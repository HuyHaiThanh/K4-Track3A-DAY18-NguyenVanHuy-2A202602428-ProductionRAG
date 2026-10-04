from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate with RAGAS 0.1.x; distinguish measured scores from fallback.

    Invalid metric cells remain None and are excluded from aggregate means.
    Caller input errors raise ValueError; provider failures return status=error.
    """
    count = len(questions)
    if not (len(answers) == len(contexts) == len(ground_truths) == count):
        raise ValueError("All evaluation inputs must have the same length")
    if any(not isinstance(value, str) for values in (questions, answers, ground_truths)
           for value in values):
        raise ValueError("Questions, answers and ground truths must contain strings")
    if any(not isinstance(group, list) or any(not isinstance(c, str) for c in group)
           for group in contexts):
        raise ValueError("Contexts must be a list of lists of strings")

    fallback = {**dict.fromkeys(METRIC_NAMES, 0.0), "per_question": [],
                "requested_questions": count, "valid_counts": dict.fromkeys(METRIC_NAMES, 0)}
    if not count:
        return {**fallback, "status": "skipped", "error": "empty_dataset"}
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        print("  ⚠️ RAGAS skipped: no valid OPENAI_API_KEY configured.")
        return {**fallback, "status": "skipped", "error": "missing_api_key"}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=ChatOpenAI(model=os.getenv("RAGAS_MODEL", "gpt-4o-mini"),
                                         temperature=0, api_key=OPENAI_API_KEY),
                          embeddings=OpenAIEmbeddings(
                              model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
                              api_key=OPENAI_API_KEY),
                          run_config=RunConfig(timeout=60, max_retries=2, max_workers=4),
                          raise_exceptions=True)
        frame = result.to_pandas()
        if len(frame) != count:
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = []
        for i, (_, row) in enumerate(frame.iterrows()):
            metrics = {name: _valid_score(row.get(name)) for name in METRIC_NAMES}
            per_question.append(EvalResult(questions[i], answers[i], list(contexts[i]),
                                           ground_truths[i], **metrics))
        scores, valid_counts = {}, {}
        for name in METRIC_NAMES:
            valid = [getattr(row, name) for row in per_question if getattr(row, name) is not None]
            valid_counts[name] = len(valid)
            scores[name] = sum(valid) / len(valid) if valid else 0.0
        status = "success" if all(n == count for n in valid_counts.values()) else "partial"
        return {**scores, "per_question": per_question, "status": status,
                "requested_questions": count, "valid_counts": valid_counts}
    except Exception as exc:
        # Do not print provider exception text: it can contain request details or secrets.
        print(f"  ⚠️ RAGAS evaluation failed ({type(exc).__name__}); scores are fallback.")
        return {**fallback, "status": "error", "error": type(exc).__name__}


def _valid_score(value) -> float | None:
    """Only finite values in [0, 1] represent usable metric measurements."""
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 <= value <= 1 else None
    except (TypeError, ValueError):
        return None


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("Possible unsupported claims in the answer",
                         "Tighten system prompt, set temperature=0, verify claims against context"),
        "context_recall": ("Possibly missing relevant chunks",
                           "Review chunking and BM25 retrieval; check multi-hop coverage"),
        "context_precision": ("Possibly irrelevant chunks ranked too highly",
                              "Review CrossEncoder reranking and filter metadata/version"),
        "answer_relevancy": ("Answer may not address the question directly",
                             "Improve prompt to answer the question directly"),
    }
    failures = []
    for result in eval_results:
        metrics = {name: _valid_score(getattr(result, name)) for name in METRIC_NAMES}
        # An incomplete evaluation cannot be diagnosed as a genuine RAG failure.
        if any(score is None for score in metrics.values()):
            continue
        worst = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({"question": result.question, "answer": result.answer,
                         "contexts": list(result.contexts), "ground_truth": result.ground_truth,
                         "score": sum(metrics.values()) / len(metrics), "metrics": metrics,
                         "worst_metric": worst, "worst_score": metrics[worst],
                         "diagnosis": diagnosis, "suggested_fix": fix,
                         "error_tree": "Output incorrect? → Context sufficient and current? "
                                       "→ Relevant chunks retrieved? → Inspect indicated module",
                         "diagnosis_note": "Hypothesis from metrics; verify using answer and contexts"})
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {name: results.get(name, 0.0) for name in METRIC_NAMES},
        "num_questions": len(results.get("per_question", [])),
        "evaluation_status": results.get("status", "unknown"),
        "requested_questions": results.get("requested_questions", len(results.get("per_question", []))),
        "valid_counts": results.get("valid_counts", {}),
        "error": results.get("error"),
        "per_question": [asdict(row) for row in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
