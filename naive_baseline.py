"""Basic RAG baseline: paragraph chunks, dense-only retrieval, shared generation."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report, METRIC_NAMES
from src.generation import generate_answer
from config import NAIVE_COLLECTION

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def main():
    print("BASIC RAG BASELINE", flush=True)
    started = time.perf_counter()
    docs = load_documents()
    chunks = [{"text": c.text, "metadata": c.metadata} for doc in docs
              for c in chunk_basic(doc["text"], metadata=doc["metadata"])]
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    build_seconds = time.perf_counter() - started
    test_set, traces = load_test_set(), []
    REPORT_DIR.mkdir(exist_ok=True)
    for i, item in enumerate(test_set):
        started = time.perf_counter()
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        retrieval_ms = (time.perf_counter() - started) * 1000
        contexts = [f"Nguồn: {r.metadata.get('source', 'unknown')}\n{r.text}" for r in results]
        started = time.perf_counter()
        answer, status = generate_answer(item["question"], contexts)
        traces.append({"question": item["question"], "answer": answer, "contexts": contexts,
                       "ground_truth": item["ground_truth"], "generation_status": status,
                       "sources": [r.metadata.get("source") for r in results],
                       "latency_ms": {"retrieval": retrieval_ms, "generation":
                                      (time.perf_counter() - started) * 1000}})
        (REPORT_DIR / "naive_trace.json").write_text(
            json.dumps({"num_chunks": len(chunks), "build_seconds": build_seconds,
                        "queries": traces}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Baseline {i+1}/{len(test_set)}] {item['question'][:65]}", flush=True)
    results = evaluate_ragas([t["question"] for t in traces], [t["answer"] for t in traces],
                             [t["contexts"] for t in traces], [t["ground_truth"] for t in traces])
    save_report(results, [], path=str(REPORT_DIR / "naive_baseline_report.json"))
    search.client.close()
    print(f"Baseline evaluation status: {results['status']}", flush=True)
    for metric in METRIC_NAMES:
        print(f"  {metric}: {results[metric]:.4f}", flush=True)
    if results["status"] != "success" or any(t["generation_status"] != "success" for t in traces):
        raise RuntimeError("Baseline evaluation or generation incomplete; inspect reports")
    return results


if __name__ == "__main__":
    main()
