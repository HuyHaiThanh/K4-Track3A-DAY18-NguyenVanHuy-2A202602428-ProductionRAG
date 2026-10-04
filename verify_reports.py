"""Verify dataset alignment, actual score means and original-source contexts offline."""
import json
from pathlib import Path
from src.m1_chunking import load_documents
from src.m4_eval import METRIC_NAMES, load_test_set


def verify_reports():
    root = Path(__file__).resolve().parent
    test_set = load_test_set()
    count = len(test_set)
    for name in ("naive_baseline_report.json", "ragas_report.json"):
        report = json.loads((root / "reports" / name).read_text(encoding="utf-8"))
        assert report["evaluation_status"] == "success", name
        assert report["num_questions"] == len(report["per_question"]) == count, name
        for row, expected in zip(report["per_question"], test_set):
            assert row["question"] == expected["question"]
            assert row["ground_truth"] == expected["ground_truth"]
        for metric in METRIC_NAMES:
            scores = [row[metric] for row in report["per_question"]]
            assert all(isinstance(score, (int, float)) and 0 <= score <= 1 for score in scores)
            assert abs(sum(scores) / count - report["aggregate"][metric]) < 1e-10
            assert report["valid_counts"][metric] == count
    documents = {d["metadata"]["source"]: d["text"] for d in load_documents()}
    checked = 0
    for name in ("naive_trace.json", "production_trace.json"):
        trace = json.loads((root / "reports" / name).read_text(encoding="utf-8"))
        assert len(trace["queries"]) == count
        for row in trace["queries"]:
            assert row["generation_status"] == "success"
            for context in row["contexts"]:
                header, body = context.split("\n", 1)
                source = header.removeprefix("Nguồn: ")
                assert body in documents[source], source
                checked += 1
    return {"evaluated_questions": count * 2, "original_contexts_verified": checked}


if __name__ == "__main__":
    print(verify_reports())
