"""Integration invariants without external APIs or model inference."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import pipeline
from src.m1_chunking import Chunk
from src.m2_search import SearchResult
from src.m3_rerank import RerankResult


def test_parent_contexts_are_original_unique_and_limited():
    search = SimpleNamespace(parents_by_id={
        "p1": Chunk("Original parent one"), "p2": Chunk("Original parent two")})
    results = [RerankResult("generated text", 1, 0.9, {"source": "a", "parent_id": "p1"}, 0),
               RerankResult("another child", 1, 0.8, {"source": "a", "parent_id": "p1"}, 1),
               RerankResult("third", 1, 0.7, {"source": "b", "parent_id": "p2"}, 2)]
    contexts, sources = pipeline.select_original_contexts(results, search, top_k=2)
    assert contexts == ["Nguồn: a\nOriginal parent one", "Nguồn: b\nOriginal parent two"]
    assert len(sources) == 2
    assert pipeline.select_original_contexts(results, search, top_k=0) == ([], [])


def test_run_query_reranks_raw_children_and_generates_from_raw_parents(monkeypatch):
    search = Mock()
    search.parents_by_id = {"p": Chunk("Original document")}
    metadata = {"source": "policy.md", "parent_id": "p", "original_text": "Original child"}
    search.search.return_value = [SearchResult("AI enrichment", 0.03, metadata, "hybrid")]
    reranker = Mock()
    reranker.rerank.return_value = [RerankResult("Original child", 0.03, 0.9, metadata, 0)]
    generate = Mock(return_value=("answer", "success"))
    monkeypatch.setattr(pipeline, "generate_answer", generate)
    answer, contexts = pipeline.run_query("q", search, reranker)
    assert reranker.rerank.call_args.args[1][0]["text"] == "Original child"
    assert contexts == ["Nguồn: policy.md\nOriginal document"]
    generate.assert_called_once_with("q", contexts)
    assert answer == "answer" and search.last_trace["generation_status"] == "success"


def test_incomplete_evaluation_cannot_report_end_to_end_success(monkeypatch, tmp_path):
    search = SimpleNamespace(build_info={"timings": {}})
    monkeypatch.setattr(pipeline, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "load_test_set", lambda: [{"question": "q", "ground_truth": "gt"}])
    def query(*args):
        search.last_trace = {"question": "q", "answer": "a", "contexts": [],
                             "generation_status": "fallback_missing_key"}
        return "a", []
    monkeypatch.setattr(pipeline, "run_query", query)
    monkeypatch.setattr(pipeline, "evaluate_ragas", lambda *args: {
        **dict.fromkeys(pipeline.METRIC_NAMES, 0.0), "per_question": [], "status": "skipped"})
    with pytest.raises(RuntimeError, match="incomplete"):
        pipeline.evaluate_pipeline(search, Mock())
    assert (tmp_path / "ragas_report.json").exists()
