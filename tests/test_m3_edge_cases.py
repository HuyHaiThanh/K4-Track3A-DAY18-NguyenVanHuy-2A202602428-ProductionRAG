"""Reranking correctness independent of model downloads or hardware latency."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


@pytest.mark.parametrize("query,docs,top_k", [("q", [], 3), (" ", [{"text": "a"}], 3),
                                             ("q", [{"text": "a"}], 0),
                                             ("q", [{"text": "a"}], -1)])
def test_noop_does_not_load_model(query, docs, top_k):
    reranker = CrossEncoderReranker()
    reranker._load_model = Mock(side_effect=AssertionError("Unexpected model load"))
    assert reranker.rerank(query, docs, top_k) == []


def test_pairs_ranks_scores_metadata_and_no_input_mutation():
    docs = [{"text": "low", "score": 0.9, "metadata": {"source": "a"}},
            {"text": "high", "score": 0.1, "metadata": {"source": "b", "parent_id": "p"}},
            {"text": "middle"}]
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = np.array([-2., 8., 1.])
    results = reranker.rerank("question", docs, top_k=2)
    reranker._model.predict.assert_called_once_with(
        [("question", "low"), ("question", "high"), ("question", "middle")])
    assert [r.text for r in results] == ["high", "middle"]
    assert [r.rank for r in results] == [0, 1]
    assert results[0].original_score == 0.1 and results[0].rerank_score == 8
    assert results[0].metadata == {"source": "b", "parent_id": "p"}
    assert results[1].original_score == 0 and results[1].metadata == {}
    results[0].metadata["source"] = "changed"
    assert docs[1]["metadata"]["source"] == "b"
    assert [doc["text"] for doc in docs] == ["low", "high", "middle"]


@pytest.mark.parametrize("scores", [0.5, np.float32(0.5), [0.5], np.array([[0.5]])])
def test_single_document_score_shapes(scores):
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = scores
    results = reranker.rerank("q", [{"text": "a"}], top_k=3)
    assert len(results) == 1 and results[0].rerank_score == pytest.approx(0.5)


@pytest.mark.parametrize("scores", [[], [1, 2], [float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_predictions_raise_instead_of_losing_documents(scores):
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = scores
    with pytest.raises(ValueError, match="score per document"):
        reranker.rerank("q", [{"text": "a"}])


def test_equal_scores_preserve_candidate_order():
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = [1., 1.]
    assert [r.text for r in reranker.rerank("q", [{"text": "a"}, {"text": "b"}])] == ["a", "b"]


def test_model_loaded_once_per_instance():
    reranker = CrossEncoderReranker("test-model")
    with patch("sentence_transformers.CrossEncoder") as factory:
        assert reranker._load_model() is factory.return_value
        assert reranker._load_model() is factory.return_value
        factory.assert_called_once_with("test-model")


def test_benchmark_excludes_warmup():
    reranker = Mock()
    with patch("src.m3_rerank.time.perf_counter", side_effect=[0., 0.01, 1., 1.03]):
        stats = benchmark_reranker(reranker, "q", [], n_runs=2)
    assert reranker.rerank.call_count == 3
    assert stats == pytest.approx({"avg_ms": 20., "min_ms": 10., "max_ms": 30.})


def test_benchmark_rejects_invalid_runs():
    with pytest.raises(ValueError, match="positive"):
        benchmark_reranker(Mock(), "q", [], n_runs=0)
