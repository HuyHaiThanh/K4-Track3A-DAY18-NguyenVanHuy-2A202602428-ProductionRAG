"""M2 regression tests; real local Qdrant with controlled embeddings."""
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.m2_search import (BM25Search, DenseSearch, HybridSearch, SearchResult,
                           reciprocal_rank_fusion, segment_vietnamese)
from config import EMBEDDING_DIM


def test_segmentation_removes_underscores():
    assert "_" not in segment_vietnamese("Nhân viên nghỉ phép năm")
    assert segment_vietnamese(" \n") == ""


def test_bm25_case_normalization_and_positive_scores():
    search = BM25Search()
    search.index([{"text": "NGHỈ PHÉP năm", "metadata": {"source": "hr"}},
                  {"text": "VPN bảo mật"}, {"text": "Lương tháng"}])
    results = search.search("nghỉ phép", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "hr"
    assert results[0].score > 0
    assert search.search("zzzzzz") == []
    assert search.search(" ") == []
    assert search.search("nghỉ", top_k=0) == []


def test_bm25_reindex_empty_clears_old_index():
    search = BM25Search()
    search.index([{"text": "nghỉ phép"}, {"text": "VPN"}, {"text": "lương"}])
    search.index([{"text": " "}])
    assert search.documents == []
    assert search.search("nghỉ") == []


def test_rrf_exact_scores_order_and_input_unchanged():
    a = SearchResult("a", 999, {"source": "a.md"}, "bm25")
    b = SearchResult("b", 0.01, {}, "dense")
    results = reciprocal_rank_fusion([[a, b], [b]], top_k=2)
    assert [r.text for r in results] == ["b", "a"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert all(r.method == "hybrid" for r in results)
    assert a.score == 999 and a.method == "bm25"


def test_rrf_does_not_count_duplicates_in_one_list():
    doc = SearchResult("same", 1, {}, "bm25")
    assert reciprocal_rank_fusion([[doc, doc]])[0].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[doc]], top_k=0) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[doc]], k=-1)


@pytest.fixture
def dense():
    # Bypass network initialization, but exercise the actual Qdrant storage/API.
    search = DenseSearch.__new__(DenseSearch)
    search.client = QdrantClient(":memory:")
    encoder = Mock()
    vectors = np.zeros((2, EMBEDDING_DIM))
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    encoder.encode.side_effect = lambda text, **kwargs: vectors if isinstance(text, list) else vectors[0]
    search._encoder = encoder
    yield search
    search.client.close()


def test_dense_round_trip_and_metadata(dense):
    dense.index([{"text": "nghỉ phép", "metadata": {"source": "hr", "parent_id": "p1"}},
                 {"text": "VPN", "metadata": {"source": "it"}}], collection="test")
    results = dense.search("nghỉ", top_k=1, collection="test")
    assert len(results) == 1 and results[0].text == "nghỉ phép"
    assert results[0].score == pytest.approx(1)
    assert results[0].method == "dense"
    assert results[0].metadata == {"source": "hr", "parent_id": "p1"}


def test_dense_empty_and_missing_collection_skip_encoding(dense):
    assert dense.search("query", collection="missing") == []
    dense.index([], collection="empty")
    assert dense.search("query", collection="empty") == []
    assert dense.search("query", top_k=0) == []
    assert dense.search(" ") == []
    dense._encoder.encode.assert_not_called()


def test_dense_reindex_replaces_old_points(dense):
    dense.index([{"text": "old1"}, {"text": "old2"}], collection="test")
    dense.index([], collection="test")
    assert dense.client.get_collection("test").points_count == 0


def test_bad_vectors_preserve_existing_collection(dense):
    dense.index([{"text": "old1"}, {"text": "old2"}], collection="test")
    dense._encoder.encode.side_effect = None
    dense._encoder.encode.return_value = np.zeros((2, 3))
    with pytest.raises(ValueError, match="dimension"):
        dense.index([{"text": "new1"}, {"text": "new2"}], collection="test")
    assert dense.client.get_collection("test").points_count == 2


def test_hybrid_calls_both_and_fuses():
    search = HybridSearch.__new__(HybridSearch)
    search.bm25, search.dense = Mock(), Mock()
    search.bm25.search.return_value = [SearchResult("shared", 50, {}, "bm25")]
    search.dense.search.return_value = [SearchResult("shared", 0.8, {}, "dense")]
    results = search.search("query", top_k=1)
    assert results[0].score == pytest.approx(2 / 61)
    search.bm25.search.assert_called_once()
    search.dense.search.assert_called_once()


def test_hybrid_empty_query_does_not_call_engines():
    search = HybridSearch.__new__(HybridSearch)
    search.bm25, search.dense = Mock(), Mock()
    assert search.search(" ") == []
    assert search.search("query", top_k=0) == []
    search.bm25.search.assert_not_called()
    search.dense.search.assert_not_called()
