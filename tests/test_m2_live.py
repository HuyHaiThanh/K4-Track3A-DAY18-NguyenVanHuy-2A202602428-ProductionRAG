"""Opt-in integration test: M1 corpus + real bge-m3 + Qdrant Docker.

PowerShell: $env:RUN_M2_LIVE='1'; python -m pytest tests/test_m2_live.py -v -s
Uses a unique temporary collection and deletes it after verification.
"""
import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from config import QDRANT_HOST, QDRANT_PORT, EMBEDDING_DIM
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch, DenseSearch


@pytest.mark.skipif(os.getenv("RUN_M2_LIVE") != "1", reason="Set RUN_M2_LIVE=1 for model/server test")
def test_live_corpus_hybrid_search():
    collection = "lab18_m2_verify_" + uuid4().hex
    dense = DenseSearch.__new__(DenseSearch)
    dense.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    dense._encoder = None
    dense.client.get_collections()  # Fail explicitly instead of using in-memory fallback.
    search = HybridSearch.__new__(HybridSearch)
    from src.m2_search import BM25Search, reciprocal_rank_fusion

    search.bm25, search.dense = BM25Search(), dense
    chunks = []
    for doc in load_documents():
        _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        chunks.extend({"text": c.text, "metadata": c.metadata} for c in children)
    try:
        search.bm25.index(chunks)
        dense.index(chunks, collection=collection)
        info = dense.client.get_collection(collection)
        assert info.points_count == len(chunks)
        assert info.config.params.vectors.size == EMBEDDING_DIM
        print(f"Indexed {len(chunks)} chunks, vector dimension {EMBEDDING_DIM}, Qdrant Docker")
        for query, expected_source in [("nghỉ phép năm", "nghi_phep"),
                                       ("Bao lâu phải đổi mật khẩu một lần?", "mat_khau")]:
            bm25 = search.bm25.search(query)
            vectors = dense.search(query, collection=collection)
            hybrid = reciprocal_rank_fusion([bm25, vectors], top_k=3)
            assert bm25 and vectors and hybrid
            assert all(r.method == "hybrid" for r in hybrid)
            assert any(expected_source in r.metadata.get("source", "") for r in hybrid)
            print(f"Query: {query}")
            for result in hybrid:
                print(f"  {result.score:.6f} | {result.metadata.get('source')} | {result.text[:100]!r}")
    finally:
        if dense.client.collection_exists(collection):
            dense.client.delete_collection(collection)
        dense.client.close()
