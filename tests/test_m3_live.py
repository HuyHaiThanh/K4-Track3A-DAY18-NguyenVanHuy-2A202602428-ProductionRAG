"""Opt-in M1→M2→M3 integration and measured latency report.

PowerShell: $env:RUN_M3_LIVE='1'; python -m pytest tests/test_m3_live.py -v -s
"""
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from config import QDRANT_HOST, QDRANT_PORT
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


@pytest.mark.skipif(os.getenv("RUN_M3_LIVE") != "1", reason="Set RUN_M3_LIVE=1 for model/server test")
def test_live_hybrid_rerank_and_latency():
    collection = "lab18_m3_verify_" + uuid4().hex
    dense = DenseSearch.__new__(DenseSearch)
    dense.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    dense._encoder = None
    dense.client.get_collections()
    bm25 = BM25Search()
    chunks = []
    for doc in load_documents():
        _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        chunks.extend({"text": c.text, "metadata": c.metadata} for c in children)
    report = {"model": "BAAI/bge-reranker-v2-m3", "num_chunks": len(chunks),
              "top_k": 3, "warm_runs": 3, "latency_target_ms": 150,
              "note": "Warm timings exclude loading and first inference; only reranking is timed.",
              "queries": []}
    try:
        bm25.index(chunks)
        dense.index(chunks, collection=collection)
        reranker = CrossEncoderReranker()
        for query, expected_source in [
                ("Nhân viên được nghỉ bao nhiêu ngày phép năm?", "nghi_phep_nam"),
                ("Bao lâu phải đổi mật khẩu một lần?", "mat_khau")]:
            candidates = reciprocal_rank_fusion([bm25.search(query),
                                                 dense.search(query, collection=collection)])
            documents = [{"text": r.text, "score": r.score, "metadata": r.metadata}
                         for r in candidates]
            started = time.perf_counter()
            reranked = reranker.rerank(query, documents)
            first_call_ms = (time.perf_counter() - started) * 1000
            assert len(reranked) == min(3, len(documents))
            assert [r.rank for r in reranked] == list(range(len(reranked)))
            assert all(a.rerank_score >= b.rerank_score for a, b in zip(reranked, reranked[1:]))
            assert expected_source in reranked[0].metadata["source"]
            stats = benchmark_reranker(reranker, query, documents, n_runs=3)
            item = {"question": query, "num_candidates": len(documents),
                    "first_call_ms": first_call_ms, "warm_latency": stats,
                    "meets_150ms_target": stats["avg_ms"] < 150,
                    "hybrid_top3_sources": [r.metadata["source"] for r in candidates[:3]],
                    "reranked_top3": [{"rank": r.rank, "score": r.rerank_score,
                                       "source": r.metadata["source"], "text": r.text}
                                      for r in reranked]}
            report["queries"].append(item)
            print(f"{query}: {len(documents)} → {len(reranked)}; warm {stats}")
            print("Top sources:", [r.metadata["source"] for r in reranked])
        report["device"] = str(reranker._model.model.device)
        output = Path(__file__).resolve().parents[1] / "reports" / "m3_latency_report.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved to {output}; device={report['device']}")
    finally:
        if dense.client.collection_exists(collection):
            dense.client.delete_collection(collection)
        dense.client.close()
