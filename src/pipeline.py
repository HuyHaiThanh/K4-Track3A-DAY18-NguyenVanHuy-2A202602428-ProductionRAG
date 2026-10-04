from __future__ import annotations
"""Production integration: enrich for retrieval, supply original parents for answers."""
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report, METRIC_NAMES
from src.m5_enrichment import enrich_chunks
from src.generation import generate_answer
from config import RERANK_TOP_K

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def build_pipeline():
    print("PRODUCTION RAG PIPELINE", flush=True)
    timings = {}
    started = time.perf_counter()
    docs = load_documents()
    parents_by_id, chunks = {}, []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parents_by_id.update({p.metadata["parent_id"]: p for p in parents})
        chunks.extend({"text": c.text, "metadata": {**c.metadata, "original_text": c.text}}
                      for c in children)
    timings["chunking_seconds"] = time.perf_counter() - started
    print(f"[1/4] {len(docs)} documents → {len(chunks)} children, {len(parents_by_id)} parents", flush=True)
    started = time.perf_counter()
    enriched = enrich_chunks(chunks)
    timings["enrichment_seconds"] = time.perf_counter() - started
    counts = dict(Counter(e.auto_metadata["enrichment_status"] for e in enriched))
    indexed = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
    print(f"[2/4] Enrichment status: {counts}", flush=True)
    started = time.perf_counter()
    search = HybridSearch()
    search.index(indexed)
    timings["index_seconds"] = time.perf_counter() - started
    search.parents_by_id = parents_by_id
    search.build_info = {"num_documents": len(docs), "num_children": len(chunks),
                         "num_parents": len(parents_by_id), "enrichment_counts": counts,
                         "timings": timings}
    print("[3/4] Hybrid index ready", flush=True)
    started = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker._load_model()
    timings["reranker_load_seconds"] = time.perf_counter() - started
    print("[4/4] Reranker loaded", flush=True)
    return search, reranker


def select_original_contexts(reranked, search, top_k=RERANK_TOP_K):
    contexts, sources, seen = [], [], set()
    if top_k <= 0:
        return contexts, sources
    for result in reranked:
        metadata = result.metadata
        parent_id = metadata.get("parent_id")
        parent = getattr(search, "parents_by_id", {}).get(parent_id)
        text = parent.text if parent is not None else metadata.get("original_text", result.text)
        source = metadata.get("source", "unknown")
        identity = (source, parent_id) if parent is not None else (source, text)
        if identity in seen:
            continue
        seen.add(identity)
        contexts.append(f"Nguồn: {source}\n{text}")
        sources.append({"source": source, "parent_id": parent_id,
                        "rerank_score": getattr(result, "rerank_score", None)})
        if len(contexts) == top_k:
            break
    return contexts, sources


def run_query(query, search, reranker):
    started = time.perf_counter()
    results = search.search(query)
    retrieval_ms = (time.perf_counter() - started) * 1000
    documents = [{"text": r.metadata.get("original_text", r.text), "score": r.score,
                  "metadata": r.metadata} for r in results]
    started = time.perf_counter()
    # Rank all candidates to retain three distinct parents, not three overlapping children.
    reranked = reranker.rerank(query, documents, top_k=len(documents))
    rerank_ms = (time.perf_counter() - started) * 1000
    contexts, sources = select_original_contexts(reranked or results, search)
    started = time.perf_counter()
    answer, generation_status = generate_answer(query, contexts)
    generation_ms = (time.perf_counter() - started) * 1000
    search.last_trace = {"question": query, "answer": answer, "contexts": contexts,
                         "sources": sources, "num_candidates": len(results),
                         "generation_status": generation_status,
                         "latency_ms": {"retrieval": retrieval_ms, "rerank": rerank_ms,
                                        "generation": generation_ms}}
    return answer, contexts


def evaluate_pipeline(search, reranker):
    test_set = load_test_set()
    traces = []
    REPORT_DIR.mkdir(exist_ok=True)
    for i, item in enumerate(test_set):
        run_query(item["question"], search, reranker)
        traces.append({**search.last_trace, "ground_truth": item["ground_truth"]})
        (REPORT_DIR / "production_trace.json").write_text(
            json.dumps({"build": search.build_info, "queries": traces}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[Production {i+1}/{len(test_set)}] {item['question'][:65]}", flush=True)
    started = time.perf_counter()
    results = evaluate_ragas([t["question"] for t in traces], [t["answer"] for t in traces],
                             [t["contexts"] for t in traces], [t["ground_truth"] for t in traces])
    search.build_info["timings"]["ragas_seconds"] = time.perf_counter() - started
    (REPORT_DIR / "production_trace.json").write_text(
        json.dumps({"build": search.build_info, "queries": traces}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    save_report(results, failure_analysis(results["per_question"], bottom_n=5),
                path=str(REPORT_DIR / "ragas_report.json"))
    print(f"Production evaluation status: {results['status']}", flush=True)
    for metric in METRIC_NAMES:
        print(f"  {metric}: {results[metric]:.4f}", flush=True)
    if results["status"] != "success" or any(t["generation_status"] != "success" for t in traces):
        raise RuntimeError("Production evaluation or answer generation incomplete; inspect reports")
    return results


if __name__ == "__main__":
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
