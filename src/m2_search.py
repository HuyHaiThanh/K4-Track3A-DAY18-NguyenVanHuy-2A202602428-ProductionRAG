from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    if not text.strip():
        return ""
    return word_tokenize(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = []
        self.corpus_tokens = []
        self.bm25 = None
        for chunk in chunks:
            tokens = segment_vietnamese(chunk["text"].lower()).split()
            if tokens:
                self.documents.append({"text": chunk["text"],
                                       "metadata": dict(chunk.get("metadata", {}))})
                self.corpus_tokens.append(tokens)
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        tokens = segment_vietnamese(query.lower()).split()
        scores = self.bm25.get_scores(tokens)
        indices = sorted((i for i, score in enumerate(scores) if score > 0),
                         key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(self.documents[i]["text"], float(scores[i]),
                             dict(self.documents[i]["metadata"]), "bm25") for i in indices]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        try:
            self.client.get_collections()
        except Exception as exc:
            print(f"  ⚠️ Qdrant unavailable; using temporary in-memory storage: {exc}")
            self.client.close()
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Replace a lab collection with the nonblank chunks and their metadata.

        Encode and validate before replacement; database writes are not atomic.
        """
        from qdrant_client.models import Distance, VectorParams, PointStruct

        documents = [c for c in chunks if c["text"].strip()]
        vectors = None
        if documents:
            vectors = self._get_encoder().encode([c["text"] for c in documents],
                                                 show_progress_bar=True)
            if vectors.shape != (len(documents), EMBEDDING_DIM):
                raise ValueError(f"Expected {len(documents)} vectors of dimension {EMBEDDING_DIM}, "
                                 f"got shape {vectors.shape}")
        # Encode first: a model failure must not erase an existing collection.
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(
            size=EMBEDDING_DIM, distance=Distance.COSINE))
        if vectors is not None:
            points = [PointStruct(id=i, vector=vector.tolist(),
                                  payload={**doc.get("metadata", {}), "text": doc["text"]})
                      for i, (doc, vector) in enumerate(zip(documents, vectors))]
            self.client.upsert(collection, points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip() or not self.client.collection_exists(collection):
            return []
        if self.client.get_collection(collection).points_count == 0:
            return []
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection, query=query_vector, limit=top_k,
                                           with_payload=True)
        return [SearchResult(pt.payload["text"], float(pt.score),
                             {k: v for k, v in pt.payload.items() if k != "text"}, "dense")
                for pt in response.points]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge by text identity: score(d) = Σ 1/(k + zero_based_rank + 1)."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if top_k <= 0:
        return []
    scores, documents = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            if result.text in seen:
                continue
            seen.add(result.text)
            documents.setdefault(result.text, result)
            scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (k + rank + 1)
    texts = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(text, scores[text], dict(documents[text].metadata), "hybrid")
            for text in texts]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        if top_k <= 0 or not query.strip():
            return []
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
