from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    groups = [[sentences[0]]]
    if len(sentences) > 1:
        import numpy as np

        embeddings = _get_semantic_encoder().encode(sentences)
        for i in range(1, len(sentences)):
            previous, current = embeddings[i - 1], embeddings[i]
            similarity = float(np.dot(previous, current) /
                               (np.linalg.norm(previous) * np.linalg.norm(current) + 1e-9))
            if similarity < threshold:
                groups.append([])
            groups[-1].append(sentences[i])
    return [Chunk(text=" ".join(group),
                  metadata={**(metadata or {}), "strategy": "semantic", "chunk_index": i})
            for i, group in enumerate(groups)]


@lru_cache(maxsize=1)
def _get_semantic_encoder():
    """Reuse the encoder across documents instead of reloading its weights."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if child_size > parent_size:
        raise ValueError("child_size must not exceed parent_size")
    if not text.strip():
        return [], []
    parents, children = [], []
    document_id = uuid4().hex
    for i, parent_text in enumerate(_split_bounded(text, parent_size)):
        pid = f"parent_{document_id}_{i}"
        parents.append(Chunk(text=parent_text, metadata={**(metadata or {}),
                             "chunk_type": "parent", "parent_id": pid, "chunk_index": i}))
        for j, child_text in enumerate(_split_bounded(parent_text, child_size)):
            children.append(Chunk(text=child_text, parent_id=pid,
                                  metadata={**(metadata or {}), "chunk_type": "child",
                                            "parent_id": pid, "chunk_index": j}))
    return parents, children


def _split_bounded(text: str, size: int) -> list[str]:
    """Prefer paragraph, sentence, or whitespace boundaries; never lose text.

    Unbroken words longer than size require splitting to respect the hard limit.
    Joining the returned chunks restores the input, including whitespace.
    """
    parts = []
    while len(text) > size:
        window = text[:size]
        boundaries = [m.end() for m in re.finditer(r'\n\s*\n', window)]
        if not boundaries:
            boundaries = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', window)]
        if not boundaries:
            boundaries = [m.end() for m in re.finditer(r'\s+', window)]
        end = boundaries[-1] if boundaries else size
        if not text[:end].strip():
            end = size
        parts.append(text[:end])
        text = text[end:]
    if text:
        parts.append(text)
    return parts


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    if not text.strip():
        return []
    chunks, lines = [], []
    section = ""
    fence_char, fence_length = "", 0

    def flush():
        content = "".join(lines).strip()
        if content:
            chunks.append(Chunk(text=content, metadata={**(metadata or {}),
                                "section": section, "strategy": "structure",
                                "chunk_index": len(chunks)}))
        lines.clear()

    for line in text.splitlines(keepends=True):
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\r\n'))
        if fence_char:
            lines.append(line)
            if (fence and fence.group(1)[0] == fence_char
                    and len(fence.group(1)) >= fence_length
                    and not fence.group(2).strip()):
                fence_char, fence_length = "", 0
            continue
        if fence:
            fence_char, fence_length = fence.group(1)[0], len(fence.group(1))
            lines.append(line)
            continue
        header = re.match(r'^ {0,3}#{1,3}[ \t]+(.+?)\s*$', line)
        if header:
            flush()
            section = header.group(1)
        lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
