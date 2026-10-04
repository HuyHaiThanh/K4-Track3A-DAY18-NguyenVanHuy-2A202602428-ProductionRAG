"""Regression tests for content preservation and advanced chunking boundaries."""
from unittest.mock import Mock

import numpy as np
import pytest

from src import m1_chunking as m1


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_documents(text):
    assert m1.chunk_semantic(text) == []
    assert m1.chunk_hierarchical(text) == ([], [])
    assert m1.chunk_structure_aware(text) == []


def test_semantic_topic_transition(monkeypatch):
    encoder = Mock()
    encoder.encode.return_value = np.array([[1., 0.], [1., 0.], [0., 1.]])
    monkeypatch.setattr(m1, "_get_semantic_encoder", lambda: encoder)
    meta = {"source": "policy.md"}
    chunks = m1.chunk_semantic("Nghỉ phép năm. Nghỉ phép thêm. Mật khẩu mới.", metadata=meta)
    assert [c.text for c in chunks] == ["Nghỉ phép năm. Nghỉ phép thêm.", "Mật khẩu mới."]
    assert all(c.metadata["source"] == "policy.md" for c in chunks)
    assert meta == {"source": "policy.md"}


def test_semantic_single_sentence_needs_no_model(monkeypatch):
    def unexpected_load():
        raise AssertionError("Single sentence should not load the encoder")
    monkeypatch.setattr(m1, "_get_semantic_encoder", unexpected_load)
    assert m1.chunk_semantic("Một câu.")[0].text == "Một câu."


@pytest.mark.parametrize("text", ["X" * 5000, ("Quy định tiếng Việt. " * 100),
                                  "Đầu.\n\n" + "dài " * 100 + "\n\nCuối."],
                         ids=["unbroken-long-text", "long-sentences", "paragraphs"])
def test_hierarchy_preserves_text_and_limits(text):
    parents, children = m1.chunk_hierarchical(text, parent_size=200, child_size=80)
    assert "".join(p.text for p in parents) == text
    assert all(0 < len(p.text) <= 200 for p in parents)
    assert all(0 < len(c.text) <= 80 for c in children)
    for parent in parents:
        pid = parent.metadata["parent_id"]
        linked = [c for c in children if c.parent_id == pid]
        assert "".join(c.text for c in linked) == parent.text
        assert all(c.metadata["parent_id"] == pid for c in linked)


def test_parent_ids_unique_across_documents():
    first, _ = m1.chunk_hierarchical("Tài liệu A")
    second, _ = m1.chunk_hierarchical("Tài liệu B")
    assert first[0].metadata["parent_id"] != second[0].metadata["parent_id"]


@pytest.mark.parametrize("parent_size,child_size", [(0, 1), (10, 0), (-1, 1), (10, 11)])
def test_invalid_sizes(parent_size, child_size):
    with pytest.raises(ValueError):
        m1.chunk_hierarchical("Văn bản", parent_size, child_size)


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        m1.chunk_semantic("Một câu.", threshold)


@pytest.mark.parametrize("fence", [chr(96) * 3, "~~~"])
def test_markdown_fences_tables_and_lists(fence):
    text = ("Mở đầu\n\n# Chính sách\n\n| Mục | Giá trị |\n|---|---|\n| Phép | 15 |\n"
            "- Giữ danh sách\n" + fence + "python\n## Không phải tiêu đề\n" + fence +
            "\n### Điều kiện\nNội dung cuối.")
    chunks = m1.chunk_structure_aware(text, {"source": "policy.md"})
    assert [c.metadata["section"] for c in chunks] == ["", "Chính sách", "Điều kiện"]
    assert "| Phép | 15 |" in chunks[1].text
    assert "- Giữ danh sách" in chunks[1].text
    assert "## Không phải tiêu đề" in chunks[1].text
    assert all(c.metadata["source"] == "policy.md" for c in chunks)


def test_structure_without_headers():
    assert m1.chunk_structure_aware("Nội dung\nkhông có header")[0].text == "Nội dung\nkhông có header"
