"""Enrichment request, fallback and metadata regression tests without API usage."""
import json
from unittest.mock import Mock

import pytest

from src import m5_enrichment as m5


SAMPLE = "Nhân viên được nghỉ 15 ngày phép năm. Thử việc không được nghỉ phép năm."


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "test-key")
    client = Mock()
    monkeypatch.setattr(m5, "_get_client", lambda key: client)
    payload = {"summary": "Nhân viên được nghỉ 15 ngày phép năm.",
               "questions": ["Nhân viên được nghỉ bao nhiêu ngày?", "Thử việc được nghỉ không?"],
               "context": "Đoạn quy định nghỉ phép trong policy.md.",
               "metadata": {"topic": "nghỉ phép", "entities": [], "category": "hr", "language": "vi"}}
    response = client.chat.completions.create.return_value
    response.choices = [Mock(finish_reason="stop", message=Mock(content=json.dumps(payload)))]
    return client, payload, response


def test_combined_one_request_preserves_text_and_indexes_hyqa(provider):
    client, _, _ = provider
    meta = {"source": "policy.md", "parent_id": "parent1", "chunk_index": 0}
    result = m5.enrich_chunks([{"text": SAMPLE, "metadata": meta}])[0]
    client.chat.completions.create.assert_called_once()
    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "gpt-4o-mini" and args["temperature"] == 0
    assert args["response_format"] == {"type": "json_object"}
    assert result.original_text == SAMPLE and result.enriched_text.endswith(SAMPLE)
    assert result.hypothesis_questions[0] in result.enriched_text
    assert result.auto_metadata["parent_id"] == "parent1"
    assert result.auto_metadata["source"] == "policy.md"
    assert result.auto_metadata["enrichment_status"] == "success"
    assert meta == {"source": "policy.md", "parent_id": "parent1", "chunk_index": 0}


def test_generated_metadata_cannot_override_source_linkage(provider):
    _, payload, response = provider
    payload["metadata"].update(source="wrong", parent_id="wrong", chunk_index=99)
    response.choices[0].message.content = json.dumps(payload)
    result = m5.enrich_chunks([{"text": SAMPLE, "metadata": {"source": "real", "parent_id": "p"}}])[0]
    assert result.auto_metadata["source"] == "real" and result.auto_metadata["parent_id"] == "p"
    assert "chunk_index" not in result.auto_metadata


@pytest.mark.parametrize("payload", ["not JSON", "[]", '{}',
    '{"summary": 1, "questions": [], "context": "", "metadata": {}}',
    '{"summary": "s", "questions": [1], "context": "c", "metadata": {}}'])
def test_malformed_payload_falls_back_without_retry(provider, payload):
    client, _, response = provider
    response.choices[0].message.content = payload
    result = m5._enrich_single_call(SAMPLE, "policy.md")
    assert result["status"] == "fallback"
    assert result["summary"] == SAMPLE
    client.chat.completions.create.assert_called_once()


def test_truncated_response_falls_back(provider):
    _, _, response = provider
    response.choices[0].finish_reason = "length"
    assert m5._enrich_single_call(SAMPLE, "policy.md")["status"] == "fallback"


def test_provider_failure_does_not_log_secret(provider, capsys):
    client, _, _ = provider
    client.chat.completions.create.side_effect = RuntimeError("sensitive-test-token")
    assert m5._enrich_single_call(SAMPLE, "policy.md")["status"] == "fallback"
    assert "sensitive-test-token" not in capsys.readouterr().out


def test_offline_functions_are_useful_and_preserve_original(monkeypatch):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "")
    assert m5.summarize_chunk(SAMPLE) == SAMPLE
    assert len(m5.generate_hypothesis_questions(SAMPLE, 1)) == 1
    assert m5.contextual_prepend(SAMPLE, "policy.md").endswith(SAMPLE)
    result = m5.enrich_chunks([{"text": SAMPLE, "metadata": {"source": "policy.md"}}])[0]
    assert result.enriched_text != SAMPLE
    assert result.auto_metadata["enrichment_status"] == "fallback"
    assert result.auto_metadata["category"] == "unknown"


def test_empty_input_and_nonpositive_question_count_skip_provider(provider):
    client, _, _ = provider
    assert m5._enrich_single_call(" ", "policy.md")["status"] == "fallback"
    assert m5.generate_hypothesis_questions(SAMPLE, 0) == []
    assert m5.generate_hypothesis_questions(SAMPLE, -1) == []
    assert m5.enrich_chunks([]) == []
    client.chat.completions.create.assert_not_called()


def test_questions_are_deduplicated_and_limited(provider):
    _, payload, response = provider
    payload["questions"] = [" q1? ", "q1?", "", "q2?", "q3?"]
    response.choices[0].message.content = json.dumps(payload)
    assert m5.generate_hypothesis_questions(SAMPLE, 2) == ["q1?", "q2?"]


@pytest.mark.parametrize("method", ["summary", "hyqa", "contextual", "metadata"])
def test_individual_modes(provider, method):
    client, _, _ = provider
    result = m5.enrich_chunks([{"text": SAMPLE}], methods=[method])[0]
    assert result.original_text == SAMPLE and SAMPLE in result.enriched_text
    assert result.auto_metadata["enrichment_status"] == "success"
    if method == "hyqa":
        assert result.hypothesis_questions[0] in result.enriched_text
    if method == "summary":
        assert result.enriched_text.startswith(result.summary)
    client.chat.completions.create.assert_called_once()


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        m5.enrich_chunks([], methods=["typo"])
