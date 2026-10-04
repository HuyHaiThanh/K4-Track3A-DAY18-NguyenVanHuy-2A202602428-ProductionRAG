"""Opt-in single synthetic chunk enrichment request (incurs API usage).

PowerShell: $env:RUN_M5_LIVE='1'; python -m pytest tests/test_m5_live.py -v -s
"""
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from src.m5_enrichment import enrich_chunks


@pytest.mark.skipif(os.getenv("RUN_M5_LIVE") != "1", reason="Set RUN_M5_LIVE=1 for API test")
def test_live_combined_enrichment():
    text = "Theo chính sách v2024, nhân viên chính thức được nghỉ 15 ngày phép năm có lương."
    result = enrich_chunks([{"text": text, "metadata": {"source": "policy_demo.md", "parent_id": "demo_parent"}}])[0]
    assert result.auto_metadata["enrichment_status"] == "success"
    assert result.original_text == text and result.enriched_text.endswith(text)
    assert result.summary and result.hypothesis_questions
    assert result.auto_metadata["source"] == "policy_demo.md"
    assert result.auto_metadata["parent_id"] == "demo_parent"
    output = Path(__file__).resolve().parents[1] / "reports" / "m5_smoke_report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Combined enrichment successful; report saved to {output}")
