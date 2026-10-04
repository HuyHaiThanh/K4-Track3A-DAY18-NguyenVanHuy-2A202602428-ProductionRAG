"""Opt-in real RAGAS smoke evaluation; not the 20-question pipeline report.

PowerShell: $env:RUN_M4_LIVE='1'; python -m pytest tests/test_m4_live.py -v -s
Requires OPENAI_API_KEY in .env and incurs evaluator API usage.
"""
import os
from pathlib import Path

import pytest

from src.m4_eval import evaluate_ragas, failure_analysis, save_report, METRIC_NAMES


@pytest.mark.skipif(os.getenv("RUN_M4_LIVE") != "1", reason="Set RUN_M4_LIVE=1 for API evaluation")
def test_real_ragas_smoke():
    result = evaluate_ragas(
        ["Theo chính sách v2024, nhân viên chính thức được nghỉ bao nhiêu ngày phép năm?"],
        ["Theo chính sách v2024, nhân viên chính thức được nghỉ 15 ngày phép năm có lương."],
        [["Chính sách nghỉ phép năm v2024: mỗi nhân viên chính thức được hưởng 15 ngày phép năm có lương."]],
        ["Nhân viên chính thức được nghỉ 15 ngày phép năm có lương theo chính sách v2024."])
    assert result["status"] == "success", f"RAGAS did not succeed: {result.get('error', result['status'])}"
    assert len(result["per_question"]) == 1
    assert all(0 <= result[name] <= 1 for name in METRIC_NAMES)
    output = Path(__file__).resolve().parents[1] / "reports" / "m4_smoke_report.json"
    save_report(result, failure_analysis(result["per_question"], bottom_n=1), str(output))
