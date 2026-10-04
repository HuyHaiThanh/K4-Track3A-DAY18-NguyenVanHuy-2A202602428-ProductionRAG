"""Validate evaluation plumbing and diagnostic logic without API charges."""
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from src import m4_eval as m4


@pytest.fixture
def evaluator(monkeypatch):
    import ragas
    import langchain_openai

    monkeypatch.setattr(m4, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", Mock())
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", Mock())
    evaluate = Mock()
    monkeypatch.setattr(ragas, "evaluate", evaluate)
    return evaluate


def inputs():
    return (["Q1", "Q2"], ["A1", "A2"], [["C1"], ["C2", "C3"]], ["GT1", "GT2"])


def test_success_means_rows_and_dataset(evaluator):
    evaluator.return_value.to_pandas.return_value = pd.DataFrame([
        dict(zip(m4.METRIC_NAMES, [0.2, 0.4, 0.6, 0.8])),
        dict(zip(m4.METRIC_NAMES, [0.8, 0.6, 0.4, 0.2]))])
    result = m4.evaluate_ragas(*inputs())
    assert result["status"] == "success"
    assert all(result[name] == pytest.approx(0.5) for name in m4.METRIC_NAMES)
    assert result["valid_counts"] == dict.fromkeys(m4.METRIC_NAMES, 2)
    assert result["per_question"][1].contexts == ["C2", "C3"]
    dataset = evaluator.call_args.args[0]
    assert dataset["question"] == ["Q1", "Q2"]
    assert set(metric.name for metric in evaluator.call_args.kwargs["metrics"]) == set(m4.METRIC_NAMES)
    assert evaluator.call_args.kwargs["raise_exceptions"] is True


def test_missing_key_is_skipped(monkeypatch):
    monkeypatch.setattr(m4, "OPENAI_API_KEY", "")
    result = m4.evaluate_ragas(*inputs())
    assert result["status"] == "skipped" and result["error"] == "missing_api_key"
    assert result["per_question"] == []
    assert all(result[name] == 0 for name in m4.METRIC_NAMES)


def test_empty_dataset_is_skipped():
    assert m4.evaluate_ragas([], [], [], [])["error"] == "empty_dataset"


@pytest.mark.parametrize("args", [(["q"], [], [[]], ["gt"]),
                                 ([1], ["a"], [["c"]], ["gt"]),
                                 (["q"], ["a"], ["c"], ["gt"]),
                                 (["q"], ["a"], [[1]], ["gt"])])
def test_invalid_input_fails_before_provider(args):
    with pytest.raises(ValueError):
        m4.evaluate_ragas(*args)


def test_api_failure_is_marked_and_secret_not_logged(evaluator, capsys):
    evaluator.side_effect = RuntimeError("sensitive-test-token")
    result = m4.evaluate_ragas(*inputs())
    assert result["status"] == "error" and result["error"] == "RuntimeError"
    assert "sensitive-test-token" not in capsys.readouterr().out
    assert "sensitive-test-token" not in str(result)


def test_missing_invalid_metrics_are_partial_not_real_zero(evaluator):
    evaluator.return_value.to_pandas.return_value = pd.DataFrame([
        {"faithfulness": 0.8, "answer_relevancy": float("nan"),
         "context_precision": float("inf"), "context_recall": 1.5},
        {"faithfulness": 0.4, "answer_relevancy": 0.6, "context_recall": 0.2}])
    result = m4.evaluate_ragas(*inputs())
    assert result["status"] == "partial"
    assert result["faithfulness"] == pytest.approx(0.6)
    assert result["answer_relevancy"] == pytest.approx(0.6)
    assert result["valid_counts"]["context_precision"] == 0
    assert result["per_question"][0].answer_relevancy is None
    assert m4.failure_analysis(result["per_question"]) == []


def test_wrong_row_count_is_error(evaluator):
    evaluator.return_value.to_pandas.return_value = pd.DataFrame()
    assert m4.evaluate_ragas(*inputs())["status"] == "error"


@pytest.mark.parametrize("worst", m4.METRIC_NAMES)
def test_diagnostic_mapping_for_each_metric(worst):
    scores = dict.fromkeys(m4.METRIC_NAMES, 0.9)
    scores[worst] = 0.1
    row = m4.EvalResult("q", "a", ["c"], "gt", **scores)
    failure = m4.failure_analysis([row], bottom_n=1)[0]
    assert failure["worst_metric"] == worst and failure["worst_score"] == 0.1
    assert failure["score"] == pytest.approx(0.7)
    assert failure["diagnosis"] and failure["suggested_fix"] and failure["error_tree"]
    assert failure["answer"] == "a" and failure["contexts"] == ["c"]


def test_bottom_n_sorting_and_incomplete_results():
    rows = [m4.EvalResult("high", "a", [], "gt", 0.8, 0.8, 0.8, 0.8),
            m4.EvalResult("low", "a", [], "gt", 0.1, 0.2, 0.3, 0.4),
            m4.EvalResult("invalid", "a", [], "gt", None, 0, 0, 0)]
    assert [f["question"] for f in m4.failure_analysis(rows, 1)] == ["low"]
    assert m4.failure_analysis(rows, 0) == []
    assert m4.failure_analysis([]) == []


def test_report_preserves_trace_and_strict_json(evaluator, tmp_path):
    evaluator.return_value.to_pandas.return_value = pd.DataFrame([
        dict.fromkeys(m4.METRIC_NAMES, 0.5), dict.fromkeys(m4.METRIC_NAMES, float("nan"))])
    result = m4.evaluate_ragas(*inputs())
    path = tmp_path / "nested" / "report.json"
    m4.save_report(result, m4.failure_analysis(result["per_question"]), str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "partial" and report["num_questions"] == 2
    assert set(report["aggregate"]) == set(m4.METRIC_NAMES)
    assert report["per_question"][1]["faithfulness"] is None
    assert report["per_question"][0]["question"] == "Q1"
    assert len(report["failures"]) == 1
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_fallback_report_does_not_claim_success(monkeypatch, tmp_path):
    monkeypatch.setattr(m4, "OPENAI_API_KEY", "")
    result = m4.evaluate_ragas(*inputs())
    path = tmp_path / "fallback.json"
    m4.save_report(result, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "skipped"
    assert report["num_questions"] == 0 and report["requested_questions"] == 2
