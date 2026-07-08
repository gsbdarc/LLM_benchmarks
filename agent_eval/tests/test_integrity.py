"""Tests for eval.integrity — save extraction, score consistency, retry detection."""

import json
from types import SimpleNamespace

from agent_eval import integrity


def assistant(*calls):
    """Build an assistant message with tool_calls = [(id, name, args_json), ...]."""
    return SimpleNamespace(tool_calls=[
        SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))
        for cid, name, args in calls
    ])


def tool_result(name, content, tool_call_id="x"):
    return {"role": "tool", "name": name, "content": content, "tool_call_id": tool_call_id}


def save_args(field, metric, scores):
    return json.dumps({"field_evaluations": [{"field": field, "metric": metric, "scores": scores}]})


def test_extract_saved_evaluation_counts():
    msgs = [
        assistant(("c1", "get_task_output", "{}")),
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
    ]
    saves, count = integrity.extract_saved_evaluation(msgs)
    assert count == 1
    assert saves[0]["field_evaluations"][0]["metric"] == "evaluate_raw_string"


def test_score_consistency_passes_when_scores_observed():
    msgs = [
        assistant(("c1", "evaluate_extracted_string", "{}")),
        tool_result("evaluate_extracted_string", '{"composite_score": 1.0, "null_accuracy": 1.0}', "c1"),
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_extracted_string", {"composite_score": 1.0}))),
    ]
    res = integrity.check_score_consistency(msgs)
    assert res["consistent"] is True
    assert res["missing"] == []


def test_score_consistency_flags_hallucinated_score():
    msgs = [
        assistant(("c1", "evaluate_extracted_string", "{}")),
        tool_result("evaluate_extracted_string", '{"composite_score": 1.0}', "c1"),
        # Agent saved 0.99, which no tool ever returned -> hallucination.
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_extracted_string", {"composite_score": 0.99}))),
    ]
    res = integrity.check_score_consistency(msgs)
    assert res["consistent"] is False
    assert res["missing"][0]["value"] == 0.99


def test_score_consistency_none_when_no_save():
    res = integrity.check_score_consistency([assistant(("c1", "evaluate_raw_string", "{}"))])
    assert res["consistent"] is None


def test_check_retry_rule_detects_retry():
    msgs = [
        assistant(("c1", "evaluate_raw_string", "{}")),
        tool_result("evaluate_raw_string", '{"error": "bad args"}', "c1"),
        assistant(("c2", "evaluate_raw_string", '{"predicted": "a", "expected": "a"}')),
        tool_result("evaluate_raw_string", '{"composite_score": 1.0}', "c2"),
    ]
    res = integrity.check_retry_rule(msgs)
    assert res["total_errors"] == 1
    assert res["tool_errors_with_retry"]["evaluate_raw_string"]["retried"] is True


def test_check_retry_rule_no_errors():
    msgs = [assistant(("c1", "evaluate_raw_string", "{}")), tool_result("evaluate_raw_string", '{"composite_score": 1.0}', "c1")]
    res = integrity.check_retry_rule(msgs)
    assert res["total_errors"] == 0


def test_run_integrity_report_happy_path():
    msgs = [
        assistant(("c1", "evaluate_extracted_string", "{}")),
        tool_result("evaluate_extracted_string", '{"composite_score": 1.0, "null_accuracy": 1.0}', "c1"),
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_extracted_string", {"composite_score": 1.0}))),
    ]
    report = integrity.run_integrity_report(
        {"messages": msgs, "stopped_reason": "answered"}, verbose=False,
    )
    assert report["save_count"] == 1
    assert report["save_failed"] == 0
    assert report["save_success"] is True
    assert report["score_consistency"]["consistent"] is True


def test_save_outcome_error_then_retry():
    # First save errors, agent retries, second succeeds: saved, but 1 attempt failed.
    msgs = [
        assistant(("c1", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
        tool_result("save_evaluation", '{"error": "db down", "saved": false}', "c1"),
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
        tool_result("save_evaluation", '{"saved": true, "n_fields": 1}', "c2"),
    ]
    assert integrity.save_outcome(msgs) == (2, 1, 1)
    rep = integrity.run_integrity_report({"messages": msgs, "stopped_reason": "answered"}, verbose=False)
    assert rep["save_success"] is True   # ultimately saved
    assert rep["save_count"] == 2 and rep["save_failed"] == 1  # nuance preserved


def test_save_outcome_redundant_double_save_is_clean():
    msgs = [
        assistant(("c1", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
        tool_result("save_evaluation", '{"saved": true}', "c1"),
        assistant(("c2", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
        tool_result("save_evaluation", '{"saved": true}', "c2"),
    ]
    rep = integrity.run_integrity_report({"messages": msgs, "stopped_reason": "answered"}, verbose=False)
    assert rep["save_success"] is True
    assert rep["save_count"] == 2 and rep["save_failed"] == 0


def test_run_integrity_report_all_saves_failed_is_not_success():
    msgs = [
        assistant(("c1", "save_evaluation", save_args("f", "evaluate_raw_string", {"composite_score": 1.0}))),
        tool_result("save_evaluation", '{"error": "db down", "saved": false}', "c1"),
    ]
    rep = integrity.run_integrity_report({"messages": msgs, "stopped_reason": "answered"}, verbose=False)
    assert rep["save_success"] is False and rep["save_failed"] == 1
