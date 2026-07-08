"""Tests for eval.scorers — the four weave.Evaluation scorers."""

import json
from types import SimpleNamespace

from agent_eval import scorers


def assistant(*calls):
    return SimpleNamespace(tool_calls=[
        SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))
        for cid, name, args in calls
    ])


def save_args(field, metric, scores):
    return json.dumps({"field_evaluations": [{"field": field, "metric": metric, "scores": scores}]})


def make_output(field="first_channel", metric="word_iou", **extra):
    msgs = [assistant(("c2", "save_evaluation", save_args(field, metric, {metric: 1.0})))]
    out = {"messages": msgs, "stopped_reason": "answered"}
    out.update(extra)
    return out


def test_save_success_scorer():
    res = scorers.save_success_scorer(make_output())
    assert res["save_success"] is True
    assert res["save_count"] == 1


def test_save_success_scorer_no_save():
    res = scorers.save_success_scorer({"messages": [], "stopped_reason": "answered"})
    assert res["save_success"] is False
    assert res["save_count"] == 0


def test_efficiency_scorer_surfaces_derived_metrics():
    out = make_output(steps=5, usage={"total_tokens": 999}, tokens_per_sec=42.0, peak_context=1200,
                      tool_errors_by_name={"word_iou": 1})
    res = scorers.efficiency_scorer(out)
    assert res["steps"] == 5
    assert res["total_tokens"] == 999
    assert res["tool_errors"] == 1
    assert res["tokens_per_sec"] == 42.0
    assert res["peak_context"] == 1200


def test_selection_accuracy_correct_routing_from_tool_name():
    # gold is now the field_type; the chosen type is derived from the type-tool name.
    gold = {("5", "first_channel"): "raw_string"}
    out = make_output(field="first_channel", metric="evaluate_raw_string")
    res = scorers.selection_accuracy_scorer(out, benchmark_id="5", gold=gold)
    assert res["selection_accuracy"] == 1.0
    assert res["correct"] == 1


def test_selection_accuracy_wrong_routing():
    gold = {("5", "first_channel"): "raw_string"}
    out = make_output(field="first_channel", metric="evaluate_list")
    res = scorers.selection_accuracy_scorer(out, benchmark_id="5", gold=gold)
    assert res["selection_accuracy"] == 0.0


def test_selection_accuracy_uses_explicit_field_type():
    # An explicit field_type on the field_evaluation wins over the tool name.
    gold = {("7", "items"): "list"}
    args = json.dumps({"field_evaluations": [
        {"field": "items", "field_type": "list", "metric": "evaluate_list",
         "scores": {"composite_score": 1.0}}
    ]})
    out = {"messages": [assistant(("c2", "save_evaluation", args))], "stopped_reason": "answered"}
    res = scorers.selection_accuracy_scorer(out, benchmark_id="7", gold=gold)
    assert res["selection_accuracy"] == 1.0


def test_selection_accuracy_none_without_gold():
    out = make_output()
    assert scorers.selection_accuracy_scorer(out, benchmark_id="5", gold={}) is None


def test_load_gold_metrics_missing_file_returns_empty(tmp_path):
    assert scorers._load_gold_metrics(tmp_path / "nope.csv") == {}


def test_load_gold_metrics_reads_field_type(tmp_path):
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text(
        "benchmark_id,benchmark_name,field_name,field_type,gold_metric,all_metrics\n"
        "5,first_channel,first_channel,raw_string,word_iou,word_iou\n"
    )
    gold = scorers._load_gold_metrics(csv_path)
    assert gold[("5", "first_channel")] == "raw_string"
