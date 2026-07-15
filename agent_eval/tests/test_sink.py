"""Tests for eval.sink — run flattening, Parquet round-trip, DuckDB query."""

import duckdb
import pandas as pd

from agent_eval.reporting import sink


def sample_result(**over):
    r = {
        "answer": "done",
        "usage": {"prompt_tokens": 220, "completion_tokens": 30, "total_tokens": 250},
        "messages": [],
        "steps": 5,
        "stopped_reason": "answered",
        "steps_detail": [],
        "tool_calls_by_name": {"get_task_output": 1, "evaluate_raw_string": 2, "save_evaluation": 1},
        "tool_time_by_name": {},
        "tool_errors_by_name": {"evaluate_raw_string": 1},
        "llm_time_total": 6.0,
        "wall_time_total": 7.0,
        "tokens_per_sec": 5.0,
        "peak_context": 1200,
        "client_ram_mb": 123.4,
        "gpu_start": {"gpu_cache_usage_perc": 0.1, "num_requests_running": 1.0, "num_requests_waiting": 0.0},
        "gpu_end": {"gpu_cache_usage_perc": 0.4, "num_requests_running": 2.0, "num_requests_waiting": 1.0},
    }
    r.update(over)
    return r


def sample_meta(**over):
    m = {
        "task_id": "2450", "run_id": 0, "benchmark_id": "5", "model_id": "1",
        "backend": "nim", "framework": "nim", "agent_model_key": 1,
        "model": "google/gemma-4-31b-it", "temperature": 0.1, "gpu_type": "NVIDIA H200",
        "reasoning_level": "thinking",
        "prompt_name": "composite_v1", "prompt_hash": "abcd1234", "tools_hash": "7540a7e5",
        "git_commit": "4d39558", "concurrency": 4, "mcp_url": "http://x/mcp",
        "weave_trace_url": "https://wandb.ai/...",
    }
    m.update(over)
    return m


def test_flatten_run_core_fields():
    row = sink.flatten_run(
        sample_result(), sample_meta(),
        integrity={"save_success": True, "save_count": 1, "save_failed": 0,
                   "score_consistency": {"consistent": True}},
        scores={"selection_accuracy": {"selection_accuracy": 1.0}},
    )
    assert row["task_id"] == "2450"
    assert row["save_failed"] == 0
    assert row["backend"] == "nim"
    assert row["reasoning_level"] == "thinking"
    assert row["tokens_per_sec"] == 5.0
    assert row["total_tokens"] == 250
    assert row["save_success"] is True
    assert row["score_consistent"] is True
    assert row["selection_accuracy"] == 1.0
    assert row["n_metric_calls"] == 2          # evaluate_raw_string x2 (data tools excluded)
    assert row["n_tool_errors"] == 1
    assert row["concurrency"] == 4


def test_flatten_run_instrumentation_fields():
    # Part 3: backend_id/framework/temperature/gpu_type flow from meta; reasoning
    # is derived from steps_detail into a bounded JSON blob.
    result = sample_result(steps_detail=[
        {"step": 1, "thinking_nim": "the value is a list of channels", "tool_calls": ["evaluate_list"]},
        {"step": 2, "thinking_nim": None, "tool_calls": ["save_evaluation"]},
    ])
    row = sink.flatten_run(result, sample_meta())
    assert row["agent_model_key"] == 1
    assert row["framework"] == "nim"
    assert row["temperature"] == 0.1
    assert row["gpu_type"] == "NVIDIA H200"
    import json
    reasoning = json.loads(row["reasoning_json"])
    assert reasoning[0]["step"] == 1
    assert "list of channels" in reasoning[0]["reasoning"]
    assert reasoning[0]["tool_calls"] == ["evaluate_list"]


def test_flatten_run_reasoning_json_none_when_no_thinking():
    row = sink.flatten_run(sample_result(steps_detail=[{"step": 1, "tool_calls": ["get_task_output"]}]),
                           sample_meta())
    assert row["reasoning_json"] is None


def test_flatten_run_tool_sequence_json():
    result = sample_result(steps_detail=[
        {"tool_calls": ["get_task_output"]},
        {"tool_calls": ["evaluate_raw_string"]},
        {"tool_calls": ["save_evaluation"]},
    ])
    row = sink.flatten_run(result, sample_meta())
    import json
    assert json.loads(row["tool_sequence_json"]) == ["get_task_output", "evaluate_raw_string", "save_evaluation"]


def test_flatten_run_gpu_fields():
    row = sink.flatten_run(sample_result(), sample_meta())
    assert row["gpu_cache_usage_start"] == 0.1
    assert row["gpu_cache_usage_end"] == 0.4
    assert row["requests_running_end"] == 2.0


def test_flatten_run_handles_gpu_error_and_none():
    row = sink.flatten_run(
        sample_result(gpu_start={"error": "down"}, gpu_end=None), sample_meta(),
    )
    assert row["gpu_cache_usage_start"] is None
    assert row["gpu_cache_usage_end"] is None


def test_flatten_run_selection_accuracy_scalar_or_dict():
    row_scalar = sink.flatten_run(sample_result(), sample_meta(), scores={"selection_accuracy": 0.5})
    assert row_scalar["selection_accuracy"] == 0.5


def test_write_run_row_creates_partitioned_file(tmp_path):
    row = sink.flatten_run(sample_result(), sample_meta())
    path = sink.write_run_row(row, base_dir=tmp_path)
    assert path.exists()
    assert "date=" in str(path.parent)
    back = pd.read_parquet(path)
    assert back.iloc[0]["task_id"] == "2450"


def test_duckdb_query_over_written_runs(tmp_path):
    # Write a few rows across two backends, then aggregate with DuckDB.
    rows = [
        sink.flatten_run(sample_result(tokens_per_sec=10.0), sample_meta(backend="nim", concurrency=1)),
        sink.flatten_run(sample_result(tokens_per_sec=20.0), sample_meta(backend="nim", concurrency=4)),
        sink.flatten_run(sample_result(tokens_per_sec=30.0), sample_meta(backend="playground", concurrency=1)),
    ]
    sink.write_runs(rows, base_dir=tmp_path)

    glob = str(tmp_path / "**" / "*.parquet")
    df = duckdb.sql(
        f"SELECT backend, avg(tokens_per_sec) AS tps FROM read_parquet('{glob}') "
        f"GROUP BY backend ORDER BY backend"
    ).df()
    assert list(df["backend"]) == ["nim", "playground"]
    nim_tps = df[df["backend"] == "nim"]["tps"].iloc[0]
    assert nim_tps == 15.0  # mean of 10 and 20
