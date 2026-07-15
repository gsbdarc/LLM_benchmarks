"""Tests for analysis.queries — DuckDB views and aggregations over Parquet runs."""

import pytest

from analysis import queries
from agent_eval.reporting import sink
from agent_eval.tests.test_sink import sample_meta, sample_result


def _write_runs(base_dir):
    rows = [
        sink.flatten_run(sample_result(tokens_per_sec=10.0),
                         sample_meta(backend="nim", concurrency=1, model="gemma"),
                         scores={"selection_accuracy": {"selection_accuracy": 1.0}}),
        sink.flatten_run(sample_result(tokens_per_sec=20.0),
                         sample_meta(backend="nim", concurrency=4, model="gemma"),
                         scores={"selection_accuracy": {"selection_accuracy": 0.5}}),
        sink.flatten_run(sample_result(tokens_per_sec=30.0),
                         sample_meta(backend="playground", concurrency=1, model="gpt-5-mini"),
                         scores={"selection_accuracy": {"selection_accuracy": 1.0}}),
    ]
    sink.write_runs(rows, base_dir=base_dir)


def test_connect_raises_when_no_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        queries.connect(tmp_path)


def test_summary_by_backend(tmp_path):
    _write_runs(tmp_path)
    con = queries.connect(tmp_path)
    df = queries.summary_by(con, "backend")
    backends = set(df["backend"])
    assert backends == {"nim", "playground"}
    nim_row = df[df["backend"] == "nim"].iloc[0]
    assert nim_row["n_runs"] == 2
    assert nim_row["avg_tokens_per_sec"] == 15.0


def test_concurrency_summary(tmp_path):
    _write_runs(tmp_path)
    con = queries.connect(tmp_path)
    df = queries.concurrency_summary(con)
    levels = sorted(df["concurrency"])
    assert levels == [1, 4]
    c1 = df[df["concurrency"] == 1].iloc[0]
    assert c1["n_runs"] == 2  # nim c1 + playground c1


def test_infra_summary_groups_by_framework_gpu_concurrency(tmp_path):
    _write_runs(tmp_path)
    con = queries.connect(tmp_path)
    df = queries.infra_summary(con)
    # sample_meta stamps framework="nim", gpu_type="NVIDIA H200"
    assert {"framework", "gpu_type", "concurrency"} <= set(df.columns)
    assert set(df["framework"]) == {"nim"}
    assert df["n_runs"].sum() == 3
