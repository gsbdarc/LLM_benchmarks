"""Tests for the CLI argument parser."""

from types import SimpleNamespace

import pytest

from agent_eval import mapping
from agent_eval.__main__ import _row_mode_work, parse_args


def test_defaults():
    args = parse_args(["--mcp-url", "http://x/mcp"])
    assert args.backend == "nim"
    assert args.limit == 5
    assert args.concurrency == 2
    assert args.no_weave is False
    assert args.model is None
    assert args.gpu_type is None


def test_model_and_gpu_type_flags():
    args = parse_args(["--mcp-url", "http://x/mcp", "--model", "gpt-4o",
                       "--gpu-type", "NVIDIA H200"])
    assert args.model == "gpt-4o"
    assert args.gpu_type == "NVIDIA H200"


def test_benchmarks_and_flags():
    args = parse_args([
        "--mcp-url", "http://x/mcp", "--backend", "playground",
        "--benchmarks", "5,6,7", "--concurrency", "4", "--no-weave", "--no-sink",
    ])
    assert args.backend == "playground"
    assert args.benchmarks == "5,6,7"
    assert args.concurrency == 4
    assert args.no_weave is True
    assert args.no_sink is True


def test_mcp_url_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_invalid_backend_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--mcp-url", "http://x/mcp", "--backend", "nope"])


def test_row_mode_flags_default_none():
    args = parse_args(["--mcp-url", "http://x/mcp"])
    assert args.eval_mapping is None
    assert args.row is None


def test_row_mode_work_reads_row_and_takes_judge_from_mapping(tmp_path):
    path = tmp_path / "eval_mapping_sample.csv"
    rows = mapping.dedupe_and_assign([], mapping.build_rows(
        [{"task_id": "2450", "run_id": 0, "benchmark_id": "5", "model_id": "1"},
         {"task_id": "2451", "run_id": None, "benchmark_id": "7", "model_id": "1"}],
        [{"judge_backend": "playground", "judge_model": "gpt-5-mini", "judge_prompt": "composite_v1"}]))
    mapping.write_csv(path, rows)

    args = SimpleNamespace(eval_mapping=str(path), row=1, backend="nim", model=None)
    backend, model, work = _row_mode_work(args)
    assert backend == "playground"          # taken from the row, not the --backend default
    assert model == "gpt-5-mini"
    assert len(work) == 1
    assert work[0]["task_id"] == "2451"
    assert work[0]["run_id"] is None        # empty run_id coerced back to None
    assert work[0]["benchmark_id"] == "7"
