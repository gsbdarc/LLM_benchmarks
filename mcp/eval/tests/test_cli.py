"""Tests for the CLI argument parser."""

import pytest

from eval.__main__ import parse_args


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
