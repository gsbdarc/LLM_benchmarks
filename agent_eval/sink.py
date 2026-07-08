"""
sink.py — flatten one agent run into one Parquet row (plan §3, the OLAP layer).

Weave stays the trace store (drill into a single run's reasoning). This module is
the aggregate store: one flat row per run, appended to a date-partitioned Parquet
dataset that DuckDB queries directly. Each run writes its OWN small file
(run-<task>-<run>-<uuid>.parquet) so concurrent workers never read-modify-write
the same file.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import AGENT_RUNS_DIR
from .observability import reasoning_blob, tool_sequence


def _gpu(gpu, key):
    """Safely pull a metric from a gpu scrape dict (which may be None or {'error':...})."""
    if not isinstance(gpu, dict) or "error" in gpu:
        return None
    return gpu.get(key)


def flatten_run(result, meta, integrity=None, scores=None):
    """Build one flat row dict from a run_agent result + context.

    meta carries config/identifier fields (backend, model, prompt_name, concurrency,
    task_id, run_id, benchmark_id, model_id, weave_trace_url, ...). integrity is the
    run_integrity_report dict; scores may carry e.g. selection_accuracy.
    """
    integrity = integrity or {}
    scores = scores or {}
    usage = result.get("usage", {}) or {}
    calls = result.get("tool_calls_by_name", {}) or {}
    errors = result.get("tool_errors_by_name", {}) or {}

    # The agent-facing scoring tools are now the three composite type-tools.
    metric_tool_names = {
        "evaluate_raw_string", "evaluate_extracted_string", "evaluate_list",
    }
    n_metric_calls = sum(n for k, n in calls.items() if k in metric_tool_names)

    consistency = integrity.get("score_consistency", {}) or {}

    row = {
        # ── identifiers ──
        "task_id": meta.get("task_id"),
        "run_id": meta.get("run_id"),
        "benchmark_id": meta.get("benchmark_id"),
        "model_id": meta.get("model_id"),
        # ── config ──
        "backend": meta.get("backend"),
        "framework": meta.get("framework"),
        "model": meta.get("model"),
        "agent_model_key": meta.get("agent_model_key"),
        "temperature": meta.get("temperature"),
        "gpu_type": meta.get("gpu_type"),
        "reasoning_level": meta.get("reasoning_level"),
        "prompt_name": meta.get("prompt_name"),
        "prompt_hash": meta.get("prompt_hash"),
        "tools_hash": meta.get("tools_hash"),
        "git_commit": meta.get("git_commit"),
        "concurrency": meta.get("concurrency"),
        "mcp_url": meta.get("mcp_url"),
        # ── performance ──
        "steps": result.get("steps"),
        "stopped_reason": result.get("stopped_reason"),
        "wall_time_total": result.get("wall_time_total"),
        "llm_time_total": result.get("llm_time_total"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "peak_context": result.get("peak_context"),
        "client_ram_mb": result.get("client_ram_mb"),
        # ── GPU / queue (from /v1/metrics scrape) ──
        "gpu_cache_usage_start": _gpu(result.get("gpu_start"), "gpu_cache_usage_perc"),
        "gpu_cache_usage_end": _gpu(result.get("gpu_end"), "gpu_cache_usage_perc"),
        "requests_running_start": _gpu(result.get("gpu_start"), "num_requests_running"),
        "requests_running_end": _gpu(result.get("gpu_end"), "num_requests_running"),
        "requests_waiting_start": _gpu(result.get("gpu_start"), "num_requests_waiting"),
        "requests_waiting_end": _gpu(result.get("gpu_end"), "num_requests_waiting"),
        # ── outcomes ──
        "save_success": integrity.get("save_success"),
        "save_count": integrity.get("save_count"),
        "save_failed": integrity.get("save_failed"),
        "score_consistent": consistency.get("consistent"),
        "selection_accuracy": (scores.get("selection_accuracy") or {}).get("selection_accuracy")
        if isinstance(scores.get("selection_accuracy"), dict) else scores.get("selection_accuracy"),
        # ── tool detail ──
        "n_tool_calls": sum(calls.values()),
        "n_metric_calls": n_metric_calls,
        "n_tool_errors": sum(errors.values()),
        "tool_calls_json": json.dumps(calls, sort_keys=True),
        "tool_sequence_json": json.dumps(tool_sequence(result.get("steps_detail"))),
        # ── reasoning (bounded; feeds Part 4 path-summaries) ──
        "reasoning_json": reasoning_blob(result.get("steps_detail")),
        # ── trace + time ──
        "weave_trace_url": meta.get("weave_trace_url"),
        "evaluated_at": datetime.now(timezone.utc),
    }
    return row


def write_run_row(row, base_dir=AGENT_RUNS_DIR):
    """Append one row as its own Parquet file under base_dir/date=YYYY-MM-DD/.

    Returns the written file path. Per-file writes are append-safe under
    concurrency (no shared file is mutated).
    """
    base_dir = Path(base_dir)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    part_dir = base_dir / f"date={date_str}"
    part_dir.mkdir(parents=True, exist_ok=True)

    fname = f"run-{row.get('task_id')}-{row.get('run_id')}-{uuid.uuid4().hex[:8]}.parquet"
    path = part_dir / fname
    pd.DataFrame([row]).to_parquet(path, engine="pyarrow", index=False)
    return path


def write_runs(rows, base_dir=AGENT_RUNS_DIR):
    """Write many flat rows (one file per row). Returns list of paths."""
    return [write_run_row(r, base_dir=base_dir) for r in rows]
