"""
runner.py — reusable orchestration for the metric-eval agent.

Holds the batch logic so both the CLI (__main__.py) and the sweep
(analysis/sweep.py) can drive runs without duplicating setup. Setup -> discover
dataset rows -> run agents (bounded parallelism) -> integrity + selection scoring
-> flatten to the Parquet sink.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from typing import Any, Optional

from .. import config
from .agent import call_mcp_tool, load_tools_from_mcp, make_llm_step, run_agent, use_session
from ..reporting.integrity import run_integrity_report
from ..reporting.observability import compute_prompt_hash, compute_tools_hash, get_git_commit
from ..prompts import resolve_prompt
from ..reporting.scorers import _load_gold_metrics, routing_path_scorer, selection_accuracy_scorer
from ..reporting.sink import flatten_run, write_run_row


def _weave_attrs(enabled: bool, attrs: dict[str, Any]) -> Any:
    """weave.attributes(attrs) if tracing is on, else a no-op context."""
    if enabled:
        import weave

        return weave.attributes(attrs)
    return nullcontext()


async def aprepare(
    backend: str,
    mcp_url: str,
    weave_enabled: bool = True,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    gpu_type: Optional[str] = None,
) -> dict[str, Any]:
    """Async setup: returns a context dict used by run_batch.

    `model` overrides the backend's default model; `prompt` selects a prompt variant
    (name or index; default = prompt #1); `gpu_type` is the GPU name of the
    (cross-host) model server, propagated explicitly for the run rows.
    """
    # Resolve the model key from the original selector before build_backend
    # reassigns `model` to the resolved model string.
    _, _, agent_model_key = config.resolve_model(backend, model)
    client, model, completion_kwargs, base_url = config.build_backend(backend, model)
    prompt_name, prompt_system, prompt_user, prompt_key = resolve_prompt(prompt)
    tools = await load_tools_from_mcp(mcp_url)
    return {
        "client": client,
        "model": model,
        "agent_model_key": agent_model_key,
        "completion_kwargs": completion_kwargs,
        "base_url": base_url,
        "tools": tools,
        "tools_hash": compute_tools_hash(tools),
        "git_commit": get_git_commit(),
        "prompt_name": prompt_name,
        "prompt_system": prompt_system,
        "prompt_user": prompt_user,
        "prompt_key": prompt_key,
        "prompt_hash": compute_prompt_hash(prompt_system),
        "reasoning_level": config.reasoning_level(backend, completion_kwargs),
        "llm_step": make_llm_step(client, model, tools, completion_kwargs),
        "backend": backend,
        "framework": config.framework(backend),
        "temperature": completion_kwargs.get("temperature"),
        "gpu_type": config.gpu_type(gpu_type),
        "mcp_url": mcp_url,
        "weave_enabled": weave_enabled,
    }


async def discover_rows(mcp_url: str, benchmark_ids: list[str], limit: int) -> list[dict[str, Any]]:
    """Use list_outputs over MCP to build the work list (one row per output)."""
    rows = []
    async with use_session(mcp_url):
        for bid in benchmark_ids:
            text = await call_mcp_tool("list_outputs", {"benchmark_id": bid, "limit": limit}, verbose=False)
            listing = json.loads(text)
            for it in listing.get("outputs", []):
                rows.append({
                    "task_id": it["task_id"],
                    "run_id": it.get("run_id"),
                    "benchmark_id": it.get("benchmark_id", bid),
                    "model_id": it.get("model_id"),
                })
            print(f"  benchmark {bid}: {len(listing.get('outputs', []))} outputs")
    return rows


async def run_one(
    row: dict[str, Any],
    ctx: dict[str, Any],
    gold: Any,
    concurrency: int,
    max_steps: int,
    verbose: bool,
    write_sink: bool,
    gpu_metrics: bool,
    write_mongo: bool = True,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run the agent on one row, score it, and write a Parquet row (and, unless
    write_mongo is off, mirror it to the central agentic_runs collection). Returns
    the row dict.

    When write_mongo and skip_existing, a row already evaluated under this code_version
    (git_commit) is SKIPPED — no agent run, no API spend — so re-runs only do new work."""
    safe_run_id = int(row["run_id"]) if row.get("run_id") is not None else 0
    metrics_url = config.metrics_url(ctx["base_url"]) if gpu_metrics else None

    # Skip-if-exists: don't re-run (or re-spend on) a (task × judge × prompt) already
    # evaluated at this code_version. Keyed the same way save_run_row keys agentic_runs.
    if write_mongo and skip_existing:
        from ..tools import run_exists
        identity = {
            "eval_id": row.get("eval_id"),
            "task_id": row["task_id"], "run_id": safe_run_id,
            "benchmark_id": row["benchmark_id"], "model_id": row.get("model_id"),
            "backend": ctx["backend"], "agent_model_key": ctx.get("agent_model_key"),
            "prompt_key": ctx["prompt_key"], "git_commit": ctx["git_commit"],
        }
        try:
            if run_exists(identity):
                print(f"[task {row['task_id']} run {safe_run_id}] SKIP — already evaluated "
                      f"at {ctx['git_commit']}")
                return {"stopped_reason": "skipped", **identity}
        except Exception as e:  # noqa: BLE001 — a check failure must not block the run
            print(f"  WARNING: skip-existing check failed ({e}); running anyway")

    attrs = {
        "prompt_hash": ctx["prompt_hash"],
        "prompt_name": ctx["prompt_name"],
        "tools_hash": ctx["tools_hash"],
        "git_commit": ctx["git_commit"],
        "completion_kwargs": json.dumps(ctx["completion_kwargs"], default=str),
        "mcp_url": ctx["mcp_url"],
        "concurrency": concurrency,
        "task_id": row["task_id"],
        "run_id": safe_run_id,
    }
    with _weave_attrs(ctx["weave_enabled"], attrs):
        result = await run_agent(
            ctx["prompt_user"].format(task_id=row["task_id"], run_id=safe_run_id),
            ctx["prompt_system"],
            ctx["llm_step"],
            ctx["mcp_url"],
            max_steps=max_steps,
            verbose=verbose,
            backend=ctx["backend"],
            model=ctx["model"],
            task_id=row["task_id"],
            run_id=safe_run_id,
            metrics_url=metrics_url,
            eval_id=row.get("eval_id"),
            git_commit=ctx["git_commit"],
        )

    integrity = run_integrity_report(result, verbose=verbose)
    sel = selection_accuracy_scorer(result, benchmark_id=row["benchmark_id"], gold=gold)
    path = routing_path_scorer(result, benchmark_id=row["benchmark_id"], gold=gold)

    meta = {
        **row,
        "run_id": safe_run_id,
        "backend": ctx["backend"],
        "framework": ctx.get("framework"),
        "model": ctx["model"],
        "agent_model_key": ctx.get("agent_model_key"),
        "temperature": ctx.get("temperature"),
        "gpu_type": ctx.get("gpu_type"),
        "reasoning_level": ctx["reasoning_level"],
        "prompt_name": ctx["prompt_name"],
        "prompt_key": ctx["prompt_key"],
        "prompt_hash": ctx["prompt_hash"],
        "tools_hash": ctx["tools_hash"],
        "git_commit": ctx["git_commit"],
        "concurrency": concurrency,
        "mcp_url": ctx["mcp_url"],
        "weave_trace_url": result.get("weave_trace_url"),
    }
    if write_sink or write_mongo:
        flat = flatten_run(result, meta, integrity, {"selection_accuracy": sel, "routing_path": path})
    if write_sink:
        path = write_run_row(flat)
        result["_sink_path"] = str(path)
    if write_mongo:
        # Central, team-queryable mirror. Never let a Mongo hiccup fail the eval —
        # the Parquet row is the authoritative local record either way.
        try:
            from ..tools import save_run_row
            save_run_row(flat)
            result["_mongo_run_saved"] = True
        except Exception as e:  # noqa: BLE001
            result["_mongo_run_saved"] = False
            print(f"  WARNING: agentic_runs mirror failed: {e}")

    print(f"[task {row['task_id']} run {safe_run_id}] steps={result['steps']} "
          f"tokens={result['usage']['total_tokens']} tps={result.get('tokens_per_sec')} "
          f"wall={result['wall_time_total']:.1f}s stopped={result['stopped_reason']} "
          f"save_success={integrity['save_success']}")
    return result


async def run_batch(rows: list[dict[str, Any]], ctx: dict[str, Any], concurrency: int = 2,
                    max_steps: int = config.MAX_STEPS, verbose: bool = False,
                    write_sink: bool = True, gpu_metrics: bool = True,
                    write_mongo: bool = True, skip_existing: bool = True) -> list[dict[str, Any]]:
    """Run all rows with bounded parallelism (asyncio.gather + Semaphore)."""
    gold = _load_gold_metrics()
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await run_one(row, ctx, gold, concurrency, max_steps, verbose,
                                 write_sink, gpu_metrics, write_mongo, skip_existing)

    return await asyncio.gather(*[_guarded(r) for r in rows])
