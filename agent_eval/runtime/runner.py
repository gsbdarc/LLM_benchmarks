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
from ..prompts import METRIC_EVAL_SYSTEM, PROMPT_NAME, eval_user_prompt
from ..reporting.scorers import _load_gold_metrics, selection_accuracy_scorer
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
    gpu_type: Optional[str] = None,
) -> dict[str, Any]:
    """Async setup: returns a context dict used by run_batch.

    `model` overrides the backend's default model; `gpu_type` is the GPU name of
    the (cross-host) model server, propagated explicitly for the run rows.
    """
    # Resolve the model key from the original selector before build_backend
    # reassigns `model` to the resolved model string.
    _, _, agent_model_key = config.resolve_model(backend, model)
    client, model, completion_kwargs, base_url = config.build_backend(backend, model)
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
        "prompt_hash": compute_prompt_hash(METRIC_EVAL_SYSTEM),
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
) -> dict[str, Any]:
    """Run the agent on one row, score it, and write a Parquet row. Returns the row dict."""
    safe_run_id = int(row["run_id"]) if row.get("run_id") is not None else 0
    metrics_url = config.metrics_url(ctx["base_url"]) if gpu_metrics else None

    attrs = {
        "prompt_hash": ctx["prompt_hash"],
        "prompt_name": PROMPT_NAME,
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
            eval_user_prompt(row["task_id"], safe_run_id),
            METRIC_EVAL_SYSTEM,
            ctx["llm_step"],
            ctx["mcp_url"],
            max_steps=max_steps,
            verbose=verbose,
            backend=ctx["backend"],
            model=ctx["model"],
            task_id=row["task_id"],
            run_id=safe_run_id,
            metrics_url=metrics_url,
        )

    integrity = run_integrity_report(result, verbose=verbose)
    sel = selection_accuracy_scorer(result, benchmark_id=row["benchmark_id"], gold=gold)

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
        "prompt_name": PROMPT_NAME,
        "prompt_hash": ctx["prompt_hash"],
        "tools_hash": ctx["tools_hash"],
        "git_commit": ctx["git_commit"],
        "concurrency": concurrency,
        "mcp_url": ctx["mcp_url"],
        "weave_trace_url": result.get("weave_trace_url"),
    }
    if write_sink:
        path = write_run_row(flatten_run(result, meta, integrity, {"selection_accuracy": sel}))
        result["_sink_path"] = str(path)

    print(f"[task {row['task_id']} run {safe_run_id}] steps={result['steps']} "
          f"tokens={result['usage']['total_tokens']} tps={result.get('tokens_per_sec')} "
          f"wall={result['wall_time_total']:.1f}s stopped={result['stopped_reason']} "
          f"save_success={integrity['save_success']}")
    return result


async def run_batch(rows: list[dict[str, Any]], ctx: dict[str, Any], concurrency: int = 2,
                    max_steps: int = config.MAX_STEPS, verbose: bool = False,
                    write_sink: bool = True, gpu_metrics: bool = True) -> list[dict[str, Any]]:
    """Run all rows with bounded parallelism (asyncio.gather + Semaphore)."""
    gold = _load_gold_metrics()
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await run_one(row, ctx, gold, concurrency, max_steps, verbose, write_sink, gpu_metrics)

    return await asyncio.gather(*[_guarded(r) for r in rows])
