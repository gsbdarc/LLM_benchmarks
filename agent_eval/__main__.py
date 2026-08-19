"""
__main__.py — CLI entrypoint for the metric-eval agent.

Examples
--------
Start the MCP server first (./run_metric_mcp.sh) and a NIM server, then:

    # one output, parity check
    python -m agent_eval --backend nim --mcp-url http://127.0.0.1:PORT/mcp \\
        --benchmarks 5 --limit 1

    # parallel batch over several benchmarks
    python -m agent_eval --backend nim --mcp-url http://127.0.0.1:PORT/mcp \\
        --benchmarks 5,6,7,10,11 --limit 5 --concurrency 4

Results land as one Parquet row per run under outputs/agent_runs/date=.../ and,
unless --no-weave, as a full trace in Weave.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dotenv import load_dotenv

from . import config
from .registry import mapping
from .runtime.runner import aprepare, discover_rows, run_batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments for the agentic metric-eval entrypoint."""
    p = argparse.ArgumentParser(prog="eval", description="Agentic metric evaluation")
    p.add_argument("--backend", choices=list(config.BACKENDS), default="nim")
    p.add_argument("--model", default=None,
                   help="override the backend's default model (same endpoint, different model)")
    p.add_argument("--prompt", default=None,
                   help="prompt variant name or index (default: prompt #1)")
    p.add_argument("--gpu-type", default=None,
                   help="GPU model name of the (cross-host) server; else read from $GPU_TYPE")
    p.add_argument("--mcp-url", required=True, help="URL printed by ./run_metric_mcp.sh")
    p.add_argument("--benchmarks", default="5", help="comma-separated benchmark ids, e.g. 5,6,7")
    p.add_argument("--limit", type=int, default=5, help="max outputs per benchmark")
    p.add_argument("--concurrency", type=int, default=2, help="max concurrent agent runs")
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--verbose", action="store_true", help="print the full agent transcript")
    p.add_argument("--no-weave", action="store_true", help="disable Weave tracing")
    p.add_argument("--no-sink", action="store_true", help="do not write Parquet rows")
    p.add_argument("--no-mongo-runs", action="store_true",
                   help="do not mirror run rows to the central agentic_runs collection")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="re-run rows even if already evaluated at this code_version (git_commit)")
    p.add_argument("--no-gpu-metrics", action="store_true", help="skip /v1/metrics scrape")
    # SLURM-array worker mode: evaluate exactly the one output named by a row of an
    # eval_mapping.csv (the backend/model come from that row's judge config).
    p.add_argument("--eval-mapping", default=None,
                   help="path to eval_mapping.csv; with --row, evaluate just that row")
    p.add_argument("--row", type=int, default=None,
                   help="0-based row index into --eval-mapping (e.g. $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--rows", default=None,
                   help="0-based INCLUSIVE slice of --eval-mapping, e.g. '0-49'. One shard "
                        "of an array job: many rows, one agent run each. The slice must "
                        "share one judge config.")
    return p.parse_args(argv)


def _work_row(row: dict[str, Any]) -> dict[str, Any]:
    """One mapping CSV row as a work row. `expected_date`/`original_value` are the
    date-fix task's grading inputs — read by the scorer, never shown to the agent,
    and absent (None) for metric-eval mappings."""
    return {
        "eval_id": int(row["eval_id"]) if row.get("eval_id") not in (None, "") else None,
        "task_id": row["task_id"],
        "run_id": mapping.coerce_run_id(row.get("run_id")),
        "benchmark_id": row["benchmark_id"],
        "model_id": row.get("model_id"),
        "expected_date": row.get("expected_date") or None,
        "original_value": row.get("original_value") or None,
    }


def _row_mode_work(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
    """Resolve (backend, model, prompt, work-rows) from one eval_mapping row. The row's
    judge config is authoritative so the array does exactly what the mapping says."""
    row = mapping.read_mapping_row(args.eval_mapping, args.row)
    return (row.get("judge_backend") or args.backend,
            row.get("judge_model") or args.model,
            row.get("judge_prompt") or args.prompt,
            [_work_row(row)])


def _rows_mode_work(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
    """Resolve a SLICE of mapping rows ("A-B", inclusive) as one shard of work.

    One process per shard, one agent run per row, so per-task attribution survives.
    The whole slice must share a judge config — one context is built for it — which
    is why mapping files are written grouped by judge. Spanning configs is an error
    rather than a silent half-run.
    """
    start, _, end = args.rows.partition("-")
    try:
        lo, hi = int(start), int(end if end else start)
    except ValueError:
        raise SystemExit(f"--rows expects 'A-B' (0-based, inclusive), got {args.rows!r}")
    if lo > hi:
        raise SystemExit(f"--rows start {lo} is after end {hi}")

    all_rows = mapping.read_csv(args.eval_mapping)
    if lo >= len(all_rows):
        raise SystemExit(f"--rows {args.rows}: mapping has only {len(all_rows)} rows")
    chunk = all_rows[lo:hi + 1]

    configs = {(r.get("judge_backend"), r.get("judge_model"), r.get("judge_prompt")) for r in chunk}
    if len(configs) > 1:
        raise SystemExit(
            f"--rows {args.rows} spans {len(configs)} judge configs {sorted(configs)}; "
            "shard within one config (mapping files are written grouped by judge)"
        )
    backend, model, prompt = configs.pop()
    print(f"Shard rows {lo}-{min(hi, len(all_rows) - 1)} ({len(chunk)} rows) of {args.eval_mapping}")
    return (backend or args.backend, model or args.model, prompt or args.prompt,
            [_work_row(r) for r in chunk])


async def _amain(args: argparse.Namespace) -> None:
    """Prepare the backend/agent context and run the (batch or single-row) eval."""
    load_dotenv()
    weave_enabled = not args.no_weave
    if weave_enabled:
        import weave

        weave.init(config.WEAVE_PROJECT)
        print(f"Weave project: {config.WEAVE_PROJECT}")

    if args.eval_mapping is not None and args.row is not None and args.rows is not None:
        raise SystemExit("pass either --row or --rows, not both")
    row_mode = args.eval_mapping is not None and (args.row is not None or args.rows is not None)
    if row_mode:
        backend, model, prompt, rows = (
            _rows_mode_work(args) if args.rows is not None else _row_mode_work(args)
        )
    else:
        backend, model, prompt = args.backend, args.model, args.prompt

    ctx = await aprepare(backend, args.mcp_url, weave_enabled=weave_enabled,
                         model=model, prompt=prompt, gpu_type=args.gpu_type)
    print(f"Backend: {backend} ({ctx['framework']}) | "
          f"Model: {ctx['model']} (#{ctx['agent_model_key']}) | "
          f"Prompt: {ctx['prompt_name']} (#{ctx['prompt_key']}) | GPU: {ctx['gpu_type']} | "
          f"tools_hash={ctx['tools_hash']} | git={ctx['git_commit']}")

    if args.row is not None:
        print(f"Row mode: eval_mapping={args.eval_mapping} row={args.row} -> "
              f"task {rows[0]['task_id']} run {rows[0]['run_id']} benchmark {rows[0]['benchmark_id']}")
    elif row_mode:
        print(f"Shard mode: eval_mapping={args.eval_mapping} rows={args.rows} -> "
              f"{len(rows)} outputs, concurrency {args.concurrency}")
    else:
        benchmark_ids = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        rows = await discover_rows(args.mcp_url, benchmark_ids, args.limit)
        print(f"Discovered {len(rows)} outputs across benchmarks {benchmark_ids}")

    results = await run_batch(
        rows, ctx,
        # Single-row mode is one run by definition; a --rows shard keeps the requested
        # concurrency (offered load = shards x concurrency, kept under the rate limit).
        concurrency=1 if args.row is not None else args.concurrency,
        max_steps=args.max_steps,
        verbose=args.verbose,
        write_sink=not args.no_sink,
        gpu_metrics=not args.no_gpu_metrics,
        write_mongo=not args.no_mongo_runs,
        skip_existing=not args.no_skip_existing,
    )

    n_ok = sum(1 for r in results if r.get("stopped_reason") == "answered")
    print(f"\nDone: {n_ok}/{len(results)} answered. "
          f"{'Parquet rows under ' + str(config.AGENT_RUNS_DIR) if not args.no_sink else 'sink disabled'}")


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: parse args and run the async eval driver."""
    args = parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
