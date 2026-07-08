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

from dotenv import load_dotenv

from . import config, mapping
from .prompts import PROMPT_NAME
from .runner import aprepare, discover_rows, run_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="eval", description="Agentic metric evaluation")
    p.add_argument("--backend", choices=list(config.BACKENDS), default="nim")
    p.add_argument("--model", default=None,
                   help="override the backend's default model (same endpoint, different model)")
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
    p.add_argument("--no-gpu-metrics", action="store_true", help="skip /v1/metrics scrape")
    # SLURM-array worker mode: evaluate exactly the one output named by a row of an
    # eval_mapping.csv (the backend/model come from that row's judge config).
    p.add_argument("--eval-mapping", default=None,
                   help="path to eval_mapping.csv; with --row, evaluate just that row")
    p.add_argument("--row", type=int, default=None,
                   help="0-based row index into --eval-mapping (e.g. $SLURM_ARRAY_TASK_ID)")
    return p.parse_args(argv)


def _row_mode_work(args):
    """Resolve (backend, model, work-rows) from one eval_mapping row. The row's
    judge config is authoritative so the array does exactly what the mapping says."""
    row = mapping.read_mapping_row(args.eval_mapping, args.row)
    backend = row.get("judge_backend") or args.backend
    model = row.get("judge_model") or args.model
    if row.get("judge_prompt") and row["judge_prompt"] != PROMPT_NAME:
        print(f"WARNING: mapping row judge_prompt={row['judge_prompt']!r} but the "
              f"installed prompt is {PROMPT_NAME!r}; running with the installed prompt.")
    work = [{
        "task_id": row["task_id"],
        "run_id": mapping.coerce_run_id(row.get("run_id")),
        "benchmark_id": row["benchmark_id"],
        "model_id": row.get("model_id"),
    }]
    return backend, model, work


async def _amain(args):
    load_dotenv()
    weave_enabled = not args.no_weave
    if weave_enabled:
        import weave

        weave.init(config.WEAVE_PROJECT)
        print(f"Weave project: {config.WEAVE_PROJECT}")

    row_mode = args.eval_mapping is not None and args.row is not None
    if row_mode:
        backend, model, rows = _row_mode_work(args)
    else:
        backend, model = args.backend, args.model

    ctx = await aprepare(backend, args.mcp_url, weave_enabled=weave_enabled,
                         model=model, gpu_type=args.gpu_type)
    print(f"Backend: {backend} ({ctx['framework']}) | "
          f"Model: {ctx['model']} (#{ctx['agent_model_key']}) | GPU: {ctx['gpu_type']} | "
          f"tools_hash={ctx['tools_hash']} | git={ctx['git_commit']}")

    if row_mode:
        print(f"Row mode: eval_mapping={args.eval_mapping} row={args.row} -> "
              f"task {rows[0]['task_id']} run {rows[0]['run_id']} benchmark {rows[0]['benchmark_id']}")
    else:
        benchmark_ids = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        rows = await discover_rows(args.mcp_url, benchmark_ids, args.limit)
        print(f"Discovered {len(rows)} outputs across benchmarks {benchmark_ids}")

    results = await run_batch(
        rows, ctx,
        concurrency=1 if row_mode else args.concurrency,
        max_steps=args.max_steps,
        verbose=args.verbose,
        write_sink=not args.no_sink,
        gpu_metrics=not args.no_gpu_metrics,
    )

    n_ok = sum(1 for r in results if r.get("stopped_reason") == "answered")
    print(f"\nDone: {n_ok}/{len(results)} answered. "
          f"{'Parquet rows under ' + str(config.AGENT_RUNS_DIR) if not args.no_sink else 'sink disabled'}")


def main(argv=None):
    args = parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
