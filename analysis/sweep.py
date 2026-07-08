"""
sweep.py — sequential-vs-parallel concurrency experiment (plan §4).

Runs the SAME set of outputs at several concurrency levels, writing one Parquet
row per run (tagged with `concurrency` by the sink) plus an optional time series
of the vLLM /v1/metrics endpoint sampled throughout. The dashboard then compares
wall-time, throughput, GPU cache usage, and num_requests_running across levels —
answering "does GPU utilization change individual vs parallel" and locating the
real concurrency ceiling on one H200.

    python -m analysis.sweep --backend nim --mcp-url http://127.0.0.1:PORT/mcp \\
        --benchmarks 5,6,7 --limit 5 --concurrency 1,2,4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the eval package importable (analysis/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_eval import config  # noqa: E402
from agent_eval.observability import scrape_vllm_metrics  # noqa: E402
from agent_eval.runner import aprepare, discover_rows, run_batch  # noqa: E402

METRICS_TS_DIR = REPO_ROOT / "outputs" / "vllm_metrics_timeseries"


async def _poller(metrics_url, stop_event, samples, interval=2.0):
    """Sample /v1/metrics until stop_event is set, appending dicts to `samples`."""
    while not stop_event.is_set():
        snap = scrape_vllm_metrics(metrics_url)
        snap["ts"] = datetime.now(timezone.utc).isoformat()
        samples.append(snap)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="analysis.sweep", description="Concurrency sweep")
    p.add_argument("--backend", choices=list(config.BACKENDS), default="nim")
    p.add_argument("--mcp-url", required=True)
    p.add_argument("--benchmarks", default="5,6,7")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--concurrency", default="1,2,4", help="comma-separated levels")
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--no-weave", action="store_true")
    p.add_argument("--poll-interval", type=float, default=2.0)
    return p.parse_args(argv)


async def _amain(args):
    load_dotenv()
    weave_enabled = not args.no_weave
    if weave_enabled:
        import weave

        weave.init(config.WEAVE_PROJECT)

    ctx = await aprepare(args.backend, args.mcp_url, weave_enabled=weave_enabled)
    benchmark_ids = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    rows = await discover_rows(args.mcp_url, benchmark_ids, args.limit)
    print(f"Sweep over concurrency {levels} on {len(rows)} outputs each")

    metrics_url = config.metrics_url(ctx["base_url"])
    METRICS_TS_DIR.mkdir(parents=True, exist_ok=True)

    for level in levels:
        print(f"\n=== concurrency = {level} ===")
        stop = asyncio.Event()
        samples: list = []
        poll_task = asyncio.create_task(_poller(metrics_url, stop, samples, args.poll_interval))

        t0 = time.perf_counter()
        await run_batch(
            rows, ctx, concurrency=level, max_steps=args.max_steps,
            verbose=False, write_sink=True, gpu_metrics=True,
        )
        wall = time.perf_counter() - t0

        stop.set()
        await poll_task
        if samples:
            df = pd.DataFrame(samples)
            df["concurrency"] = level
            ts_path = METRICS_TS_DIR / f"sweep-c{level}-{int(t0)}.parquet"
            df.to_parquet(ts_path, index=False)
            print(f"  wall={wall:.1f}s  metrics samples={len(samples)} -> {ts_path.name}")
        else:
            print(f"  wall={wall:.1f}s  (no metrics samples)")

    print("\nSweep complete. Compare with: python -m analysis.dashboard")


def main(argv=None):
    asyncio.run(_amain(parse_args(argv)))


if __name__ == "__main__":
    main()
