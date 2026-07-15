#!/usr/bin/env python3
"""
Generate gold_metrics.csv from benchmarks.json.

Reads the ground_truth block for each specified benchmark and writes one row per
(benchmark, field) with the primary metric (first in the metrics list) as gold_metric.
The agent is scored against this primary metric in selection_accuracy_scorer.

Run once from the repo root:
    python -m agent_eval.registry.generate_gold_metrics

Re-run only if benchmarks.json changes or you add new benchmark IDs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

BENCHMARK_IDS = ["5", "6", "7", "10", "11"]
# agent_eval/registry/ -> agent_eval/ -> repo root
BENCHMARKS_FILE = Path(__file__).resolve().parents[2] / "inputs" / "benchmarks.json"
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "gold_metrics.csv"
FIELDNAMES = ["benchmark_id", "benchmark_name", "field_name", "field_type", "gold_metric", "all_metrics"]


def main() -> None:
    """Read benchmarks.json and write one gold_metrics.csv row per (benchmark, field)."""
    with open(BENCHMARKS_FILE) as f:
        benchmarks = json.load(f)

    rows = []
    for bid in BENCHMARK_IDS:
        bm = benchmarks.get(bid)
        if not bm:
            print(f"Warning: benchmark {bid} not found in {BENCHMARKS_FILE}")
            continue
        gt = bm.get("ground_truth", {})
        if not gt:
            print(f"Warning: benchmark {bid} ({bm.get('task_name')}) has no ground_truth block — skipping")
            continue
        for field_key, spec in gt.items():
            metrics = spec.get("metrics", [])
            rows.append({
                "benchmark_id": bid,
                "benchmark_name": bm.get("task_name", ""),
                "field_name": spec.get("output_field", field_key),
                "field_type": spec.get("type", ""),
                "gold_metric": metrics[0] if metrics else "",
                "all_metrics": ", ".join(metrics),
            })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {OUTPUT_CSV}\n")
    print(f"{'benchmark_id':<14} {'benchmark_name':<20} {'field_name':<28} {'gold_metric'}")
    print("-" * 80)
    for r in rows:
        print(f"{r['benchmark_id']:<14} {r['benchmark_name']:<20} {r['field_name']:<28} {r['gold_metric']}")


if __name__ == "__main__":
    main()
