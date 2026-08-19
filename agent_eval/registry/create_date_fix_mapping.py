"""
create_date_fix_mapping.py — build the work list for the date-fix task.

Sibling of create_eval_mapping.py, for the OTHER task: instead of crossing outputs
with judge configs to grade metric routing, this selects benchmark-3 (tv_guide_date)
outputs to repair and records the truth needed to grade the repair.

Two extra columns beyond the standard mapping fields:
  * expected_date  — the true guide date from ground_truths
  * original_value — what the model originally output
Both are read only by `date_fix_scorer`; the agent never sees them. Judging a
regression from a value the agent reported itself would let it grade its own baseline.

The sample is deliberately mixed so all three behaviours show up in one batch:
  * wrong   — the derived date disagrees with the truth, and the prerequisites
              (newspaper_date, day_of_week) were both extracted correctly. These are
              the 476 the demo is about: the agent should FIX them.
  * control — already correct. The agent should CONFIRM them, and any change here is
              a regression. This is what proves the agent doesn't break good rows.
  * range   — ground truth is a span ("April 4-8 2005"), so no single date is right.
              The agent should ABSTAIN.

Rows are written GROUPED BY JUDGE so `--rows A-B` shards never span judge configs.

    python -m agent_eval.registry.create_date_fix_mapping --wrong 30 --control 30 --range 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# agent_eval/registry/ -> agent_eval/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.registry import mapping  # noqa: E402

DATE_FIX_BENCHMARK = "3"
FIELDS = mapping.FIELDS + ["expected_date", "original_value"]
OUT = REPO_ROOT / "inputs" / "date_fix_mapping.csv"


def build_candidates() -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Bucket benchmark-3 outputs into wrong / control / range.

    EVERY bucket requires the model to have read both inputs correctly, so each row
    is internally consistent and the agent is judged on the derivation alone.

    That check used to apply only to "wrong", which let a contradictory row into the
    controls: one model landed on the right final date (May 26 2002) while reporting
    the publication year as 1902. The agent recomputed from the bad input and
    "corrected" a right answer — counted as its failure when the row was the problem.
    Rows with unsound inputs are excluded and counted, not silently dropped.

    Also returns population counts for the cost story: how many answers exist, and how
    many a FREE deterministic check would flag for review (the model's own three answers
    disagreeing) — that is how candidates would be picked in production, where there is no
    answer key to select on.
    """
    from agent_eval.tools import get_db, parse_date, compute_guide_date, DATE_FIX_BENCHMARKS

    db = get_db()
    truth = {g["_id"]: g for g in db["ground_truths"].find()}

    # The model's own answers for all three related benchmarks, keyed by identity.
    answers: dict[tuple, dict[str, Any]] = {}
    for d in db["llm_outputs"].find(
        {"benchmark_id": {"$in": list(DATE_FIX_BENCHMARKS)}, "status": "processed"},
        {"task_id": 1, "run_id": 1, "benchmark_id": 1, "model_id": 1,
         "image_id": 1, "output": 1},
    ):
        key = (d.get("model_id"), d.get("image_id"), d.get("run_id"))
        answers.setdefault(key, {})[DATE_FIX_BENCHMARKS[d["benchmark_id"]]] = d

    buckets: dict[str, list[dict[str, Any]]] = {"wrong": [], "control": [], "range": []}
    excluded = 0
    scope = {"answers": 0, "flagged_by_free_check": 0}
    for (model_id, image_id, run_id), got in answers.items():
        anchor = got.get("tv_guide_date")
        gt = truth.get(image_id)
        if anchor is None or gt is None:
            continue
        scope["answers"] += 1

        # The production candidate filter: does the model's own derived date follow from
        # its own two inputs? No ground truth needed, no model call, so it costs nothing.
        derived = compute_guide_date((got.get("newspaper_date") or {}).get("output"),
                                     (got.get("day_of_week") or {}).get("output"))
        stated = parse_date(anchor.get("output"))
        if ("ambiguous" in derived or stated is None
                or derived.get("date") != stated.strftime("%Y-%m-%d")):
            scope["flagged_by_free_check"] += 1

        # Both inputs must be sound, whatever the final answer turned out to be.
        np_ok = parse_date((got.get("newspaper_date") or {}).get("output")) == parse_date(
            gt.get("newspaper_date"))
        dow_ok = (str((got.get("day_of_week") or {}).get("output") or "").strip().lower()
                  == str(gt.get("day_of_week") or "").strip().lower())
        if not (np_ok and dow_ok):
            excluded += 1
            continue

        expected_raw = gt.get("tv_guide_date")
        expected = parse_date(expected_raw)
        row = {
            "task_id": anchor["task_id"],
            "run_id": run_id,
            "benchmark_id": DATE_FIX_BENCHMARK,
            "model_id": model_id,
            "expected_date": expected_raw,
            "original_value": anchor.get("output"),
        }
        if expected is None:                      # truth is a span of days
            buckets["range"].append(row)
        elif parse_date(anchor.get("output")) == expected:
            buckets["control"].append(row)
        else:
            buckets["wrong"].append(row)
    print(f"excluded {excluded} rows whose inputs the model misread "
          f"(judging the derivation needs sound inputs)")
    return buckets, scope


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agent_eval.registry.create_date_fix_mapping")
    p.add_argument("--wrong", type=int, default=30, help="rows the agent should FIX")
    p.add_argument("--control", type=int, default=30,
                   help="already-correct rows; any change is a regression")
    p.add_argument("--range", type=int, default=6, dest="n_range",
                   help="range-truth rows the agent should ABSTAIN on")
    p.add_argument("--judge-backends", default="playground")
    p.add_argument("--judge-models", default="gemini-2.5-pro",
                   help="comma-separated; the reasoning-capable judge by default")
    p.add_argument("--prompt", default="date_fix_v1")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    buckets, scope = build_candidates()
    print("candidates: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print(f"population: {scope['answers']} answers, {scope['flagged_by_free_check']} flagged by the "
          f"free consistency check (that is the set worth sending to an agent)")

    import random
    rng = random.Random(args.seed)
    picked: list[dict[str, Any]] = []
    for name, n in (("wrong", args.wrong), ("control", args.control), ("range", args.n_range)):
        pool = sorted(buckets[name], key=lambda r: (str(r["task_id"]), str(r["run_id"])))
        take = pool if n >= len(pool) else rng.sample(pool, n)
        if n > len(pool):
            print(f"  NOTE: asked for {n} {name} rows, only {len(pool)} exist — taking all")
        for r in take:
            picked.append({**r, "bucket": name})

    # Group by judge so --rows shards never span configs.
    rows, eval_id = [], 1
    for backend in [b.strip() for b in args.judge_backends.split(",") if b.strip()]:
        for model in [m.strip() for m in args.judge_models.split(",") if m.strip()]:
            for r in picked:
                rows.append({
                    "eval_id": eval_id,
                    "task_id": r["task_id"],
                    "run_id": r["run_id"],
                    "benchmark_id": r["benchmark_id"],
                    "model_id": r["model_id"],
                    "judge_backend": backend,
                    "judge_model": model,
                    "judge_prompt": args.prompt,
                    "expected_date": r["expected_date"],
                    "original_value": r["original_value"],
                })
                eval_id += 1

    mapping.write_csv(args.out, rows, fields=FIELDS)
    by_bucket: dict[str, int] = {}
    for r in picked:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.out}")
    print("  per judge: " + ", ".join(f"{k}={v}" for k, v in by_bucket.items()))


if __name__ == "__main__":
    main()
