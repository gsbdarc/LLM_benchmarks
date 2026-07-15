"""
create_eval_mapping.py — build the agentic-eval work registry.

The eval analogue of scripts/3_create_mapping.py. Where that maps INFERENCE jobs
(benchmark × model × image → an output), this maps EVALUATION jobs: each existing
llm_output crossed with a judge/agent config, keyed by an append-only integer
`eval_id`.

  (1) pull processed llm_outputs (labeled benchmarks) from Mongo
  (2) cross them with the judge config(s) — for now just playground/gpt-5-mini/composite_v1
  (3) append new unique combinations to inputs/eval_mapping.csv with a fresh eval_id
  (4) optionally write a stratified sample (inputs/eval_mapping_sample.csv) that a
      SLURM array indexes by $SLURM_ARRAY_TASK_ID

Only benchmarks with a gold field_type in agent_eval/gold_metrics.csv are useful (routing
accuracy is null otherwise), so the default set is {5,6,7,10,11}.

    python -m agent_eval.registry.create_eval_mapping --sample 100

Pure list/dedupe/sample logic lives in agent_eval/registry/mapping.py (unit-tested); this file is
the thin Mongo-facing wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# agent_eval/registry/ -> agent_eval/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:               # so `import agent_eval` resolves
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval import config  # noqa: E402
from agent_eval.registry import mapping  # noqa: E402

# Benchmarks with gold field_type labels in agent_eval/gold_metrics.csv.
DEFAULT_BENCHMARKS = ["5", "6", "7", "10", "11"]
INPUTS_DIR = REPO_ROOT / "inputs"
REGISTRY = INPUTS_DIR / "eval_mapping.csv"
SAMPLE = INPUTS_DIR / "eval_mapping_sample.csv"


def fetch_outputs(benchmark_ids, limit_per_benchmark):
    """Pull processed outputs per benchmark via the same query the MCP tool uses."""
    from agent_eval.tools import list_pending_outputs  # lazy: keeps Mongo out of imports

    outs = []
    for bid in benchmark_ids:
        res = list_pending_outputs(benchmark_id=bid, limit=limit_per_benchmark)
        got = res.get("outputs", [])
        outs.extend(got)
        print(f"  benchmark {bid}: {len(got)} processed outputs")
    return outs


def judge_configs():
    """The judge/agent config(s) that do the grading. One for now."""
    from agent_eval.prompts import PROMPT_NAME

    model, _, _ = config.resolve_model("playground")
    return [{"judge_backend": "playground", "judge_model": model, "judge_prompt": PROMPT_NAME}]


def main(argv=None):
    p = argparse.ArgumentParser(prog="create_eval_mapping")
    p.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS),
                   help="comma-separated benchmark ids (default: the labeled ones)")
    p.add_argument("--limit-per-benchmark", type=int, default=100000,
                   help="cap on outputs pulled per benchmark")
    p.add_argument("--sample", type=int, default=None,
                   help="also write a stratified sample of this many rows for the array")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (deterministic)")
    args = p.parse_args(argv)

    benchmark_ids = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    outputs = fetch_outputs(benchmark_ids, args.limit_per_benchmark)
    candidates = mapping.build_rows(outputs, judge_configs())

    existing = mapping.read_csv(REGISTRY)
    new_rows = mapping.dedupe_and_assign(existing, candidates)
    mapping.append_csv(REGISTRY, new_rows)
    print(f"Registry {REGISTRY}: +{len(new_rows)} new "
          f"(was {len(existing)}, now {len(existing) + len(new_rows)})")

    if args.sample is not None:
        full = mapping.read_csv(REGISTRY)
        picked = mapping.stratified_sample(full, args.sample, seed=args.seed)
        mapping.write_csv(SAMPLE, picked)
        print(f"Sample {SAMPLE}: {len(picked)} rows  ->  --array=0-{max(0, len(picked) - 1)}")


if __name__ == "__main__":
    main()
