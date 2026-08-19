"""
create_eval_mapping.py — build the agentic-eval work registry.

The eval analogue of scripts/3_create_mapping.py. Where that maps INFERENCE jobs
(benchmark × model × image → an output), this maps EVALUATION jobs: each existing
llm_output crossed with a judge/agent config, keyed by an append-only integer
`eval_id`.

  (1) pull processed llm_outputs (labeled benchmarks) from Mongo
  (2) cross them with the judge config(s) — each playground model × each --prompts variant
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
from typing import Any

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


def fetch_outputs(
    benchmark_ids: list[str], limit_per_benchmark: int
) -> list[dict[str, Any]]:
    """Pull processed outputs per benchmark via the same query the MCP tool uses."""
    from agent_eval.tools import list_pending_outputs  # lazy: keeps Mongo out of imports

    outs = []
    for bid in benchmark_ids:
        res = list_pending_outputs(benchmark_id=bid, limit=limit_per_benchmark)
        got = res.get("outputs", [])
        outs.extend(got)
        print(f"  benchmark {bid}: {len(got)} processed outputs")
    return outs


DEFAULT_JUDGE_BACKENDS = ["playground"]


def judge_configs(
    prompt_names: list[str], judge_backends: list[str] | None = None
) -> list[dict[str, Any]]:
    """One judge config per (backend model × prompt), across the given judge backends.

    Each output is graded by every (backend, model, prompt) judge. Passing >1 prompt
    builds the prompt A/B (e.g. v1 + v2); adding a backend (e.g. the local 'qwen')
    adds that judge alongside the hosted ones."""
    cfgs = []
    for backend in (judge_backends or DEFAULT_JUDGE_BACKENDS):
        if backend not in config.BACKENDS:
            raise ValueError(
                f"unknown judge backend {backend!r}; choose from {list(config.BACKENDS)}"
            )
        for key in sorted(config.BACKENDS[backend].get("models") or {}, key=int):
            model, _, _ = config.resolve_model(backend, int(key))
            for pname in prompt_names:
                cfgs.append({"judge_backend": backend, "judge_model": model, "judge_prompt": pname})
    return cfgs


def resolve_prompt_names(spec: str | None) -> list[str]:
    """Resolve a `--prompts` spec (comma-separated names/indices) to canonical names.
    None → every registered variant, ordered by index."""
    from agent_eval.prompts import prompt_names, resolve_prompt

    if not spec:
        return prompt_names()
    names = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            names.append(resolve_prompt(tok)[0])  # validates + canonicalizes name/index
    return names


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="create_eval_mapping")
    p.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS),
                   help="comma-separated benchmark ids (default: the labeled ones)")
    p.add_argument("--limit-per-benchmark", type=int, default=100000,
                   help="cap on outputs pulled per benchmark")
    p.add_argument("--prompts", default=None,
                   help="comma-separated judge prompt names/indices (default: all registered variants)")
    p.add_argument("--judge-backends", default=",".join(DEFAULT_JUDGE_BACKENDS),
                   help="comma-separated judge backends crossed with prompts (default: playground). "
                        "e.g. 'playground,qwen' to add the local judge, or 'qwen' for local only")
    p.add_argument("--sample", type=int, default=None,
                   help="write a paired sample of this many OUTPUTS (each crossed with all judges)")
    p.add_argument("--sample-like", default=None,
                   help="write a sample over the EXACT outputs in this CSV (same outputs, current "
                        "--prompts judges); use to reuse a prior sample's outputs for a new prompt")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (deterministic)")
    args = p.parse_args(argv)

    judge_backends = [b.strip() for b in args.judge_backends.split(",") if b.strip()]
    prompt_names = resolve_prompt_names(args.prompts)
    cfgs = judge_configs(prompt_names, judge_backends)
    print(f"Judge configs: {len(cfgs)} = backends {judge_backends} x prompts {prompt_names}")

    benchmark_ids = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    outputs = fetch_outputs(benchmark_ids, args.limit_per_benchmark)
    candidates = mapping.build_rows(outputs, cfgs)

    existing = mapping.read_csv(REGISTRY)
    new_rows = mapping.dedupe_and_assign(existing, candidates)
    mapping.append_csv(REGISTRY, new_rows)
    print(f"Registry {REGISTRY}: +{len(new_rows)} new "
          f"(was {len(existing)}, now {len(existing) + len(new_rows)})")

    if args.sample_like is not None:
        full = mapping.read_csv(REGISTRY)
        reference = mapping.read_csv(args.sample_like)  # read fully before we overwrite SAMPLE
        wanted_prompts = set(prompt_names)
        wanted_backends = set(judge_backends)
        picked = [r for r in mapping.select_by_outputs(full, reference)
                  if r.get("judge_prompt") in wanted_prompts
                  and r.get("judge_backend") in wanted_backends]
        mapping.write_csv(SAMPLE, picked)
        n_out = len({mapping._output_key(r) for r in picked})
        print(f"Sample {SAMPLE}: {n_out} outputs (from {args.sample_like}) x backends "
              f"{judge_backends} x prompts {prompt_names} = {len(picked)} rows "
              f" ->  --array=0-{max(0, len(picked) - 1)}")
    elif args.sample is not None:
        full = mapping.read_csv(REGISTRY)
        picked = mapping.sample_paired(full, args.sample, seed=args.seed)
        mapping.write_csv(SAMPLE, picked)
        n_out = len({mapping._output_key(r) for r in picked})
        n_judges = len(cfgs)
        print(f"Sample {SAMPLE}: {n_out} outputs x {n_judges} judges = {len(picked)} rows "
              f" ->  --array=0-{max(0, len(picked) - 1)}")


if __name__ == "__main__":
    main()
