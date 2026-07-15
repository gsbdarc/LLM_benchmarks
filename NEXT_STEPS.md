# Next steps — team review (Tue 2026-07-21), then composite_v2

First real agentic-eval batch is **done** and in the dashboard. Immediate next step is a
**team review of the results + dashboard on Tuesday morning (2026-07-21) to collect feedback
before deciding what to build next.**

## Where we landed
- **Batch 376856 complete:** 120 evals = **40 outputs × 3 judges** (gpt-5-mini, DeepSeek-V3.2,
  claude-sonnet-4-6), all via the playground API, weave on. 119 converged, 1 hit `max_steps`.
- **Judges:** Llama-4 was dropped — the Stanford key has **no Llama access** (401); swapped in
  **DeepSeek-V3.2** (open-weight, validated end-to-end).
- **Data landed in all stores:** local Parquet, **`agentic_runs`** (central Mongo mirror, the
  team-queryable copy — 120/120), and **`agentic_evaluations`** (per-judge verdicts, keyed by
  the shared **`eval_id`** so multiple judges of the same output no longer collapse).
- **Dashboard:** `images/agent_dashboard_376856.html` (self-contained; built from the clean
  `agentic_runs` rows). Group-by model/benchmark, latency split (`llm_time` vs `overhead_time`),
  full-size tool-path/routing DAG.

## Headline results (for the review)
- **Routing accuracy ~0.44–0.54 overall**, but wildly uneven by benchmark:
  bench 11 = 1.00, bench 7 = 0.985, bench 10 = 0.33, bench 6 = 0.17, **bench 5 = 0.083**.
- The **raw_string ↔ extracted_string confusion is concentrated in single-/ambiguous-field
  benchmarks (5, 6, 10)** — e.g. bench 5's `first_channel` is gold `raw_string` but every judge
  routes it `extracted_string`. Bench 7 (3 clearly-different-shaped fields) routes near-perfectly.
- **Save reliability high** (DeepSeek 100%, gpt-5-mini & claude 97.5%); the misses were 1
  `max_steps` and 1 answered-but-didn't-save (caught by `save_success`). No true tool-hedging
  (the few >1-tool-per-field cases were all sanctioned retry-after-error).
- **Latency** (service, not pure inference — playground is remote): DeepSeek fastest (~20s),
  gpt-5-mini ~24s, claude ~26s. `overhead_time` ~0.9s and backend-independent (local tools).

## Queued (after team feedback picks the direction)
- **Benchmark 5/6/10 routing deep-dive** → the concrete input for **composite_v2**: sharpen the
  raw-vs-extracted distinction (raw = always-present printed line; extracted = a single value
  that may be absent), *and* review whether some gold `field_type` labels are too strict.
- **deepseek-r1 as a 4th judge** — the "does a reasoning model route better?" experiment.
- **Local NIM run** to populate the GPU-pressure / concurrency panel (needs the GPU).
- **Dashboard path-summaries** (needs the gemma summarizer server): rebuild with
  `python -m analysis.build_dashboard --base-dir outputs/dash_batch376856 --refresh-summaries --out images/agent_dashboard_376856.html`.

## Rebuild / re-run recipes (from the repo root)

```bash
source ~/venv/bin/activate

# regenerate the dashboard from the clean batch rows (Mongo export lives in outputs/dash_batch376856)
python -m analysis.build_dashboard --base-dir outputs/dash_batch376856 --no-summaries \
    --out images/agent_dashboard_376856.html

# a fresh batch: rebuild the sample, then fire it (SLURM picks a compute node, starts its own server)
python -m agent_eval.registry.create_eval_mapping --sample 40 --seed 0
sbatch agent_eval/scripts/run_eval_batch.slurm       # knobs: --export=ALL,MAXPAR=8,WEAVE=1
```

Tests: `cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest` (123 green).
Design log + layout: `agent_eval/README.md`.
