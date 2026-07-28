# Next steps — v2 measurement, then local Qwen judge

Post-review roadmap Phase 1 (+ parts of 2/3) is **built, committed, and v1-baselined**. The
immediate next step is the **composite_v2 measurement** (the real test of the prompt rework).

Branch `mcp-metric-calc`; 3 new commits this session — `082117c` (routing + cost + versioned
stores), `a8d6f22` (living dashboard), `06fe561` (skip-existing ignores errored + error_detail).
**136 tests green.**

## Shipped
- **composite_v2** prompt (goal-based; drops the numbered procedure, the "Call get_task_output"
  imperative, and "retry once") — APPROVED. Selectable via `--prompt` / the mapping `judge_prompt`.
- **args-parse fix** (malformed tool-call JSON → error back to the model, no phantom `get_task_output({})`).
- **bench-10 `[]`→`""` normalization** (type-aware) in `fetch_evaluable_output`.
- **`routing_path_correct`** scorer (whole clean path) beside `selection_accuracy`; wired to
  sink / `agentic_runs` / dashboard.
- **#1 cost:** per-model prices in `backends/*.json` + `config.model_price` + `$` in `flatten_run`
  + dashboard Cost column. DeepSeek-V3.2 unpriced (no published price → null); **claude-sonnet-4-6
  assumed 3/15 — CONFIRM**.
- **#2 living dashboard:** `analysis/export_runs.py` + `agent_eval/scripts/refresh_dashboard.slurm`
  (self-rescheduling; `sbatch` once to go live). Generated `images/agent_dashboard_live.html` gitignored.
- **Versioned Mongo stores:** `code_version` (git_commit) in BOTH collection keys (compound unique
  index eval_id+git_commit), `get_git_commit` `-dirty` flag, `run_exists` skip-if-exists (skips only
  NON-errored), `error_detail` persisted on API failures.

## v1 baseline findings (job 394413 @ `a8d6f22`; 376856 @ `5b5616a` preserved, coexisting)
Re-ran v1 (40 outputs × 3 hosted judges) under the new code vs 376856:
- **`selection_accuracy` 0.483 → 0.651 (+0.17)** — the bench-10 `[]` normalization, *same prompt*. A win.
- **double `get_task_output` ~79%, unchanged old→new** — it is **prompt-driven, not the args bug**.
  This is the baseline v2 must beat.
- **`routing_path_correct` validated** (4/120 clean; most fail on double-fetch + routing errors).
- A transient playground outage errored gpt-5-mini 40/40 on the first attempt (job 394400); confirmed
  it wasn't our code (burst test 8/8 clean), deleted the errored docs, re-ran clean. `error_detail`
  now makes such failures visible.

## IMMEDIATE NEXT: the v2 measurement (Phase 1 finale)
Does composite_v2 cut the ~79% double-fetch and hold/improve routing?
1. Teach `create_eval_mapping` to emit **composite_v2** judge configs (today it uses only the default
   prompt). Add a `--prompts` option / enumerate both; the `judge_prompt` column drives the prompt.
2. Build the v2 sample (40 outputs × 3 judges = 120, same 40 outputs).
3. Commit that change first (clean `code_version`), then fire:
   `sbatch agent_eval/scripts/run_eval_batch.slurm` (weave on, MAXPAR 8, skip-existing on).
4. Compare v2 vs v1 per judge: double-fetch rate, `routing_path_correct`, `selection_accuracy`.
   Success = v2 improves both AND cuts double-fetch.
   Note: v1 baseline is `a8d6f22`; the `06fe561` changes since are behaviorally inert to agent/scoring,
   so v2@<new> vs v1@`a8d6f22` is a valid prompt A/B. For perfect rigor, re-run v1 at the new commit too
   (skip-existing won't skip a new version).

## Then
- **#4 Local Qwen judge** — Qwen3.6-35B-A3B via vLLM (TP=2, 2× A40, `Vllm_testing/multi_gpu_serve/`).
  Serve with `--enable-auto-tool-choice --tool-call-parser <hermes|qwen3>` (else no tool calls). Add
  `backends/qwen.json` (framework `vllm`, `base_url` via `$LOCAL_MODEL_URL` env, prices 0/0). Fix
  `config.metrics_url` → `/metrics` for vLLM. Pre-flight one eval, then add as a 4th judge under v2,
  GPU metrics ON.
- Confirm the **claude-sonnet-4-6 price** (assumed 3/15).
- `sbatch agent_eval/scripts/refresh_dashboard.slurm` once to make the dashboard "living".

## Recipes (from repo root)
```bash
source ~/venv/bin/activate
cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest ; cd ..          # 136 tests
# rebuild dashboard from the central store:
python -m analysis.export_runs --out outputs/dashboard_cache
python -m analysis.build_dashboard --base-dir outputs/dashboard_cache --no-summaries \
    --out images/agent_dashboard_live.html
# compare versions in agentic_runs by git_commit: a8d6f22 = v1-new, 5b5616a = 376856
```
Design log + layout: `agent_eval/README.md`.
