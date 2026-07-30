# CLAUDE.md — repo guide for AI coding agents

Project-specific conventions and gotchas. Merge with the user's global guidelines.

## What this repo is (TWO pipelines — don't conflate them)
- **Upstream inference — `scripts/`** (`1_…`–`7_…`, `yens.slurm`/`sherlock.slurm`): runs LLMs over
  images and **writes** `llm_outputs` to Mongo. Config in `inputs/` (`models.json`, `benchmarks.json`).
- **Agentic metric-eval — `agent_eval/` + `analysis/`**: an MCP agent that **judges** those outputs.
  This is the **active work**, on branch **`mcp-metric-calc`** (not `main`). Config-as-data in
  `agent_eval/backends/*.json`, `agent_eval/prompts/*.json`, `agent_eval/gold_metrics.csv`.
- Name collisions: `scripts/3_create_mapping.py` (inference tasks) ≠ `registry/create_eval_mapping.py`
  (eval jobs); `model_id` = model **under test** ≠ `judge_model` / backend int-key = the **judge**.

Read `agent_eval/README.md` (design + module layout) and `NEXT_STEPS.md` (live status) before editing.

## Conventions
- **Venv:** one repo-local `.venv/` for both pipelines (`source .venv/bin/activate`), built from
  `requirements.txt`; gitignored. (Sherlock is a separate filesystem, so build its own `.venv` there.)
- **Tests:** `cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest` (offline; ~145 tests). CI runs
  the same on push. `EVAL_DISABLE_WEAVE=1` makes Weave a no-op — always set it in tests/local runs.
- **Run from the repo root:** `python -m agent_eval …`, `python -m analysis.…`, and `sbatch` the
  SLURM scripts from the root (they check `SLURM_SUBMIT_DIR`).
- **Secrets:** live in a gitignored `.env` (template: `.env.example`). Never commit `.env`, and never
  print secret **values** — variable names only.
- **Commit** only when asked; the active branch is `mcp-metric-calc`.

## Reuse map (what's generic vs task-specific)
- **Generic / data-driven** (reuse for a new task with no/low code): the agent loop
  (`runtime/agent.py`), batch runner (`runtime/runner.py`), reporting (`reporting/*`), and the
  registries — add an endpoint via `backends/<name>.json`, a judge prompt via `prompts/<name>.json`
  (both pure data changes; `config.resolve_model` / `prompts.resolve_prompt` load them).
- **Task-specific** (edit for a new task): the MCP tools in `server.py` + `tools.py`, the routing key
  `gold_metrics.csv`, and the defaults in `registry/create_eval_mapping.py`.

## Gotchas
- **MongoDB is required** for the data tools (`list_outputs`/`get_task_output`/`save_evaluation`) and
  the run mirror. Atlas hosts, replica set, and DB (`usf-internship`) are **hardcoded** in `tools.py`;
  creds come from `.env` (`MONGO_DB_USERNAME`/`PASSWORD`). The 3 `evaluate_*` composite tools are
  pure-compute and work without Mongo.
- `agent_eval/tools.py` imports scoring from `scripts/evaluator.py` via a `sys.path` insert (`BASE_DIR
  = parents[1]`). **Don't move `agent_eval/` away from being a sibling of `scripts/`** — it breaks.
- The eval CLI **does not start the MCP server** — `--mcp-url` is required and must point at an
  already-running server (`run_eval_batch.slurm` starts/stops one per job; transport is
  streamable-HTTP at path `/mcp`).
- `eval_id` and `git_commit` are **stamped client-side** in `runtime/agent.py` on `save_evaluation`;
  any values the LLM passes are overwritten by design.
- Mongo stores are **version-keyed by `git_commit`** (skip-existing skips only non-errored runs at the
  current commit). Results: local Parquet under `outputs/agent_runs/` + Mongo `agentic_runs`
  (run rows) / `agentic_evaluations` (per-field verdicts), joined by `eval_id`.
- Metric names in the dashboard: **"Metric identified"** = `selection_accuracy` (declared routing),
  **"Optimal route"** = `routing_path_correct` (clean tool-call path).
