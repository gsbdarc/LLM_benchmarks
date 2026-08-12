# CLAUDE.md — repo guide for AI coding agents

Project-specific conventions and gotchas. Merge with the user's global guidelines.

## What this repo is (TWO pipelines — don't conflate them)
- **Upstream inference — `scripts/`** (`1_…`–`7_…`, `yens.slurm`/`sherlock.slurm`): runs LLMs over
  images and **writes** `llm_outputs` to Mongo. Config in `inputs/` (`models.json`, `benchmarks.json`).
- **Agentic harness — `agent_eval/` + `analysis/`**: an MCP agent that **reads** those outputs and
  does something with them. This is the **active work**, on branch **`mcp-metric-calc`** (not `main`).
  Config-as-data in `agent_eval/backends/*.json`, `agent_eval/prompts/*.json`,
  `agent_eval/gold_metrics.csv`.
- Name collisions: `scripts/3_create_mapping.py` (inference tasks) ≠ `registry/create_eval_mapping.py`
  (eval jobs); `model_id` = model **under test** ≠ `judge_model` / backend int-key = the **judge**.

Read `agent_eval/README.md` (design + module layout) and `NEXT_STEPS.md` (live status) before editing.

## TWO TASKS run on that one harness — check which one you're in
The prompt variant selects the task, and `runtime/agent.py:tools_for_prompt` gives the agent only
that task's tools. Nothing else differs: same agent loop, runner, backends, reporting, sink.

| | **metric-eval** | **date-fix** |
|---|---|---|
| Prompt | `composite_v1` / `composite_v2` | `date_fix_v1` |
| Tools shown | `get_task_output`, `evaluate_raw_string` / `_extracted_string` / `_list`, `save_evaluation` | `get_guide_date_case`, `compute_guide_date`, `save_correction` |
| Does what | routes each field to the right scoring tool and saves a verdict | repairs `tv_guide_date` (derivable from `newspaper_date` + `day_of_week`) and saves a correction |
| Graded by | `selection_accuracy`, `routing_path_correct` | `date_fix_scorer` → `fix_outcome`, `fix_regression`, `fix_needs_review` |
| Truth from | `gold_metrics.csv` (gold `field_type`) | `expected_date` / `original_value` columns on the work-list row |
| Work list | `registry/create_eval_mapping.py` | `registry/create_date_fix_mapping.py` |
| Saves to | `agentic_evaluations` | `agentic_corrections` |
| Read results | `analysis/build_dashboard.py`, `serve_dashboard.py` (:8787) | `analysis/build_date_fix_demo.py`, `serve_demo.py` (:8788) |

Both share `agentic_runs` (one row per run) and `outputs/agent_runs/` Parquet.

## Conventions
- **Venv:** one repo-local `.venv/` for both pipelines (`source .venv/bin/activate`), built from
  `requirements.txt`; gitignored. (Sherlock is a separate filesystem, so build its own `.venv` there.)
- **Tests:** `cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest` (offline; ~183 tests). CI runs
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
- **Task-specific** (write for a new task): the MCP tools in `server.py` + `tools.py`, a scorer in
  `reporting/scorers.py`, a work-list builder in `registry/`, and its grading key (`gold_metrics.csv`
  for metric-eval; per-row columns for date-fix).
- The date-fix task is the **worked example** that this split is real: adding a whole second task
  touched no backend, no runner, no sink schema — one prompt JSON, three tools, one scorer, one
  mapping builder, and one line in `_TOOLS_BY_TASK`. Copy that shape.

## Gotchas
- **MongoDB is required** for the data tools (`list_outputs`/`get_task_output`/`save_evaluation`,
  `get_guide_date_case`/`save_correction`) and the run mirror. Atlas hosts, replica set, and DB
  (`usf-internship`) are **hardcoded** in `tools.py`; creds come from `.env`
  (`MONGO_DB_USERNAME`/`PASSWORD`). The 3 `evaluate_*` composites and `compute_guide_date` are
  pure-compute and work without Mongo — which is why most of the test suite runs offline.
- `agent_eval/tools.py` imports scoring from `scripts/evaluator.py` via a `sys.path` insert (`BASE_DIR
  = parents[1]`). **Don't move `agent_eval/` away from being a sibling of `scripts/`** — it breaks.
- The eval CLI **does not start the MCP server** — `--mcp-url` is required and must point at an
  already-running server (`run_eval_batch.slurm` starts/stops one per job; transport is
  streamable-HTTP at path `/mcp`).
- `eval_id` and `git_commit` are **stamped client-side** in `runtime/agent.py` on `save_evaluation`
  and `save_correction`; any values the LLM passes are overwritten by design.
- Mongo stores are **version-keyed by `git_commit`** (skip-existing skips only non-errored runs at the
  current commit). Results: local Parquet under `outputs/agent_runs/` + Mongo `agentic_runs`
  (run rows, both tasks) / `agentic_evaluations` (metric-eval per-field verdicts) /
  `agentic_corrections` (date-fix decisions), all joined to a run by `eval_id`.
- **Tool subsetting is by prompt-NAME PREFIX** (`_TOOLS_BY_TASK` in `runtime/agent.py`). A prompt
  named off-pattern silently falls back to the metric-eval tool set, and the agent then can't see the
  tools its prompt tells it to call. Name a new date-fix variant `date_fix_*`, or add a prefix entry.
- **The date-fix agent is never shown ground truth.** `expected_date` / `original_value` ride the
  mapping row (`__main__.py:_work_row`) straight to `date_fix_scorer` and are never put in a prompt or
  a tool result. Don't "helpfully" pass them to the agent — grading a regression against a value the
  agent reported itself would let it grade its own baseline.
- Metric names in the dashboard: **"Metric identified"** = `selection_accuracy` (declared routing),
  **"Optimal route"** = `routing_path_correct` (clean tool-call path). In the paths table,
  **Misrouted** (measured wrong) and **Unscored** (never gradeable, usually no save) are counted
  separately on purpose — collapsing them displays "never measured" as "no errors".
