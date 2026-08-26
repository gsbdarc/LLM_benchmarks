---
name: agentic-eval
description: >-
  Work with this repo's agentic harness — an MCP agent driven over LLM outputs. It runs two
  tasks: metric-eval (judging how outputs are scored) and date-fix (repairing derived
  tv_guide_date values). Use when running either task, reading the dashboard or date-fix demo,
  or extending the harness — adding a judge backend, a prompt variant, a new MCP tool, or a
  whole new task. Covers the agent_eval/ + analysis/ pipeline.
---

# Agentic metric-eval harness

## Re-enter the archived project

For a cold start, inspect the current branch and working tree, then read these in order:

1. `README.md` — project overview and the boundary between the two pipelines.
2. `PROJECT_STATUS.md` — dated archival snapshot: verified behavior, incomplete work, setup quirks,
   and the GitHub follow-up index.
3. `agent_eval/README.md` — stable harness layout and design rationale.
4. `NEXT_STEPS.md` — chronological experiments, recorded results, and old runbooks.

Branch `mcp-metric-calc` intentionally preserves WIP, notebooks, examples, and infrastructure
experiments. Do not remove or reorganize them merely because they are unfinished. Treat prices,
model availability, infrastructure status, "immediate next" labels, and GitHub issue state as dated
evidence; verify them before starting new work.

## Orient
This repo has **two pipelines**: upstream inference (`scripts/`, writes `llm_outputs` to Mongo) and
this **agentic harness** (`agent_eval/` + `analysis/`, reads those outputs). Inspect the current
branch and working tree before acting; do not assume a branch name. An MCP FastMCP server
(`agent_eval/server.py`) exposes the tools; an async agent
loop (`agent_eval/runtime/agent.py`) drives an LLM through one output per run.

**The harness runs TWO tasks — establish which one you're in before editing.** The prompt variant
picks it, and `runtime/agent.py:tools_for_prompt` shows the agent only that task's tools (mapping a
prompt-name **prefix** via `_TOOLS_BY_TASK`; an off-pattern name silently falls back to metric-eval).

| | **metric-eval** (`composite_v*`) | **date-fix** (`date_fix_v1`) |
|---|---|---|
| Does | routes each field to the right `evaluate_*` tool, saves a verdict | repairs `tv_guide_date` (derived from `newspaper_date` + `day_of_week`), saves a correction |
| Tools | `get_task_output`, `evaluate_raw_string`/`_extracted_string`/`_list`, `save_evaluation` | `get_guide_date_case`, `compute_guide_date`, `save_correction` |
| Scored by | `selection_accuracy`, `routing_path_correct` (key: `gold_metrics.csv`) | `date_fix_scorer` → `fix_outcome` / `fix_regression` / `fix_needs_review` |
| Work list | `registry/create_eval_mapping.py` | `registry/create_date_fix_mapping.py` |
| Answers in | `agentic_evaluations` | `agentic_corrections` |
| Results page | `analysis/build_dashboard.py` / `serve_dashboard.py` | `analysis/build_date_fix_demo.py` / `serve_demo.py` |

Both write one row per run to `agentic_runs` + `outputs/agent_runs/` Parquet.

Authoritative depth: **`PROJECT_STATUS.md`** (dated handoff), **`agent_eval/README.md`** (module
layout + "choices & why"), **`AGENTS.md`** / **`CLAUDE.md`** (repository rules), and
**`NEXT_STEPS.md`** (historical work log). Read those before non-trivial edits.

Setup every command assumes: `source .venv/bin/activate` (repo-local venv), run **from the repo root**, secrets in
`.env` (see `.env.example` — needs `MONGO_DB_USERNAME/PASSWORD` + `STANFORD_API_KEY`). Tests:
`cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest`.

## Run an eval
The CLI does **not** start the MCP server — `--mcp-url` points at a running one. The SLURM batch
starts/stops its own server per job (the easy path).

```bash
# build a work sample (outputs × judge configs -> inputs/eval_mapping_sample.csv)
python -m agent_eval.registry.create_eval_mapping --sample 100
# run the batch: one job starts a local MCP server + fans out MAXPAR workers over the sample
sbatch agent_eval/scripts/run_eval_batch.slurm
#   tune: sbatch --export=ALL,MAXPAR=12,WEAVE=0 agent_eval/scripts/run_eval_batch.slurm
```

The **date-fix** task uses the same batch job, pointed at its own work list:

```bash
python -m agent_eval.registry.create_date_fix_mapping        # -> inputs/date_fix_mapping.csv
sbatch --export=ALL,SAMPLE=inputs/date_fix_mapping.csv agent_eval/scripts/run_eval_batch.slurm
```

Its sample is deliberately mixed — **wrong** rows the agent should fix, **control** rows already
correct (any change is a regression), **range** rows where truth is a span and the agent should
abstain. Tune with `--wrong/--control/--range`.

Single run by hand (needs a running server URL): `python -m agent_eval --backend playground
--mcp-url <url> --benchmarks 5 --limit 1 --verbose`. SLURM-array worker mode does one mapping row:
`--eval-mapping <csv> --row <i>` (backend/model/prompt come from that row's judge config); `--rows
A-B` does an inclusive slice as one shard, which must share a single judge config.

Results land in: local Parquet `outputs/agent_runs/date=…/`, and Mongo `agentic_runs` (run metrics)
+ `agentic_evaluations` / `agentic_corrections` (the answers), joined by `eval_id`. Stores are
version-keyed by `git_commit`; skip-existing skips only non-errored runs at the current commit.
Dollar costs are stamped into each run using the prices in that commit. Updating backend prices
changes future runs only; do not silently recalculate archived run costs with newer rates.

## Read results
```bash
python -m analysis.serve_dashboard      # metric-eval, live from Mongo -> http://127.0.0.1:8787
python -m analysis.build_dashboard --no-summaries --open   # static self-contained HTML

python -m analysis.serve_demo           # date-fix demo, live -> http://127.0.0.1:8788
python -m analysis.build_date_fix_demo  # static -> images/date_fix_demo.html (gitignored)
```
Both servers bind localhost on a headless node — forward the port to view them. `serve_demo` prints
the exact `ssh -N -L …` line; VS Code's Ports panel does the same.

**Dashboard (metric-eval).** Group by **Version** to compare `git_commit`s. Key metrics:
**double-fetch** (agent called `get_task_output` >1×), **"Metric identified"** =
`selection_accuracy` (declared routing correct), **"Optimal route"** = `routing_path_correct`
(whole clean tool-call path). In the paths table **Misrouted** (measured wrong) and **Unscored**
(no gradeable verdict, usually no save) are counted separately — don't merge them, or "never
measured" renders as "no errors".

**Demo (date-fix).** Reader-facing, four outcomes: Fixed / Confirmed / Left for a person / Wrong.
The number that matters is **`confidently_wrong`** — wrong *and* not flagged for review. A wrong
answer the agent flagged still reached a human; an unflagged one is a silent error.

## Extend the harness
- **New judge model/endpoint** → add/edit `agent_eval/backends/<name>.json` (pure data:
  `framework`, `base_url` (may use `$ENV_VAR`), `api_key_env`, int-keyed `models` with
  `completion_kwargs` + dated price snapshots). Select with `--backend <name> --model <int>`.
- **New judge prompt** → add `agent_eval/prompts/<name>.json` (`index`, `name`, `system[]`, `user`).
  Cross prompts into a run with `create_eval_mapping --prompts v1,v2` / `--judge-backends …`.
- **New MCP tool** → add a function in `agent_eval/tools.py` and register it with `@mcp.tool()` in
  `agent_eval/server.py` (return errors as data, don't raise). The agent auto-discovers tools via
  `list_tools()`. **But** it is only *shown* the tools in its task's set — a new tool for an existing
  task must be added to that frozenset in `runtime/agent.py`, or the agent will never see it.
- **New routing grading (metric-eval)** → update `agent_eval/gold_metrics.csv` (regenerate via
  `registry/generate_gold_metrics.py`) and the defaults in `create_eval_mapping.py`.
- **A whole new task** → follow what date-fix did; it needed no backend, runner or sink change:
  1. tools in `tools.py` + `@mcp.tool()` wrappers in `server.py`;
  2. `prompts/<task>_v1.json`, named so its prefix identifies the task;
  3. a frozenset of its tools + one `_TOOLS_BY_TASK` entry in `runtime/agent.py`;
  4. a scorer in `reporting/scorers.py` returning `None` for other tasks' runs (that's how
     `date_fix_scorer` skips metric-eval rows), and its columns added to `sink.py` + `RUN_COLUMNS`;
  5. a work-list builder in `registry/`, carrying any grading truth as CSV columns that
     `__main__.py:_work_row` forwards to the scorer — **never into a prompt or tool result**.

## Refresh Stanford Gateway prices

Prices in `agent_eval/backends/playground.json` are a dated configuration snapshot, not a promise
that they remain current. Before a new hosted batch:

1. Open Stanford's official rate page: `https://uit.stanford.edu/service/ai-api-gateway/rates`, and
   confirm each configured model is still available.
2. Use Stanford's fixed per-million-token rate when one is published. When Stanford states a
   discount instead, follow its vendor-pricing link, apply the stated discount, and record the
   derivation in the model's `_price_note`.
3. Update each model's `input_price` / `output_price` and the top-level `pricing.verified_at` date.
   Keep the configured unit as USD per 1 million tokens.
4. If no supported price is published, leave both values `null`; the dashboard will show
   **Unpriced**. Never substitute a guess merely to populate a cost column. Local models remain 0/0.
5. Update the recorded-price assertions in `tests/test_dashboard_review_feedback.py`, run that
   focused test, then run the full offline suite with `EVAL_DISABLE_WEAVE=1`.

Historical run rows already contain the cost calculated when they were written. Use `git_commit` to
recover that run's price configuration; do not rewrite old rows after a rate change.

## Gotchas
- MongoDB required for the data tools + run mirror (Atlas hosts/DB `usf-internship` hardcoded in
  `tools.py`; the 3 `evaluate_*` tools are pure-compute and need no Mongo).
- `agent_eval/tools.py` imports scoring from `scripts/evaluator.py` via `sys.path` — keep
  `agent_eval/` a sibling of `scripts/`.
- `eval_id` / `git_commit` are stamped **client-side** on `save_evaluation` / `save_correction`;
  LLM-passed values are overwritten by design.
- The date-fix agent **never sees ground truth**: `expected_date` / `original_value` ride the mapping
  row to `date_fix_scorer` only. Don't pass them to the agent — grading a regression off a value the
  agent reported itself would let it grade its own baseline.
- One repo-local `.venv/` for both pipelines (built from `requirements.txt`); run from the repo root.
- Don't confuse `scripts/3_create_mapping.py` (inference) with `registry/create_eval_mapping.py`
  (eval), or `model_id` (model under test) with `judge_model` (the judge).
