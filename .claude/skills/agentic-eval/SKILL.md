---
name: agentic-eval
description: >-
  Work with this repo's agentic metric-evaluation harness (an MCP agent that judges LLM
  outputs against ground truth). Use when running an eval, reading the results dashboard,
  or extending the harness — adding a judge backend, a prompt variant, or a new MCP tool.
  Covers the agent_eval/ + analysis/ pipeline on branch mcp-metric-calc.
---

# Agentic metric-eval harness

## Orient
This repo has **two pipelines**: upstream inference (`scripts/`, writes `llm_outputs` to Mongo) and
this **agentic eval** (`agent_eval/` + `analysis/`, judges those outputs). Active branch:
`mcp-metric-calc`. An MCP FastMCP server (`agent_eval/server.py`) exposes data + scoring tools; an
async agent loop (`agent_eval/runtime/agent.py`) drives an LLM judge to fetch one output, classify
each field's data shape, route it to the right `evaluate_*` tool, and save a verdict.

Authoritative depth: **`agent_eval/README.md`** (module layout + "choices & why"), **`CLAUDE.md`**
(conventions + gotchas), **`NEXT_STEPS.md`** (live status). Read those before non-trivial edits.

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

Single eval by hand (needs a running server URL): `python -m agent_eval --backend playground
--mcp-url <url> --benchmarks 5 --limit 1 --verbose`. SLURM-array worker mode does one mapping row:
`--eval-mapping <csv> --row <i>` (backend/model/prompt come from that row's judge config).

Results land in: local Parquet `outputs/agent_runs/date=…/`, and Mongo `agentic_runs` (run metrics)
+ `agentic_evaluations` (per-field verdicts), joined by `eval_id`. Stores are version-keyed by
`git_commit`; skip-existing skips only non-errored runs at the current commit.

## Read results
```bash
python -m analysis.serve_dashboard      # live, reads Mongo each load -> http://127.0.0.1:8787
python -m analysis.build_dashboard --no-summaries --open   # static self-contained HTML
```
Group by **Version** to compare `git_commit`s. Key metrics: **double-fetch** (agent called
`get_task_output` >1×), **"Metric identified"** = `selection_accuracy` (declared routing correct),
**"Optimal route"** = `routing_path_correct` (whole clean tool-call path correct).

## Extend the harness
- **New judge model/endpoint** → add/edit `agent_eval/backends/<name>.json` (pure data:
  `framework`, `base_url` (may use `$ENV_VAR`), `api_key_env`, int-keyed `models` with
  `completion_kwargs` + prices). Select with `--backend <name> --model <int>`.
- **New judge prompt** → add `agent_eval/prompts/<name>.json` (`index`, `name`, `system[]`, `user`).
  Cross prompts into a run with `create_eval_mapping --prompts v1,v2` / `--judge-backends …`.
- **New MCP tool** → add a function in `agent_eval/tools.py` and register it with `@mcp.tool()` in
  `agent_eval/server.py` (return errors as data, don't raise). The agent auto-discovers tools via
  `list_tools()`, so no agent-loop change is needed.
- **New task/routing grading** → update `agent_eval/gold_metrics.csv` (regenerate via
  `registry/generate_gold_metrics.py`) and the task-specific defaults in `create_eval_mapping.py`.

## Gotchas
- MongoDB required for the data tools + run mirror (Atlas hosts/DB `usf-internship` hardcoded in
  `tools.py`; the 3 `evaluate_*` tools are pure-compute and need no Mongo).
- `agent_eval/tools.py` imports scoring from `scripts/evaluator.py` via `sys.path` — keep
  `agent_eval/` a sibling of `scripts/`.
- `eval_id` / `git_commit` are stamped **client-side** on `save_evaluation`; LLM-passed values are
  overwritten by design.
- One repo-local `.venv/` for both pipelines (built from `requirements.txt`); run from the repo root.
- Don't confuse `scripts/3_create_mapping.py` (inference) with `registry/create_eval_mapping.py`
  (eval), or `model_id` (model under test) with `judge_model` (the judge).
