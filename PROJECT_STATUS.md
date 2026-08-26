# Project status and handoff

**As of:** 2026-08-26

**Handoff branch:** `mcp-metric-calc`

This repository contains two related pipelines for evaluating multimodal LLM extraction from
historical TV-guide images. The original inference pipeline runs models over images. The newer
agentic evaluation harness reads those outputs and measures or repairs them through MCP tools.

This document began as an AI-assisted repository review and was corrected against the code, tests,
configuration, Git history, and recorded experiments. It separates locally verified behavior from
prior experiment results and unfinished infrastructure work. For the detailed agentic design
rationale, read [`agent_eval/README.md`](agent_eval/README.md); for the chronological work log and
recorded run results, read [`NEXT_STEPS.md`](NEXT_STEPS.md).

For a cold start, read `README.md` first, this handoff second, `agent_eval/README.md` for stable
design decisions, and `NEXT_STEPS.md` for historical run details. The `agentic-eval` skill under
`.agents/skills/` contains the operational commands and extension procedures used by coding agents.
Anything time-sensitive here—including prices, model availability, infrastructure state, and issue
status—is a dated snapshot that must be rechecked before new work.

## Current state at a glance

| Area | State | Evidence or limit |
|---|---|---|
| Python environment | Working locally | Python 3.10; all 26 direct dependencies are covered by the reproducible lock; `pip check` passes. |
| Upstream inference (`scripts/`) | Implemented; locally configured | Local prerequisite checks pass. No live model/Mongo inference batch was rerun during wrap-up. |
| Metric-eval agent | Working offline; previously run live | Composite routing, versioned storage, scoring, observability, batching, and dashboard code are covered by tests. Live results cited below come from earlier jobs recorded in `NEXT_STEPS.md`. |
| Date-fix agent | Shipped | Tool isolation, mixed work-list construction, correction scoring, and the demo are covered by tests; the real-corpus rule is pinned at 30 matched / 3 ambiguous / 2 known misses. |
| Metric dashboard | Working | It now excludes date-fix rows from run counts, prompts, and path summaries. |
| Local Qwen judge | Scaffolded, not completed | Backend/config and 2× A40 serving scripts exist, but no server was active at wrap-up and `LOCAL_MODEL_URL` was unset. |
| Analysis notebooks | Archived results | `plot_metrics.ipynb` preserves upstream graphics/tables; `mongo_evals.ipynb` preserves MongoDB-backed evaluation analysis. Their live paths were not reproduced during handoff. |

## What works

### Upstream inference

The numbered scripts under `scripts/` cover PDF conversion, input indexing, mapping generation,
ground-truth extraction, model execution, result combination, and deterministic scoring. Models and
benchmarks remain data-driven through `inputs/models.json` and `inputs/benchmarks.json`. MongoDB
provides shared storage and idempotent writes for parallel SLURM jobs.

The local prerequisite checker is `scripts/check_setup.py`. Run its inference checks from the
repository root:

```bash
python scripts/check_setup.py inference
```

This confirms the Python environment, Poppler, `BASE_DIR`, Stanford API credential variable, and
MongoDB credential variables without contacting those services. It establishes local readiness, not
current external-service availability: model access can change when the Stanford Playground retires
models or issues replacement keys.

### Agentic evaluation

One shared runtime supports two tasks selected by prompt-name prefix:

- `composite_*` is metric-eval. The agent reads an output, routes each field to one of three
  composite type tools, and saves the evaluation. `selection_accuracy` measures the declared type
  choice; `routing_path_correct` measures the whole tool-call path.
- `date_fix_*` repairs the derived `tv_guide_date` from `newspaper_date` and `day_of_week`, recording
  `corrected`, `confirmed`, or `abstained`. Ground-truth columns bypass the agent and go directly to
  the scorer, so the agent never sees its grading answer.

Both tasks use the same runner, backend registry, MCP server, observability layer, Parquet sink, and
MongoDB `agentic_runs` mirror. Task answers remain separate in `agentic_evaluations` and
`agentic_corrections`, joined by `eval_id`. Client-side `eval_id` and `git_commit` stamping and
version-keyed skip-existing behavior are intentional compatibility contracts.

The offline suite passes **211 tests** with Weave disabled. The suite covers the agent loop, backend
configuration, CLI, mapping, scorers, sinks, dashboards, setup checks, inference-script contracts,
and the real date-fix corpus. The metric dashboard now fails closed for rows without a recognizable
metric prompt and excludes date-fix rows from metric counts and summaries.

### Recorded experiment results

These are historical results documented in `NEXT_STEPS.md`, not reruns performed during wrap-up:

- The `composite_v1` baseline improved selection accuracy from 0.483 to 0.651 after type-aware
  normalization, while redundant double-fetch behavior remained about 79%.
- `composite_v2` reduced double-fetches from 79% to 60%, cost by about 22%, and tokens by about 15%,
  but save success regressed from 98% to 79%. On a usable-run basis it did not improve routing.
- The date-fix task and reader-facing demo were completed, including explicit abstention and
  confidently-wrong reporting.

## Incomplete work and known risks

### Reliability bugs

- Sometimes a model sends a tool request in an invalid format. The system mistakenly keeps that
  broken request and sends it back to the model on the next attempt, which can cause the retry to
  fail immediately. This still needs a test and a fix that removes the broken request before
  retrying.
- Twenty-two historical metric-eval runs saved successfully but had no field matching
  `gold_metrics.csv`. They are now honestly displayed as **Unscored**, but the mismatch has not been
  explained.
- Unknown prompt-name prefixes silently fall back to the metric-eval tool set. A typo can therefore
  expose the wrong tools instead of failing during configuration.

### Local-model and prompt work

- `agent_eval/backends/qwen.json` and `multi_gpu_serve/` scaffold Qwen3.6-35B-A3B with vLLM on two
  A40 GPUs. The end-to-end judge run, concurrency measurement, and GPU telemetry validation remain
  undone. The archived serving plan also contains stale path/venv assumptions and needs
  reconciliation before it is treated as a runbook.
- Generation-time enforcement of valid function-call JSON should be tested on vLLM and, where
  supported, hosted Playground backends.
- A possible `composite_v2.1` prompt change—a final reminder that the agent is not done until it
  calls `save_evaluation`—was deliberately deferred until prompt-independent reliability problems
  are fixed. Any future comparison should reuse the same 40 outputs and report save success alongside
  routing accuracy to avoid survivorship bias.

### Reporting and documentation

- The self-refreshing dashboard SLURM script exists but still needs an owner to submit and monitor
  it. `routing_path_reason` is not yet surfaced in the runs table.
- Hosted-judge prices are snapshots, not permanent facts. Earlier runs used the prices present in
  their recorded `git_commit`, including the former 3/15 Claude configuration. Commit `0c58635`
  updated the configured Stanford Gateway rates to 1.5/7.5 for Claude Sonnet 4.6 and 0.125/1 for
  GPT-5-mini, based on Stanford's 50% discount and verified on 2026-08-18. DeepSeek-V3.2 remains
  unpriced because Stanford published no supported rate. Before new runs, recheck Stanford's rate
  page and follow the refresh procedure in the `agentic-eval` skill; do not rewrite historical run
  costs with newer rates.
- The README's upstream-inference cost findings are supported by saved tables in
  `notebooks/plot_metrics.ipynb`, even though the referenced `images/token_cost.png` chart was never
  committed. For the March 17, 2026 results, the notebook records an estimated $41.35 total for o1
  versus $8.76 for gemini-2.5-pro (about 4.7 times as much). It also supports the equal-accuracy
  comparisons for Newspaper Name (claude-3-haiku $0.06 versus gemini-2.5-pro $1.34) and Newspaper
  Date (gemini-2.0-flash-lite-001 $0.03 versus gemini-2.5-pro $1.49). It does not support the old
  claim that claude-3-haiku matched gemini-2.5-pro on Newspaper Date: their accuracies were 88.6%
  and 100%. These dollar amounts were calculated from recorded tokens and the static rates in
  `inputs/models.json`; before producing new figures, check Stanford's current Gateway rate page,
  update that file, run `python scripts/7_compute_metrics.py`, and rerun
  `notebooks/plot_metrics.ipynb`. Keep the old figures tied to their date and original rates.
- `notebooks/plot_metrics.ipynb` preserves the dated upstream model, benchmark, and image graphics
  and summary tables. `notebooks/mongo_evals.ipynb` preserves MongoDB-backed evaluation cleaning,
  weighted-score analysis, heatmaps, and spot checks. Both contain saved outputs and
  environment-specific paths; treat them as historical analysis records, not reproducible tests or
  the supported runtime for either pipeline.

## Setup and data-location knowledge

- Use one repository-local Python 3.10 environment: `source .venv/bin/activate`. Install the exact
  lock with `python -m pip install -r requirements.txt`.
- Run agentic Python modules and SLURM launchers from the repository root. The upstream inference
  launcher is the exception documented in the README: submit `scripts/yens.slurm` from `scripts/`.
- Run offline tests with `cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest`. Without
  `EVAL_DISABLE_WEAVE=1`, local tests may attempt tracing initialization.
- Secrets live only in the ignored `.env`; `.env.example` is the template. Never copy secret values
  into notebooks, generated dashboards, logs, issues, or commits.
- To obtain `STANFORD_API_KEY`, use Stanford UIT's
  [AI API Gateway service page](https://uit.stanford.edu/service/ai-api-gateway). Its **Get started**
  section links the new-key request and the current access instructions. A valid PTA and its approval
  are required; the same page links help for changing an existing key's model access. Store the
  issued credential only in the ignored `.env`.
- Source images and ground-truth files live below the ignored `inputs/data/` tree. Generated agent
  work lists live under `inputs/`; generated run Parquet lives under `outputs/agent_runs/`.
- The current code targets the DARC Atlas deployment and `usf-internship` MongoDB database. Hosts and
  database name are hardcoded in the data tools; only credentials are configured through `.env`.
- The agent CLI does not start MCP. Use the self-contained batch job or point `--mcp-url` at an
  already-running server. Both dashboard servers bind to localhost on headless nodes, so use SSH
  port forwarding or the VS Code Ports panel.
- Do not confuse `scripts/3_create_mapping.py` (inference work) with
  `agent_eval/registry/create_eval_mapping.py` (judge work), or `model_id` (model under test) with
  the judge backend/model identifier.
- Prompt names are currently part of task routing: `composite_*` receives metric tools and
  `date_fix_*` receives date tools. A new task requires both registered server tools and inclusion in
  that task's allowed-tool subset.
- Never expose expected dates, original values, or other grading truth to the date-fix agent. Those
  values belong only on the work-list-to-scorer path.
- Generated dashboards are intentionally ignored because they embed result data. Rebuild or serve
  them from the source templates instead of treating an HTML snapshot as source code.

## Prioritized follow-up backlog

The wrap-up backlog was checked against the repository's existing issues before filing:

1. [#26 — Fix malformed tool-call JSON poisoning retry history](https://github.com/gsbdarc/LLM_benchmarks/issues/26)
2. [#27 — Enforce structured tool-call generation where backends support it](https://github.com/gsbdarc/LLM_benchmarks/issues/27)
3. [#28 — Investigate saved-but-Unscored metric-eval runs](https://github.com/gsbdarc/LLM_benchmarks/issues/28)
4. [#29 — Fail fast on unknown prompt task prefixes](https://github.com/gsbdarc/LLM_benchmarks/issues/29)
5. [#30 — Complete the Qwen judge run and serving runbook on 2× A40](https://github.com/gsbdarc/LLM_benchmarks/issues/30)
6. [#31 — Revisit `composite_v2.1` after reliability fixes](https://github.com/gsbdarc/LLM_benchmarks/issues/31)
7. [#32 — Operationalize the living dashboard and expose routing diagnostics](https://github.com/gsbdarc/LLM_benchmarks/issues/32)
8. [#34 — Make the MongoDB deployment and database configurable](https://github.com/gsbdarc/LLM_benchmarks/issues/34)

## Verification performed for this handoff

The initial environment wrap-up on 2026-08-25 recorded:

```text
python scripts/check_setup.py python                  5 passed, 0 failed
python scripts/check_setup.py inference              10 passed, 0 failed
python scripts/check_setup.py agent-eval --backend playground
                                                     8 passed, 0 failed
python scripts/check_setup.py agent-eval --backend qwen --batch
                                                     8 passed, 1 failed
                                                     (LOCAL_MODEL_URL not set)
cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest
                                                   211 passed
```

The final archival pass on 2026-08-26 repeated the Python setup profile (5 passed), validated all
tracked Markdown file targets and notebook JSON, rendered the metric dashboard from the 240-run
local cache, rendered the date-fix demo from MongoDB (66 answers for each of two judges), and reran
the offline suite (211 passed in 8.46 seconds). The pricing metadata test was exercised red/green:
it first failed on the missing snapshot metadata and then passed after that metadata was added.

The final date-fix render proves read access to the configured MongoDB at that moment; it did not
write data or rerun a batch. Stanford Playground, Weave, SLURM jobs, GPU serving, and browser-based
visual review were not exercised, so their current status should not be inferred from these checks.
