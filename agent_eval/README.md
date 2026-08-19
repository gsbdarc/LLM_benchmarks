# Agentic Metric-Eval — design log

A running, concise record of **what the pieces do, the choices we made, and why.**
Keep it current as the code changes. (Team-facing: the parallel-local-model
findings at the bottom are shared learnings.)

## What this is

The benchmark's metric-calculation step, reimagined as an **agent**. Instead of
deterministic code picking a metric per field, an LLM agent looks at each field's
`predicted` vs `expected` values, infers the **data shape**, and routes it to the
right scoring tool over MCP. We measure how well it routes, plus performance/infra.

The deterministic path (`scripts/6b_mongo_eval.py` + `scripts/evaluator.py`) still
exists and is the ground-truth baseline the agent is checked against.

That was the first task. There are now **two** — see [Two tasks on one
harness](#two-tasks-on-one-harness) before assuming a run is a metric eval.

## Layout

`agent_eval/` is one package (a sibling of `scripts/` and `analysis/` at the repo root),
with the library modules grouped by concern into subpackages. Entry points + foundational
modules sit at the top; run entry points as `python -m agent_eval …` from the repo root.

| Path | Role |
|---|---|
| `__main__.py` | CLI entry — `python -m agent_eval` (batch, `--eval-mapping … --row` worker mode, `--rows A-B` shard mode). |
| `server.py` | FastMCP HTTP server (`python -m agent_eval.server`): 3 data tools + **3 composite type-tools** (`evaluate_raw_string/extracted_string/list`) + **3 date-fix tools** (`get_guide_date_case`, `compute_guide_date`, `save_correction`). Every task's tools live on one server; the *agent* is shown a subset. |
| `tools.py` | MCP tool business logic; MongoDB access + metric/composite imports from `scripts/evaluator.py` (one source of truth). |
| `config.py` | Backend resolution + clients; loads `backends/*.json`; path anchor (`PKG_DIR`/`REPO_ROOT`). |
| `prompts/` | Prompt registry: one JSON per variant (`composite_v1.json`, `date_fix_v1.json`); `__init__.py` loader exposes `PROMPT_NAME`/`METRIC_EVAL_SYSTEM`/`eval_user_prompt`/`resolve_prompt`. The prompt name also selects the **task** (see below). |
| **`runtime/`** | The agent execution engine: `agent.py` (persistent MCP session + agent loop + `tools_for_prompt`), `runner.py` (batch orchestration). |
| **`reporting/`** | Measuring + recording a run: `observability.py` (traces/derivations/scrape), `scorers.py` (`selection_accuracy`, `routing_path_scorer`, `date_fix_scorer`), `integrity.py`, `sink.py` (one Parquet row per run). |
| **`registry/`** | The work registry: `mapping.py` (pure build/dedupe/sample), `create_eval_mapping.py` (metric-eval), `create_date_fix_mapping.py` (date-fix), `generate_gold_metrics.py`. |
| **`scripts/`** | Launchers / SLURM jobs: `run_metric_mcp.sh`, `run_eval_batch.slurm` (self-contained batch — the easy path), `run_eval_array.slurm` (array variant). |
| `backends/*.json` | One file per **endpoint** (URL + framework + auth) with an int-keyed `models` map. |
| `gold_metrics.csv` | Gold `field_type` per benchmark field (routing-accuracy key; regenerate via `registry/generate_gold_metrics.py`). |
| `tests/` | Unit tests. `cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest`. |
| `agent_dashboard_sample_v2.html` | Hand-authored dashboard mockup (fake data) — the visual target. |
| `../analysis/queries.py`, `../analysis/build_dashboard.py`, `../analysis/serve_dashboard.py` | DuckDB views + HTML dashboard generator + live server (:8787) — **metric-eval**. |
| `../analysis/build_date_fix_demo.py`, `../analysis/serve_demo.py` | Demo-page generator + live server (:8788) — **date-fix**. |

## Two tasks on one harness

The harness now runs a **second task**, and everything above it is shared. The prompt
variant is the switch: `runtime/agent.py:tools_for_prompt` maps a prompt-name **prefix**
to a tool set (`_TOOLS_BY_TASK`), so a `date_fix_*` run is shown only the date-fix tools
and a `composite_*` run only the metric-eval ones. The server still serves everything —
the restriction is on what the *agent* sees, so `runner.discover_rows` is unaffected.

**metric-eval** (`composite_v*`) — the original: read one output's fields, infer each
field's data shape, route to the right `evaluate_*` composite, `save_evaluation`. Graded
on routing (`selection_accuracy`, `routing_path_correct`) against `gold_metrics.csv`.

**date-fix** (`date_fix_v1`) — a repair task, not a grading task. A TV guide covers one
day, and that date is *not printed on the page*: it has to be derived from the newspaper's
publication date and which weekday the guide is for. Earlier models extracted all three
values independently, so `tv_guide_date` is often inconsistent with the other two. The
agent loads the case (`get_guide_date_case`), derives the right date
(`compute_guide_date` — the first occurrence of the weekday on or after publication), and
records one decision (`save_correction`): `corrected`, `confirmed`, or `abstained`.

Reproduce the demo end to end, from the repo root:

```bash
python -m agent_eval.registry.create_date_fix_mapping        # -> inputs/date_fix_mapping.csv
sbatch --export=ALL,SAMPLE=inputs/date_fix_mapping.csv \
       agent_eval/scripts/run_eval_batch.slurm               # same batch job as metric-eval
python -m analysis.serve_demo                                # -> http://localhost:8788 (+ ssh -L hint)
#   or a static file:  python -m analysis.build_date_fix_demo
```

`tests/test_date_fix.py` pins the date rule against the real 35-image corpus
(30 matched / 3 ambiguous / 2 known misses) — if a change to the rule or the parser moves
those numbers the suite fails and names the image.

## Choices & why

- **Ground truth never reaches the date-fix agent.** `expected_date` and `original_value`
  are columns on the work-list row; `__main__.py:_work_row` carries them past the agent
  straight to `date_fix_scorer`. Two reasons. Judging a *regression* against a value the
  agent reported itself would let it grade its own baseline. And the agent's real job is
  to decide whether the date is consistent with the two values it derives from — showing
  it the answer would test the wrong thing.
- **`confirmed` is split by whether it was right.** `fix_outcome` distinguishes
  `confirmed_correct` from `confirmed_wrong` (and `fixed_correct` from `fixed_wrong`)
  because "confirmed" is a *recordable success*: an agent that blesses every row would
  otherwise score perfectly. The work list is mixed on purpose —
  `create_date_fix_mapping.py` samples **wrong** rows (it should fix), **control** rows
  (already right; any change is a regression), and **range** rows — so one batch can show
  a regression rather than only rewarding action.
- **Abstention is a first-class answer.** Some images' ground truth is a *span*
  ("April 4-8 2005"), and a weekday range like "Monday-Friday" doesn't describe a single
  day. `compute_guide_date` returns `{"ambiguous": reason}` with no date, and the right
  move is `action="abstained"`, not a guess. `truth_parseable` marks these rows so a
  confident date there scores as wrong.
- **`needs_review` is orthogonal to correctness.** The agent's own uncertainty flag is
  recorded separately from whether it was right, so we can ask the question that matters:
  does the flag fire on the rows it actually gets wrong? A flag that never fires on real
  errors is decoration. The demo reports `confidently_wrong` (wrong *and* unflagged)
  separately from wrong-but-flagged — the latter still reached a human.
- **Rows with unsound inputs are excluded, and counted.** Every bucket requires the model
  to have read *both* inputs correctly, so each row is internally consistent and the agent
  is judged on the derivation alone. That check used to apply only to "wrong", which let a
  contradictory row into the controls: one model landed on the right final date while
  reporting the publication year as 1902, so recomputing from the bad input "corrected" a
  right answer and counted as the agent's failure when the row was the problem.
- **3 composite type-tools, not 7 raw metrics.** There is one *correct* calculation
  per field (from the original manual scoring). Collapsing to `evaluate_<type>`
  makes "did the agent route to the right shape?" simple to monitor while we
  experiment with prompts. `selection_accuracy` now = correct type-tool routing,
  graded against the gold `field_type` in `gold_metrics.csv`. The 7 metrics live on
  as private helpers inside the composites.
- **Single source of truth for scoring.** The per-type composites live in
  `scripts/evaluator.py` and are called by *both* the deterministic pipeline and the
  MCP type-tools, so agent vs baseline scores match by construction.
- **Backends = per-endpoint JSON files, int-keyed models.** `backends/<endpoint>.json`
  declares the endpoint once (base_url/framework/api-key-env) and lists models under
  integer keys. Endpoints are few & stable; models are many. Run any model on an
  endpoint with `--model <int>` (or a raw model-id) — no new file per model. The int
  is the *agent/eval* model (stored as `agent_model_key`), distinct from `model_id`
  (the model being evaluated).
- **Two observability stores, plus a central mirror.** Weave = deep per-run traces
  (drill into reasoning); Parquet+DuckDB = wide, many-run rows for the dashboard (the
  authoritative local record). `reasoning_json` is a bounded per-run blob that feeds the
  dashboard's path summaries. Each run row is also mirrored to the `agentic_runs` Mongo
  collection so the whole team can query results (perf, `selection_accuracy`, `save_success`,
  `weave_trace_url`, …) without Yen access — upsert-keyed on the output × judge-config
  identity, and failures never break the run. Disable with `--no-mongo-runs`. `agentic_runs`
  holds runs of **both** tasks; the agent's actual answers live in the per-task collections
  `agentic_evaluations` (metric-eval per-field verdicts) and `agentic_corrections`
  (date-fix decisions), joined back by `eval_id`.
- **GPU type is propagated, not sniffed.** The eval client and GPU server are
  different hosts, so `nvidia-smi` is never read on the client. `run_nim_server.slurm`
  echoes the GPU name; pass it via `--gpu-type` or `$GPU_TYPE`. The infra panel uses
  the per-run scraped Prometheus numbers (`gpu_cache_usage_end`, `requests_running_end`);
  wiring the SLURM per-second CSV is a deferred follow-up.
- **Dashboard is generated, not hand-built.** `build_dashboard.py` injects DuckDB rows
  + a prompt registry + glossary + LLM path-summaries into `analysis/dashboard_template.html`.
  Path summaries are a **sync one-shot step**, cached **by path signature** in a JSON
  file — a path is summarized once and reused across builds (later builds make zero LLM
  calls unless a new path appears; `--refresh-summaries` forces regeneration). They
  **degrade gracefully**: if the summarizer endpoint is down, the build still succeeds
  without blurbs. Summarizer currently reuses the working gemma NIM server
  (`summarizer` backend).
- **Dashboard views.** Group breakdown is regroupable by **model or benchmark** (also
  prompt / framework / GPU / temperature); each group shows tokens, **latency split into
  `llm_time` (model/service) vs `overhead_time` (local tool + loop)**, routing accuracy,
  and save rate. The **tool-call path / routing** DAG renders full-size (scrolls
  horizontally) with divergent (mis-routed) paths flagged. `eval_id` joins each run row
  to its `agentic_evaluations` verdict.

## Running

All commands run **from the repo root**.

```bash
# tests (offline; weave disabled)
source .venv/bin/activate && cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest -q; cd ..

# MCP server (hand) + one agent run
agent_eval/scripts/run_metric_mcp.sh
python -m agent_eval --backend nim --mcp-url <url> --benchmarks 5 --limit 1 --verbose

# dashboard (metric-eval)
python -m analysis.build_dashboard --open              # summaries on (needs summarizer up)
python -m analysis.build_dashboard --no-summaries      # skip summaries
python -m analysis.serve_dashboard                     # live -> http://127.0.0.1:8787

# demo page (date-fix)
python -m analysis.serve_demo                          # live -> http://127.0.0.1:8788
```

Both servers bind localhost on a headless node, so forward the port to see them
(`serve_demo` prints the exact `ssh -N -L …` line for the host it's on; VS Code's Ports
panel does the same thing).

## Running a playground batch

The playground API can't take many concurrent requests *on one connection* (in-process
`--concurrency` chokes), but it's fine with **many independent processes** each doing one
request at a time. So we parallelize with separate worker processes, and because the API is
remote (no GPU, no `/metrics`) we skip GPU metrics. Everything runs on **one compute node**
(`yen10-16/20` — not the login hosts) over `127.0.0.1`.

**Easy path — `run_eval_batch.slurm` (self-contained: one job starts its own server + fans out).**
From the repo root:
```bash
python -m agent_eval.registry.create_eval_mapping --sample 100      # -> inputs/eval_mapping_sample.csv
sbatch agent_eval/scripts/run_eval_batch.slurm                     # picks a node, runs everything
#   tune:  sbatch --export=ALL,MAXPAR=12,WEAVE=0 agent_eval/scripts/run_eval_batch.slurm
python -m analysis.build_dashboard --no-summaries          # after it finishes
```
The registry crosses sampled outputs × judge-config; today one config
(`playground` / gpt-5-mini / `composite_v1`), so it's 1:1 with the sample. Only benchmarks with
a gold field_type ({5,6,7,10,11}) are included (else routing accuracy is null). `MAXPAR` caps
concurrent playground calls; failed rows are listed in the job `.out` — rerun by re-running the
batch (idempotent upsert, so re-runs are safe). The array variant (`run_eval_array.slurm`) is
there if you prefer a persistent shared server pinned to a node with `--array=…%N`.

What this gives the dashboard: real routing accuracy, tokens, steps, tool-call paths,
save-success, per-benchmark comparisons. NOT the sequential-vs-parallel / GPU-pressure panel
— that's inherently a local-model story (cross-process parallelism is invisible to the per-run
`concurrency` field, and the playground exposes no GPU metrics).

## Parallel local models — findings log

> Goal: make `--concurrency` actually parallelize LLM calls against local servers,
> and record what serving configs do/don't batch. (Populated during Part 6.)

- **Confirmed problem:** `--concurrency` did NOT parallelize LLM calls — the client
  was a synchronous `OpenAI(...)` called blocking inside the async agent loop, so
  concurrent runs serialized at the LLM layer (`requests_running` stayed ~1).
- **Fix applied (Part 6):** `config.build_backend` now builds `AsyncOpenAI`, and
  `agent.make_llm_step` is a coroutine the loop `await`s, so `run_batch`'s
  `asyncio.gather` genuinely overlaps requests. `tenacity @retry` works on the
  coroutine (verified in `test_agent.py`); the dashboard builder uses the separate
  sync `config.sync_openai_client` so it needs no event loop.
- **Client-side check:** run `--concurrency 4` vs `1` against a local backend and
  confirm scraped `requests_running_end` rises above 1 and batch wall-time drops.
- **Server layer matters too:** the engine must accept concurrency.
  - vLLM / NIM: continuous-batch natively — expect `requests_running` to climb.
  - **Ollama: serializes unless `OLLAMA_NUM_PARALLEL` is set** on the server. Set it
    in the launch script before concluding "async didn't help."
- _What worked / what didn't (fill in as we test live):_
  - …
