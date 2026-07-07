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

## Layout

| Path | Role |
|---|---|
| `metric_server.py` | FastMCP HTTP server. 3 data tools (`list_outputs`, `get_task_output`, `save_evaluation`) + **3 composite type-tools** (`evaluate_raw_string/extracted_string/list`). |
| `metric_tools.py` | MongoDB business logic; imports metrics/composites from `scripts/evaluator.py` (one source of truth). |
| `eval/config.py` | Backend resolution + clients. Loads `eval/backends/*.json`. |
| `eval/backends/*.json` | One file per **endpoint** (URL + framework + auth) with an int-keyed `models` map. |
| `eval/prompts.py` | Agent system/user prompts; `PROMPT_NAME` version. |
| `eval/agent.py` | MCP session + agent loop (per-step reasoning, usage, GPU scrape). |
| `eval/observability.py` | Hashes, derivations (tokens/sec, peak ctx, `reasoning_blob`), vLLM/NIM metrics scrape. Weave optional (`EVAL_DISABLE_WEAVE=1`). |
| `eval/integrity.py` | Score-consistency + retry checks. |
| `eval/scorers.py` | Weave scorers incl. routing `selection_accuracy`. |
| `eval/sink.py` | Flatten one run → one Parquet row (the OLAP layer). |
| `eval/runner.py`, `eval/__main__.py` | Batch orchestration + CLI (`python -m eval`). |
| `../analysis/queries.py` | DuckDB views over the Parquet runs. |
| `../analysis/build_dashboard.py` | Generate the self-contained HTML dashboard. |
| `agent_dashboard_sample_v2.html` | Hand-authored mockup (fake data) — the visual target. |

## Choices & why

- **3 composite type-tools, not 7 raw metrics.** There is one *correct* calculation
  per field (from the original manual scoring). Collapsing to `evaluate_<type>`
  makes "did the agent route to the right shape?" simple to monitor while we
  experiment with prompts. `selection_accuracy` now = correct type-tool routing,
  graded against the gold `field_type` in `gold_metrics.csv`. The 7 metrics live on
  as private helpers inside the composites.
- **Single source of truth for scoring.** The per-type composites live in
  `scripts/evaluator.py` and are called by *both* the deterministic pipeline and the
  MCP type-tools, so agent vs baseline scores match by construction.
- **Backends = per-endpoint JSON files, int-keyed models.** `eval/backends/<endpoint>.json`
  declares the endpoint once (base_url/framework/api-key-env) and lists models under
  integer keys. Endpoints are few & stable; models are many. Run any model on an
  endpoint with `--model <int>` (or a raw model-id) — no new file per model. The int
  is the *agent/eval* model (stored as `agent_model_key`), distinct from `model_id`
  (the model being evaluated).
- **Two observability stores.** Weave = deep per-run traces (drill into reasoning);
  Parquet+DuckDB = wide, many-run rows for the dashboard. `reasoning_json` is a
  bounded per-run blob that feeds the dashboard's path summaries.
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

## Running

```bash
# tests (offline; weave disabled)
cd mcp && source ~/venv/bin/activate
EVAL_DISABLE_WEAVE=1 python -m pytest eval/tests/ -q

# MCP server + one agent run
./run_metric_mcp.sh
python -m eval --backend nim --mcp-url <url> --benchmarks 5 --limit 1 --verbose

# dashboard
python -m analysis.build_dashboard --open              # summaries on (needs summarizer up)
python -m analysis.build_dashboard --no-summaries      # skip summaries
```

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
