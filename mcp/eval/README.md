# Agentic Metric Evaluation — a guide

*Audience: you know the `LLM_benchmarks` pipeline (`5_main.py` → MongoDB
`llm_outputs`, `evaluator.py`, `6b_mongo_eval.py`, `plot_metrics.ipynb`) but are
new to MCP, agents, and observability. This guide starts with the big picture,
then explains the three new ideas, then shows how to run it.*

---

## 1. The big picture: why this exists

In the pipeline today, evaluation is **deterministic**. For each benchmark we
*declare* which metric to use (in the benchmark's `ground_truth` block), and
`6b_mongo_eval.py` applies exactly that metric from `evaluator.py`. That works,
but it bakes in an assumption: that a human already knew the right metric for
every field.

The harder question is: **given a predicted value and a ground-truth value, which
metric is even appropriate?** A free-form OCR line wants `word_iou`; a single
extracted value that might be missing wants `null_accuracy` plus a content score;
an unordered list wants `set_f1`. Choosing well requires *looking at the data*.

This project hands that judgment to an **LLM agent**. The agent reads one output,
reasons about each field's shape, calls the matching metric, and writes its
verdict to a new MongoDB collection, `agentic_evaluations` — sitting alongside the
deterministic results rather than replacing them.

That raises a second, equally important goal. Once an LLM is making decisions in a
loop, you need to **see what it did** — which metric it chose and why, how many
steps it took, how many tokens, how fast, whether it ever invented a number. That
visibility is what "observability" means here, and it's the larger half of this
work.

So there are two aims:

1. **Get good, well-reasoned metric choices** from the agent.
2. **Observe the agent across many runs** so we can trust it, compare prompt and
   model variants, and find performance limits.

---

## 2. Three new ideas

### 2a. An MCP server and its "tools"

An LLM can only emit text. To let it *do* things, you give it **tools** — named
functions with typed arguments that it can ask to call. **MCP** (Model Context
Protocol) is just a standard way to expose those tools over a connection, so the
same tools can be reused by any agent.

Here, [`metric_server.py`](../metric_server.py) is the MCP server. It wraps things
you already know:

- the seven metric functions from [`evaluator.py`](../../scripts/evaluator.py)
  (one tool each: `word_iou`, `null_accuracy`, …), and
- three data tools backed by MongoDB: `list_outputs`, `get_task_output`,
  `save_evaluation`.

Crucially, `get_task_output` returns the predicted/expected values but **hides the
declared metric** — otherwise there'd be nothing for the agent to reason about.

### 2b. The agent loop

An "agent" is an LLM run in a loop:

```
think  →  call a tool  →  read the result  →  think again  →  …  →  final answer
```

Each turn, the model either asks to call a tool or decides it's done. Our loop
lives in [`agent.py`](agent.py) (`run_agent`). For one output it typically looks
like: `get_task_output` → classify the field → call a metric tool → `save_evaluation`
→ summarize. The system prompt that teaches it the classification rules is in
[`prompts.py`](prompts.py).

### 2c. Observability: two stores, not one

You can't inspect an LLM's reasoning after the fact unless you **record it as it
happens**. We record at two altitudes — the same OLTP-vs-OLAP split you'd use for
any data system:

- **Weave (the trace store)** — *one run, deep.* Every LLM call, tool call,
  reasoning block, and integrity check for a single run, with a clickable URL.
  Great for debugging "what did it do on task 2450?". This is the transactional,
  drill-in view.
- **Parquet + DuckDB (the analytics store)** — *many runs, wide.* One flat row per
  run ([`sink.py`](sink.py)), written to `outputs/agent_runs/`. Great for "average
  tokens/sec by model" or "selection accuracy across 500 runs". This is the
  aggregate, SQL-friendly view — the natural successor to what
  [`plot_metrics.ipynb`](../../notebooks/plot_metrics.ipynb) does today.

On top of the *what-it-decided* trace, we also record *how-it-performed*:
tokens/sec, peak context size, reasoning level, and GPU/queue pressure. The GPU
numbers come from two places because the model runs on a **different machine**
(`yen-gpu4`) than this code:

- the vLLM/NIM **`/v1/metrics`** endpoint (scraped over HTTP — cache usage, how
  many requests were running), and
- an **`nvidia-smi` sampler** added to [`run_nim_server.slurm`](../../run_nim_server.slurm)
  for true SM utilization and GPU RAM, joined back by timestamp.

---

## 3. How it all fits together

```
 MongoDB llm_outputs  (from 5_main.py)
          │
          ▼
   ┌──────────────────┐     calls tools over MCP      ┌──────────────────┐
   │   agent loop      │ ───────────────────────────▶ │  metric_server.py │
   │   (run_agent)     │ ◀─────────────────────────── │  (evaluator.py +  │
   └──────────────────┘     tool results               │   MongoDB)        │
       │     │                                          └──────────────────┘
       │     └──────────────▶ MongoDB agentic_evaluations  (save_evaluation)
       │
       ├──▶ Weave trace           (one run, deep)
       └──▶ Parquet row → DuckDB  (many runs, wide) → dashboard
```

---

## 4. How to run it

From the `mcp/` directory:

1. **Start the MCP server**: `./run_metric_mcp.sh` — copy the printed
   `http://HOST:PORT/mcp` URL.
2. **Start a model.** For the `nim` backend: `sbatch ../run_nim_server.slurm`
   (serves `google/gemma-4-31b-it` on `yen-gpu4:8000` and writes the GPU sampler
   CSV). The `playground` backend uses the Stanford API instead — no GPU job
   needed.
3. **Run the agent:**

   ```bash
   # one output, to sanity-check the whole path
   python -m eval --backend nim --mcp-url http://127.0.0.1:PORT/mcp \
       --benchmarks 5 --limit 1

   # a parallel batch across several benchmarks
   python -m eval --backend nim --mcp-url http://127.0.0.1:PORT/mcp \
       --benchmarks 5,6,7,10,11 --limit 5 --concurrency 4
   ```

Handy flags: `--verbose` (print the full transcript), `--no-weave` (skip tracing),
`--no-sink` (skip Parquet), `--no-gpu-metrics` (skip the `/v1/metrics` scrape).

### What you get back

- A **Weave URL** per run (unless `--no-weave`).
- **Parquet rows** under `outputs/agent_runs/date=YYYY-MM-DD/` — query them:

  ```python
  from analysis.queries import connect, summary_by, concurrency_summary
  con = connect()
  print(summary_by(con, "backend", "reasoning_level"))
  print(concurrency_summary(con))     # sequential vs parallel
  ```

- New **`agentic_evaluations`** documents in MongoDB.
- The **GPU CSV** at `$NIM_ROOT/gpu-usage-<jobid>.csv` on the NIM box.

---

## 5. Going parallel, and the sweep experiment

Earlier the batch was capped at two concurrent runs because the server dropped
connections under load. The root cause was that every tool call opened a brand-new
connection; `agent.py` now keeps **one MCP connection per run** and retries
transient drops, so you can safely raise `--concurrency`.

Because the model lives on a single H200, there's a real ceiling. To find it (and
to answer "does GPU utilization change when we go parallel?"), run the sweep — it
runs the same outputs at several concurrency levels and records GPU/throughput
throughout:

```bash
python -m analysis.sweep --backend nim --mcp-url http://127.0.0.1:PORT/mcp \
    --benchmarks 5,6,7 --limit 5 --concurrency 1,2,4
```

---

## 6. The dashboard

A **single self-contained HTML file** — same convention as the team's "Stanford AI
API Token Usage Dashboard": the data is embedded as JSON and rendered by vanilla
JS + inline SVG, so there's no server or ports to manage. Generate it after some
runs exist:

```bash
python -m analysis.build_dashboard            # -> images/agent_dashboard.html
python -m analysis.build_dashboard --open     # also open in a browser
```

The headline view is **averages across prompts × models** — tool calls, tokens,
tokens/sec, steps, selection accuracy — with a group-by toggle (Model / Prompt /
Model×Prompt), a sequential-vs-parallel panel, and a per-run table that links back
to each Weave trace. A demo with synthetic data lives at
`images/agent_dashboard_sample.html`. The template is
[`analysis/dashboard_template.html`](../../analysis/dashboard_template.html);
the generator is [`analysis/build_dashboard.py`](../../analysis/build_dashboard.py).

---

## 7. Tests

```bash
cd mcp && python -m pytest      # ~70 unit tests, fully offline
```

Tests never hit the network, MongoDB, a GPU, or Weave: the agent loop runs against
fake LLM/MCP objects, and Weave tracing is auto-disabled
(`EVAL_DISABLE_WEAVE=1`, set in `conftest.py`). See [`eval/tests/`](tests).

---

## 8. File map

| File | Role |
|---|---|
| `config.py` | backends, OpenAI client, paths, `reasoning_level`, `/v1/metrics` URL |
| `prompts.py` | the system prompt + `PROMPT_NAME` (bump it when the prompt changes) |
| `observability.py` | hashes, token/reasoning derivations, Prometheus scrape, Weave toggle |
| `agent.py` | tool discovery + the agent loop (persistent MCP session, retries) |
| `integrity.py` | post-run checks (did it save once? did it invent any scores?) |
| `scorers.py` | the four `weave.Evaluation` scorers |
| `sink.py` | flatten one run → one Parquet row |
| `runner.py` | shared setup + bounded-parallel batch |
| `__main__.py` | the CLI (`python -m eval`) |
| `../../analysis/` | `queries.py` (DuckDB), `sweep.py`, `build_dashboard.py` + `dashboard_template.html` |
