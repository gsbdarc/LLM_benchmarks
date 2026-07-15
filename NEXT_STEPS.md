# Next steps — pick up Tuesday

Goal: run the first real agentic-eval batch on the **playground** (gpt-5-mini) over ~100
real `llm_outputs`, get the data into the dashboard. All tooling is built, tested, and the
directory reorg is done.

## Where we left off (Fri)
- Single-row smoke **passed** end-to-end (worker → MCP server → Mongo → playground agent →
  composite tool → save → Parquet + Weave trace URL), and the 3 fixes verified on the real row
  (`agent_model_key`, `weave_trace_url`, `save_success`/`save_failed`).
- `weave` installed into `~/venv` (was missing).
- **Directory reorg DONE:** `mcp/` → **`agent_eval/`** (one flat package at the repo root,
  sibling of `scripts/`/`analysis/`). `python -m eval` is now **`python -m agent_eval`**;
  `metric_server.py`→`server.py`, `metric_tools.py`→`tools.py`; the `eval/` subpackage is gone.
  Non-eval bits moved out: `redivis_example/`→`reference/`, notebooks→`notebooks/`. **112 tests green.**
- **Uncommitted:** the whole reorg + eval-mapping pipeline + worker mode + slurm scripts + the
  save/weave/model fixes. **Commit first thing** (it's a big rename — commit so it's safe).

## Run it (all from the REPO ROOT)

```bash
cd /yen/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks

# 0. commit the reorg + tooling (big uncommitted change)
git add -A && git commit -m "Reorg mcp/ -> agent_eval package; eval-mapping run tooling + fixes"

# 1. clean slate (drops the smoke rows + capped validation registry)
rm -rf outputs/agent_runs/* ; rm -f inputs/eval_mapping*.csv

# 2. build the 100-row sample (queries Mongo; ~seconds)
source ~/venv/bin/activate
python -m agent_eval.registry.create_eval_mapping --sample 100

# 3. fire the self-contained batch (SLURM picks a compute node, runs its own server)
sbatch agent_eval/scripts/run_eval_batch.slurm
#    knobs:  --export=ALL,MAXPAR=12   (more concurrency)
#            --export=ALL,WEAVE=0     (if Weave/W&B can't be reached from compute nodes)

# 4. check results
cat eval-batch-<jobid>.out            # "Batch complete: X/N rows produced a Done line"
ls eval-logs/                         # per-row logs; failed rows listed in the .out

# 5. build the dashboard from the real rows
python -m analysis.build_dashboard --no-summaries      # -> images/agent_dashboard.html
```

## Watch for
- **`weave.init` errors in `eval-logs/row-*.out`** = compute nodes can't reach wandb.ai.
  Re-run with `sbatch --export=ALL,WEAVE=0 agent_eval/scripts/run_eval_batch.slurm` (data still flows to
  Parquet; only the trace deep-link is lost). This is the one thing the batch will tell us that
  we couldn't test in advance.
- **Routing accuracy / agent behavior**: the smoke showed gpt-5-mini double-saving and
  misrouting benchmark 6 (raw_string → extracted_string). Expected signal, not a bug — look at
  the spread across the batch; it's fodder for a prompt tweak later.

## Reorg landed (reference)
`agent_eval/` is now the one package. Test it with `cd agent_eval && EVAL_DISABLE_WEAVE=1
python -m pytest`. `agent_eval/README.md` is the design log (layout + run recipe + findings).

## Later (needs the GPU, currently busy)
- Local-model (NIM) run to populate the sequential-vs-parallel / GPU-pressure panel and the
  `--concurrency N` study (`requests_running_end` should rise >1 vs sequential).
- Path-summaries on the dashboard (needs the gemma summarizer server up).
