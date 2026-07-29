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

## v2 measurement — DONE (job 397060 @ `fdfd6e5`, 120/120). Result: MIXED, not a clear win.
Same 40 outputs as v1 (`--sample-like`), composite_v2, 3 judges. v2 vs v1 (`a8d6f22`):
- **Double-fetch 79% → 60%** (v2's goal, partially met; gpt-5-mini 78%→40%). **Cost −22%, tokens −15%.**
- **Metric identified (usable basis, unsaved=0) 57% → 53%** — slightly WORSE. The raw 65→70% was
  survivorship bias (v2 graded 91 vs v1's 106; harder runs failed to save).
- **Save success 98% → 79%** — REGRESSION. 18/25 misses: gpt-5-mini (16) ended in prose after
  `get_task_output → list_outputs`, never calling `evaluate_*` or `save_evaluation` (v2 dropped the
  procedure → weak models lose the workflow). Other 7 = DeepSeek 400s (malformed tool-call JSON).
Verdict: v2 is leaner but less reliable and no better at routing. Two follow-ups (TBD, see below).

## TBD (both prompt-independent; do before trusting any prompt on weak models)
1. **History-sanitation** for the malformed-tool-JSON 400: in `agent.py`, the raw assistant msg with the
   bad `tool_call.arguments` is appended (line ~289) BEFORE the parse-error branch, so it re-poisons the
   next request → hard 400 (DeepSeek, 7/40). Fix: repair/drop the unparseable tool_call in the stored
   message so history stays clean. + test that reproduces the poisoned transcript.
2. **Generation-time enforcement**: guided/constrained decoding or strict function schemas where we
   control serving (the local Qwen vLLM server) so invalid tool JSON is impossible; try the strict-schema
   request flag for the playground models too (may be ignored by DeepSeek-via-Azure).
(A v2.1 prompt that restores a terminal "you're not done until save_evaluation" nudge is the OTHER lever,
but deferred — user wants to hold on prompt iteration.)

## IMMEDIATE NEXT: #4 Local Qwen judge (scaffolding BUILT; needs a GPU allocation to run)
Qwen3.6-35B-A3B via vLLM, TP=2 on **2× A40** (yen-gpu2/gpu3 are the A40 nodes, 48 GiB each — check with
`sinfo -p gpu -N -h -o "%N | %f" | grep A40`; bf16 fits in 96 GiB). Serve script: `multi_gpu_serve/serve.sh`.
Already done (committed): `backends/qwen.json` (framework vllm, base_url `$LOCAL_MODEL_URL`, prices 0/0),
`config._resolve_base_url` (env base_url), `config.metrics_url(..., "vllm")` → root `/metrics`,
`create_eval_mapping --judge-backends` (adds the local judge). `serve.sh` now has
`--enable-auto-tool-choice --tool-call-parser hermes` (this vLLM 0.25.1 has NO bare `qwen3` tool
parser — `hermes` for Qwen3 instruct; `qwen3_xml` fallback). To run once a node is allocated:
```bash
# 1. grab 2 A40 + serve (INSIDE the allocation):
srun -p gpu -C GPU_MODEL:A40 -G 2 -c 16 --mem=100G -t 2:00:00 --pty /bin/bash
cd multi_gpu_serve && bash serve.sh            # note the host it prints, e.g. yen-gpu2:40777
# NB: venv_1/bin/vllm has a stale shebang; if activating venv_1, use `python -m vllm serve` instead.
# 2. from the login node / eval host:
export LOCAL_MODEL_URL=http://yen-gpu2:40777/v1
# pre-flight ONE eval against the local judge (GPU metrics on, pass the GPU name):
python -m agent_eval --backend qwen --mcp-url http://127.0.0.1:PORT/mcp \
    --benchmarks 5 --limit 1 --gpu-type "NVIDIA A40"
# 3. build a qwen sample over the SAME 40 outputs (composite_v2) and run the batch:
python -m agent_eval.registry.create_eval_mapping \
    --judge-backends qwen --prompts composite_v2 --sample-like inputs/eval_mapping_sample_v1.csv
GPU_TYPE="NVIDIA A40" sbatch agent_eval/scripts/run_eval_batch.slurm   # (drop --no-gpu-metrics for GPU stats)
```

## Then
- Confirm the **claude-sonnet-4-6 price** (assumed 3/15).
- `sbatch agent_eval/scripts/refresh_dashboard.slurm` once to make the dashboard "living".
- Surface `routing_path_reason` in the dashboard runs table (deferred until the local summarizer is up).

## Recipes (from repo root)
```bash
source ~/venv/bin/activate
cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest ; cd ..          # 145 tests
# rebuild dashboard from the central store:
python -m analysis.export_runs --out outputs/dashboard_cache
python -m analysis.build_dashboard --base-dir outputs/dashboard_cache --no-summaries \
    --out images/agent_dashboard_live.html
# compare versions in agentic_runs by git_commit: a8d6f22 = v1-new, 5b5616a = 376856
```
Design log + layout: `agent_eval/README.md`.
