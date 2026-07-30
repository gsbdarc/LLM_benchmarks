#!/bin/bash
# run_eval_array.slurm — evaluate a sampled eval_mapping as a SLURM array.
#
# One array task = one row of inputs/eval_mapping_sample.csv (index = $SLURM_ARRAY_TASK_ID)
# = one output judged by the playground agent. Each task is its own process (its own
# connection to the playground), which is what the API tolerates.
#
# NOTE: run_eval_batch.slurm is the simpler path (self-contained: one job starts its own
# server + fans out). Use this array variant only if you want a persistent shared server.
#
# BEFORE submitting (all from the REPO ROOT):
#   1. Get a shell ON A COMPUTE NODE (yen10-16/20 — NOT the login host yen4/yen5)
#      and start the MCP server there:
#        srun -p normal -c 2 -t 3:00:00 --pty bash
#        hostname -s                                # <-- the compute node, e.g. yen12
#        agent_eval/scripts/run_metric_mcp.sh               # <-- note the port it prints
#   2. Build the sample (from a login shell is fine):
#        python -m agent_eval.registry.create_eval_mapping --sample 100
#   3. Submit this array PINNED TO THAT COMPUTE NODE, throttled with %N so you don't
#      hit the playground with all tasks at once (the node has 100s of free cores):
#        sbatch -w <compute-node> --array=0-99%8 \
#          --export=ALL,MCP_URL="http://127.0.0.1:<port>/mcp" agent_eval/scripts/run_eval_array.slurm
#   4. Rerun failures by resubmitting just their indices, e.g. --array=3,17,42%8
#
#SBATCH -J eval-array
#SBATCH -p normal
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0:30:00
#SBATCH -o eval-array-%A_%a.out

set -euo pipefail

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

: "${MCP_URL:?set MCP_URL to the running server, e.g. http://127.0.0.1:PORT/mcp}"

# Run from the repo root (where the `agent_eval` package + inputs/ live). Under SLURM the
# script runs from a spool copy, so $0 is useless — use the submit dir instead.
cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the repo root}"
[ -d agent_eval ] || { echo "ERROR: no agent_eval/ package in $PWD — run sbatch from the repo root"; exit 1; }

# --no-gpu-metrics: the playground has no /metrics endpoint. Row mode forces
# --concurrency 1 (parallelism is across array tasks, not in-process).
python -m agent_eval \
    --mcp-url "$MCP_URL" \
    --eval-mapping inputs/eval_mapping_sample.csv \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --no-gpu-metrics
