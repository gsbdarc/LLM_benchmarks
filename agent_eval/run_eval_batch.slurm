#!/bin/bash
# run_eval_batch.slurm — self-contained agentic-eval batch.
#
# ONE job, ONE node: it starts the MCP server locally on the allocated compute
# node, evaluates every row of the sample (MAXPAR at a time = that many concurrent
# playground calls, each its own process/connection), then stops the server. No
# manual server, no -w, no port juggling — SLURM picks the node; the server lives
# on 127.0.0.1 for the job's lifetime.
#
#   # from the REPO ROOT:
#   python -m agent_eval.create_eval_mapping --sample 100   # once: builds inputs/eval_mapping_sample.csv
#   sbatch agent_eval/run_eval_batch.slurm                  # then this
#   # tune:  sbatch --export=ALL,MAXPAR=12,WEAVE=0 agent_eval/run_eval_batch.slurm
#
#SBATCH -J eval-batch
#SBATCH -p normal
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -o eval-batch-%j.out

set -uo pipefail   # NOT -e: one failed row must not abort the batch / server teardown

MAXPAR="${MAXPAR:-8}"                                   # concurrent evals (playground throttle)
SAMPLE="${SAMPLE:-inputs/eval_mapping_sample.csv}"     # the sampled work list (repo-root relative)
WEAVE="${WEAVE:-1}"                                     # 1 = W&B/Weave tracing on, 0 = off

cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
[ -d agent_eval ] || { echo "ERROR: no agent_eval/ package in $PWD — run sbatch from the repo root"; exit 1; }
[ -f "$SAMPLE" ] || { echo "ERROR: $SAMPLE missing — run create_eval_mapping.py --sample N first"; exit 1; }
source ~/venv/bin/activate

# 1. start the MCP server on a free local port
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
export MCP_HOST=127.0.0.1 MCP_PORT="$PORT"
python -m agent_eval.server >"mcp-server-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
MCP_URL="http://127.0.0.1:${PORT}/mcp"

# 2. wait until it accepts connections (up to ~60s)
for _ in $(seq 1 60); do
  python -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',$PORT))==0 else 1)" && break
  sleep 1
done

WEAVE_FLAG=""; [ "$WEAVE" = "0" ] && WEAVE_FLAG="--no-weave"
N=$(($(wc -l < "$SAMPLE") - 1))     # data rows = lines minus header
mkdir -p eval-logs
echo "MCP=$MCP_URL | evals=$N | MAXPAR=$MAXPAR | weave=$WEAVE"

# 3. fan out: MAXPAR concurrent workers, one per sample row, each logged separately
seq 0 $((N - 1)) | xargs -P "$MAXPAR" -I{} bash -c '
  python -m agent_eval --mcp-url "$1" --eval-mapping "$2" --row "$3" --no-gpu-metrics $4 \
      >"eval-logs/row-$3.out" 2>&1
' _ "$MCP_URL" "$SAMPLE" {} "$WEAVE_FLAG"

# 4. summarize
ok=$(grep -l "Done:" eval-logs/row-*.out 2>/dev/null | wc -l)
echo "Batch complete: ${ok}/${N} rows produced a Done line. Per-row logs in eval-logs/."
echo "Rows with NO Done line (rerun these):"
grep -L "Done:" eval-logs/row-*.out 2>/dev/null || echo "  (none)"
