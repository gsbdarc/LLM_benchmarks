#!/bin/bash
# Self-rescheduling dashboard refresh — the "living" dashboard (tokenviz / RCPedia pattern).
# Each run materializes the central agentic_runs collection to a Parquet cache, rebuilds the
# standalone HTML from it, then schedules the NEXT run ~24h later — so one submission keeps
# the dashboard perpetually fresh with no long-running server.
#
#   Submit ONCE (from the repo root):   sbatch agent_eval/scripts/refresh_dashboard.slurm
#   Watch:                              squeue -u $USER -n dash-refresh ; cat dash-refresh-<jobid>.out
#   Stop the loop:                      scancel the PENDING dash-refresh job
#   Knobs:                              --export=ALL,OUT=/path/to/shared/dash.html,CACHE=outputs/dashboard_cache
#
#SBATCH -J dash-refresh
#SBATCH -p normal
#SBATCH -c 1
#SBATCH -t 0:30:00
#SBATCH -o dash-refresh-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
[ -d agent_eval ] || { echo "ERROR: no agent_eval/ in $PWD — sbatch from the repo root"; exit 1; }
source ~/venv/bin/activate
# Mongo creds for agentic_runs (harmless if the vars are already in the environment).
set -a; source .env 2>/dev/null || true; set +a

OUT="${OUT:-images/agent_dashboard_live.html}"   # Jupyter-viewable standalone HTML (data-laden; gitignored)
CACHE="${CACHE:-outputs/dashboard_cache}"          # Parquet cache dumped from agentic_runs

# Reschedule FIRST so the loop survives even if this build fails (tokenviz reschedules on
# success OR failure). Reschedule by repo path, not $0 (SLURM copies the script to spool).
sbatch --begin=now+24hours agent_eval/scripts/refresh_dashboard.slurm \
  || echo "WARN: could not schedule the next refresh"

# Cache -> render.
python -m analysis.export_runs --out "$CACHE"
python -m analysis.build_dashboard --base-dir "$CACHE" --no-summaries --out "$OUT"
echo "Refreshed $OUT from agentic_runs at $(date -u)"
