#!/usr/bin/env bash
set -euo pipefail

LOWERPORT=32768
UPPERPORT=60999

find_available_port() {
  local p
  while :; do
    p=$(shuf -i "${LOWERPORT}-${UPPERPORT}" -n1)
    if ! ss -tuln | grep -q ":${p} "; then
      echo "$p"
      return
    fi
  done
}

PORT=$(find_available_port)
HOST=$(hostname -s)

echo "Starting Metric-Eval MCP server"
echo "Host: $HOST"
echo "Port: $PORT"
echo "URL:  http://${HOST}:${PORT}/mcp"

# For local-only testing, do THIS:
export MCP_HOST="127.0.0.1"
export MCP_PORT="$PORT"

# Run from the repo root so the `agent_eval` package is importable, then launch
# the server as a module (server.py uses package-relative imports).
cd "$(dirname "$0")/.."
python -m agent_eval.server
