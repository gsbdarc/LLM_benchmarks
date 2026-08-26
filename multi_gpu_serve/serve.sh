#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B as an OpenAI-compatible API, sharded across 2 A40s.
#
# Run this INSIDE a 2-GPU allocation on a yen-gpu node, e.g.:
#   srun -p gpu -C GPU_MODEL:A40 -G 2 -c 16 --mem=100G -t 2:00:00 --pty /bin/bash
#   bash serve.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- config (override inline, e.g. PORT=8001 bash serve.sh) ------------------
MODEL="Qwen/Qwen3.6-35B-A3B"
TP=2                      # tensor-parallel size == number of GPUs
MAX_LEN="${MAX_LEN:-32768}"   # context cap => bounds KV-cache size; raise later
PORT="${PORT:-40777}"         # high port, reachable from the Yen login node
# ~70 GB of weights download here on first run — keep it off home:
export HF_HOME="${HF_HOME:-/scratch/shared/$USER/hf_cache}"

# Single-node multi-GPU only needs intra-node NCCL transport (P2P/shared memory).
# The yen-gpu host exposes a RoCE device (bnxt_re0) whose rdma-core/mlx5 provider
# is older than NCCL 2.28 expects; NCCL's built-in IB transport segfaults while
# probing it. NCCL_NET_PLUGIN=none only disables *external* plugins, so we must
# also turn off the internal IB/RoCE transport. We don't need it for one node.
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1

mkdir -p "$HF_HOME"
#source "$HERE/venv_1/bin/activate"

echo "Serving $MODEL (TP=$TP, max_len=$MAX_LEN) on $(hostname):$PORT"
echo "HF cache: $HF_HOME"

# --enable-auto-tool-choice + --tool-call-parser: WITHOUT these, vLLM emits tool
# calls as plain text and the OpenAI API returns no structured `tool_calls`, so the
# agentic-eval agent (which is entirely tool-call driven) never evaluates or saves.
# `hermes` is the parser for Qwen3 instruct models in this vLLM build (0.25.1);
# there is no bare `qwen3` tool parser (only qwen3_coder/qwen3_xml/mimo) — `qwen3_xml`
# is the fallback if hermes mis-parses.
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len "$MAX_LEN" \
  --host 0.0.0.0 \
  --port "$PORT"
