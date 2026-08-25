# Plan: Serve Qwen3.6-35B-A3B across 2× A40 (venv-based, step-by-step)

## Context

`Qwen/Qwen3.6-35B-A3B` (hybrid Gated-DeltaNet MoE, 35B total / 3B active) won't
fit on one A40 (48 GB). In BF16 the weights are ≈70 GB, so we shard across **two
A40s with tensor parallelism (TP=2)**. This is an **explainability exercise**:
we build it up incrementally, one verifiable step at a time, using a **plain
Python venv — no Apptainer** — so nothing about how vLLM runs is hidden behind a
container.

Decisions locked in:
- Folder: `Vllm_testing/multi_gpu_serve/`.
- venv (`pip install vllm`), not the container `vllm_helper` uses. vLLM's wheel
  bundles its CUDA userspace via torch; it only needs the node's NVIDIA driver.
- Model needs **vLLM ≥ 0.17.0** plus flags `--trust-remote-code` and
  `--reasoning-parser qwen3`.

## Why venv is fine here
vLLM pip wheel ships torch + CUDA userspace libraries; the GPU node only needs a
compatible NVIDIA driver (yen-gpu nodes have it). No system CUDA build required.
Container is a fallback only if pip ever hits a driver/CUDA wall.

## Fit analysis on 2× A40 (96 GB total)
- BF16 weights ≈70 GB → ~35 GB/GPU, leaving ~13 GB/GPU.
- Hybrid arch: only ~10 layers do real attention (2 KV heads); DeltaNet layers
  use fixed-size recurrent state (no KV growth) → KV cache is small → headroom OK.
- Quant options don't help on A40 (Ampere): FP8 = H100/H200, NVFP4 = Blackwell.
  So **BF16**. Start `--max-model-len 32768` and raise once it fits.

## Steps (each: what → why → verify)
1. **Create `multi_gpu_serve/` + venv** (`/usr/bin/python3 -m venv venv`, Python
   3.10). Verify: `venv/bin/python --version`.  ✅ done
2. **Install vLLM** (`pip install "vllm>=0.17"`). Verify:
   `python -c "import vllm; print(vllm.__version__)"` ≥ 0.17.
3. **Interactive 2-GPU allocation:**
   `srun -p gpu -C GPU_MODEL:A40 -G 2 -c 16 --mem=100G -t 2:00:00 --pty /bin/bash`.
   Verify: `nvidia-smi` shows 2 A40s.
4. **Launch the server** on the GPU node:
   `vllm serve Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 2 --trust-remote-code
   --reasoning-parser qwen3 --max-model-len 32768 --host 0.0.0.0 --port <PORT>`.
   Explain every flag. Verify: log shows "Uvicorn running…" and both GPUs load.
5. **Query from the login node** (`curl`/openai client to
   `http://<gpu-node>:<PORT>/v1/chat/completions`). Verify: coherent reply.
6. **Freeze into `serve.sbatch`** once the interactive path works — a background
   batch job modeled on the Yen template (`-p gpu -G 2 -C GPU_MODEL:A40`).

## Files that will land in `multi_gpu_serve/`
- `PLAN.md` — this plan.
- `requirements.txt` — `vllm>=0.17`.
- `README.md` — the step-by-step walkthrough with explanations.
- `serve.sh` — the documented launch command (built at step 4).
- `serve.sbatch` — background batch version (step 6).
- `test_client.py` — minimal OpenAI-style smoke test (step 5).
- `.gitignore` — `venv/`, `__pycache__/`, `*.out`.

## Port / access
Pick a high port (e.g. 8000 or a value in 32768–60999); bind `0.0.0.0`; reach it
from the login node via the GPU node's hostname (Yen allows intra-cluster high
ports — `vllm_helper` relies on this). SSH tunnel is the fallback.

## Model download
Set `HF_HOME` to scratch/project space (not home — quota) so the ~70 GB download
doesn't fill `~`. e.g. `export HF_HOME=/scratch/shared/$USER/hf_cache`.

## Verification (end-to-end)
- `nvidia-smi` on the node shows ~35 GB used on **each** GPU (TP actually sharding).
- `/v1/models` lists `Qwen/Qwen3.6-35B-A3B`; a chat request returns a reply.

## Risks / open items
- pip vLLM vs node driver (fallback: container).
- TP NCCL over PCIe on A40 (no NVLink) — works, just slower; venv avoids the
  container `/dev/shm` gotcha since host `/dev/shm` is normal-sized.
- If BF16 won't fit: lower `--max-model-len`, then consider `-G 4` / TP=4.
