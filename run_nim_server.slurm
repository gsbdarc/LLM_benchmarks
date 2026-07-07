#!/bin/bash
#SBATCH -J nim-server
#SBATCH -p gpu
#SBATCH -C "GPU_MODEL:H200"
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 25
#SBATCH --mem=100G
#SBATCH -t 5:00:00
#SBATCH -o nim-server-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@stanford.edu

# 1. Define Paths & Credentials
export NGC_API_KEY=$(cat /home/users/$USER/.secrets/ngc_api_key)

export NIM_ROOT="/scratch/shared/$USER/nim"
export SINGULARITY_CACHEDIR="$NIM_ROOT/singularity_cache"
export SINGULARITY_TMPDIR="$NIM_ROOT/tmp"
export SINGULARITY_CONFIGDIR="$NIM_ROOT/singularity_config"
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD="$NGC_API_KEY"

ml singularity

# 2. Setup directories
mkdir -p "$NIM_ROOT"/{cache,work/{nginx,configs},tmp,singularity_cache,singularity_config,triton}
chmod -R 700 "$NIM_ROOT"

# 3. Download SIF if not found
if [ ! -f "$NIM_ROOT/gemma-4-31b-it.sif" ]; then
    echo "SIF file not found. Downloading..."
    singularity pull "$NIM_ROOT/gemma-4-31b-it.sif" docker://nvcr.io/nim/google/gemma-4-31b-it:latest
else
    echo "SIF file found. Skipping download."
fi

# 3b. Background GPU sampler (observability, plan §5).
# Writes a timestamped CSV of SM utilization + GPU RAM for the job's lifetime;
# the analysis layer joins it to agent runs by wall-clock window. SM% and GPU RAM
# cannot be read from the client (the agent runs on a different node), so we
# sample them here on the GPU box. The trap stops the sampler when the job ends.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 1 > "$NIM_ROOT/gpu-usage-${SLURM_JOB_ID}.csv" &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT
echo "GPU sampler PID $NVSMI_PID -> $NIM_ROOT/gpu-usage-${SLURM_JOB_ID}.csv"

# GPU model name for the run rows (plan §3). The eval client runs on a DIFFERENT
# host and cannot read nvidia-smi, so we surface the name here for the client to
# pass explicitly as `--gpu-type` (or export GPU_TYPE before `python -m eval`).
export GPU_TYPE="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
echo "GPU_TYPE=$GPU_TYPE   # pass to the eval client:  --gpu-type \"$GPU_TYPE\""

# 4. Launch NIM
singularity run --nv --cleanenv \
    --writable-tmpfs \
    --bind "$NIM_ROOT/cache:/opt/nim/.cache" \
    --bind "$NIM_ROOT/work/nginx:/opt/nim/nginx" \
    --bind "$NIM_ROOT/work/configs:/opt/nim/generated_configs" \
    --bind "$NIM_ROOT/triton:/tmp/triton_cache" \
    --env TRITON_CACHE_DIR="/tmp/triton_cache" \
    --env NGC_API_KEY=$NGC_API_KEY \
    --env NIM_TENSOR_PARALLEL_SIZE=1 \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    "$NIM_ROOT/gemma-4-31b-it.sif"