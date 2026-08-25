#!/bin/bash
# Submit from the Yen login node:   sbatch serve.sbatch
# Then find the node+port in the log: cat serve-<jobid>.out   (look for "Serving ... on <host>:<port>")
# Query it from the login node:      curl http://<host>:<port>/v1/models
# Stop it:                           scancel <jobid>
#SBATCH -J qwen-vllm
#SBATCH -p gpu
#SBATCH -C GPU_MODEL:A40
#SBATCH -G 2                 # 2 A40s -> tensor-parallel-size 2
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 8:00:00
#SBATCH -o serve-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ltdarc@stanford.edu

cd "$SLURM_SUBMIT_DIR"
# serve.sh activates the venv and exec's `vllm serve` (this job process becomes
# the server; it stays up until the walltime or `scancel`).
bash serve.sh
