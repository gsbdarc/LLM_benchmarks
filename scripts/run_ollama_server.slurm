#!/bin/bash
#SBATCH -J ollama-server              # job name
#SBATCH -p gpu                        # partition
#SBATCH -C "GPU_MODEL:A40"            # constraint
#SBATCH -G 1                          # gpus
#SBATCH -n 1                          # tasks
#SBATCH -c 4                          # cpus-per-task
#SBATCH -t 12:00:00                   # time
#SBATCH -o ollama-server-%j.out       # output file

ml apptainer
export SCRATCH_BASE=/scratch/shared/$USER
source /zfs/projects/students/ltdarc-usf-intern-2025/ollama_helper/ollama.sh

# start the server
ollama serve