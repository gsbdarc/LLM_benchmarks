#!/bin/bash
#SBATCH -J main                       # job name
#SBATCH -p gsb,normal                 # partition
#SBATCH -n 1                          # tasks
#SBATCH -c 1                          # cpus-per-task
#SBATCH -t 1:30:00                    # time
#SBATCH -o out/main-%a.out            # output file
#SBATCH --array=1-1000                # array_job
#SBATCH --mail-type=ALL               # email
#SBATCH --mail-user=ltdarc@stanford.edu

# Activate venv
source /home/groups/fmcnich/darc/ltdarc/LLM_benchmarks/venv/bin/activate

# Run python script
python main.py ${SLURM_ARRAY_TASK_ID}
python main.py $((SLURM_ARRAY_TASK_ID+1000))
python main.py $((SLURM_ARRAY_TASK_ID+2000))
