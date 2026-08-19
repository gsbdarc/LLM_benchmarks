#!/bin/bash
#SBATCH -J main                       # job name
#SBATCH -p gsb,normal                 # partition
#SBATCH -n 1                          # tasks
#SBATCH -c 1                          # cpus-per-task
#SBATCH -t 1:30:00                    # time
#SBATCH -o out/main-%a.out            # output file
#SBATCH --array=1-840                 # array_job
#SBATCH --mail-type=ALL               # email
#SBATCH --mail-user=youremail@stanford.edu # change as needed

# remove all prior venvs
module purge

# load python + the repo-local .venv (submit from the scripts/ dir → repo root is one level up).
# Sherlock is a separate filesystem from the Yens, so build its own .venv there — same convention.
module load python/3.12
source ../.venv/bin/activate

# Run python script
#for offset in $(seq 0 1000 2000)
#do
    #python 5_main.py $((SLURM_ARRAY_TASK_ID + offset))
#done

python 5_main.py $((SLURM_ARRAY_TASK_ID + 2939))