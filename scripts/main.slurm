#!/bin/bash
#SBATCH -J main                       # job name
#SBATCH -p normal                     # partition
#SBATCH -n 1                          # tasks
#SBATCH -c 1                          # cpus-per-task
#SBATCH -t 1:00:00                    # time
#SBATCH -o main-%j.out                # output file
#SBATCH --array=0-500              # array_job


# Activate venv
source /zfs/projects/students/ltdarc-usf-intern-2025/venv/bin/activate

# Run python script
python main.py $(sed -n "${SLURM_ARRAY_TASK_ID}p" config_2.txt)