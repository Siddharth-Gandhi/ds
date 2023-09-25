#!/bin/bash
#SBATCH --job-name=ssg2_scrape_git
#SBATCH --output=logs/my_job_output_%A_%a.log
#SBATCH --error=logs/my_job_error_%A_%a.log
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=8                     # Number of CPUs
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                       # Memory per node (e.g., 4GB)
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --exclude=boston-2-34
# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

# clear logs
# rm logs/*
# Conditionally clear logs only for the first task
# if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
#     rm logs/*
# fi
FILE_PATH=test_repos.txt
n=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)
TOTAL_ITEMS=$n

TOTAL_TASKS=8

# Rest of the script
ITEMS_PER_TASK=$(( $TOTAL_ITEMS / $TOTAL_TASKS ))
REMAINING_ITEMS=$(( $TOTAL_ITEMS % $TOTAL_TASKS ))
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))
END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))

# echo "Task ID: $SLURM_ARRAY_TASK_ID, Start Index: $START_INDEX, End Index: $END_INDEX"

# Adjust for remainder
if [ $SLURM_ARRAY_TASK_ID -eq $(( $TOTAL_TASKS - 1 )) ]; then
    END_INDEX=$(( $END_INDEX + $REMAINING_ITEMS ))
fi
# ...

srun -N1 python3 scrape_local.py $FILE_PATH $START_INDEX $END_INDEX