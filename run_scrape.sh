#!/bin/bash
#SBATCH --job-name=ssg2_scrape_git
#SBATCH --output=logs/my_job_output_%j.log  # Standard output log file with job ID
#SBATCH --error=logs/my_job_error_%j.log    # Standard error log file with job ID
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                       # Memory per node (e.g., 4GB)
#SBATCH --time=2:00:00
#SBATCH --array=0-19

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

TOTAL_ITEMS=200
TOTAL_TASKS=8

# Rest of the script
ITEMS_PER_TASK=$(( $TOTAL_ITEMS / $TOTAL_TASKS ))
REMAINING_ITEMS=$(( $TOTAL_ITEMS % $TOTAL_TASKS ))
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))
END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))

# Adjust for remainder
if [ $SLURM_ARRAY_TASK_ID -eq $(( $TOTAL_TASKS - 1 )) ]; then
    END_INDEX=$(( $END_INDEX + $REMAINING_ITEMS ))
fi
# ...

python3 scrape_local.py $START_INDEX $END_INDEX