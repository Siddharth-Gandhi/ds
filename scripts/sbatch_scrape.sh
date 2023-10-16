#!/bin/bash
#SBATCH --job-name=git_build_scrape
#SBATCH --output=logs_build/git_scrape_output_%A_%a.log
#SBATCH --error=logs_build/git_scrape_error_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

FILE_PATH=test_repos.txt
TOTAL_ITEMS=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)

TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT

# Indexing the repo list and assigning each task it's index
ITEMS_PER_TASK=$(( $TOTAL_ITEMS / $TOTAL_TASKS ))
REMAINING_ITEMS=$(( $TOTAL_ITEMS % $TOTAL_TASKS ))
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))
END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))

echo "Task ID: $SLURM_ARRAY_TASK_ID, Start Index: $START_INDEX, End Index: $END_INDEX"
echo $FILE_PATH $START_INDEX $END_INDEX | sort

# Adjust for remainder
if [ $SLURM_ARRAY_TASK_ID -eq $(( $TOTAL_TASKS - 1 )) ]; then
    END_INDEX=$(( $END_INDEX + $REMAINING_ITEMS ))
fi

srun --wait=0 python -u scrape_local.py $FILE_PATH $START_INDEX $END_INDEX