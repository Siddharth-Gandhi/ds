#!/bin/bash
#SBATCH --job-name=git_clone
#SBATCH --output=logs/git_clone_output_%A_%a.log
#SBATCH --error=logs/git_clone_error_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-63

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

FILE_PATH=top_repos.txt
TOTAL_ITEMS=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)
TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT
ITEMS_PER_TASK=$(( $TOTAL_ITEMS / $TOTAL_TASKS ))
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))
END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))

# Adjust for any remaining items
if [ $SLURM_ARRAY_TASK_ID -eq $(( $TOTAL_TASKS - 1 )) ]; then
    REMAINING_ITEMS=$(( $TOTAL_ITEMS % $TOTAL_TASKS ))
    END_INDEX=$(( $END_INDEX + $REMAINING_ITEMS ))
fi

srun --wait=0 python -u clone_repos.py $FILE_PATH $START_INDEX $END_INDEX