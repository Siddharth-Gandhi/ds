#!/bin/bash
#SBATCH --job-name=big_scrape
#SBATCH -p ssd
#SBATCH --nodelist=boston-2-10
#SBATCH --output=big_logs/big_scrape__%A_%N_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --time=24:00:00

# list current files
# check if hostname is among the correct hosts and only then proceed
if [ "$HOSTNAME" != "boston-2-7" ] && [ "$HOSTNAME" != "boston-2-8" ] && [ "$HOSTNAME" != "boston-2-9" ] && [ "$HOSTNAME" != "boston-2-10" ]; then
    echo "Wrong host $HOSTNAME, exiting"
    exit 1
fi

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "($HOSTNAME) Current directory: $PWD"

# remove current versions of src and misc
rm /ssd/ssg2/src -r
rm /ssd/ssg2/misc -r

# # copy clone_repo.py and test_repos.txt to /ssd/ssg2 (overwriting if necessary)
cp src/ /ssd/ssg2 -r
cp misc/ /ssd/ssg2 -r

echo "Changing directory to /ssd/ssg2"
cd /ssd/ssg2
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
which python


SCRIPT_PATH=src/scrape_local_v3.py
FILE_PATH=misc/batch2.txt
TOTAL_ITEMS=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)
TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT

# Divide the tasks among nodes
if [ "$HOSTNAME" == "boston-2-7" ]; then
    NODE_START_INDEX=0
    NODE_END_INDEX=$((TOTAL_ITEMS/4 - 1))
elif [ "$HOSTNAME" == "boston-2-8" ]; then
    NODE_START_INDEX=$((TOTAL_ITEMS/4))
    NODE_END_INDEX=$((TOTAL_ITEMS/2 - 1))
elif [ "$HOSTNAME" == "boston-2-9" ]; then
    NODE_START_INDEX=$((TOTAL_ITEMS/2))
    NODE_END_INDEX=$((3*TOTAL_ITEMS/4 - 1))
elif [ "$HOSTNAME" == "boston-2-10" ]; then
    NODE_START_INDEX=$((3*TOTAL_ITEMS/4))
    NODE_END_INDEX=$((TOTAL_ITEMS - 1))
fi

NODE_TOTAL_ITEMS=$((NODE_END_INDEX - NODE_START_INDEX + 1))

echo "Found $TOTAL_ITEMS items in $FILE_PATH to be processed in total"
echo "$HOSTNAME will process lines $((NODE_START_INDEX+1)) to $((NODE_END_INDEX+1)) out of $TOTAL_ITEMS in $FILE_PATH"

# Calculate work distribution for each task within the node
ITEMS_PER_TASK=$(( NODE_TOTAL_ITEMS / $TOTAL_TASKS ))
REMAINING_ITEMS=$(( NODE_TOTAL_ITEMS % $TOTAL_TASKS ))
START_INDEX=$(( NODE_START_INDEX + $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))

# Distribute the remaining items among the tasks
if [ $SLURM_ARRAY_TASK_ID -lt $REMAINING_ITEMS ]; then
    START_INDEX=$(( $START_INDEX + $SLURM_ARRAY_TASK_ID ))
    END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK ))
else
    START_INDEX=$(( $START_INDEX + $REMAINING_ITEMS ))
    END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID, Start Index: $START_INDEX, End Index: $END_INDEX"

echo "Running python $SCRIPT_PATH $FILE_PATH $START_INDEX $END_INDEX in 5 seconds"

sleep 5

srun --wait=0 python -u $SCRIPT_PATH $FILE_PATH $START_INDEX $END_INDEX