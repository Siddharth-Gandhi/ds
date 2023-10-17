#!/bin/bash
#SBATCH --job-name=2_7_scrape
#SBATCH -p ssd
#SBATCH --nodelist=boston-2-7
#SBATCH --output=logs/2_7_scrape_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --array=0-0
#SBATCH --nodelist=boston-2-7
#SBATCH --time=12:00:00

# list current files
# check if hostname is correct and only then proceed

if [ "$HOSTNAME" != "boston-2-7" ]; then
    echo "Wrong host $HOSTNAME, exiting"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "($HOSTNAME) Current directory: $PWD"

pwd
# ls -l

which python

# copy clone_repo.py and test_repos.txt to /ssd/ssg2 (overwriting if necessary)
# cp clone_repos.py /ssd/ssg2

# copy dependencies to /ssd/ssg2 (overwriting if necessary) for scrape_local.py
# cp misc/test.txt /ssd/ramdisk
cp src/scrape_local_v2.py /ssd/ramdisk
# cp misc/code_extensions.json /ssd/ramdisk
cp misc/ /ssd/ramdisk -r

echo "Changing directory to /ssd/ramdisk"
cd /ssd/ramdisk
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
which python

# get number of lines in test_repos.txt
# num_lines=$(wc -l < test_repos.txt)

# run scrape_local.py
echo "Running scrape_local.py"

# Check resources allocated to this job (CPU, memory, disk space)
# scontrol show job $SLURM_JOB_ID
# sleep for 10 seconds
FILE_PATH=test.txt
TOTAL_ITEMS=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)


TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT

echo "Found $TOTAL_ITEMS items in $FILE_PATH to be processed by $TOTAL_TASKS tasks"

# handle the case where there are more tasks than items
if [ $TOTAL_TASKS -gt $TOTAL_ITEMS ]; then
    echo "WARNING: More tasks than items, reducing number of tasks to $TOTAL_ITEMS"
    TOTAL_TASKS=$TOTAL_ITEMS
fi

# Indexing the repo list and assigning each task it's index
ITEMS_PER_TASK=$(( $TOTAL_ITEMS / $TOTAL_TASKS ))
REMAINING_ITEMS=$(( $TOTAL_ITEMS % $TOTAL_TASKS ))
START_INDEX=$(( $SLURM_ARRAY_TASK_ID * $ITEMS_PER_TASK ))
END_INDEX=$(( $START_INDEX + $ITEMS_PER_TASK - 1 ))

echo "Task ID: $SLURM_ARRAY_TASK_ID, Start Index: $START_INDEX, End Index: $END_INDEX"
# sleep 5


# python clone_repos.py test_repos.txt 0 $num_lines (chunk size is 1000 by default)
srun --wait=0 python -u scrape_local_v2.py $FILE_PATH  $START_INDEX $END_INDEX