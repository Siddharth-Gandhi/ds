#!/bin/bash
#SBATCH --job-name=big_clone
#SBATCH -p ssd
#SBATCH --nodelist=boston-2-10
#SBATCH --output=big_logs/hi-big_clone_b2r_%N.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

# list current files
# check if hostname is correct and only then proceed

if [ "$HOSTNAME" != "boston-2-7" ] && [ "$HOSTNAME" != "boston-2-8" ] && [ "$HOSTNAME" != "boston-2-9" ] && [ "$HOSTNAME" != "boston-2-10" ]; then
    echo "Wrong host $HOSTNAME, exiting"
    exit 1
fi

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "($HOSTNAME) Current directory: $PWD"
# ls -l

# remove current versions of src and misc
rm /ssd/ssg2/src/ -r
rm /ssd/ssg2/misc/ -r

# # copy clone_repo.py and test_repos.txt to /ssd/ssg2 (overwriting if necessary)
cp src/ /ssd/ssg2 -r
cp misc/ /ssd/ssg2 -r

echo "Changing directory to /ssd/ssg2"
cd /ssd/ssg2
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
which python

FILE_PATH=misc/batch2.txt

# get number of lines in test_repos.txt
num_lines=$(wc -l < $FILE_PATH)

echo "Running clone_repos.py"

# Divide number of lines by 4, if on boston-2-7, first 1/4, if on boston-2-8, second 1/4, etc.
TOTAL_ITEMS=$(grep -v "^[[:space:]]*$" $FILE_PATH | wc -l)

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

echo "$HOSTNAME will process lines $((NODE_START_INDEX+1)) to $((NODE_END_INDEX+1)) out of $TOTAL_ITEMS in $FILE_PATH"
# # run clone_repo.py
# echo "Running clone_repo.py"
python src/clone_repos.py $FILE_PATH $NODE_START_INDEX $NODE_END_INDEX