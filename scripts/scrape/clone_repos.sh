#!/bin/bash
#SBATCH --job-name=test_clone
#SBATCH -p ssd
#SBATCH --nodelist=boston-2-7
#SBATCH --output=logs/test_clone_output.log

# list current files
# check if hostname is correct and only then proceed

if [ "$HOSTNAME" != "boston-2-7" ]; then
    echo "Wrong host $HOSTNAME, exiting"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "($HOSTNAME) Current directory: $PWD"
# ls -l

which python

# copy clone_repo.py and test_repos.txt to /ssd/ssg2 (overwriting if necessary)
cp clone_repos.py /ssd/ssg2
cp test_repos.txt /ssd/ssg2

echo "Changing directory to /ssd/ssg2"
cd /ssd/ssg2
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
which python

# get number of lines in test_repos.txt
num_lines=$(wc -l < test_repos.txt)

# run clone_repo.py
echo "Running clone_repo.py"
python clone_repos.py test_repos.txt 0 $num_lines
