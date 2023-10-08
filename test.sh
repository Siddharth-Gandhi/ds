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

cd /ssd/ssg2
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
