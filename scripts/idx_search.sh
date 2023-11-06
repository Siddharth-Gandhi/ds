#!/bin/bash
#SBATCH --job-name=idx_search
#SBATCH -p ssd
#SBATCH --nodelist=boston-2-7
#SBATCH --output=big_logs/idx_search__%A_%N_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0
#SBATCH --time=24:00:00

# list current files
if [ "$HOSTNAME" != "boston-2-7" ] && [ "$HOSTNAME" != "boston-2-8" ] && [ "$HOSTNAME" != "boston-2-9" ] && [ "$HOSTNAME" != "boston-2-10" ]; then
    echo "Wrong host $HOSTNAME, exiting"
    exit 1
fi

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
echo "($HOSTNAME) Current directory: $PWD"

# # copy clone_repo.py and test_repos.txt to /ssd/ssg2 (overwriting if necessary)
cp src/ /ssd/ssg2 -r
cp misc/ /ssd/ssg2 -r
cp scripts/ /ssd/ssg2 -r
cp logging.conf /ssd/ssg2

echo "Changing directory to /ssd/ssg2"
cd /ssd/ssg2
echo "($HOSTNAME) Current directory: $PWD"
echo "Current files: in $PWD"
ls -l
which python


# make jsonl files for pyserini
# python src/convert_to_jsonl.py data/ --use_tokenizer --content_option commit

# build index
# bash scripts/build_index.sh data commit true 1

# evaluate and store
bash scripts/eval_repos.sh data commit true