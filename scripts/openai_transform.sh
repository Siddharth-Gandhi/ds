#!/bin/bash
#SBATCH --job-name=oai
#SBATCH --output=logs/oai/transform_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# source conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

# list current files
echo "Current directory: $PWD"
ls -l

python -u src/openai_transform.py