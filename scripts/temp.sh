#!/bin/bash
#SBATCH --job-name=m_c_df
#SBATCH --output=logs/output_%A.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate the conda environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

echo "On host $(hostname)"

python -u src/get_multi_code_df.py


echo "Job completed"


