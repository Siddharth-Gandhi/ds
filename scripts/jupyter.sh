#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/jupyter.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888