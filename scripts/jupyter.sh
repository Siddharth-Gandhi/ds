#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name="jupyter"
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/jupyter.log
#SBATCH --nodelist=boston-1-7

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

# cat /etc/hosts
node=$(hostname -s)
port=8888

jupyter-lab --port=${port} --ip=${node} --no-browser