#!/bin/bash
#SBATCH --job-name=ssg2_scrape_git
#SBATCH --output=my_job_output_%j.log  # Standard output log file with job ID
#SBATCH --error=my_job_error_%j.log    # Standard error log file with job ID
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G                       # Memory per node (e.g., 4GB)
#SBATCH --time=2:00:00
#SBATCH --array=0-19

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate git_scrape

# Rest of the script
# ...

python3 scrape_local.py $START_INDEX $END_INDEX