#!/bin/bash

# Remove old log files
rm logs/*

# Submit the SLURM job
sbatch sbatch_scrape.sh