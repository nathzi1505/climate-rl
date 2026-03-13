#!/bin/bash
#SBATCH --job-name=mnist-training
#SBATCH --output=/gws/ssde/j25a/ai4er/users/pn341/climate-rl/scripts/mnist_multicpu/output_%j.log
#SBATCH --error=/gws/ssde/j25a/ai4er/users/pn341/climate-rl/scripts/mnist_multicpu/error_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=orchid
#SBATCH --account=orchid

# Load modules or set environment variables here
conda activate venv

# Run the Python script
srun python /gws/ssde/j25a/ai4er/users/pn341/climate-rl/scripts/mnist_multicpu/main.py

# TO RUN THIS SCRIPT USE:
# sbatch submit.sh
