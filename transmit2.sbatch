#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=prompt            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=transmit-output2.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=1:00:00              # Max execution time
#SBATCH --partition=tesla           # Partition
#

# Activate your Anaconda environment
conda activate transform

# Run your Python script
cd /home/sam/llama
python mistral.py