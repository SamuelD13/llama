#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=prompt            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=prompt-output.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=1:00:00              # Max execution time
#SBATCH --partition=tesla           # Partition
#

# Activate your Anaconda environment
conda activate llama2

# Run your Python script
cd /home/sam/llama     # CHANGEME
python -m torch.distributed.launch --nproc_per_node=1 example.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6