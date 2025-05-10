#!/bin/bash
#SBATCH --job-name=ddp_overlap_bench
#SBATCH --account=ECE491B
#SBATCH --partition=ece491b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=ddp_overlap_benchmark.out

# Load Anaconda
module load lang/Anaconda3
source activate cs336_systems

# Set PyTorch distributed env variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0  # Will be set automatically by mp.spawn

# Run the benchmark
python benchmark_ddp_overlap.py
