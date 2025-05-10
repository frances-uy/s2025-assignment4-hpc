#!/bin/bash
#SBATCH --job-name=naive_ddp_bench
#SBATCH --account=ECE491B
#SBATCH --partition=ece491b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=naive_ddp_benchmark.out

module load lang/Anaconda3
source activate cs336_systems

python benchmark_naive_ddp.py
