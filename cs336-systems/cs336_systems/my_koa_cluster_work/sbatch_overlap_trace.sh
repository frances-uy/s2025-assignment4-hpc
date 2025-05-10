#!/bin/bash
#SBATCH --job-name=overlap_trace
#SBATCH --account=ECE491B
#SBATCH --partition=ece491b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=overlap_trace.out

module load lang/Anaconda3
source activate cs336_systems

python benchmark_ddp_overlap.py --trace_dir trace_overlap
