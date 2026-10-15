#!/usr/bin/env bash
#SBATCH --job-name=reasoning-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the absolute project directory}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the training environment python}"
: "${CONFIG:?Set CONFIG to the generated absolute YAML path}"
cd "$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# The caller activates any site-specific CUDA modules before submission.
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 LLaMA-Factory/src/train.py "$CONFIG"
