#!/usr/bin/env bash
#SBATCH --job-name=reasoning-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT}"
: "${EVAL_PYTHON:?Set EVAL_PYTHON}"
: "${MODEL:?Set MODEL to the model ID or full/merged checkpoint path}"
: "${RUN_OUTPUT:?Set a unique RUN_OUTPUT directory}"
cd "$PROJECT_ROOT"
"$EVAL_PYTHON" -m reasoning.evaluate --model "$MODEL" --output "$RUN_OUTPUT" "$@"
