#!/bin/bash
#SBATCH --job-name=sft-extension-source-readiness-v2-calibration-retry
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-source-readiness-v2-calibration-retry-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-source-readiness-v2-calibration-retry-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_output="$PWD/extension/runs/source-readiness-v2-retry-${SLURM_JOB_ID:?Expected CPU allocation}"
"$PWD/.venv/bin/python" -m extension.source_readiness_v2_retry_qa --root "$PWD" --output "$run_output"
"$PWD/.venv/bin/python" -m extension.source_readiness_v2_retry --root "$PWD" \
  --plan extension/source_readiness_v2_retry_plan.json --output "$run_output" --qa-receipt "$run_output/unit_qa.json"
