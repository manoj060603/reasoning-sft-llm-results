#!/bin/bash
# Optional recovery publication only; submit after primary publisher succeeds.
#SBATCH --job-name=sft-publish-optional-recovery
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/optional-publication-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/optional-publication-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(2)'
"$PWD/.venv/bin/python" -m deadline.publish_optional_model \
  --root deadline \
  --recovery deadline/recovery-runs/base-anchor-75-25-20261020 \
  --state deadline/hf/optional-recovery-publication \
  --primary-receipt deadline/hf/publication/publication_complete.json \
  --expected-owner adversarialbandit \
  --timeout-seconds 5400
