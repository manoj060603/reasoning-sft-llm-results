#!/bin/bash
# Separate optional publication; queue only after the confirmed candidate's all-nine evaluation succeeds.
#SBATCH --job-name=sft-publish-validation-anchor
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-publication-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-publication-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
followup="${1:?Expected exact confirmed search output directory}"
qa_receipt="${2:?Expected scoped successful native CPU publication QA receipt}"
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(2)'
"$PWD/.venv/bin/python" -m deadline.publish_anchor_followup \
  --root deadline --followup "$followup" --state deadline/hf/anchor-v2-publication \
  --qa-receipt "$qa_receipt" --expected-owner adversarialbandit
