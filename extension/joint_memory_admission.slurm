#!/bin/bash
#SBATCH --job-name=sft-extension-joint-memory-admission
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-joint-memory-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-joint-memory-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
[[ "$#" -eq 1 ]]
qa_run="$1"
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PWD/.venv/bin/python" -c 'from extension.cpu_feasibility import deadline_guard; deadline_guard(180000)'
admission_rel="extension/operations/joint-memory-admission-job-${SLURM_JOB_ID:?Expected B200 allocation}.json"
for attempt in {1..30}; do
  [[ -f "$admission_rel" ]] && break
  sleep 1
done
[[ -f "$admission_rel" ]]
"$PWD/.venv/bin/python" -m extension.joint_memory_admission --root "$PWD" \
  --plan extension/joint_memory_admission_plan.json --output "$PWD/extension/runs/joint-memory-admission-${SLURM_JOB_ID}" \
  --qa-receipt "$qa_run/native_qa.json" --admission-receipt "$admission_rel"
