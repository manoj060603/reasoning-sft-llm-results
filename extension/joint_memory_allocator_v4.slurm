#!/bin/bash
#SBATCH --job-name=sft-extension-joint-memory-allocator-v4
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
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-joint-allocator-v4-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-joint-allocator-v4-%j.err
set -euo pipefail
umask 077
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
[[ "$#" -eq 1 ]]
qa_run="$1"
[[ ! -v PYTORCH_NO_CUDA_MEMORY_CACHING && ! -v PYTORCH_ALLOC_CONF && ! -v PYTORCH_HIP_ALLOC_CONF ]]
[[ ! -v PYTORCH_CUDA_ALLOC_CONF || "$PYTORCH_CUDA_ALLOC_CONF" == 'backend:native,expandable_segments:True' ]]
export PYTORCH_CUDA_ALLOC_CONF='backend:native,expandable_segments:True'
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PWD/.venv/bin/python" -c 'from extension.cpu_feasibility import deadline_guard; deadline_guard(180000)'
admission_rel="extension/operations/joint-memory-allocator-v4-job-${SLURM_JOB_ID:?ExpectedB200}.json"
for attempt in {1..30}; do
  [[ -f "$admission_rel" ]] && break
  sleep 1
done
[[ -f "$admission_rel" ]]
"$PWD/.venv/bin/python" -m extension.joint_memory_allocator_v4 --root "$PWD" \
  --plan extension/joint_memory_admission_plan.json \
  --output "$PWD/extension/runs/joint-memory-allocator-v4-${SLURM_JOB_ID}" \
  --qa-receipt "$qa_run/native_qa.json" --admission-receipt "$admission_rel"
