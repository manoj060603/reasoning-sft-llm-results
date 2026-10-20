#!/bin/bash
#SBATCH --job-name=sft-recovery-source-dev
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-dev-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-dev-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -d /apps/common/software/CUDA/12.8.0 ]]; then
  export CUDA_HOME=/apps/common/software/CUDA/12.8.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
DEADLINE_PYTHON="$PWD/.venv/bin/python"
recovery_root="${1:?Expected recovery output directory}"
qa_receipt="${2:?Expected successful native CPU-QA receipt}"
"$DEADLINE_PYTHON" -c 'from deadline.pipeline import guard_start; guard_start(3.75)'
"$DEADLINE_PYTHON" -m deadline.operations.recovery_preflight --verify "$qa_receipt"
if [[ ! -f "$recovery_root/development/prepared.json" ]]; then
  printf '%s\n' 'CPU preparation must seal the source-heldout screen before GPU admission.' >&2
  exit 1
fi
for role in base advanced recovery; do
  "$DEADLINE_PYTHON" -m deadline.operations.recovery_preflight --verify "$qa_receipt"
  "$DEADLINE_PYTHON" -m deadline.recovery.screen run \
    --output "$recovery_root/development" --role "$role"
done
CUDA_VISIBLE_DEVICES='' "$DEADLINE_PYTHON" -m deadline.recovery.screen summarize \
  --output "$recovery_root/development"
