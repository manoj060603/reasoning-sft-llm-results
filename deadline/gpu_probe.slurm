#!/bin/bash
#SBATCH --job-name=sft-transfer-probe
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/probe-%A_%a.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/probe-%A_%a.err
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
"$DEADLINE_PYTHON" -m deadline.pipeline guard-start --reserve-hours 20
BASE_MODEL=/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
case "${SLURM_ARRAY_TASK_ID:?SLURM array identity is required}" in
  0) probe_name=probeA; training_fold=0; scoring_fold=1 ;;
  1) probe_name=probeB; training_fold=1; scoring_fold=0 ;;
  *) printf '%s\n' 'Unexpected probe array index.' >&2; exit 2 ;;
esac
"$DEADLINE_PYTHON" -m deadline.transfer_probe train \
  --base "$BASE_MODEL" \
  --cache deadline/artifacts/transfer_cache.jsonl \
  --cache-meta deadline/artifacts/cache_meta.json \
  --train-ids "deadline/artifacts/${probe_name}_ids.json" \
  --fold "$training_fold" --probe-id "$probe_name" \
  --output "deadline/runs/$probe_name" \
  --steps 64 --batch-size 1 --accumulation 8 --lr 5e-5 --lora-rank 8 --seed 42
"$DEADLINE_PYTHON" -m deadline.transfer_probe score \
  --base "$BASE_MODEL" \
  --cache deadline/artifacts/transfer_cache.jsonl \
  --cache-meta deadline/artifacts/cache_meta.json \
  --candidate-ids "deadline/artifacts/fold${scoring_fold}_ids.json" \
  --fold "$scoring_fold" --probe-id "$probe_name" \
  --probe "deadline/runs/$probe_name" \
  --output "deadline/runs/scores_fold${scoring_fold}" \
  --batch-size 4 --chunk-tokens 128
