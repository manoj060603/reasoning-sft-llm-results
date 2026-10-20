#!/bin/bash
#SBATCH --job-name=sft-transfer-train
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/train-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/train-%j.err
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
"$DEADLINE_PYTHON" -m deadline.pipeline guard-start --reserve-hours 12
BASE_MODEL=/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
role="${1:?Usage: gpu_train.slurm random|advanced}"
case "$role" in
  random|advanced) ;;
  *) printf '%s\n' 'Training role must be random or advanced.' >&2; exit 2 ;;
esac
"$DEADLINE_PYTHON" -m deadline.train_full \
  --base "$BASE_MODEL" \
  --cache "deadline/artifacts/training/$role/training.jsonl" \
  --cache-meta "deadline/artifacts/training/$role/cache_meta.json" \
  --output "deadline/runs/$role" --steps 1875 --accumulation 8
if [[ ! -f "deadline/runs/$role/complete.json" ]]; then
  printf '%s\n' 'Training returned without its completion receipt.' >&2
  exit 1
fi
"$DEADLINE_PYTHON" -m deadline.pipeline submit-eval --role "$role"
