#!/bin/bash
#SBATCH --job-name=sft-transfer-eval-resume
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/eval-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/eval-%j.err
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
"$DEADLINE_PYTHON" -m deadline.pipeline guard-start --reserve-hours 19
BASE_MODEL=/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
role="${1:?Usage: gpu_eval_extended.slurm random|advanced}"
case "$role" in
  random|advanced)
    if [[ ! -f "deadline/runs/$role/complete.json" ]]; then
      printf '%s\n' 'Final model has no training completion receipt.' >&2
      exit 1
    fi
    EVALUATION_MODEL="$PWD/deadline/runs/$role/model"
    ;;
  *) printf '%s\n' 'Continuation role must be random or advanced.' >&2; exit 2 ;;
esac
"$DEADLINE_PYTHON" -m reasoning.evaluate \
  --model "$EVALUATION_MODEL" \
  --data-dir deadline/artifacts/benchmarks \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "deadline/results/$role" \
  --seed 0 --batch-size 1024 --tensor-parallel-size 1 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85 \
  --max-new-tokens 32768 --max-model-len 32768 --record-hardware
"$DEADLINE_PYTHON" -m deadline.pipeline report
