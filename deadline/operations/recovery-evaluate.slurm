#!/bin/bash
# Separate optional post-hoc evaluation. Admission requires a verified dev gate,
# terminal resource reclamation, six GPU hours and three closeout hours.
#SBATCH --job-name=sft-recovery-benchmarks
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
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-eval-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-eval-%j.err
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
"$DEADLINE_PYTHON" -c 'from deadline.pipeline import guard_start; guard_start(9)'
"$DEADLINE_PYTHON" -m deadline.operations.recovery_preflight --verify "$qa_receipt"
CUDA_VISIBLE_DEVICES='' "$DEADLINE_PYTHON" - "$recovery_root" <<'PY'
from pathlib import Path
import sys
from deadline.recovery.common import digest, require
from deadline.recovery.screen import model_inputs, validate_summary
root = Path(sys.argv[1]).resolve()
summary = validate_summary(root / 'development')
require(summary['gate']['accepted'] is True, 'Fixed source-heldout gate did not pass')
_, models = model_inputs(root / 'complete.json', verify_roles=('recovery',))
require(Path(models['recovery']['path']) == root / 'model', 'Recovery model path differs')
require(digest('reasoning/evaluate.py') ==
        '55181149a1010cd60a5979e859e2fcd6ee5d1a3abb8d4fbf33cee4b18586c316',
        'Frozen primary benchmark evaluator changed')
PY
"$DEADLINE_PYTHON" -m reasoning.evaluate \
  --model "$recovery_root/model" \
  --data-dir deadline/artifacts/benchmarks \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$recovery_root/benchmarks" \
  --seed 0 --batch-size 1024 --tensor-parallel-size 1 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85 \
  --max-new-tokens 32768 --max-model-len 32768 --record-hardware
