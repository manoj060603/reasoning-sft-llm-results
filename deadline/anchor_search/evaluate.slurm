#!/bin/bash
#SBATCH --job-name=sft-anchor-v2-benchmarks
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-eval-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-eval-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -d /apps/common/software/CUDA/12.8.0 ]]; then
  export CUDA_HOME=/apps/common/software/CUDA/12.8.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
anchor_root="${1:?Expected confirmed anchor run directory}"
qa_receipt="${2:?Expected successful native QA receipt}"
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(5)'
CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" -m deadline.anchor_search.preflight --verify "$qa_receipt"
model="$(CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" - "$anchor_root" <<'PY'
import sys
from pathlib import Path
from deadline.anchor_search.core import load_prepared
from deadline.anchor_search.screen import validate_decision
from deadline.recovery.common import digest,require
root=Path(sys.argv[1]);prepared=load_prepared(root);decision=validate_decision(root)
model=prepared['models'][decision['winner']]
require(str(Path(model['path']).resolve())==str(Path(decision['model_path']).resolve()),'Winner model path changed')
for name,expected in model['files'].items():
    require(digest(Path(model['path'])/name)==expected,'Selected model bytes changed: '+name)
require(digest('reasoning/evaluate.py')=='55181149a1010cd60a5979e859e2fcd6ee5d1a3abb8d4fbf33cee4b18586c316','Frozen evaluator changed')
print(model['path'])
PY
)"
"$PWD/.venv/bin/python" -m reasoning.evaluate --model "$model" \
  --data-dir deadline/artifacts/benchmarks \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$anchor_root/benchmarks" --seed 0 --batch-size 1024 \
  --tensor-parallel-size 1 --max-num-seqs 128 --gpu-memory-utilization 0.85 \
  --max-new-tokens 32768 --max-model-len 32768 --record-hardware
