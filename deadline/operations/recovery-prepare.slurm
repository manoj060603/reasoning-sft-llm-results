#!/bin/bash
#SBATCH --job-name=sft-recovery-cpu-prepare
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-prepare-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-prepare-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -d /apps/common/software/CUDA/12.8.0 ]]; then
  export CUDA_HOME=/apps/common/software/CUDA/12.8.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
DEADLINE_PYTHON="$PWD/.venv/bin/python"
recovery_root="${1:?Expected unique recovery output directory}"
qa_receipt="${2:?Expected unique native CPU-QA receipt path}"
"$DEADLINE_PYTHON" -c 'from deadline.pipeline import guard_start; guard_start(4.75)'
"$DEADLINE_PYTHON" -m deadline.operations.recovery_preflight --output "$qa_receipt"
"$DEADLINE_PYTHON" -m deadline.recovery.merge \
  --root deadline \
  --base /home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --output "$recovery_root"
"$DEADLINE_PYTHON" -m deadline.recovery.screen prepare \
  --artifacts deadline/artifacts --merge-receipt "$recovery_root/complete.json" \
  --output "$recovery_root/development"
"$DEADLINE_PYTHON" -m deadline.operations.recovery_preflight --verify "$qa_receipt"
"$DEADLINE_PYTHON" - "$recovery_root" "$qa_receipt" <<'PY'
import os
from pathlib import Path
import sys
from deadline.pipeline import submit
submit('recovery_development', Path('deadline/operations/recovery-development.slurm'),
       arguments=sys.argv[1:], dependency=os.environ['SLURM_JOB_ID'],
       gpu_hours=.75, required_hours=3.75)
PY
