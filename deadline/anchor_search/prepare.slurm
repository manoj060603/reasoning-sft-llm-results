#!/bin/bash
#SBATCH --job-name=sft-anchor-v2-prepare
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-prepare-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-prepare-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
anchor_root="${1:?Expected new anchor run directory}"
qa_receipt="${2:?Expected unique native QA receipt path}"
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(7.5)'
"$PWD/.venv/bin/python" -m deadline.anchor_search.preflight --output "$qa_receipt"
"$PWD/.venv/bin/python" - "$anchor_root" <<'PY'
import sys
from deadline.anchor_search.core import prepare
prepare(sys.argv[1], root='deadline', base='/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1')
PY
"$PWD/.venv/bin/python" -m deadline.anchor_search.preflight --verify "$qa_receipt"
"$PWD/.venv/bin/python" - "$anchor_root" "$qa_receipt" <<'PY'
import os,sys
from pathlib import Path
from deadline.pipeline import submit
submit('anchor_v2_development',Path('deadline/anchor_search/development.slurm'),
       arguments=sys.argv[1:],dependency=os.environ['SLURM_JOB_ID'],gpu_hours=1.5,required_hours=6.5)
PY
