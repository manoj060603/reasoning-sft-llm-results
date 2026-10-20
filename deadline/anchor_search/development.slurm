#!/bin/bash
#SBATCH --job-name=sft-anchor-v2-source
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-source-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/anchor-source-%j.err
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
anchor_root="${1:?Expected sealed anchor run directory}"
qa_receipt="${2:?Expected successful native QA receipt}"
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(6.5)'
for role in original incumbent alpha0125 alpha0375; do
  CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" -m deadline.anchor_search.preflight --verify "$qa_receipt"
  "$PWD/.venv/bin/python" -m deadline.anchor_search.screen run --output "$anchor_root" --split search --role "$role"
done
CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" -m deadline.anchor_search.screen select --output "$anchor_root"
winner="$(CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" - "$anchor_root" <<'PY'
import sys
from pathlib import Path
from deadline.recovery.common import read_json
d=read_json(Path(sys.argv[1])/'selection.json')
print(d['winner'] if d['accepted'] else 'NONE')
PY
)"
if [[ "$winner" == NONE ]]; then
  printf '%s\n' 'No new candidate passed the frozen source-search gate; retain published incumbent.'
  exit 0
fi
for role in original incumbent "$winner"; do
  CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" -m deadline.anchor_search.preflight --verify "$qa_receipt"
  "$PWD/.venv/bin/python" -m deadline.anchor_search.screen run --output "$anchor_root" --split confirmation --role "$role"
done
CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" -m deadline.anchor_search.screen confirm --output "$anchor_root"
CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" - "$anchor_root" "$qa_receipt" <<'PY'
import os,sys
from pathlib import Path
from deadline.recovery.common import read_json
from deadline.anchor_search.screen import validate_decision
from deadline.pipeline import submit
root=Path(sys.argv[1]);decision=read_json(root/'decision.json')
if not decision['accepted']:
    print('The single candidate failed confirmation; no fallback or benchmark admission.',flush=True)
else:
    validate_decision(root)
    submit('anchor_v2_benchmarks',Path('deadline/anchor_search/evaluate.slurm'),
           arguments=sys.argv[1:],dependency=os.environ['SLURM_JOB_ID'],gpu_hours=3,required_hours=5)
PY
