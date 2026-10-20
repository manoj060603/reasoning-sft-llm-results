#!/bin/bash
# First bounded allocation for the unchanged all-nine recovery evaluation.
# Later continuation is conditional on fresh terminal resource reclamation.
#SBATCH --job-name=sft-recovery-eval-initial
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
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-eval-initial-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-eval-initial-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PYTHONPATH="$PWD"
recovery_root="${1:?Expected recovery output directory}"
qa_receipt="${2:?Expected original successful native CPU-QA receipt}"
initial_preflight="${3:?Expected successful bounded-stage preflight receipt}"
CUDA_VISIBLE_DEVICES='' "$PWD/.venv/bin/python" - "$initial_preflight" <<'PY'
from pathlib import Path
import sys
from deadline.pipeline import guard_start
from deadline.recovery.common import digest, read_json, require
receipt = read_json(sys.argv[1])
require(receipt.get('schema') == 'deadline-recovery-initial-eval-preflight-v1' and
        receipt.get('success') is True and receipt.get('gpu_seconds_reserved') == 2700,
        'Bounded benchmark stage lacks its successful native preflight')
for path, expected in receipt['source_sha256'].items():
    require(digest(path) == expected, 'Bounded-stage source changed after preflight')
for path, expected in receipt['input_receipt_sha256'].items():
    require(digest(path) == expected, 'Bounded-stage input receipt changed')
# Reserve this 45-minute stage, a potential later six-hour continuation, and
# three hours for checks/publication. No continuation is admitted by this file.
guard_start(9.75)
PY
# Only this entry script's SBATCH directives are interpreted by Slurm. Running
# the QA-bound common wrapper through bash preserves every model/evaluation
# setting; its six-hour SBATCH comment does not alter this 45-minute allocation.
exec bash "$PWD/deadline/operations/recovery-evaluate.slurm" "$recovery_root" "$qa_receipt"
