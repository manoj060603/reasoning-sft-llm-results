#!/bin/bash
#SBATCH --job-name=sft-extension-native-tensor-comparison-qa
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-native-tensor-comparison-qa-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-native-tensor-comparison-qa-%j.err
set -euo pipefail
umask 077
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# A job can start while its submitter still seals the accepted scheduler ID.
"$PWD/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import re
import time

role = "native_tensor_comparison_qa_20261021_v1"
job = os.environ.get("SLURM_JOB_ID", "")
if re.fullmatch(r"[1-9][0-9]*", job) is None:
    raise RuntimeError("Expected the assigned CPU job identity")
path = Path("extension/job_ledger.json")
until = time.monotonic() + 30.0

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise RuntimeError("Duplicate admission metadata key")
        result[key] = value
    return result

def nonfinite(_):
    raise RuntimeError("Nonfinite admission metadata")

while True:
    if time.monotonic() >= until:
        raise RuntimeError("CPU admission readiness timed out; no retry")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Expected the existing admission ledger")
    with path.open("rb") as stream:
        payload = stream.read((2 << 20) + 1)
    if len(payload) > 2 << 20:
        raise RuntimeError("Admission metadata exceeds its bound")
    ledger = json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)
    if (type(ledger) is not dict or ledger.get("schema") != "reasoning-sft-extension-job-ledger-v1"
            or type(ledger.get("jobs")) is not dict or type(ledger["jobs"].get(role)) is not dict):
        raise RuntimeError("Malformed or missing current CPU admission")
    entry = ledger["jobs"][role]
    if "job_id" in entry and entry["job_id"] != job:
        raise RuntimeError("CPU admission belongs to another job")
    if time.monotonic() >= until:
        raise RuntimeError("CPU admission readiness timed out; no retry")
    if entry.get("status") == "submitted" and entry.get("job_id") == job:
        break
    if entry.get("status") not in {"submitting", "uncertain"}:
        raise RuntimeError("CPU admission is not awaiting its submission seal")
    time.sleep(min(.25, max(0., until - time.monotonic())))
PY
"$PWD/.venv/bin/python" -m extension.native_tensor_comparison_qa --root "$PWD" --plan extension/native_tensor_comparison_qa_plan.json --output "$PWD/extension/runs/native-tensor-comparison-qa-${SLURM_JOB_ID:?Expected CPU allocation}"
