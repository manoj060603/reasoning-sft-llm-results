#!/bin/bash
#SBATCH --job-name=sft-extension-source-generation-v3-qa
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-source-generation-v3-qa-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-source-generation-v3-qa-%j.err
set -euo pipefail
umask 077
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The scheduler can start this allocation before the submitter finishes sealing
# its accepted identity. Wait only for that exact in-flight CPU admission.
"$PWD/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import re
import time

role = "source_generation_v3_qa_20261021_v1"
job_id = os.environ.get("SLURM_JOB_ID", "")
if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
    raise RuntimeError("Expected the assigned CPU job identity")
ledger_path = Path("extension/job_ledger.json")
deadline = time.monotonic() + 30.0

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("Duplicate admission metadata key")
        result[key] = value
    return result

def reject_constant(_):
    raise RuntimeError("Nonfinite admission metadata")

while True:
    if time.monotonic() >= deadline:
        raise RuntimeError("CPU admission readiness timed out; no retry")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise RuntimeError("Expected the existing admission ledger")
    with ledger_path.open("rb") as stream:
        payload = stream.read((2 << 20) + 1)
    if len(payload) > 2 << 20:
        raise RuntimeError("Admission metadata exceeds its bound")
    ledger = json.loads(payload, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if (not isinstance(ledger, dict) or ledger.get("schema") != "reasoning-sft-extension-job-ledger-v1"
            or not isinstance(ledger.get("jobs"), dict) or not isinstance(ledger["jobs"].get(role), dict)):
        raise RuntimeError("Malformed or missing CPU admission metadata")
    entry = ledger["jobs"][role]
    status = entry.get("status")
    if "job_id" in entry and entry["job_id"] != job_id:
        raise RuntimeError("CPU admission belongs to another job")
    if time.monotonic() >= deadline:
        raise RuntimeError("CPU admission readiness timed out; no retry")
    if status == "submitted" and entry.get("job_id") == job_id:
        break
    if status not in {"submitting", "uncertain"}:
        raise RuntimeError("CPU admission is not awaiting its submission seal")
    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
PY
"$PWD/.venv/bin/python" -m extension.source_generation_v3_qa "$PWD/extension/runs/source-generation-v3-qa-${SLURM_JOB_ID:?Expected CPU allocation}"
