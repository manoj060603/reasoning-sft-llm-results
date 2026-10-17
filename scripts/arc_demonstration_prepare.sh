#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-demonstration-${SLURM_JOB_ID:-local}"
PY="$PWD/.venv-arc/bin/python"
TRAIN_PY="$PWD/.venv-lmf/bin/python"
BASE="$PWD/artifacts/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
DATA=artifacts/datasets-demonstration-response-15k
test "${SLURM_JOB_PARTITION:-}" = b200_normal_q
nvidia-smi --query-gpu=name --format=csv,noheader | "$PY" -c 'import sys; names=sys.stdin.read(); assert "B200" in names,names'
mkdir -p logs artifacts/demonstration-response "$DATA"
"$PY" - <<'PY'
import json,os
from pathlib import Path
from reasoning.data import digest,write_json
from reasoning.demonstration_response import load_plan
plan=load_plan('artifacts/demonstration-response/plan.json')
assert plan['prefix_usable']>=12000
path=Path('artifacts/demonstration-response/allocations.json')
receipt=json.load(open(path)) if path.exists() else {'plan_sha256':digest('artifacts/demonstration-response/plan.json'),'job_ids':[]}
assert receipt['plan_sha256']==digest('artifacts/demonstration-response/plan.json')
job=os.environ['SLURM_JOB_ID']
if job not in receipt['job_ids']:
 assert len(receipt['job_ids'])<5,'Frozen preparation budget exhausted: five eight-hour B200 allocations'
 receipt['job_ids'].append(job)
write_json(path,receipt)
from reasoning.grader_preflight import check_grader
check_grader()
from vllm import SamplingParams
SamplingParams(temperature=.6,top_p=.95,top_k=-1,max_tokens=2048,n=1,seed=42)
PY
"$PY" -m reasoning.demonstration_response score --batch-size 256
"$PY" -m reasoning.demonstration_response export
"$PY" -m reasoning.demonstration_response audit --batch-size 128
"$TRAIN_PY" -m reasoning.demonstration_preflight --model "$BASE" --directory "$DATA"
"$PY" - <<'PY'
import json
from reasoning.data import digest,write_json
data='artifacts/datasets-demonstration-response-15k/'
pre=json.load(open(data+'training_preflight.json'))
audit=json.load(open(data+'audit_receipt.json'))
assert audit['passed'] and pre['examples']==15000 and pre['updates']==1875
write_json('artifacts/demonstration-response/preparation_receipt.json',{
 'schema':'demonstration-response-v1','examples':15000,'updates':1875,
 'data_sha256':digest(data+'advanced.json'),
 'selection_sha256':digest(data+'selection_manifest.json'),
 'audit_sha256':digest(data+'audit_receipt.json'),
 'preflight_sha256':digest(data+'training_preflight.json'),
 'plan_sha256':digest('artifacts/demonstration-response/plan.json'),
 'original_targets_unchanged':True,'cpu_training_preflight':pre})
print('Full preparation complete; no training job is submitted from inside this allocation.',flush=True)
PY
