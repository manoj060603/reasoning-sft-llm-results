#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-fast-2b200-eval-${SLURM_JOB_ID}"
PY="$PWD/.venv-arc/bin/python"
[[ "${SLURM_JOB_PARTITION}" = b200_normal_q || "${SLURM_JOB_PARTITION}" = b200_preemptable_q ]]
nvidia-smi --query-gpu=name --format=csv,noheader | "$PY" -c 'import sys; assert "B200" in sys.stdin.read()'
"$PY" - <<'VERIFY'
import json,os
from pathlib import Path
from reasoning.fast_2b200 import verify_final
from reasoning.data import write_json,digest
run=Path('runs/lmf-full-length-15k-16k-3epoch-fast-2b200')
verify_final(run)
a=json.load(open(run/'allocations.json'))
assert a['binding']==json.load(open('artifacts/datasets-full-length-15k/fast_2b200_execution_receipt.json'))
assert all(digest(name)==sha for name,sha in a['binding']['execution_sha256'].items())
assert a['binding']['prior_charged_gpu_hours']==8
job=os.environ['SLURM_JOB_ID']
if job not in a['evaluation_job_ids']:
 assert 8+12*len(a['training_job_ids'])+8*(len(a['evaluation_job_ids'])+1)<=40
 a['evaluation_job_ids'].append(job)
write_json(run/'allocations.json',a)
VERIFY
RUN=runs/lmf-full-length-15k-16k-3epoch-fast-2b200
RESULT=results/lmf-full-length-15k-16k-3epoch-fast-2b200-suite
"$PY" -m reasoning.evaluate --model "$RUN" --data-dir artifacts/benchmarks_extended \
 --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
 --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
 --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$PY" -m reasoning.compare --baseline results/full-base-suite-v2 --candidate "$RESULT" \
 --output results/lmf-full-length-15k-16k-3epoch-fast-2b200-suite-comparison.json
set +e
"$PY" -m reasoning.domination results/lmf-full-length-15k-16k-3epoch-fast-2b200-suite-comparison.json \
 --output results/lmf-full-length-15k-16k-3epoch-fast-2b200-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
