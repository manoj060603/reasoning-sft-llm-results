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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-full-length-${SLURM_JOB_ID:-local}"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
RUN=runs/lmf-full-length-15k-16k-3epoch-4a30
RESULT=results/lmf-full-length-15k-16k-3epoch-4a30-suite
test "${SLURM_JOB_PARTITION:-}" = b200_normal_q
nvidia-smi --query-gpu=name --format=csv,noheader | "$EVAL_PYTHON" -c 'import sys; names=sys.stdin.read(); assert "B200" in names,names'
mkdir -p "$RUN" logs
"$EVAL_PYTHON" - <<'VERIFY'
import json,os
from pathlib import Path
from reasoning.a30_runtime import verify_final
from reasoning.data import write_json
run=Path('runs/lmf-full-length-15k-16k-3epoch-4a30')
verify_final(run)
a=json.load(open(run/'allocations.json'))
jobs=a.setdefault('evaluation_job_ids',[])
job=os.environ['SLURM_JOB_ID']
if job not in jobs:
 assert 8*len(a['job_ids'])+8*(len(jobs)+1)<=40,'40 physical GPU-hour ceiling exhausted'
 jobs.append(job)
write_json(run/'allocations.json',a)
VERIFY
"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
 --data-dir artifacts/benchmarks_extended \
 --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
 --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
 --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.compare --baseline results/full-base-suite-v2 \
 --candidate "$RESULT" --output results/lmf-full-length-15k-16k-3epoch-4a30-suite-comparison.json
set +e
"$EVAL_PYTHON" -m reasoning.domination results/lmf-full-length-15k-16k-3epoch-4a30-suite-comparison.json \
 --output results/lmf-full-length-15k-16k-3epoch-4a30-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
