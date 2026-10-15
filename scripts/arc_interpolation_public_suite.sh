#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD"
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-interp-public-${SLURM_JOB_ID:-local}"
PYTHON="$PWD/.venv-arc/bin/python"
MODEL="$PWD/runs/lmf-full-15000-interp-050"
RESULT=results/lmf-full-15000-interp-050-suite

test -f "$MODEL/interpolation.json"
"$PYTHON" - <<'PY'
import json
r=json.load(open('runs/lmf-full-15000-interp-050/interpolation.json'))
assert r['alpha']==0.5
assert r['tensor_count']==434
assert r['output_dtype']=='bfloat16'
PY

"$PYTHON" -m reasoning.evaluate --model "$MODEL" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 --candidate "$RESULT" \
  --output results/lmf-full-15000-interp-050-suite-comparison.json
