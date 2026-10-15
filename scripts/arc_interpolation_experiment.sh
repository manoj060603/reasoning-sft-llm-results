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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-interpolation-${SLURM_JOB_ID:-local}"
PYTHON="$PWD/.venv-arc/bin/python"
BASE=$($PYTHON -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
FULL="$PWD/runs/lmf-full-15000-sft"

test -f "$FULL/model.safetensors.index.json"
test -f artifacts/dev-rescue/acereason_dev/test.jsonl
test -f artifacts/dev-concise-confirm/acereason_dev/test.jsonl

for SPEC in 025:0.25 050:0.50 075:0.75; do
  NAME=${SPEC%%:*}; ALPHA=${SPEC#*:}
  "$PYTHON" -m reasoning.interpolate_weights --base "$BASE" --tuned "$FULL" \
    --alpha "$ALPHA" --output "runs/lmf-full-15000-interp-${NAME}"
done

evaluate_dev () {
  local NAME=$1 MODEL=$2
  "$PYTHON" -m reasoning.evaluate --model "$MODEL" \
    --data-dir artifacts/dev-rescue --benchmarks acereason_dev \
    --output "results/interpolation-${NAME}-dev32k" --batch-size 1024 \
    --max-num-seqs 128 --max-new-tokens 32768 --record-hardware \
    --gpu-memory-utilization .85
}

evaluate_dev base "$BASE"
evaluate_dev full "$FULL"
for NAME in 025 050 075; do
  evaluate_dev "$NAME" "$PWD/runs/lmf-full-15000-interp-${NAME}"
done

"$PYTHON" -m reasoning.choose_interpolation \
  --base results/interpolation-base-dev32k \
  --full results/interpolation-full-dev32k \
  --candidate alpha-025=results/interpolation-025-dev32k \
  --candidate alpha-050=results/interpolation-050-dev32k \
  --candidate alpha-075=results/interpolation-075-dev32k \
  --output results/interpolation-selection.json

SELECTED=$($PYTHON -c 'import json;print(json.load(open("results/interpolation-selection.json"))["selected"])')
NAME=${SELECTED#alpha-}
"$PYTHON" -m reasoning.evaluate --model "$BASE" \
  --data-dir artifacts/dev-concise-confirm --benchmarks acereason_dev \
  --output results/interpolation-confirm-base32k --batch-size 1024 \
  --max-num-seqs 128 --max-new-tokens 32768 --record-hardware \
  --gpu-memory-utilization .85
"$PYTHON" -m reasoning.evaluate --model "$PWD/runs/lmf-full-15000-interp-${NAME}" \
  --data-dir artifacts/dev-concise-confirm --benchmarks acereason_dev \
  --output "results/interpolation-confirm-${NAME}-32k" --batch-size 1024 \
  --max-num-seqs 128 --max-new-tokens 32768 --record-hardware \
  --gpu-memory-utilization .85
"$PYTHON" -m reasoning.compare \
  --baseline results/interpolation-confirm-base32k \
  --candidate "results/interpolation-confirm-${NAME}-32k" \
  --output results/interpolation-confirm-comparison.json
