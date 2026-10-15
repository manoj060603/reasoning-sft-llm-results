#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-full-triton-${SLURM_JOB_ID:-local}"
# Queue each full benchmark so short completions replenish active GPU sequences
# while long tails continue. The GPU still processes at most 128 sequences.
export EVAL_BATCH_SIZE=1024 EVAL_MAX_SEQS=128
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m reasoning.evaluate --model "$MODEL" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output results/full-base-suite-v2 --batch-size "$EVAL_BATCH_SIZE" \
  --max-num-seqs "$EVAL_MAX_SEQS" --record-hardware --gpu-memory-utilization .85
# Cross-cluster pilot dependency: validate complete development predictions.
# Bound the wait to 30 minutes; no public scores enter recipe selection.
READY=0
for ATTEMPT in $(seq 1 120); do
  if "$PYTHON" -m reasoning.choose_pilot > artifacts/full-pilot-choice.txt 2> artifacts/full-pilot-wait.log; then
    READY=1; break
  fi
  sleep 15
done
if [[ "$READY" != 1 ]]; then
  echo 'Baseline complete; pilot not ready within 30 minutes. Resume this job later.' >&2
  exit 3
fi
bash scripts/arc_final.sh train
bash scripts/arc_final.sh evaluate
