#!/usr/bin/env bash
# Explicit full-size continuation. Not submitted by the pilot driver.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=${1:?Use train or evaluate}; shift
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
# Recompute the choice from complete, matching development results only.
WINNER=$("$PYTHON" -m reasoning.choose_pilot)
OBJECTIVE=${WINNER##*-}
DATA=advanced
if [[ "$WINNER" == filtered_random-sft ]]; then DATA=filtered_random; fi
RUN="runs/final-15000-${WINNER}"
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-32}
case "$STAGE" in
  train)
    "$PYTHON" - "$DATA" <<'PY'
import json,sys
from reasoning.data import read_rows
root='artifacts/datasets-final-8k/'
manifest=json.load(open(root+'selection_manifest.json'))
assert manifest['arguments']['cutoff']==8192
assert len(read_rows(root+sys.argv[1]+'.json'))==15000
assert len(set(manifest['selected_ids'][sys.argv[1]]))==15000
PY
    CUDA_HOME=/apps/common/software/CUDA/12.8.0 \
      TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-final-triton-${SLURM_JOB_ID:-local}" \
      "$PYTHON" -m reasoning.train_adapter --model "$MODEL" \
      --data "artifacts/datasets-final-8k/$DATA.json" --output "$RUN" \
      --objective "$OBJECTIVE" --steps 1875 --accumulation 8 --cutoff 8192 \
      --save-every 100 --resume
    ;;
  evaluate)
    test -f "$RUN/complete.json"
    "$PYTHON" - "$RUN" <<'PY'
import json,sys
stats=json.load(open(sys.argv[1]+'/data_stats.json'))
if stats['used']!=15000 or stats['dropped_overlength']:
    raise ValueError('Full-size evaluation requires 15000 used examples with no dropped rows')
PY
    SCOPE=suite
    if [[ $# -eq 0 ]]; then
      set -- amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa
    else
      SCOPE=$(IFS=_; echo "$*")
    fi
    # One engine for the suite; complete batches are still atomically resumable.
    "$PYTHON" -m reasoning.evaluate --model "$MODEL" \
      --data-dir artifacts/benchmarks_extended --benchmarks "$@" \
      --output "results/full-base-$SCOPE-v2" --batch-size "$EVAL_BATCH_SIZE" \
      --max-num-seqs "$EVAL_MAX_SEQS" --record-hardware --gpu-memory-utilization .85
    "$PYTHON" -m reasoning.evaluate --model "$RUN/merged" \
      --data-dir artifacts/benchmarks_extended --benchmarks "$@" \
      --output "results/final-15000-${WINNER}-$SCOPE" --batch-size "$EVAL_BATCH_SIZE" \
      --max-num-seqs "$EVAL_MAX_SEQS" --record-hardware --gpu-memory-utilization .85
    "$PYTHON" -m reasoning.compare --baseline "results/full-base-$SCOPE-v2" \
      --candidate "results/final-15000-${WINNER}-$SCOPE" \
      --output "results/final-15000-${WINNER}-$SCOPE-comparison.json"
    ;;
  *) echo 'Use train or evaluate [benchmark ...]' >&2; exit 2;;
esac
