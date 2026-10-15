#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
REASONING_CUDA_HOME=/apps/common/software/CUDA/12.8.0
REASONING_TRITON_CACHE="${TMPDIR:-/tmp}/reasoning-triton-${SLURM_JOB_ID:-local}"
export OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
"$PYTHON" -m unittest discover -s tests -v
CUDA_HOME="$REASONING_CUDA_HOME" TRITON_CACHE_DIR="$REASONING_TRITON_CACHE" "$PYTHON" -m reasoning.export_preflight
"$PYTHON" -m reasoning.select --strategy route --size 512 --cutoff 4096 --output artifacts/datasets-pilot
"$PYTHON" -m reasoning.devset --size 32 --output artifacts/dev-pilot/acereason_dev/test.jsonl
for specification in filtered_random:sft route:sft route:dft route:asft; do
  variant=${specification%:*}; objective=${specification#*:}
  DATA=filtered_random
  if [[ "$variant" == route ]]; then DATA=advanced; fi
  RUN="runs/pilot-${variant}-${objective}"
  CUDA_HOME="$REASONING_CUDA_HOME" TRITON_CACHE_DIR="$REASONING_TRITON_CACHE" "$PYTHON" -m reasoning.train_adapter --model "$MODEL" --data "artifacts/datasets-pilot/${DATA}.json" --output "$RUN" --objective "$objective" --steps 64 --cutoff 4096 --resume
  "$PYTHON" -m reasoning.evaluate --model "$RUN/merged" --data-dir artifacts/dev-pilot --benchmarks acereason_dev --output "results/pilot-${variant}-${objective}-dev" --batch-size 32 --max-new-tokens 8192 --gpu-memory-utilization .85
 done
"$PYTHON" -m reasoning.compare --baseline results/pilot-filtered_random-sft-dev --candidate results/pilot-route-sft-dev --output results/pilot-selection-comparison.json
"$PYTHON" -m reasoning.compare --baseline results/pilot-route-sft-dev --candidate results/pilot-route-dft-dev --output results/pilot-objective-comparison.json
WINNER=$("$PYTHON" -m reasoning.choose_pilot)
for variant in "$WINNER" filtered_random-sft; do
  "$PYTHON" -m reasoning.evaluate --model "runs/pilot-${variant}/merged" --data-dir eval/data --benchmarks amc --output "results/pilot-${variant}-amc" --batch-size 8 --gpu-memory-utilization .85
  if [[ "$variant" == "$WINNER" ]]; then
    "$PYTHON" -m reasoning.compare --baseline results/base-amc --candidate "results/pilot-${WINNER}-amc" --output results/pilot-versus-base-amc.json
  fi
done
"$PYTHON" -m reasoning.compare --baseline results/pilot-filtered_random-sft-amc --candidate "results/pilot-${WINNER}-amc" --output results/pilot-versus-random-amc.json
