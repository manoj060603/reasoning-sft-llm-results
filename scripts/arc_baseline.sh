#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PYTHON="$PWD/.venv-arc/bin/python"
nvidia-smi
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
"$PYTHON" -m reasoning.evaluate --model "$MODEL" --data-dir eval/data --benchmarks amc --output results/base-amc --batch-size 8 --gpu-memory-utilization .85
