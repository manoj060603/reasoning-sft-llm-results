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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-gen-ssd-v2-${SLURM_JOB_ID:-local}"

PYTHON="$PWD/.venv-arc/bin/python"
BASE=$($PYTHON -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
TEACHER=$($PYTHON -c 'import json;print(json.load(open("artifacts/gen-ssd-2k/teacher_provenance.json"))["path"])')
test -f artifacts/datasets-rescue-4k/advanced.json

# A fresh small batch must pass generation, verification, export, exact
# chat-template length checks, matched IDs, and datasets loading before 2K.
rm -rf artifacts/gen-ssd-preflight-v2
"$PYTHON" -m reasoning.gen_ssd \
  --input artifacts/datasets-rescue-4k/advanced.json \
  --teacher "$TEACHER" --student "$BASE" \
  --output artifacts/gen-ssd-preflight-v2 \
  --target-size 4 --max-source 32 --attempts 2 \
  --batch-size 8 --score-batch-size 2 \
  --chunk-tokens 1024 --max-new-tokens 4096 \
  --first-candidates 4 --later-candidates 2 --training-cutoff 4096
"$PYTHON" scripts/validate_gen_ssd_export.py \
  --dataset-dir artifacts/gen-ssd-preflight-v2 --model "$BASE" \
  --expected 4 --cutoff 4096 \
  --receipt artifacts/gen-ssd-preflight-v2/export_validation.json

# Do not reuse receipts from the earlier, pre-cutoff implementation.
rm -rf artifacts/gen-ssd-2k/records artifacts/gen-ssd-2k/generation_manifest.json \
  artifacts/gen-ssd-2k/generation_summary.json artifacts/gen-ssd-2k/guided.json \
  artifacts/gen-ssd-2k/existing.json artifacts/gen-ssd-2k/dataset_info.json
bash scripts/arc_gen_ssd_2k.sh
"$PYTHON" scripts/validate_gen_ssd_export.py \
  --dataset-dir artifacts/gen-ssd-2k --model "$BASE" \
  --expected 2000 --cutoff 4096 \
  --receipt artifacts/gen-ssd-2k/export_validation.json
