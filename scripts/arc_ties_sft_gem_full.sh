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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-ties-${SLURM_JOB_ID:-local}"

PYTHON="$PWD/.venv-arc/bin/python"
BASE=$($PYTHON -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
SFT="$PWD/runs/lmf-full-15000-sft"
GEM="$PWD/runs/lmf-gem-15000-8k"
MERGED="$PWD/runs/lmf-ties-sft-gem-15k"
RESULT=results/lmf-ties-sft-gem-15k-suite

test -f "$SFT/model.safetensors.index.json"
test -f "$GEM/model.safetensors.index.json"
mkdir -p logs

"$PYTHON" -m reasoning.ties_merge --base "$BASE" --expert "$SFT" --expert "$GEM" \
  --density 0.20 --scale 1.0 --output "$MERGED"

"$PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

root = Path('.')
receipt = json.load(open('runs/lmf-ties-sft-gem-15k/ties_merge.json'))
assert receipt['density'] == .20 and receipt['scale'] == 1.0
assert receipt['tensor_count'] > 0 and receipt['output_dtype'] == 'bfloat16'
provenance = {
    'schema': 'reasoning-sft-ties-sft-gem-15k-v1',
    'literature': ['https://arxiv.org/abs/2306.01708', 'https://arxiv.org/abs/2203.05482'],
    'merge': receipt,
    'experts': ['runs/lmf-full-15000-sft', 'runs/lmf-gem-15000-8k'],
    'public_benchmark_scores_used_for_merge': False,
    'selection_or_checkpoint_tuning': False,
    'base_model': json.load(open('artifacts/model.json')),
    'source_sha256': sha('reasoning/ties_merge.py'),
}
(root / 'runs/lmf-ties-sft-gem-15k/provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
PY

"$PYTHON" -m reasoning.evaluate --model "$MERGED" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 --candidate "$RESULT" \
  --output results/lmf-ties-sft-gem-15k-suite-comparison.json

set +e
"$PYTHON" -m reasoning.domination \
  results/lmf-ties-sft-gem-15k-suite-comparison.json \
  --output results/lmf-ties-sft-gem-15k-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
