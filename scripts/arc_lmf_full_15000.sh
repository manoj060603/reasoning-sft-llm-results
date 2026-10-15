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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-lmf-full-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
CONFIG=configs/lmf-full-15000-sft.yaml
RUN=runs/lmf-full-15000-sft
RESULT=results/lmf-full-15000-sft-suite

test -x "$TRAIN_PYTHON"
test -f artifacts/datasets-final-8k/advanced.json
test -f artifacts/datasets-final-8k/dataset_info.json

mkdir -p "$RUN"
"$EVAL_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path

root = Path('.')
data = root / 'artifacts/datasets-final-8k/advanced.json'
manifest = json.load(open(root / 'artifacts/datasets-final-8k/selection_manifest.json'))
rows = json.load(open(data))
assert len(rows) == 15000
assert len(manifest['selected_ids']['advanced']) == 15000
assert len(set(manifest['selected_ids']['advanced'])) == 15000
assert all(row.get('instruction') and row.get('output') for row in rows)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = []
for path in sorted((root / 'LLaMA-Factory/src').rglob('*.py')):
    source.append((str(path.relative_to(root)), sha(path)))
receipt = {
    'schema': 'reasoning-sft-lmf-full-15000-v1',
    'training_rows': len(rows),
    'unique_selected_ids': len(set(manifest['selected_ids']['advanced'])),
    'data_sha256': sha(data),
    'selection_manifest_sha256': sha(root / 'artifacts/datasets-final-8k/selection_manifest.json'),
    'dataset_info_sha256': sha(root / 'artifacts/datasets-final-8k/dataset_info.json'),
    'config_sha256': sha('configs/lmf-full-15000-sft.yaml'),
    'llamafactory_source_tree': hashlib.sha256(json.dumps(source, separators=(',', ':')).encode()).hexdigest(),
    'base_model': json.load(open('artifacts/model.json')),
}
(root / 'runs/lmf-full-15000-sft/provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
PY

"$TRAIN_PYTHON" - <<'PY' > "$RUN/environment.txt"
from llamafactory.extras.env import print_env
print_env()
PY

if [[ ! -f "$RUN/trainer_state.json" ]]; then
  "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$CONFIG"
fi

test -f "$RUN/model.safetensors.index.json" -o -f "$RUN/model.safetensors"
"$EVAL_PYTHON" - <<'PY'
import json, math
state = json.load(open('runs/lmf-full-15000-sft/trainer_state.json'))
assert state['global_step'] == 1875, state['global_step']
logs = state.get('log_history', [])
for row in logs:
    for key in ('loss', 'grad_norm'):
        if key in row:
            assert math.isfinite(float(row[key])), (key, row[key])
json.dump({'global_step': state['global_step'], 'finite_logs': True, 'logged_rows': len(logs)},
          open('runs/lmf-full-15000-sft/completion.json', 'w'), indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 \
  --candidate "$RESULT" \
  --output results/lmf-full-15000-sft-suite-comparison.json
