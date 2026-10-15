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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-trimsft-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
CONFIG=configs/lmf-trimsft-15000.yaml
RUN=runs/lmf-trimsft-15000
RESULT=results/lmf-trimsft-15000-suite

test -x "$TRAIN_PYTHON"
test -f artifacts/datasets-final-8k/advanced.json
test -f artifacts/datasets-final-8k/dataset_info.json
mkdir -p "$RUN" logs

"$EVAL_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path

root = Path('.')
data = root / 'artifacts/datasets-final-8k/advanced.json'
manifest_path = root / 'artifacts/datasets-final-8k/selection_manifest.json'
manifest = json.load(open(manifest_path))
rows = json.load(open(data))
expected_data_hash = 'dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert len(rows) == 15000, len(rows)
ids = manifest['selected_ids']['advanced']
assert len(ids) == 15000 and len(set(ids)) == 15000
assert all(row.get('instruction') and row.get('output') for row in rows)
assert sha(data) == expected_data_hash, (sha(data), expected_data_hash)

source = []
for path in sorted((root / 'LLaMA-Factory/src').rglob('*.py')):
    source.append((str(path.relative_to(root)), sha(path)))
for path in sorted((root / 'reasoning').rglob('*.py')):
    source.append((str(path.relative_to(root)), sha(path)))
receipt = {
    'schema': 'reasoning-sft-trimsft-15000-v1',
    'paper': 'https://arxiv.org/abs/2609.09707',
    'method': {'margin': 1.5, 'bandwidth': 0.8, 'weight_detached': True},
    'training_rows': len(rows),
    'unique_selected_ids': len(set(ids)),
    'data_sha256': sha(data),
    'selection_manifest_sha256': sha(manifest_path),
    'dataset_info_sha256': sha(root / 'artifacts/datasets-final-8k/dataset_info.json'),
    'config_sha256': sha('configs/lmf-trimsft-15000.yaml'),
    'source_tree_sha256': hashlib.sha256(json.dumps(source, separators=(',', ':')).encode()).hexdigest(),
    'base_model': json.load(open('artifacts/model.json')),
}
(root / 'runs/lmf-trimsft-15000/provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
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
state = json.load(open('runs/lmf-trimsft-15000/trainer_state.json'))
assert state['global_step'] == 1875, state['global_step']
logs = state.get('log_history', [])
for row in logs:
    for key in ('loss', 'grad_norm'):
        if key in row:
            assert math.isfinite(float(row[key])), (key, row[key])
json.dump({'global_step': state['global_step'], 'finite_logs': True, 'logged_rows': len(logs)},
          open('runs/lmf-trimsft-15000/completion.json', 'w'), indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 \
  --candidate "$RESULT" \
  --output results/lmf-trimsft-15000-suite-comparison.json

set +e
"$EVAL_PYTHON" -m reasoning.domination \
  results/lmf-trimsft-15000-suite-comparison.json \
  --output results/lmf-trimsft-15000-domination.json
domination_status=$?
set -e
if [[ "$domination_status" -ne 0 && "$domination_status" -ne 2 ]]; then
  exit "$domination_status"
fi
