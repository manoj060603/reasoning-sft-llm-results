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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-prift-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
CONFIG=configs/lmf-prift-mass-15000-8k.yaml
RUN=runs/lmf-prift-mass-15000-8k
RESULT=results/lmf-prift-mass-15000-8k-suite

test -x "$TRAIN_PYTHON"
mkdir -p "$RUN" logs

"$EVAL_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path

root = Path('.')
data = root / 'artifacts/datasets-final-8k/advanced.json'
manifest_path = root / 'artifacts/datasets-final-8k/selection_manifest.json'
manifest = json.load(open(manifest_path))
rows = json.load(open(data))
expected = 'dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

ids = manifest['selected_ids']['advanced']
assert len(rows) == 15000 and len(ids) == 15000 and len(set(ids)) == 15000
assert manifest['arguments']['cutoff'] == 8192
assert all(row.get('instruction') and row.get('output') for row in rows)
assert sha(data) == expected, (sha(data), expected)
source = []
for directory in ('LLaMA-Factory/src', 'reasoning'):
    for path in sorted((root / directory).rglob('*.py')):
        source.append((str(path.relative_to(root)), sha(path)))
receipt = {
    'schema': 'reasoning-sft-prift-mass-15000-8k-v1',
    'paper': 'https://arxiv.org/abs/2606.09396',
    'official_repository': 'https://github.com/wang-kee/PriFT',
    'official_repository_commit_reviewed': 'f0a8f03ffc93915887e983943c11b9d4e420f427',
    'method': {'name': 'prift_mass', 'threshold': 0.5, 'frozen_reference': True},
    'training_rows': len(rows),
    'unique_selected_ids': len(set(ids)),
    'selection_cutoff': manifest['arguments']['cutoff'],
    'data_sha256': sha(data),
    'selection_manifest_sha256': sha(manifest_path),
    'dataset_info_sha256': sha(root / 'artifacts/datasets-final-8k/dataset_info.json'),
    'config_sha256': sha('configs/lmf-prift-mass-15000-8k.yaml'),
    'source_tree_sha256': hashlib.sha256(json.dumps(source, separators=(',', ':')).encode()).hexdigest(),
    'base_model': json.load(open('artifacts/model.json')),
}
(root / 'runs/lmf-prift-mass-15000-8k/provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
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
state = json.load(open('runs/lmf-prift-mass-15000-8k/trainer_state.json'))
assert state['global_step'] == 1875, state['global_step']
logs = state.get('log_history', [])
for row in logs:
    for key in ('loss', 'grad_norm'):
        if key in row:
            assert math.isfinite(float(row[key])), (key, row[key])
json.dump({'global_step': state['global_step'], 'finite_logs': True, 'logged_rows': len(logs)},
          open('runs/lmf-prift-mass-15000-8k/completion.json', 'w'), indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 \
  --candidate "$RESULT" \
  --output results/lmf-prift-mass-15000-8k-suite-comparison.json

set +e
"$EVAL_PYTHON" -m reasoning.domination \
  results/lmf-prift-mass-15000-8k-suite-comparison.json \
  --output results/lmf-prift-mass-15000-8k-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
