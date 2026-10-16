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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-gem-modelaware-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
MODEL="$PWD/artifacts/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
POOL=artifacts/prepared_extended/pool.json
STATIC=artifacts/features_extended.json
FEATURES=artifacts/features-modelaware-nll512.json
DATASETS=artifacts/datasets-modelaware-15k
CONFIG=configs/lmf-gem-modelaware-15000-8k.yaml
RUN=runs/lmf-gem-modelaware-15000-8k
RESULT=results/lmf-gem-modelaware-15000-8k-suite

test -x "$TRAIN_PYTHON"
test -x "$EVAL_PYTHON"
mkdir -p "$RUN" logs

# Score the complete sealed pool, not any public benchmark.  The operation is
# append-only/resumable and preserves the already frozen TF-IDF cluster labels.
if [[ ! -f "$FEATURES" ]]; then
  "$EVAL_PYTHON" -m reasoning.nll \
    --input "$POOL" --features "$STATIC" --output "$FEATURES" \
    --model "$MODEL" --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
    --nll-tokens 512 --max-batch-tokens 8192 --flush-every 256
fi

if [[ ! -f "$DATASETS/selection_manifest.json" ]]; then
  "$EVAL_PYTHON" -m reasoning.select \
    --pool "$POOL" --validation artifacts/prepared_extended/validation.json \
    --features "$FEATURES" --validation-features artifacts/validation_features_extended.json \
    --strategy hybrid --cutoff 8192 --size 15000 --seed 42 --output "$DATASETS"
fi

"$EVAL_PYTHON" - <<'PY'
import hashlib, json, math
from pathlib import Path

root = Path('.')
pool = root / 'artifacts/prepared_extended/pool.json'
static = root / 'artifacts/features_extended.json'
features_path = root / 'artifacts/features-modelaware-nll512.json'
data = root / 'artifacts/datasets-modelaware-15k/advanced.json'
manifest_path = root / 'artifacts/datasets-modelaware-15k/selection_manifest.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(pool) == 'c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9'
assert sha(static) == '039cd63692d528dcce2caf2be2bfed85be917a66298a2532933ef1107f9db979'
manifest = json.load(open(manifest_path))
rows = json.load(open(data))
features = json.load(open(features_path))
ids = manifest['selected_ids']['advanced']
assert manifest['arguments']['strategy'] == 'hybrid'
assert manifest['arguments']['cutoff'] == 8192
assert len(rows) == len(ids) == 15000 and len(set(ids)) == 15000
assert len(features) == 83037 and len({f['id'] for f in features}) == 83037
assert all(math.isfinite(float(f['nll'])) for f in features)
source = []
for directory in ('LLaMA-Factory/src', 'reasoning'):
    for path in sorted((root / directory).rglob('*.py')):
        source.append((str(path.relative_to(root)), sha(path)))
receipt = {
    'schema': 'reasoning-sft-gem-modelaware-15000-8k-v1',
    'literature': [
        'https://arxiv.org/abs/2505.17266',
        'https://arxiv.org/abs/2408.16673',
        'https://arxiv.org/abs/2506.13284',
    ],
    'selection': {
        'name': 'base_nll_diversity_hybrid', 'nll_prefix_tokens': 512,
        'within_length_rank_target': 0.65, 'clusters': 128,
        'public_benchmark_scores_used': False,
    },
    'objective': {'name': 'linear_gem', 'beta': 0.7},
    'training_rows': len(rows),
    'unique_selected_ids': len(set(ids)),
    'pool_sha256': sha(pool),
    'static_features_sha256': sha(static),
    'nll_features_sha256': sha(features_path),
    'data_sha256': sha(data),
    'selection_manifest_sha256': sha(manifest_path),
    'config_sha256': sha('configs/lmf-gem-modelaware-15000-8k.yaml'),
    'source_tree_sha256': hashlib.sha256(json.dumps(source, separators=(',', ':')).encode()).hexdigest(),
    'base_model': json.load(open('artifacts/model.json')),
}
(root / 'runs/lmf-gem-modelaware-15000-8k/provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
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
state = json.load(open('runs/lmf-gem-modelaware-15000-8k/trainer_state.json'))
assert state['global_step'] == 1875, state['global_step']
logs = state.get('log_history', [])
for row in logs:
    for key in ('loss', 'grad_norm'):
        if key in row:
            assert math.isfinite(float(row[key])), (key, row[key])
json.dump({'global_step': 1875, 'finite_logs': True, 'logged_rows': len(logs)},
          open('runs/lmf-gem-modelaware-15000-8k/completion.json', 'w'), indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 --candidate "$RESULT" \
  --output results/lmf-gem-modelaware-15000-8k-suite-comparison.json

set +e
"$EVAL_PYTHON" -m reasoning.domination \
  results/lmf-gem-modelaware-15000-8k-suite-comparison.json \
  --output results/lmf-gem-modelaware-15000-8k-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
