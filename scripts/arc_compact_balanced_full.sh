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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-compact-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
CONFIG=configs/lmf-compact-balanced-15000.yaml
DATA=artifacts/datasets-compact-balanced-15k/compact_balanced.json
RUN=runs/lmf-compact-balanced-15000
RESULT=results/lmf-compact-balanced-15000-suite

test -x "$TRAIN_PYTHON"
test -f "$DATA"
test -f artifacts/datasets-compact-balanced-15k/selection_manifest.json
mkdir -p "$RUN" logs

"$EVAL_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
data = Path('artifacts/datasets-compact-balanced-15k/compact_balanced.json')
manifest_path = Path('artifacts/datasets-compact-balanced-15k/selection_manifest.json')
manifest = json.loads(manifest_path.read_text())
rows = json.loads(data.read_text())
assert len(rows) == len({x['instruction'] for x in rows}) == 15000
assert manifest['size'] == len(manifest['selected_ids']) == 15000
assert all(x.get('instruction') and x.get('output') and '<think>' not in x['output'].lower() for x in rows)
assert max(manifest['domain_counts'].values()) < 15000
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = {
    'schema': 'reasoning-sft-compact-balanced-full-v1',
    'training_rows': len(rows), 'data_sha256': sha(data),
    'selection_manifest_sha256': sha(manifest_path),
    'config_sha256': sha('configs/lmf-compact-balanced-15000.yaml'),
    'base_model': json.load(open('artifacts/model.json')),
}
Path('runs/lmf-compact-balanced-15000/provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
PY

if [[ ! -f "$RUN/trainer_state.json" ]]; then
  "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$CONFIG"
fi

test -f "$RUN/model.safetensors.index.json" -o -f "$RUN/model.safetensors"
"$EVAL_PYTHON" - <<'PY'
import json, math
state = json.load(open('runs/lmf-compact-balanced-15000/trainer_state.json'))
assert state['global_step'] == 1875
for row in state.get('log_history', []):
    for key in ('loss', 'grad_norm'):
        if key in row: assert math.isfinite(float(row[key])), (key, row[key])
json.dump({'global_step': 1875, 'finite_logs': True},
          open('runs/lmf-compact-balanced-15000/completion.json','w'), indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85

"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 --candidate "$RESULT" \
  --output results/lmf-compact-balanced-15000-suite-comparison.json

set +e
"$EVAL_PYTHON" -m reasoning.domination \
  results/lmf-compact-balanced-15000-suite-comparison.json \
  --output results/lmf-compact-balanced-15000-domination.json
GATE_STATUS=$?
set -e
[[ "$GATE_STATUS" -eq 0 || "$GATE_STATUS" -eq 2 ]]

