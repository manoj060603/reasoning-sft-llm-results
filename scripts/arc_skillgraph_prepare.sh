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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-skillgraph-${SLURM_JOB_ID:-local}"

PY="$PWD/.venv-arc/bin/python"
POOL=artifacts/prepared_extended/pool.json
FEATURES=artifacts/features_extended.json
TEACHER="$PWD/artifacts/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
BASE="$PWD/artifacts/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
LABELS=artifacts/skillgraph-asft-15k/skill_labels.jsonl
DATA=artifacts/datasets-skillgraph-asft-15k
AUDIT=artifacts/skillgraph-asft-15k/alternative_audit.json

test -x "$PY" && test -d "$TEACHER" && test -d "$BASE"
mkdir -p artifacts/skillgraph-asft-15k "$DATA" logs
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
for path, expected in [
 ('artifacts/prepared_extended/pool.json','c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9'),
 ('artifacts/features_extended.json','039cd63692d528dcce2caf2be2bfed85be917a66298a2532933ef1107f9db979')]:
 h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 assert h==expected,(path,h)
PY

# Four live teacher examples are an end-to-end schema preflight.  A failed
# structured decode stops the job before the 83,037-example generation pass.
"$PY" -m reasoning.skill_graph label --pool "$POOL" --teacher "$TEACHER" \
  --output "$LABELS" --batch-size 4 --limit 4
"$PY" -m reasoning.skill_graph label --pool "$POOL" --teacher "$TEACHER" \
  --output "$LABELS" --batch-size 64
"$PY" -m reasoning.skill_graph export --pool "$POOL" --features "$FEATURES" \
  --labels "$LABELS" --output "$DATA" --tokenizer "$BASE" --size 15000 --cutoff 8192

# Independently generated answer checks audit the hardest-composition selected
# subset.  They are receipt-only and cannot tune training or public scores.
"$PY" -m reasoning.skill_graph_verify --pool "$POOL" \
  --selection "$DATA/selection_manifest.json" --labels "$LABELS" \
  --teacher "$TEACHER" --base-tokenizer "$BASE" --output "$AUDIT" \
  --count 4 --batch-size 4 --cutoff 8192
"$PY" -m reasoning.skill_graph_verify --pool "$POOL" \
  --selection "$DATA/selection_manifest.json" --labels "$LABELS" \
  --teacher "$TEACHER" --base-tokenizer "$BASE" --output "$AUDIT" \
  --count 256 --batch-size 16 --cutoff 8192

"$PY" - <<'PY'
import json,math
from pathlib import Path
from reasoning.data import digest,read_rows
from reasoning.skill_graph import load_labels
pool=read_rows('artifacts/prepared_extended/pool.json')
labels=load_labels(Path('artifacts/skillgraph-asft-15k/skill_labels.jsonl'),[r['id'] for r in pool])
selection=json.load(open('artifacts/datasets-skillgraph-asft-15k/selection_manifest.json'))
data=read_rows('artifacts/datasets-skillgraph-asft-15k/advanced.json')
audit=read_rows('artifacts/skillgraph-asft-15k/alternative_audit.json')
ids=selection['selected_ids']
assert len(pool)==len(labels)==83037 and len({x['id'] for x in labels})==83037
assert len(ids)==len(set(ids))==len(data)==15000
assert max(selection['exact_chat_template_tokens'].values())<=8192
assert len(audit)==256 and all(math.isfinite(x['divergence']) for x in audit)
receipt={'schema':'reasoning-sft-skillgraph-asft-preparation-v1',
 'pool_rows':len(pool),'skill_labels':len(labels),'selected_ids':len(ids),
 'verified_alternative_answers':sum(x['answer_match'] for x in audit),
 'pool_sha256':digest('artifacts/prepared_extended/pool.json'),
 'features_sha256':digest('artifacts/features_extended.json'),
 'labels_sha256':digest('artifacts/skillgraph-asft-15k/skill_labels.jsonl'),
 'data_sha256':digest('artifacts/datasets-skillgraph-asft-15k/advanced.json'),
 'selection_sha256':digest('artifacts/datasets-skillgraph-asft-15k/selection_manifest.json'),
 'audit_sha256':digest('artifacts/skillgraph-asft-15k/alternative_audit.json')}
Path('artifacts/skillgraph-asft-15k/preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt,flush=True)
PY
