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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-skillgraph-asft-${SLURM_JOB_ID:-local}"

TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
CONFIG=configs/lmf-skillgraph-asft-15000-8k.yaml
RUN=runs/lmf-skillgraph-asft-15000-8k
RESULT=results/lmf-skillgraph-asft-15000-8k-suite
test -x "$TRAIN_PYTHON" && test -x "$EVAL_PYTHON"
test -f artifacts/skillgraph-asft-15k/preparation_receipt.json
mkdir -p "$RUN" logs

"$EVAL_PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
from reasoning.data import digest,read_rows
receipt=json.load(open('artifacts/skillgraph-asft-15k/preparation_receipt.json'))
assert receipt['pool_rows']==receipt['skill_labels']==83037
assert receipt['selected_ids']==15000
for key,path in [
 ('pool_sha256','artifacts/prepared_extended/pool.json'),
 ('features_sha256','artifacts/features_extended.json'),
 ('labels_sha256','artifacts/skillgraph-asft-15k/skill_labels.jsonl'),
 ('data_sha256','artifacts/datasets-skillgraph-asft-15k/advanced.json'),
 ('selection_sha256','artifacts/datasets-skillgraph-asft-15k/selection_manifest.json'),
 ('audit_sha256','artifacts/skillgraph-asft-15k/alternative_audit.json')]:
 assert digest(path)==receipt[key],path
selection=json.load(open('artifacts/datasets-skillgraph-asft-15k/selection_manifest.json'))
assert len(selection['selected_ids'])==len(set(selection['selected_ids']))==15000
assert max(selection['exact_chat_template_tokens'].values())<=8192
sources=[]
for directory in ('LLaMA-Factory/src','reasoning'):
 for path in sorted(Path(directory).rglob('*.py')):
  sources.append((str(path),digest(path)))
provenance={'schema':'reasoning-sft-skillgraph-asft-15000-8k-v1',
 'literature':['https://arxiv.org/abs/2503.14917','https://arxiv.org/abs/2509.23753'],
 'source_tree_sha256':hashlib.sha256(json.dumps(sources,separators=(',',':')).encode()).hexdigest(),
 'config_sha256':digest('configs/lmf-skillgraph-asft-15000-8k.yaml'),
 'preparation_receipt':receipt,'public_benchmark_data_used_for_selection':False,
 'base_model':json.load(open('artifacts/model.json'))}
Path('runs/lmf-skillgraph-asft-15000-8k/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
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
import json,math
state=json.load(open('runs/lmf-skillgraph-asft-15000-8k/trainer_state.json'))
assert state['global_step']==1875,state['global_step']
logs=state.get('log_history',[])
for row in logs:
 for key in ('loss','grad_norm'):
  if key in row: assert math.isfinite(float(row[key])),(key,row[key])
json.dump({'global_step':1875,'finite_logs':True,'logged_rows':len(logs)},
          open('runs/lmf-skillgraph-asft-15000-8k/completion.json','w'),indent=2)
PY

"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.compare \
  --baseline results/full-base-suite-v2 --candidate "$RESULT" \
  --output results/lmf-skillgraph-asft-15000-8k-suite-comparison.json
set +e
"$EVAL_PYTHON" -m reasoning.domination \
  results/lmf-skillgraph-asft-15000-8k-suite-comparison.json \
  --output results/lmf-skillgraph-asft-15000-8k-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
