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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-lmf-control-${SLURM_JOB_ID:-local}"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$($EVAL_PYTHON -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
CONFIG=configs/lmf-concise-lora-a.yaml
RUN=runs/lmf-concise-2000-lora-a

test -x "$TRAIN_PYTHON"
test -f artifacts/datasets-concise-2k/concise.json
test -f artifacts/datasets-concise-2k/dataset_info.json
$EVAL_PYTHON - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
selection = json.load(open('artifacts/datasets-concise-2k/selection_manifest.json'))
data = root/'artifacts/datasets-concise-2k/concise.json'
assert hashlib.sha256(data.read_bytes()).hexdigest() == selection['data_sha256']
assert selection['selected_rows'] == 2000 and selection['complete_box_rows'] == 2000
PY

mkdir -p "$RUN"
$TRAIN_PYTHON - <<'PY' > "$RUN/environment.txt"
from llamafactory.extras.env import print_env
print_env()
PY
$EVAL_PYTHON - <<'PY'
import hashlib, json
from pathlib import Path
root=Path('.')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=[]
for p in sorted((root/'LLaMA-Factory/src').rglob('*.py')):
    source.append((str(p.relative_to(root)),sha(p)))
receipt={
  'schema':'reasoning-sft-lmf-control-a-v1',
  'data_sha256':sha('artifacts/datasets-concise-2k/concise.json'),
  'dataset_info_sha256':sha('artifacts/datasets-concise-2k/dataset_info.json'),
  'config_sha256':sha('configs/lmf-concise-lora-a.yaml'),
  'llamafactory_source_tree':hashlib.sha256(json.dumps(source,separators=(',',':')).encode()).hexdigest(),
  'base_model':json.load(open('artifacts/model.json')),
}
(root/'runs/lmf-concise-2000-lora-a/provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY

if [[ ! -f "$RUN/trainer_state.json" ]]; then
  "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$CONFIG"
fi

for STEP in 50 100 250; do
  ADAPTER="$RUN/checkpoint-${STEP}"
  OUTPUT="results/lmf-concise-lora-a-step-${STEP}-dev8k"
  test -f "$ADAPTER/adapter_model.safetensors"
  "$EVAL_PYTHON" -m reasoning.evaluate --model "$MODEL" --lora-adapter "$ADAPTER" \
    --data-dir artifacts/dev-rescue --benchmarks acereason_dev --output "$OUTPUT" \
    --batch-size 1024 --max-num-seqs 128 --max-new-tokens 8192 \
    --record-hardware --gpu-memory-utilization .85
  "$EVAL_PYTHON" -m reasoning.compare --baseline results/rescue-base-dev \
    --candidate "$OUTPUT" --output "results/lmf-concise-lora-a-step-${STEP}-comparison8k.json"
done

"$EVAL_PYTHON" -m reasoning.choose_lmf_control \
  --base results/rescue-base-dev \
  --candidate step-50=results/lmf-concise-lora-a-step-50-dev8k \
  --candidate step-100=results/lmf-concise-lora-a-step-100-dev8k \
  --candidate step-250=results/lmf-concise-lora-a-step-250-dev8k \
  --output results/lmf-concise-lora-a-selection8k.json

SELECTED=$($EVAL_PYTHON -c 'import json;print(json.load(open("results/lmf-concise-lora-a-selection8k.json"))["selected"] or "")')
if [[ -z "$SELECTED" ]]; then
  echo "No LLaMA-Factory checkpoint passed the 8K termination screen; 32K evaluation skipped"
  exit 0
fi
STEP=${SELECTED#step-}
"$EVAL_PYTHON" -m reasoning.evaluate --model "$MODEL" \
  --data-dir artifacts/dev-rescue --benchmarks acereason_dev \
  --output results/lmf-control-base-dev32k --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.evaluate --model "$MODEL" --lora-adapter "$RUN/checkpoint-${STEP}" \
  --data-dir artifacts/dev-rescue --benchmarks acereason_dev \
  --output results/lmf-concise-lora-a-finalist-dev32k --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.compare --baseline results/lmf-control-base-dev32k \
  --candidate results/lmf-concise-lora-a-finalist-dev32k \
  --output results/lmf-concise-lora-a-finalist-comparison32k.json
