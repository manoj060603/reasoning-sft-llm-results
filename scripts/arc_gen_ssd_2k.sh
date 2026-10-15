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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-gen-ssd-${SLURM_JOB_ID:-local}"

PYTHON="$PWD/.venv-arc/bin/python"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
BASE=$($PYTHON -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
mkdir -p artifacts/gen-ssd-2k runs results

TEACHER=$($PYTHON - <<'PY'
from huggingface_hub import model_info, snapshot_download
from pathlib import Path
import json
repo = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
commit = model_info(repo).sha
path = snapshot_download(repo, revision=commit)
Path('artifacts/gen-ssd-2k/teacher_provenance.json').write_text(
    json.dumps({'repo': repo, 'commit': commit, 'path': path}, indent=2) + '\n')
print(path)
PY
)

"$PYTHON" -m reasoning.gen_ssd \
  --input artifacts/datasets-rescue-4k/advanced.json \
  --teacher "$TEACHER" --student "$BASE" \
  --output artifacts/gen-ssd-2k \
  --target-size 2000 --max-source 4000 \
  --attempts 3 --batch-size 24 --score-batch-size 2 \
  --chunk-tokens 1024 --max-new-tokens 4096 \
  --first-candidates 4 --later-candidates 2 --training-cutoff 4096

for ARM in existing guided; do
  RUN="runs/gen-ssd-${ARM}-2k-full"
  CONFIG="configs/gen-ssd-${ARM}-2k-full.yaml"
  if [[ ! -f "$RUN/trainer_state.json" ]]; then
    "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$CONFIG"
  fi
  test -f "$RUN/model.safetensors.index.json" -o -f "$RUN/model.safetensors"
  "$PYTHON" - "$RUN" <<'PY'
import json, math, sys
run=sys.argv[1]
state=json.load(open(f'{run}/trainer_state.json'))
assert state['global_step']==250, state['global_step']
for row in state.get('log_history', []):
    for key in ('loss','grad_norm'):
        if key in row: assert math.isfinite(float(row[key])), (key,row[key])
json.dump({'global_step':250,'finite_logs':True},open(f'{run}/completion.json','w'),indent=2)
PY
done

evaluate () {
  local NAME=$1 MODEL=$2 DATA=$3 OUT=$4
  "$PYTHON" -m reasoning.evaluate --model "$MODEL" \
    --data-dir "$DATA" --benchmarks acereason_dev --output "$OUT" \
    --batch-size 1024 --max-num-seqs 128 --max-new-tokens 32768 \
    --record-hardware --gpu-memory-utilization .85
}

evaluate base "$BASE" artifacts/dev-rescue results/gen-ssd-base-dev32k
evaluate existing "$PWD/runs/gen-ssd-existing-2k-full" artifacts/dev-rescue results/gen-ssd-existing-dev32k
evaluate guided "$PWD/runs/gen-ssd-guided-2k-full" artifacts/dev-rescue results/gen-ssd-guided-dev32k

"$PYTHON" -m reasoning.choose_gen_ssd \
  --base results/gen-ssd-base-dev32k \
  --existing results/gen-ssd-existing-dev32k \
  --guided results/gen-ssd-guided-dev32k \
  --output results/gen-ssd-2k-selection.json
"$PYTHON" -m reasoning.compare \
  --baseline results/gen-ssd-existing-dev32k \
  --candidate results/gen-ssd-guided-dev32k \
  --output results/gen-ssd-2k-dev-comparison.json

SELECTED=$($PYTHON -c 'import json;print(json.load(open("results/gen-ssd-2k-selection.json"))["selected"] or "")')
if [[ "$SELECTED" == guided ]]; then
  evaluate existing-confirm "$PWD/runs/gen-ssd-existing-2k-full" artifacts/dev-concise-confirm results/gen-ssd-existing-confirm32k
  evaluate guided-confirm "$PWD/runs/gen-ssd-guided-2k-full" artifacts/dev-concise-confirm results/gen-ssd-guided-confirm32k
  "$PYTHON" -m reasoning.compare \
    --baseline results/gen-ssd-existing-confirm32k \
    --candidate results/gen-ssd-guided-confirm32k \
    --output results/gen-ssd-2k-confirm-comparison.json
else
  echo "Student-guided arm did not pass the preregistered development gate; confirmation skipped."
fi
