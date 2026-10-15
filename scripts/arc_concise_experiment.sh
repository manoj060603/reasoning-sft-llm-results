#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-concise-triton-${SLURM_JOB_ID:-local}"
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')

test -f artifacts/datasets-concise-2k/selection_manifest.json
test -f artifacts/dev-concise-confirm/acereason_dev/test.jsonl
"$PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('.')
manifest=json.load(open('artifacts/datasets-concise-2k/selection_manifest.json'))
paths={
 'protocol':root/'CONCISE_TARGET_EXPERIMENT.md',
 'selector':root/'reasoning/concise.py',
 'trainer':root/'reasoning/train_adapter.py',
 'evaluator':root/'reasoning/evaluate.py',
 'finalizer':root/'reasoning/choose_concise.py',
 'runner':root/'scripts/arc_concise_experiment.sh',
 'launcher':root/'scripts/concise_experiment.slurm',
}
for name,path in paths.items():
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==manifest['source_sha256'][name],(name,actual,manifest['source_sha256'][name])
actual=hashlib.sha256((root/'artifacts/datasets-concise-2k/concise.json').read_bytes()).hexdigest()
assert actual==manifest['data_sha256']
assert manifest['selected_rows']==2000 and manifest['complete_box_rows']==2000
PY

# Prespecified early/middle/late checkpoints; the final checkpoint is already evaluated.
for STEP in 000300 000900 001500; do
  ADAPTER="runs/rescue-15000-capacity-route-dft/milestones/step-${STEP}/adapter"
  OUTPUT="results/rescue-milestone-step-${STEP}-dev"
  "$PYTHON" -m reasoning.evaluate --model "$MODEL" --lora-adapter "$ADAPTER" \
    --data-dir artifacts/dev-rescue --benchmarks acereason_dev --output "$OUTPUT" \
    --batch-size 1024 --max-num-seqs 128 --max-new-tokens 8192 \
    --record-hardware --gpu-memory-utilization .85
  "$PYTHON" -m reasoning.compare --baseline results/rescue-base-dev \
    --candidate "$OUTPUT" --output "results/rescue-milestone-step-${STEP}-comparison.json"
done

for OBJECTIVE in sft dft; do
  RUN="runs/concise-2000-${OBJECTIVE}"
  "$PYTHON" -m reasoning.train_adapter --model "$MODEL" \
    --data artifacts/datasets-concise-2k/concise.json --output "$RUN" \
    --objective "$OBJECTIVE" --steps 250 --accumulation 8 --cutoff 4096 --lr 1e-5 \
    --save-every 50 --keep-every 50 --resume
  test -f "$RUN/complete.json"
  "$PYTHON" -m reasoning.evaluate --model "$RUN/merged" \
    --data-dir artifacts/dev-rescue --benchmarks acereason_dev \
    --output "results/concise-2000-${OBJECTIVE}-dev" \
    --batch-size 1024 --max-num-seqs 128 --max-new-tokens 8192 \
    --record-hardware --gpu-memory-utilization .85
  "$PYTHON" -m reasoning.compare --baseline results/rescue-base-dev \
    --candidate "results/concise-2000-${OBJECTIVE}-dev" \
    --output "results/concise-2000-${OBJECTIVE}-comparison.json"
done

"$PYTHON" -m reasoning.choose_concise \
  --base results/rescue-base-dev \
  --sft results/concise-2000-sft-dev \
  --dft results/concise-2000-dft-dev \
  --output results/concise-2000-selection.json

SELECTED=$("$PYTHON" -c 'import json;print(json.load(open("results/concise-2000-selection.json"))["selected"] or "")')
if [[ -z "$SELECTED" ]]; then
  echo "No concise-target arm passed the development gate; fresh confirmation intentionally skipped"
  exit 0
fi

"$PYTHON" -m reasoning.evaluate --model "$MODEL" \
  --data-dir artifacts/dev-concise-confirm --benchmarks acereason_dev \
  --output results/concise-confirm-base --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 8192 --record-hardware --gpu-memory-utilization .85
"$PYTHON" -m reasoning.evaluate --model "runs/concise-2000-${SELECTED}/merged" \
  --data-dir artifacts/dev-concise-confirm --benchmarks acereason_dev \
  --output "results/concise-confirm-${SELECTED}" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 8192 --record-hardware --gpu-memory-utilization .85
"$PYTHON" -m reasoning.compare --baseline results/concise-confirm-base \
  --candidate "results/concise-confirm-${SELECTED}" \
  --output results/concise-confirm-comparison.json
