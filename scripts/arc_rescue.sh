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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-rescue-triton-${SLURM_JOB_ID:-local}"
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
RUN=runs/rescue-15000-capacity-route-dft
test -f artifacts/datasets-rescue-4k/selection_manifest.json
"$PYTHON" -m reasoning.train_adapter --model "$MODEL" \
  --data artifacts/datasets-rescue-4k/advanced.json --output "$RUN" \
  --objective dft --steps 1875 --accumulation 8 --cutoff 4096 --lr 1e-5 \
  --save-every 100 --keep-every 300 --resume
test -f "$RUN/complete.json"
"$PYTHON" - "$RUN" <<'PY'
import json,math,sys
from pathlib import Path
run=Path(sys.argv[1]); stats=json.load(open(run/'data_stats.json'))
assert stats['used']==15000 and stats['dropped_overlength']==0
rows=[json.loads(x) for x in open(run/'train.jsonl')]
assert len(rows)==1875 and all(math.isfinite(r[k]) for r in rows for k in ('objective_loss','grad_norm'))
PY
# A held-out gate measures both accuracy and stopping before public benchmarks.
for SPEC in "base:$MODEL" "candidate:$RUN/merged"; do
  NAME=${SPEC%%:*}; TARGET=${SPEC#*:}
  "$PYTHON" -m reasoning.evaluate --model "$TARGET" --data-dir artifacts/dev-rescue \
    --benchmarks acereason_dev --output "results/rescue-$NAME-dev" \
    --batch-size 1024 --max-num-seqs 128 --max-new-tokens 8192 \
    --record-hardware --gpu-memory-utilization .85
done
"$PYTHON" -m reasoning.compare --baseline results/rescue-base-dev \
  --candidate results/rescue-candidate-dev --output results/rescue-dev-comparison.json
"$PYTHON" - <<'PY'
import json
base=json.load(open('results/rescue-base-dev/metrics.json'))['acereason_dev']
cand=json.load(open('results/rescue-candidate-dev/metrics.json'))['acereason_dev']
if cand['pass_at_8'] < base['pass_at_8'] or cand['truncated_samples'] > base['truncated_samples'] + 40:
    raise SystemExit('Development gate rejected rescue model; public evaluation intentionally skipped')
PY
"$PYTHON" -m reasoning.evaluate --model "$RUN/merged" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output results/rescue-15000-capacity-route-dft-suite --batch-size 1024 \
  --max-num-seqs 128 --record-hardware --gpu-memory-utilization .85
"$PYTHON" -m reasoning.compare --baseline results/full-base-suite-v2 \
  --candidate results/rescue-15000-capacity-route-dft-suite \
  --output results/rescue-15000-capacity-route-dft-suite-comparison.json
