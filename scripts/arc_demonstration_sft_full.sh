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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-demonstration-full-${SLURM_JOB_ID:-local}"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
RUN=runs/lmf-demonstration-response-15000-8k
RESULT=results/lmf-demonstration-response-15000-8k-suite
test "${SLURM_JOB_PARTITION:-}" = b200_normal_q
nvidia-smi --query-gpu=name --format=csv,noheader | "$EVAL_PYTHON" -c 'import sys; names=sys.stdin.read(); assert "B200" in names,names'
mkdir -p "$RUN" logs
"$TRAIN_PYTHON" - <<'PY'
import json,yaml
from pathlib import Path
from reasoning.data import digest,write_json
receipt=json.load(open('artifacts/demonstration-response/preparation_receipt.json'))
data=Path('artifacts/datasets-demonstration-response-15k')
for key,name in [('data_sha256','advanced.json'),('selection_sha256','selection_manifest.json'),
                 ('audit_sha256','audit_receipt.json'),('preflight_sha256','training_preflight.json')]:
 assert digest(data/name)==receipt[key],name
assert receipt['examples']==15000 and receipt['updates']==1875
assert json.load(open(data/'audit_receipt.json'))['passed']
config=yaml.safe_load(open('configs/lmf-demonstration-response-15000-8k.yaml'))
assert config['disable_shuffling'] and config['max_steps']==1875 and not config.get('packing')
run=Path(config['output_dir']); checkpoints=[]
for path in run.glob('checkpoint-*'):
 try:
  state=json.load(open(path/'trainer_state.json')); step=state['global_step']
  if 0<step<1875 and (path/'optimizer.pt').is_file() and (path/'rng_state.pth').is_file():
   checkpoints.append((step,path))
 except (ValueError,OSError,KeyError): pass
if checkpoints:
 config['resume_from_checkpoint']=str(max(checkpoints)[1])
(run/'effective_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
write_json(run/'provenance.json',{'schema':'demonstration-response-v1',
 'preparation_receipt':receipt,'config_sha256':digest('configs/lmf-demonstration-response-15000-8k.yaml'),
 'base_model':json.load(open('artifacts/model.json')),'original_targets_unchanged':True,
 'checkpoint_selection':False,'public_scores_used_for_selection':False})
PY
if [[ ! -f "$RUN/trainer_state.json" ]]; then
  "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$RUN/effective_config.yaml"
fi
"$EVAL_PYTHON" - <<'PY'
import json,math
from pathlib import Path
from reasoning.data import write_json
run=Path('runs/lmf-demonstration-response-15000-8k')
state=json.load(open(run/'trainer_state.json')); assert state['global_step']==1875
logs=[x for x in state['log_history'] if 'loss' in x]
assert len(logs)==1875 and {x['step'] for x in logs}==set(range(1,1876))
assert all(math.isfinite(float(x[k])) for x in logs for k in ('loss','grad_norm'))
index=run/'model.safetensors.index.json'
if index.exists():
 shards=set(json.load(open(index))['weight_map'].values())
 assert shards and all((run/x).is_file() and (run/x).stat().st_size>0 for x in shards)
else: assert (run/'model.safetensors').is_file()
assert (run/'config.json').is_file() and (run/'tokenizer_config.json').is_file()
write_json(run/'completion.json',{'global_step':1875,'finite_updates':1875,'model_files_present':True})
PY
"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
  --data-dir artifacts/benchmarks_extended \
  --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
  --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
  --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.compare --baseline results/full-base-suite-v2 \
  --candidate "$RESULT" --output results/lmf-demonstration-response-15000-8k-suite-comparison.json
set +e
"$EVAL_PYTHON" -m reasoning.domination results/lmf-demonstration-response-15000-8k-suite-comparison.json \
  --output results/lmf-demonstration-response-15000-8k-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
