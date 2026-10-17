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
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-full-length-${SLURM_JOB_ID:-local}"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
RUN=runs/lmf-full-length-15k-16k-3epoch
RESULT=results/lmf-full-length-15k-16k-3epoch-suite
test "${SLURM_JOB_PARTITION:-}" = b200_normal_q
nvidia-smi --query-gpu=name --format=csv,noheader | "$EVAL_PYTHON" -c 'import sys; names=sys.stdin.read(); assert "B200" in names,names'
mkdir -p "$RUN" logs
"$TRAIN_PYTHON" - <<'PYCHECK'
import json,os,yaml,importlib.metadata
from pathlib import Path
from reasoning.data import digest,write_json
from reasoning.full_length import POOL_SHA,FEATURES_SHA
path=Path('artifacts/datasets-full-length-15k')
r=json.load(open(path/'preparation_receipt.json'))
m=json.load(open(path/'selection_manifest.json'))
assert r['unique_prompts']==15000 and len(set(m['selected_ids']))==15000
assert r['updates']==5625 and r['epochs']==3 and r['example_presentations']==45000
assert r['selected_above_8k']>0 and r['max_exact_tokens']<=16384
assert r['pool_sha256']==POOL_SHA and r['features_sha256']==FEATURES_SHA
assert r['module_sha256']==digest('reasoning/full_length.py')
assert r['allocation_module_sha256']==digest('reasoning/select.py')
assert r['original_targets_complete'] and r['prompt_tokens_masked'] and r['eos_supervised']
assert r['actual_dataset_converter_verified'] and r['system_column_preserved']
assert r['dataset_info_sha256']==digest(path/'dataset_info.json')
assert r['data_sha256']==digest(path/'advanced.json')==m['data_sha256']
assert r['model_manifest']==json.load(open('artifacts/model.json'))
assert r['model_manifest']['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
cfg_path=Path('configs/lmf-full-length-15k-16k-3epoch.yaml')
cfg=yaml.safe_load(cfg_path.read_text())
assert cfg['max_steps']==5625 and cfg['num_train_epochs']==3 and cfg['cutoff_len']==16384
assert cfg['learning_rate']==5e-6 and cfg['gradient_accumulation_steps']==8 and cfg['per_device_train_batch_size']==1
assert not cfg['disable_shuffling'] and not cfg['packing'] and not cfg['train_on_prompt']
from llamafactory.hparams import FinetuningArguments
loss_flags={k:cfg[k] for k in ('use_dft_loss','use_trimsft_loss','use_prift_loss','use_gem_loss','use_asft_loss')}
assert not any(loss_flags.values())
FinetuningArguments(finetuning_type='full',disable_shuffling=False,**loss_flags)
run=Path(cfg['output_dir']);run.mkdir(parents=True,exist_ok=True)
alloc_path=run/'allocations.json'
binding={'config_sha256':digest(cfg_path),'preparation_sha256':digest(path/'preparation_receipt.json'),
         'driver_sha256':digest('scripts/arc_full_length_full.sh'),
         'native_data_source_sha256':{name:digest('LLaMA-Factory/src/llamafactory/data/'+name) for name in ('parser.py','converter.py','template.py','processor/supervised.py')},
         'runtime_versions':{k:importlib.metadata.version(k) for k in ('torch','transformers')}}
alloc=json.load(open(alloc_path)) if alloc_path.exists() else {**binding,'job_ids':[]}
assert all(alloc[k]==v for k,v in binding.items()),'Frozen training configuration/runtime changed'
job=os.environ['SLURM_JOB_ID']
if job not in alloc['job_ids']:
 assert len(alloc['job_ids'])<5,'Frozen full-run budget exhausted (five eight-hour B200 allocations)'
 alloc['job_ids'].append(job)
write_json(alloc_path,alloc)
flag=run/'skip_training';flag.unlink(missing_ok=True)
state_path=run/'trainer_state.json'
if state_path.exists() and json.load(open(state_path))['global_step']==5625:
 flag.write_text('Final step saved; full verification follows before evaluation.\n')
else:
 checkpoints=[]
 for checkpoint in run.glob('checkpoint-*'):
  try:
   state=json.load(open(checkpoint/'trainer_state.json')); step=state['global_step']
   if 0<step<5625 and (checkpoint/'optimizer.pt').is_file() and (checkpoint/'rng_state.pth').is_file():
    checkpoints.append((step,checkpoint))
  except (ValueError,OSError,KeyError):pass
 if checkpoints:cfg['resume_from_checkpoint']=str(max(checkpoints)[1])
(run/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
write_json(run/'provenance.json',{'schema':'full-length-original-15k-16k-three-epoch-v1',
 'preparation_receipt':r,'config_sha256':digest(cfg_path),'original_targets_unchanged':True,
 'serialized_tokens_per_epoch':sum(x['exact_tokens'] for x in m['selected']),
 'total_serialized_tokens':3*sum(x['exact_tokens'] for x in m['selected']),
 'total_supervised_tokens':3*r['supervised_tokens_per_epoch'],
 'checkpoint_selection':False,'public_scores_used_for_selection':False})
PYCHECK
if [[ ! -f "$RUN/skip_training" ]]; then
 "$TRAIN_PYTHON" LLaMA-Factory/src/train.py "$RUN/effective_config.yaml"
fi
"$EVAL_PYTHON" - <<'PYFINAL'
import json,math
from pathlib import Path
from reasoning.data import write_json
run=Path('runs/lmf-full-length-15k-16k-3epoch')
state=json.load(open(run/'trainer_state.json'))
assert state['global_step']==5625 and abs(state['epoch']-3.)<.001
logs=[x for x in state['log_history'] if 'loss' in x]
assert len(logs)==5625 and {x['step'] for x in logs}==set(range(1,5626))
assert all(math.isfinite(float(x[k])) for x in logs for k in ('loss','grad_norm'))
index=run/'model.safetensors.index.json'
if index.exists():
 shards=set(json.load(open(index))['weight_map'].values())
 assert shards and all((run/x).is_file() and (run/x).stat().st_size>0 for x in shards)
else:assert (run/'model.safetensors').is_file()
assert (run/'config.json').is_file() and (run/'tokenizer_config.json').is_file()
write_json(run/'completion.json',{'global_step':5625,'finite_updates':5625,'epochs':3,
 'unique_prompts':15000,'example_presentations':45000,'model_files_present':True})
PYFINAL
"$EVAL_PYTHON" -m reasoning.evaluate --model "$RUN" \
 --data-dir artifacts/benchmarks_extended \
 --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
 --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
 --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$EVAL_PYTHON" -m reasoning.compare --baseline results/full-base-suite-v2 \
 --candidate "$RESULT" --output results/lmf-full-length-15k-16k-3epoch-suite-comparison.json
set +e
"$EVAL_PYTHON" -m reasoning.domination results/lmf-full-length-15k-16k-3epoch-suite-comparison.json \
 --output results/lmf-full-length-15k-16k-3epoch-domination.json
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
