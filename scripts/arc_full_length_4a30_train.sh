#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-full-length-${SLURM_JOB_ID:-local}"
TRAIN_PYTHON="$PWD/.venv-lmf/bin/python"
EVAL_PYTHON="$PWD/.venv-arc/bin/python"
RUN=runs/lmf-full-length-15k-16k-3epoch-4a30
[[ "${SLURM_JOB_PARTITION:-}" = a30_normal_q || "${SLURM_JOB_PARTITION:-}" = a30_preemptable_q ]]
nvidia-smi --query-gpu=name --format=csv,noheader | "$EVAL_PYTHON" -c 'import sys; names=sys.stdin.read(); assert len(names.splitlines())==4 and all("A30" in x for x in names.splitlines()),names'
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
cfg_path=Path('configs/lmf-full-length-15k-16k-3epoch-4a30.yaml')
cfg=yaml.safe_load(cfg_path.read_text())
assert cfg['max_steps']==5625 and cfg['num_train_epochs']==3 and cfg['cutoff_len']==16384
assert cfg['learning_rate']==5e-6 and cfg['gradient_accumulation_steps']==2 and cfg['per_device_train_batch_size']==1
assert not cfg['disable_shuffling'] and not cfg['packing'] and not cfg['train_on_prompt']
from llamafactory.hparams import FinetuningArguments
loss_flags={k:cfg[k] for k in ('use_dft_loss','use_trimsft_loss','use_prift_loss','use_gem_loss','use_asft_loss')}
assert not any(loss_flags.values())
FinetuningArguments(finetuning_type='full',disable_shuffling=False,**loss_flags)
run=Path(cfg['output_dir']);run.mkdir(parents=True,exist_ok=True)
from reasoning.a30_runtime import register_allocation,valid_checkpoint
register_allocation(run,cfg_path)
flag=run/'skip_training';flag.unlink(missing_ok=True)
state_path=run/'trainer_state.json'
if state_path.exists() and json.load(open(state_path))['global_step']==5625:
 flag.write_text('Final step saved; full verification follows before evaluation.\n')
else:
 checkpoints=[]
 for checkpoint in run.glob('checkpoint-*'):
  try:
   state=json.load(open(checkpoint/'trainer_state.json')); step=state['global_step']
   if 0<step<5625 and valid_checkpoint(checkpoint):
    checkpoints.append((step,checkpoint))
  except (ValueError,OSError,KeyError):pass
 if checkpoints:cfg['resume_from_checkpoint']=str(max(checkpoints)[1])
(run/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
write_json(run/'provenance.json',{'schema':'full-length-original-15k-16k-three-epoch-4a30-v1',
 'preparation_receipt':r,'config_sha256':digest(cfg_path),'original_targets_unchanged':True,
 'serialized_tokens_per_epoch':sum(x['exact_tokens'] for x in m['selected']),
 'total_serialized_tokens':3*sum(x['exact_tokens'] for x in m['selected']),
 'total_supervised_tokens':3*r['supervised_tokens_per_epoch'],
 'checkpoint_selection':False,'public_scores_used_for_selection':False})
PYCHECK
if [[ ! -f "$RUN/skip_training" ]]; then
 "$TRAIN_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
   scripts/train_memory_ce.py "$RUN/effective_config.yaml"
fi
"$EVAL_PYTHON" - <<'PYFINAL'
from reasoning.a30_runtime import verify_final
verify_final('runs/lmf-full-length-15k-16k-3epoch-4a30')
PYFINAL
# Evaluation is a separate single-B200 allocation after this job terminates.
