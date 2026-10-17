"""Provenance and recovery for the user-authorized two-A30 execution."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
from .data import digest,write_json

FILES=('configs/lmf-full-length-15k-16k-3epoch-2a30.yaml',
       'configs/ds-full-length-4a30.json','scripts/arc_full_length_2a30_train.sh',
       'scripts/train_memory_ce.py','reasoning/memory_ce.py','reasoning/a30_pair_runtime.py')


def valid_checkpoint(checkpoint):
    p=Path(checkpoint)
    try:
        state=json.loads((p/'trainer_state.json').read_text())
        step=state['global_step']
        shards=list((p/f'global_step{step}').glob('*optim_states.pt'))
        models=list((p/f'global_step{step}').glob('*model_states.pt'))
        return (0<step<5625 and bool(models) and all(x.stat().st_size>0 for x in models) and len(shards)==2 and all(x.stat().st_size>0 for x in shards)
                and all((p/f'rng_state_{rank}.pth').is_file() for rank in range(2)))
    except (KeyError,ValueError,OSError):
        return False


def register_allocation(run,cfg_path):
    receipt=json.load(open('artifacts/datasets-full-length-15k/a30_pair_execution_receipt.json'))
    hashes={name:digest(name) for name in FILES}
    assert hashes==receipt['execution_sha256'],'A30 execution files changed'
    assert receipt['preparation_sha256']==digest('artifacts/datasets-full-length-15k/preparation_receipt.json')
    versions={k:importlib.metadata.version(k) for k in ('torch','transformers','deepspeed')}
    assert versions==receipt['runtime_versions']
    ds=json.load(open('configs/ds-full-length-4a30.json'))
    assert ds['zero_optimization']['stage']==3
    assert ds['zero_optimization']['offload_optimizer']['device']=='cpu'
    assert ds['zero_optimization']['offload_param']['device']=='cpu'
    assert int(os.environ['SLURM_GPUS_ON_NODE'])==2
    path=run/'allocations.json'
    alloc=json.load(open(path)) if path.exists() else {'execution_sha256':hashes,'job_ids':[],
        'training_gpu_hours_cap':32,'evaluation_gpu_hours_reserved':8,'total_gpu_hours_cap':40,
        'allocation_gpu_count':2,'allocation_hours':2,'automatic_requeue':False}
    assert alloc['execution_sha256']==hashes
    job=os.environ['SLURM_JOB_ID']
    if job not in alloc['job_ids']:
        assert len(alloc['job_ids'])<8,'Eight 2-hour two-A30 chunks exhausted; preserve state and report budget blocker'
        alloc['job_ids'].append(job)
    write_json(path,alloc)


def verify_final(directory):
    run=Path(directory)
    state=json.load(open(run/'trainer_state.json'))
    assert state['global_step']==5625 and abs(state['epoch']-3.)<.001
    logs=[x for x in state['log_history'] if 'loss' in x]
    assert len(logs)==5625 and {x['step'] for x in logs}==set(range(1,5626))
    assert all(math.isfinite(float(x[k])) for x in logs for k in ('loss','grad_norm'))
    index=run/'model.safetensors.index.json'
    shards=set(json.load(open(index))['weight_map'].values()) if index.exists() else {'model.safetensors'}
    from safetensors import safe_open
    import torch
    for name in shards:
        path=run/name
        assert path.is_file() and path.stat().st_size>0
        with safe_open(str(path),framework='pt',device='cpu') as f:
            for key in f.keys():
                tensor=f.get_tensor(key)
                for chunk in tensor.reshape(-1).split(1000000):
                    assert torch.isfinite(chunk).all(),(name,key)
    assert (run/'config.json').is_file() and (run/'tokenizer_config.json').is_file()
    write_json(run/'completion.json',{'global_step':5625,'finite_updates':5625,'epochs':3,
       'unique_prompts':15000,'example_presentations':45000,'model_files_present':True,
       'all_model_tensors_finite':True,'training_world_size':2,'effective_batch_size':8})
