"""Launch preparation for the original full-length recipe on one B200.

Preserves the spent A30 budget. Never imports a two-rank optimizer checkpoint.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import yaml
from .data import digest, write_json

ROOT = Path('artifacts/datasets-full-length-15k')
CFG = Path('configs/lmf-full-length-early-b200.yaml')
FILES = (str(CFG), 'scripts/arc_full_length_early_b200.sh',
         'scripts/full_length_early_b200.slurm', 'reasoning/early_b200.py',
         'scripts/arc_full_length_early_b200_eval.sh', 'scripts/full_length_early_b200_eval.slurm',
         'scripts/train_memory_ce.py', 'reasoning/memory_ce.py')


def checkpoint_valid(p):
    try:
        s = json.load(open(p/'trainer_state.json'))
        index=p/'model.safetensors.index.json'
        shards=set(json.load(open(index))['weight_map'].values()) if index.exists() else {'model.safetensors'}
        return (bool(shards) and 0 < s['global_step'] <= 5625 and
                all((p/n).is_file() and (p/n).stat().st_size > 0
                    for n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', *shards)))
    except (OSError, ValueError, KeyError):
        return False


def validate():
    from .full_length import POOL_SHA, FEATURES_SHA
    from llamafactory.hparams import ModelArguments, FinetuningArguments
    from llamafactory.hparams.parser import _check_extra_dependencies
    c = yaml.safe_load(CFG.read_text())
    old = yaml.safe_load(Path('configs/lmf-full-length-15k-16k-3epoch.yaml').read_text())
    assert {k:v for k,v in c.items() if k not in ('output_dir','save_steps')} == {
        k:v for k,v in old.items() if k not in ('output_dir','save_steps')}
    assert c['save_steps'] == 50 and c['max_steps'] == 5625 and c['num_train_epochs'] == 3
    assert c['gradient_accumulation_steps'] == 8 and c['per_device_train_batch_size'] == 1
    assert 'deepspeed' not in c
    flags = {k:c[k] for k in ('use_dft_loss','use_trimsft_loss','use_prift_loss','use_gem_loss','use_asft_loss')}
    assert not any(flags.values())
    _check_extra_dependencies(ModelArguments(model_name_or_path=c['model_name_or_path'],flash_attn=c['flash_attn']),
        FinetuningArguments(finetuning_type='full',disable_shuffling=False,plot_loss=c['plot_loss'],**flags),
        SimpleNamespace(deepspeed=None,predict_with_generate=False))
    r = json.load(open(ROOT/'preparation_receipt.json'))
    m = json.load(open(ROOT/'selection_manifest.json'))
    assert r['unique_prompts']==15000 and len(set(m['selected_ids']))==15000
    assert r['updates']==5625 and r['epochs']==3 and r['example_presentations']==45000
    assert r['pool_sha256']==POOL_SHA and r['features_sha256']==FEATURES_SHA
    assert r['module_sha256']==digest('reasoning/full_length.py')
    assert r['allocation_module_sha256']==digest('reasoning/select.py')
    assert r['data_sha256']==m['data_sha256']==digest(ROOT/'advanced.json')
    assert r['dataset_info_sha256']==digest(ROOT/'dataset_info.json')
    assert all(r[k] for k in ('original_targets_complete','prompt_tokens_masked','eos_supervised',
        'actual_dataset_converter_verified','system_column_preserved'))
    assert r['model_manifest']==json.load(open('artifacts/model.json'))
    assert r['model_manifest']['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert r['max_exact_tokens']<=16384 and r['selected_above_8k']>0
    pair = json.load(open('runs/lmf-full-length-15k-16k-3epoch-2a30/allocations.json'))
    assert pair['job_ids']==['656067','656673'] and not pair.get('evaluation_job_ids')
    receipt = {'execution_sha256':{f:digest(f) for f in FILES},
        'preparation_sha256':digest(ROOT/'preparation_receipt.json'),
        'selection_sha256':digest(ROOT/'selection_manifest.json'),
        'runtime_versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','deepspeed')},
        'prior_training_job_ids':pair['job_ids'],'prior_charged_gpu_hours':8,
        'single_b200_training_allocation_hours':8,'total_gpu_hours_cap':40,
        'evaluation_gpu_hours_reserved':8,'original_base_restart':True,
        'two_rank_checkpoint_imported':False,'gpu_fit_and_throughput_unverified':True}
    return c, receipt


def register(c, receipt):
    assert int(os.environ['SLURM_GPUS_ON_NODE'])==1
    run = Path(c['output_dir']); run.mkdir(parents=True,exist_ok=True)
    path = run/'allocations.json'
    ledger = json.load(open(path)) if path.exists() else {
        'binding':receipt,'training_job_ids':[],'evaluation_job_ids':[]}
    assert ledger['binding']==receipt
    job = os.environ['SLURM_JOB_ID']
    if job not in ledger['training_job_ids']:
        assert 8 + 8*(len(ledger['training_job_ids'])+1) + 8*len(ledger['evaluation_job_ids']) + 8 <= 40
        ledger['training_job_ids'].append(job)
    write_json(path,ledger)
    checkpoints = [p for p in run.glob('checkpoint-*') if checkpoint_valid(p)]
    if checkpoints:
        c['resume_from_checkpoint']=str(max(checkpoints,key=lambda p:json.load(open(p/'trainer_state.json'))['global_step']))
    (run/'effective_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))


def verify_final(run):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    s=json.load(open(run/'trainer_state.json'))
    assert s['global_step']==5625 and abs(s['epoch']-3)<.001
    records=[r for r in s['log_history'] if 'loss' in r]
    assert len(records)==5625 and {r['step'] for r in records}==set(range(1,5626))
    assert all(math.isfinite(float(r[k])) for r in records for k in ('loss','grad_norm'))
    index=run/'model.safetensors.index.json'
    shards=set(json.load(open(index))['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for name in shards:
        with safe_open(str(run/name),framework='pt',device='cpu') as f:
            for key in f.keys():
                for chunk in f.get_tensor(key).reshape(-1).split(1000000):
                    assert torch.isfinite(chunk).all(),(name,key)
    assert all((run/n).is_file() for n in ('config.json','tokenizer_config.json'))
    write_json(run/'completion.json',{'global_step':5625,'epochs':3,'finite_updates':5625,
        'unique_prompts':15000,'example_presentations':45000,'all_model_tensors_finite':True,
        'training_world_size':1,'effective_batch_size':8})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--register',action='store_true');p.add_argument('--verify-final',action='store_true');a=p.parse_args()
    c,r=validate();path=ROOT/'early_b200_execution_receipt.json'
    if a.prepare:
        if path.exists(): assert json.load(open(path))==r
        else: write_json(path,r)
        print('CPU_NATIVE_GUARD_DATA_RECIPE_AND_BUDGET_PREFLIGHT_PASSED')
    else:
        assert json.load(open(path))==r
        if a.register: register(c,r)
        if a.verify_final: verify_final(Path(c['output_dir']))


if __name__=='__main__': main()
