"""Two-B200 execution of the frozen full SFT recipe; carries prior GPU spending."""
import argparse
import json
import os
from pathlib import Path
import yaml
from .data import digest, write_json
from .early_b200 import validate as validate_base, verify_final as verify_base

ROOT = Path('artifacts/datasets-full-length-15k')
CFG = Path('configs/lmf-full-length-fast-2b200.yaml')
FILES = (str(CFG), 'reasoning/fast_2b200.py',
         'scripts/full_length_fast_2b200.slurm', 'scripts/arc_full_length_fast_2b200.sh',
         'scripts/full_length_fast_2b200_eval.slurm', 'scripts/arc_full_length_fast_2b200_eval.sh')


def checkpoint_valid(p):
    try:
        s = json.load(open(p/'trainer_state.json'))
        index = p/'model.safetensors.index.json'
        shards = set(json.load(open(index))['weight_map'].values()) if index.exists() else {'model.safetensors'}
        return bool(shards) and 0 < s['global_step'] <= 5625 and all(
            (p/n).is_file() and (p/n).stat().st_size > 0 for n in
            ('optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth', *shards))
    except (OSError, ValueError, KeyError):
        return False


def validate():
    base, prior = validate_base()
    # Prevent separate single-B200 allocations from being omitted from the ledger.
    old = Path(base['output_dir'])/'allocations.json'
    if old.exists():
        a = json.load(open(old))
        assert not a['training_job_ids'] and not a['evaluation_job_ids']
    c = yaml.safe_load(CFG.read_text())
    allowed = {'output_dir', 'gradient_accumulation_steps', 'average_tokens_across_devices'}
    assert {k:v for k,v in c.items() if k not in allowed} == {k:v for k,v in base.items() if k not in allowed}
    assert c['gradient_accumulation_steps'] == 4 and c['average_tokens_across_devices'] is True
    assert c['per_device_train_batch_size'] * 2 * c['gradient_accumulation_steps'] == 8
    receipt = dict(prior)
    receipt['execution_sha256'] = {**prior['execution_sha256'], **{f:digest(f) for f in FILES}}
    receipt.pop('single_b200_training_allocation_hours')
    receipt.update(training_world_size=2, training_allocation_hours=6,
                   training_allocation_gpu_hours=12, global_token_denominator=True,
                   cpu_optimizer_offload=False, original_base_restart=True,
                   distributed_order_not_bitwise_equivalent=True)
    return c, receipt


def register(c, receipt):
    assert int(os.environ['SLURM_GPUS_ON_NODE']) == 2
    run = Path(c['output_dir']); run.mkdir(parents=True, exist_ok=True)
    path = run/'allocations.json'
    a = json.load(open(path)) if path.exists() else {'binding':receipt, 'training_job_ids':[], 'evaluation_job_ids':[]}
    assert a['binding'] == receipt
    job = os.environ['SLURM_JOB_ID']
    if job not in a['training_job_ids']:
        assert 8 + 12*(len(a['training_job_ids'])+1) + 8*len(a['evaluation_job_ids']) + 8 <= 40
        a['training_job_ids'].append(job)
    write_json(path, a)
    checkpoints = [p for p in run.glob('checkpoint-*') if checkpoint_valid(p)]
    if checkpoints:
        c['resume_from_checkpoint'] = str(max(checkpoints, key=lambda p:json.load(open(p/'trainer_state.json'))['global_step']))
    else:
        assert not list(run.glob('checkpoint-*')), 'Incomplete recovery state; preserve and inspect before restarting'
    (run/'effective_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))


def verify_final(run):
    verify_base(run)
    p = run/'completion.json'
    a = json.load(open(p)); a['training_world_size'] = 2
    write_json(p, a)


def main():
    p = argparse.ArgumentParser()
    for name in ('prepare', 'register', 'verify-final'): p.add_argument('--'+name, action='store_true')
    args = p.parse_args(); c, r = validate()
    path = ROOT/'fast_2b200_execution_receipt.json'
    if args.prepare:
        if path.exists(): assert json.load(open(path)) == r
        else: write_json(path, r)
        print('TWO_B200_CPU_NATIVE_DATA_RECIPE_AND_BUDGET_PREFLIGHT_PASSED')
    else:
        assert json.load(open(path)) == r
        if args.register: register(c, r)
        if args.verify_final: verify_final(Path(c['output_dir']))


if __name__ == '__main__': main()
