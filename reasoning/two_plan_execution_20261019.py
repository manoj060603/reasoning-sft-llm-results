"""Guarded execution for two NEW candidates. No budget authorization is bundled.

CPU --prepare binds audited data and drafts. Runtime registration is impossible
without explicit additional GPU budget and overlap authorization. Existing 40
GPU-hours remain charged. Explicit submission modes journal and reserve actual scheduler IDs; no automatic retry or requeue.
"""
import argparse
import fcntl
import json
import math
import os
import uuid
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import yaml
from .data import digest, write_json

ROOT = Path('/projects/tealab/rohit/reasoning-sft-20261011')
NEW = Path('artifacts/two-plans-20261019')
AUTH = NEW / 'additional_budget_authorization.json'
PLANS = {'assistance_coverage': {
    'config': 'configs/lmf-assistance-coverage-20261019.draft.yaml',
    'receipt': 'selection_manifest.json', 'data': 'advanced.json', 'rows': 15000,
    'epochs': 3, 'updates': 5625, 'world_size': 4, 'allocation_hours': 6},
    'stage2': {'config': 'configs/stage2-hard3k-20261019.draft.yaml',
              'receipt': 'stage2_preparation.json', 'data': 'stage2_hard3k.json',
              'rows': 2997, 'epochs': 2, 'updates': 750, 'world_size': 2, 'allocation_hours': 6}}
SCRIPT_FILES = ('reasoning/two_plan_execution_20261019.py',
                'scripts/two_plan_20261019_train.slurm', 'scripts/two_plan_20261019_train.sh',
                'scripts/two_plan_20261019_eval.slurm', 'scripts/two_plan_20261019_eval.sh',
                'scripts/train_memory_ce.py', 'reasoning/memory_ce.py',
                'reasoning/evaluate.py', 'reasoning/metrics.py', 'reasoning/compare.py',
                'reasoning/domination.py', 'reasoning/data.py', 'reasoning/full_length.py',
                'reasoning/select.py', 'eval/utils/grader.py', 'eval/utils/parser.py',
                'eval/utils/math_normalization.py')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def require(ok, why):
    if not ok: raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def authorization(path):
    a = read(path)  # File must be supplied by user-authorized process; never invented.
    extra = a.get('additional_gpu_hours')
    require(type(extra) in (int, float) and math.isfinite(extra) and extra > 0,
            'explicit numeric additional_gpu_hours required')
    require(a.get('prior_committed_gpu_hours') == 40, 'must preserve all 40 prior committed GPU-hours')
    require(a.get('allow_overlap') is True and set(a.get('plans', [])) == set(PLANS),
            'explicit overlap authorization for both named plans required')
    require(isinstance(a.get('authorization_evidence'), str) and a['authorization_evidence'].strip(),
            'authorization evidence required; documents are not budget approval')
    other = a.get('other_committed_gpu_hours')
    require(type(other) in (int, float) and math.isfinite(other) and other >= 0,
            'explicit other_committed_gpu_hours needed for dev/convergence/generation costs')
    return a


def validate_ledger(ledger):
    require(ledger.get('schema') == 'two-plan-ledger-v1' and ledger.get('prior_committed_gpu_hours') == 40,
            'invalid central ledger schema or reset prior budget')
    require(isinstance(ledger.get('jobs'), list), 'invalid central ledger jobs')
    seen = set()
    for job in ledger['jobs']:
        require(job.get('plan') in PLANS and job.get('phase') in ('train', 'eval')
                and isinstance(job.get('job_id'), str) and job['job_id'].isdigit()
                and job['job_id'] not in seen, 'unknown or duplicate ledger plan/phase/job')
        require(job.get('charged_gpu_hours') == allocation_charge(job['plan'], job['phase']),
                'ledger charge cannot be reduced')
        seen.add(job['job_id'])


def allocation_charge(plan, phase):
    return PLANS[plan]['world_size'] * PLANS[plan]['allocation_hours'] if phase == 'train' else 8


def projected_total(ledger, plan, phase, job):
    jobs = ledger.get('jobs', [])
    if not any(x['job_id'] == job for x in jobs):
        jobs = jobs + [dict(plan=plan, phase=phase, job_id=job)]
    charges = sum(allocation_charge(x['plan'], x['phase']) for x in jobs)
    reserves = sum(8 for p in PLANS if not any(x['plan'] == p and x['phase'] == 'eval' for x in jobs))
    return 40 + charges + reserves


@contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield read(path) if path.exists() else {'schema': 'two-plan-ledger-v1', 'prior_committed_gpu_hours': 40, 'jobs': []}
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def training_terminal(job):
    result = subprocess.run(['sacct', '-j', job, '-n', '-X', '--format=JobIDRaw,State', '-P'],
                            check=True, capture_output=True, text=True)
    states = [line.split('|')[1].split()[0].rstrip('+') for line in result.stdout.splitlines()
              if line.split('|')[0] == job]
    require(len(states) == 1 and states[0] in TERMINAL, f'prior job {job} not verified terminal')


def checkpoint_valid(p, updates, world_size=2):
    try:
        s = read(p / 'trainer_state.json')
        require(type(s['global_step']) is int and 0 < s['global_step'] <= updates, 'checkpoint step invalid')
        records = [r for r in s.get('log_history', []) if 'loss' in r]
        require(len(records) == s['global_step'] and {r['step'] for r in records} == set(range(1, s['global_step'] + 1))
                and all(math.isfinite(float(r[k])) for r in records for k in ('loss', 'grad_norm')),
                'checkpoint loss/gradient records incomplete or nonfinite')
        shards = model_shards(p)
        return all((p / n).is_file() and (p / n).stat().st_size > 0 for n in
                   ('optimizer.pt', 'scheduler.pt', *(f'rng_state_{i}.pth' for i in range(world_size)),
                    'config.json', 'tokenizer_config.json', *shards))
    except (OSError, ValueError, KeyError): return False


def model_shards(run):
    index = run / 'model.safetensors.index.json'
    if index.exists():
        mapping = read(index)['weight_map']
        require(bool(mapping) and all('/' not in x and '..' not in x for x in mapping.values()), 'invalid shard map')
        return set(mapping.values())
    require((run / 'model.safetensors').is_file(), 'no consolidated model weights')
    return {'model.safetensors'}


def convergence_binding(receipt, parent, allow_pending=False):
    """Bind a real CPU panel and actual auxiliary job; require measured runtime gate."""
    input_path = NEW / 'stage2-convergence-input.json'
    expected = read(input_path)
    require(expected.get('schema') == 'stage2-final-parent-training-token-accuracy-input-v1',
            'unknown convergence input schema')
    source = Path('artifacts/datasets-full-length-15k')
    for key, path in [('source_data_sha256', source / 'advanced.json'),
                      ('source_selection_sha256', source / 'selection_manifest.json'),
                      ('source_dataset_info_sha256', source / 'dataset_info.json')]:
        require(expected.get(key) == digest(path), 'convergence source binding changed')
    require(expected['source_data_sha256'] == receipt['source_data_sha256']
            and expected['source_selection_sha256'] == receipt['source_selection_sha256'],
            'convergence panel and stage2 subset must share exact parent training source')
    require(Path(expected['parent']).resolve() == parent.resolve()
            and expected['parent_completion_sha256'] == digest(parent / 'completion.json'),
            'convergence parent completion binding changed')
    names = model_shards(parent) | {'config.json', 'tokenizer_config.json', 'generation_config.json'}
    require(expected.get('parent_files_sha256') == {n: digest(parent / n) for n in sorted(names)},
            'convergence parent weights or metadata changed')
    require(expected.get('threshold') == .95 and len(expected.get('rows', [])) == 200
            and len({r['id'] for r in expected['rows']}) == 200, 'invalid fixed convergence panel')
    require(all(digest(name) == checksum for name, checksum in expected['execution_sha256'].items()),
            'convergence execution files changed')
    auxiliary = read(NEW / 'other_allocations.json')
    jobs = [j for j in auxiliary.get('jobs', []) if j.get('purpose') == 'stage2_convergence'
            and j.get('input_receipt_sha256') == digest(input_path)]
    require(len(jobs) == 1 and str(jobs[0].get('job_id', '')).isdigit()
            and jobs[0].get('gpu_hours') == 1, 'no exact actual auxiliary GPU job binding')
    job_id = str(jobs[0]['job_id'])
    decision_path = NEW / 'stage2-convergence-decision.json'
    if decision_path.exists():
        measured = read(decision_path)
        hit, total = measured.get('correct_tokens'), measured.get('supervised_tokens')
        ratio = measured.get('token_accuracy')
        require(measured.get('schema') == 'stage2-convergence-measured-v1'
                and measured.get('binding') == expected and measured.get('job_id') == job_id
                and measured.get('gpu_used') is True and 'B200' in measured.get('hardware', ''),
                'convergence measurement does not match exact panel/model/actual GPU job')
        require(type(hit) is int and type(total) is int and 0 <= hit <= total and total > 0
                and type(ratio) in (int, float) and math.isfinite(ratio)
                and abs(ratio - hit / total) <= 1e-12, 'invalid measured token accuracy arithmetic')
        counts = measured.get('per_row_counts', [])
        require(len(counts) == 200 and {r['id'] for r in counts} == {r['id'] for r in expected['rows']}
                and all(type(r.get('correct')) is int and type(r.get('total')) is int
                        and 0 <= r['correct'] <= r['total'] and r['total'] > 0 for r in counts)
                and sum(r['correct'] for r in counts) == hit and sum(r['total'] for r in counts) == total,
                'convergence per-row counts do not support aggregate')
        gate = measured.get('convergence_gate_decision', {})
        require(gate.get('approved') is (ratio < .95)
                and gate.get('basis') == 'measured_training_token_accuracy'
                and gate.get('threshold') == .95 and gate.get('accuracy') == ratio,
                'convergence gate differs from measured threshold rule')
        require(ratio < .95, 'measured training-token accuracy does not authorize stage2')
    else:
        require(allow_pending, 'actual measured convergence decision required; no synthetic waiver')
    return {'convergence_input_receipt_sha256': digest(input_path), 'convergence_job_id': job_id}


def validate(plan, training_runtime=True, allow_pending_convergence=False):
    spec = PLANS[plan]
    config_path = Path(spec['config']); config = yaml.safe_load(config_path.read_text())
    require(config.get('do_train') is False and config.get('overwrite_output_dir') is False,
            'only disabled, non-overwriting draft configurations may bind')
    expected = {'cutoff_len': 16384, 'template': 'qwen', 'finetuning_type': 'full',
                'train_on_prompt': False, 'packing': False, 'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 8 // spec['world_size'], 'average_tokens_across_devices': True,
                'num_train_epochs': spec['epochs'], 'learning_rate': 5e-6, 'seed': 42}
    require(all(config.get(k) == v for k, v in expected.items()) and 'deepspeed' not in config,
            'draft recipe differs from plan-specific native DDP effective batch eight')
    require(not any(config.get(k) for k in ('use_dft_loss', 'use_trimsft_loss', 'use_prift_loss', 'use_gem_loss', 'use_asft_loss')),
            'custom loss not permitted')
    require(config.get('warmup_ratio') == (.03 if plan == 'stage2' else 0.0)
            and config.get('flash_attn') == 'sdpa' and config.get('lr_scheduler_type') == 'cosine'
            and config.get('bf16') is True and config.get('data_seed') == 42, 'fixed native recipe drift')
    require((plan == 'stage2' and 'max_steps' not in config)
            or (plan == 'assistance_coverage' and config.get('max_steps') == 5625), 'native schedule override drift')
    data_dir = Path(config['dataset_dir']); receipt_path = data_dir / spec['receipt']; receipt = read(receipt_path)
    data_path = data_dir / spec['data']; data = read(data_path)
    data_sha = receipt['data_sha256'] if plan == 'assistance_coverage' else receipt['prepared_dataset_sha256']
    require(digest(data_path) == data_sha and len(data) == spec['rows'], 'audited dataset hash/count mismatch')
    require(digest(data_dir / 'dataset_info.json') == receipt['dataset_info_sha256'], 'column mapping hash mismatch')
    for row in data:
        require(all(isinstance(row.get(k), str) for k in ('instruction', 'input', 'output', 'system')),
                'original four text fields required')
    if plan == 'stage2':
        require(receipt.get('native_subset_serialization_audit_complete') is True
                and receipt.get('decontamination_audit_complete') is True,
                'native subset/decontamination audits required')
        require(receipt.get('config_do_train') is False and receipt['draft_config_sha256'] == digest(config_path),
                'stage2 draft differs from actual CPU audit binding')
        require(receipt['subset_unique_prompts'] == spec['rows'], 'stage2 row-count contract changed')
        schedule_path = Path('research/stage2_native_schedule_20261019.json')
        schedule = read(schedule_path)
        require(schedule.get('max_steps') == spec['updates'] and schedule.get('epochs') == spec['epochs']
                and schedule.get('world_size') == 2 and schedule.get('dataset_rows') == spec['rows']
                and schedule.get('rank_batches_per_epoch') == 1499 and schedule.get('updates_per_epoch') == 375
                and schedule.get('planned_ddp_sample_presentations_with_padding') == 5996,
                'native 2997-row distributed padding/update schedule not yet verified')
        if training_runtime:
            import inspect
            from transformers import Trainer
            import hashlib
            require(hashlib.sha256(inspect.getsource(Trainer.set_initial_training_values).encode()).hexdigest()
                    == schedule['trainer_method_sha256'], 'native Trainer step arithmetic changed')
        parent = Path(config['model_name_or_path']); completion = read(parent / 'completion.json')
        require(completion.get('global_step') == 5625 and completion.get('epochs') == 3
                and completion.get('all_model_tensors_finite') is True, 'stage2 parent must be verified final epoch3')
        parent_binding = {str(parent / n): digest(parent / n) for n in model_shards(parent)}
        convergence = convergence_binding(receipt, parent, allow_pending_convergence)
    else:
        convergence = None
        require(receipt.get('status') == 'cpu_selected_and_native_audited_not_gpu_authorized'
                and receipt.get('exact_selected_unique') == 15000 and len(set(receipt['selected_ids'])) == 15000
                and receipt.get('native_eligible_audit_count', 0) >= 15000,
                'assistance coverage native audit not complete')
        require(Path(config['model_name_or_path']).name == 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
                'coverage candidate must restart pinned original base')
        parent_binding = {}
    if training_runtime:
        from llamafactory.hparams import ModelArguments, FinetuningArguments
        from llamafactory.hparams.parser import _check_extra_dependencies
        from types import SimpleNamespace
        _check_extra_dependencies(ModelArguments(model_name_or_path=config['model_name_or_path'], flash_attn='sdpa'),
                                  FinetuningArguments(finetuning_type='full'),
                                  SimpleNamespace(deepspeed=None, predict_with_generate=False))
    binding = {'schema': 'two-plan-execution-v1', 'plan': plan, 'rows': spec['rows'],
               'epochs': spec['epochs'], 'updates': spec['updates'], 'prior_committed_gpu_hours': 40,
               'execution_sha256': {p: digest(p) for p in (*SCRIPT_FILES, spec['config'])},
               'preparation_receipt_path': str(receipt_path), 'preparation_receipt_sha256': digest(receipt_path),
               'data_sha256': data_sha, 'parent_weights_sha256': parent_binding,
               'native_schedule_sha256': digest('research/stage2_native_schedule_20261019.json') if plan == 'stage2' else None,
               'convergence': convergence,
               'training_gpu_hours_per_allocation': allocation_charge(plan, 'train'), 'evaluation_gpu_hours_per_allocation': 8,
               'stage2_new_optimizer': plan == 'stage2', 'training_world_size': spec['world_size'],
               'training_allocation_hours': spec['allocation_hours'], 'effective_batch_size': 8,
               'distributed_order_not_bitwise_equivalent': True, 'deepspeed_enabled': False,
               'training_throughput_gpu_verified': False, 'completion_speed_guaranteed': False,
               'maximum_training_allocations': 1 if plan == 'assistance_coverage' else 1}
    return config, binding


def verify_model_tensors(run):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    shards = model_shards(run); tensors = set()
    for name in shards:
        with safe_open(str(run / name), framework='pt', device='cpu') as f:
            for key in f.keys():
                require(key not in tensors, 'duplicate model tensor across shards'); tensors.add(key)
                require(all(torch.isfinite(x).all().item() for x in f.get_tensor(key).reshape(-1).split(1000000)),
                        f'nonfinite final weight {name}:{key}')
    index = run / 'model.safetensors.index.json'
    if index.exists(): require(tensors == set(read(index)['weight_map']), 'model index tensor coverage incomplete')
    return shards


def verify_final(plan, config, binding):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    run = Path(config['output_dir']); spec = PLANS[plan]; state = read(run / 'trainer_state.json')
    require(state['global_step'] == spec['updates'] and abs(state['epoch'] - spec['epochs']) < .001,
            'incomplete exact updates/epochs')
    records = [r for r in state['log_history'] if 'loss' in r]
    require(len(records) == spec['updates'] and {r['step'] for r in records} == set(range(1, spec['updates'] + 1)),
            'missing per-update finite training records')
    require(all(math.isfinite(float(r[k])) for r in records for k in ('loss', 'grad_norm')), 'nonfinite loss/gradient')
    shards = verify_model_tensors(run)
    require(all((run / n).is_file() for n in ('config.json', 'tokenizer_config.json', 'generation_config.json')), 'tokenizer/model metadata missing')
    write_json(run / 'two_plan_completion.json', {'plan': plan, 'global_step': spec['updates'],
               'epochs': spec['epochs'], 'unique_prompts': spec['rows'], 'world_size': spec['world_size'],
               'effective_batch_size': 8, 'all_model_tensors_finite': True, 'finite_updates': len(records),
               'binding': binding, 'final_shards_sha256': {n: digest(run / n) for n in shards},
               'source_example_presentations': spec['rows'] * spec['epochs'],
               'ddp_sampler_padding': plan == 'stage2',
               'convergence_decision_sha256': digest(NEW / 'stage2-convergence-decision.json') if plan == 'stage2' else None})


def deadline_fit(phase, now=None):
    require(phase in ('train', 'eval'), 'unknown execution phase')
    now = now or datetime.now(ZoneInfo('America/New_York'))
    deadline = datetime(2026, 10, 20, 23, 59, tzinfo=ZoneInfo('America/New_York'))
    # Runtime startup checks one allocation plus its final evaluation.
    hours = 14 if phase == 'train' else 8
    require(now + timedelta(hours=hours) <= deadline,
            'allocation plus full evaluation reserve cannot fit October20 23:59Eastern deadline')


def submission_deadline_fit(plan, phase, initial_training, now=None):
    """Conservative request duration; scheduler forecast must fit this margin too."""
    now = now or datetime.now(ZoneInfo('America/New_York'))
    deadline = datetime(2026, 10, 20, 23, 59, tzinfo=ZoneInfo('America/New_York'))
    hours = PLANS[plan]['allocation_hours'] + 8 if phase == 'train' else 8
    require(now + timedelta(hours=hours) <= deadline,
            f'{hours} hour requested training/evaluation margin does not fit deadline')
    return hours


def submit(plan, phase, config, binding, auth_path):
    """Only submission API: reserve actual ID under lock; never retry uncertainty.

    A persisted attempting/unknown journal entry blocks further submissions until
    an operator reconciles scheduler evidence. No sbatch retry occurs here.
    """
    require(plan in PLANS and phase in ('train', 'eval'), 'unknown submission plan/phase')
    a = authorization(auth_path)
    if plan == 'stage2' and phase == 'train':
        require(isinstance(binding.get('convergence'), dict)
                and str(binding['convergence'].get('convergence_job_id', '')).isdigit(),
                'stage2 submission requires bound actual convergence dependency job')
    receipt_path = NEW / f'{plan}.execution_receipt.json'
    require(read(receipt_path) == binding, 'submission differs from exact CPU execution receipt')
    run = Path(config['output_dir'])
    require(run.name in {'lmf-assistance-coverage-20261019', 'stage2-hard3k-20261019'},
            'unexpected candidate run path')
    ledger_path = NEW / 'allocations.json'
    with locked(ledger_path) as ledger:
        validate_ledger(ledger)
        other_path = NEW / 'other_allocations.json'
        if other_path.exists():
            require(not any(x.get('status') in ('attempting', 'unknown')
                            for x in read(other_path).get('submission_journal', [])),
                    'unreconciled ancillary sbatch outcome; manual reconciliation required')
        journal = ledger.setdefault('submission_journal', [])
        require(not any(x.get('status') in ('attempting', 'unknown') for x in journal),
                'unreconciled sbatch outcome; inspect scheduler and journal before any retry')
        own = [x for x in ledger['jobs'] if x['plan'] == plan]
        for old in own:
            # sacct terminal proof rejects queued/running/unknown jobs alike.
            training_terminal(old['job_id'])
        require(not any(x['phase'] == 'eval' for x in own), 'no second final evaluation or training after evaluation')
        training = [x for x in own if x['phase'] == 'train']
        require(not (phase == 'train' and training),
                'one training allocation per plan only; no continuation authorized')
        if phase == 'eval':
            require(bool(training), 'evaluation requires registered terminal training')
            verify_final(plan, config, binding)
        else:
            require(not (run / 'two_plan_completion.json').exists(), 'completed candidate cannot retrain')
            if training:
                require((run / 'two_plan_binding.json').is_file()
                        and read(run / 'two_plan_binding.json') == binding,
                        'continuation requires exact own run binding')
                checkpoints = [p for p in run.glob('checkpoint-*') if checkpoint_valid(p, PLANS[plan]['updates'], PLANS[plan]['world_size'])]
                require(bool(checkpoints), 'no complete progressing native recovery checkpoint; inspect instead of resubmitting')
                verify_model_tensors(max(checkpoints, key=lambda p: read(p / 'trainer_state.json')['global_step']))
            elif run.exists():
                require(not list(run.iterdir()), 'new candidate cannot adopt existing run state')
        margin = submission_deadline_fit(plan, phase, not training)
        total = projected_total(ledger, plan, phase, '__unsubmitted__') + a['other_committed_gpu_hours']
        require(total <= 40 + a['additional_gpu_hours'],
                'submission exceeds incremental cap including reserved evaluations and ancillary costs')
        # All refusal checks above occur before any scheduler call or journal write.
        intent = {'intent_id': uuid.uuid4().hex, 'plan': plan, 'phase': phase,
                  'status': 'attempting', 'authorization_sha256': digest(auth_path),
                  'binding': binding, 'deadline_margin_hours': margin,
                  'created_at': datetime.now(ZoneInfo('America/New_York')).isoformat()}
        journal.append(intent)
        write_json(ledger_path, ledger)  # durable ambiguity marker before sbatch
        command = ['sbatch', '--parsable', f'--job-name=reasoning_{plan}_{phase}',
                   *( [f'--gres=gpu:{PLANS[plan]["world_size"]}',
                       f'--time={PLANS[plan]["allocation_hours"]:02d}:00:00'] if phase == 'train' else []),
                   *( [f'--dependency=afterok:{binding["convergence"]["convergence_job_id"]}']
                      if plan == 'stage2' and phase == 'train' and binding.get('convergence') else []),
                   f'scripts/two_plan_20261019_{phase}.slurm', plan]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            actual = result.stdout.strip().split(';')[0]
            require(actual.isdigit() and int(actual) > 0, 'sbatch returned no unambiguous actual job ID')
            require(not any(x['job_id'] == actual for x in ledger['jobs']), 'sbatch returned a duplicate job ID')
        except Exception as error:
            intent.update(status='unknown', error_type=type(error).__name__,
                          reconciliation_required=True)
            write_json(ledger_path, ledger)
            raise RuntimeError('sbatch outcome unknown; no automatic retry; reconcile scheduler with submission journal') from error
        ledger['jobs'].append({'plan': plan, 'phase': phase, 'job_id': actual,
                               'authorization_sha256': digest(auth_path), 'binding': binding,
                               'status': 'submitted_reserved',
                               'charged_gpu_hours': allocation_charge(plan, phase)})
        intent.update(status='confirmed', job_id=actual)
        write_json(ledger_path, ledger)
        print(json.dumps({'submitted_actual_job_id': actual, 'plan': plan, 'phase': phase,
                          'status': 'submitted_reserved', 'projected_total_gpu_hours': total}), flush=True)
        return actual


def charge_start(plan, phase, auth_path):
    """Record an authorized started allocation before optional runtime/data guards.

    This is accounting, not a submission API. Scheduler queue feasibility must be
    checked before sbatch. Startup failures retain their conservative full charge.
    """
    require(plan in PLANS and phase in ('train', 'eval'), 'unknown plan or phase')
    a = authorization(auth_path); job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and int(os.environ['SLURM_GPUS_ON_NODE']) == (PLANS[plan]['world_size'] if phase == 'train' else 1),
            'wrong GPU allocation')
    ledger_path = NEW / 'allocations.json'
    with locked(ledger_path) as ledger:
        validate_ledger(ledger)
        require(not any(x['job_id'] == job and (x['plan'] != plan or x['phase'] != phase)
                        for x in ledger['jobs']), 'job ID reused across plan or phase')
        total = projected_total(ledger, plan, phase, job) + a['other_committed_gpu_hours']
        require(total <= 40 + a['additional_gpu_hours'], 'incremental cap exceeded including reserved evals and other costs')
        if not any(x['job_id'] == job for x in ledger['jobs']):
            ledger['jobs'].append({'plan': plan, 'phase': phase, 'job_id': job,
                                  'authorization_sha256': digest(auth_path), 'binding': None,
                                  'status': 'started_before_runtime_preflight',
                                  'charged_gpu_hours': allocation_charge(plan, phase)})
            write_json(ledger_path, ledger)
        else:
            existing = next(x for x in ledger['jobs'] if x['job_id'] == job)
            if existing.get('status') == 'submitted_reserved':
                existing['status'] = 'started_before_runtime_preflight'
                write_json(ledger_path, ledger)
    # A started allocation failing this gate stays charged in the ledger above.
    submission_deadline_fit(plan, phase, False)


def register(plan, phase, config, binding, auth_path):
    a = authorization(auth_path); job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and int(os.environ['SLURM_GPUS_ON_NODE']) == (PLANS[plan]['world_size'] if phase == 'train' else 1), 'wrong GPU allocation')
    run = Path(config['output_dir'])
    require(run.name in {'lmf-assistance-coverage-20261019', 'stage2-hard3k-20261019'}, 'unexpected new run output')
    ledger_path = NEW / 'allocations.json'
    with locked(ledger_path) as ledger:
        validate_ledger(ledger)
        for old in ledger['jobs']:
            if old['plan'] == plan and old['job_id'] != job:
                require(not (phase == 'train' and old['phase'] == 'train'),
                        'one training allocation per plan only; no continuation authorized')
                training_terminal(old['job_id'])
                require(not (phase == 'train' and old['phase'] == 'eval'), 'no retraining after final eval')
                require(not (phase == 'eval' and old['phase'] == 'eval'), 'one final evaluation per plan only')
        total = projected_total(ledger, plan, phase, job) + a['other_committed_gpu_hours']
        require(total <= 40 + a['additional_gpu_hours'], 'additional budget exceeded including both reserved final evaluations')
        require(not any(x['job_id'] == job and (x['plan'] != plan or x['phase'] != phase) for x in ledger['jobs']), 'job ID reused across plans/phases')
        if phase == 'eval':
            require(any(x['plan'] == plan and x['phase'] == 'train' for x in ledger['jobs']), 'no registered training')
            verify_final(plan, config, binding)
        run.mkdir(parents=True, exist_ok=True)
        owner = run / 'two_plan_binding.json'
        if owner.exists(): require(read(owner) == binding, 'run receipt differs; no state adoption')
        else:
            require(not list(run.iterdir()), 'refusing to adopt pre-existing unbound output')
            write_json(owner, binding)
        existing = next((x for x in ledger['jobs'] if x['job_id'] == job), None)
        require(existing is not None, 'started allocation must be charged before preflight')
        require(existing.get('binding') in (None, binding), 'started job binding changed')
        existing['binding'] = binding; existing['status'] = 'runtime_registered'
        write_json(ledger_path, ledger)
        if phase == 'train':
            require(not (run / 'two_plan_completion.json').exists(), 'completed candidate cannot retrain')
            effective = dict(config); effective['do_train'] = True
            checkpoints = [p for p in run.glob('checkpoint-*') if checkpoint_valid(p, PLANS[plan]['updates'], PLANS[plan]['world_size'])]
            if checkpoints:
                highest = max(checkpoints, key=lambda p: read(p / 'trainer_state.json')['global_step'])
                verify_model_tensors(highest)
                effective['resume_from_checkpoint'] = str(highest)
            else: require(not list(run.glob('checkpoint-*')), 'invalid own recovery state; inspect rather than restart')
            write_path = run / 'effective_config.yaml'
            write_path.write_text(yaml.safe_dump(effective, sort_keys=False))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', required=True, choices=PLANS)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--register-train', action='store_true')
    mode.add_argument('--register-eval', action='store_true'); mode.add_argument('--verify-final', action='store_true')
    mode.add_argument('--submit-train', action='store_true'); mode.add_argument('--submit-eval', action='store_true')
    mode.add_argument('--prepare-stage2-pending-convergence', action='store_true')
    p.add_argument('--authorization', type=Path, default=AUTH); a = p.parse_args()
    if a.register_train or a.register_eval:
        charge_start(a.plan, 'train' if a.register_train else 'eval', a.authorization)
    require(not a.prepare_stage2_pending_convergence or a.plan == 'stage2', 'pending convergence only applies to stage2')
    config, binding = validate(a.plan, training_runtime=not (a.register_eval or a.submit_eval),
                               allow_pending_convergence=a.prepare_stage2_pending_convergence or (a.submit_train and a.plan == 'stage2'))
    path = NEW / f'{a.plan}.execution_receipt.json'
    if a.prepare or a.prepare_stage2_pending_convergence:
        if path.exists(): require(read(path) == binding, 'existing execution receipt differs')
        else: write_json(path, binding)
        print('CPU_EXECUTION_BINDING_PREPARED_NO_GPU_AUTHORIZATION'); return
    require(read(path) == binding, 'execution files/dataset differ from CPU receipt')
    if a.submit_train: submit(a.plan, 'train', config, binding, a.authorization); return
    if a.submit_eval: submit(a.plan, 'eval', config, binding, a.authorization); return
    if a.register_train: register(a.plan, 'train', config, binding, a.authorization)
    if a.register_eval: register(a.plan, 'eval', config, binding, a.authorization)
    if a.verify_final: verify_final(a.plan, config, binding)


if __name__ == '__main__': main()
