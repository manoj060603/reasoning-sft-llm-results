"""Final-parent training-token accuracy diagnostic; no training or benchmark use.

One fixed source-only 200-row panel. Native serialization, shifted masked argmax,
CPU-bound provenance, one authorized B200/hour. The document's .95 cutoff is an
experimental scheduling decision, not an established convergence guarantee.
"""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import importlib.metadata
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import digest, write_json
from .two_plan_execution_20261019 import authorization, model_shards, require, validate_ledger, PLANS

ROOT = Path('/projects/tealab/rohit/reasoning-sft-20261011')
ART = Path('artifacts/two-plans-20261019')
DATA = Path('artifacts/datasets-full-length-15k')
PARENT = Path('runs/lmf-full-length-15k-16k-3epoch-fast-2b200')
DATA_SHA = '0ccdb3b3e7800f181cd99cb16f433d81d359d86e6d6fa0d089f61c36f56185cc'
SELECTION_SHA = 'f4a43fbc36ce72e74cb9a5f1afa92a31fd5071d753420657eb7440026f5ee7ed'
AUTH = ART/'additional_budget_authorization.json'
INPUT_RECEIPT = ART/'stage2-convergence-input.json'
OUTPUT_RECEIPT = ART/'stage2-convergence-decision.json'
FILES = ('reasoning/stage2_convergence_20261019.py', 'scripts/stage2_convergence_20261019.slurm',
         'scripts/stage2_convergence_20261019.sh', 'reasoning/full_length.py',
         'reasoning/two_plan_execution_20261019.py',
         'LLaMA-Factory/src/llamafactory/data/converter.py',
         'LLaMA-Factory/src/llamafactory/data/processor/supervised.py',
         'LLaMA-Factory/src/llamafactory/data/template.py')


def read(p): return json.loads(Path(p).read_text())


def user_content(row):
    return '\n'.join(row[k] for k in ('instruction', 'input') if row[k])


def fixed_indices(ids):
    require(len(ids) == len(set(ids)) == 15000, 'fixed original 15K IDs required')
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(('stage2-convergence-42:' + ids[i]).encode()).hexdigest())[:200]


def native_batch(row, tok, template, proc, converter):
    converted = converter(row)
    prompt = [{'role': 'user', 'content': user_content(row)}]
    response = [{'role': 'assistant', 'content': row['output']}]
    require(converted['_system'] == row['system'] and converted['_prompt'] == prompt
            and converted['_response'] == response, 'native original system/input/prompt/response drift')
    pairs = template.encode_multiturn(tok, prompt + response, row['system'], None)
    require(len(pairs) == 1, 'native panel row is not single turn')
    source, target = pairs[0]
    expected = target + ([tok.eos_token_id] if template.efficient_eos else [])
    batch = proc.preprocess_dataset({k: [v] for k, v in converted.items()})
    require(len(batch['input_ids']) == 1, 'native processor dropped fixed training row')
    ids, labels = batch['input_ids'][0], batch['labels'][0]
    require(len(source) + len(expected) <= 16384 and ids == source + expected
            and labels == [-100]*len(source) + expected and tok.eos_token_id in expected,
            'native complete-target/shift/prompt-mask/EOS mismatch')
    return ids, labels


def shifted_chunk_accuracy(hidden, labels, lm_head, chunk=128):
    """labels[t] is predicted by hidden[t-1]; masks never enter denominator."""
    import torch
    require(hidden.ndim == 3 and labels.ndim == 2 and hidden.shape[:2] == labels.shape,
            'hidden/labels shapes differ')
    require(type(chunk) is int and chunk > 0, 'positive logit chunk required')
    hit = total = 0
    for start in range(0, hidden.shape[1] - 1, chunk):
        end = min(start + chunk, hidden.shape[1] - 1)
        targets = labels[:, start+1:end+1]
        mask = targets != -100
        logits = lm_head(hidden[:, start:end])
        require(torch.isfinite(logits).all().item(), 'nonfinite diagnostic logits')
        predicted = logits.argmax(dim=-1)
        hit += ((predicted == targets) & mask).sum().item()
        total += mask.sum().item()
    require(total > 0, 'no supervised shifted tokens')
    return int(hit), int(total)


def validate_panel():
    require(digest(DATA/'advanced.json') == DATA_SHA and digest(DATA/'selection_manifest.json') == SELECTION_SHA,
            'fixed original training data or selection mismatch')
    source = read(DATA/'advanced.json'); selection = read(DATA/'selection_manifest.json')
    require(len(source) == 15000, 'wrong training row count')
    final = read(PARENT/'completion.json')
    require(final.get('global_step') == 5625 and final.get('epochs') == 3
            and final.get('all_model_tensors_finite') is True and final.get('finite_updates') == 5625,
            'only verified final epoch3 parent may be measured')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    tensor_keys = set()
    for shard in model_shards(PARENT):
        with safe_open(str(PARENT/shard), framework='pt', device='cpu') as weights:
            for key in weights.keys():
                require(key not in tensor_keys, 'duplicate parent tensor')
                tensor_keys.add(key)
                require(all(torch.isfinite(part).all().item() for part in weights.get_tensor(key).reshape(-1).split(1000000)),
                        'parent final weight is nonfinite')
    index = PARENT/'model.safetensors.index.json'
    if index.exists(): require(tensor_keys == set(read(index)['weight_map']), 'parent tensor index incomplete')
    from .full_length import processor
    model = selection['model_manifest']['path']
    tok, template, proc, converter = processor(model, DATA)
    indices = fixed_indices(selection['selected_ids']); batches = []
    labels_hash = hashlib.sha256(); records = []
    for i in indices:
        row = source[i]; entry = selection['selected'][i]
        require(entry['id'] == selection['selected_ids'][i]
                and hashlib.sha256(row['output'].encode()).hexdigest() == entry['original_response_sha256'],
                'panel target bytes differ from original native audit')
        ids, labels = native_batch(row, tok, template, proc, converter)
        require(sum(x != -100 for x in labels) == entry['supervised_tokens'], 'native supervised-count drift')
        batches.append((ids, labels)); labels_hash.update(json.dumps([ids, labels], separators=(',', ':')).encode())
        records.append({'id': entry['id'], 'source_index': i, 'tokens': len(ids),
                        'supervised_tokens': entry['supervised_tokens'],
                        'original_fields_sha256': {k: hashlib.sha256(row[k].encode()).hexdigest()
                                                   for k in ('instruction', 'input', 'output', 'system')}})
    # Exercise nonempty query preservation before any GPU allocation.
    fixture = {'system': 'Explicit original system.', 'instruction': 'Keep this prompt.',
               'input': 'Keep this nonempty query.', 'output': 'Reason carefully. \\boxed{5}'}
    native_batch(fixture, tok, template, proc, converter)
    binding = {'schema': 'stage2-final-parent-training-token-accuracy-input-v1',
               'source_data_sha256': DATA_SHA, 'source_selection_sha256': SELECTION_SHA,
               'panel_selection_rule': 'smallest SHA256(stage2-convergence-42:originalID), 200 rows',
               'rows': records, 'serialized_inputs_and_labels_sha256': labels_hash.hexdigest(),
               'parent': str(PARENT), 'parent_completion_sha256': digest(PARENT/'completion.json'),
               'all_parent_weight_tensors_cpu_finite': True, 'parent_tensor_count': len(tensor_keys),
               'parent_files_sha256': {n: digest(PARENT/n) for n in sorted(model_shards(PARENT) |
                                       {'config.json', 'tokenizer_config.json', 'generation_config.json'})},
               'source_tokenizer_files_sha256': {p.name: digest(p) for p in sorted(Path(model).glob('*.json'))},
               'source_dataset_info_sha256': digest(DATA/'dataset_info.json'),
               'execution_sha256': {p: digest(p) for p in FILES},
               'runtime_versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'safetensors')},
               'native_complete_targets_input_system_mask_eos': True,
               'benchmark_data_or_outcomes_used': False, 'early_epoch_checkpoints_available': False,
               'missing_checkpoint_steps': [1875, 3750], 'threshold': .95,
               'threshold_scientifically_validated': False, 'gpu_used': False}
    return batches, binding


def require_deadline_fit(now=None):
    deadline = datetime(2026, 10, 20, 23, 59, tzinfo=ZoneInfo('America/New_York'))
    now = now or datetime.now(ZoneInfo('America/New_York'))
    require(now + timedelta(hours=15) <= deadline,
            'convergence1h plus training6h plus finaleval8h cannot fit deadline')


def journal_clean(ledger):
    journal = ledger.get('submission_journal', [])
    values = journal.values() if isinstance(journal, dict) else journal
    require(all(r.get('status') not in ('attempting', 'unknown') for r in values),
            'uncertain scheduler submission requires manual reconciliation before another job')


def check_other_budget(a, central, ledger):
    if central:
        validate_ledger(central)
    jobs = central.get('jobs', [])
    # Main ledger validates exact plan-specific charges (coverage4B200x6h=24,
    # stage2 2B200x6h=12). Never reconstruct every training allocation as12.
    charged = sum(j['charged_gpu_hours'] for j in jobs)
    reserve = sum(8 for plan in PLANS
                  if not any(j['plan'] == plan and j['phase'] == 'eval' for j in jobs))
    require(charged + reserve + a['other_committed_gpu_hours'] <= a['additional_gpu_hours'],
            'total incremental commitments exceed explicit additional budget')
    require(all(type(j.get('gpu_hours')) in (int, float) and math.isfinite(j['gpu_hours']) and j['gpu_hours'] > 0
                for j in ledger['jobs']), 'invalid auxiliary cost ledger')
    require(sum(j['gpu_hours'] for j in ledger['jobs']) + 1 <= a['other_committed_gpu_hours'],
            'other committed GPU reserve exhausted')


def submit(auth_path):
    a = authorization(auth_path)
    require(a['other_committed_gpu_hours'] >= 1, 'one authorized convergence GPU-hour required')
    _, binding = validate_panel()
    require(read(INPUT_RECEIPT) == binding, 'actual CPU native/provenance preflight missing or stale')
    require(not OUTPUT_RECEIPT.exists(), 'convergence decision already exists')
    require_deadline_fit()
    ART.mkdir(parents=True, exist_ok=True); path = ART/'other_allocations.json'
    with open(str(ART/'allocations.json') + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            a = authorization(auth_path)
            central = read(ART/'allocations.json') if (ART/'allocations.json').exists() else {}
            journal_clean(central)
            ledger = read(path) if path.exists() else {'schema': 'two-plan-other-allocations-v1', 'jobs': [], 'submission_journal': []}
            require(ledger.get('schema') == 'two-plan-other-allocations-v1', 'unknown other-cost ledger')
            journal_clean(ledger)
            require(not any(r['purpose'] == 'stage2_convergence' for r in ledger['jobs']),
                    'one convergence allocation only; no repeat submission')
            require(not ledger.get('submission_journal'), 'prior auxiliary submission attempt requires manual reconciliation')
            check_other_budget(a, central, ledger)
            attempt = {'purpose': 'stage2_convergence', 'status': 'attempting',
                       'authorization_sha256': digest(auth_path), 'input_receipt_sha256': digest(INPUT_RECEIPT)}
            ledger['submission_journal'] = [attempt]; write_json(path, ledger)
            try:
                result = subprocess.run(['sbatch', '--parsable', 'scripts/stage2_convergence_20261019.slurm'],
                                        capture_output=True, text=True, timeout=45)
                match = re.fullmatch(r'(\d+)(?:;[\w.-]+)?', result.stdout.strip())
                require(result.returncode == 0 and match is not None,
                        'scheduler outcome unconfirmed; inspect job queue and journal manually')
                job = match.group(1)
            except Exception:
                attempt['status'] = 'unknown'; write_json(path, ledger)
                raise
            attempt['status'] = 'confirmed'; attempt['job_id'] = job
            ledger['jobs'].append({'purpose': 'stage2_convergence', 'job_id': job, 'gpu_hours': 1,
                                   'authorization_sha256': digest(auth_path),
                                   'input_receipt_sha256': digest(INPUT_RECEIPT), 'status': 'submitted_reserved'})
            write_json(path, ledger)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps({'submitted_job_id': job, 'gpus': 1, 'hours': 1, 'purpose': 'stage2_convergence'}))


def charge_start(auth_path):
    a = authorization(auth_path)
    require(a['other_committed_gpu_hours'] >= 1, 'one GPU-hour convergence reserve not authorized')
    job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and int(os.environ['SLURM_GPUS_ON_NODE']) == 1, 'one-B200 allocation required')
    ART.mkdir(parents=True, exist_ok=True)
    path = ART/'other_allocations.json'
    # Same lock as the two-plan train/eval ledger; its authorization reserves all
    # other_committed_gpu_hours, so this hour is not charged there a second time.
    with open(str(ART/'allocations.json') + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            ledger = read(path) if path.exists() else {'schema': 'two-plan-other-allocations-v1', 'jobs': []}
            require(ledger.get('schema') == 'two-plan-other-allocations-v1', 'unknown other-cost ledger')
            old = [r for r in ledger['jobs'] if r['purpose'] == 'stage2_convergence']
            require(not old or (len(old) == 1 and old[0]['job_id'] == job),
                    'one convergence allocation only; no automatic retries')
            if not old:
                require(sum(r['gpu_hours'] for r in ledger['jobs']) + 1 <= a['other_committed_gpu_hours'],
                        'other authorized GPU cost reserve exhausted')
                ledger['jobs'].append({'purpose': 'stage2_convergence', 'job_id': job, 'gpu_hours': 1,
                                       'authorization_sha256': digest(auth_path), 'status': 'started_before_preflight'})
                write_json(path, ledger)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)
    require_deadline_fit()


def gate_exit_status(accuracy):
    require(math.isfinite(accuracy) and 0 <= accuracy <= 1, 'invalid measured training accuracy')
    # afterok must release stage2 only for this predeclared experimental condition.
    return 0 if accuracy < .95 else 2


def run(auth_path):
    charge_start(auth_path)
    batches, binding = validate_panel()
    require(read(INPUT_RECEIPT) == binding, 'CPU native panel/provenance preflight missing or stale')
    require(not OUTPUT_RECEIPT.exists(), 'existing convergence result must not be overwritten')
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(PARENT), torch_dtype=torch.bfloat16,
                                                device_map={'': 'cuda:0'}, attn_implementation='sdpa',
                                                trust_remote_code=False, local_files_only=True).eval()
    require(model.config.model_type == 'qwen2' and hasattr(model, 'model') and hasattr(model, 'lm_head'),
            'bounded hidden/lm-head diagnostic only implemented for original Qwen2 architecture')
    hit = total = 0; row_counts = []
    with torch.inference_mode():
        for i, (ids, labels) in enumerate(batches):
            inputs = torch.tensor([ids], dtype=torch.long, device='cuda:0')
            targets = torch.tensor([labels], dtype=torch.long, device='cuda:0')
            hidden = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                 use_cache=False, return_dict=True).last_hidden_state
            row_hit, row_total = shifted_chunk_accuracy(hidden, targets, model.lm_head)
            hit += row_hit; total += row_total
            row_counts.append({'id': binding['rows'][i]['id'], 'correct': row_hit, 'total': row_total})
            del hidden, inputs, targets
            if (i+1) % 10 == 0: print(f'Final-parent native training-token accuracy rows {i+1}/200', flush=True)
    accuracy = hit / total
    require(math.isfinite(accuracy) and 0 <= accuracy <= 1, 'invalid aggregate accuracy')
    approved = accuracy < .95
    decision = {'approved': approved, 'basis': 'measured_training_token_accuracy',
                'evidence': str(OUTPUT_RECEIPT), 'accuracy': accuracy, 'threshold': .95,
                'rule': 'stage2 condition supported only when final fixed-panel accuracy is below .95',
                'threshold_scientifically_validated': False}
    write_json(OUTPUT_RECEIPT, {'schema': 'stage2-convergence-measured-v1', 'binding': binding,
               'correct_tokens': hit, 'supervised_tokens': total, 'token_accuracy': accuracy,
               'per_row_counts': row_counts, 'gpu_used': True, 'hardware': torch.cuda.get_device_name(0),
               'job_id': os.environ['SLURM_JOB_ID'], 'evaluation_decoding_protocol_changed': False,
               'chunk_positions': 128, 'diagnostic_dtype': 'bfloat16',
               'convergence_gate_decision': decision,
               'only_final_epoch_measured': True, 'missing_early_steps': [1875, 3750],
               'arithmetic_not_claimed_bitwise_identical_to_full_sequence_lm_head': True})
    print(json.dumps({'correct_tokens': hit, 'supervised_tokens': total, 'accuracy': accuracy,
                      'stage2_gate_approved': approved, 'threshold_validated': False}), flush=True)
    return gate_exit_status(accuracy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    mode.add_argument('--submit', action='store_true')
    p.add_argument('--authorization', type=Path, default=AUTH); a = p.parse_args()
    if a.prepare:
        _, binding = validate_panel()
        if INPUT_RECEIPT.exists(): require(read(INPUT_RECEIPT) == binding, 'existing CPU diagnostic binding differs')
        else: write_json(INPUT_RECEIPT, binding)
        print('FINAL_PARENT_200_ROW_NATIVE_CPU_DIAGNOSTIC_PREFLIGHT_PASSED_NO_GPU')
    elif a.submit: submit(a.authorization)
    else: return run(a.authorization)


if __name__ == '__main__': raise SystemExit(main())
