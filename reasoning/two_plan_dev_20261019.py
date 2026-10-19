"""Frozen 200-question development control and Stage-2 screen.

This module creates one source-only panel before model outcomes exist, binds the
original final parent and the Stage-2 candidate separately, and submits at most
one two-hour B200 evaluation for each.  It never selects training data, reads
public benchmark outcomes, or launches a terminal benchmark evaluation.
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
from pathlib import Path
from zoneinfo import ZoneInfo

from .benchmark import record_ids
from .compare import check_protocol, load_run
from .data import boxed, digest, normalize, prompt_id, question, read_rows, write_json
from .metrics import summarize
from .two_plan_execution_20261019 import (
    PLANS, authorization, model_shards, require, validate_ledger,
)

ROOT = Path('/projects/tealab/rohit/reasoning-sft-20261011')
ART = Path('artifacts/two-plans-20261019')
AUTH = ART / 'additional_budget_authorization.json'
DEV_ROOT = ART / 'dev-200'
DEV = DEV_ROOT / 'acereason_dev' / 'test.jsonl'
DEV_META = Path(str(DEV) + '.meta.json')
VALIDATION = Path('artifacts/prepared_extended/validation.json')
EXTENDED_MANIFEST = Path('artifacts/prepared_extended/manifest.json')
POOL = Path('artifacts/prepared_extended/pool.json')
SELECTION = Path('artifacts/datasets-full-length-15k/selection_manifest.json')
PARENT = Path('runs/lmf-full-length-15k-16k-3epoch-fast-2b200')
STAGE2 = Path('runs/stage2-hard3k-20261019')
PARENT_OUT = Path('results/two-plan-parent-dev-20261019')
STAGE2_OUT = Path('results/two-plan-stage2-dev-20261019')
PANEL_RECEIPT = ART / 'dev-200-panel.json'
PARENT_INPUT = ART / 'parent-dev-input.json'
PARENT_DECISION = ART / 'parent-dev-decision.json'
STAGE2_INPUT = ART / 'stage2-dev-input.json'
STAGE2_DECISION = ART / 'stage2-dev-decision.json'

VALIDATION_SHA = '5737267040ec4d3e2f3ea9cf1ebf25cbc9e95f5ad7da03b60a347263e0deca71'
POOL_SHA = 'c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9'
SELECTION_SHA = 'f4a43fbc36ce72e74cb9a5f1afa92a31fd5071d753420657eb7440026f5ee7ed'
PANEL_TAG = 'two-plan-dev-42:'
DEADLINE = datetime(2026, 10, 20, 23, 59, tzinfo=ZoneInfo('America/New_York'))
FILES = (
    'reasoning/two_plan_dev_20261019.py', 'scripts/two_plan_dev_20261019.slurm',
    'scripts/two_plan_dev_20261019.sh', 'reasoning/evaluate.py', 'reasoning/metrics.py',
    'reasoning/compare.py', 'reasoning/data.py', 'reasoning/benchmark.py',
    'reasoning/grader_preflight.py', 'eval/utils/grader.py', 'eval/utils/parser.py',
    'eval/utils/math_normalization.py',
)
PHASES = {
    'parent': dict(purpose='parent_dev', model=PARENT, output=PARENT_OUT,
                   input=PARENT_INPUT, decision=PARENT_DECISION,
                   label='C1_original_final5625_control'),
    'stage2': dict(purpose='stage2_dev', model=STAGE2, output=STAGE2_OUT,
                   input=STAGE2_INPUT, decision=STAGE2_DECISION,
                   label='Stage2_candidate'),
}


def read(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)
    if path.exists():
        require(path.read_text() == text, f'existing frozen artifact differs: {path}')
        return
    temp = path.with_suffix(path.suffix + '.tmp'); temp.write_text(text); temp.replace(path)


def iter_json_array(path, chunk_size=1 << 20):
    """Stream a top-level JSON array without holding the 2 GB pool in memory."""
    decoder = json.JSONDecoder(); buffer = ''; pos = 0; started = False; eof = False
    with Path(path).open(encoding='utf-8') as stream:
        while True:
            if pos > (1 << 20): buffer, pos = buffer[pos:], 0
            while pos >= len(buffer) and not eof:
                block = stream.read(chunk_size); eof = not block; buffer += block
            while pos < len(buffer) and buffer[pos].isspace(): pos += 1
            if not started:
                require(pos < len(buffer) and buffer[pos] == '[', 'source is not a JSON array')
                started = True; pos += 1; continue
            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ','): pos += 1
                if pos < len(buffer): break
                require(not eof, 'truncated JSON array'); buffer, pos = '', 0
                block = stream.read(chunk_size); eof = not block; buffer += block
            if buffer[pos] == ']':
                pos += 1
                trailing = buffer[pos:] + stream.read()
                require(not trailing.strip(), 'trailing bytes after JSON array')
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos); pos = end; yield value; break
                except json.JSONDecodeError:
                    require(not eof, 'invalid or truncated JSON array record')
                    buffer, pos = buffer[pos:], 0
                    block = stream.read(chunk_size); eof = not block; buffer += block


def normalized_id(row):
    value = prompt_id(question(row))
    require(re.fullmatch(r'[0-9a-f]{64}', value) is not None, 'invalid normalized prompt identity')
    return value


def official_answer_supported(answer):
    require(isinstance(answer, str) and answer.strip(), 'empty reference answer')
    # The unchanged assignment grader must be able to compare the answer with itself.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
    from utils.grader import check_is_correct
    try: return bool(check_is_correct(answer, answer))
    except Exception: return False


def benchmark_question_bindings():
    files = sorted({p for pattern in ('artifacts/benchmarks*/**/test.jsonl',)
                    for p in Path('.').glob(pattern) if p.is_file()})
    identities = set(); bindings = {}
    for path in files:
        rows = read_rows(path); bindings[str(path)] = {'sha256': digest(path), 'questions': len(rows)}
        for row in rows: identities.add(normalized_id(row))
    require(files and identities, 'actual benchmark question files required for contamination-only audit')
    return identities, bindings


def prior_dev_bindings():
    # Development panels have historically lived both directly under
    # artifacts/dev-* and inside experiment directories.  Exclude every such
    # source panel so the new 200 questions are independent of all earlier
    # screens, regardless of where the panel was stored.
    paths = sorted(p for p in Path('artifacts').glob('**/test.jsonl')
                   if p.is_file() and p.resolve() != DEV.resolve()
                   and any(part.startswith('dev') for part in p.parts))
    identities = set(); bindings = {}
    for path in paths:
        rows = read_rows(path); ids = {normalized_id(row) for row in rows}
        identities |= ids; bindings[str(path)] = {'sha256': digest(path), 'records': len(rows),
                                                   'unique_normalized_ids': len(ids)}
    require(paths, 'the original development panel must exist and be excluded')
    return identities, bindings


def pool_identities():
    require(digest(POOL) == POOL_SHA, 'sealed 83,037-row pool source hash changed')
    ids = set(); count = 0
    for row in iter_json_array(POOL):
        require(isinstance(row, dict), 'invalid pool row')
        pid = normalized_id(row)
        require(row.get('id') == pid, 'pool row ID differs from normalized prompt ID')
        require(pid not in ids, 'duplicate normalized pool prompt ID')
        ids.add(pid); count += 1
    require(count == len(ids) == 83037, 'sealed training pool must contain 83,037 unique prompts')
    return ids


def select_panel(validation, excluded, benchmark_ids, train_ids):
    eligible = []
    for row in validation:
        pid = normalized_id(row)
        require(row.get('id') == pid, 'validation row ID differs from normalized prompt ID')
        answer = boxed(row.get('output', ''))
        if answer is None or not official_answer_supported(answer): continue
        if pid in excluded or pid in benchmark_ids or pid in train_ids: continue
        eligible.append((hashlib.sha256((PANEL_TAG + pid).encode()).hexdigest(), pid, row, answer))
    require(len({pid for _, pid, _, _ in eligible}) == len(eligible), 'duplicate eligible validation identity')
    require(len(eligible) >= 200, 'fewer than 200 eligible held-out rows after frozen exclusions; no backfill')
    chosen = sorted(eligible)[:200]
    return [dict(question=question(row), answer=answer, id=pid) for _, pid, row, answer in chosen]


def prepare_panel():
    require(digest(VALIDATION) == VALIDATION_SHA, 'original validation source hash changed')
    manifest = read(EXTENDED_MANIFEST)
    require(manifest.get('original_pool_sha256') == POOL_SHA
            and manifest.get('original_validation_sha256') == VALIDATION_SHA
            and manifest.get('train_before') == manifest.get('train_after') == 83037
            and manifest.get('validation_before') == manifest.get('validation_after') == 1000
            and manifest.get('heldout_rows_never_moved_to_train') is True,
            'extended source provenance does not establish the frozen held-out split')
    validation = read_rows(VALIDATION)
    require(len(validation) == 1000, 'frozen validation source must contain 1,000 rows')
    validation_ids = {normalized_id(r) for r in validation}
    require(len(validation_ids) == 1000, 'validation source identities are not unique')
    prior_ids, prior_files = prior_dev_bindings()
    benchmark_ids, benchmark_files = benchmark_question_bindings()
    train_ids = pool_identities()
    require(not (validation_ids & train_ids), 'held-out validation overlaps normalized training prompts')
    selected = select_panel(validation, prior_ids, benchmark_ids, train_ids)
    selected_ids = {r['id'] for r in selected}
    require(len(selected) == len(selected_ids) == 200 and not selected_ids & prior_ids
            and not selected_ids & train_ids and not selected_ids & benchmark_ids,
            'new development panel identity/disjointness contract failed')
    by_id = {normalized_id(r): r for r in validation}
    records = []
    for row in selected:
        source = by_id[row['id']]
        require(row['question'] == question(source) and row['answer'] == boxed(source['output']),
                'development reference differs from original held-out rationale bytes')
        records.append({'id': row['id'], 'question_sha256': hashlib.sha256(row['question'].encode()).hexdigest(),
                        'reference_answer_sha256': hashlib.sha256(row['answer'].encode()).hexdigest(),
                        'source_output_sha256': hashlib.sha256(source['output'].encode()).hexdigest()})
    write_jsonl(DEV, selected)
    meta = {'schema': 'two-plan-frozen-development-panel-v1', 'source': str(VALIDATION),
            'source_sha256': VALIDATION_SHA, 'source_rows': 1000,
            'pool_source': str(POOL), 'pool_sha256': POOL_SHA, 'train_unique_ids': 83037,
            'selection_rule': f'smallest SHA256({PANEL_TAG}<normalized_prompt_id>) after frozen exclusions',
            'panel_rows': 200, 'panel_sha256': digest(DEV), 'records': records,
            'excluded_prior_development_files': prior_files,
            'contamination_question_files': benchmark_files,
            'prior_development_unique_ids': len(prior_ids),
            'benchmark_question_unique_ids': len(benchmark_ids),
            'all_ids_normalized_with_reasoning_data_prompt_id': True,
            'all_rows_from_original_validation_bytes': True,
            'all_answers_supported_by_unchanged_assignment_grader': True,
            'training_ids_disjoint': True, 'prior_development_ids_disjoint': True,
            'benchmark_question_ids_disjoint': True,
            'benchmark_answers_or_outcomes_used_for_selection': False,
            'generation_outcomes_used': False, 'no_backfill': True}
    if DEV_META.exists(): require(read(DEV_META) == meta, 'existing frozen panel metadata differs')
    else: write_json(DEV_META, meta)
    if PANEL_RECEIPT.exists(): require(read(PANEL_RECEIPT) == meta, 'existing panel receipt differs')
    else: write_json(PANEL_RECEIPT, meta)
    print(json.dumps({'panel': str(DEV), 'sha256': meta['panel_sha256'], 'rows': 200}))


def model_binding(phase):
    spec = PHASES[phase]; run = spec['model']
    completion_name = 'completion.json' if phase == 'parent' else 'two_plan_completion.json'
    completion = read(run / completion_name)
    if phase == 'parent':
        require(completion.get('global_step') == 5625 and completion.get('epochs') == 3
                and completion.get('finite_updates') == 5625
                and completion.get('all_model_tensors_finite') is True,
                'C1 parent is not the complete finite original FINAL5625 model')
    else:
        require(completion.get('plan') == 'stage2' and completion.get('global_step') == 750
                and completion.get('epochs') == 2 and completion.get('finite_updates') == 750
                and completion.get('all_model_tensors_finite') is True,
                'Stage2 candidate is not complete and finite')
    names = model_shards(run) | {'config.json', 'tokenizer_config.json', 'generation_config.json'}
    require(all((run / n).is_file() and (run / n).stat().st_size > 0 for n in names),
            'model weights or required metadata missing')
    return {'path': str(run), 'phase_label': spec['label'], 'completion_file': completion_name,
            'completion_sha256': digest(run / completion_name),
            'files_sha256': {n: digest(run / n) for n in sorted(names)}}


def input_binding(phase):
    require(PANEL_RECEIPT.exists() and DEV.exists() and DEV_META.exists(), 'frozen 200-row panel not prepared')
    panel = read(PANEL_RECEIPT)
    require(panel.get('panel_rows') == 200 and panel.get('panel_sha256') == digest(DEV)
            and read(DEV_META) == panel, 'frozen panel receipt or bytes changed')
    rows = read_rows(DEV); ids = record_ids(rows)
    require(len(rows) == len(set(ids)) == 200 and ids == [r['id'] for r in rows],
            'development panel does not have exactly 200 unique normalized record IDs')
    model = model_binding(phase)
    output = PHASES[phase]['output']
    outcome_unknown = not output.exists() or not any(output.iterdir())
    binding = {'schema': 'two-plan-development-input-v1', 'phase': phase,
               'phase_label': PHASES[phase]['label'], 'panel_receipt_sha256': digest(PANEL_RECEIPT),
               'panel_sha256': digest(DEV), 'panel_meta_sha256': digest(DEV_META),
               'panel_problem_ids': ids, 'model': model,
               'protocol': {'benchmark': 'acereason_dev', 'temperature': .6, 'top_p': .95,
                            'samples_per_problem': 8, 'max_new_tokens': 32768,
                            'max_model_len': 32768, 'batch_size': 1024, 'max_num_seqs': 128,
                            'tensor_parallel_size': 1, 'gpu_memory_utilization': .85,
                            'seed': 0, 'evaluator': 'reasoning.evaluate'},
               'execution_sha256': {p: digest(p) for p in FILES},
               'source_validation_sha256': VALIDATION_SHA, 'source_pool_sha256': POOL_SHA,
               'source_selection_sha256': digest(SELECTION),
               'training_prompt_count': 83037, 'development_prompt_count': 200,
               'benchmark_data_or_outcomes_used_for_model_or_panel_selection': False,
               'development_outcome_unknown_at_freeze': outcome_unknown,
               'missing_C2_early_checkpoint_not_substituted': True,
               'C5_merge_not_run': True, 'gpu_used': False}
    require(binding['source_selection_sha256'] == SELECTION_SHA, 'original selected-training receipt changed')
    return binding


def prepare(phase):
    spec = PHASES[phase]
    require(not spec['decision'].exists(), f'{phase} development decision already exists')
    if spec['output'].exists(): require(not any(spec['output'].iterdir()), 'development outcomes exist before input freeze')
    if phase == 'stage2':
        require(PARENT_DECISION.exists(), 'measured parent development control required before Stage2 preparation')
        control = read(PARENT_DECISION)
        require(control.get('qualified_control') is True, 'parent truncation rejection leaves no qualified control; fail closed')
    binding = input_binding(phase)
    if spec['input'].exists(): require(read(spec['input']) == binding, f'stale {phase} input binding')
    else: write_json(spec['input'], binding)
    print(f'{phase.upper()}_DEV_CPU_PREFLIGHT_PASSED_NO_GPU')


def journal_clean(ledger):
    values = ledger.get('submission_journal', [])
    values = values.values() if isinstance(values, dict) else values
    require(all(r.get('status') not in ('attempting', 'unknown') for r in values),
            'uncertain scheduler submission requires manual reconciliation')


def auxiliary_budget(auth, central, auxiliary, extra=0):
    if central: validate_ledger(central)
    jobs = central.get('jobs', [])
    charged = sum(j['charged_gpu_hours'] for j in jobs)
    reserves = sum(8 for plan in PLANS
                   if not any(j['plan'] == plan and j['phase'] == 'eval' for j in jobs))
    require(charged + reserves + auth['other_committed_gpu_hours'] <= auth['additional_gpu_hours'],
            'central commitments plus auxiliary reserve exceed authorized incremental budget')
    costs = [j.get('gpu_hours') for j in auxiliary.get('jobs', [])]
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in costs),
            'invalid auxiliary allocation cost')
    require(sum(costs) + extra <= auth['other_committed_gpu_hours'],
            'five-hour auxiliary reserve exhausted; never double-charge or exceed it')


def deadline_fit(now=None):
    now = now or datetime.now(ZoneInfo('America/New_York'))
    require(now + timedelta(hours=10) <= DEADLINE,
            'two-hour screen plus reserved eight-hour final evaluation cannot fit deadline')


def submit(phase, auth_path=AUTH):
    spec = PHASES[phase]; auth = authorization(auth_path)
    expected = input_binding(phase)
    require(spec['input'].exists() and read(spec['input']) == expected, 'exact CPU input preflight missing or stale')
    require(not spec['decision'].exists(), 'development decision already exists; no repeated screen')
    if phase == 'stage2':
        require(PARENT_DECISION.exists() and read(PARENT_DECISION).get('qualified_control') is True,
                'no qualified parent control; Stage2 screen forbidden')
    deadline_fit(); ART.mkdir(parents=True, exist_ok=True)
    ledger_path = ART / 'other_allocations.json'
    with open(str(ART / 'allocations.json') + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            auth = authorization(auth_path)
            central = read(ART/'allocations.json') if (ART/'allocations.json').exists() else {}
            journal_clean(central)
            ledger = read(ledger_path) if ledger_path.exists() else {
                'schema': 'two-plan-other-allocations-v1', 'jobs': [], 'submission_journal': []}
            require(ledger.get('schema') == 'two-plan-other-allocations-v1', 'unknown auxiliary ledger schema')
            journal_clean(ledger)
            require(not any(j.get('purpose') == spec['purpose'] for j in ledger['jobs']),
                    f'one {spec["purpose"]} allocation only')
            auxiliary_budget(auth, central, ledger, 2)
            attempt = {'purpose': spec['purpose'], 'status': 'attempting',
                       'authorization_sha256': digest(auth_path),
                       'input_receipt_sha256': digest(spec['input'])}
            ledger.setdefault('submission_journal', []).append(attempt); write_json(ledger_path, ledger)
            try:
                result = subprocess.run(['sbatch', '--parsable', 'scripts/two_plan_dev_20261019.slurm', phase],
                                        capture_output=True, text=True, timeout=45)
                match = re.fullmatch(r'(\d+)(?:;[\w.-]+)?', result.stdout.strip())
                require(result.returncode == 0 and match is not None,
                        'scheduler outcome unconfirmed; inspect queue and journal manually')
                job = match.group(1)
            except Exception:
                attempt['status'] = 'unknown'; write_json(ledger_path, ledger); raise
            attempt.update(status='confirmed', job_id=job)
            ledger['jobs'].append({'purpose': spec['purpose'], 'job_id': job, 'gpu_hours': 2,
                                   'authorization_sha256': digest(auth_path),
                                   'input_receipt_sha256': digest(spec['input']),
                                   'status': 'submitted_reserved'})
            write_json(ledger_path, ledger)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps({'submitted_job_id': job, 'gpus': 1, 'hours': 2, 'purpose': spec['purpose']}))


def charge_start(phase, auth_path=AUTH):
    spec = PHASES[phase]; auth = authorization(auth_path); job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and int(os.environ['SLURM_GPUS_ON_NODE']) == 1, 'one-B200 allocation required')
    with open(str(ART / 'allocations.json') + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            central = read(ART/'allocations.json') if (ART/'allocations.json').exists() else {}
            ledger = read(ART/'other_allocations.json')
            matches = [j for j in ledger['jobs'] if j.get('purpose') == spec['purpose']]
            require(len(matches) == 1 and matches[0].get('job_id') == job
                    and matches[0].get('gpu_hours') == 2,
                    'running allocation does not match the one reserved actual scheduler ID')
            matches[0]['status'] = 'started_before_preflight'
            auxiliary_budget(auth, central, ledger, 0); write_json(ART/'other_allocations.json', ledger)
        finally: fcntl.flock(lock, fcntl.LOCK_UN)
    deadline_fit()


def result_summary(path):
    manifest, runs = load_run(path)
    require(set(runs) == {'acereason_dev'}, 'screen contains anything other than frozen development panel')
    rows = runs['acereason_dev']; expected = read_rows(DEV); ids = record_ids(expected)
    require(len(rows) == 200 and {r['problem_id'] for r in rows} == set(ids),
            'development results do not contain all and only 200 frozen identities')
    require(all(len(r['generated_responses']) == len(r['answers_correctness'])
                == len(r['finish_reasons']) == len(r['output_tokens']) == 8 for r in rows),
            'development screen is not complete 200x8')
    stats = summarize(rows); trunc = sum(x == 'length' for r in rows for x in r['finish_reasons'])
    solved = sum(any(r['answers_correctness']) for r in rows)
    require(stats['problems'] == 200 and stats['samples_per_problem'] == 8
            and abs(stats['pass_at_8'] - solved/200) <= 1e-12,
            'development metrics arithmetic mismatch')
    return manifest, rows, {'pass_at_1': stats['pass_at_1'], 'pass_at_8': stats['pass_at_8'],
                            'solved_questions': solved, 'truncated_samples': trunc,
                            'total_samples': 1600, 'truncation_fraction': trunc/1600}


def compare_gate(parent, candidate):
    require(parent['total_samples'] == candidate['total_samples'] == 1600, 'incomplete screen totals')
    qualified = parent['truncated_samples'] <= 80
    if not qualified:
        return {'qualified_control': False, 'approved_for_terminal_evaluation': False,
                'reason': 'parent_control_exceeds_five_percent_truncation_no_qualified_control_fail_closed'}
    delta_questions = candidate['solved_questions'] - parent['solved_questions']
    pass_gate = delta_questions >= 2 and candidate['truncated_samples'] <= 80
    informational = None
    if abs(delta_questions) <= 1:
        key_parent = (-parent['truncated_samples'], parent['pass_at_1'])
        key_candidate = (-candidate['truncated_samples'], candidate['pass_at_1'])
        informational = ('candidate' if key_candidate > key_parent else
                         'parent' if key_parent > key_candidate else 'tie')
    return {'qualified_control': True, 'approved_for_terminal_evaluation': pass_gate,
            'candidate_minus_parent_pass_at_8': candidate['pass_at_8'] - parent['pass_at_8'],
            'candidate_minus_parent_solved_questions': delta_questions,
            'required_additional_solved_questions': 2,
            'candidate_truncation_qualified': candidate['truncated_samples'] <= 80,
            'informational_tie_policy_winner': informational,
            'tie_policy_scope': 'informational only when within one solved question: fewer truncations, then pass@1',
            'reason': 'strict_predeclared_gate_passed' if pass_gate else 'strict_predeclared_gate_failed'}


def run(phase, auth_path=AUTH):
    charge_start(phase, auth_path); spec = PHASES[phase]
    expected = input_binding(phase)
    require(read(spec['input']) == expected, 'runtime input/model/source binding differs from CPU preflight')
    require(not spec['decision'].exists(), 'development decision already exists; refuse overwrite')
    # Use the same pinned evaluation environment as the existing full-suite
    # runs.  The training environment does not expose all package metadata
    # recorded by reasoning.evaluate's immutable manifest.
    command = [str(ROOT/'.venv-arc/bin/python'), '-m', 'reasoning.evaluate',
               '--model', str(spec['model']), '--data-dir', str(DEV_ROOT),
               '--benchmarks', 'acereason_dev', '--output', str(spec['output']),
               '--seed', '0', '--batch-size', '1024', '--tensor-parallel-size', '1',
               '--max-num-seqs', '128', '--record-hardware', '--gpu-memory-utilization', '.85',
               '--max-new-tokens', '32768', '--max-model-len', '32768']
    subprocess.run(command, check=True)
    manifest, _, measured = result_summary(spec['output'])
    require(manifest['data'] == {'acereason_dev': digest(DEV)}
            and manifest['problem_counts'] == {'acereason_dev': 200}
            and manifest.get('temperature') == .6 and manifest.get('top_p') == .95
            and manifest.get('n') == 8 and manifest.get('max_tokens') == 32768
            and manifest.get('max_model_len') == 32768 and manifest.get('batch_size') == 1024
            and manifest.get('max_num_seqs') == 128 and manifest.get('tensor_parallel_size') == 1
            and manifest.get('gpu_memory_utilization') == .85,
            'actual evaluator manifest differs from frozen development protocol')
    base = {'schema': 'two-plan-development-decision-v1', 'phase': phase,
            'phase_label': spec['label'], 'job_id': os.environ['SLURM_JOB_ID'],
            'input_binding': expected, 'input_receipt_sha256': digest(spec['input']),
            'result_manifest_sha256': digest(spec['output']/'manifest.json'), 'metrics': measured,
            'all_200_questions_x_8_samples_complete': True,
            'truncation_rejection_rule': '>80 of 1600 (>5%)',
            'public_benchmark_outcomes_used': False,
            'missing_C2_early_checkpoint_not_substituted': True, 'C5_merge_not_run': True}
    if phase == 'parent':
        qualified = measured['truncated_samples'] <= 80
        base.update(qualified_control=qualified,
                    approved_to_submit_stage2_screen=qualified,
                    reason=('qualified_parent_control' if qualified else
                            'parent_control_exceeds_five_percent_truncation_no_qualified_control_fail_closed'))
        write_json(PARENT_DECISION, base)
        return 0 if qualified else 2
    require(PARENT_DECISION.exists(), 'parent decision missing at Stage2 completion')
    parent_decision = read(PARENT_DECISION)
    require(parent_decision.get('qualified_control') is True, 'parent is not a qualified control')
    parent_manifest, _, parent_metrics = result_summary(PARENT_OUT)
    check_protocol(parent_manifest, manifest)
    gate = compare_gate(parent_metrics, measured)
    require(gate['qualified_control'] is True, 'qualified-control receipt and recomputed result disagree')
    base.update(parent_job_id=parent_decision['job_id'],
                parent_input_receipt_sha256=digest(PARENT_INPUT),
                parent_result_manifest_sha256=digest(PARENT_OUT/'manifest.json'),
                parent_metrics=parent_metrics, gate=gate,
                source_model_identities={'parent': parent_decision['input_binding']['model'],
                                         'candidate': expected['model']})
    write_json(STAGE2_DECISION, base)
    return 0 if gate['approved_for_terminal_evaluation'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ('prepare-panel', 'prepare-parent', 'submit-parent', 'prepare-stage2',
                 'submit-stage2', 'run-parent', 'run-stage2'):
        mode.add_argument('--' + name, action='store_true')
    p.add_argument('--authorization', type=Path, default=AUTH); a = p.parse_args()
    if a.prepare_panel: prepare_panel()
    elif a.prepare_parent: prepare('parent')
    elif a.submit_parent: submit('parent', a.authorization)
    elif a.prepare_stage2: prepare('stage2')
    elif a.submit_stage2: submit('stage2', a.authorization)
    elif a.run_parent: return run('parent', a.authorization)
    else: return run('stage2', a.authorization)


if __name__ == '__main__': raise SystemExit(main())
