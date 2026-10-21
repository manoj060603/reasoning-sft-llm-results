"""Metadata-only admission for one original-only, interface-checking calibration.

No private prompt cache, model shard, inference or grader is opened here. Native
CPU fixtures and terminal accounting must precede this finite GPU request.
"""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess
from extension import submission as m
from extension import joint_gpu_admission as shared
from extension import source_generation as generation
from extension.cpu_feasibility import value_digest

ROLE = generation.ROLE
QA_ROLE = 'source_generation_qa_20261021_v1'
NAME = 'sft-extension-original-source-calibration'
QA_NAME = 'sft-extension-source-generation-qa'
SCRIPT = 'extension/source_generation.slurm'
PARENT_SNAPSHOT = generation.PARENT_DIRECTORY + '/job_ledger_snapshot.json'
PARENT_SNAPSHOT_SHA256 = '56db3ba37128483d25917c3f18713aba1988a6b4b37d084a1020754febe8f6ed'
PARENT_ENTRIES_SHA256 = '79f6394b067d4022d7c1e92f153c5320faa6ac2f72e2192dd9501119c68a00b8'
METADATA_TESTS, COLLECTOR_TESTS = 16, 32
CPU_SECONDS, GPU_SECONDS, RESERVE_SECONDS = 1200, 7200, 172800
NEW_METADATA = frozenset({'extension/source_generation_submission.py', 'tests/test_source_generation_submission.py',
    'extension/source_generation_qa.py', 'extension/source_generation_qa.slurm'})
OLD_METADATA = frozenset({'extension/joint_gpu_admission.py', 'tests/test_joint_gpu_admission.py',
    'extension/joint_gpu_metadata_qa.py', 'extension/joint_gpu_metadata_qa.slurm'})

def metadata_inventory(root):
    return {name: shared._record(root, name) for name in sorted(NEW_METADATA | OLD_METADATA)}

def guard(now=None):
    now = datetime.now(timezone.utc) if now is None else now
    shared.require(isinstance(now, datetime) and now.tzinfo is not None and
        (m.DEADLINE - now).total_seconds() >= GPU_SECONDS + RESERVE_SECONDS,
        'Keep the full finite job and48-hour closeout reserve')
    return now

def initial_evidence(root):
    """Read only bounded source, plan, ledger and native receipt metadata."""
    root = Path(root).resolve()
    assets = {}
    def read(name, checksum=None):
        assets[name] = shared._record(root, name)
        shared.require(checksum is None or assets[name]['sha256'] == checksum, 'Pinned metadata changed: ' + name)
        return m._read(root / name)
    snapshot = read(PARENT_SNAPSHOT, PARENT_SNAPSHOT_SHA256)
    shared._ledger(snapshot)
    initial = snapshot['jobs']
    shared.require(len(initial) == 16 and value_digest(initial) == PARENT_ENTRIES_SHA256 and
        QA_ROLE not in initial and ROLE not in initial and len(shared.scheduler_identities(initial)) == 15,
        'Keep all sixteen actual admissions and only the sealed no-ID exception')
    parents = {name: read(generation.PARENT_DIRECTORY + '/' + name, checksum)
        for name, checksum in generation.PARENT_PINS.items()}
    complete, qa = parents['complete.json'], parents['unit_qa.json']
    shared.require(complete.get('complete') is True and complete.get('job_id') == '1226240' and
        complete.get('tests_run') == 478 and complete.get('gold_self_comparisons_passed') == 32 and
        complete.get('native_gold_mutation_invariance_passed') is True and complete.get('input_snapshot_unchanged') is True and
        complete.get('output_sha256', {}).get('calibration_prompt_cache.json') == generation.CACHE['sha256'] and
        complete.get('native_qa_sha256') == generation.PARENT_PINS['unit_qa.json'], 'Exact source-only readiness478/cache binding required')
    shared._passed_qa(qa, generation.prepared_stage.QA_SCHEMA)
    shared.validate_accounting(parents['accounting.json'], '1226240', name='sft-extension-source-readiness-v2-calibration-retry')
    core = generation.source_inventory(root)
    shared.require(len(core) == 95 and all(core.get(n) == item for n, item in complete['source_inventory'].items()),
        'All frozen parent91 source bytes must remain unchanged')
    plan = generation.validate_plan(read(generation.PLAN))
    parser = read('extension/report/joint-gpu-metadata-qa-20261021-1225720/native_qa.json',
        'd15f2a5de33c4fdb20a1a4b20b91e55ac3fb6d247f998a9712c061484deef39b')
    metadata = metadata_inventory(root)
    shared.require(set(parser['source_inventory']) == OLD_METADATA and
        all(metadata[n] == item for n, item in parser['source_inventory'].items()), 'Tested parser4 must remain exact')
    assets['deadline/job_ledger.json'] = shared._record(root, 'deadline/job_ledger.json')
    shared.require(assets['deadline/job_ledger.json']['sha256'] == shared.qa.HISTORICAL_LEDGER_SHA256, 'Historical ledger changed')
    return {'initial': initial, 'assets': assets, 'sources': core, 'metadata': metadata,
        'plan_sha256': assets[generation.PLAN]['sha256'], 'plan': plan}

def validate_wrapper(root):
    text = m._small_bytes(Path(root) / SCRIPT, Path(root)).decode()
    expected = {'job-name': NAME, 'account': 'tml_2026', 'partition': 'b200_normal_q', 'qos': 'owl_b200_normal_short',
        'nodes': '1', 'ntasks': '1', 'gres': 'gpu:1', 'cpus-per-task': '8', 'mem': '128G', 'time': '02:00:00', 'no-requeue': ''}
    allowed, found, header = set(expected) | {'output', 'error'}, {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith('#SBATCH'):
            if line.strip() and not line.lstrip().startswith('#'): header = False
            continue
        shared.require(header, 'Late SBATCH directive')
        parts = shlex.split(line.lstrip()[7:].strip())
        shared.require(bool(parts), 'Empty directive')
        key, separator, arg = parts[0].partition('=')
        shared.require(key.startswith('--') and key[2:] in allowed and key[2:] not in found and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), 'Unknown/duplicate/ambiguous directive')
        found[key[2:]] = arg if separator else parts[1] if len(parts) == 2 else ''
    shared.require(set(found) == allowed and all(found[k] == v for k, v in expected.items()), 'Exact singleB200/8CPU/128G/2h wrapper required')
    for key in ('output', 'error'):
        p = Path(found[key]); shared.require(p.is_absolute() and '..' not in p.parts and p.parent == Path(root) / 'extension/logs', 'Bounded extension logs required')
    return found

def validate_history(ledger, initial, cpu_job):
    shared._ledger(ledger)
    shared.require(set(ledger['jobs']) <= set(initial) | {QA_ROLE, ROLE} and
        all(ledger['jobs'].get(k) == v for k, v in initial.items()), 'Unknown or rewritten historical admission')
    entry = ledger['jobs'].get(QA_ROLE, {})
    shared._entry(entry, QA_ROLE, gpu_seconds=0, job=cpu_job)
    shared.require(entry['binding'].get('prior_entries_sha256') == PARENT_ENTRIES_SHA256 and
        entry['binding'].get('wall_time_seconds') == CPU_SECONDS, 'CPU QA must bind all sixteen prior entries and20min ceiling')
    shared.require(cpu_job not in shared.scheduler_identities(initial), 'CPU QA must have a distinct actual allocation')
    return {**initial, QA_ROLE: entry}

def validate_native_qa(qa, proof, prior):
    counts = {'test_source_generation.py': COLLECTOR_TESTS, 'test_source_generation_submission.py': METADATA_TESTS}
    shared.require(qa.get('tests_run') == 48 and qa.get('test_counts') == counts and qa.get('parent_tests_previously_passed') == 478 and
        qa.get('parent_tests_replayed') is False and qa.get('source_inventory') == proof['sources'] and
        qa.get('code_sha256') == {n: v['sha256'] for n, v in proof['sources'].items()} and
        qa.get('metadata_source_inventory') == proof['metadata'] and qa.get('plan_sha256') == proof['plan_sha256'] and
        qa.get('cache_sha256') == generation.CACHE['sha256'] and qa.get('parent_complete_sha256') == generation.PARENT_PINS['complete.json'] and
        qa.get('actual_vllm_interface_established') is False and qa.get('admission_binding_sha256') == prior[QA_ROLE]['binding_sha256'],
        'Exact native48 test/source/cache/admission bindings are required; old478 not replayed')
    shared.require(prior[QA_ROLE].get('source_sha256') == proof['sources'] == prior[QA_ROLE]['binding'].get('source_sha256') and
        prior[QA_ROLE]['binding'].get('metadata_source_inventory') == proof['metadata'], 'CPU source103 inventory changed')
    return qa

def load_evidence(root, qa_run, qa_sha256, ledger):
    root = Path(root).resolve(); proof = initial_evidence(root)
    directory = shared._inside(root, qa_run, runs=True, directory=True)
    qa_path = directory / 'native_qa.json'
    shared.require(shared._checksum(qa_sha256) and shared._record(root, qa_path)['sha256'] == qa_sha256, 'Caller-pinned native QA required')
    qa = m._read(qa_path); shared._passed_qa(qa, generation.QA_SCHEMA)
    shared.require(directory.name == 'source-generation-qa-' + qa['job_id'] and directory.parent == root / 'extension/runs', 'Exact native CPU run namespace required')
    prior = validate_history(ledger, proof['initial'], qa['job_id'])
    validate_native_qa(qa, proof, prior)
    terminal_path = directory / 'accounting.json'; terminal = m._read(terminal_path)
    shared.validate_accounting(terminal, qa['job_id'], name=QA_NAME)
    shared.require(terminal['elapsed_seconds'] <= CPU_SECONDS, 'CPU QA exceeded finite ceiling')
    proof['assets'].update({str(qa_path.relative_to(root)): shared._record(root, qa_path),
        str(terminal_path.relative_to(root)): shared._record(root, terminal_path)})
    validate_wrapper(root)
    return {**proof, 'prior': prior, 'qa_run': str(directory.relative_to(root)), 'qa_sha256': qa_sha256,
        'qa_job': qa['job_id'], 'qa_value': qa}

def known_queue(root, snapshot, prior):
    identities = shared.scheduler_identities(prior)
    by_job, by_name = {}, {}
    for value in prior.values():
        command = value.get('command', value.get('binding', {}).get('command', []))
        scripts = [part for part in command if isinstance(part, str) and part.endswith('.slurm')]
        shared.require(len(scripts) == 1, 'Exactly one bound wrapper must identify each prior role')
        script = Path(scripts[0]); script = script if script.is_absolute() else Path(root) / script
        shared.require(script.resolve().is_relative_to(Path(root)), 'Prior wrapper escaped project')
        names = [line.strip().split('=', 1)[1] for line in m._small_bytes(script, Path(root)).decode().splitlines()
            if line.strip().startswith('#SBATCH --job-name=')]
        shared.require(len(names) == 1 and names[0], 'Prior wrapper lacks one exact job name')
        job = value.get('job_id'); by_name.setdefault(names[0], set()).add(job)
        if shared._numeric_job(job): by_job[job] = names[0]
    for row in snapshot['jobs']:
        if row['job_id'] in identities:
            expected = identities[row['job_id']]
            shared.require(row['name'] == by_job[row['job_id']] and
                row['gpus'] in ({0} if expected.get('requested_gpu_seconds') == 0 else {0, 1}), 'Known name/resource changed')
            if expected.get('requested_gpu_seconds') and row['state'] in shared.NONTERMINAL:
                shared.require(row['gpus'] == 1, 'Known live GPU resource changed')
        elif any(row['job_id'].split('_')[0] == job for job in identities):
            raise m.SubmissionError('Unknown scheduler array or duplicate identity')
        elif row['name'] in by_name or row['name'] == NAME:
            raise m.SubmissionError('Untracked duplicate prior/current role')
    return by_job

def handoff(root, entry, proof):
    job = entry['job_id']; path = Path(root) / 'extension/operations' / ('source-generation-original-job-' + job + '.json')
    value = {'schema': generation.ADMISSION_SCHEMA, 'status': 'admitted', 'job_id': job,
        'plan_sha256': proof['plan_sha256'], 'qa_sha256': proof['qa_sha256'],
        'code_sha256': {n: v['sha256'] for n, v in proof['sources'].items()}, 'binding_sha256': entry['binding_sha256'],
        'cache_sha256': generation.CACHE['sha256'], 'authorize_first_interface_checking_generation': True, 'automatic_downstream': False}
    return path, value

def submit(root, qa_run, qa_sha256, *, now=None, runner=None):
    root = Path(root).resolve(); runner = subprocess.run if runner is None else runner; guard(now)
    path, lockpath = root / 'extension/job_ledger.json', root / 'extension/job_ledger.lock'
    shared.require(not path.is_symlink() and not lockpath.is_symlink(), 'No ledger aliases')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); ledger = m._read(path); proof = load_evidence(root, qa_run, qa_sha256, ledger)
        command = ['sbatch', '--clusters=owl', '--parsable', str(root / SCRIPT), proof['qa_run'] + '/native_qa.json']
        binding = {'role': ROLE, 'command': command, 'source_sha256': proof['sources'], 'metadata_source_inventory': proof['metadata'],
            'plan_sha256': proof['plan_sha256'], 'qa_sha256': proof['qa_sha256'], 'cache_sha256': generation.CACHE['sha256'],
            'profile_sha256': generation.PROFILE_SHA256, 'parent_complete_sha256': generation.PARENT_PINS['complete.json'],
            'model_role': 'original', 'authorize_first_interface_checking_generation': True, 'requested_gpu_seconds': GPU_SECONDS,
            'requested_gpu_hours': 2, 'wall_time_seconds': GPU_SECONDS, 'closeout_reserve_seconds': RESERVE_SECONDS,
            'prior_entries_sha256': value_digest(proof['prior']), 'initial_evidence': proof['assets'],
            'automatic_retry': False, 'automatic_downstream': False}
        checksum = value_digest(binding); snapshot = shared.queue_snapshot(runner)
        existing = ledger['jobs'].get(ROLE)
        known_queue(root, snapshot, {**proof['prior'], **({ROLE: existing} if existing is not None else {})})
        shared.require(m._read(path) == ledger and load_evidence(root, qa_run, qa_sha256, ledger) == proof, 'Source/evidence/ledger changed during queue snapshot')
        same = {row['job_id'] for row in snapshot['jobs'] if row['name'] == NAME}
        if existing is not None:
            shared._entry(existing, ROLE, gpu_seconds=GPU_SECONDS, sources=proof['sources'])
            shared.require(existing.get('binding') == binding and existing.get('binding_sha256') == checksum and not same - {existing['job_id']}, 'Never retry existing or ambiguous calibration')
            shared.check_capacity(snapshot, already_submitted=True)
            target, expected = handoff(root, existing, proof)
            shared.require(target.is_file() and m._read(target) == expected, 'Existing handoff missing; reconcile without resubmission')
            return existing['job_id']
        shared.require(not same, 'Untracked calibration requires reconciliation'); shared.check_capacity(snapshot)
        when = guard(now); environment, removed = m._submission_environment()
        entry = {'status': 'submitting', 'binding': binding, 'binding_sha256': checksum, 'source_sha256': proof['sources'],
            'command': command, 'requested_gpu_seconds': GPU_SECONDS, 'requested_gpu_hours': 2,
            'requested_at_utc': when.isoformat(), 'scheduler_snapshot': snapshot, 'removed_scheduler_environment_keys': removed}
        ledger['jobs'][ROLE] = entry; m._write(path, ledger)
        try: response = runner(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status='uncertain', error_type=type(error).__name__); m._write(path, ledger); raise
        entry.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:], scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r'([1-9][0-9]*)(?:;owl)?', response.stdout.strip()) if isinstance(response.stdout, str) else None
        if response.returncode or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            entry['status'] = 'submission_failed' if response.returncode else 'uncertain'
            if match: entry['possible_job_id'] = match.group(1)
            m._write(path, ledger); raise m.SubmissionError('Preserve rejected/ambiguous calibration request; no retry')
        job = match.group(1); entry.update(status='uncertain', job_id=job); m._write(path, ledger)
        shared.require(job not in shared.scheduler_identities(proof['prior']) and job not in {r['job_id'] for r in snapshot['jobs']}, 'Returned scheduler identity collision')
        target, value = handoff(root, entry, proof)
        shared.require(not target.exists() and not target.is_symlink(), 'Never overwrite existing handoff')
        m._write(target, value); entry.update(status='submitted', submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            admission_receipt_sha256=shared._record(root, target)['sha256']); m._write(path, ledger)
        return job

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', default='.')
    parser.add_argument('--qa-run', required=True); parser.add_argument('--qa-sha256', required=True); a = parser.parse_args()
    print(submit(a.root, a.qa_run, a.qa_sha256), flush=True)

if __name__ == '__main__': main()
