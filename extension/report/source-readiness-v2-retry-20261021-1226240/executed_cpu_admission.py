"""One metadata-only admission for an explicitly versioned CPU fixture repair.

Preserve fifteen prior entries and both pending/running GPU diagnostics. All
private calibration and test work belongs to the new compute allocation.
"""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import submission as m
from extension import joint_gpu_admission as gpu
from extension import source_readiness_v2_retry as stage
from extension.cpu_feasibility import Evidence, value_digest

SELF = 'extension/operations/submit-source-readiness-v2-retry-20261021.py'
SCRIPT = 'extension/source_readiness_v2_retry.slurm'
NAME = 'sft-extension-source-readiness-v2-calibration-retry'
INSTALL = 'extension/source_readiness_v2_retry_source_receipt.json'
INSTALL_SHA256 = '63eff3e99ae3e0b3933926d4fdef95674b869f4af9257af3830eb8e547e925ca'
PLAN_SHA256 = '7526ff1b21dd2a8a53c39f47c91981ffe91496cd5f1270b9cddf50974cf3c39b'

def rec(root, name):
    return gpu._record(root, name)

def guard():
    now = datetime.now(timezone.utc)
    m.require((m.DEADLINE - now).total_seconds() >= 180000, 'Keep full2h work plus48h reserve')
    return now

def priors(ledger):
    gpu._ledger(ledger)
    values = {k: v for k, v in ledger['jobs'].items() if k != stage.ROLE}
    m.require(len(values) == 15 and value_digest(values) == stage.PRIOR_ENTRIES_SHA256,
        'Preserve all fifteen exact prior entries, including failed1225820')
    m.require(len(gpu.scheduler_identities(values)) == 14, 'Only the sealed DNS rejection may lack a jobID')
    return values

def prerequisites(root, prior):
    m.require(re.fullmatch('[0-9a-f]{64}', INSTALL_SHA256) and re.fullmatch('[0-9a-f]{64}', PLAN_SHA256),
        'Root must first freeze exact source installation and prospective plan')
    install_record = rec(root, INSTALL)
    m.require(install_record['sha256'] == INSTALL_SHA256, 'Exact reviewed installed91-source manifest required')
    install = m._read(root / INSTALL)
    sources = stage.source_inventory(root)
    m.require(len(sources) == 91 and install.get('source_inventory') == sources and install.get('installed_on_arc') is True,
        'The installed ninety-one sources differ from their root freeze')
    evidence = Evidence()
    plan, _, _ = stage._read_plans(root, root / stage.PLAN, evidence)
    m.require(rec(root, stage.PLAN)['sha256'] == PLAN_SHA256 and plan['native_qa']['new_readiness_tests'] == 28,
        'Exact prospective478-check execution plan required')
    values = stage.failure_evidence(root, evidence)
    m.require(values['job_ledger_snapshot.json']['jobs'] == prior, 'Failure snapshot and current fifteen entries differ')
    fourteen = {k: v for k, v in prior.items() if k != stage.FAILED_ROLE}
    parent = {'schema': m.SCHEMA, 'historical_ledger_unchanged': 'deadline/job_ledger.json', 'gpu_hour_cap': None, 'jobs': fourteen}
    inherited = gpu.load_evidence(root, 'extension/runs/joint-memory-qa-1225557',
        'extension/runs/joint-gpu-metadata-qa-1225720/native_qa.json',
        'd15f2a5de33c4fdb20a1a4b20b91e55ac3fb6d247f998a9712c061484deef39b', parent)
    m.require(set(fourteen) == set(inherited['prior']) | {gpu.ROLE}, 'Keep the established tested GPU handoff')
    current = prior[gpu.ROLE]
    gpu._entry(current, gpu.ROLE, gpu_seconds=7200, sources=inherited['sources'], job='1225727')
    path, expected = gpu._admission_receipt(root, current, inherited)
    m.require(m._read(path) == expected and rec(root, str(path.relative_to(root)))['sha256'] == current['admission_receipt_sha256'],
        'The already-admitted GPU handoff must remain exact')
    assets = {**inherited['assets'], INSTALL: install_record, stage.PLAN: rec(root, stage.PLAN), SELF: rec(root, SELF)}
    for name, value in evidence.files.items():
        p = Path(name)
        m.require(p.is_relative_to(root), 'Retry metadata must stay inside project')
        assets[str(p.relative_to(root))] = rec(root, str(p.relative_to(root)))
    assets[str(path.relative_to(root))] = rec(root, str(path.relative_to(root)))
    directives = [line.strip() for line in (root / SCRIPT).read_text().splitlines() if line.lstrip().startswith('#SBATCH')]
    expected = ['#SBATCH --job-name=' + NAME, '#SBATCH --account=tml_2026', '#SBATCH --partition=normal_q',
        '#SBATCH --qos=owl_normal_short', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8',
        '#SBATCH --mem=64G', '#SBATCH --time=02:00:00', '#SBATCH --no-requeue',
        '#SBATCH --output=' + str(root / 'extension/logs/extension-source-readiness-v2-calibration-retry-%j.out'),
        '#SBATCH --error=' + str(root / 'extension/logs/extension-source-readiness-v2-calibration-retry-%j.err')]
    m.require(directives == expected, 'Only exactCPU8/64G/2h/zeroGPU/no-requeue wrapper is allowed')
    evidence.unchanged()
    return {'sources': sources, 'assets': assets, 'inherited': inherited}

def check_queue(snapshot, prior, proof):
    inherited = proof['inherited']
    gpu._known_queue(snapshot, inherited['prior'], inherited['metadata_value'], inherited['metadata_name'])
    fixed = {'1225727': (gpu.NAME, 1), '1225820': ('sft-extension-source-readiness-v2-calibration', 0)}
    for row in snapshot['jobs']:
        if row['job_id'] in fixed:
            name, gpus = fixed[row['job_id']]
            m.require(row['name'] == name and row['gpus'] in ({0, 1} if gpus else {0}), 'Known scheduler identity/resources changed')
            if row['job_id'] == '1225820': m.require(row['state'] == 'FAILED', 'Earlier failed CPU stage cannot be active')
            elif row['state'] in gpu.NONTERMINAL: m.require(row['gpus'] == 1, 'Known live GPU must still request one')
        elif row['name'] in {v[0] for v in fixed.values()}:
            raise m.SubmissionError('Untracked duplicate prior stage')
    m.require(len(gpu.scheduler_identities(prior)) == 14, 'Known prior identities changed')

def main():
    root = Path.cwd().resolve()
    m.require(Path(__file__).resolve() == root / SELF, 'Run only this installed one-use adapter from root')
    guard()
    path, lockpath = root / 'extension/job_ledger.json', root / 'extension/job_ledger.lock'
    m.require(not path.is_symlink() and not lockpath.is_symlink(), 'No ledger/lock aliases')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = m._read(path); prior = priors(ledger); proof = prerequisites(root, prior)
        command = ['sbatch', '--clusters=owl', '--parsable', str(root / SCRIPT)]
        binding = {'role': stage.ROLE, 'command': command, 'source_sha256': proof['sources'],
            'admission_code_sha256': proof['assets'][SELF]['sha256'], 'plan_sha256': PLAN_SHA256,
            'scientific_plan_sha256': stage.SCIENTIFIC_PLAN['sha256'], 'legacy_plan_sha256': stage.LEGACY_PLAN['sha256'],
            'worker_protocol_sha256': stage.PROTOCOL_PIN['sha256'], 'prior_entries_sha256': value_digest(prior),
            'initial_evidence': proof['assets'], 'metadata_source_sha256': proof['inherited']['parser_sources'],
            'expected_native_tests': 478, 'requested_gpu_hours': 0, 'requested_gpu_seconds': 0, 'wall_time_seconds': 7200,
            'closeout_reserve_seconds': 172800, 'deadline_utc': m.DEADLINE.isoformat(), 'automatic_downstream': False}
        checksum = value_digest(binding)
        snapshot = gpu.queue_snapshot(subprocess.run); check_queue(snapshot, prior, proof)
        m.require(m._read(path) == ledger and priors(ledger) == prior and prerequisites(root, prior) == proof,
            'All source/evidence/self bytes and ledger must remain unchanged during snapshot')
        same = {row['job_id'] for row in snapshot['jobs'] if row['name'] == NAME}
        current = ledger['jobs'].get(stage.ROLE)
        if current is not None:
            m.require(current.get('status') == 'submitted' and current.get('binding') == binding and
                current.get('binding_sha256') == checksum and current.get('source_sha256') == proof['sources'] and
                current.get('requested_gpu_seconds') == 0 and current.get('requested_gpu_hours') == 0 and
                gpu._numeric_job(current.get('job_id')) and not same - {current['job_id']},
                'Preserve any existing/rejected/uncertain retry; never resubmit')
            for row in snapshot['jobs']:
                if row['job_id'] == current['job_id']:
                    m.require(row['name'] == NAME and row['gpus'] == 0, 'Existing current CPU identity/resources changed')
            print(current['job_id'], flush=True); return
        m.require(not same, 'Untracked current CPU job needs reconciliation')
        when = guard(); environment, removed = m._submission_environment()
        m.require((root / 'extension/logs').is_dir() and not (root / 'extension/logs').is_symlink(), 'Reviewed log directory required')
        current = {'status': 'submitting', 'binding': binding, 'binding_sha256': checksum, 'source_sha256': proof['sources'],
            'plan_sha256': PLAN_SHA256, 'command': command, 'requested_gpu_hours': 0, 'requested_gpu_seconds': 0,
            'requested_at_utc': when.isoformat(), 'scheduler_snapshot': snapshot, 'removed_scheduler_environment_keys': removed}
        ledger['jobs'][stage.ROLE] = current; m._write(path, ledger)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            current.update(status='uncertain', error_type=type(error).__name__); m._write(path, ledger); raise
        current.update(scheduler_returncode=result.returncode, scheduler_stdout=str(result.stdout)[-2000:], scheduler_stderr=str(result.stderr)[-2000:])
        match = re.fullmatch(r'([1-9][0-9]*)(?:;owl)?', result.stdout.strip()) if isinstance(result.stdout, str) else None
        if result.returncode or match is None or not isinstance(result.stderr, str) or result.stderr.strip():
            current['status'] = 'submission_failed' if result.returncode else 'uncertain'
            if match is not None: current['possible_job_id'] = match.group(1)
            m._write(path, ledger); raise m.SubmissionError('Preserve rejected/ambiguous request without automatic retry')
        job = match.group(1); current.update(status='uncertain', job_id=job); m._write(path, ledger)
        m.require(job not in gpu.scheduler_identities(prior) and job not in {v['job_id'] for v in snapshot['jobs']}, 'Scheduler identity collision')
        current.update(status='submitted', submitted_at_utc=datetime.now(timezone.utc).isoformat()); m._write(path, ledger)
        print(job, flush=True)

if __name__ == '__main__': main()
