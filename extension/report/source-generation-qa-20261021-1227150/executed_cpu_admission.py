"""One fixed metadata-only CPU48 admission; no model/private cache/vLLM opens."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import source_generation_submission as stage
from extension.cpu_feasibility import value_digest

SELF = 'extension/operations/submit-source-generation-qa-20261021.py'
SCRIPT = 'extension/source_generation_qa.slurm'
INSTALL = 'extension/source_generation_source_receipt.json'
INSTALL_SHA256 = '79cf65df73f2f18930b67ac830cd5bb71bf0ff2e0fc93d846892b77347ae3415'
PLAN_SHA256 = 'f1b5fd13f75e4c7e5916caf4210a397605828cb0cd896c1609a3edc1a726a19e'

def guard():
    now = datetime.now(timezone.utc)
    stage.shared.require((stage.m.DEADLINE - now).total_seconds() >= 174000, 'Keep20min work plus48h reserve')
    return now

def prerequisites(root):
    stage.shared.require(stage.shared._checksum(INSTALL_SHA256) and stage.shared._checksum(PLAN_SHA256), 'Root must first freeze103 sources')
    install = stage.shared._record(root, INSTALL)
    stage.shared.require(install['sha256'] == INSTALL_SHA256, 'Exact source installation required')
    manifest = stage.m._read(root / INSTALL); proof = stage.initial_evidence(root)
    stage.shared.require(manifest.get('installed_on_arc') is True and manifest.get('core_source_inventory') == proof['sources'] and
        manifest.get('metadata_source_inventory') == proof['metadata'] and proof['plan_sha256'] == PLAN_SHA256,
        'All95 core and8 metadata sources must equal the reviewed installation')
    proof['assets'][INSTALL] = install; proof['assets'][SELF] = stage.shared._record(root, SELF)
    stage.validate_wrapper(root)
    directives = [line.strip() for line in (root / SCRIPT).read_text().splitlines() if line.lstrip().startswith('#SBATCH')]
    expected = ['#SBATCH --job-name=' + stage.QA_NAME, '#SBATCH --account=tml_2026', '#SBATCH --partition=normal_q',
        '#SBATCH --qos=owl_normal_short', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=2',
        '#SBATCH --mem=8G', '#SBATCH --time=00:20:00', '#SBATCH --no-requeue',
        '#SBATCH --output=' + str(root / 'extension/logs/extension-source-generation-qa-%j.out'),
        '#SBATCH --error=' + str(root / 'extension/logs/extension-source-generation-qa-%j.err')]
    stage.shared.require(directives == expected, 'Only exactCPU2/8G/20min/noGPU/no-requeue wrapper allowed')
    return proof

def main():
    root = Path.cwd().resolve(); stage.shared.require(Path(__file__).resolve() == root / SELF, 'Run installed adapter from root')
    guard(); path, lockpath = root / 'extension/job_ledger.json', root / 'extension/job_ledger.lock'
    stage.shared.require(not path.is_symlink() and not lockpath.is_symlink(), 'No ledger/lock aliases')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); ledger = stage.m._read(path); stage.shared._ledger(ledger); proof = prerequisites(root)
        before = {key: value for key, value in ledger['jobs'].items() if key != stage.QA_ROLE}
        stage.shared.require(before == proof['initial'] and len(before) == 16, 'Keep every one of sixteen actual prior admissions')
        command = ['sbatch', '--clusters=owl', '--parsable', str(root / SCRIPT)]
        binding = {'role': stage.QA_ROLE, 'command': command, 'source_sha256': proof['sources'],
            'metadata_source_inventory': proof['metadata'], 'plan_sha256': proof['plan_sha256'],
            'cache_sha256': stage.generation.CACHE['sha256'], 'initial_evidence': proof['assets'],
            'admission_code_sha256': proof['assets'][SELF]['sha256'], 'prior_entries_sha256': stage.PARENT_ENTRIES_SHA256,
            'requested_gpu_seconds': 0, 'requested_gpu_hours': 0, 'wall_time_seconds': 1200, 'closeout_reserve_seconds': 172800,
            'expected_native_tests': 48, 'parent_tests_previously_passed': 478, 'parent_tests_replayed': False,
            'automatic_downstream': False}
        checksum = value_digest(binding); snapshot = stage.shared.queue_snapshot(subprocess.run)
        stage.known_queue(root, snapshot, proof['initial']); stage.shared.check_capacity(snapshot, already_submitted=True)
        stage.shared.require(stage.m._read(path) == ledger and prerequisites(root) == proof, 'Sources/evidence/self/ledger changed during snapshot')
        same = {row['job_id'] for row in snapshot['jobs'] if row['name'] == stage.QA_NAME}; current = ledger['jobs'].get(stage.QA_ROLE)
        if current is not None:
            stage.shared._entry(current, stage.QA_ROLE, gpu_seconds=0, sources=proof['sources'])
            stage.shared.require(current.get('binding') == binding and current.get('binding_sha256') == checksum and not same - {current['job_id']},
                'Preserve any existing/uncertain/rejected CPU QA without retry')
            for row in snapshot['jobs']:
                if row['job_id'] == current['job_id']: stage.shared.require(row['name'] == stage.QA_NAME and row['gpus'] == 0, 'CPU role identity/resources changed')
            print(current['job_id'], flush=True); return
        stage.shared.require(not same, 'Untracked CPU QA requires reconciliation'); when = guard()
        environment, removed = stage.m._submission_environment()
        current = {'status': 'submitting', 'binding': binding, 'binding_sha256': checksum, 'source_sha256': proof['sources'],
            'metadata_source_inventory': proof['metadata'], 'plan_sha256': proof['plan_sha256'], 'command': command,
            'requested_gpu_seconds': 0, 'requested_gpu_hours': 0, 'requested_at_utc': when.isoformat(),
            'scheduler_snapshot': snapshot, 'removed_scheduler_environment_keys': removed}
        ledger['jobs'][stage.QA_ROLE] = current; stage.m._write(path, ledger)
        try: result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            current.update(status='uncertain', error_type=type(error).__name__); stage.m._write(path, ledger); raise
        current.update(scheduler_returncode=result.returncode, scheduler_stdout=str(result.stdout)[-2000:], scheduler_stderr=str(result.stderr)[-2000:])
        match = re.fullmatch(r'([1-9][0-9]*)(?:;owl)?', result.stdout.strip()) if isinstance(result.stdout, str) else None
        if result.returncode or match is None or not isinstance(result.stderr, str) or result.stderr.strip():
            current['status'] = 'submission_failed' if result.returncode else 'uncertain'
            if match: current['possible_job_id'] = match.group(1)
            stage.m._write(path, ledger); raise stage.m.SubmissionError('Rejected/ambiguous CPU QA preserved; no retry')
        job = match.group(1); current.update(status='uncertain', job_id=job); stage.m._write(path, ledger)
        stage.shared.require(job not in stage.shared.scheduler_identities(before) and job not in {v['job_id'] for v in snapshot['jobs']}, 'Scheduler identity collision')
        current.update(status='submitted', submitted_at_utc=datetime.now(timezone.utc).isoformat()); stage.m._write(path, ledger)
        print(job, flush=True)

if __name__ == '__main__': main()
