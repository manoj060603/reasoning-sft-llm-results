"""Compute-only native QA receipt for the isolated post-hoc recovery experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import unittest

from deadline.recovery.common import digest, read_json, require, require_compute, write_new_json


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT / 'deadline/recovery/plan.json'


def sources():
    paths = list((PROJECT / 'deadline/recovery').glob('*.py'))
    paths += list((PROJECT / 'tests').glob('test*recovery*.py'))
    paths += [PLAN, Path(__file__).resolve(),
              PROJECT / 'deadline/operations/recovery-prepare.slurm',
              PROJECT / 'deadline/operations/recovery-development.slurm',
              PROJECT / 'deadline/operations/recovery-evaluate.slurm']
    require(any(path.name.startswith('test') and 'recovery' in path.name for path in paths),
            'recovery native tests are missing')
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(set(paths))}


def verify(receipt):
    require_compute()
    value = read_json(receipt)
    require(value.get('schema') == 'deadline-recovery-native-qa-v1', 'wrong QA receipt schema')
    require(value.get('success') is True and value.get('tests_run', 0) > 0,
            'native recovery tests did not pass')
    require(all(value.get(key) == 0 for key in ('failures', 'errors', 'skips')),
            'native recovery QA has failures, errors or skips')
    require(value.get('plan_sha256') == digest(PLAN), 'recovery plan changed after QA')
    require(value.get('code_sha256') == sources(), 'recovery source changed after QA')
    require(value.get('allocation', {}).get('cuda_visible_devices') == '',
            'native recovery QA was not CPU-only')
    print(json.dumps({'verified_qa_receipt': str(Path(receipt).resolve()),
                      'sha256': digest(receipt), 'tests_run': value['tests_run']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    require((args.output is None) != (args.verify is None), 'choose exactly one QA action')
    if args.verify is not None:
        verify(args.verify)
        return
    allocation = require_compute(cpu_only=True)
    require(not args.output.exists(), 'QA output exists; preserve it and use a unique path')
    before = sources()
    wrappers = ('recovery-prepare.slurm', 'recovery-development.slurm', 'recovery-evaluate.slurm')
    for name in wrappers:
        subprocess.run(['bash', '-n', str(PROJECT / 'deadline/operations' / name)], check=True)
    suite = unittest.defaultTestLoader.discover(str(PROJECT / 'tests'), pattern='test*recovery*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(before == sources(), 'recovery source changed during QA')
    success = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    value = {'schema': 'deadline-recovery-native-qa-v1',
             'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'hostname': socket.gethostname(), 'allocation': allocation,
             'plan_sha256': digest(PLAN), 'code_sha256': before,
             'tests_run': result.testsRun, 'failures': len(result.failures),
             'errors': len(result.errors), 'skips': len(result.skipped), 'success': success,
             'shell_syntax_checks': list(wrappers)}
    write_new_json(args.output, value)
    print(json.dumps(value), flush=True)
    if not success:
        raise SystemExit('native recovery tests failed; no GPU development admission')


if __name__ == '__main__':
    main()
