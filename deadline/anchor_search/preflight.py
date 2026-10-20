"""Native CPU-only checks for the isolated validation-guided follow-up."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import unittest

from deadline.recovery.common import digest, read_json, require, require_compute, write_new_json

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sources():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.slurm')) + [HERE / 'plan.json']
    tests = list((PROJECT / 'tests').glob('test_anchor_*.py'))
    require(len(tests) >= 2, 'Both independent anchor test suites are required')
    paths += tests
    paths += [PROJECT / name for name in (
        'deadline/recovery/common.py', 'deadline/recovery/merge.py',
        'deadline/recovery/screen.py', 'deadline/pipeline.py', 'deadline/gpu_budget.py',
        'reasoning/evaluate.py', 'reasoning/data.py', 'reasoning/features.py',
        'reasoning/benchmark.py', 'eval/utils/grader.py', 'eval/utils/parser.py',
        'eval/utils/math_normalization.py')]
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(set(paths))}


def verify(path):
    require_compute()
    value = read_json(path)
    require(value.get('schema') == 'deadline-anchor-native-qa-v1' and
            value.get('success') is True and value.get('tests_run', 0) > 0 and
            all(value.get(key) == 0 for key in ('failures', 'errors', 'skips')),
            'Successful complete native anchor QA is required')
    require(value.get('code_sha256') == sources(), 'Anchor code changed after native QA')
    require(value.get('allocation', {}).get('cuda_visible_devices') == '',
            'Native anchor checks must be CPU-only')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    require((args.output is None) != (args.verify is None), 'Choose one QA action')
    if args.verify is not None:
        value = verify(args.verify)
        print(json.dumps({'verified': str(args.verify), 'tests_run': value['tests_run']}), flush=True)
        return
    allocation = require_compute(cpu_only=True)
    require(not args.output.exists(), 'Preserve existing QA receipts')
    before = sources()
    wrappers = sorted(HERE.glob('*.slurm'))
    require(len(wrappers) >= 3, 'All three stage wrappers are required')
    for wrapper in wrappers:
        subprocess.run(['bash', '-n', str(wrapper)], check=True)
    suite = unittest.defaultTestLoader.discover(str(PROJECT / 'tests'), pattern='test_anchor_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(before == sources(), 'Anchor source changed during native QA')
    success = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    value = dict(schema='deadline-anchor-native-qa-v1',
                 created_at_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                 allocation=allocation, code_sha256=before, tests_run=result.testsRun,
                 failures=len(result.failures), errors=len(result.errors),
                 skips=len(result.skipped), success=success,
                 shell_syntax_checks=[str(p.relative_to(PROJECT)) for p in wrappers])
    write_new_json(args.output, value)
    print(json.dumps(value), flush=True)
    if not success:
        raise SystemExit('Native anchor checks failed; no preparation or GPU admission')


if __name__ == '__main__':
    main()
