#!/bin/bash
# Pinned labeled training-data capacity audit; no model or downstream job.
#SBATCH --job-name=sft-extension-labeled-feasibility
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-labeled-feasibility-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-labeled-feasibility-%j.err

set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

run_output="$PWD/extension/runs/labeled-feasibility-20261020-${SLURM_JOB_ID:?Expected CPU allocation}"
qa_receipt="$run_output/native_qa.json"

"$PWD/.venv/bin/python" - "$run_output" <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import deadline_guard, require_cpu_allocation
from extension.labeled_submission import ROLE, initial_evidence, source_inventory, validate_plan

compute = require_cpu_allocation()
deadline_guard(duration_seconds=180000)
root = Path.cwd()
validate_plan(json.loads((root / 'extension/labeled_plan.json').read_text()))
initial_before = initial_evidence(root)
before = source_inventory(root)
ledger = json.loads((root / 'extension/job_ledger.json').read_text())
admission = ledger.get('jobs', {}).get(ROLE, {})
if admission.get('job_id') != os.environ['SLURM_JOB_ID'] or admission.get('status') != 'submitted' or admission.get('source_sha256') != before:
    raise SystemExit('Current job/source differs from the admitted labeled-stage binding')
if admission.get('binding', {}).get('initial_evidence') != initial_before[0]:
    raise SystemExit('Initial evidence differs from the admitted labeled-stage binding')
subprocess.run(['bash', '-n', str(root / 'extension/labeled_feasibility.slurm')], check=True)
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.mkdir(exist_ok=False)

loader = unittest.defaultTestLoader
old = loader.discover(str(root / 'tests'), pattern='test_extension_*.py', top_level_dir=str(root / 'tests'))
new = loader.discover(str(root / 'tests'), pattern='test_labeled_*.py', top_level_dir=str(root / 'tests'))
old_count, new_count = old.countTestCases(), new.countTestCases()
if old_count != 64 or new_count < 24:
    raise SystemExit('Expected all 64 regression tests and both new labeled-stage test modules')
suite = unittest.TestSuite([old, new])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun != old_count + new_count or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
    raise SystemExit('Native labeled-stage QA requires all tests with zero failures, errors or skips')
after = source_inventory(root)
if after != before or initial_evidence(root) != initial_before:
    raise SystemExit('Source or historical receipt changed during native QA')
receipt = {
    'schema': 'extension-labeled-feasibility-native-qa-v1',
    'success': True,
    'tests_run': result.testsRun,
    'regression_tests': old_count,
    'new_labeled_tests': new_count,
    'failures': len(result.failures),
    'errors': len(result.errors),
    'skipped': len(result.skipped),
    'expected_failures': len(result.expectedFailures),
    'unexpected_successes': len(result.unexpectedSuccesses),
    'wrapper_syntax_passed': True,
    'source_inventory': after,
    'code_sha256': {name: item['sha256'] for name, item in after.items()},
    'initial_evidence': initial_before[0],
    'admission_binding_sha256': admission['binding_sha256'],
    'compute_proof': compute,
    'job_id': os.environ['SLURM_JOB_ID'],
    'step_id': os.environ.get('SLURM_STEP_ID', 'batch'),
    'node': compute['node'],
    'gpu_disabled': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
    'versions': {'python': sys.version, **{
        name: importlib.metadata.version(name)
        for name in ('torch', 'pyarrow', 'huggingface_hub')
    }},
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
}
with (output / 'native_qa.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps({'native_qa_complete': True, 'tests': result.testsRun, 'output': str(output)}), flush=True)
PY

"$PWD/.venv/bin/python" -m extension.labeled_feasibility \
  --root "$PWD" --plan extension/labeled_plan.json \
  --output "$run_output" --qa-receipt "$qa_receipt"
printf 'Labeled feasibility complete: %s\n' "$run_output"
