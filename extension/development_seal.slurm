#!/bin/bash
# CPU-only native training-hook QA and prospective development split sealing.
#SBATCH --job-name=sft-extension-development-seal
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-development-seal-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-development-seal-%j.err

set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_output="$PWD/extension/runs/development-seal-20261021-${SLURM_JOB_ID:?Expected CPU allocation}"
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
from extension.development_submission import ROLE, initial_evidence, source_inventory, validate_plan

compute = require_cpu_allocation()
deadline_guard(duration_seconds=180000)
root = Path.cwd()
validate_plan(json.loads((root / 'extension/development_plan.json').read_text()))
initial_before = initial_evidence(root)
before = source_inventory(root)
ledger = json.loads((root / 'extension/job_ledger.json').read_text())
admission = ledger.get('jobs', {}).get(ROLE, {})
if admission.get('job_id') != os.environ['SLURM_JOB_ID'] or admission.get('status') != 'submitted' or admission.get('source_sha256') != before:
    raise SystemExit('Current job/source differs from admitted development-stage binding')
if admission.get('binding', {}).get('initial_evidence') != initial_before[0]:
    raise SystemExit('Initial evidence differs from admitted development-stage binding')
if any(ledger.get('jobs', {}).get(name) != entry for name, entry in initial_before[1].items()):
    raise SystemExit('An earlier extension job entry changed')
subprocess.run(['bash', '-n', str(root / 'extension/development_seal.slurm')], check=True)
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.mkdir(exist_ok=False)

loader = unittest.defaultTestLoader
patterns = ('test_extension_*.py', 'test_labeled_*.py', 'test_preserving_*.py', 'test_development_*.py')
suites = [loader.discover(str(root / 'tests'), pattern=pattern, top_level_dir=str(root / 'tests')) for pattern in patterns]
counts = [suite.countTestCases() for suite in suites]
if counts[:3] != [64, 44, 36] or counts[3] < 2:
    raise SystemExit('Expected all 144 existing tests and both new development-stage modules')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
if result.testsRun != sum(counts) or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
    raise SystemExit('Native development QA requires every test with zero failures, errors and skips')
after = source_inventory(root)
if after != before or initial_evidence(root) != initial_before:
    raise SystemExit('Sources or prior native receipts changed during native QA')
receipt = {
    'schema': 'extension-development-seal-native-qa-v1',
    'success': True,
    'tests_run': result.testsRun,
    'test_counts': dict(zip(patterns, counts)),
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
    'synthetic_tiny_cpu_models_only': True,
    'experiment_model_loaded': False,
    'versions': {'python': sys.version, **{
        name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'accelerate', 'pyarrow', 'huggingface_hub')
    }},
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
}
with (output / 'native_qa.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps({'native_qa_complete': True, 'tests': result.testsRun, 'output': str(output)}), flush=True)
PY

"$PWD/.venv/bin/python" -m extension.development_seal \
  --root "$PWD" --plan extension/development_plan.json \
  --output "$run_output" --qa-receipt "$qa_receipt"
printf 'Development seal complete: %s\n' "$run_output"
