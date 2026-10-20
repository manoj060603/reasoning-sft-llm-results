#!/bin/bash
# Read-only CPU feasibility inventory; does not admit any downstream work.
#SBATCH --job-name=sft-extension-feasibility
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-feasibility-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-feasibility-%j.err

set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The submission operator creates this before sbatch so Slurm can open its logs.
mkdir -p "$PWD/extension/logs"

run_output="$PWD/extension/runs/feasibility-20261020-${SLURM_JOB_ID:?Expected existing CPU allocation}"
qa_receipt="$run_output/native_qa.json"

"$PWD/.venv/bin/python" - "$run_output" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
import unittest

from extension.cpu_feasibility import deadline_guard, require_cpu_allocation

# These guards precede native tests and every inventory read.
compute = require_cpu_allocation()
deadline_guard(duration_seconds=5400)

root = Path.cwd()
subprocess.run(['bash', '-n', str(root / 'extension/cpu_feasibility.slurm')], check=True)
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.mkdir(exist_ok=False)
def inventory():
    # Bind the joint extension preflight, including the fixed planning inputs.
    # Mutable ledgers, run outputs and logs are deliberately outside this set.
    paths = sorted({
        *(str(path.relative_to(root)) for path in (root / 'extension').glob('*.py')),
        *(str(path.relative_to(root)) for path in (root / 'tests').glob('test_extension_*.py')),
        'extension/cpu_feasibility.slurm',
        'extension/campaign.json',
        'extension/research.json',
        'extension/development_source_metadata.json',
        'reasoning/data.py',
    })
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
    }

before = inventory()
suite = unittest.defaultTestLoader.discover(
    str(root / 'tests'), pattern='test_extension_*.py',
    top_level_dir=str(root / 'tests'),
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun <= 0:
    raise SystemExit('Native feasibility QA discovered no tests')
if not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
    raise SystemExit('Native feasibility QA requires zero failures, errors and skips')
after = inventory()
if after != before:
    raise SystemExit('Feasibility source inventory changed during native QA')

receipt = {
    'schema': 'extension-feasibility-native-qa-v1',
    'success': True,
    'wrapper_syntax_passed': True,
    'versions': {'python': sys.version, 'torch': importlib.metadata.version('torch')},
    'tests_run': result.testsRun,
    'failures': len(result.failures),
    'errors': len(result.errors),
    'skipped': len(result.skipped),
    'unexpected_successes': len(result.unexpectedSuccesses),
    'expected_failures': len(result.expectedFailures),
    'code_sha256': after,
    'discovery_pattern': 'test_extension_*.py',
    'source_scope': 'extension/*.py, tests/test_extension_*.py, wrapper, reasoning/data.py and three fixed extension planning JSON files; excludes runs/logs/ledger',
    'job_id': os.environ['SLURM_JOB_ID'],
    'step_id': os.environ.get('SLURM_STEP_ID', 'batch'),
    'node': compute['node'],
    'compute_proof': compute,
    'cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
    'gpu_disabled': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
}
with (output / 'native_qa.json').open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
    handle.write('\n')
print(f'Native QA complete: {output / "native_qa.json"}', flush=True)
PY

"$PWD/.venv/bin/python" -m extension.cpu_feasibility \
  --root deadline --output "$run_output" --qa-receipt "$qa_receipt"
printf 'Feasibility complete: %s\n' "$run_output"
