#!/bin/bash
#SBATCH --job-name=sft-extension-smoke-preparation
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-smoke-preparation-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-smoke-preparation-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_output="$PWD/extension/runs/native-smoke-preparation-${SLURM_JOB_ID:?Expected CPU allocation}"
"$PWD/.venv/bin/python" - "$run_output" <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from extension.cpu_feasibility import deadline_guard, digest, read_json, require, require_cpu_allocation, write_new_json
from extension.smoke_submission import CPU_ROLE, initial_evidence, source_inventory
compute = require_cpu_allocation()
deadline_guard(180000)
root, output = Path.cwd(), Path(sys.argv[1])
before, parents = source_inventory(root), initial_evidence(root)
require(len(before) == 51, 'Bind the exact 51 reviewed sources')
admission = read_json(root / 'extension/job_ledger.json')['jobs'][CPU_ROLE]
require(admission.get('job_id') == os.environ['SLURM_JOB_ID'] and admission.get('status') == 'submitted' and
    admission.get('source_sha256') == before and admission.get('requested_gpu_seconds') == 0,
    'Current CPU job differs from the reviewed source admission')
for name, entry in parents[1].items():
    require(read_json(root / 'extension/job_ledger.json')['jobs'].get(name) == entry, 'Earlier ledger entry changed')
for name in ('extension/smoke_preparation.slurm', 'extension/native_model_admission.slurm'):
    subprocess.run(['bash', '-n', str(root / name)], check=True)
output.parent.mkdir(parents=True, exist_ok=True)
output.mkdir(exist_ok=False)
started = datetime.now(timezone.utc).isoformat()
patterns = ('test_extension_*.py', 'test_labeled_*.py', 'test_preserving_*.py', 'test_development_*.py',
    'test_preservation_development_seal_v2.py', 'test_preservation_retry_submission.py',
    'test_preservation_data.py', 'test_native_model_admission.py', 'test_smoke_preparation.py', 'test_smoke_submission.py')
loader = unittest.defaultTestLoader
suites = [loader.discover(str(root / 'tests'), pattern=pattern, top_level_dir=str(root / 'tests')) for pattern in patterns]
counts = [suite.countTestCases() for suite in suites]
require(counts[:6] == [64, 44, 36, 48, 17, 21] and all(count > 0 for count in counts[6:]),
    'Run all 230 old checks and all four exact new unit modules')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
require(result.testsRun == sum(counts) and result.wasSuccessful() and not result.skipped and
    not result.expectedFailures and not result.unexpectedSuccesses, 'Every native check must pass without skips')
require(source_inventory(root) == before and initial_evidence(root) == parents, 'Sources or prior evidence changed during QA')
receipt = {'schema': 'extension-native-smoke-preparation-unit-qa-v1', 'success': True,
    'job_id': os.environ['SLURM_JOB_ID'], 'node': compute['node'], 'compute_proof': compute,
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'started_at_utc': started,
    'tests_run': result.testsRun, 'test_counts': dict(zip(patterns, counts)), 'failures': len(result.failures),
    'errors': len(result.errors), 'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
    'unexpected_successes': len(result.unexpectedSuccesses), 'gpu_disabled': True,
    'wrapper_syntax_passed': True, 'source_inventory': before,
    'code_sha256': {name: meta['sha256'] for name, meta in before.items()},
    'plan_sha256': digest(root / 'extension/smoke_preparation_plan.json'),
    'native_plan_sha256': digest(root / 'extension/native_model_admission_plan.json'),
    'initial_evidence': parents[0], 'admission_binding_sha256': admission['binding_sha256'],
    'synthetic_fixture_models_only': True, 'experiment_model_loaded': False,
    'versions': {'python': sys.version, **{name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'accelerate', 'safetensors', 'pyarrow', 'huggingface_hub')}}}
write_new_json(output / 'unit_qa.json', receipt)
print(json.dumps({'unit_qa_complete': True, 'tests_run': result.testsRun}), flush=True)
PY
"$PWD/.venv/bin/python" -m extension.smoke_preparation --root "$PWD" \
  --plan extension/smoke_preparation_plan.json --output "$run_output" --qa-receipt "$run_output/unit_qa.json"
"$PWD/.venv/bin/python" - "$run_output" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
from extension.cpu_feasibility import digest, read_json, require, require_cpu_allocation, write_new_json
from extension.native_model_admission import validate_plan, validate_prepared
from extension.smoke_submission import CPU_ROLE, initial_evidence, source_inventory
root, output = Path.cwd(), Path(sys.argv[1])
compute = require_cpu_allocation()
unit = read_json(output / 'unit_qa.json')
require(unit['job_id'] == compute['job_id'] and unit['compute_proof']['node'] == compute['node'] and
    unit['source_inventory'] == source_inventory(root) and unit['initial_evidence'] == initial_evidence(root)[0],
    'CPU closeout source/input proof differs')
plan_path = root / 'extension/native_model_admission_plan.json'
plan = validate_plan(read_json(plan_path))
prepared_path = output / 'prepared.json'
prepared = validate_prepared(read_json(prepared_path), plan, digest(plan_path))
require(prepared['source_audit']['preparation']['sha256'] == digest(output / 'preparation.json'), 'Private preparation binding differs')
qa = {**unit, 'schema': 'extension-native-model-admission-native-qa-v1',
    'cpu_preparation_plan_sha256': unit['plan_sha256'], 'plan_sha256': digest(plan_path),
    'prepared_sha256': digest(prepared_path), 'unit_qa_sha256': digest(output / 'unit_qa.json'),
    'compute_end': compute, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
write_new_json(output / 'native_qa.json', qa)
pin = lambda path: {'path': str(path.relative_to(root)), 'sha256': digest(path)}
files = {file.name: digest(file) for file in sorted(output.iterdir()) if file.is_file()}
complete = {'schema': 'extension-native-smoke-preparation-complete-v1', 'complete': True, 'cpu_only': True,
    'job_id': compute['job_id'], 'gpu_seconds': 0, 'input_snapshot_unchanged': True,
    'plan_sha256': unit['plan_sha256'], 'native_plan_sha256': digest(plan_path),
    'code_sha256': unit['code_sha256'], 'source_inventory': unit['source_inventory'],
    'prepared': pin(prepared_path), 'native_qa': pin(output / 'native_qa.json'),
    'unit_qa': pin(output / 'unit_qa.json'), 'preparation': pin(output / 'preparation.json'),
    'compute_start': unit['compute_proof'], 'compute_end': compute, 'output_sha256': files,
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'automatic_downstream': False,
    'model_inference': False, 'training_pilot': False, 'raw_witnesses_private': True}
write_new_json(output / 'complete.json', complete)
print(json.dumps({'complete': True, 'job_id': compute['job_id'], 'tests_run': unit['tests_run'],
    'private_prepared_sha256': digest(prepared_path), 'gpu_seconds': 0}), flush=True)
PY
