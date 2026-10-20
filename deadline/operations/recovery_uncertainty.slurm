#!/bin/bash
# Final post-hoc analysis only; no model selection, generation or publication.
#SBATCH --job-name=sft-recovery-uncertainty
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-uncertainty-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/recovery-uncertainty-%j.err
set -euo pipefail
# Check the submitted source only after execution has entered its CPU allocation.
bash -n "$0"
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Fifteen minutes of CPU work plus one hour for final delivery.
"$PWD/.venv/bin/python" -c 'from deadline.pipeline import guard_start; guard_start(1.25)'
"$PWD/.venv/bin/python" - "$0" <<'PY'
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import sys

from deadline import compare_uncertainty as analysis
from deadline import publish_huggingface as primary
from deadline.audit_results import EXPECTED, validate_metric
from deadline.publish_optional_model import REPO_ID, validate_primary_receipt
from deadline.recovery.common import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


primary.compute_only()
root = Path.cwd()
primary_receipt_path = root / 'deadline/hf/publication/publication_complete.json'
optional_state = root / 'deadline/hf/optional-recovery-publication'
optional_receipt_path = optional_state / 'publication_complete.json'
evaluation_path = optional_state / 'package/evaluation_summary.json'
recovery = root / 'deadline/recovery-runs/base-anchor-75-25-20261020'
analysis_path = Path(analysis.__file__).resolve()
require(digest(analysis_path) ==
        'bbc9f515302d0cd2b19059e07140efaa778b1245c81515cd51021ef2a78b5b6f',
        'The unchanged native bootstrap implementation is required')
primary_receipt = read(primary_receipt_path)
validate_primary_receipt(primary_receipt)
optional = read(optional_receipt_path)
require(optional.get('schema') == 'deadline-optional-recovery-publication-complete-v1' and
        optional.get('complete') is True and optional.get('owner') == primary.OWNER and
        optional.get('repo_id') == REPO_ID and
        optional.get('url') == 'https://huggingface.co/' + REPO_ID and
        optional.get('commit') == '51814ba26ab0dc052447aee1612bebcffdba43f2' and
        optional.get('public_access_verified') is True and
        optional.get('evaluated_and_published_weight_bytes_identical') is True and
        optional.get('primary_four_artifacts_unchanged') is True,
        'The completed, anonymously verified fixed recovery publication is required')
require(optional.get('evaluation_summary_sha256') == digest(evaluation_path),
        'The native optional evaluation summary differs from publication')
require(optional.get('merge_receipt_sha256') == digest(recovery / 'complete.json'),
        'The recovery merge identity differs from publication')
evaluation = read(evaluation_path)
require(evaluation.get('schema') == 'deadline-optional-recovery-all-nine-evaluation-v1' and
        evaluation.get('all_nine_matched_raw') is True and
        evaluation.get('primary_comparison_unmodified') is True and
        set(evaluation.get('models', {})) == {'original', 'random', 'advanced', 'recovery'},
        'The sealed native all-four, all-nine raw evaluation is required')
for role, model in evaluation['models'].items():
    require(set(model.get('benchmarks', {})) == set(EXPECTED),
            'Missing native benchmark summary: ' + role)
    for benchmark, count in EXPECTED.items():
        row = model['benchmarks'][benchmark]
        require(row.get('status') == 'complete' and not row.get('errors') and
                row.get('evidence') == 'raw eight-sample predictions verified',
                'Incomplete native raw evidence: ' + role + '/' + benchmark)
        validate_metric(row.get('metrics', {}), count)

output = root / 'deadline/report/recovery-final-20261020T1730Z'
native_path = output / 'uncertainty-native-candidate-alias.json'
summary_path = output / 'uncertainty.json'
require(not native_path.exists() and not summary_path.exists(),
        'Preserve existing recovery uncertainty outputs; do not overwrite them')
inputs = {
    'primary_publication_complete': primary_receipt_path,
    'optional_publication_complete': optional_receipt_path,
    'optional_evaluation_summary': evaluation_path,
    'recovery_merge_complete': recovery / 'complete.json',
}
input_hashes = {name: digest(path) for name, path in inputs.items()}
# The old API names its candidate "advanced". Its only input change is this
# explicit role alias; the original primary advanced model is not overwritten.
paths = {'original': root / 'deadline/results/original',
         'random': root / 'deadline/results/random',
         'advanced': recovery / 'benchmarks'}
native = analysis.compare_uncertainty(paths)
require(not native['unavailable_benchmarks'] and not native['audit_errors'] and
        set(native['generation_diagnostics']) == set(EXPECTED) and
        native['seed'] == 20261019 and native['resamples'] == 10000 and
        native['multiple_comparison_adjustment'] is False,
        'The unchanged bootstrap did not produce complete matched all-nine evidence')
for pair, comparator in (('advanced_vs_random', 'random'),
                         ('advanced_vs_original', 'original')):
    require(set(native['comparisons'][pair]['benchmarks']) == set(EXPECTED),
            'Missing per-benchmark native uncertainty')
    for label, macro in native['comparisons'][pair]['macros'].items():
        require(macro.get('status') == 'complete', 'Partial recovery macro is forbidden')
        for metric, estimate in macro['metrics'].items():
            require(estimate['advanced'] == evaluation['models']['recovery'][label][metric] and
                    estimate['comparator'] == evaluation['models'][comparator][label][metric],
                    'Bootstrap point estimate differs from the published native summary')
require(input_hashes == {name: digest(path) for name, path in inputs.items()} and
        digest(analysis_path) ==
        'bbc9f515302d0cd2b19059e07140efaa778b1245c81515cd51021ef2a78b5b6f',
        'Analysis code or sealed inputs changed during the native computation')
output.mkdir(parents=True, exist_ok=True)
write_new(native_path, native)


def recovery_keys(value):
    """Relabel existing macro fields only; never change estimates or statistics."""
    if isinstance(value, dict):
        return {('recovery' if key == 'advanced' else key): recovery_keys(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [recovery_keys(item) for item in value]
    return value


summary = {
    'schema': 'deadline-posthoc-recovery-uncertainty-summary-v1',
    'complete': True,
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'candidate_role': 'recovery',
    'native_alias_map': {'advanced': 'recovery'},
    'posthoc_after_primary_benchmark_observations': True,
    'alpha_search': False,
    'alpha_base': .75,
    'alpha_advanced': .25,
    'analysis_purpose': 'Final uncertainty analysis of one fixed published candidate; no model tuning or primary-output modification.',
    'analysis_code_sha256': digest(analysis_path),
    'wrapper_source_sha256': digest(root / 'deadline/operations/recovery_uncertainty.slurm'),
    'submitted_wrapper_sha256': digest(Path(sys.argv[1]).resolve()),
    'analysis_dependencies_sha256': {
        name: digest(root / name) for name in
        ('deadline/audit_results.py', 'reasoning/benchmark.py', 'reasoning/data.py')},
    'compute': {key.lower(): os.environ.get(key) for key in
        ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME', 'SLURM_JOB_PARTITION',
         'SLURM_CPUS_PER_TASK', 'SLURM_JOB_GPUS', 'SLURM_GPUS', 'CUDA_VISIBLE_DEVICES')},
    'hostname': socket.gethostname(),
    'python': platform.python_version(),
    'seed': native['seed'],
    'resamples': native['resamples'],
    'confidence': native['confidence'],
    'interval': native['interval'],
    'resampling_unit': native['resampling_unit'],
    'metric_weighting': native['metric_weighting'],
    'conditioning': native['conditioning'],
    'multiple_comparison_adjustment': native['multiple_comparison_adjustment'],
    'historical_leaderboard_equivalence_verified': False,
    'native_artifact': {'path': str(native_path), 'sha256': digest(native_path),
                        'bytes': native_path.stat().st_size,
                        'candidate_field_name': 'advanced (alias of recovery only)'},
    'input_receipts': {name: {'path': str(path), 'sha256': input_hashes[name]}
                       for name, path in inputs.items()},
    'sources_under_native_aliases': copy.deepcopy(native['sources']),
    'comparisons': {
        pair.replace('advanced_vs_', 'recovery_vs_'): {
            'macros': recovery_keys(copy.deepcopy(value['macros']))}
        for pair, value in native['comparisons'].items()},
    'limitations': ['Exploratory recovery was designed after primary outcomes.',
        'Intervals condition on the fixed models and eight recorded outputs.',
        'Training-seed and additional generation-sampling variance are excluded.',
        'Multiple comparisons are unadjusted; no coefficient or candidate search was performed.'],
}
write_new(summary_path, summary)
print(json.dumps({'complete': True, 'candidate_role': 'recovery',
                  'native_artifact': summary['native_artifact'],
                  'summary_path': str(summary_path),
                  'summary_sha256': digest(summary_path)}, indent=2), flush=True)
PY
