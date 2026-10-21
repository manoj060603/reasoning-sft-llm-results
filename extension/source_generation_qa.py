"""Native48 CPU-only fixtures for new collector and finite GPU admission.

The unchanged parent478 is verified by receipts and source hashes, not replayed.
No private cache, tokenizer, model or vLLM is opened by this QA stage.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import unittest
from extension.cpu_feasibility import require_cpu_allocation, deadline_guard, write_new_json, value_digest, digest
from extension import source_generation_submission as metadata

def main():
    root = Path.cwd().resolve(); proposed = Path(sys.argv[1])
    compute = require_cpu_allocation(); deadline_guard(174000)
    metadata.shared.require(root == Path(__file__).resolve().parents[1] and not proposed.is_symlink() and
        compute['allocated_cpus'] == 2, 'Exact current two-CPU native QA required')
    output = proposed.resolve()
    metadata.shared.require(output.parent == root / 'extension/runs' and output.name == 'source-generation-qa-' + compute['job_id'] and
        not output.exists(), 'Exclusive current CPU run required')
    proof = metadata.initial_evidence(root); ledger = metadata.m._read(root / 'extension/job_ledger.json')
    metadata.shared._ledger(ledger); current = ledger['jobs'].get(metadata.QA_ROLE, {})
    metadata.shared._entry(current, metadata.QA_ROLE, gpu_seconds=0, sources=proof['sources'], job=compute['job_id'])
    before = {key: value for key, value in ledger['jobs'].items() if key != metadata.QA_ROLE}
    metadata.shared.require(before == proof['initial'] and len(before) == 16 and
        current['binding'].get('metadata_source_inventory') == proof['metadata'] and
        current['binding'].get('prior_entries_sha256') == metadata.PARENT_ENTRIES_SHA256 and
        current['binding'].get('plan_sha256') == proof['plan_sha256'] and
        current['binding'].get('wall_time_seconds') == 1200 and current['binding'].get('expected_native_tests') == 48 and
        current['binding'].get('automatic_downstream') is False, 'Exact finite current CPU admission and all16 priors required')
    metadata.validate_wrapper(root)
    for name in ('extension/source_generation.slurm', 'extension/source_generation_qa.slurm'):
        subprocess.run(['bash', '-n', str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / 'started.json', {'schema': metadata.generation.QA_SCHEMA, 'job_id': compute['job_id'],
        'source_inventory': proof['sources'], 'metadata_source_inventory': proof['metadata'], 'expected_tests': 48,
        'parent_tests_previously_passed': 478, 'parent_tests_replayed': False})
    patterns = ['test_source_generation.py', 'test_source_generation_submission.py']
    suites = [unittest.defaultTestLoader.discover(str(root / 'tests'), pattern=name, top_level_dir=str(root / 'tests')) for name in patterns]
    counts = [suite.countTestCases() for suite in suites]
    metadata.shared.require(counts == [32, 16], 'Exact32 collector plus16 metadata checks only')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {'tests_run': result.testsRun, 'test_counts': dict(zip(patterns, counts)), 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped), 'expected_failures': len(result.expectedFailures),
        'unexpected_successes': len(result.unexpectedSuccesses)}
    if result.testsRun != 48 or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
        write_new_json(output / 'failure.json', {'schema': metadata.generation.QA_SCHEMA, 'success': False, 'job_id': compute['job_id'],
            **metrics, 'source_inventory': proof['sources'], 'metadata_source_inventory': proof['metadata'], 'compute_proof': compute,
            'automatic_retry': False, 'automatic_downstream': False, 'parent_tests_replayed': False})
        raise RuntimeError('New native collector/metadata QA failed; preserve without automatic retry')
    metadata.shared.require(metadata.initial_evidence(root) == proof and metadata.m._read(root / 'extension/job_ledger.json') == ledger,
        'All source103/protocol/parent/current admission bytes must remain unchanged during QA')
    ending = require_cpu_allocation()
    metadata.shared.require(ending['job_id'] == compute['job_id'] and ending['node'] == compute['node'], 'CPU allocation changed')
    receipt = {'schema': metadata.generation.QA_SCHEMA, 'success': True, 'job_id': compute['job_id'], 'node': compute['node'],
        'compute_proof': compute, 'compute_end': ending, 'started_at_utc': started, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        **metrics, 'source_inventory': proof['sources'], 'metadata_source_inventory': proof['metadata'],
        'code_sha256': {name: item['sha256'] for name, item in proof['sources'].items()}, 'plan_sha256': proof['plan_sha256'],
        'cache_sha256': metadata.generation.CACHE['sha256'], 'parent_complete_sha256': metadata.generation.PARENT_PINS['complete.json'],
        'parent_tests_previously_passed': 478, 'parent_tests_replayed': False, 'collector_tests': 32, 'metadata_tests': 16,
        'gpu_disabled': True, 'wrapper_syntax_passed': True, 'actual_vllm_interface_established': False,
        'experiment_model_loaded': False, 'private_cache_opened': False, 'automatic_downstream': False,
        'admission_binding_sha256': current['binding_sha256'], 'prior_entries_sha256': value_digest(before),
        'started_sha256': digest(output / 'started.json')}
    metadata.validate_native_qa(receipt, proof, {metadata.QA_ROLE: current})
    write_new_json(output / 'native_qa.json', receipt)
    write_new_json(output / 'complete.json', {'schema': 'extension-source-generation-qa-complete-v1', 'complete': True,
        'job_id': compute['job_id'], 'tests_run': 48, 'cpu_only': True, 'gpu_seconds': 0,
        'native_qa_sha256': digest(output / 'native_qa.json'), 'actual_vllm_interface_established': False,
        'automatic_downstream': False})
    print(json.dumps({'native_collector_qa_complete': True, 'tests_run': 48, 'gpu_seconds': 0, 'parent478_replayed': False}), flush=True)

if __name__ == '__main__': main()
