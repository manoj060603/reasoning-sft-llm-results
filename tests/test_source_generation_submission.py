"""Native-CPU-only admission oracles; no private cache/model/vLLM access."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from extension import source_generation_submission as module
from extension.cpu_feasibility import value_digest

def entry(role='fixture', job='100', seconds=0, sources=None):
    binding = {'role': role, 'automatic_downstream': False, 'wall_time_seconds': 1200,
        'prior_entries_sha256': module.PARENT_ENTRIES_SHA256}
    if sources is not None: binding['source_sha256'] = sources
    return {'status': 'submitted', 'job_id': job, 'requested_gpu_seconds': seconds,
        'requested_gpu_hours': seconds / 3600, 'binding': binding, 'binding_sha256': value_digest(binding),
        'source_sha256': sources, 'command': ['sbatch', 'fixture.slurm']}

def wrapper(name=None):
    name = module.NAME if name is None else name
    return '\n'.join(['#!/bin/bash', '#SBATCH --job-name=' + name, '#SBATCH --account=tml_2026',
        '#SBATCH --partition=b200_normal_q', '#SBATCH --qos=owl_b200_normal_short', '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1', '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=8', '#SBATCH --mem=128G',
        '#SBATCH --time=02:00:00', '#SBATCH --no-requeue', '#SBATCH --output=ROOT/extension/logs/sample-%j.out',
        '#SBATCH --error=ROOT/extension/logs/sample-%j.err', 'set -euo pipefail']) + '\n'

class Flow:
    def __init__(self, directory, stdout='999;owl\n', stderr='', returncode=0, queue='', transport=None):
        self.root = Path(directory); (self.root / 'extension/operations').mkdir(parents=True)
        self.path = self.root / 'extension/job_ledger.json'
        self.prior = {'fixture': entry()}
        self.ledger = {'schema': module.m.SCHEMA, 'gpu_hour_cap': None,
            'historical_ledger_unchanged': 'deadline/job_ledger.json', 'jobs': deepcopy(self.prior)}
        self.path.write_text(json.dumps(self.ledger))
        self.proof = {'initial': deepcopy(self.prior), 'prior': deepcopy(self.prior), 'sources': {'core.py': {'sha256': 'a' * 64, 'bytes': 1}},
            'metadata': {'metadata.py': {'sha256': 'b' * 64, 'bytes': 1}}, 'plan_sha256': 'c' * 64,
            'qa_sha256': 'd' * 64, 'qa_run': 'extension/runs/source-generation-qa-100', 'assets': {}}
        self.stdout, self.stderr, self.returncode, self.queue, self.transport = stdout, stderr, returncode, queue, transport
        self.calls, self.at_submit = [], None

    def runner(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if command[0] == 'squeue': return SimpleNamespace(returncode=0, stdout=self.queue, stderr='')
        if command[0] != 'sbatch': raise AssertionError('Unexpected fixture command')
        self.at_submit = json.loads(self.path.read_text())
        if self.transport: raise self.transport
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)

    def run(self, *, changed=False):
        with ExitStack() as stack:
            if changed:
                altered = deepcopy(self.proof); altered['plan_sha256'] = 'e' * 64
                stack.enter_context(patch.object(module, 'load_evidence', side_effect=[self.proof, altered]))
            else: stack.enter_context(patch.object(module, 'load_evidence', return_value=self.proof))
            stack.enter_context(patch.object(module, 'known_queue'))
            return module.submit(self.root, self.proof['qa_run'], self.proof['qa_sha256'], runner=self.runner)

    def submissions(self): return [call for call in self.calls if call[0][0] == 'sbatch']
    def current(self): return json.loads(self.path.read_text())['jobs'].get(module.ROLE)

def qa_fixture():
    sources = {'core.py': {'sha256': 'a' * 64, 'bytes': 1}}
    meta = {'metadata.py': {'sha256': 'b' * 64, 'bytes': 1}}
    current = entry(module.QA_ROLE, sources=sources)
    current['binding']['metadata_source_inventory'] = meta
    current['binding_sha256'] = value_digest(current['binding'])
    proof = {'sources': sources, 'metadata': meta, 'plan_sha256': 'c' * 64}
    qa = {'tests_run': 48, 'test_counts': {'test_source_generation.py': 32, 'test_source_generation_submission.py': 16},
        'parent_tests_previously_passed': 478, 'parent_tests_replayed': False, 'source_inventory': sources,
        'code_sha256': {n: v['sha256'] for n, v in sources.items()}, 'metadata_source_inventory': meta,
        'plan_sha256': 'c' * 64, 'cache_sha256': module.generation.CACHE['sha256'],
        'parent_complete_sha256': module.generation.PARENT_PINS['complete.json'], 'actual_vllm_interface_established': False,
        'admission_binding_sha256': current['binding_sha256']}
    return qa, proof, {module.QA_ROLE: current}

class SourceGenerationAdmissionTests(unittest.TestCase):
    def test_deadline_has_exact_full_job_and_reserve_boundary(self):
        boundary = module.m.DEADLINE - timedelta(seconds=180000)
        self.assertEqual(module.guard(boundary), boundary)
        for bad in (boundary + timedelta(microseconds=1), boundary.replace(tzinfo=None), 'tomorrow'):
            with self.assertRaises(module.m.SubmissionError): module.guard(bad)

    def test_wrapper_allows_only_exact_finite_b200_resources_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p = root / module.SCRIPT; p.parent.mkdir()
            text = wrapper().replace('ROOT', str(root)); p.write_text(text)
            self.assertEqual(module.validate_wrapper(root)['gres'], 'gpu:1')
            for changed in (text.replace('--gres=gpu:1', '--gres=gpu:2'), text.replace('02:00:00', '03:00:00'),
                    text.replace('--mem=128G', '--mem=256G'), text.replace('--no-requeue', '--requeue'),
                    text + '#SBATCH --gres=gpu:1\n', text.replace('set -euo pipefail', '#SBATCH --exclusive\nset -euo pipefail'),
                    text.replace(str(root / 'extension/logs'), '/tmp')):
                p.write_text(changed)
                with self.assertRaises(module.m.SubmissionError): module.validate_wrapper(root)

    def test_actual_parent_metadata_and_all_source_bytes_are_bound(self):
        root = Path(module.__file__).resolve().parents[1]
        proof = module.initial_evidence(root)
        self.assertEqual((len(proof['initial']), len(proof['sources']), len(proof['metadata'])), (16, 95, 8))
        self.assertEqual(value_digest(proof['initial']), module.PARENT_ENTRIES_SHA256)
        self.assertNotIn(module.generation.CACHE['path'], proof['assets'])
        self.assertTrue(all(not name.endswith('.safetensors') for name in proof['assets']))
        self.assertEqual(module.validate_wrapper(root)['job-name'], module.NAME)

    def test_history_refuses_rewritten_unknown_or_uncertain_admissions(self):
        initial = {'fixture': entry()}; current = entry(module.QA_ROLE, job='101')
        ledger = {'schema': module.m.SCHEMA, 'gpu_hour_cap': None, 'historical_ledger_unchanged': 'deadline/job_ledger.json',
            'jobs': {**deepcopy(initial), module.QA_ROLE: current}}
        self.assertEqual(module.validate_history(ledger, initial, '101')[module.QA_ROLE], current)
        variants = []
        x = deepcopy(ledger); x['jobs']['fixture']['job_id'] = '200'; variants.append(x)
        x = deepcopy(ledger); x['jobs']['unknown'] = entry(job='201'); variants.append(x)
        x = deepcopy(ledger); x['jobs'][module.QA_ROLE]['status'] = 'uncertain'; variants.append(x)
        x = deepcopy(ledger); x['jobs'][module.QA_ROLE]['requested_gpu_seconds'] = 1; variants.append(x)
        for value in variants:
            with self.assertRaises(module.m.SubmissionError): module.validate_history(value, initial, '101')

    def test_known_queue_checks_identity_resources_arrays_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p = root / 'fixture.slurm'; p.write_text('#SBATCH --job-name=fixture-name\n')
            prior = {'fixture': entry(seconds=7200)}
            row = {'job_id': '100', 'name': 'fixture-name', 'state': 'PENDING', 'gpus': 1}
            self.assertEqual(module.known_queue(root, {'jobs': [row]}, prior), {'100': 'fixture-name'})
            for change in ({'name': 'renamed'}, {'gpus': 2}, {'gpus': 0}, {'job_id': '100_2'}, {'job_id': '200'}):
                with self.assertRaises(module.m.SubmissionError): module.known_queue(root, {'jobs': [{**row, **change}]}, prior)

    def test_success_journals_before_submit_and_reuses_exact_existing_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary); self.assertEqual(f.run(), '999')
            self.assertEqual(f.at_submit['jobs'][module.ROLE]['status'], 'submitting')
            c = f.current(); self.assertEqual(c['status'], 'submitted'); self.assertEqual(c['requested_gpu_seconds'], 7200)
            receipt = f.root / 'extension/operations/source-generation-original-job-999.json'
            self.assertEqual(json.loads(receipt.read_text())['binding_sha256'], c['binding_sha256'])
            self.assertEqual(f.submissions()[0][0][-1], f.proof['qa_run'] + '/native_qa.json')
            self.assertEqual(f.run(), '999'); self.assertEqual(len(f.submissions()), 1)

    def test_two_pending_or_unrelated_gpu_jobs_block_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary, queue='200|PENDING|other|gres/gpu:1\n201|RUNNING|another|gpu:A100:1\n')
            before = f.path.read_bytes()
            with self.assertRaisesRegex(module.m.SubmissionError, 'At most two'): f.run()
            self.assertEqual(f.path.read_bytes(), before); self.assertEqual(f.submissions(), [])

    def test_changed_source_after_snapshot_blocks_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary); before = f.path.read_bytes()
            with self.assertRaises(module.m.SubmissionError): f.run(changed=True)
            self.assertEqual(f.path.read_bytes(), before); self.assertEqual(f.submissions(), [])

    def test_rejected_ambiguous_or_warning_output_is_preserved_without_retry(self):
        for rc, stdout, stderr, state in ((1, '', 'denied', 'submission_failed'), (0, '', '', 'uncertain'),
                (0, '999;owl\n', 'warning', 'uncertain'), (0, '999;other\n', '', 'uncertain')):
            with tempfile.TemporaryDirectory() as temporary:
                f = Flow(temporary, stdout, stderr, rc)
                with self.assertRaises(module.m.SubmissionError): f.run()
                self.assertEqual(f.current()['status'], state)
                with self.assertRaises(module.m.SubmissionError): f.run()
                self.assertEqual(len(f.submissions()), 1)

    def test_transport_exception_is_uncertain_and_cannot_be_resubmitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary, transport=subprocess.TimeoutExpired('sbatch', 60))
            with self.assertRaises(subprocess.TimeoutExpired): f.run()
            self.assertEqual(f.current()['status'], 'uncertain')
            with self.assertRaises(module.m.SubmissionError): f.run()
            self.assertEqual(len(f.submissions()), 1)

    def test_returned_existing_scheduler_identity_stays_uncertain(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary, stdout='100;owl\n')
            with self.assertRaises(module.m.SubmissionError): f.run()
            self.assertEqual((f.current()['status'], f.current()['job_id']), ('uncertain', '100'))
            self.assertFalse((f.root / 'extension/operations/source-generation-original-job-100.json').exists())

    def test_handoff_write_failure_preserves_accepted_id_without_second_submit(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary); original = module.m._write
            def failing(path, value):
                if Path(path).name.startswith('source-generation-original-job-'): raise OSError('fixture disk failure')
                return original(path, value)
            with patch.object(module.m, '_write', side_effect=failing), self.assertRaises(OSError): f.run()
            self.assertEqual((f.current()['status'], f.current()['job_id']), ('uncertain', '999'))
            with self.assertRaises(module.m.SubmissionError): f.run()
            self.assertEqual(len(f.submissions()), 1)

    def test_existing_changed_binding_or_missing_handoff_never_resubmits(self):
        for kind in ('binding', 'missing'):
            with tempfile.TemporaryDirectory() as temporary:
                f = Flow(temporary); f.run()
                if kind == 'missing': (f.root / 'extension/operations/source-generation-original-job-999.json').unlink()
                else:
                    ledger = json.loads(f.path.read_text()); ledger['jobs'][module.ROLE]['binding']['model_role'] = 'trained'
                    ledger['jobs'][module.ROLE]['binding_sha256'] = value_digest(ledger['jobs'][module.ROLE]['binding']); f.path.write_text(json.dumps(ledger))
                with self.assertRaises(module.m.SubmissionError): f.run()
                self.assertEqual(len(f.submissions()), 1)

    def test_native_qa_requires_exact_new48_and_parent478_not_replayed(self):
        qa, proof, prior = qa_fixture(); self.assertIs(module.validate_native_qa(qa, proof, prior), qa)
        for change in ({'tests_run': 47}, {'parent_tests_replayed': True}, {'parent_tests_previously_passed': 0},
                {'actual_vllm_interface_established': True}, {'cache_sha256': '0' * 64}, {'metadata_source_inventory': {}},
                {'test_counts': {'test_source_generation.py': 32, 'test_source_generation_submission.py': 0}}):
            with self.assertRaises(module.m.SubmissionError): module.validate_native_qa({**qa, **change}, proof, prior)

    def test_native_qa_binds_exact_source_parent_admission_and_code_maps(self):
        qa, proof, prior = qa_fixture()
        for change in ({'source_inventory': {}}, {'code_sha256': {}}, {'plan_sha256': '0' * 64},
                {'parent_complete_sha256': '0' * 64}, {'admission_binding_sha256': '0' * 64}):
            with self.assertRaises(module.m.SubmissionError): module.validate_native_qa({**qa, **change}, proof, prior)

    def test_existing_role_still_counts_unrelated_live_gpu_allocations(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = Flow(temporary); f.run()
            f.queue = '999|RUNNING|'+module.NAME+'|gpu:1\n200|PENDING|other|gpu:1\n201|RUNNING|third|gpu:1\n'
            with self.assertRaisesRegex(module.m.SubmissionError, 'At most two'): f.run()
            self.assertEqual(len(f.submissions()), 1)

if __name__ == '__main__': unittest.main()
