"""CPU tests; tensor arithmetic tests require the training environment's torch."""
import unittest
import json
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

from reasoning import stage2_convergence_20261019 as x
try:
    import torch
except ImportError:
    torch = None


class ConvergenceTests(unittest.TestCase):
    def test_fixed_source_panel_ignores_input_order(self):
        ids = [f'id{i:05}' for i in range(15000)]
        panel = {ids[i] for i in x.fixed_indices(ids)}
        reverse = list(reversed(ids))
        self.assertEqual(panel, {reverse[i] for i in x.fixed_indices(reverse)})
        self.assertEqual(len(panel), 200)
        with self.assertRaises(ValueError): x.fixed_indices(ids[:-1])

    def test_native_conversion_preserves_input_and_masks_eos(self):
        row = dict(system='Explicit system', instruction='Question', input='Nonempty query', output='Solution')
        class Template:
            efficient_eos = False
            def encode_multiturn(self, tok, messages, system, unused):
                assert messages[0]['content'] == 'Question\nNonempty query'
                assert system == 'Explicit system'
                return [([10, 11], [2, 3, 0])]
        class Processor:
            def preprocess_dataset(self, batch):
                return {'input_ids': [[10, 11, 2, 3, 0]], 'labels': [[-100, -100, 2, 3, 0]]}
        def converter(r):
            return {'_system': r['system'], '_prompt': [{'role': 'user', 'content': x.user_content(r)}],
                    '_response': [{'role': 'assistant', 'content': r['output']}]}
        ids, labels = x.native_batch(row, SimpleNamespace(eos_token_id=0), Template(), Processor(), converter)
        self.assertEqual(labels[-1], 0)
        def broken(r):
            d = converter(r); d['_prompt'][0]['content'] = r['instruction']; return d
        with self.assertRaisesRegex(ValueError, 'input'):
            x.native_batch(row, SimpleNamespace(eos_token_id=0), Template(), Processor(), broken)

    @unittest.skipIf(torch is None, 'Run in native training CPU environment for torch arithmetic')
    def test_shifted_masked_argmax_includes_eos_across_chunk_boundaries(self):
        hidden = torch.nn.functional.one_hot(torch.tensor([[1, 2, 3, 0, 2]]), num_classes=4).float()
        labels = torch.tensor([[-100, -100, 2, 3, 0]])
        head = torch.nn.Identity()
        for chunk in (1, 2, 128):
            self.assertEqual(x.shifted_chunk_accuracy(hidden, labels, head, chunk), (3, 3))
        labels[0, 3] = 1
        self.assertEqual(x.shifted_chunk_accuracy(hidden, labels, head, 2), (2, 3))
        hidden[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            x.shifted_chunk_accuracy(hidden, labels, head)

    def test_submit_persists_real_scheduler_id_before_return_and_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp, 'art'); art.mkdir()
            proof = art/'input.json'; proof.write_text(json.dumps({'fixture': True}))
            decision = art/'decision.json'
            fixture = {'additional_gpu_hours': 57, 'other_committed_gpu_hours': 5}
            with patch.object(x, 'ART', art), patch.object(x, 'INPUT_RECEIPT', proof), patch.object(x, 'OUTPUT_RECEIPT', decision), \
                 patch.object(x, 'authorization', return_value=fixture), patch.object(x, 'digest', return_value='0'*64), \
                 patch.object(x, 'validate_panel', return_value=([], {'fixture': True})), patch.object(x, 'require_deadline_fit'), \
                 patch.object(x.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='12345;owl\n')) as scheduler:
                x.submit(Path(tmp, 'TEST_ONLY_AUTHORIZATION.json'))
                ledger = x.read(art/'other_allocations.json')
                self.assertEqual(ledger['jobs'][0]['job_id'], '12345')
                self.assertEqual(ledger['jobs'][0]['gpu_hours'], 1)
                self.assertEqual(ledger['submission_journal'][0]['status'], 'confirmed')
                with self.assertRaisesRegex(ValueError, 'one convergence'):
                    x.submit(Path(tmp, 'TEST_ONLY_AUTHORIZATION.json'))
                self.assertEqual(scheduler.call_count, 1)

    def test_submission_timeout_is_unknown_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp, 'art'); art.mkdir()
            proof = art/'input.json'; proof.write_text(json.dumps({'fixture': True}))
            with patch.object(x, 'ART', art), patch.object(x, 'INPUT_RECEIPT', proof), patch.object(x, 'OUTPUT_RECEIPT', art/'none.json'), \
                 patch.object(x, 'authorization', return_value={'additional_gpu_hours': 57, 'other_committed_gpu_hours': 5}), \
                 patch.object(x, 'digest', return_value='0'*64), patch.object(x, 'validate_panel', return_value=([], {'fixture': True})), \
                 patch.object(x, 'require_deadline_fit'), patch.object(x.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 45)) as scheduler:
                with self.assertRaises(subprocess.TimeoutExpired): x.submit(Path('TEST_ONLY'))
                self.assertEqual(x.read(art/'other_allocations.json')['submission_journal'][0]['status'], 'unknown')
                with self.assertRaisesRegex(ValueError, 'reconciliation'): x.submit(Path('TEST_ONLY'))
                self.assertEqual(scheduler.call_count, 1)

    def test_aux_budget_and_deadline_fail_closed(self):
        with self.assertRaises(ValueError):
            x.check_other_budget({'additional_gpu_hours': 1, 'other_committed_gpu_hours': 5}, {}, {'jobs': []})
        with self.assertRaises(ValueError):
            x.require_deadline_fit(datetime(2026, 10, 20, 10, 0, tzinfo=ZoneInfo('America/New_York')))
        with self.assertRaises(ValueError):
            x.journal_clean({'submission_journal': [{'status': 'attempting'}]})

    def test_four_b200_coverage_charge_and_97_hour_boundary(self):
        central = {'schema': 'two-plan-ledger-v1', 'prior_committed_gpu_hours': 40,
                   'jobs': [dict(plan='assistance_coverage', phase='train', job_id='1', charged_gpu_hours=24),
                            dict(plan='stage2', phase='train', job_id='2', charged_gpu_hours=12)]}
        # 24+12 training +16 reserved evals +5 reserved other =57 extra/97total.
        auth = {'additional_gpu_hours': 57, 'other_committed_gpu_hours': 5}
        x.check_other_budget(auth, central, {'jobs': []})
        x.check_other_budget(auth, central, {'jobs': [dict(gpu_hours=1)]})
        with self.assertRaises(ValueError):
            x.check_other_budget({**auth, 'additional_gpu_hours': 56}, central, {'jobs': []})
        central['jobs'][0]['charged_gpu_hours'] = 12
        with self.assertRaises(ValueError): x.check_other_budget(auth, central, {'jobs': []})

    def test_afterok_releases_only_below_experimental_threshold(self):
        self.assertEqual(x.gate_exit_status(.949999), 0)
        self.assertEqual(x.gate_exit_status(.95), 2)
        self.assertEqual(x.gate_exit_status(1.), 2)
        with self.assertRaises(ValueError): x.gate_exit_status(float('nan'))

    def test_missing_authorization_does_not_create_auxiliary_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(x, 'ART', Path(tmp, 'new')):
                with self.assertRaises(FileNotFoundError): x.charge_start(Path(tmp, 'missing.json'))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__': unittest.main()
