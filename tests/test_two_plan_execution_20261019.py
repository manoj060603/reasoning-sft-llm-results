"""CPU checks for incremental-budget accounting and recovery boundaries."""
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import unittest
from unittest.mock import patch, Mock

from reasoning import two_plan_execution_20261019 as x


class ExecutionGuardTests(unittest.TestCase):
    def test_budget_preserves_40_and_both_eval_reservations(self):
        ledger = {'jobs': []}
        self.assertEqual(x.projected_total(ledger, 'assistance_coverage', 'train', '1'), 80)
        ledger['jobs'] = [dict(plan='assistance_coverage', phase='train', job_id='1')]
        self.assertEqual(x.projected_total(ledger, 'stage2', 'train', '2'), 92)
        ledger['jobs'].append(dict(plan='stage2', phase='train', job_id='2'))
        self.assertEqual(x.projected_total(ledger, 'assistance_coverage', 'eval', '3'), 92)
        self.assertEqual(x.projected_total(ledger, 'assistance_coverage', 'train', '1'), 92)
        self.assertEqual(x.projected_total(ledger, 'assistance_coverage', 'train', '4'), 116)

    def test_missing_authorization_cannot_create_ledger_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth = Path(tmp, 'missing.json')
            with patch.dict(os.environ, SLURM_JOB_ID='123', SLURM_GPUS_ON_NODE='2'):
                with self.assertRaises(FileNotFoundError):
                    x.register('stage2', 'train', {'output_dir': str(Path(tmp, 'run'))}, {}, auth)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_authorization_rejects_implicit_reset_bool_nan_overlap(self):
        valid = dict(additional_gpu_hours=57, prior_committed_gpu_hours=40,
                     allow_overlap=True, plans=list(x.PLANS), other_committed_gpu_hours=5,
                     authorization_evidence='TEST FIXTURE ONLY; not a real user authorization')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'fixture.json')
            p.write_text(json.dumps(valid)); self.assertEqual(x.authorization(p)['additional_gpu_hours'], 57)
            for key, value in [('additional_gpu_hours', True), ('additional_gpu_hours', float('nan')),
                               ('prior_committed_gpu_hours', 0), ('allow_overlap', False),
                               ('other_committed_gpu_hours', -1), ('authorization_evidence', '')]:
                p.write_text(json.dumps({**valid, key: value}))
                with self.assertRaises(ValueError): x.authorization(p)

    def test_deadline_reserves_training_plus_evaluation(self):
        zone = ZoneInfo('America/New_York')
        x.deadline_fit('train', datetime(2026, 10, 20, 9, 0, tzinfo=zone))
        with self.assertRaises(ValueError):
            x.deadline_fit('train', datetime(2026, 10, 20, 10, 0, tzinfo=zone))
        with self.assertRaises(ValueError):
            x.deadline_fit('eval', datetime(2026, 10, 20, 16, 0, tzinfo=zone))
        with self.assertRaises(ValueError): x.deadline_fit('unrecognized')

    def test_started_preflight_failure_stays_charged_idempotently(self):
        fixture = dict(additional_gpu_hours=57, prior_committed_gpu_hours=40,
                       allow_overlap=True, plans=list(x.PLANS), other_committed_gpu_hours=5,
                       authorization_evidence='TEST ONLY')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'fixture.json'); p.write_text(json.dumps(fixture))
            store = Path(tmp, 'new-artifacts')
            with patch.object(x, 'NEW', store), patch.object(x, 'submission_deadline_fit', side_effect=ValueError('expired')):
                with patch.dict(os.environ, SLURM_JOB_ID='123', SLURM_GPUS_ON_NODE='2'):
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError, 'expired'):
                            x.charge_start('stage2', 'train', p)
            ledger = x.read(store/'allocations.json')
            self.assertEqual(len(ledger['jobs']), 1)
            self.assertEqual(ledger['jobs'][0]['charged_gpu_hours'], 12)
            self.assertIsNone(ledger['jobs'][0]['binding'])
            x.validate_ledger(ledger)
            ledger['jobs'][0]['phase'] = 'invented'
            with self.assertRaises(ValueError): x.validate_ledger(ledger)

    def test_no_second_final_eval_registration(self):
        fixture = dict(additional_gpu_hours=57, prior_committed_gpu_hours=40,
                       allow_overlap=True, plans=list(x.PLANS), other_committed_gpu_hours=5,
                       authorization_evidence='TEST ONLY')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'fixture.json'); p.write_text(json.dumps(fixture))
            store = Path(tmp, 'new-artifacts'); store.mkdir()
            ledger = dict(schema='two-plan-ledger-v1', prior_committed_gpu_hours=40, jobs=[
                dict(plan='stage2', phase='eval', job_id='1', charged_gpu_hours=8),
                dict(plan='stage2', phase='eval', job_id='2', charged_gpu_hours=8)])
            x.write_json(store/'allocations.json', ledger)
            with patch.object(x, 'NEW', store), patch.object(x, 'training_terminal'):
                with patch.dict(os.environ, SLURM_JOB_ID='2', SLURM_GPUS_ON_NODE='1'):
                    with self.assertRaisesRegex(ValueError, 'one final evaluation'):
                        x.register('stage2', 'eval', {'output_dir': str(Path(tmp, 'stage2-hard3k-20261019'))}, {}, p)

    def submission_fixture(self, tmp):
        auth = Path(tmp, 'authorization.json')
        auth.write_text(json.dumps(dict(additional_gpu_hours=57, prior_committed_gpu_hours=40,
            allow_overlap=True, plans=list(x.PLANS), other_committed_gpu_hours=5,
            authorization_evidence='TEST ONLY: explicit bounded allocation authorization')))
        store = Path(tmp, 'artifacts'); store.mkdir()
        binding = {'fixture': 'bound CPU receipt', 'convergence': {'convergence_job_id': '445'}}
        x.write_json(store / 'stage2.execution_receipt.json', binding)
        return auth, store, {'output_dir': str(Path(tmp, 'stage2-hard3k-20261019'))}, binding

    def test_submit_records_actual_id_and_idempotent_start_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            completed = Mock(stdout='123456;owl\n')
            with patch.object(x, 'NEW', store), patch.object(x, 'submission_deadline_fit', return_value=14), \
                    patch.object(x.subprocess, 'run', return_value=completed) as call:
                self.assertEqual(x.submit('stage2', 'train', config, binding, auth), '123456')
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[0], ['sbatch', '--parsable',
                    '--job-name=reasoning_stage2_train', '--gres=gpu:2', '--time=06:00:00', '--dependency=afterok:445',
                    'scripts/two_plan_20261019_train.slurm', 'stage2'])
                ledger = x.read(store/'allocations.json')
                self.assertEqual(ledger['jobs'][0]['charged_gpu_hours'], 12)
                self.assertEqual(ledger['jobs'][0]['status'], 'submitted_reserved')
                self.assertEqual(ledger['submission_journal'][0]['status'], 'confirmed')
                with patch.dict(os.environ, SLURM_JOB_ID='123456', SLURM_GPUS_ON_NODE='2'), \
                        patch.object(x, 'submission_deadline_fit'):
                    x.charge_start('stage2', 'train', auth)
                    x.charge_start('stage2', 'train', auth)
                ledger = x.read(store/'allocations.json')
                self.assertEqual(len(ledger['jobs']), 1)
                self.assertEqual(ledger['jobs'][0]['charged_gpu_hours'], 12)
                self.assertEqual(ledger['jobs'][0]['binding'], binding)

    def test_coverage_submission_reserves_four_gpu_six_hours_and_blocks_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            x.write_json(store/'assistance_coverage.execution_receipt.json', binding)
            config['output_dir'] = str(Path(tmp, 'lmf-assistance-coverage-20261019'))
            with patch.object(x, 'NEW', store), patch.object(x, 'submission_deadline_fit', return_value=14), \
                    patch.object(x.subprocess, 'run', return_value=Mock(stdout='888\n')) as call:
                x.submit('assistance_coverage', 'train', config, binding, auth)
                self.assertIn('--gres=gpu:4', call.call_args.args[0])
                self.assertIn('--time=06:00:00', call.call_args.args[0])
                ledger = x.read(store/'allocations.json')
                self.assertEqual(ledger['jobs'][0]['charged_gpu_hours'], 24)
                with patch.object(x, 'training_terminal'):
                    with self.assertRaisesRegex(ValueError, 'one training allocation per plan'):
                        x.submit('assistance_coverage', 'train', config, binding, auth)
                self.assertEqual(call.call_count, 1)

    def test_auxiliary_unknown_submission_and_stale_cpu_binding_block_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            with patch.object(x, 'NEW', store), patch.object(x.subprocess, 'run') as call:
                with self.assertRaisesRegex(ValueError, 'exact CPU'):
                    x.submit('stage2', 'train', config, {'changed': True, 'convergence': {'convergence_job_id': '445'}}, auth)
                x.write_json(store/'other_allocations.json', {'submission_journal': [{'status': 'unknown'}]})
                with self.assertRaisesRegex(ValueError, 'ancillary'):
                    x.submit('stage2', 'train', config, binding, auth)
                call.assert_not_called()

    def test_submit_rejects_budget_before_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            authorization = x.read(auth); authorization['additional_gpu_hours'] = 20
            auth.write_text(json.dumps(authorization))
            with patch.object(x, 'NEW', store), patch.object(x, 'submission_deadline_fit', return_value=14), \
                    patch.object(x.subprocess, 'run') as call:
                with self.assertRaisesRegex(ValueError, 'incremental cap'):
                    x.submit('stage2', 'train', config, binding, auth)
                call.assert_not_called()
                self.assertFalse((store/'allocations.json').exists())

    def test_submit_rejects_same_plan_active_and_any_prior_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            ledger = dict(schema='two-plan-ledger-v1', prior_committed_gpu_hours=40,
                jobs=[dict(plan='stage2', phase='train', job_id='77', charged_gpu_hours=12)])
            x.write_json(store/'allocations.json', ledger)
            with patch.object(x, 'NEW', store), patch.object(x, 'training_terminal', side_effect=ValueError('active')), \
                    patch.object(x.subprocess, 'run') as call:
                with self.assertRaisesRegex(ValueError, 'active'):
                    x.submit('stage2', 'train', config, binding, auth)
                call.assert_not_called()
            ledger['jobs'][0].update(phase='eval', charged_gpu_hours=8)
            x.write_json(store/'allocations.json', ledger)
            with patch.object(x, 'NEW', store), patch.object(x, 'training_terminal'), \
                    patch.object(x.subprocess, 'run') as call:
                for phase in ('train', 'eval'):
                    with self.assertRaisesRegex(ValueError, 'no second final'):
                        x.submit('stage2', phase, config, binding, auth)
                call.assert_not_called()

    def test_ambiguous_submission_persists_incident_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth, store, config, binding = self.submission_fixture(tmp)
            with patch.object(x, 'NEW', store), patch.object(x, 'submission_deadline_fit', return_value=14), \
                    patch.object(x.subprocess, 'run', side_effect=TimeoutError('connection lost')) as call:
                with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                    x.submit('stage2', 'train', config, binding, auth)
                self.assertEqual(x.read(store/'allocations.json')['submission_journal'][0]['status'], 'unknown')
                with self.assertRaisesRegex(ValueError, 'unreconciled'):
                    x.submit('stage2', 'train', config, binding, auth)
                self.assertEqual(call.call_count, 1)

    def test_initial_coverage_margin_and_continuation(self):
        zone = ZoneInfo('America/New_York')
        now = datetime(2026, 10, 20, 10, 0, tzinfo=zone)
        with self.assertRaisesRegex(ValueError, '14 hour'):
            x.submission_deadline_fit('assistance_coverage', 'train', True, now)
        self.assertEqual(x.submission_deadline_fit('assistance_coverage', 'train', False, now.replace(hour=9)), 14)
        self.assertEqual(x.submission_deadline_fit('stage2', 'train', True, now.replace(hour=9)), 14)
        self.assertEqual(x.submission_deadline_fit('stage2', 'eval', False, now), 8)

    def test_own_checkpoint_needs_both_rngs_and_finite_complete_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            state = dict(global_step=1, log_history=[dict(step=1, loss=1., grad_norm=1.)])
            (p/'trainer_state.json').write_text(json.dumps(state))
            for n in ('optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'config.json',
                      'tokenizer_config.json', 'model.safetensors'):
                (p/n).write_bytes(b'fixture')
            self.assertFalse(x.checkpoint_valid(p, 750))
            (p/'rng_state_1.pth').write_bytes(b'fixture')
            self.assertTrue(x.checkpoint_valid(p, 750))
            state['log_history'][0]['loss'] = float('inf')
            (p/'trainer_state.json').write_text(json.dumps(state))
            self.assertFalse(x.checkpoint_valid(p, 750))

    def test_measured_convergence_gate_pending_then_numeric_and_model_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp, 'new'); store.mkdir()
            parent = Path(tmp, 'parent')
            source = Path('artifacts/datasets-full-length-15k')
            fake_digest = lambda p: 'digest:' + str(p)
            names = {'model.safetensors', 'config.json', 'tokenizer_config.json', 'generation_config.json'}
            expected = dict(schema='stage2-final-parent-training-token-accuracy-input-v1',
                source_data_sha256=fake_digest(source/'advanced.json'),
                source_selection_sha256=fake_digest(source/'selection_manifest.json'),
                source_dataset_info_sha256=fake_digest(source/'dataset_info.json'),
                parent=str(parent), parent_completion_sha256=fake_digest(parent/'completion.json'),
                parent_files_sha256={n: fake_digest(parent/n) for n in names}, threshold=.95,
                rows=[{'id': str(i)} for i in range(200)], execution_sha256={'code.py': fake_digest('code.py')})
            receipt = {k: expected[k] for k in ('source_data_sha256', 'source_selection_sha256')}
            auxiliary = {'jobs': [dict(purpose='stage2_convergence', job_id='445', gpu_hours=1,
                input_receipt_sha256=fake_digest(store/'stage2-convergence-input.json'))]}
            measured = dict(schema='stage2-convergence-measured-v1', binding=expected,
                job_id='445', gpu_used=True, hardware='NVIDIA B200', correct_tokens=200,
                supervised_tokens=400, token_accuracy=.5,
                per_row_counts=[dict(id=str(i), correct=1, total=2) for i in range(200)],
                convergence_gate_decision=dict(approved=True, basis='measured_training_token_accuracy',
                    threshold=.95, accuracy=.5))
            def fake_read(path):
                if Path(path).name == 'stage2-convergence-input.json': return expected
                if Path(path).name == 'other_allocations.json': return auxiliary
                if Path(path).name == 'stage2-convergence-decision.json': return measured
                raise AssertionError('unexpected receipt read')
            with patch.object(x, 'NEW', store), patch.object(x, 'read', side_effect=fake_read), \
                    patch.object(x, 'digest', side_effect=fake_digest), \
                    patch.object(x, 'model_shards', return_value={'model.safetensors'}):
                pending_binding = x.convergence_binding(receipt, parent, allow_pending=True)
                with self.assertRaisesRegex(ValueError, 'actual measured'):
                    x.convergence_binding(receipt, parent)
                (store/'stage2-convergence-decision.json').write_text('fixture')
                self.assertEqual(x.convergence_binding(receipt, parent), pending_binding)
                measured['correct_tokens'] = 300
                with self.assertRaisesRegex(ValueError, 'arithmetic'):
                    x.convergence_binding(receipt, parent)
                measured.update(correct_tokens=1900, supervised_tokens=2000, token_accuracy=.95,
                    per_row_counts=[dict(id=str(i), correct=10 if i < 190 else 0, total=10) for i in range(200)],
                    convergence_gate_decision=dict(approved=False, basis='measured_training_token_accuracy',
                        threshold=.95, accuracy=.95))
                with self.assertRaisesRegex(ValueError, 'does not authorize'):
                    x.convergence_binding(receipt, parent)
                measured['job_id'] = '999'
                with self.assertRaisesRegex(ValueError, 'actual GPU job'):
                    x.convergence_binding(receipt, parent)

    def test_world_four_checkpoint_requires_all_four_rngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'trainer_state.json').write_text(json.dumps(dict(global_step=1,
                log_history=[dict(step=1, loss=1., grad_norm=1.)])))
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth',
                         'config.json', 'tokenizer_config.json', 'model.safetensors'):
                (p/name).write_bytes(b'fixture')
            self.assertFalse(x.checkpoint_valid(p, 5625, 4))
            (p/'rng_state_2.pth').write_bytes(b'fixture')
            self.assertFalse(x.checkpoint_valid(p, 5625, 4))
            (p/'rng_state_3.pth').write_bytes(b'fixture')
            self.assertTrue(x.checkpoint_valid(p, 5625, 4))

    def test_actual_ce_four_rank_two_microbatch_normalization(self):
        import torch
        from reasoning.memory_ce import causal_ce
        torch.set_num_threads(2)
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(314)
            inputs = torch.randn(8, 17, 7, dtype=dtype)
            reference = torch.randn(7, 13, dtype=dtype, requires_grad=True)
            labels = torch.randint(0, 13, (8, 17))
            for i in range(8):
                labels[i, :i+1] = -100
            labels[2, 12:] = -100
            count = (labels[:, 1:] != -100).sum()
            loss = causal_ce(inputs @ reference, labels, 13, num_items_in_batch=count)
            expected = torch.autograd.grad(loss, reference)[0]
            rank_gradients = []
            for rank in range(4):
                weights = reference.detach().clone().requires_grad_()
                # Two accumulation microbatches/rank. Native global-token loss is
                # scaled by world size before DDP averages four rank gradients.
                rank_loss = sum(4 * causal_ce(inputs[i:i+1] @ weights, labels[i:i+1],
                    13, num_items_in_batch=count) for i in (rank*2, rank*2+1))
                rank_gradients.append(torch.autograd.grad(rank_loss, weights)[0])
            actual = torch.stack(rank_gradients).mean(dim=0)
            torch.testing.assert_close(actual, expected,
                rtol=.03 if dtype == torch.bfloat16 else 2e-5,
                atol=.002 if dtype == torch.bfloat16 else 1e-7)
            logits = (inputs @ reference.detach()).requires_grad_()
            causal_ce(logits, labels, 13, num_items_in_batch=count).backward()
            # Causal shift: a label masked at t+1 must give zero gradient at t.
            shifted_mask = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:] == -100
            self.assertEqual(torch.count_nonzero(logits.grad[shifted_mask]).item(), 0)

    def test_path_traversal_shard_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../other.safetensors'}}))
            with self.assertRaises(ValueError): x.model_shards(p)

    def test_enabled_draft_is_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'enabled.yaml'); p.write_text('do_train: true\noverwrite_output_dir: false\n')
            with patch.dict(x.PLANS, {'fixture': dict(config=str(p))}):
                with self.assertRaisesRegex(ValueError, 'disabled'):
                    x.validate('fixture', training_runtime=False)


if __name__ == '__main__': unittest.main()
