import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from reasoning import two_plan_dev_20261019 as x
from reasoning.data import digest, prompt_id


class TwoPlanDevelopmentTests(unittest.TestCase):
    def test_streaming_array_and_normalized_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'rows.json')
            rows = [{'id': i, 'text': 'x' * (i + 9)} for i in range(8)]
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            self.assertEqual(list(x.iter_json_array(path, chunk_size=7)), rows)
            row = {'question': '  A\u00a0 +  B  '}
            self.assertEqual(x.normalized_id(row), prompt_id('A + B'))

    def test_panel_selection_is_order_invariant_exact_and_no_backfill(self):
        rows = [dict(id=prompt_id(f'Question {i}'), instruction=f'Question {i}', input='',
                     output=f'Reason. \\boxed{{{i}}}', system='s') for i in range(240)]
        excluded = {rows[0]['id']}; benchmark = {rows[1]['id']}; train = {rows[2]['id']}
        with patch.object(x, 'official_answer_supported', return_value=True):
            first = x.select_panel(rows, excluded, benchmark, train)
            second = x.select_panel(list(reversed(rows)), excluded, benchmark, train)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len({r['id'] for r in first}), 200)
            self.assertFalse({r['id'] for r in first} & (excluded | benchmark | train))
            with self.assertRaisesRegex(ValueError, 'fewer than 200'):
                x.select_panel(rows[:202], excluded, benchmark, train)

    def test_prepare_panel_binds_original_heldout_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); art = root/'art'; validation = root/'validation.json'
            pool = root/'pool.json'; manifest = root/'manifest.json'; selection = root/'selection.json'
            dev = art/'dev/acereason_dev/test.jsonl'; meta = Path(str(dev)+'.meta.json')
            receipt = art/'panel.json'
            rows = [dict(id=prompt_id(f'Q {i}'), instruction=f'Q {i}', input='',
                         output=f'Original reasoning {i}. \\boxed{{{i}}}', system='s') for i in range(1000)]
            validation.write_text(json.dumps(rows)); pool.write_text('[]'); selection.write_text('{}')
            manifest.write_text(json.dumps(dict(original_pool_sha256='pool',
                original_validation_sha256=digest(validation), train_before=83037, train_after=83037,
                validation_before=1000, validation_after=1000, heldout_rows_never_moved_to_train=True)))
            old = {rows[0]['id']}; bench = {rows[1]['id']}; train = {prompt_id('separate training row')}
            with patch.object(x, 'ART', art), patch.object(x, 'DEV_ROOT', art/'dev'), \
                 patch.object(x, 'DEV', dev), patch.object(x, 'DEV_META', meta), \
                 patch.object(x, 'PANEL_RECEIPT', receipt), patch.object(x, 'VALIDATION', validation), \
                 patch.object(x, 'EXTENDED_MANIFEST', manifest), patch.object(x, 'POOL', pool), \
                 patch.object(x, 'SELECTION', selection), patch.object(x, 'VALIDATION_SHA', digest(validation)), \
                 patch.object(x, 'POOL_SHA', 'pool'), patch.object(x, 'prior_dev_bindings', return_value=(old, {'old': {}})), \
                 patch.object(x, 'benchmark_question_bindings', return_value=(bench, {'bench': {}})), \
                 patch.object(x, 'pool_identities', return_value=train), \
                 patch.object(x, 'official_answer_supported', return_value=True):
                x.prepare_panel()
                made = x.read_rows(dev); proof = x.read(receipt)
                self.assertEqual(len(made), 200)
                self.assertTrue(proof['all_rows_from_original_validation_bytes'])
                by_id = {r['id']: r for r in rows}
                self.assertTrue(all(r['answer'] in by_id[r['id']]['output'] for r in made))

    def test_complete_200x8_result_and_truncation_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dev = root/'dev.jsonl'; run = root/'run'; bench = run/'acereason_dev'
            rows = [dict(question=f'Q{i}', answer=str(i), id=prompt_id(f'Q{i}')) for i in range(200)]
            dev.write_text(''.join(json.dumps(r)+'\n' for r in rows)); bench.mkdir(parents=True)
            ids = [r['id'] for r in rows]
            manifest = {'data': {'acereason_dev': digest(dev)}, 'problem_counts': {'acereason_dev': 200},
                        'problem_ids': {'acereason_dev': ids}}
            (run/'manifest.json').write_text(json.dumps(manifest))
            generated = [dict(problem_id=pid, generated_responses=['x']*8,
                answers_correctness=[i % 3 == 0] + [False]*7,
                finish_reasons=(['length'] if i < 80 else ['stop']) + ['stop']*7,
                output_tokens=[1]*8) for i, pid in enumerate(ids)]
            (bench/'batch_000000.json').write_text(json.dumps(generated))
            with patch.object(x, 'DEV', dev):
                _, _, score = x.result_summary(run)
            self.assertEqual(score['truncated_samples'], 80)
            self.assertEqual(score['total_samples'], 1600)
            self.assertEqual(score['solved_questions'], 67)

    def test_frozen_gate_needs_two_questions_and_at_most_eighty_truncations(self):
        parent = dict(pass_at_1=.2, pass_at_8=.5, solved_questions=100,
                      truncated_samples=80, total_samples=1600)
        candidate = dict(pass_at_1=.19, pass_at_8=.51, solved_questions=102,
                         truncated_samples=80, total_samples=1600)
        self.assertTrue(x.compare_gate(parent, candidate)['approved_for_terminal_evaluation'])
        candidate.update(pass_at_8=.505, solved_questions=101, truncated_samples=79)
        gate = x.compare_gate(parent, candidate)
        self.assertFalse(gate['approved_for_terminal_evaluation'])
        self.assertEqual(gate['informational_tie_policy_winner'], 'candidate')
        candidate.update(pass_at_8=.51, solved_questions=102, truncated_samples=81)
        self.assertFalse(x.compare_gate(parent, candidate)['approved_for_terminal_evaluation'])
        parent['truncated_samples'] = 81
        self.assertFalse(x.compare_gate(parent, candidate)['qualified_control'])

    def test_auxiliary_budget_exact_five_hour_boundary(self):
        auth = {'additional_gpu_hours': 57, 'other_committed_gpu_hours': 5}
        central = {'schema': 'two-plan-ledger-v1', 'prior_committed_gpu_hours': 40,
                   'jobs': [dict(plan='assistance_coverage', phase='train', job_id='1', charged_gpu_hours=24),
                            dict(plan='stage2', phase='train', job_id='2', charged_gpu_hours=12)]}
        auxiliary = {'jobs': [dict(purpose='stage2_convergence', job_id='3', gpu_hours=1)]}
        x.auxiliary_budget(auth, central, auxiliary, 2)
        auxiliary['jobs'].append(dict(purpose='parent_dev', job_id='4', gpu_hours=2))
        x.auxiliary_budget(auth, central, auxiliary, 2)
        with self.assertRaisesRegex(ValueError, 'reserve exhausted'):
            x.auxiliary_budget(auth, central, auxiliary, 3)

    def test_submit_records_actual_id_once_and_unknown_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp, 'art'); art.mkdir(); auth_path = art/'auth.json'; proof = art/'input.json'
            auth_path.write_text('{}'); proof.write_text('{}')
            fake_auth = {'additional_gpu_hours': 57, 'other_committed_gpu_hours': 5}
            phase = dict(x.PHASES['parent']); phase.update(input=proof, decision=art/'decision.json')
            with patch.object(x, 'ART', art), patch.dict(x.PHASES, {'parent': phase}, clear=False), \
                 patch.object(x, 'authorization', return_value=fake_auth), \
                 patch.object(x, 'input_binding', return_value={}), patch.object(x, 'deadline_fit'), \
                 patch.object(x, 'digest', return_value='0'*64), \
                 patch.object(x.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='123;owl\n')) as sbatch:
                x.submit('parent', auth_path)
                ledger = x.read(art/'other_allocations.json')
                self.assertEqual(ledger['jobs'][0]['job_id'], '123')
                self.assertEqual(ledger['jobs'][0]['gpu_hours'], 2)
                with self.assertRaisesRegex(ValueError, 'one parent_dev'):
                    x.submit('parent', auth_path)
                self.assertEqual(sbatch.call_count, 1)
            (art/'other_allocations.json').unlink()
            with patch.object(x, 'ART', art), patch.dict(x.PHASES, {'parent': phase}, clear=False), \
                 patch.object(x, 'authorization', return_value=fake_auth), \
                 patch.object(x, 'input_binding', return_value={}), patch.object(x, 'deadline_fit'), \
                 patch.object(x, 'digest', return_value='0'*64), \
                 patch.object(x.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 45)) as sbatch:
                with self.assertRaises(subprocess.TimeoutExpired): x.submit('parent', auth_path)
                self.assertEqual(x.read(art/'other_allocations.json')['submission_journal'][0]['status'], 'unknown')
                with self.assertRaisesRegex(ValueError, 'reconciliation'): x.submit('parent', auth_path)
                self.assertEqual(sbatch.call_count, 1)

    def test_deadline_and_runtime_protocol_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            x.deadline_fit(datetime(2026, 10, 20, 15, 0, tzinfo=ZoneInfo('America/New_York')))
        self.assertIsNone(x.deadline_fit(datetime(2026, 10, 20, 13, 0,
                                                  tzinfo=ZoneInfo('America/New_York'))))
        self.assertEqual(x.PHASES['parent']['label'], 'C1_original_final5625_control')
        self.assertEqual(x.PHASES['stage2']['label'], 'Stage2_candidate')


if __name__ == '__main__': unittest.main()
