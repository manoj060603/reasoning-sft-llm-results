import json
import tempfile
import unittest
from pathlib import Path

from deadline.audit_results import OFFICIAL, PROTOCOL, REQUIRED_SETTINGS, audit_run, compare_coverage, markdown_report


class ResultAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name='run', raw=False):
        path = self.root / name
        path.mkdir()
        manifest = {key: 0 for key in PROTOCOL}
        manifest.update(REQUIRED_SETTINGS)
        manifest.update(n=8, problem_counts={'amc': 2}, problem_ids={'amc': ['a', 'b']}, data={'amc': 'sha256'})
        metric = dict(problems=2, samples_per_problem=8, pass_at_1=.5, pass_at_8=1.)
        (path / 'manifest.json').write_text(json.dumps(manifest))
        (path / 'metrics.json').write_text(json.dumps({'amc': metric}))
        if raw:
            (path / 'amc').mkdir()
            rows = [dict(problem_id=i, answers_correctness=[True] * 4 + [False] * 4,
                         generated_responses=['response'] * 8) for i in ('a', 'b')]
            (path / 'amc' / 'batch_000000.json').write_text(json.dumps(rows))
        return path

    def audit(self, path):
        return audit_run(path, expected={'amc': 2})

    def test_metrics_only_is_labeled(self):
        run = self.audit(self.fixture())
        self.assertEqual(run['benchmarks']['amc']['status'], 'complete')
        self.assertIn('aggregate receipt only', run['benchmarks']['amc']['evidence'])
        self.assertIsNone(run['official_eight_macro'])

    def test_raw_recomputed_and_verified(self):
        run = self.audit(self.fixture(raw=True))
        self.assertEqual(run['benchmarks']['amc']['evidence'], 'raw eight-sample predictions verified')
        self.assertEqual(run['benchmarks']['amc']['metrics']['pass_at_1'], .5)

    def test_incomplete_raw_cannot_fall_back_to_complete_metrics(self):
        path = self.fixture(raw=True)
        p = path / 'amc' / 'batch_000000.json'
        p.write_text(json.dumps(json.loads(p.read_text())[:1]))
        self.assertEqual(self.audit(path)['benchmarks']['amc']['status'], 'invalid')

    def test_seven_flags_rejected(self):
        path = self.fixture(raw=True)
        p = path / 'amc' / 'batch_000000.json'
        rows = json.loads(p.read_text())
        rows[0]['answers_correctness'].pop()
        p.write_text(json.dumps(rows))
        self.assertEqual(self.audit(path)['benchmarks']['amc']['status'], 'invalid')

    def test_duplicate_question_record_ids_rejected(self):
        path = self.fixture(raw=True)
        p = path / 'amc' / 'batch_000000.json'
        rows = json.loads(p.read_text())
        rows[1]['problem_id'] = rows[0]['problem_id']
        p.write_text(json.dumps(rows))
        self.assertEqual(self.audit(path)['benchmarks']['amc']['status'], 'invalid')

    def test_aggregate_disagrees_with_raw_rejected(self):
        path = self.fixture(raw=True)
        p = path / 'metrics.json'
        data = json.loads(p.read_text())
        data['amc']['pass_at_1'] = .25
        p.write_text(json.dumps(data))
        self.assertEqual(self.audit(path)['benchmarks']['amc']['status'], 'invalid')

    def test_mismatched_data_or_sampling_prevents_matched_claim(self):
        runs = {r: self.audit(self.fixture(r)) for r in ('original', 'random', 'advanced')}
        self.assertTrue(compare_coverage(runs, names=('amc',))['matched'])
        runs['random']['manifest']['data']['amc'] = 'different hash'
        self.assertFalse(compare_coverage(runs, names=('amc',))['matched'])
        runs['random']['manifest']['data']['amc'] = 'sha256'
        runs['advanced']['manifest']['temperature'] = .7
        self.assertFalse(compare_coverage(runs, names=('amc',))['matched'])

    def test_missing_random_is_not_a_completed_comparison(self):
        runs = {'original': self.audit(self.fixture())}
        result = compare_coverage(runs, names=('amc',))
        self.assertFalse(result['matched'])
        report = markdown_report(runs, result)
        self.assertIn('random: no evaluation folder supplied.', report)
        self.assertIn('| amc | 50.00% / 100.00% | — | — |', report)

    def test_changed_evaluator_source_prevents_matched_claim(self):
        runs = {r: self.audit(self.fixture(r)) for r in ('original', 'random', 'advanced')}
        runs['advanced']['manifest']['evaluator_sha256'] = 'changed evaluator'
        result = compare_coverage(runs, names=('amc',))
        self.assertFalse(result['matched'])
        self.assertTrue(any('evaluator_sha256' in error for error in result['errors']))

    def test_nonfinite_or_impossible_metrics_rejected(self):
        path = self.fixture()
        p = path / 'metrics.json'
        data = json.loads(p.read_text())
        data['amc']['pass_at_1'] = float('nan')
        p.write_text(json.dumps(data))
        self.assertEqual(self.audit(path)['benchmarks']['amc']['status'], 'invalid')

    def test_macro_requires_every_named_benchmark(self):
        path = self.fixture()
        manifest = json.loads((path / 'manifest.json').read_text())
        metrics = {}
        manifest.update(problem_counts=dict(OFFICIAL),
                        problem_ids={b: [str(i) for i in range(n)] for b, n in OFFICIAL.items()},
                        data={b: 'sha256' for b in OFFICIAL})
        for benchmark, count in OFFICIAL.items():
            metrics[benchmark] = dict(problems=count, samples_per_problem=8,
                                      pass_at_1=.5, pass_at_8=1.)
        (path / 'manifest.json').write_text(json.dumps(manifest))
        (path / 'metrics.json').write_text(json.dumps(metrics))
        run = audit_run(path)
        self.assertEqual(run['official_eight_macro'], dict(pass_at_1=.5, pass_at_8=1.))
        self.assertIsNone(run['supplemented_historical_seven_macro'])
        del metrics['gpqa']
        (path / 'metrics.json').write_text(json.dumps(metrics))
        self.assertIsNone(audit_run(path)['official_eight_macro'])

    def test_same_but_non_assignment_sampling_rejected(self):
        path = self.fixture()
        p = path / 'manifest.json'
        manifest = json.loads(p.read_text())
        manifest['max_tokens'] = 8192
        p.write_text(json.dumps(manifest))
        run = self.audit(path)
        self.assertNotIn('manifest', run)
        self.assertIn('max_tokens', run['errors'][0])


if __name__ == '__main__':
    unittest.main()
