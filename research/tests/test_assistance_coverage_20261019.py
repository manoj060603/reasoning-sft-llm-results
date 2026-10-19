import copy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1] / 'assistance_coverage_20261019.py'
spec = importlib.util.spec_from_file_location('assist', P)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def feature(i, cluster=0, tokens=200):
    return dict(id=f'{i:05d}', cluster=cluster, response_tokens=tokens,
                total_tokens=tokens+100, complete_box=True, repetition=0)


def records(row_id, normal=(False, False), assisted=(True, True)):
    from reasoning.demonstration_response import stable_seed
    out = []
    for c, values in [('normal', normal), ('assisted', assisted)]:
        for s, correct in enumerate(values):
            text = 'result'
            out.append(dict(key=f'{row_id}:{c}:{s}', id=row_id, condition=c, sample=s,
                seed=stable_seed(42, row_id, c, s), prefix_sha256=a.text_hash('first paragraph' if c == 'assisted' else ''),
                text=text, text_sha256=a.text_hash(text), finish_reason='stop', supported=True, correct=correct))
    return out


class AssistanceCoverageTests(unittest.TestCase):
    def test_exact_size_order_and_half(self):
        fs = [feature(i, i % 3, 128+i) for i in range(17000)]
        scores = {f['id']: .375 for f in fs[:9000]}
        picked, cells = a.select(fs, scores)
        reverse, reverse_cells = a.select(list(reversed(fs)), dict(reversed(list(scores.items()))))
        self.assertEqual(picked, reverse)
        self.assertEqual(cells, reverse_cells)
        self.assertEqual(len({f['id'] for f in picked}), 15000)
        self.assertTrue(all(c['selected_by_utility'] <= c['quota']//2 for c in cells))

    def test_unscored_fill_and_undersupplied_cells(self):
        fs = [feature(i, 0 if i < 2 else 1) for i in range(100)]
        picked, cells = a.select(fs, {'00001': .2}, 80)
        self.assertEqual(len(picked), 80)
        self.assertTrue(all(c['quota'] <= c['capacity'] for c in cells))
        self.assertGreater(sum(f['assistance_utility'] is None for f in picked), 0)
        self.assertTrue(all(c['selected_by_coverage'] + c['selected_by_utility'] == c['quota'] for c in cells))

    def test_quotas_match_frozen_candidate_a_and_ignore_unrelated_scores(self):
        from reasoning.full_length import select as reference_select
        fs = [feature(i, i % 5, 128+i) for i in range(110)]
        # Deliberately include benchmark/loss metadata; neither is a selector input.
        _, frozen = reference_select(fs, 70)
        chosen, cells = a.select(fs, {}, 70)
        for cell in cells:
            expected = frozen[str(cell['cluster'])]['quartile_quotas'][cell['length_quartile']]
            self.assertEqual(cell['quota'], expected)
        contaminated = [{**f, 'nll': float('inf'), 'benchmark_failed': True} for f in reversed(fs)]
        other, _ = a.select(contaminated, {'foreign_id': 1000}, 70)
        self.assertEqual([f['id'] for f in chosen], [f['id'] for f in other])

    def test_infeasible_and_duplicate_fail(self):
        with self.assertRaises(ValueError):
            a.select([feature(0)], {}, 2)
        with self.assertRaises(ValueError):
            a.select([feature(0), feature(0)], {}, 1)

    def test_utility_laplace_and_censor(self):
        rows = records('a')
        self.assertEqual(a.utility(rows), .375)
        self.assertEqual(a.utility(records('a', (True, True), (False, False))), 0)
        self.assertIsNone(a.utility(rows[:3]))
        for field, val in [('finish_reason', 'length'), ('supported', False), ('correct', 1)]:
            changed = copy.deepcopy(rows)
            changed[0][field] = val
            self.assertIsNone(a.utility(changed))

    def test_provenance_and_prefix_join(self):
        original = dict(id='a', instruction='problem', output='first paragraph\n\nrest')
        c = dict(id='a', prefix='first paragraph', prefix_reason=None,
                 prefix_sha256=a.text_hash('first paragraph'),
                 response_sha256=a.text_hash(original['output']), prompt_sha256=a.text_hash('problem'))
        check = lambda _: ('first paragraph', None)
        score, _ = a.joined_scores({'a': original}, [c], records('a'), check)
        self.assertEqual(score, {'a': .375})
        for field in ['response_sha256', 'prompt_sha256', 'prefix_sha256']:
            bad = dict(c, **{field: 'wrong'})
            score, _ = a.joined_scores({'a': original}, [bad], records('a'), check)
            self.assertEqual(score, {})
        for field, value in [('seed', 0), ('prefix_sha256', 'wrong'), ('text_sha256', 'wrong')]:
            bad = records('a'); bad[0][field] = value
            self.assertEqual(a.joined_scores({'a': original}, [c], bad, check)[0], {})
        self.assertEqual(a.joined_scores({'a': original}, [c], records('a'), lambda _: ('', 'answer_leak'))[0], {})
        with self.assertRaises(ValueError):
            a.joined_scores({'a': original}, [c], records('a')*2, check)


if __name__ == '__main__':
    unittest.main()
