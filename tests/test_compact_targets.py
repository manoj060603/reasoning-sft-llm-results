import unittest

from reasoning.compact_targets import domain, quotas, repeated_ngram_fraction, visible_solution
from reasoning.domination import required_correct


class CompactTargetsTest(unittest.TestCase):
    def test_visible_solution_removes_hidden_trace(self):
        text = "<think>long scratch work</think>Proof with steps. \\boxed{7}"
        self.assertEqual(visible_solution(text), "Proof with steps. \\boxed{7}")

    def test_malformed_nested_trace_is_rejected(self):
        self.assertEqual(visible_solution("answer <think>unfinished"), "")

    def test_domain_rules(self):
        self.assertEqual(domain("Find the remainder modulo 17 for this prime"), "number_theory")
        self.assertEqual(domain("A triangle is tangent to a circle"), "geometry")

    def test_quota_sum_and_capacity(self):
        caps = {k: 100 for k in ("geometry", "number_theory", "algebra", "combinatorics", "calculus", "probability", "other_math")}
        q = quotas(500, caps)
        self.assertEqual(sum(q.values()), 500)
        self.assertTrue(all(q[k] <= caps[k] for k in q))

    def test_repetition(self):
        repeated = " ".join(["a b c d e f g h"] * 4)
        self.assertGreater(repeated_ngram_fraction(repeated), 0)

    def test_strict_discrete_targets(self):
        self.assertEqual(required_correct(7), 8)
        self.assertEqual(required_correct(450), 451)


if __name__ == "__main__":
    unittest.main()

