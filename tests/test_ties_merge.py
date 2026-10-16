import unittest


class TiesMergeTests(unittest.TestCase):
    def test_sign_conflicts_follow_larger_mass(self):
        import torch
        from reasoning.ties_merge import ties_merge_tensor

        base = torch.zeros(4)
        first = torch.tensor([4.0, 2.0, -5.0, 1.0])
        second = torch.tensor([-1.0, 3.0, 2.0, -4.0])
        actual = ties_merge_tensor(base, [first, second], density=1.0, scale=1.0, chunk_size=2)
        expected = torch.tensor([4.0, 2.5, -5.0, -4.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_scale_rebases_merged_delta(self):
        import torch
        from reasoning.ties_merge import ties_merge_tensor

        base = torch.ones(3)
        experts = [torch.tensor([3.0, 5.0, 7.0]), torch.tensor([5.0, 3.0, 9.0])]
        actual = ties_merge_tensor(base, experts, density=1.0, scale=0.5)
        self.assertTrue(torch.equal(actual, torch.tensor([2.5, 2.5, 4.5], dtype=torch.bfloat16)))

    def test_nonfloating_values_must_match(self):
        import torch
        from reasoning.ties_merge import ties_merge_tensor

        base = torch.tensor([1, 2])
        with self.assertRaises(ValueError):
            ties_merge_tensor(base, [base.clone(), torch.tensor([1, 3])], 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
