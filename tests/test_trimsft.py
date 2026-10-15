import unittest

import torch

from reasoning.trimsft import trimsft_cross_entropy


class TrimSFTLossTests(unittest.TestCase):
    def test_matches_published_gaussian_weighting(self):
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.5]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = trimsft_cross_entropy(logits, labels, margin=1.5, bandwidth=0.8)

        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        gaps = torch.tensor([1.5, -0.5])
        weights = torch.exp(-((gaps - 1.5) ** 2) / (2 * 0.8**2))
        self.assertTrue(torch.allclose(loss, (ce * weights).mean()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ignored_tokens_do_not_contribute(self):
        logits = torch.tensor([[2.0, 0.5], [100.0, -100.0]], requires_grad=True)
        labels = torch.tensor([0, -100])
        one = trimsft_cross_entropy(logits, labels)
        reference = trimsft_cross_entropy(logits[:1], labels[:1])
        self.assertTrue(torch.allclose(one, reference))

    def test_all_ignored_is_differentiable_zero(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss = trimsft_cross_entropy(logits, torch.full((2,), -100))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_rejects_nonpositive_bandwidth(self):
        with self.assertRaises(ValueError):
            trimsft_cross_entropy(torch.randn(1, 3), torch.tensor([0]), bandwidth=0.0)


if __name__ == "__main__":
    unittest.main()
