import unittest

import torch

from reasoning.gem import gem_cross_entropy


class GEMLossTest(unittest.TestCase):
    def test_matches_naive_forward_and_gradient(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, 5, -100], [3, 8, 2]])
        loss = gem_cross_entropy(logits, labels, beta=0.7, vocab_chunk=4)
        loss.backward()
        actual_grad = logits.grad.clone()

        reference_logits = logits.detach().clone().requires_grad_(True)
        valid = labels.reshape(-1) != -100
        selected = reference_logits.reshape(-1, 11)[valid]
        targets = labels.reshape(-1)[valid]
        q = torch.softmax(selected / 0.7, dim=-1).detach()
        reference = ((q * selected).sum(-1) - selected.gather(1, targets[:, None]).squeeze(1)).mean()
        reference.backward()

        self.assertTrue(torch.allclose(loss, reference, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_grad, reference_logits.grad, atol=1e-10, rtol=1e-10))

    def test_all_ignored_is_differentiable_zero(self):
        logits = torch.randn(2, 4, 9, requires_grad=True)
        loss = gem_cross_entropy(logits, torch.full((2, 4), -100))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_rejects_invalid_parameters(self):
        logits = torch.randn(1, 2, 3)
        labels = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            gem_cross_entropy(logits, labels, beta=0.0)
        with self.assertRaises(ValueError):
            gem_cross_entropy(logits, labels, vocab_chunk=0)


if __name__ == "__main__":
    unittest.main()
