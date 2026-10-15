import unittest

import torch

from reasoning.prift import prift_cross_entropy, reference_mass


class PriFTLossTests(unittest.TestCase):
    def test_reference_mass_matches_brute_force(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.5, 1.5]])
        labels = torch.tensor([1, 0])
        probs = torch.softmax(logits, dim=-1)
        expected = torch.stack(
            [probs[i][probs[i] <= probs[i, labels[i]]].sum() for i in range(2)]
        )
        self.assertTrue(torch.allclose(reference_mass(logits, labels, vocab_chunk=2), expected))

    def test_prob_weighted_loss(self):
        online = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        labels = torch.tensor([[0, 1]])
        loss = prift_cross_entropy(online, reference, labels, method="prob")
        ce = torch.nn.functional.cross_entropy(online.reshape(-1, 2), labels.reshape(-1), reduction="none")
        weights = torch.softmax(reference.reshape(-1, 2), -1).gather(1, labels.reshape(-1, 1)).squeeze(1)
        self.assertTrue(torch.allclose(loss, (ce * weights).sum() / 2))
        loss.backward()
        self.assertTrue(torch.isfinite(online.grad).all())

    def test_mass_masks_unsupported_tokens_and_ignores_prompt(self):
        online = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        reference = torch.tensor([[[3.0, 0.0], [3.0, 0.0]]])
        labels = torch.tensor([[0, -100]])
        loss = prift_cross_entropy(online, reference, labels, method="mass", mass_threshold=0.5)
        expected = torch.nn.functional.cross_entropy(online[:, 0], torch.tensor([0]))
        self.assertTrue(torch.allclose(loss, expected))

    def test_all_ignored_is_differentiable_zero(self):
        online = torch.randn(1, 2, 3, requires_grad=True)
        loss = prift_cross_entropy(online, torch.randn_like(online), torch.full((1, 2), -100))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(online.grad, torch.zeros_like(online)))


if __name__ == "__main__":
    unittest.main()
