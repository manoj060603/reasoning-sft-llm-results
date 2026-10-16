import unittest

import torch
import torch.nn.functional as F

from reasoning.asft import anchored_loss


class AnchoredLossTests(unittest.TestCase):
    def test_matches_direct_dft_plus_kl(self):
        torch.manual_seed(7)
        online = torch.randn(2, 4, 11, requires_grad=True)
        ref = torch.randn(2, 4, 11)
        labels = torch.tensor([[-100, 2, 4, 1], [-100, 3, -100, 6]])
        actual = anchored_loss(online, ref, labels, kl_weight=.03, token_chunk=2)
        mask = labels != -100
        logp = F.log_softmax(online[mask], -1)
        gold = labels[mask]
        gold_logp = logp.gather(1, gold[:, None]).squeeze(1)
        p_ref = F.softmax(ref[mask], -1)
        expected = (-gold_logp * gold_logp.detach().exp()
                    + .03 * (p_ref * (p_ref.log() - logp)).sum(-1)).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        grad = torch.autograd.grad(actual, online)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.equal(grad[~mask], torch.zeros_like(grad[~mask])))

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            anchored_loss(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
