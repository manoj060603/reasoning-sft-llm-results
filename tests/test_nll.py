import unittest

from reasoning.nll import make_batches, response_nll


class NllTests(unittest.TestCase):
    def test_make_batches_respects_padded_budget(self):
        records = [("a", 1, [1, 2]), ("b", 1, [1, 2, 3]), ("c", 1, list(range(7)))]
        batches = list(make_batches(records, 8))
        self.assertEqual([[x[0] for x in b] for b in batches], [["a", "b"], ["c"]])
        self.assertTrue(all(max(len(x[2]) for x in b) * len(b) <= 8 for b in batches))

    def test_response_nll_masks_prompt_and_padding(self):
        import torch
        vocab = 5
        ids = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 1, 0, 0]])
        logits = torch.zeros(2, 5, vocab)
        # Make only the response predictions perfect. Example zero response is
        # original tokens 2,3,4; example one response is original token 2.
        for batch, positions in ((0, (1, 2, 3)), (1, (1,))):
            for position in positions:
                logits[batch, position, ids[batch, position + 1]] = 20
        losses = response_nll(logits, ids, [2, 2], [3, 1])
        self.assertTrue(torch.all(losses < 1e-6), losses)


if __name__ == "__main__":
    unittest.main()
