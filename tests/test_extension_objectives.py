"""Numerical objective oracles; execute only in a native CPU allocation."""

import math
import unittest

import torch
import torch.nn.functional as F

from extension.objectives import objective_loss


class ObjectiveOracleTests(unittest.TestCase):
    def assertClose(self, actual, expected, atol=1e-6):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-6)

    def test_sft_equals_manual_equal_example_cross_entropy(self):
        logits = torch.tensor([[[1., 2., -1.], [2., 0., 1.]],
                               [[0., 2., 3.], [1., 1., 1.]]], requires_grad=True)
        targets = torch.tensor([[1, 2], [0, -100]])
        loss, metrics = objective_loss(logits, targets, chunk_tokens=1)
        manual = (F.cross_entropy(logits[0], targets[0]) +
                  F.cross_entropy(logits[1, :1], targets[1, :1])) / 2
        self.assertClose(loss, manual)
        self.assertClose(metrics["ce"], manual.detach())
        self.assertClose(metrics["weighted_ce"], manual.detach())
        self.assertEqual(metrics["effective_weight"].item(), 1.)
        self.assertEqual(metrics["active_counts"].tolist(), [2, 1])
        self.assertEqual(metrics["active_tokens"].item(), 3)
        loss.backward()
        self.assertEqual(logits.grad[1, 1].abs().sum().item(), 0.)

    def test_forward_kl_direction_matches_manual_distribution(self):
        p = torch.tensor([.7, .2, .1])
        q = torch.tensor([.2, .3, .5])
        logits = p.log().reshape(1, 1, 3).requires_grad_()
        reference = q.log().reshape(1, 1, 3).requires_grad_()
        loss, metrics = objective_loss(logits, torch.tensor([[1]]),
                                      objective="asft", reference_logits=reference, beta=.4)
        forward = (q * (q.log() - p.log())).sum()
        reverse = (p * (p.log() - q.log())).sum()
        self.assertClose(metrics["kl"], forward)
        self.assertGreater(abs(metrics["kl"].item() - reverse.item()), .01)
        self.assertClose(loss, -p[1] * p[1].log() + .4 * forward)
        loss.backward()
        self.assertIsNone(reference.grad)

    def test_identical_reference_has_zero_kl(self):
        logits = torch.tensor([[[2., -1., 0.], [.1, .2, .3]]], requires_grad=True)
        for mode in ("asft", "prior_anchored"):
            loss, metrics = objective_loss(logits, torch.tensor([[0, 2]]),
                                          objective=mode, reference_logits=logits)
            self.assertEqual(metrics["kl"].item(), 0.)
            self.assertTrue(loss.requires_grad)

    def test_online_probability_weights_are_detached(self):
        logits = torch.tensor([[[.4, -.3, 1.2]]], requires_grad=True)
        loss, _ = objective_loss(logits, torch.tensor([[0]]), objective="asft",
                                 reference_logits=torch.zeros_like(logits), beta=0.)
        p = logits.detach().softmax(-1)
        expected = p.clone()
        expected[0, 0, 0] -= 1
        expected *= p[0, 0, 0]
        loss.backward()
        self.assertClose(logits.grad, expected)

    def test_frozen_probability_weights_and_kl_reference_are_detached(self):
        logits = torch.tensor([[[.4, -.3, 1.2]]], requires_grad=True)
        reference = torch.tensor([[[1.4, .7, -.2]]], requires_grad=True)
        beta = .35
        loss, _ = objective_loss(logits, torch.tensor([[0]]), objective="prior_anchored",
                                 reference_logits=reference, beta=beta)
        p, q = logits.detach().softmax(-1), reference.detach().softmax(-1)
        expected = p.clone()
        expected[0, 0, 0] -= 1
        expected = expected * q[0, 0, 0] + beta * (p - q)
        loss.backward()
        self.assertClose(logits.grad, expected)
        self.assertIsNone(reference.grad)

    def test_normalization_uses_all_active_tokens_not_sum_of_weights(self):
        logits = torch.zeros((1, 2, 2), requires_grad=True)
        reference = torch.tensor([[[math.log(.1), math.log(.9)],
                                   [math.log(.3), math.log(.7)]]])
        loss, metrics = objective_loss(logits, torch.tensor([[0, 0]]),
                                      objective="prior_anchored", reference_logits=reference,
                                      beta=0.)
        self.assertClose(loss, torch.tensor(.2 * math.log(2)))
        self.assertClose(metrics["effective_weight"], torch.tensor(.2))
        self.assertGreater(abs(loss.item() - math.log(2)), .4)

    def test_chunks_and_checkpointing_preserve_loss_metrics_and_gradient(self):
        values = torch.arange(42, dtype=torch.float32).reshape(2, 7, 3) / 9
        reference = values.flip(-1).clone().requires_grad_()
        targets = torch.tensor([[-100, -100, 0, 1, 2, 0, 1],
                                [2, 1, -100, 0, 1, -100, 2]])
        for mode in ("sft", "asft", "prior_anchored"):
            baseline, baseline_metrics, baseline_grad = None, None, None
            for chunk, recompute in ((7, False), (1, True), (3, True), (4, False)):
                logits = values.clone().requires_grad_()
                loss, metrics = objective_loss(logits, targets, objective=mode,
                                              reference_logits=reference, beta=.2,
                                              chunk_tokens=chunk, checkpoint_chunks=recompute)
                loss.backward()
                if baseline is None:
                    baseline, baseline_metrics, baseline_grad = loss.detach(), metrics, logits.grad
                else:
                    self.assertClose(loss.detach(), baseline)
                    self.assertClose(logits.grad, baseline_grad)
                    for key in metrics:
                        self.assertClose(metrics[key], baseline_metrics[key])
                self.assertIsNone(reference.grad)

    def test_large_extreme_logits_and_gradients_remain_finite(self):
        for mode in ("sft", "asft", "prior_anchored"):
            logits = torch.tensor([[[1e20, -1e20, 0.], [-1e20, 1e20, 0.]]],
                                  requires_grad=True)
            reference = torch.zeros_like(logits)
            loss, metrics = objective_loss(logits, torch.tensor([[1, 0]]),
                                          objective=mode, reference_logits=reference,
                                          chunk_tokens=1)
            self.assertTrue(torch.isfinite(loss).item())
            for value in metrics.values():
                self.assertTrue(torch.isfinite(value).all().item())
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_float64_input_uses_fp32_objective_and_gradients_reach_input(self):
        logits = torch.tensor([[[.1, .4, -.2]]], dtype=torch.float64, requires_grad=True)
        loss, _ = objective_loss(logits, torch.tensor([[1]]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertEqual(logits.grad.dtype, torch.float64)
        self.assertGreater(logits.grad.abs().sum().item(), 0.)

    def test_metrics_never_retain_autograd_graph(self):
        logits = torch.ones((1, 2, 3), requires_grad=True)
        _, metrics = objective_loss(logits, torch.tensor([[0, 1]]), objective="asft",
                                    reference_logits=torch.zeros_like(logits))
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None
                            for value in metrics.values()))

    def test_no_grad_mode_is_supported(self):
        with torch.no_grad():
            loss, _ = objective_loss(torch.zeros((1, 2, 3)), torch.tensor([[0, 1]]),
                                      objective="prior_anchored",
                                      reference_logits=torch.ones((1, 2, 3)))
        self.assertFalse(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).item())


class ObjectiveInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.zeros((2, 3, 4), requires_grad=True)
        self.targets = torch.tensor([[0, 1, 2], [-100, 2, 3]])

    def test_anchored_modes_require_reference(self):
        for mode in ("asft", "prior_anchored"):
            with self.assertRaisesRegex(ValueError, "reference_logits required"):
                objective_loss(self.logits, self.targets, objective=mode)

    def test_invalid_modes_rejected(self):
        for mode in ("DFT", "", None, 3):
            with self.assertRaisesRegex(ValueError, "unknown objective"):
                objective_loss(self.logits, self.targets, objective=mode)

    def test_invalid_beta_rejected(self):
        for beta in (-.1, float("inf"), float("nan"), True, "0.05"):
            with self.assertRaisesRegex(ValueError, "beta must be finite"):
                objective_loss(self.logits, self.targets, beta=beta)

    def test_invalid_chunk_configuration_rejected(self):
        for chunk in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                objective_loss(self.logits, self.targets, chunk_tokens=chunk)
        with self.assertRaisesRegex(ValueError, "must be bool"):
            objective_loss(self.logits, self.targets, checkpoint_chunks=1)

    def test_invalid_student_shapes_or_dtypes_rejected(self):
        for logits in (torch.zeros((3, 4)), torch.zeros((2, 3, 4), dtype=torch.int64),
                       [[1., 2.]], None):
            with self.assertRaisesRegex(ValueError, "floating"):
                objective_loss(logits, self.targets)
        for shape in ((0, 3, 4), (2, 0, 4), (2, 3, 0)):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                objective_loss(torch.zeros(shape), self.targets)

    def test_target_shape_and_dtype_rejected(self):
        for target in (self.targets[:, :2], self.targets.float(), self.targets.bool(), None):
            with self.assertRaisesRegex(ValueError, "targets must be int64"):
                objective_loss(self.logits, target)

    def test_only_minus100_is_an_ignored_id(self):
        for bad in (-1, -101, 4):
            targets = self.targets.clone()
            targets[0, 0] = bad
            with self.assertRaisesRegex(ValueError, "invalid ID"):
                objective_loss(self.logits, targets)

    def test_empty_example_or_entire_mask_fails_closed(self):
        for targets in (torch.full_like(self.targets, -100), self.targets.clone()):
            targets[0] = -100
            with self.assertRaisesRegex(ValueError, "every example"):
                objective_loss(self.logits, targets)

    def test_invalid_reference_shape_and_dtype_rejected(self):
        for reference in (torch.zeros((2, 3, 5)), torch.zeros((2, 3, 4), dtype=torch.int64),
                          [[1., 2.]]):
            with self.assertRaisesRegex(ValueError, "same-shaped floating"):
                objective_loss(self.logits, self.targets, objective="asft",
                               reference_logits=reference)

    def test_cross_device_metadata_fails_before_tensor_operations(self):
        # Meta tensors need no GPU, allocation, model, or data; the guard must
        # reject their device before attempting any value-dependent operation.
        with self.assertRaisesRegex(ValueError, "targets.*share a device"):
            objective_loss(self.logits, torch.empty((2, 3), dtype=torch.int64, device="meta"))
        with self.assertRaisesRegex(ValueError, "reference_logits.*share a device"):
            objective_loss(self.logits, self.targets, objective="asft",
                           reference_logits=torch.empty((2, 3, 4), device="meta"))

    def test_nonfinite_student_even_at_masked_positions_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            logits = self.logits.detach().clone()
            logits[1, 0, 2] = invalid
            with self.assertRaisesRegex(ValueError, "must be finite"):
                objective_loss(logits, self.targets)

    def test_nonfinite_reference_rejected(self):
        reference = torch.zeros_like(self.logits)
        reference[1, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "reference_logits.*finite"):
            objective_loss(self.logits, self.targets, objective="asft",
                           reference_logits=reference)

    def test_fp32_conversion_overflow_fails_closed(self):
        logits = torch.tensor([[[1e300, 0.]]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "FP32 student log probabilities"):
            objective_loss(logits, torch.tensor([[0]]))


if __name__ == "__main__":
    unittest.main()
