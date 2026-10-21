"""Small numerical oracles; execute later in a native CPU allocation only."""
import math
import unittest

import torch
import torch.nn.functional as F

from extension import objectives as prototype
from extension.preserving_objectives import MODES, objective_loss


class PreservationObjectiveTests(unittest.TestCase):
    def assertClose(self, actual, expected, atol=1e-6, rtol=1e-6):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_modes_are_exactly_the_three_existing_modes_plus_ce_kl(self):
        self.assertEqual(MODES, frozenset({"sft", "ce_kl", "asft", "prior_anchored"}))

    def test_all_old_modes_preserve_prototype_loss_gradient_and_metrics(self):
        values = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5) / 11
        targets = torch.tensor([[-100, 0, 2, 4], [1, -100, 3, -100]])
        for mode in sorted(prototype.MODES):
            for dtype in (torch.float32, torch.bfloat16, torch.float64):
                for recompute in (False, True):
                    with self.subTest(mode=mode, dtype=dtype, checkpoint=recompute):
                        new = values.to(dtype).clone().requires_grad_()
                        old = values.to(dtype).clone().requires_grad_()
                        new_reference = values.flip(-1).to(dtype).clone().requires_grad_()
                        old_reference = new_reference.detach().clone().requires_grad_()
                        kwargs = dict(objective=mode, beta=.27, chunk_tokens=2,
                                      checkpoint_chunks=recompute)
                        actual, actual_metrics = objective_loss(new, targets,
                            reference_logits=new_reference, **kwargs)
                        expected, expected_metrics = prototype.objective_loss(old, targets,
                            reference_logits=old_reference, **kwargs)
                        actual.backward()
                        expected.backward()
                        self.assertClose(actual, expected, atol=0, rtol=0)
                        self.assertClose(new.grad, old.grad, atol=0, rtol=0)
                        self.assertEqual(set(actual_metrics), set(expected_metrics))
                        for key in actual_metrics:
                            self.assertClose(actual_metrics[key], expected_metrics[key], atol=0, rtol=0)
                        self.assertIsNone(new_reference.grad)
                        self.assertIsNone(old_reference.grad)

    def test_ce_kl_matches_manual_forward_kl_and_analytic_gradient(self):
        p = torch.tensor([.7, .2, .1])
        q = torch.tensor([.2, .3, .5])
        logits = p.log().reshape(1, 1, 3).requires_grad_()
        reference = q.log().reshape(1, 1, 3).requires_grad_()
        beta = .4
        loss, metrics = objective_loss(logits, torch.tensor([[1]]), objective="ce_kl",
            reference_logits=reference, beta=beta, checkpoint_chunks=False)
        forward = (q * (q.log() - p.log())).sum()
        reverse = (p * (p.log() - q.log())).sum()
        self.assertClose(loss, -p[1].log() + beta * forward)
        self.assertClose(metrics["kl"], forward)
        self.assertGreater(abs(metrics["kl"].item() - reverse.item()), .01)
        self.assertClose(metrics["weighted_ce"], -p[1].log())
        self.assertEqual(metrics["effective_weight"].item(), 1.)
        expected_gradient = p.clone()
        expected_gradient[1] -= 1
        expected_gradient += beta * (p - q)
        loss.backward()
        self.assertClose(logits.grad[0, 0], expected_gradient)
        self.assertIsNone(reference.grad)

    def test_ce_kl_uses_every_vocabulary_entry_not_only_target_probability(self):
        student = torch.tensor([[[.6, .2, .2]]]).log()
        first = torch.tensor([[[.4, .5, .1]]]).log()
        second = torch.tensor([[[.4, .3, .3]]]).log()
        target = torch.tensor([[0]])
        first_loss, first_metrics = objective_loss(student, target, objective="ce_kl",
            reference_logits=first, beta=1.)
        second_loss, second_metrics = objective_loss(student, target, objective="ce_kl",
            reference_logits=second, beta=1.)
        self.assertClose(first_metrics["ce"], second_metrics["ce"])
        self.assertGreater(abs(first_loss.item() - second_loss.item()), .05)
        self.assertGreater(abs(first_metrics["kl"].item() - second_metrics["kl"].item()), .05)

    def test_zero_beta_equals_plain_ce_loss_and_gradient(self):
        values = torch.tensor([[[.1, .7, -.2], [1., .3, 0.]],
                               [[2., -1., .5], [.2, .1, -.1]]])
        targets = torch.tensor([[1, 2], [0, -100]])
        actual_logits = values.clone().requires_grad_()
        expected_logits = values.clone().requires_grad_()
        actual, metrics = objective_loss(actual_logits, targets, objective="ce_kl",
            reference_logits=values.flip(-1), beta=0., chunk_tokens=1)
        expected, _ = prototype.objective_loss(expected_logits, targets, chunk_tokens=1)
        actual.backward()
        expected.backward()
        self.assertClose(actual, expected)
        self.assertClose(actual_logits.grad, expected_logits.grad)
        self.assertGreater(metrics["kl"].item(), 0.)

    def test_equal_student_reference_has_zero_kl_and_plain_ce_gradient(self):
        values = torch.tensor([[[2., -1., 0.], [.1, .2, .3]]])
        student = values.clone().requires_grad_()
        reference = values.clone().requires_grad_()
        baseline = values.clone().requires_grad_()
        targets = torch.tensor([[0, 2]])
        loss, metrics = objective_loss(student, targets, objective="ce_kl",
            reference_logits=reference, beta=3.)
        expected, _ = objective_loss(baseline, targets)
        self.assertEqual(metrics["kl"].item(), 0.)
        self.assertClose(loss, expected)
        loss.backward()
        expected.backward()
        self.assertClose(student.grad, baseline.grad)
        self.assertIsNone(reference.grad)

    def test_masking_and_unequal_lengths_use_equal_example_reduction(self):
        values = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5) / 5
        student = values.clone().requires_grad_()
        reference = values.flip(-1).clone().requires_grad_()
        targets = torch.tensor([[-100, 0, 1, 2], [4, -100, -100, -100]])
        beta = .3
        loss, metrics = objective_loss(student, targets, objective="ce_kl",
            reference_logits=reference, beta=beta, chunk_tokens=2)
        expected_rows, expected_ce, expected_kl = [], [], []
        for index in range(2):
            mask = targets[index] != -100
            log_p = student[index, mask].float().log_softmax(-1)
            log_q = reference[index, mask].detach().float().log_softmax(-1)
            ce = F.nll_loss(log_p, targets[index, mask], reduction="mean")
            kl = (log_q.exp() * (log_q - log_p)).sum(-1).mean()
            expected_rows.append(ce + beta * kl)
            expected_ce.append(ce)
            expected_kl.append(kl)
        self.assertClose(loss, torch.stack(expected_rows).mean())
        self.assertClose(metrics["ce"], torch.stack(expected_ce).mean().detach())
        self.assertClose(metrics["kl"], torch.stack(expected_kl).mean().detach())
        pooled = (expected_rows[0] * 3 + expected_rows[1]) / 4
        self.assertGreater(abs(loss.item() - pooled.item()), .01)
        self.assertEqual(metrics["active_counts"].tolist(), [3, 1])
        self.assertEqual(metrics["active_tokens"].item(), 4)
        loss.backward()
        self.assertEqual(student.grad[targets == -100].abs().sum().item(), 0.)
        self.assertIsNone(reference.grad)

    def test_ce_kl_does_not_shift_aligned_targets_or_special_case_eos(self):
        logits = torch.tensor([[[5., 0., -2., -3.], [0., 0., -1., 3.]]], requires_grad=True)
        targets = torch.tensor([[0, 3]])  # Token3 can represent supervised EOS.
        loss, metrics = objective_loss(logits, targets, objective="ce_kl",
            reference_logits=torch.zeros_like(logits), beta=0.)
        self.assertClose(loss, F.cross_entropy(logits[0], targets[0]))
        self.assertEqual(metrics["active_tokens"].item(), 2)
        loss.backward()
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0.)

    def test_ce_kl_chunk_checkpoint_loss_metrics_and_gradient_equivalence(self):
        values = torch.arange(70, dtype=torch.float32).reshape(2, 7, 5) / 13
        targets = torch.tensor([[-100, -100, 0, 1, 2, 3, 4], [4, 3, -100, 2, 1, -100, 0]])
        baseline = None
        for chunk, recompute in ((7, False), (1, True), (3, True), (4, False)):
            logits = values.clone().requires_grad_()
            reference = values.flip(-1).clone().requires_grad_()
            loss, metrics = objective_loss(logits, targets, objective="ce_kl",
                reference_logits=reference, beta=.2, chunk_tokens=chunk, checkpoint_chunks=recompute)
            loss.backward()
            if baseline is None:
                baseline = loss.detach(), metrics, logits.grad.detach().clone()
            else:
                self.assertClose(loss, baseline[0])
                self.assertClose(logits.grad, baseline[2])
                for key in metrics:
                    self.assertClose(metrics[key], baseline[1][key])
            self.assertIsNone(reference.grad)

    def test_all_modes_metrics_have_exact_keys_and_no_graphs(self):
        expected_keys = {"loss", "ce", "weighted_ce", "kl", "effective_weight", "active_counts", "active_tokens"}
        for mode in sorted(MODES):
            logits = torch.ones((1, 2, 3), requires_grad=True)
            loss, metrics = objective_loss(logits, torch.tensor([[0, 1]]), objective=mode,
                reference_logits=torch.zeros_like(logits))
            self.assertTrue(loss.requires_grad)
            self.assertEqual(set(metrics), expected_keys)
            self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in metrics.values()))
            self.assertEqual(metrics["active_counts"].dtype, torch.int64)
            self.assertEqual(metrics["active_tokens"].dtype, torch.int64)

    def test_ce_kl_bfloat16_and_float64_inputs_use_fp32_objective(self):
        for dtype in (torch.bfloat16, torch.float64):
            logits = torch.tensor([[[.1, .4, -.2]]], dtype=dtype, requires_grad=True)
            reference = torch.tensor([[[.9, -.4, .3]]], dtype=dtype, requires_grad=True)
            loss, _ = objective_loss(logits, torch.tensor([[1]]), objective="ce_kl", reference_logits=reference)
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertEqual(logits.grad.dtype, dtype)
            self.assertGreater(logits.grad.abs().sum().item(), 0.)
            self.assertIsNone(reference.grad)

    def test_ce_kl_extreme_finite_logits_keep_finite_loss_metrics_and_gradient(self):
        logits = torch.tensor([[[1e20, -1e20, 0.], [-1e20, 1e20, 0.]]], requires_grad=True)
        loss, metrics = objective_loss(logits, torch.tensor([[1, 0]]), objective="ce_kl",
            reference_logits=torch.zeros_like(logits), chunk_tokens=1)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(torch.isfinite(value).all().item() for value in metrics.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_ce_kl_supports_no_grad_and_frozen_student(self):
        with torch.no_grad():
            loss, metrics = objective_loss(torch.zeros((1, 2, 3)), torch.tensor([[0, 1]]),
                objective="ce_kl", reference_logits=torch.ones((1, 2, 3)))
        self.assertFalse(loss.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))


class PreservationInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.zeros((2, 3, 4), requires_grad=True)
        self.targets = torch.tensor([[0, 1, 2], [-100, 2, 3]])

    def loss(self, **kwargs):
        return objective_loss(kwargs.pop("student_logits", self.logits), kwargs.pop("targets", self.targets),
            objective=kwargs.pop("objective", "ce_kl"),
            reference_logits=kwargs.pop("reference_logits", torch.zeros_like(self.logits)), **kwargs)

    def test_ce_kl_requires_reference_even_when_beta_zero(self):
        for beta in (0., .1):
            with self.assertRaisesRegex(ValueError, "reference_logits required"):
                self.loss(reference_logits=None, beta=beta)

    def test_invalid_mode_or_coefficient_rejected(self):
        for mode in ("CE_KL", "", None, 1):
            with self.assertRaisesRegex(ValueError, "unknown objective"):
                self.loss(objective=mode)
        for beta in (-.1, float("inf"), float("nan"), True, "0.05"):
            with self.assertRaisesRegex(ValueError, "beta must be finite"):
                self.loss(beta=beta)

    def test_invalid_chunk_controls_rejected(self):
        for chunk in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.loss(chunk_tokens=chunk)
        with self.assertRaisesRegex(ValueError, "must be bool"):
            self.loss(checkpoint_chunks=1)

    def test_invalid_student_reference_and_target_metadata_rejected(self):
        for logits in (None, torch.zeros((3, 4)), torch.zeros((2, 3, 4), dtype=torch.int64)):
            with self.assertRaisesRegex(ValueError, "floating"):
                self.loss(student_logits=logits)
        for shape in ((0, 3, 4), (2, 0, 4), (2, 3, 0)):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                self.loss(student_logits=torch.zeros(shape))
        for target in (None, self.targets.float(), self.targets[:, :2]):
            with self.assertRaisesRegex(ValueError, "targets must be int64"):
                self.loss(targets=target)
        for reference in ([[1., 2.]], torch.zeros((2, 3, 5)), torch.ones_like(self.logits, dtype=torch.int64)):
            with self.assertRaisesRegex(ValueError, "same-shaped floating"):
                self.loss(reference_logits=reference)

    def test_invalid_ids_and_empty_supervised_examples_rejected(self):
        for bad in (-1, -101, 4):
            target = self.targets.clone()
            target[0, 0] = bad
            with self.assertRaisesRegex(ValueError, "invalid ID"):
                self.loss(targets=target)
        for all_masked in (False, True):
            target = self.targets.clone()
            target[:] = -100 if all_masked else target
            target[0] = -100
            with self.assertRaisesRegex(ValueError, "every example"):
                self.loss(targets=target)

    def test_cross_device_metadata_rejected_without_a_gpu(self):
        with self.assertRaisesRegex(ValueError, "targets.*share a device"):
            self.loss(targets=torch.empty((2, 3), dtype=torch.int64, device="meta"))
        with self.assertRaisesRegex(ValueError, "reference_logits.*share a device"):
            self.loss(reference_logits=torch.empty((2, 3, 4), device="meta"))

    def test_nonfinite_student_or_reference_including_masked_positions_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for name in ("student_logits", "reference_logits"):
                values = self.logits.detach().clone()
                values[1, 0, 0] = invalid  # Position is masked; still fail closed.
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, "must be finite"):
                    self.loss(**{name: values})

    def test_fp32_student_and_reference_conversion_overflow_rejected(self):
        for name, message in (("student_logits", "FP32 student"), ("reference_logits", "FP32 reference")):
            values = torch.zeros((2, 3, 4), dtype=torch.float64)
            values[0, 0, 0] = 1e300
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                self.loss(**{name: values})


if __name__ == "__main__":
    unittest.main()
