"""Independent dense CPU oracles, authored but not executed or admitted.

Only synthetic tensors and tiny modules are used. No Transformers model,
tokenizer, cache, corpus, GPU, scheduler or production training runner is used.
The native-EOS vocabulary fixture is synthetic and contains no model weights.
Run only in a separately admitted native CPU allocation. Passing these tests
does not establish BF16 CUDA parity, long-context memory or pilot feasibility.
"""
from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from extension import checkpointed_head_objectives as module


MODES = ("sft", "ce_kl", "asft", "prior_anchored")
TERMS = ("ce", "weighted_ce", "kl", "effective_weight")
LOSS_ATOL = METRIC_ATOL = 1e-6
GRAD_RTOL, GRAD_ATOL = 1e-5, 1e-6


def head(hidden=4, vocabulary=9, *, offset=0., frozen=False):
    result = nn.Linear(hidden, vocabulary, bias=False)
    with torch.no_grad():
        values = torch.arange(hidden * vocabulary, dtype=torch.float32).reshape(vocabulary, hidden)
        result.weight.copy_(((values.remainder(23) - 11) / 16) + offset)
    if frozen:
        result.eval().requires_grad_(False)
    return result


def fixture():
    hidden = ((torch.arange(56, dtype=torch.float32).reshape(2, 7, 4).remainder(19) - 9) / 16)
    targets = torch.tensor([[-100, 0, 8, 1, 2, 8, 3], [4, -100, 8, -100, 5, -100, -100]], dtype=torch.int64)
    return hidden, targets, head(), hidden.flip(1).clone() * .5 + .125, head(offset=.125, frozen=True)


def dense_oracle(hidden, targets, weight, *, reference_hidden=None, reference_weight=None,
                 objective="sft", beta=.05, eos_token_id=8):
    """Independent dense projection, token formula and masked batch reduction.

    This deliberately does not call any production/prototype loss helper or
    chunk implementation. FP32 F.linear forms the complete synthetic logits.
    Gold weights and the full reference distribution are explicitly detached.
    """
    logits = F.linear(hidden, weight)
    logp = F.log_softmax(logits.float(), dim=-1)
    active = targets != -100
    safe = targets.clamp_min(0)
    gold_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    ce = -gold_logp
    if objective == "sft":
        weights, kl = torch.ones_like(ce), torch.zeros_like(ce)
    else:
        with torch.no_grad():
            logq = F.log_softmax(F.linear(reference_hidden.detach(), reference_weight.detach()).float(), dim=-1)
            q = logq.exp()
            if objective == "asft":
                weights = gold_logp.detach().exp()
            elif objective == "prior_anchored":
                weights = logq.gather(-1, safe.unsqueeze(-1)).squeeze(-1).exp()
            else:
                weights = torch.ones_like(ce)
        kl = torch.sum(q * (logq - logp), dim=-1)
    values = {"ce": ce, "weighted_ce": weights * ce, "kl": kl, "effective_weight": weights}
    counts = active.sum(dim=1)
    eos = active & (targets == eos_token_id)
    other = active & ~eos
    eos_counts, other_counts = eos.sum(dim=1), other.sum(dim=1)
    loss = (((values["weighted_ce"] + beta * kl) * active).sum(dim=1) / counts).mean()
    metrics = {"loss": loss.detach(), "active_counts": counts.detach(), "active_tokens": counts.sum().detach(),
        "eos_counts": eos_counts.detach(), "eos_tokens": eos_counts.sum().detach(),
        "non_eos_counts": other_counts.detach(), "non_eos_tokens": other_counts.sum().detach()}
    for name, value in values.items():
        sums = (value * active).sum(dim=1)
        eos_sums, other_sums = (value * eos).sum(dim=1), (value * other).sum(dim=1)
        metrics[name] = (sums / counts).mean().detach()
        metrics[name + "_sums"] = sums.detach()
        metrics["eos_" + name + "_sums"] = eos_sums.detach()
        metrics["non_eos_" + name + "_sums"] = other_sums.detach()
        for prefix, relevant_sums, relevant_counts in (("eos_", eos_sums, eos_counts), ("non_eos_", other_sums, other_counts)):
            rows = [relevant_sums[index] / count if int(count) else relevant_sums[index] * 0
                for index, count in enumerate(relevant_counts)]
            metrics[prefix + name] = torch.stack(rows).mean().detach()
    return loss, metrics


def projected(hidden, targets, student_head, reference_hidden=None, reference_head=None, **kwargs):
    return module.checkpointed_head_loss(hidden, targets, student_head=student_head,
        student_embedding_weight=student_head.weight, reference_hidden=reference_hidden,
        reference_head=reference_head,
        reference_embedding_weight=None if reference_head is None else reference_head.weight,
        eos_token_id=kwargs.pop("eos_token_id", 8), **kwargs)


class TinyTied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.body = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 9, bias=False)
        self.head.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight.copy_(head().weight)
            self.body.weight.copy_((torch.arange(16, dtype=torch.float32).reshape(4, 4) - 6) / 17)

    def hidden(self, ids):
        return torch.tanh(self.body(self.embedding(ids)))


def frozen_tied_reference():
    """Use a distinct distribution so tied anchored tests exercise nonzero KL."""
    result = TinyTied().eval().requires_grad_(False)
    with torch.no_grad():
        result.body.weight.mul_(-.5)
    return result


class DenseHeadNumericalTests(unittest.TestCase):
    def assertValue(self, actual, expected, atol=LOSS_ATOL):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=0)

    def assertGrad(self, actual, expected):
        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, expected, atol=GRAD_ATOL, rtol=GRAD_RTOL)

    def assertMetrics(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(metric=name):
                self.assertValue(actual[name], expected[name], METRIC_ATOL)

    def test_modes_and_fixed_chunk_envelope_are_explicit(self):
        self.assertEqual(module.MODES, frozenset(MODES))
        self.assertEqual(module.TERM_NAMES, TERMS)
        self.assertEqual(module.MAX_CHUNK_TOKENS, 128)

    def test_all_four_losses_metrics_and_hidden_head_gradients_match_independent_dense(self):
        values, targets, original, ref_values, reference = fixture()
        for mode in MODES:
            for chunk, checkpoint in ((1, False), (2, True), (3, True), (7, False)):
                with self.subTest(mode=mode, chunk=chunk, checkpoint=checkpoint):
                    actual_hidden = values.clone().requires_grad_()
                    expected_hidden = values.clone().requires_grad_()
                    actual_head, expected_head = deepcopy(original), deepcopy(original)
                    actual_ref = ref_values.clone().requires_grad_()
                    expected_ref = ref_values.clone().requires_grad_()
                    actual, metrics = projected(actual_hidden, targets, actual_head, actual_ref, reference,
                        objective=mode, beta=.23, chunk_tokens=chunk, checkpoint_chunks=checkpoint)
                    expected, oracle = dense_oracle(expected_hidden, targets, expected_head.weight,
                        reference_hidden=expected_ref, reference_weight=reference.weight, objective=mode, beta=.23)
                    self.assertValue(actual, expected)
                    self.assertMetrics(metrics, oracle)
                    actual.backward()
                    expected.backward()
                    self.assertGrad(actual_hidden.grad, expected_hidden.grad)
                    self.assertGrad(actual_head.weight.grad, expected_head.weight.grad)
                    self.assertGreater(float(expected_hidden.grad.abs().sum()), 0.)
                    self.assertGreater(float(expected_head.weight.grad.abs().sum()), 0.)
                    self.assertIsNone(actual_ref.grad)
                    self.assertIsNone(expected_ref.grad)
                    self.assertIsNone(reference.weight.grad)

    def test_analytic_four_mode_gradient_proves_weight_detach_and_forward_kl_direction(self):
        p, q = torch.tensor([.7, .2, .1]), torch.tensor([.2, .3, .5])
        beta, label = .4, 1
        for mode in MODES:
            for checkpoint in (False, True):
                with self.subTest(mode=mode, checkpoint=checkpoint):
                    student, reference = head(3, 3), head(3, 3, frozen=True)
                    with torch.no_grad():
                        student.weight.copy_(torch.eye(3)); reference.weight.copy_(torch.eye(3))
                    hidden = p.log().reshape(1, 1, 3).requires_grad_()
                    ref = q.log().reshape(1, 1, 3).requires_grad_()
                    loss, metrics = projected(hidden, torch.tensor([[label]]), student, ref, reference,
                        objective=mode, beta=beta, eos_token_id=2, checkpoint_chunks=checkpoint)
                    weight = p[label] if mode == "asft" else q[label] if mode == "prior_anchored" else 1.
                    base = p.clone(); base[label] -= 1
                    expected_grad = weight * base
                    if mode != "sft": expected_grad = expected_grad + beta * (p - q)
                    loss.backward()
                    self.assertGrad(hidden.grad[0, 0], expected_grad)
                    self.assertGrad(student.weight.grad, expected_grad[:, None] * p.log()[None, :])
                    expected_kl = torch.tensor(0.) if mode == "sft" else (q * (q.log() - p.log())).sum()
                    self.assertValue(metrics["kl"], expected_kl)
                    self.assertIsNone(ref.grad)
                    self.assertIsNone(reference.weight.grad)

    def test_asft_probability_weight_is_detached_rather_than_a_differentiated_factor(self):
        student, reference = head(2, 2), head(2, 2, frozen=True)
        with torch.no_grad():
            student.weight.copy_(torch.eye(2)); reference.weight.copy_(torch.eye(2))
        logits = torch.tensor([[[.9, .1]]]).log().requires_grad_()
        loss, _ = projected(logits, torch.tensor([[1]]), student, torch.zeros_like(logits), reference,
            objective="asft", beta=0., eos_token_id=1, chunk_tokens=1)
        wrong = logits.detach().clone().requires_grad_()
        logp = wrong.log_softmax(-1)[0, 0, 1]
        wrong_loss = -logp.exp() * logp
        loss.backward(); wrong_loss.backward()
        expected = .1 * torch.tensor([.9, -.9])
        self.assertGrad(logits.grad[0, 0], expected)
        self.assertGreater(float((logits.grad - wrong.grad).abs().max()), .05)

    def test_kl_uses_the_entire_reference_distribution_beyond_gold_probability(self):
        student, reference = head(3, 3), head(3, 3, frozen=True)
        with torch.no_grad():
            student.weight.copy_(torch.eye(3)); reference.weight.copy_(torch.eye(3))
        p = torch.tensor([[[.6, .2, .2]]]).log()
        refs = (torch.tensor([[[.4, .5, .1]]]).log(), torch.tensor([[[.4, .3, .3]]]).log())
        results = [projected(p, torch.tensor([[0]]), student, ref, reference, objective="prior_anchored",
            beta=1., eos_token_id=2, checkpoint_chunks=False) for ref in refs]
        self.assertValue(results[0][1]["weighted_ce"], results[1][1]["weighted_ce"])
        self.assertGreater(abs(float(results[0][1]["kl"] - results[1][1]["kl"])), .05)
        self.assertGreater(abs(float(results[0][0] - results[1][0])), .05)

    def test_equal_example_normalization_differs_from_pooled_active_token_mean(self):
        student = head(2, 2)
        with torch.no_grad(): student.weight.copy_(torch.eye(2))
        values = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]], [[4., -4.], [0., 0.], [0., 0.]]])
        targets = torch.tensor([[0, 1, 0], [1, -100, -100]])
        loss, metrics = projected(values, targets, student, objective="sft", eos_token_id=1, chunk_tokens=2)
        row_ce = torch.stack([F.cross_entropy(values[0], targets[0]), F.cross_entropy(values[1, :1], targets[1, :1])])
        self.assertValue(loss, row_ce.mean())
        self.assertGreater(abs(float(loss - (row_ce * torch.tensor([3., 1.])).sum() / 4)), 1.)
        self.assertEqual(metrics["active_counts"].tolist(), [3, 1])

    def test_weight_sum_is_never_the_denominator(self):
        student, reference = head(2, 2), head(2, 2, frozen=True)
        with torch.no_grad(): student.weight.copy_(torch.eye(2)); reference.weight.copy_(torch.eye(2))
        values = torch.tensor([[[.9, .1], [.2, .8]]]).log()
        loss, _ = projected(values, torch.tensor([[0, 0]]), student, torch.zeros_like(values), reference,
            objective="asft", beta=0., eos_token_id=1, checkpoint_chunks=False)
        weights = torch.tensor([.9, .2]); ce = -weights.log()
        self.assertValue(loss, (weights * ce).sum() / 2)
        self.assertGreater(abs(float(loss - (weights * ce).sum() / weights.sum())), .1)

    def test_already_aligned_inputs_are_not_shifted_or_shortened(self):
        student = head(3, 3)
        with torch.no_grad(): student.weight.copy_(torch.eye(3))
        values = torch.eye(3).mul(5).unsqueeze(0).requires_grad_()
        targets = torch.tensor([[0, 1, 2]])
        loss, metrics = projected(values, targets, student, objective="sft", eos_token_id=2, chunk_tokens=1)
        self.assertValue(loss, F.cross_entropy(values[0], targets[0]))
        self.assertEqual(metrics["active_tokens"].item(), 3)
        loss.backward()
        self.assertGreater(float(values.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(values.grad[0, -1].abs().sum()), 0.)

    def test_zero_beta_ce_kl_equals_plain_ce_loss_and_all_student_gradients(self):
        values, targets, original, ref_values, reference = fixture()
        first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
        first_head, second_head = deepcopy(original), deepcopy(original)
        anchored, _ = projected(first, targets, first_head, ref_values, reference, objective="ce_kl", beta=0., chunk_tokens=2)
        ce, _ = projected(second, targets, second_head, objective="sft", beta=99., chunk_tokens=2)
        self.assertValue(anchored, ce)
        anchored.backward(); ce.backward()
        self.assertGrad(first.grad, second.grad); self.assertGrad(first_head.weight.grad, second_head.weight.grad)

    def test_equal_reference_has_zero_kl_and_ce_kl_plain_ce_gradient(self):
        values, targets, student, _, _ = fixture()
        reference = deepcopy(student).eval().requires_grad_(False)
        hidden = values.clone().requires_grad_()
        loss, metrics = projected(hidden, targets, student, values.clone(), reference, objective="ce_kl", beta=.8, chunk_tokens=2)
        expected_hidden, expected_head = values.clone().requires_grad_(), deepcopy(student)
        expected, _ = dense_oracle(expected_hidden, targets, expected_head.weight, objective="sft")
        self.assertValue(metrics["kl"], torch.tensor(0.))
        self.assertValue(loss, expected)
        loss.backward(); expected.backward()
        self.assertGrad(hidden.grad, expected_hidden.grad); self.assertGrad(student.weight.grad, expected_head.weight.grad)

    def test_actual_native_eos_and_post_eos_tail_both_contribute_in_a_synthetic_vocabulary(self):
        student = head(2, 151936)
        with torch.no_grad(): student.weight.zero_()
        values = torch.tensor([[[.1, .3], [.3, -.2], [-.5, .4], [.2, .6]]], requires_grad=True)
        targets = torch.tensor([[-100, -100, 151645, 198]])  # Native EOS followed by native terminal tail.
        loss, metrics = projected(values, targets, student, eos_token_id=151645, chunk_tokens=2)
        expected, oracle = dense_oracle(values, targets, student.weight, eos_token_id=151645)
        self.assertValue(loss, expected); self.assertMetrics(metrics, oracle)
        self.assertEqual((metrics["active_tokens"].item(), metrics["eos_tokens"].item(), metrics["non_eos_tokens"].item()), (2, 1, 1))
        loss.backward()
        self.assertGreater(float(student.weight.grad[151645].abs().sum()), 0.)
        self.assertGreater(float(student.weight.grad[198].abs().sum()), 0.)

    def test_eos_and_non_eos_diagnostics_have_equal_example_means_with_zero_groups(self):
        values = torch.tensor([[[.2, -.3], [.8, .1]], [[-.2, .4], [.1, -.6]]])
        targets = torch.tensor([[1, 1], [0, 0]])
        student = head(2, 2)
        loss, metrics = projected(values, targets, student, eos_token_id=1, chunk_tokens=1)
        expected, oracle = dense_oracle(values, targets, student.weight, eos_token_id=1)
        self.assertValue(loss, expected); self.assertMetrics(metrics, oracle)
        self.assertEqual(metrics["eos_counts"].tolist(), [2, 0]); self.assertEqual(metrics["non_eos_counts"].tolist(), [0, 2])

    def test_metrics_are_exactly_bounded_detached_scalars_or_per_example_vectors(self):
        values, targets, student, ref_values, reference = fixture()
        loss, metrics = projected(values.requires_grad_(), targets, student, ref_values, reference, objective="asft", chunk_tokens=2)
        _, expected = dense_oracle(values, targets, student.weight, reference_hidden=ref_values,
            reference_weight=reference.weight, objective="asft")
        self.assertEqual(set(metrics), set(expected)); self.assertEqual(loss.ndim, 0)
        for name, value in metrics.items():
            self.assertIsInstance(value, torch.Tensor)
            self.assertLessEqual(value.numel(), 2); self.assertLessEqual(value.ndim, 1)
            self.assertFalse(value.requires_grad); self.assertIsNone(value.grad_fn)
            self.assertTrue(bool(torch.isfinite(value).all()))
        for name in ("active_counts", "active_tokens", "eos_counts", "eos_tokens", "non_eos_counts", "non_eos_tokens"):
            self.assertEqual(metrics[name].dtype, torch.int64)

    def test_checkpoint_keeps_trainable_head_gradient_when_hidden_has_no_gradient(self):
        values, targets, original, ref_values, reference = fixture()
        for mode in MODES:
            actual_head, expected_head = deepcopy(original), deepcopy(original)
            actual, _ = projected(values, targets, actual_head, ref_values, reference, objective=mode, chunk_tokens=2)
            expected, _ = dense_oracle(values, targets, expected_head.weight, reference_hidden=ref_values,
                reference_weight=reference.weight, objective=mode)
            actual.backward(); expected.backward()
            self.assertGrad(actual_head.weight.grad, expected_head.weight.grad)

    def test_delayed_backward_across_many_distinct_chunks_does_not_capture_the_last_iteration(self):
        values = (torch.arange(232, dtype=torch.float32).reshape(2, 29, 4).remainder(31) - 15) / 11
        targets = torch.full((2, 29), -100, dtype=torch.int64)
        targets[0, [0, 5, 14, 28]] = torch.tensor([0, 3, 8, 1])
        targets[1, [2, 9, 18]] = torch.tensor([4, 2, 8])
        original, reference = head(), head(offset=.17, frozen=True)
        for mode in MODES:
            student, expected_head = deepcopy(original), deepcopy(original)
            actual_hidden, expected_hidden = values.clone().requires_grad_(), values.clone().requires_grad_()
            ref = values.flip(1).clone().requires_grad_()
            actual, metrics = projected(actual_hidden, targets, student, ref, reference,
                objective=mode, beta=.31, chunk_tokens=2, checkpoint_chunks=True)
            expected, oracle = dense_oracle(expected_hidden, targets, expected_head.weight,
                reference_hidden=ref, reference_weight=reference.weight, objective=mode, beta=.31)
            self.assertValue(actual, expected); self.assertMetrics(metrics, oracle)
            actual.backward(); expected.backward()
            self.assertGrad(actual_hidden.grad, expected_hidden.grad); self.assertGrad(student.weight.grad, expected_head.weight.grad)
            self.assertIsNone(ref.grad)

    def test_reference_hidden_is_detached_and_frozen_reference_head_never_changes(self):
        values, targets, student, ref_values, reference = fixture()
        before = reference.weight.detach().clone()
        ref = ref_values.requires_grad_()
        loss, _ = projected(values.requires_grad_(), targets, student, ref, reference, objective="prior_anchored", chunk_tokens=1)
        loss.backward()
        self.assertIsNone(ref.grad); self.assertIsNone(reference.weight.grad)
        self.assertFalse(reference.training); self.assertFalse(reference.weight.requires_grad)
        self.assertValue(reference.weight, before, 0)


class TiedHeadGradientAndUpdateTests(unittest.TestCase):
    assertValue = DenseHeadNumericalTests.assertValue
    assertGrad = DenseHeadNumericalTests.assertGrad
    def test_tied_embedding_plus_projection_gradient_is_the_sum_of_both_independent_paths(self):
        ids = torch.tensor([[0, 1, 2, 1, 4, 5, 6], [7, 1, 8, 0, 1, 3, 2]])
        _, targets, _, _, _ = fixture()
        for mode in MODES:
            for checkpoint in (False, True):
                model, reference = TinyTied(), frozen_tied_reference()
                self.assertIs(model.head.weight, model.embedding.weight)
                ref_hidden = reference.hidden(ids).detach().requires_grad_()
                embedding = model.embedding.weight.detach().clone().requires_grad_()
                projection = model.head.weight.detach().clone().requires_grad_()
                body = model.body.weight.detach().clone().requires_grad_()
                expected_hidden = torch.tanh(F.linear(F.embedding(ids, embedding), body))
                expected, oracle = dense_oracle(expected_hidden, targets, projection, reference_hidden=ref_hidden,
                    reference_weight=reference.head.weight, objective=mode, beta=.17)
                if mode != "sft": self.assertGreater(float(oracle["kl"]), 0.)
                actual, _ = projected(model.hidden(ids), targets, model.head, ref_hidden, reference.head,
                    objective=mode, beta=.17, chunk_tokens=2, checkpoint_chunks=checkpoint)
                actual.backward(); expected.backward()
                self.assertGreater(float(embedding.grad.abs().sum()), 0.); self.assertGreater(float(projection.grad.abs().sum()), 0.)
                self.assertGrad(model.embedding.weight.grad, embedding.grad + projection.grad)
                self.assertGrad(model.body.weight.grad, body.grad)
                self.assertEqual(len(list(model.parameters())), 2)
                self.assertIsNone(ref_hidden.grad); self.assertIsNone(reference.head.weight.grad)

    def test_fp32_adam_update_and_both_moments_match_dense_for_every_unique_student_parameter(self):
        ids = torch.tensor([[0, 1, 2, 1, 4, 5, 6], [7, 1, 8, 0, 1, 3, 2]])
        _, targets, _, _, _ = fixture()
        for mode in MODES:
            for checkpoint in (False, True):
                actual, expected = TinyTied(), TinyTied()
                reference = frozen_tied_reference()
                ref_before = {name: value.detach().clone() for name, value in reference.named_parameters()}
                optimizers = [torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=.01) for model in (actual, expected)]
                ref_hidden = reference.hidden(ids).detach()
                first, _ = projected(actual.hidden(ids), targets, actual.head, ref_hidden, reference.head,
                    objective=mode, beta=.13, chunk_tokens=3, checkpoint_chunks=checkpoint)
                second, oracle = dense_oracle(expected.hidden(ids), targets, expected.head.weight,
                    reference_hidden=ref_hidden, reference_weight=reference.head.weight, objective=mode, beta=.13)
                if mode != "sft": self.assertGreater(float(oracle["kl"]), 0.)
                self.assertValue(first, second)
                first.backward(); second.backward()
                params = [dict(model.named_parameters()) for model in (actual, expected)]
                self.assertEqual(set(params[0]), {"embedding.weight", "body.weight"})
                before = {name: value.detach().clone() for name, value in params[0].items()}
                for name in params[0]: self.assertGrad(params[0][name].grad, params[1][name].grad)
                optimizers[0].step(); optimizers[1].step()
                self.assertIs(actual.head.weight, actual.embedding.weight)
                for name in params[0]:
                    left, right = params[0][name], params[1][name]
                    self.assertEqual(left.dtype, torch.float32); self.assertGrad(left, right)
                    self.assertGreater(float((left.detach() - before[name]).abs().sum()), 0.)
                    for key in ("exp_avg", "exp_avg_sq"):
                        first_state, second_state = optimizers[0].state[left][key], optimizers[1].state[right][key]
                        self.assertEqual(first_state.dtype, torch.float32); self.assertTrue(bool(torch.isfinite(first_state).all()))
                        self.assertGrad(first_state, second_state)
                for name, value in reference.named_parameters():
                    self.assertValue(value, ref_before[name], 0); self.assertIsNone(value.grad)


class ProjectionBoundaryTests(unittest.TestCase):
    def test_all_original_head_calls_are_bounded_active_rows_and_metrics_add_no_projection(self):
        values, targets, original, ref_values, frozen = fixture()
        for mode in MODES:
            for checkpoint in (False, True):
                student, reference = deepcopy(original), deepcopy(frozen)
                seen, ref_seen = [], []
                def capture(destination):
                    return lambda layer, inputs, outputs: destination.append((tuple(inputs[0].shape), tuple(outputs.shape)))
                handles = [student.register_forward_hook(capture(seen)), reference.register_forward_hook(capture(ref_seen))]
                try:
                    loss, metrics = projected(values.clone().requires_grad_(), targets, student, ref_values, reference,
                        objective=mode, chunk_tokens=2, checkpoint_chunks=checkpoint)
                    expected_calls = sum(bool((targets[b, t:t + 2] != -100).any()) for b in range(2) for t in range(0, 7, 2))
                    self.assertEqual(len(seen), expected_calls)
                    self.assertEqual(sum(shape[0][0] for shape in seen), int((targets != -100).sum()))
                    self.assertEqual(len(ref_seen), 0 if mode == "sft" else expected_calls)
                    self.assertEqual(loss.ndim, 0)
                    self.assertTrue(all(value.numel() <= 2 for value in metrics.values()))
                    loss.backward()
                    for input_shape, output_shape in seen + ref_seen:
                        self.assertEqual(len(input_shape), 2); self.assertEqual(input_shape[1], 4)
                        self.assertTrue(1 <= input_shape[0] <= 2)
                        self.assertEqual(output_shape, (input_shape[0], 9))
                finally:
                    for handle in handles: handle.remove()

    def test_sft_accepts_complete_reference_but_does_not_project_it(self):
        values, targets, student, ref_values, reference = fixture()
        calls = []
        handle = reference.register_forward_hook(lambda *args: calls.append(True))
        try:
            loss, _ = projected(values, targets, student, ref_values, reference, objective="sft", chunk_tokens=1)
            loss.backward()
            self.assertEqual(calls, [])
        finally: handle.remove()

    def test_fully_masked_windows_are_not_projected_but_valid_tail_windows_are(self):
        values = torch.ones((1, 7, 4), requires_grad=True)
        targets = torch.tensor([[-100, -100, -100, -100, 8, 1, 2]])
        student, calls = head(), []
        handle = student.register_forward_hook(lambda layer, inputs, output: calls.append(inputs[0].shape[0]))
        try:
            _, metrics = projected(values, targets, student, chunk_tokens=2, checkpoint_chunks=False)
            self.assertEqual(calls, [2, 1]); self.assertEqual(metrics["active_tokens"].item(), 3)
        finally: handle.remove()

    def test_noncontiguous_normal_hidden_views_match_dense(self):
        storage = torch.arange(112, dtype=torch.float32).reshape(2, 14, 4) / 128
        values = storage[:, ::2]
        self.assertFalse(values.is_contiguous())
        _, targets, student, ref_values, reference = fixture()
        actual, metrics = projected(values, targets, student, ref_values, reference, objective="ce_kl", chunk_tokens=3)
        expected, oracle = dense_oracle(values, targets, student.weight,
            reference_hidden=ref_values, reference_weight=reference.weight, objective="ce_kl")
        torch.testing.assert_close(actual, expected, rtol=0, atol=LOSS_ATOL)
        for name in metrics: torch.testing.assert_close(metrics[name], oracle[name], rtol=0, atol=METRIC_ATOL)


class CheckpointedHeadGuardTests(unittest.TestCase):
    def setUp(self):
        self.hidden, self.targets, self.student, self.ref_hidden, self.reference = fixture()

    def call(self, **kwargs):
        return module.checkpointed_head_loss(kwargs.pop("student_hidden", self.hidden), kwargs.pop("targets", self.targets),
            student_head=kwargs.pop("student_head", self.student),
            student_embedding_weight=kwargs.pop("student_embedding_weight", self.student.weight),
            reference_hidden=kwargs.pop("reference_hidden", self.ref_hidden),
            reference_head=kwargs.pop("reference_head", self.reference),
            reference_embedding_weight=kwargs.pop("reference_embedding_weight", self.reference.weight),
            eos_token_id=kwargs.pop("eos_token_id", 8), objective=kwargs.pop("objective", "ce_kl"), **kwargs)

    def test_invalid_mode_beta_or_checkpoint_control_fails_closed(self):
        for mode in (None, "CE_KL", "", 0):
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.call(objective=mode)
        for beta in (-1., float("nan"), float("inf"), True, "0.05"):
            with self.subTest(beta=beta), self.assertRaises(ValueError): self.call(beta=beta)
        for checkpoint in (0, 1, None, "True"):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError): self.call(checkpoint_chunks=checkpoint)

    def test_chunk_size_and_eos_id_are_strictly_bounded_plain_integers(self):
        for chunk in (0, -1, 129, True, 1.5, "128"):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError): self.call(chunk_tokens=chunk)
        for eos in (-1, 9, True, .5, None):
            with self.subTest(eos=eos), self.assertRaises(ValueError): self.call(eos_token_id=eos)

    def test_empty_malformed_or_unsupported_hidden_metadata_is_rejected(self):
        for value in (None, torch.ones(2, 4), self.hidden.double(), self.hidden.half(), self.hidden.long(),
                      torch.empty(0, 7, 4), torch.empty(2, 0, 4), torch.empty(2, 7, 0)):
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError): self.call(student_hidden=value)

    def test_target_shape_dtype_and_invalid_ids_reject_every_ignore_code_except_minus100(self):
        for target in (None, self.targets.float(), self.targets[:, :6], torch.ones((2, 7, 1), dtype=torch.int64)):
            with self.subTest(target_type=type(target)), self.assertRaises(ValueError): self.call(targets=target)
        for token in (-1, -101, 9):
            target = self.targets.clone(); target[0, 0] = token
            with self.subTest(token=token), self.assertRaises(ValueError): self.call(targets=target)

    def test_every_example_must_have_an_active_target_even_if_other_rows_have_many(self):
        for all_rows in (False, True):
            target = self.targets.clone(); target[0] = -100
            if all_rows: target[:] = -100
            with self.subTest(all_rows=all_rows), self.assertRaises(ValueError): self.call(targets=target)

    def test_head_must_be_original_bias_free_fp32_trainable_linear_with_exact_tied_parameter(self):
        class WrappedLinear(nn.Linear): pass
        for value in (nn.Linear(4, 9, bias=True), WrappedLinear(4, 9, bias=False), nn.Sequential(head()), head(hidden=5)):
            weight = value.weight if isinstance(value, nn.Linear) else self.student.weight
            with self.subTest(head_type=type(value)), self.assertRaises(ValueError):
                self.call(student_head=value, student_embedding_weight=weight)
        for weight in (None, self.student.weight.detach(), nn.Parameter(self.student.weight.detach().clone())):
            with self.subTest(weight_type=type(weight)), self.assertRaises(ValueError): self.call(student_embedding_weight=weight)
        for dtype in (torch.bfloat16, torch.float64):
            value = deepcopy(self.student).to(dtype)
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                self.call(student_head=value, student_embedding_weight=value.weight)
        value = deepcopy(self.student).requires_grad_(False)
        with self.assertRaises(ValueError): self.call(student_head=value, student_embedding_weight=value.weight)

    def test_anchored_modes_require_complete_eval_frozen_disjoint_reference_tuple(self):
        for mode in ("ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.call(objective=mode, reference_hidden=None, reference_head=None, reference_embedding_weight=None)
        for missing in ("reference_hidden", "reference_head", "reference_embedding_weight"):
            with self.subTest(missing=missing), self.assertRaises(ValueError): self.call(**{missing: None})
        for reference in (deepcopy(self.reference).train(), deepcopy(self.reference).requires_grad_(True)):
            with self.assertRaises(ValueError): self.call(reference_head=reference, reference_embedding_weight=reference.weight)
        reference = head(frozen=True)
        reference.weight = nn.Parameter(self.student.weight.detach(), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.call(reference_head=reference, reference_embedding_weight=reference.weight)

    def test_partial_optional_sft_reference_is_invalid(self):
        with self.assertRaises(ValueError): self.call(objective="sft", reference_head=None)

    def test_nonfinite_hidden_at_masked_positions_is_rejected_before_any_projection(self):
        for name in ("student_hidden", "reference_hidden"):
            for invalid in (float("nan"), float("inf"), -float("inf")):
                values = self.hidden.clone() if name == "student_hidden" else self.ref_hidden.clone()
                values[0, 0, 0] = invalid  # targets[0,0] == -100.
                calls = []
                handle = self.student.register_forward_hook(lambda *args: calls.append(True))
                try:
                    with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError): self.call(**{name: values})
                    self.assertEqual(calls, [])
                finally: handle.remove()

    def test_nonfinite_head_weights_and_finite_input_projection_overflow_fail_closed(self):
        for which in ("student", "reference"):
            candidate = deepcopy(self.student if which == "student" else self.reference)
            with torch.no_grad(): candidate.weight[0, 0] = float("nan")
            kwargs = {which + "_head": candidate, ("student_embedding_weight" if which == "student" else "reference_embedding_weight"): candidate.weight}
            with self.subTest(which=which), self.assertRaises(ValueError): self.call(**kwargs)
        candidate = deepcopy(self.student)
        with torch.no_grad(): candidate.weight.fill_(1e30)
        with self.assertRaises(ValueError):
            self.call(student_hidden=torch.full_like(self.hidden, 1e30), student_head=candidate, student_embedding_weight=candidate.weight)

    def test_wrong_reference_shape_vocabulary_and_cross_device_metadata_are_rejected(self):
        for value in (self.ref_hidden[:, :6], self.ref_hidden.double(), torch.empty(self.ref_hidden.shape, device="meta")):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError): self.call(reference_hidden=value)
        reference = head(vocabulary=10, frozen=True)
        with self.assertRaises(ValueError): self.call(reference_head=reference, reference_embedding_weight=reference.weight)
        with self.assertRaises(ValueError): self.call(targets=torch.empty(self.targets.shape, dtype=torch.int64, device="meta"))

    def test_inference_hidden_tensors_are_rejected_instead_of_silently_cloned(self):
        with torch.inference_mode():
            inference = torch.ones(self.hidden.shape)
        with self.assertRaises(ValueError): self.call(student_hidden=inference)
        with self.assertRaises(ValueError): self.call(reference_hidden=inference)

    def test_bf16_hidden_without_explicit_caller_autocast_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "autocast"):
            self.call(student_hidden=self.hidden.bfloat16(), reference_hidden=self.ref_hidden.bfloat16())

    def test_no_grad_scoring_and_zero_eos_groups_return_finite_detached_metrics(self):
        targets = self.targets.clone(); targets[targets == 8] = 7
        with torch.no_grad():
            loss, metrics = self.call(targets=targets)
        self.assertFalse(loss.requires_grad); self.assertEqual(metrics["eos_tokens"].item(), 0)
        for name in TERMS:
            self.assertEqual(metrics["eos_" + name].item(), 0.)
            self.assertTrue(bool(torch.isfinite(metrics[name]).all()))
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
