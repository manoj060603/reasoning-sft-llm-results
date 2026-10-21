"""Native CPU-only integration fixtures; no downloads, original models or data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import TrainingArguments

from extension.preserving_trainer import (check_optimizer, finite_update_guard,
    make_trainer_class, response_batch, scalar_metrics)


EOS, SUFFIX = 6, (6, 7)


class TinyCausal(torch.nn.Module):
    """Prefix-only dependence makes future tokens unable to affect past logits."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)
        self.calls, self.last_grad_enabled, self.last_inference = 0, None, None

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, use_cache=False):
        if labels is not None or use_cache:
            raise ValueError("Hook must omit labels and disable cache")
        self.calls += 1
        hidden = self.embedding(input_ids).cumsum(dim=1).tanh()
        logits = self.head(hidden)
        self.last_grad_enabled, self.last_inference = torch.is_grad_enabled(), torch.is_inference(logits)
        return {"logits": logits}


def rows():
    return [
        {"input_ids": [1, 2, 6, 7], "labels": [-100, -100, 6, 7]},
        {"input_ids": [1, 3, 4, 5, 6, 7], "labels": [-100, -100, 4, 5, 6, 7]},
        {"input_ids": [2, 4, 1, 6, 7], "labels": [-100, -100, 1, 6, 7]}]


def collate(batch):
    assert len(batch) == 1
    row = batch[0]
    result = {name: torch.tensor([value], dtype=torch.int64) for name, value in row.items()}
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    return result


def args(path, **changes):
    values = dict(output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1,
        num_train_epochs=1, max_grad_norm=0., save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, remove_unused_columns=False, dataloader_pin_memory=False,
        label_smoothing_factor=0.)
    values.update(changes)
    return TrainingArguments(**values)


def manual_loss(model, batch, mode="sft", reference=None, beta=.2):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1]
    targets = batch["labels"][:, 1:]
    mask = targets != -100
    logp = logits.float().log_softmax(-1)
    gold = logp.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
    ce = -gold
    weight, kl = torch.ones_like(ce), torch.zeros_like(ce)
    if mode != "sft":
        with torch.no_grad():
            logq = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1].float().log_softmax(-1)
            q = logq.exp()
        if mode == "asft":
            weight = gold.detach().exp()
        elif mode == "prior_anchored":
            weight = logq.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1).exp()
        kl = (q * (logq - logp)).sum(-1)
    return ((weight * ce + beta * kl) * mask).sum() / mask.sum()


Base = make_trainer_class()


class OrderedTrainer(Base):
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)


class HookIntegrationTests(unittest.TestCase):
    def test_all_four_hooks_match_manual_parameter_gradients_after_one_shift(self):
        torch.manual_seed(21)
        initial = TinyCausal()
        batch = collate([rows()[1]])
        batch["row_index"] = torch.tensor([17])
        original_labels = batch["labels"].clone()
        for mode in ("sft", "ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                student, expected = deepcopy(initial), deepcopy(initial)
                reference = None if mode == "sft" else deepcopy(initial)
                trainer = OrderedTrainer(model=student, args=args(temporary), objective=mode,
                    reference_model=reference, eos_token_id=EOS, terminal_suffix=SUFFIX,
                    beta=.2, chunk_tokens=2, checkpoint_chunks=True)
                actual = trainer.compute_loss(student, batch, num_items_in_batch=torch.tensor(999))
                oracle = manual_loss(expected, batch, mode, reference, .2)
                actual.backward()
                oracle.backward()
                torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-6)
                for (_, first), (_, second) in zip(student.named_parameters(), expected.named_parameters()):
                    torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-6)
                self.assertTrue(torch.equal(batch["labels"], original_labels))
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                self.assertEqual(trainer.last_metrics["supervised_tokens"], 4)
                self.assertEqual(trainer.last_metrics["eos_tokens"], 1)

    def test_returned_metrics_and_inspection_outputs_retain_no_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyCausal()
            trainer = OrderedTrainer(model=model, args=args(temporary), objective="sft",
                eos_token_id=EOS, terminal_suffix=SUFFIX)
            loss, outputs = trainer.compute_loss(model, collate([rows()[0]]), return_outputs=True)
            self.assertTrue(loss.requires_grad)
            self.assertTrue(all(type(value) in (int, float) for value in trainer.last_metrics.values()))
            self.assertEqual(set(outputs), {"loss", "metrics"})
            self.assertFalse(outputs["loss"].requires_grad)
            self.assertTrue(all(type(value) in (int, float) for value in outputs["metrics"].values()))
            self.assertNotIn("logits", outputs)

    def test_eos_and_non_eos_diagnostics_match_independent_token_means(self):
        torch.manual_seed(10)
        model, reference = TinyCausal(), TinyCausal()
        batch = collate([rows()[1]])
        for mode in ("sft", "ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                student, frozen = deepcopy(model), deepcopy(reference)
                trainer = OrderedTrainer(model=student, args=args(temporary), objective=mode,
                    reference_model=None if mode == "sft" else frozen, eos_token_id=EOS, terminal_suffix=SUFFIX,
                    chunk_tokens=1)
                trainer.compute_loss(student, batch)
                with torch.no_grad():
                    logp = student(**{key: value for key, value in batch.items() if key != "labels"})["logits"][:, :-1].log_softmax(-1)
                    targets = batch["labels"][:, 1:]
                    mask, eos = targets != -100, targets == EOS
                    ce = -logp.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
                    weights, kl = torch.ones_like(ce), torch.zeros_like(ce)
                    if mode != "sft":
                        logq = frozen(**{key: value for key, value in batch.items() if key != "labels"})["logits"][:, :-1].log_softmax(-1)
                        if mode == "asft":
                            weights = logp.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1).exp()
                        elif mode == "prior_anchored":
                            weights = logq.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1).exp()
                        kl = (logq.exp() * (logq - logp)).sum(-1)
                    for name, values in (("ce", ce), ("weighted_ce", weights * ce), ("kl", kl), ("effective_weight", weights)):
                        for prefix, selected in (("eos_", eos), ("non_eos_", mask & ~eos)):
                            self.assertAlmostEqual(trainer.last_metrics[prefix + name], values[selected].mean().item(), places=5)

    def test_native_accumulation_unequal_lengths_and_partial_tail_matches_two_updates(self):
        torch.manual_seed(11)
        initial = TinyCausal()
        actual, expected = deepcopy(initial), deepcopy(initial)
        reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.05)
        data = rows()
        for group in (data[:2], data[2:]):
            reference_optimizer.zero_grad()
            for row in group:
                (manual_loss(expected, collate([row])) / len(group)).backward()
            reference_optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = torch.optim.SGD(actual.parameters(), lr=.05)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            trainer = OrderedTrainer(model=actual, args=args(temporary, gradient_accumulation_steps=2, max_steps=2),
                train_dataset=data, data_collator=collate, optimizers=(optimizer, scheduler),
                objective="sft", eos_token_id=EOS, terminal_suffix=SUFFIX)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(trainer.training_forward_examples, 3)
            for first, second in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_reference_zero_step_kl_isolation_and_saved_student_keys(self):
        torch.manual_seed(5)
        student = TinyCausal()
        reference = deepcopy(student)
        before = {name: value.clone() for name, value in reference.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            trainer = OrderedTrainer(model=student, args=args(temporary), objective="ce_kl",
                reference_model=reference, eos_token_id=EOS, terminal_suffix=SUFFIX,
                train_dataset=rows()[:1], data_collator=collate, beta=.2)
            trainer.compute_loss(student, collate([rows()[0]]))
            self.assertEqual(trainer.last_metrics["kl"], 0.)
            self.assertFalse(reference.last_grad_enabled)
            self.assertFalse(reference.last_inference)
            trainer.train()
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in reference.parameters()))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertEqual(set(student.state_dict()), set(before))
            saved = Path(temporary) / "saved-student"
            trainer.save_model(str(saved))
            from safetensors import safe_open
            with safe_open(str(saved / "model.safetensors"), framework="pt", device="cpu") as handle:
                self.assertEqual(set(handle.keys()), set(student.state_dict()))

    def test_nonfinite_backward_aborts_before_native_optimizer_step(self):
        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=.1)
                self.step_calls = 0
            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)
        student = TinyCausal()
        before = {name: value.clone() for name, value in student.state_dict().items()}
        hook = next(student.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = CountingSGD(student.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            trainer = OrderedTrainer(model=student, args=args(temporary), objective="sft",
                eos_token_id=EOS, terminal_suffix=SUFFIX, train_dataset=rows()[:1],
                data_collator=collate, optimizers=(optimizer, scheduler))
            with self.assertRaises((ValueError, RuntimeError)):
                trainer.train()
            self.assertEqual(optimizer.step_calls, 0)
        hook.remove()
        for name, value in student.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)


class BatchGuardTests(unittest.TestCase):
    def test_native_eos_may_precede_template_tail_and_padding(self):
        batch = collate([rows()[0]])
        batch["input_ids"] = F.pad(batch["input_ids"], (0, 2), value=0)
        batch["labels"] = F.pad(batch["labels"], (0, 2), value=-100)
        batch["attention_mask"] = F.pad(batch["attention_mask"], (0, 2), value=0)
        forward, labels, metrics = response_batch(batch, EOS, SUFFIX)
        self.assertEqual(metrics["supervised_tokens"], 2)
        self.assertEqual(metrics["eos_tokens"], 1)
        self.assertNotIn("labels", forward)
        self.assertEqual(labels[0, -1].item(), -100)

    def test_prompt_holes_target_rewrites_or_missing_terminal_suffix_rejected(self):
        for labels in ([-100, 2, -100, 7], [-100, -100, 5, 7], [-100, -100, -100, -100]):
            batch = collate([rows()[0]])
            batch["labels"] = torch.tensor([labels])
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                response_batch(batch, EOS, SUFFIX)
        batch = collate([rows()[0]])
        batch["input_ids"][0, -1] = batch["labels"][0, -1] = 5
        with self.assertRaisesRegex(ValueError, "terminal suffix"):
            response_batch(batch, EOS, SUFFIX)

    def test_microbatch_two_and_left_padding_are_explicitly_unsupported(self):
        batch = collate([rows()[0]])
        with self.assertRaises(ValueError):
            response_batch({key: value.repeat(2, 1) for key, value in batch.items()}, EOS, SUFFIX)
        batch["attention_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "right padding"):
            response_batch(batch, EOS, SUFFIX)

    def test_scalar_metrics_reject_graphs_nonfinite_and_vectors(self):
        for metrics in ({"x": torch.tensor(1., requires_grad=True)}, {"x": torch.tensor(float("nan"))},
                        {"x": torch.ones(2)}):
            with self.assertRaises(ValueError):
                scalar_metrics(metrics)


class PrecisionAndConfigurationTests(unittest.TestCase):
    def test_fp32_adam_moments_and_nonfinite_state_guard(self):
        model = TinyCausal()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        manual_loss(model, collate([rows()[0]])).backward()
        finite_update_guard(model, optimizer)
        optimizer.step()
        check_optimizer(model, optimizer)
        state = next(iter(optimizer.state.values()))
        state["exp_avg"] = state["exp_avg"].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            finite_update_guard(model, optimizer)
        state["exp_avg"] = state["exp_avg"].float()
        state["exp_avg"].flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            finite_update_guard(model, optimizer)

    def test_optimizer_cannot_include_reference_or_omit_student_parameters(self):
        model, reference = TinyCausal(), TinyCausal()
        reference.eval().requires_grad_(False)
        for parameters in (list(model.parameters())[:1], list(model.parameters()) + list(reference.parameters())):
            optimizer = torch.optim.SGD(parameters, lr=.1)
            with self.assertRaises(ValueError):
                check_optimizer(model, optimizer, reference)

    def test_reference_alias_is_rejected_before_student_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyCausal()
            with self.assertRaisesRegex(ValueError, "separate models"):
                OrderedTrainer(model=model, args=args(temporary), objective="ce_kl", reference_model=model,
                    eos_token_id=EOS, terminal_suffix=SUFFIX)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_bf16_student_native_clipping_and_missing_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "FP32"):
                OrderedTrainer(model=TinyCausal().to(torch.bfloat16), args=args(temporary), objective="sft",
                    eos_token_id=EOS, terminal_suffix=SUFFIX)
            with self.assertRaisesRegex(ValueError, "native clipping disabled"):
                OrderedTrainer(model=TinyCausal(), args=args(temporary, max_grad_norm=1.), objective="sft",
                    eos_token_id=EOS, terminal_suffix=SUFFIX)
            with self.assertRaisesRegex(ValueError, "reference"):
                OrderedTrainer(model=TinyCausal(), args=args(temporary), objective="ce_kl",
                    eos_token_id=EOS, terminal_suffix=SUFFIX)

    def test_resume_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = OrderedTrainer(model=TinyCausal(), args=args(temporary), objective="sft",
                eos_token_id=EOS, terminal_suffix=SUFFIX)
            with self.assertRaisesRegex(ValueError, "not implemented"):
                trainer.train(resume_from_checkpoint="some-checkpoint")
            with self.assertRaises(NotImplementedError):
                trainer.evaluate()
            with self.assertRaises(NotImplementedError):
                trainer.predict([])


if __name__ == "__main__":
    unittest.main()
