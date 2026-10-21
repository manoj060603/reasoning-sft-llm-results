"""Prospective32 CPU fixtures; no test/model workload was executed in authoring.

Actual tiny random original Qwen classes and pinned HF Trainer are used only in
future admitted CPU QA. The head's tiny-count override is explicit and scoped;
it never fakes434 or original checkpoint provenance. Small fixture-only tensor
clones implement independent dense math/Adam oracles, not a production snapshot
or substitute for bounded native comparisons/shadows. CPU observation hooks
observe real tiny forwards/backwards but assert no native BF16/GPU admission.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from transformers import Qwen2Config, Trainer, TrainerCallback, TrainingArguments

from extension import checkpointed_head_trainer as head
from extension import paired_training_step_controller as module
from extension import preserving_trainer as preserving


EOS, SUFFIX = 15, (15, 16)
PARAM_ATOL, PARAM_RTOL = 1e-7, 1e-6
GRAD_ATOL, GRAD_RTOL = 3e-6, 5e-5


def tiny(seed=71):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=19, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True,
        use_cache=False, eos_token_id=EOS, bos_token_id=1, pad_token_id=0)
    config._attn_implementation = "sdpa"
    value = head.Qwen2ForCausalLM(config).float()
    value.tie_weights()
    return value


def examples():
    values = [
        {"input_ids": [1, 2, EOS, 16], "labels": [-100, -100, EOS, 16]},
        {"input_ids": [1, 3, 4, 5, EOS, 16], "labels": [-100, -100, 4, 5, EOS, 16]},
        {"input_ids": [2, 4, 1, EOS, 16], "labels": [-100, -100, 1, EOS, 16]},
    ]
    return [{**deepcopy(values[index % 3]), "row_index": index} for index in range(9)]


def collate(rows):
    if len(rows) != 1:
        raise AssertionError("The fixture must retain microbatch1")
    row = rows[0]
    result = {name: torch.tensor([row[name]], dtype=torch.int64) for name in ("input_ids", "labels")}
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    result["row_index"] = torch.tensor([row["row_index"]], dtype=torch.int64)
    return result


def arguments(path):
    return TrainingArguments(output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=1, max_steps=2, max_grad_norm=0.,
        optim="adamw_torch", learning_rate=2e-6, weight_decay=.01,
        adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
        lr_scheduler_type="cosine", warmup_steps=57,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no", logging_strategy="no", eval_strategy="no", report_to=[],
        disable_tqdm=True, remove_unused_columns=False, dataloader_drop_last=False,
        dataloader_num_workers=0, dataloader_pin_memory=False, label_smoothing_factor=0.,
        seed=20261021, data_seed=20261021, torch_empty_cache_steps=None)


def dense_loss(model, batch, reference, mode):
    """Independent full-vocabulary equal-example formula, sole causal shift."""
    forward = {key: batch[key] for key in ("input_ids", "attention_mask")}
    logits = model(**forward, use_cache=False).logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    active, safe = targets != -100, targets.clamp_min(0)
    logp = logits.log_softmax(-1)
    target_logp = logp.gather(-1, safe[..., None]).squeeze(-1)
    weights, kl = torch.ones_like(target_logp), torch.zeros_like(target_logp)
    if mode != "sft":
        with torch.no_grad():
            logq = reference(**forward, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            if mode == "asft": weights = target_logp.detach().exp()
            elif mode == "prior_anchored": weights = logq.gather(-1, safe[..., None]).squeeze(-1).exp()
        kl = (logq.exp() * (logq - logp)).sum(-1)
    return (((-weights * target_logp + .05 * kl) * active).sum(1) / active.sum(1)).mean()


def clone_parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def close_maps(actual, expected, *, gradients=False):
    if set(actual) != set(expected):
        raise AssertionError("The fixture oracle itself requires exact FQN coverage")
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name],
            atol=GRAD_ATOL if gradients else PARAM_ATOL,
            rtol=GRAD_RTOL if gradients else PARAM_RTOL, msg=lambda text: name + ": " + text)


class CPUObserver:
    """Real tiny hook counts with explicitly CPU-only, non-native metadata."""
    def __init__(self, student, reference, branch, *, stochastic=False):
        self.student, self.reference, self.branch = student, reference, branch
        self.stack, self.handles, self.events, self.draws = [], [], [], []
        self.calls = {"forward_head": 0, "backward_head": 0, "reference": 0}
        self.layers, self.gradients = {}, {}
        self.entered, self.exited, self.stochastic = False, False, stochastic
        self.draw_count = 1

    def __enter__(self):
        if self.entered:
            raise AssertionError("Fixture observer cannot enter twice")
        self.entered = True
        def head_hook(_module, _arguments):
            if self.stack:
                phase = self.stack[-1][0]
                self.calls[phase + "_head"] += 1
        self.handles.append(self.student.lm_head.register_forward_pre_hook(head_hook))
        for index, layer in enumerate(self.student.model.layers):
            def layer_hook(_module, _arguments, index=index):
                if self.stack:
                    key = (index, self.stack[-1][0])
                    self.layers[key] = self.layers.get(key, 0) + 1
            self.handles.append(layer.register_forward_pre_hook(layer_hook))
        def reference_hook(_module, _arguments):
            if self.stack: self.calls["reference"] += 1
        self.handles.append(self.reference.lm_head.register_forward_pre_hook(reference_hook))
        for name, value in self.student.named_parameters():
            def grad_hook(gradient, name=name):
                if not self.stack or self.stack[-1][0] != "backward":
                    raise AssertionError("Tiny backward escaped its owned phase")
                self.gradients[name] = self.gradients.get(name, 0) + 1
                return gradient
            self.handles.append(value.register_hook(grad_hook))
        return self

    def __exit__(self, *error):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.exited = True
        return False

    @contextmanager
    def phase(self, name, *, presentation_index):
        self.stack.append((name, presentation_index))
        self.events.append(("enter", name, presentation_index))
        if name == "forward" and self.stochastic:
            values = []
            for _ in range(self.draw_count):
                values.append((random.random(), float(np.random.random()), float(torch.rand(()))))
            self.draws.append(values)
        try:
            yield
        finally:
            self.events.append(("exit", name, presentation_index))
            self.stack.pop()

    def snapshot(self):
        return {"cpu_fixture_only": True, "native_dtype_or_recompute_proven": False,
            "forward_head": self.calls["forward_head"], "backward_head": self.calls["backward_head"],
            "reference_calls": self.calls["reference"], "gradient_calls": dict(self.gradients)}

    def validate(self, *, require_backward_recompute, require_all_layers, require_reference):
        if self.stack or set(self.gradients) != set(dict(self.student.named_parameters())) or any(count != 9 for count in self.gradients.values()):
            raise AssertionError("The actual tiny gradient hooks need all nine backwards")
        if not all(self.layers.get((index, "forward"), 0) >= 9 for index in range(len(self.student.model.layers))):
            raise AssertionError("Missing actual tiny decoder forward observations")
        if require_backward_recompute and (not self.calls["backward_head"] or
                not any(self.layers.get((index, "backward"), 0) for index in range(len(self.student.model.layers)))):
            raise AssertionError("Missing actual tiny CPU candidate recomputation")
        if require_reference and self.calls["reference"] < 9:
            raise AssertionError("Missing active-branch reference calls")
        if not require_reference and self.calls["reference"]:
            raise AssertionError("SFT unexpectedly used the reference")
        return {**self.snapshot(), "required_all_layers_checked": bool(require_all_layers)}


def observed_trainer(base):
    class Observed(base):
        def __init__(self, *args, paired_observer, **kwargs):
            self.paired_observer = paired_observer
            self.paired_presentation_index = None
            super().__init__(*args, **kwargs)

        def compute_loss(self, *args, **kwargs):
            with self.paired_observer.phase("forward", presentation_index=self.paired_presentation_index):
                return super().compute_loss(*args, **kwargs)
    return Observed


DenseTrainer = observed_trainer(preserving.make_trainer_class())
CandidateTrainer = observed_trainer(head.make_trainer_class())


class FixtureHooks:
    """Independent tiny-only dense gradients, actual Adam states and deltas."""
    def __init__(self, dense, candidate, reference, batches, mode, stack):
        self.trainers = {"dense": dense, "candidate": candidate}
        self.reference, self.batches, self.mode, self.stack = reference, batches, mode, stack
        self.initial = clone_parameters(reference)
        self.manual = deepcopy(reference).train().requires_grad_(True)
        self.manual.gradient_checkpointing_disable()
        names = dict(self.manual.named_parameters())
        decay = [name for name in names if not name.endswith(".bias") and "norm" not in name.lower()]
        excluded = [name for name in names if name not in decay]
        self.manual_optimizer = torch.optim.AdamW([
            {"params": [names[name] for name in decay], "weight_decay": .01},
            {"params": [names[name] for name in excluded], "weight_decay": 0.}],
            lr=2e-6, betas=(.9, .999), eps=1e-8)
        self.manual_updates, self.manual_norm, self.events = 0, None, []
        self.raw, self.clipped, self.norms, self.callback_calls = {}, {}, {}, []
        self.presentations, self.shadow_handles = [], []
        self.before = clone_parameters(reference)

    def receipt(self, stage, *, count=False, **values):
        return {"complete": True, "passed": True, "stage": stage, "cpu_fixture_only": True,
            "native_provenance_established": False,
            **({"compared_tensor_count": len(self.initial)} if count else {}), **values}

    def verify_invariants(self, *, stage, **kwargs):
        self.events.append(stage)
        if self.reference.training or any(value.requires_grad or value.grad is not None for value in self.reference.parameters()):
            raise AssertionError("Reference was not isolated/frozen")
        close_maps(dict(self.reference.named_parameters()), self.before)
        storages = []
        for trainer in self.trainers.values():
            if trainer.model.lm_head.weight is not trainer.model.get_input_embeddings().weight:
                raise AssertionError("Lost actual tiny tied head")
            raw = preserving.check_optimizer(trainer.model, trainer.optimizer, self.reference)
            if any(value.dtype != torch.float32 for value in trainer.model.parameters()):
                raise AssertionError("Lost FP32 Parameters")
            storages.append({value.untyped_storage().data_ptr() for value in trainer.model.parameters()})
            for state in raw.state.values():
                if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                    raise AssertionError("Nonfinite actual tiny optimizer state")
        if storages[0] & storages[1]: raise AssertionError("Students share storage")
        if stage == "prepared":
            for branch, trainer in self.trainers.items():
                callback = trainer.callback_handler.callbacks[-1]
                original = callback.on_pre_optimizer_step
                def recorded(*args, _original=original, _branch=branch, **values):
                    self.callback_calls.append((_branch, "clip"))
                    return _original(*args, **values)
                self.stack.enter_context(patch.object(callback, "on_pre_optimizer_step", side_effect=recorded))
        return self.receipt(stage)

    def verify_identity(self, *, stage, **kwargs):
        self.events.append(stage)
        for trainer in self.trainers.values():
            for name, value in trainer.model.named_parameters():
                if not torch.equal(value.detach(), self.initial[name]):
                    raise AssertionError("Actual stored FP32 weight changed at LR0/original identity")
        return self.receipt(stage, count=True)

    def compare_scalars(self, *, stage, dense, candidate, **kwargs):
        self.events.append(stage)
        if stage.startswith("presentation_"):
            index = int(stage.rsplit("_", 1)[1]); size = 8 if index < 8 else 1
            with torch.no_grad():
                expected = float(dense_loss(self.reference, self.batches[index], self.reference, self.mode))
            active = int((self.batches[index]["labels"] != -100).sum())
            eos = int((self.batches[index]["labels"] == EOS).sum())
            for values in (dense, candidate):
                if values["group_size"] != size or not math.isclose(values["scaled_loss"], expected / size, rel_tol=3e-5, abs_tol=2e-6):
                    raise AssertionError("Actual training_step did not divide exactly once by8/1")
                if not math.isclose(values["metrics"]["loss"], expected, rel_tol=3e-5, abs_tol=2e-6):
                    raise AssertionError("Unscaled objective was replaced")
                if values["metrics"]["supervised_tokens"] != active or values["metrics"]["eos_tokens"] != eos:
                    raise AssertionError("Lost native target/EOS count")
            self.presentations.append((index, size, expected, dense["scaled_loss"], candidate["scaled_loss"]))
        else:
            if not all(math.isclose(values["clip_norm"], self.manual_norm, rel_tol=5e-5, abs_tol=3e-6) for values in (dense, candidate)):
                raise AssertionError("Actual clip norm differs from independent dense gradients")
        return self.receipt(stage)

    def compare_tensors(self, *, stage, **kwargs):
        self.events.append(stage)
        update = int(stage.rsplit("_", 1)[1])
        if stage.startswith("raw_gradients"):
            self.manual_optimizer.zero_grad(set_to_none=True)
            indices = range(8) if update == 1 else (8,)
            for index in indices:
                (dense_loss(self.manual, self.batches[index], self.reference, self.mode) / len(indices)).backward()
            expected = {name: value.grad for name, value in self.manual.named_parameters()}
            for branch, trainer in self.trainers.items():
                actual = {name: value.grad for name, value in trainer.model.named_parameters()}
                close_maps(actual, expected, gradients=True)
                self.raw[(branch, update)] = {name: value.detach().clone() for name, value in actual.items()}
            for group in self.manual_optimizer.param_groups: group["lr"] = 2e-6 * (update - 1) / 57
        elif stage.startswith("clipped_gradients"):
            self.manual_norm = float(torch.nn.utils.clip_grad_norm_(self.manual.parameters(), 1., error_if_nonfinite=True))
            self.norms[update] = self.manual_norm
            expected = {name: value.grad for name, value in self.manual.named_parameters()}
            for branch, trainer in self.trainers.items():
                actual = {name: value.grad for name, value in trainer.model.named_parameters()}
                close_maps(actual, expected, gradients=True)
                self.clipped[(branch, update)] = {name: value.detach().clone() for name, value in actual.items()}
        elif stage.startswith("adam_first_moments"):
            self.manual_optimizer.step(); self.manual_updates += 1
            if self.manual_updates != update: raise AssertionError("Independent oracle updated twice")
            self._compare_state("exp_avg")
        elif stage.startswith("adam_second_moments"):
            self._compare_state("exp_avg_sq")
        elif stage.startswith("updated_weights"):
            for trainer in self.trainers.values(): close_maps(dict(trainer.model.named_parameters()), dict(self.manual.named_parameters()))
            self.manual_optimizer.zero_grad(set_to_none=True)
        else:
            raise AssertionError("The fixture did not implement a required numerical comparison")
        return self.receipt(stage, count=True)

    def _compare_state(self, key):
        manual = dict(self.manual.named_parameters())
        for trainer in self.trainers.values():
            raw = preserving._optimizer(trainer.optimizer)
            for name, value in trainer.model.named_parameters():
                expected = self.manual_optimizer.state[manual[name]][key]
                torch.testing.assert_close(raw.state[value][key], expected, atol=3e-7 if key == "exp_avg" else 1e-10,
                    rtol=1e-4)
                if raw.state[value][key].dtype != torch.float32: raise AssertionError("Non-FP32 Adam state")

    def create_shadow(self, *, stage, **kwargs):
        self.events.append(stage)
        initial = {branch: clone_parameters(trainer.model) for branch, trainer in self.trainers.items()}
        self.shadow_handles.append(initial)
        return module.ShadowArtifact(initial, self.receipt(stage, count=True, fixture_small_clone_only=True))

    def compare_deltas(self, *, stage, initial_shadow, **kwargs):
        self.events.append(stage)
        positive, deltas = {}, {}
        for branch, trainer in self.trainers.items():
            deltas[branch] = {name: value.detach().double() - initial_shadow[branch][name].double()
                for name, value in trainer.model.named_parameters()}
            positive[branch] = {"tied": bool(deltas[branch]["model.embed_tokens.weight"].abs().sum() > 0),
                "backbone": any(bool(value.abs().sum() > 0) for name, value in deltas[branch].items() if name != "model.embed_tokens.weight")}
        for name in deltas["dense"]:
            torch.testing.assert_close(deltas["candidate"][name], deltas["dense"][name], atol=1e-8, rtol=1e-4)
        return self.receipt(stage, count=True, positive_update_groups=positive)

    def api(self):
        return module.PairedStepHooks(self.verify_invariants, self.verify_identity, self.compare_scalars,
            self.compare_tensors, self.create_shadow, self.compare_deltas)


@contextmanager
def pair_fixture(mode="sft", *, stochastic=False):
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with ExitStack() as stack:
        temporary = stack.enter_context(tempfile.TemporaryDirectory())
        original = tiny()
        reference = deepcopy(original).eval().requires_grad_(False)
        models = {name: deepcopy(original) for name in module.BRANCHES}
        observers = {name: CPUObserver(models[name], reference, name, stochastic=stochastic) for name in module.BRANCHES}
        count = len(list(original.parameters()))
        stack.enter_context(patch.object(head, "EXPECTED_PARAMETER_TENSORS", count))
        trainers = {}
        for name, cls in (("dense", DenseTrainer), ("candidate", CandidateTrainer)):
            trainers[name] = cls(model=models[name], args=arguments(Path(temporary) / name),
                train_dataset=examples(), data_collator=collate, paired_observer=observers[name],
                reference_model=reference, objective=mode, eos_token_id=EOS, terminal_suffix=SUFFIX,
                beta=.05, chunk_tokens=128, checkpoint_chunks=True, clip_norm=1.)
        batches = [collate([row]) for row in examples()]
        hooks = FixtureHooks(trainers["dense"], trainers["candidate"], reference, batches, mode, stack)
        value = {"models": models, "reference": reference, "observers": observers,
            "trainers": trainers, "batches": batches, "hooks": hooks, "count": count, "stack": stack}
        try:
            yield value
        finally:
            for trainer in trainers.values():
                trainer.accelerator.gradient_accumulation_steps = 1
                trainer.accelerator.sync_gradients = True
            random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])


def run_pair(fixture, *, hooks=None, expected=None):
    return module.run_paired_training_steps(fixture["trainers"]["dense"], fixture["trainers"]["candidate"],
        fixture["batches"], reference_model=fixture["reference"], hooks=fixture["hooks"].api() if hooks is None else hooks,
        observers=fixture["observers"], expected_parameters=fixture["count"] if expected is None else expected)


class StockRecorder(TrainerCallback):
    def __init__(self): self.raw, self.steps = [], []
    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.raw.append({name: value.grad.detach().clone() for name, value in model.named_parameters()})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.steps.append({"weights": clone_parameters(model), "lrs": [float(group["lr"]) for group in optimizer.param_groups]})


class StockDenseTrainer(Trainer):
    """Actual stock train-loop fixture with independent dense objective only."""
    def __init__(self, *args, reference, mode, **kwargs):
        self.reference, self.mode = reference, mode
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
    def create_optimizer_and_scheduler(self, num_training_steps):
        # The full loop still stops after two updates; its scheduler is the
        # predeclared full1875 horizon, not a fresh two-step cosine schedule.
        return super().create_optimizer_and_scheduler(num_training_steps=1875)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = dense_loss(model, inputs, self.reference, self.mode)
        return (loss, {"loss": loss.detach()}) if return_outputs else loss


class PairedTrainingStepControllerTests(unittest.TestCase):
    def failure(self, fixture, **kwargs):
        with self.assertRaises(module.PairedControllerFailure) as caught:
            run_pair(fixture, **kwargs)
        value = caught.exception.result
        self.assertFalse(value["complete"])
        self.assertTrue(value["caller_must_preserve_private_partial_result_and_abort"])
        self.assertFalse(value["training_or_pilot_admitted"])
        json.dumps(value, allow_nan=False)
        return value

    def stock_oracle(self, mode):
        with pair_fixture(mode) as fixture:
            result = run_pair(fixture)
            with tempfile.TemporaryDirectory() as temporary:
                original = deepcopy(fixture["reference"]).train().requires_grad_(True)
                recorder = StockRecorder()
                trainer = StockDenseTrainer(model=original, args=arguments(temporary),
                    reference=fixture["reference"], mode=mode,
                    train_dataset=examples(), data_collator=collate, callbacks=[recorder])
                trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                self.assertEqual(len(recorder.raw), 2)
                self.assertEqual(len(recorder.steps), 2)
                for branch in module.BRANCHES:
                    for update in (1, 2):
                        close_maps(fixture["hooks"].raw[(branch, update)], recorder.raw[update - 1], gradients=True)
                    close_maps(dict(fixture["models"][branch].named_parameters()), dict(original.named_parameters()))
                self.assertEqual(recorder.steps[0]["lrs"], [0., 0.])
                self.assertTrue(all(math.isclose(lr, 2e-6 / 57, rel_tol=1e-12) for lr in recorder.steps[1]["lrs"]))
                for name, initial in fixture["hooks"].initial.items():
                    self.assertTrue(torch.equal(recorder.steps[0]["weights"][name], initial))
                self.assertTrue(result["complete"])

    def test_fixed_schedule_and_required_hook_contract_are_explicit(self):
        self.assertEqual(module.ROLES, ("short0", "short1", "short2") * 3)
        self.assertEqual(module.GROUPS, (tuple(range(8)), (8,)))
        self.assertEqual((module.WARMUP_STEPS, module.TOTAL_STEPS, module.SEED), (57, 1875, 20261021))
        self.assertEqual((module.PEAK_LR, module.WEIGHT_DECAY, module.BETA), (2e-6, .01, .05))
        with self.assertRaises(ValueError): module.PairedStepHooks(*([None] * 6))

    def test_tiny_fixture_never_satisfies_default434_or_claims_original_checkpoint_provenance(self):
        with pair_fixture() as fixture:
            self.assertNotEqual(fixture["count"], 434)
            result = self.failure(fixture, expected=434)
            self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})
            self.assertFalse(result["native_provenance_established"])
        self.assertEqual(head.EXPECTED_PARAMETER_TENSORS, 434)

    def test_observation_subclasses_keep_the_actual_inherited_hf_training_step(self):
        with pair_fixture() as fixture:
            for trainer in fixture["trainers"].values():
                self.assertIs(trainer.training_step.__func__, Trainer.training_step)
                self.assertIsNot(trainer.compute_loss.__func__, Trainer.compute_loss)
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                self.assertIsNone(trainer.compute_loss_func)

    def test_actual_sft_nine_step_completion_is_explicitly_not_full_loop_or_native_admission(self):
        with pair_fixture() as fixture:
            result = run_pair(fixture)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_backwards"], {"dense": 9, "candidate": 9})
            self.assertEqual(result["optimizer_updates"], {"dense": 2, "candidate": 2})
            self.assertEqual(result["actual_group_sizes"], [8, 1])
            self.assertTrue(result["fixture_count_is_not_native_proof"])
            for name in ("trainer_train_called", "full_trainer_train_loop_established", "unique_row_driver_established",
                         "long_context_memory_established", "native_provenance_established", "training_or_pilot_admitted"):
                self.assertFalse(result[name], name)

    def test_actual_gradients_clipping_adam_and_stored_update2_match_independent_dense_math(self):
        with pair_fixture() as fixture:
            result = run_pair(fixture)
            self.assertEqual(fixture["hooks"].manual_updates, 2)
            self.assertEqual(set(fixture["hooks"].raw), {(branch, update) for branch in module.BRANCHES for update in (1, 2)})
            for branch, trainer in fixture["trainers"].items():
                self.assertTrue(any(not torch.equal(value, fixture["hooks"].initial[name]) for name, value in trainer.model.named_parameters()))
                self.assertTrue(all(value.grad is None for value in trainer.model.parameters()))
                raw = preserving._optimizer(trainer.optimizer)
                self.assertEqual(len(raw.state), fixture["count"])
                self.assertTrue(all(float(value["step"]) == 2 for value in raw.state.values()))
                self.assertTrue(all(value["exp_avg"].dtype == value["exp_avg_sq"].dtype == torch.float32 for value in raw.state.values()))
            self.assertEqual(result["optimizer_step_attempts"], {"dense": 2, "candidate": 2})

    def test_all_anchored_objectives_use_independent_dense_loss_and_gradient_oracles(self):
        for mode in ("ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode), pair_fixture(mode) as fixture:
                result = run_pair(fixture)
                self.assertTrue(result["complete"])
                self.assertEqual(len(fixture["hooks"].presentations), 9)
                self.assertTrue(all(observer.calls["reference"] >= 9 for observer in fixture["observers"].values()))
                self.assertTrue(all(value.grad is None and not value.requires_grad for value in fixture["reference"].parameters()))

    def test_independent_stock_trainer_train_loop_matches_actual_sft_training_steps(self):
        self.stock_oracle("sft")

    def test_independent_stock_trainer_train_loop_matches_actual_cekl_training_steps(self):
        self.stock_oracle("ce_kl")

    def test_independent_stock_trainer_train_loop_matches_probability_weighted_modes(self):
        for mode in ("asft", "prior_anchored"):
            with self.subTest(mode=mode): self.stock_oracle(mode)

    def test_event_choreography_compares_raw_then_once_clipped_then_both_steps_and_states(self):
        with pair_fixture() as fixture:
            events = run_pair(fixture)["events"]
            for update in (1, 2):
                def position(predicate): return next(index for index, row in enumerate(events) if predicate(row))
                raw = position(lambda row: row.get("stage") == "raw_gradients_" + str(update))
                clipped = position(lambda row: row.get("stage") == "clipped_gradients_" + str(update))
                begins = [position(lambda row, branch=branch: row["event"] == "optimizer_step_begin" and row.get("branch") == branch and row.get("update") == update) for branch in module.BRANCHES]
                ends = [position(lambda row, branch=branch: row["event"] == "optimizer_step_complete" and row.get("branch") == branch and row.get("update") == update) for branch in module.BRANCHES]
                moment = position(lambda row: row.get("stage") == "adam_first_moments_" + str(update))
                final = position(lambda row: row["event"] == "group_finalized" and row.get("update") == update)
                self.assertLess(raw, clipped); self.assertLess(clipped, min(begins))
                self.assertLess(begins[0], ends[0]); self.assertLess(ends[0], begins[1]); self.assertLess(begins[1], ends[1])
                self.assertLess(max(ends), moment); self.assertLess(moment, final)

    def test_owned_finite_clip_callback_is_invoked_exactly_twice_per_branch(self):
        with pair_fixture() as fixture:
            run_pair(fixture)
            self.assertEqual(fixture["hooks"].callback_calls, [("dense", "clip"), ("candidate", "clip")] * 2)
            self.assertEqual(set(fixture["hooks"].norms), {1, 2})
            self.assertTrue(all(math.isfinite(value) for value in fixture["hooks"].norms.values()))

    def test_preparation_keeps_original_models_raw_scheduler_and_exact_adam_groups(self):
        with pair_fixture() as fixture:
            run_pair(fixture)
            for branch, trainer in fixture["trainers"].items():
                raw = preserving._optimizer(trainer.optimizer)
                self.assertIs(trainer.model, fixture["models"][branch])
                self.assertIs(trainer.lr_scheduler.optimizer, raw)
                self.assertEqual(trainer.accelerator.gradient_accumulation_steps, 1)
                self.assertEqual(list(trainer.accelerator._models), [trainer.model])
                self.assertEqual(list(trainer.accelerator._schedulers), [])
                self.assertIs(type(raw), torch.optim.AdamW)
                self.assertIsNone(raw.defaults["foreach"]); self.assertIsNone(raw.defaults["fused"])
                self.assertEqual([group["weight_decay"] for group in raw.param_groups], [.01, 0.])
                self.assertEqual(trainer.lr_scheduler.last_epoch, 2)

    def test_actual_backward_consumption_is_nine_ordered_repetitions_with_exact_native_targets(self):
        with pair_fixture() as fixture:
            result = run_pair(fixture)
            events = [row for row in result["events"] if row["event"] == "training_step_begin"]
            for branch, trainer in fixture["trainers"].items():
                self.assertEqual(trainer.training_forward_examples, 9)
                rows = [row for row in events if row["branch"] == branch]
                self.assertEqual([row["presentation_index"] for row in rows], list(range(9)))
                self.assertEqual([row["source_role"] for row in rows], list(module.ROLES))
                self.assertTrue(all(value == 9 for value in fixture["observers"][branch].gradients.values()))
            self.assertEqual(sum(int((batch["labels"] == EOS).sum()) for batch in fixture["batches"]), 9)

    def test_returned_loss_is_divided_once_by_eight_then_once_by_one(self):
        with pair_fixture() as fixture:
            run_pair(fixture)
            values = fixture["hooks"].presentations
            self.assertEqual([size for _, size, *_ in values], [8] * 8 + [1])
            for index, size, raw, dense, candidate in values:
                self.assertTrue(math.isclose(dense, raw / size, rel_tol=3e-5, abs_tol=2e-6))
                self.assertTrue(math.isclose(candidate, raw / size, rel_tol=3e-5, abs_tol=2e-6))
                if index < 8: self.assertGreater(abs(dense - raw / 64), .001)

    def test_rng_replay_advances_python_numpy_and_torch_only_one_logical_stream(self):
        with pair_fixture(stochastic=True) as fixture:
            random.seed(321); np.random.seed(456); torch.manual_seed(789)
            python = random.Random(321); numpy = np.random.RandomState(456)
            generator = torch.Generator().manual_seed(789)
            expected = [(python.random(), float(numpy.random_sample()), float(torch.rand((), generator=generator))) for _ in range(9)]
            result = run_pair(fixture)
            for observer in fixture["observers"].values(): self.assertEqual(observer.draws, [[row] for row in expected])
            self.assertEqual(random.random(), python.random())
            self.assertEqual(float(np.random.random()), float(numpy.random_sample()))
            self.assertEqual(float(torch.rand(())), float(torch.rand((), generator=generator)))
            self.assertFalse(result["seed_contract"]["controller_reseeds_generators"])

    def test_rng_divergence_fails_after_actual_backward_and_removes_both_observer_hooks(self):
        with pair_fixture(stochastic=True) as fixture:
            fixture["observers"]["candidate"].draw_count = 2
            result = self.failure(fixture)
            self.assertEqual(result["completed_backwards"], {"dense": 1, "candidate": 1})
            self.assertEqual(result["optimizer_step_attempts"], {"dense": 0, "candidate": 0})
            self.assertTrue(all(value.exited and not value.handles for value in fixture["observers"].values()))

    def test_missing_hooks_or_observer_contract_is_rejected_without_backward(self):
        with pair_fixture() as fixture:
            result = self.failure(fixture, hooks=object())
            self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})
        with pair_fixture() as fixture:
            fixture["observers"]["dense"].snapshot = None
            self.assertEqual(self.failure(fixture)["optimizer_step_attempts"], {"dense": 0, "candidate": 0})

    def test_hook_receipts_cannot_skip_stage_count_finite_or_passed_requirements(self):
        for change in ({"stage": "wrong"}, {"compared_tensor_count": 434}, {"passed": False}, {"value": float("nan")}):
            with self.subTest(change=change), pair_fixture() as fixture:
                original = fixture["hooks"].verify_identity
                def invalid(**kwargs): return {**original(**kwargs), **change}
                result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), verify_identity=invalid))
                self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_native_short_batch_field_dtype_shape_and_mask_mutations_are_detected(self):
        baseline = collate([examples()[0]])
        for change in ("dtype", "long", "extra", "row_index_shape"):
            value = deepcopy(baseline)
            if change == "dtype": value["labels"] = value["labels"].float()
            elif change == "long": value = {name: torch.ones((1, 513), dtype=torch.int64) for name in ("input_ids", "labels", "attention_mask")}
            elif change == "extra": value["secret"] = torch.tensor([1])
            else: value["row_index"] = torch.tensor([0, 1])
            with self.subTest(change=change), self.assertRaises(ValueError): module._batch_record(value)
        with pair_fixture() as fixture:
            fixture["batches"][0]["labels"][0, -1] = -100
            # The semantic repeat hash rejects this before any training call.
            self.assertEqual(self.failure(fixture)["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_duplicate_semantic_witnesses_and_wrong_repetition_order_are_rejected(self):
        for change in ("duplicates", "order"):
            with self.subTest(change=change), pair_fixture() as fixture:
                if change == "duplicates":
                    fixture["batches"] = [deepcopy(fixture["batches"][0]) for _ in range(9)]
                    for index, batch in enumerate(fixture["batches"]): batch["row_index"] = torch.tensor([index])
                else: fixture["batches"][0], fixture["batches"][1] = fixture["batches"][1], fixture["batches"][0]
                self.assertEqual(self.failure(fixture)["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_unique_presentation_indices_do_not_turn_three_repeated_sources_into_nine_unique_rows(self):
        with pair_fixture() as fixture:
            records = [module._batch_record(row) for row in fixture["batches"]]
            self.assertEqual(len({json.dumps(row["row_index"]) for row in records}), 9)
            semantics = [{name: value for name, value in row.items() if name != "row_index"} for row in records]
            self.assertEqual(len({json.dumps(row, sort_keys=True) for row in semantics}), 3)
            self.assertTrue(all(semantics[index] == semantics[index % 3] for index in range(9)))

    def test_batch_mutation_after_a_presentation_blocks_the_next_backward(self):
        with pair_fixture() as fixture:
            original = fixture["hooks"].compare_scalars
            def altered(**kwargs):
                value = original(**kwargs)
                if kwargs["stage"] == "presentation_0": fixture["batches"][8]["input_ids"][0, 0] += 1
                return value
            result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), compare_scalars=altered))
            self.assertEqual(result["completed_backwards"], {"dense": 1, "candidate": 1})
            self.assertEqual(result["optimizer_step_attempts"], {"dense": 0, "candidate": 0})

    def test_fresh_noresume_and_no_reentry_guards_do_not_reuse_completed_trajectory(self):
        with pair_fixture() as fixture:
            first = run_pair(fixture)
            second = self.failure(fixture)
            self.assertTrue(first["complete"])
            self.assertEqual(second["completed_backwards"], {"dense": 0, "candidate": 0})
        with pair_fixture() as fixture:
            fixture["trainers"]["dense"]._prototype_train_started = True
            self.assertEqual(self.failure(fixture)["optimizer_step_attempts"], {"dense": 0, "candidate": 0})
        with pair_fixture() as fixture, self.assertRaises(TypeError):
            module.run_paired_training_steps(fixture["trainers"]["dense"], fixture["trainers"]["candidate"], fixture["batches"],
                reference_model=fixture["reference"], hooks=fixture["hooks"].api(), observers=fixture["observers"], resume_from_checkpoint="fixture")

    def test_replaced_training_step_is_not_accepted_as_an_actual_hf_backward(self):
        with pair_fixture() as fixture:
            class Replaced(DenseTrainer):
                def training_step(self, *args, **kwargs): return torch.tensor(0.)
            fixture["trainers"]["dense"].__class__ = Replaced
            self.assertEqual(self.failure(fixture)["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_nonfresh_optimizer_prepared_accelerator_or_extra_callback_blocks_preparation(self):
        for change in ("optimizer", "prepared", "callback"):
            with self.subTest(change=change), pair_fixture() as fixture:
                trainer = fixture["trainers"]["dense"]
                if change == "optimizer": trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
                elif change == "prepared": trainer.accelerator._models.append(trainer.model)
                else: trainer.add_callback(TrainerCallback())
                self.assertEqual(self.failure(fixture)["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_wrong_accumulation_precision_objective_or_scheduler_configuration_is_not_adapted(self):
        for change in ("ga", "accelerator_divisor", "beta", "chunk", "scheduler"):
            with self.subTest(change=change), pair_fixture() as fixture:
                trainer = fixture["trainers"]["dense"]
                if change == "ga": trainer.args.gradient_accumulation_steps = 4
                elif change == "accelerator_divisor": trainer.accelerator.gradient_accumulation_steps = 8
                elif change == "beta": trainer.beta = .1
                elif change == "chunk": trainer.chunk_tokens = 64
                else: trainer.args.lr_scheduler_type = "linear"
                self.assertEqual(self.failure(fixture)["optimizer_step_attempts"], {"dense": 0, "candidate": 0})

    def test_reference_training_gradients_or_cross_student_storage_aliases_are_rejected(self):
        for change in ("reference_train", "reference_grad", "student_alias"):
            with self.subTest(change=change), pair_fixture() as fixture:
                if change == "reference_train": fixture["reference"].train()
                elif change == "reference_grad": next(fixture["reference"].parameters()).grad = torch.ones_like(next(fixture["reference"].parameters()))
                else:
                    left = fixture["models"]["dense"].get_input_embeddings().weight
                    fixture["models"]["candidate"].get_input_embeddings().weight.data = left.data
                self.assertEqual(self.failure(fixture)["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_nonfinite_raw_gradient_stops_owned_clip_before_any_optimizer_attempt(self):
        with pair_fixture() as fixture:
            original = fixture["hooks"].compare_tensors
            def bad_gradient(**kwargs):
                receipt = original(**kwargs)
                if kwargs["stage"] == "raw_gradients_1": next(fixture["models"]["dense"].parameters()).grad.fill_(float("nan"))
                return receipt
            result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), compare_tensors=bad_gradient))
            self.assertEqual(result["completed_backwards"], {"dense": 8, "candidate": 8})
            self.assertEqual(result["optimizer_step_attempts"], {"dense": 0, "candidate": 0})

    def test_actual_optimizer_state_guard_rejects_wrong_step_nonfinite_and_non_fp32_moments(self):
        with pair_fixture() as fixture:
            run_pair(fixture)
            trainer = fixture["trainers"]["dense"]; parameters = dict(trainer.model.named_parameters())
            raw = preserving._optimizer(trainer.optimizer); parameter = next(iter(parameters.values()))
            original = deepcopy(raw.state[parameter])
            for change in ("step", "nan", "bf16", "negative_second"):
                raw.state[parameter] = deepcopy(original)
                if change == "step": raw.state[parameter]["step"].fill_(3.)
                elif change == "nan": raw.state[parameter]["exp_avg"].fill_(float("nan"))
                elif change == "bf16": raw.state[parameter]["exp_avg"] = raw.state[parameter]["exp_avg"].bfloat16()
                else: raw.state[parameter]["exp_avg_sq"].fill_(-1.)
                with self.subTest(change=change), self.assertRaises(ValueError): module._state(trainer, parameters, 2)
            raw.state[parameter] = original

    def test_second_branch_optimizer_failure_preserves_partial_mutation_without_retry_or_rollback(self):
        with pair_fixture() as fixture:
            original_invariant = fixture["hooks"].verify_invariants
            calls = []
            def inject(**kwargs):
                receipt = original_invariant(**kwargs)
                if kwargs["stage"] == "prepared":
                    raw = preserving._optimizer(fixture["trainers"]["candidate"].optimizer); original = raw.step
                    def failing(*args, **values):
                        calls.append(1)
                        if len(calls) == 2: raise RuntimeError("original synthetic second optimizer fault")
                        return original(*args, **values)
                    fixture["stack"].enter_context(patch.object(raw, "step", side_effect=failing))
                return receipt
            result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), verify_invariants=inject))
            self.assertEqual(result["completed_backwards"], {"dense": 9, "candidate": 9})
            self.assertEqual(result["optimizer_updates"], {"dense": 2, "candidate": 1})
            self.assertEqual(result["optimizer_step_attempts"], {"dense": 2, "candidate": 2})
            self.assertTrue(result["partial_optimizer_state_may_have_changed"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(any(not torch.equal(value, fixture["hooks"].initial[name]) for name, value in fixture["models"]["dense"].named_parameters()))
            self.assertTrue(all(torch.equal(value, fixture["hooks"].initial[name]) for name, value in fixture["models"]["candidate"].named_parameters()))

    def test_missing_initial_shadow_or_false_positive_update_group_closes_the_attempt(self):
        with pair_fixture() as fixture:
            def missing(**kwargs): return module.ShadowArtifact(None, fixture["hooks"].receipt(kwargs["stage"], count=True))
            result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), create_shadow=missing))
            self.assertEqual(result["optimizer_updates"], {"dense": 1, "candidate": 1})
        with pair_fixture() as fixture:
            original = fixture["hooks"].compare_deltas
            def unproven(**kwargs):
                receipt = original(**kwargs); receipt["positive_update_groups"]["dense"]["backbone"] = False
                return receipt
            result = self.failure(fixture, hooks=replace(fixture["hooks"].api(), compare_deltas=unproven))
            self.assertEqual(result["optimizer_updates"], {"dense": 2, "candidate": 2})
            self.assertFalse(result["complete"])

    def test_bounded_plain_json_receipts_reject_tensors_nonfinite_and_oversized_events(self):
        for value in ({"tensor": torch.tensor(1.)}, {"number": float("nan")}, {"payload": "x" * (module.MAX_RECEIPT_BYTES + 1)}):
            with self.assertRaises(ValueError): module._json_copy(value)
        journal = module._Journal()
        for _ in range(module.MAX_EVENTS): journal.add("fixture", complete=False)
        with self.assertRaises(ValueError): journal.add("extra")
        self.assertEqual(module._json_copy({"cpu_fixture_only": True, "native_provenance_established": False}),
            {"cpu_fixture_only": True, "native_provenance_established": False})


if __name__ == "__main__":
    unittest.main()

