"""Unexecuted prospective24 CPU fixtures; no downloads or experiment inputs.

The real-HF cases use a tiny, randomly initialized original Qwen2 class. Its
tensor-count guard is explicitly patched ONLY inside the fixture; unpatched434
rejection is tested separately. FP32 CPU dense-loss/gradient/update oracles do
not prove the original3B weights, native BF16/CUDA/recompute/save or memory fit.
Mocked Accelerator seams test guards/context placement, never native admission.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import SequentialSampler
from transformers import Qwen2Config, TrainingArguments

from extension import checkpointed_head_trainer as module
from extension.preserving_trainer import check_optimizer, finite_update_guard


EOS, SUFFIX = 15, (15, 16)
MODES = ("sft", "ce_kl", "asft", "prior_anchored")


def tiny(seed=41):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=19, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True,
        use_cache=False, bos_token_id=1, eos_token_id=EOS, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = module.Qwen2ForCausalLM(config).float()
    model.tie_weights()
    return model


def rows(count=3):
    examples = [
        {"input_ids": [1, 2, EOS, 16], "labels": [-100, -100, EOS, 16]},
        {"input_ids": [1, 3, 4, 5, EOS, 16], "labels": [-100, -100, 4, 5, EOS, 16]},
        {"input_ids": [2, 4, 1, EOS, 16], "labels": [-100, -100, 1, EOS, 16]},
    ]
    return [deepcopy(examples[index % 3]) for index in range(count)]


def collate(values):
    if len(values) != 1:
        raise AssertionError("The fixture itself must retain microbatch1")
    batch = {name: torch.tensor([value], dtype=torch.int64) for name, value in values[0].items()}
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return batch


def arguments(directory, **changes):
    value = dict(output_dir=str(directory), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        num_train_epochs=1, max_steps=1, max_grad_norm=0., optim="adamw_torch",
        save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True,
        remove_unused_columns=False, dataloader_pin_memory=False, dataloader_num_workers=0,
        label_smoothing_factor=0., seed=20261021, data_seed=20261021)
    value.update(changes)
    return TrainingArguments(**value)


def dense_loss(model, batch, *, mode="sft", reference=None, beta=.2):
    """Independent full-vocabulary, one-shift FP32 CPU mathematical oracle."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
    targets = batch["labels"][:, 1:]
    mask = targets != -100
    logp = logits.float().log_softmax(-1)
    target_logp = logp.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
    ce, weights, kl = -target_logp, torch.ones_like(target_logp), torch.zeros_like(target_logp)
    if mode != "sft":
        with torch.no_grad():
            qlogits = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
            logq = qlogits.float().log_softmax(-1)
            if mode == "asft":
                weights = target_logp.detach().exp()
            elif mode == "prior_anchored":
                weights = logq.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1).exp()
        kl = (logq.exp() * (logq - logp)).sum(-1)
    return ((weights * ce + beta * kl) * mask).sum() / mask.sum()


Base = module.make_trainer_class()


class OrderedTrainer(Base):
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)


@contextmanager
def trainer_fixture(student=None, reference=None, *, mode="sft", changes=None, data=None,
                    optimizers=(None, None), chunk_tokens=2, checkpoint_chunks=True):
    student = tiny() if student is None else student
    # This explicit tiny-model oracle must never be presented as434/native proof.
    tensor_count = len(list(student.parameters()))
    with tempfile.TemporaryDirectory() as temporary, patch.object(module, "EXPECTED_PARAMETER_TENSORS", tensor_count):
        trainer = OrderedTrainer(model=student, args=arguments(temporary, **(changes or {})),
            reference_model=reference, objective=mode, eos_token_id=EOS,
            terminal_suffix=SUFFIX, beta=.2, chunk_tokens=chunk_tokens,
            checkpoint_chunks=checkpoint_chunks, train_dataset=rows() if data is None else data,
            data_collator=collate, optimizers=optimizers)
        yield trainer, Path(temporary)


def state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def no_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return False
    if isinstance(value, dict):
        return all(no_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(no_tensor_tree(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


class CheckpointedHeadTrainerTests(unittest.TestCase):
    def test_tiny_real_qwen_fixture_does_not_satisfy_unpatched434_admission(self):
        model = tiny()
        self.assertIs(type(model), module.Qwen2ForCausalLM)
        self.assertIs(type(model.model), module.Qwen2Model)
        self.assertNotEqual(len(list(model.parameters())), 434)
        self.assertEqual(module.EXPECTED_PARAMETER_TENSORS, 434)
        with self.assertRaisesRegex(ValueError, "434"):
            module._qwen_parts(model, name="Synthetic", trainable=True)
        with trainer_fixture(model) as (trainer, _):
            self.assertIs(trainer.model, model)

    def test_original_model_forward_and_checkpoint_target_are_not_rebound(self):
        model = tiny()
        forward_function = model.forward.__func__
        before_modules = tuple(model.named_modules())
        with trainer_fixture(model) as (trainer, _):
            unwrap = trainer.accelerator.unwrap_model
            with patch.object(trainer.accelerator, "unwrap_model", wraps=unwrap) as observed:
                trainer.compute_loss(model, collate([rows()[0]]))
            self.assertIs(trainer.model, trainer._checkpointed_head_original_student)
            self.assertIs(model.forward.__func__, forward_function)
            self.assertEqual(tuple(model.named_modules()), before_modules)
            self.assertTrue(any(call.kwargs.get("keep_fp32_wrapper") is True for call in observed.call_args_list))

    def test_explicit_accelerator_context_surrounds_reference_student_and_head_helper(self):
        student, reference = tiny(), tiny(43)
        with trainer_fixture(student, reference, mode="ce_kl") as (trainer, _):
            depth, events = {"value": 0}, []
            original_context, helper = trainer.accelerator.autocast, module.checkpointed_head_loss
            @contextmanager
            def observed_context():
                depth["value"] += 1
                events.append(("enter", depth["value"]))
                try:
                    with original_context():
                        yield
                finally:
                    events.append(("exit", depth["value"]))
                    depth["value"] -= 1
            def observe(label):
                return lambda *_: events.append((label, depth["value"]))
            def observed_helper(*args, **kwargs):
                events.append(("helper", depth["value"]))
                return helper(*args, **kwargs)
            handles = [reference.model.register_forward_pre_hook(observe("reference")),
                student.model.register_forward_pre_hook(observe("student")),
                student.lm_head.register_forward_pre_hook(observe("student_head")),
                reference.lm_head.register_forward_pre_hook(observe("reference_head"))]
            try:
                with patch.object(trainer.accelerator, "autocast", side_effect=observed_context), \
                     patch.object(module, "checkpointed_head_loss", side_effect=observed_helper):
                    trainer.compute_loss(student, collate([rows()[1]]))
            finally:
                for handle in handles:
                    handle.remove()
            for label in ("reference", "student", "helper", "student_head", "reference_head"):
                self.assertTrue(any(name == label for name, _ in events))
                self.assertTrue(all(value == 1 for name, value in events if name == label))
            self.assertEqual(depth["value"], 0)
            self.assertEqual(trainer.last_execution_metadata["explicit_accelerator_autocast"],
                {"device_type": "cpu", "enabled": False, "dtype": None})

    def test_sole_shift_preserves_eos_tail_and_omits_root_full_logits_forward(self):
        model = tiny()
        batch = collate([rows()[1]])
        original = {name: value.clone() for name, value in batch.items()}
        calls, helper = [], module.checkpointed_head_loss
        def capture(hidden, labels, **kwargs):
            calls.append((tuple(hidden.shape), labels.detach().clone()))
            return helper(hidden, labels, **kwargs)
        with trainer_fixture(model) as (trainer, _), patch.object(model, "forward", side_effect=AssertionError("Root logits path used")), \
             patch.object(module, "checkpointed_head_loss", side_effect=capture):
            trainer.compute_loss(model, batch)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0][:2], (1, batch["labels"].shape[1] - 1))
            self.assertTrue(torch.equal(calls[0][1], original["labels"][:, 1:]))
            self.assertEqual(trainer.last_metrics["eos_tokens"], 1)
            self.assertEqual(trainer.last_metrics["supervised_tokens"], 4)
            for name, value in original.items():
                self.assertTrue(torch.equal(batch[name], value))

    def test_all_four_modes_match_independent_dense_loss_and_tied_parameter_gradients(self):
        initial, teacher = tiny(), tiny(47)
        batch = collate([rows()[1]])
        for mode in MODES:
            with self.subTest(mode=mode):
                actual, expected = deepcopy(initial), deepcopy(initial)
                reference = None if mode == "sft" else deepcopy(teacher).eval().requires_grad_(False)
                with trainer_fixture(actual, reference, mode=mode) as (trainer, _):
                    loss = trainer.compute_loss(actual, batch)
                    oracle = dense_loss(expected, batch, mode=mode, reference=reference)
                    loss.backward()
                    oracle.backward()
                    torch.testing.assert_close(loss, oracle, atol=2e-6, rtol=2e-6)
                    for (name, value), (_, target) in zip(actual.named_parameters(), expected.named_parameters()):
                        with self.subTest(parameter=name):
                            self.assertEqual(value.grad is None, target.grad is None)
                            if value.grad is not None:
                                torch.testing.assert_close(value.grad, target.grad, atol=3e-6, rtol=3e-5)
                    self.assertIs(actual.lm_head.weight, actual.model.embed_tokens.weight)

    def test_num_items_is_ignored_without_extra_loss_scaling_and_outputs_are_scalar(self):
        with trainer_fixture() as (trainer, _):
            batch = collate([rows()[0]])
            first = trainer.compute_loss(trainer.model, batch)
            second, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True,
                num_items_in_batch=torch.tensor(999999))
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertEqual(set(outputs), {"loss", "metrics"})
            self.assertEqual(second.numel(), 1)
            self.assertTrue(second.requires_grad)
            self.assertFalse(outputs["loss"].requires_grad)
            self.assertNotIn("logits", outputs)
            self.assertTrue(all(type(value) in (int, float) for value in outputs["metrics"].values()))
            self.assertFalse(trainer.model_accepts_loss_kwargs)

    def test_reference_forward_is_no_grad_normal_detached_and_unchanged(self):
        student, reference = tiny(), tiny(53)
        before, observations = state(reference), []
        def reference_hook(_, __, output):
            value = output.last_hidden_state
            observations.append((torch.is_grad_enabled(), value.requires_grad, value.grad_fn, torch.is_inference(value)))
        with trainer_fixture(student, reference, mode="ce_kl") as (trainer, _):
            handle = reference.model.register_forward_hook(reference_hook)
            try:
                trainer.compute_loss(student, collate([rows()[1]])).backward()
            finally:
                handle.remove()
            self.assertEqual(observations, [(False, False, None, False)])
            self.assertFalse(reference.training)
            self.assertTrue(all(not value.requires_grad and value.grad is None for value in reference.parameters()))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertFalse(any(value is reference for value in student.modules()))

    def test_optimizer_contains_student_once_and_fp32_moments_exclude_reference(self):
        student, reference = tiny(), tiny(59)
        with trainer_fixture(student, reference, mode="ce_kl") as (trainer, _):
            trainer.create_optimizer()
            optimizer = check_optimizer(student, trainer.optimizer, reference)
            trainer.compute_loss(student, collate([rows()[1]])).backward()
            finite_update_guard(student, optimizer, reference)
            optimizer.step()
            check_optimizer(student, optimizer, reference)
            self.assertEqual({id(value) for group in optimizer.param_groups for value in group["params"]},
                {id(value) for value in student.parameters()})
            self.assertFalse({id(value) for value in reference.parameters()} &
                {id(value) for group in optimizer.param_groups for value in group["params"]})
            self.assertTrue(optimizer.state)
            self.assertTrue(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for item in optimizer.state.values() for name, value in item.items() if name in {"exp_avg", "exp_avg_sq"}))
            moments = next(iter(optimizer.state.values()))
            moments["exp_avg"] = moments["exp_avg"].bfloat16()
            with self.assertRaisesRegex(ValueError, "finite FP32"):
                finite_update_guard(student, optimizer, reference)
            moments["exp_avg"] = moments["exp_avg"].float()
            moments["exp_avg"].flatten()[0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite FP32"):
                finite_update_guard(student, optimizer, reference)

    def test_standard_save_reload_keeps_tied_student_without_reference_checkpoint_state(self):
        student, reference = tiny(), tiny(61)
        with trainer_fixture(student, reference, mode="ce_kl") as (trainer, directory):
            before_keys = set(student.state_dict())
            trainer.create_optimizer()
            trainer.compute_loss(student, collate([rows()[0]])).backward()
            trainer.optimizer.step()
            saved = directory / "student-only"
            trainer.save_model(str(saved))
            from safetensors import safe_open
            with safe_open(str(saved / "model.safetensors"), framework="pt", device="cpu") as handle:
                self.assertTrue(set(handle.keys()) <= before_keys)
                self.assertFalse(any("reference" in name for name in handle.keys()))
            loaded = module.Qwen2ForCausalLM.from_pretrained(saved, local_files_only=True,
                dtype=torch.float32, attn_implementation="sdpa")
            self.assertEqual(set(loaded.state_dict()), before_keys)
            self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, student.state_dict()[name], atol=0, rtol=0)
            self.assertIs(trainer.model, student)

    def test_fp32_leaf_and_full_trainability_guards_reject_lower_precision_or_frozen_student(self):
        for kind in ("bf16", "frozen"):
            model = tiny()
            count = len(list(model.parameters()))
            if kind == "bf16":
                model.bfloat16()
            else:
                next(model.parameters()).requires_grad_(False)
            with self.subTest(kind=kind), patch.object(module, "EXPECTED_PARAMETER_TENSORS", count), self.assertRaisesRegex(ValueError, "FP32"):
                module._qwen_parts(model, name="Student", trainable=True)

    def test_original_bias_free_tied_linear_boundary_rejects_untie_bias_and_subclass(self):
        class WrappedLinear(torch.nn.Linear):
            pass
        for kind in ("untied", "bias", "subclass"):
            model = tiny()
            if kind == "untied":
                model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            elif kind == "bias":
                model.lm_head.bias = torch.nn.Parameter(torch.zeros(model.config.vocab_size))
            else:
                replacement = WrappedLinear(model.config.hidden_size, model.config.vocab_size, bias=False)
                replacement.weight = model.model.embed_tokens.weight
                model.lm_head = replacement
            with self.subTest(kind=kind), patch.object(module, "EXPECTED_PARAMETER_TENSORS", len(list(model.parameters()))), self.assertRaises(ValueError):
                module._qwen_parts(model, name="Student", trainable=True)

    def test_non_original_root_or_backbone_class_is_refused(self):
        model = tiny()
        with self.assertRaisesRegex(ValueError, "original unwrapped"):
            module._qwen_parts(torch.nn.Sequential(model), name="Student", trainable=True)
        model.model = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "original unwrapped"):
            module._qwen_parts(model, name="Student", trainable=True)

    def test_cache_enabled_or_non_sdpa_configuration_is_refused(self):
        for name, value in (("use_cache", True), ("_attn_implementation", "eager")):
            model = tiny()
            setattr(model.config, name, value)
            with self.subTest(name=name), patch.object(module, "EXPECTED_PARAMETER_TENSORS", len(list(model.parameters()))), self.assertRaisesRegex(ValueError, "SDPA"):
                module._qwen_parts(model, name="Student", trainable=True)

    def test_compile_is_rejected_before_trainer_or_model_mutation(self):
        model = tiny()
        with tempfile.TemporaryDirectory() as temporary:
            settings = arguments(temporary)
            settings.torch_compile = True
            with self.assertRaisesRegex(ValueError, "Compiled"):
                OrderedTrainer(model=model, args=settings, objective="sft", eos_token_id=EOS, terminal_suffix=SUFFIX)
        self.assertTrue(all(value.requires_grad for value in model.parameters()))

    def test_distributed_fsdp_deepspeed_and_nonidentity_unwrap_are_refused(self):
        with trainer_fixture() as (trainer, _):
            original = trainer.accelerator
            class AcceleratorProxy:
                def __init__(self, **changes):
                    self.num_processes = changes.get("num_processes", 1)
                    self.state = SimpleNamespace(distributed_type=changes.get("distributed_type", "NO"))
                    self.mixed_precision = changes.get("mixed_precision", "no")
                    self.foreign = changes.get("foreign", False)
                def unwrap_model(self, model, keep_fp32_wrapper=True):
                    return object() if self.foreign else model
            for change in ({"num_processes": 2}, {"distributed_type": "MULTI_CPU"}, {"foreign": True}, {"mixed_precision": "bf16"}):
                with self.subTest(change=change), patch.object(trainer, "accelerator", AcceleratorProxy(**change)), self.assertRaises(ValueError):
                    trainer._original_parts(trainer.model)
            self.assertIs(trainer.accelerator, original)
            for name in ("is_deepspeed_enabled", "is_fsdp_enabled"):
                with patch.object(trainer, name, True), self.assertRaises(ValueError):
                    trainer._original_parts(trainer.model)

    def test_adapter_dispatch_compile_and_offload_markers_are_refused(self):
        for name, value in (("peft_config", {}), ("hf_device_map", {"": "cpu"}), ("_orig_mod", object()), ("_hf_hook", object())):
            model = tiny()
            setattr(model, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "adapter|compiled|dispatched|offloaded"):
                module._qwen_parts(model, name="Student", trainable=True)

    def test_reference_overlap_wrong_class_and_non_fp32_storage_are_refused(self):
        student = tiny()
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, "EXPECTED_PARAMETER_TENSORS", len(list(student.parameters()))):
            for reference in (student, tiny(67).bfloat16(), torch.nn.Linear(8, 8)):
                with self.subTest(reference=type(reference).__name__), self.assertRaises(ValueError):
                    OrderedTrainer(model=student, args=arguments(temporary), reference_model=reference,
                        objective="ce_kl", eos_token_id=EOS, terminal_suffix=SUFFIX)
            self.assertTrue(all(value.requires_grad for value in student.parameters()))

    def test_invalid_masks_zero_active_and_missing_eos_fail_before_backbone(self):
        with trainer_fixture() as (trainer, _):
            calls = []
            handle = trainer.model.model.register_forward_pre_hook(lambda *_: calls.append(True))
            try:
                for kind in ("zero", "hole", "missing_tail", "left_padding"):
                    batch = collate([rows()[1]])
                    if kind == "zero": batch["labels"].fill_(-100)
                    if kind == "hole": batch["labels"][0, -2] = -100
                    if kind == "missing_tail": batch["labels"][0, -1] = batch["input_ids"][0, -1] = 3
                    if kind == "left_padding": batch["attention_mask"][0, 0] = 0
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        trainer.compute_loss(trainer.model, batch)
                self.assertEqual(calls, [])
            finally:
                handle.remove()

    def test_scalar_metric_boundary_rejects_graph_vector_and_nonfinite_values(self):
        captured, helper = {}, module.checkpointed_head_loss
        def capture(*args, **kwargs):
            result = helper(*args, **kwargs)
            captured.update(result[1])
            return result
        batch = collate([rows()[1]])
        with trainer_fixture() as (trainer, _), patch.object(module, "checkpointed_head_loss", side_effect=capture):
            trainer.compute_loss(trainer.model, batch)
            _, _, response = module.response_batch(batch, EOS, SUFFIX)
            self.assertTrue(all(type(value) in (int, float) for value in trainer.last_metrics.values()))
            for value in (torch.tensor(1., requires_grad=True), torch.ones(2), torch.tensor(float("nan"))):
                changed = dict(captured)
                changed["ce"] = value
                with self.assertRaises(ValueError):
                    module._scalar_metrics(changed, response)

    def test_forward_exception_removes_only_temporary_hooks_without_masking_original_error(self):
        with trainer_fixture() as (trainer, _):
            def original_hook(*_):
                raise RuntimeError("synthetic original head failure")
            original = trainer.model.lm_head.register_forward_hook(original_hook)
            before = tuple(trainer.model.lm_head._forward_hooks)
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic original head failure"):
                    trainer.compute_loss(trainer.model, collate([rows()[1]]))
                self.assertEqual(tuple(trainer.model.lm_head._forward_hooks), before)
                self.assertFalse(trainer.last_execution_metadata["complete"])
                self.assertTrue(no_tensor_tree(trainer.last_execution_metadata))
            finally:
                original.remove()

    def test_forward_metadata_is_bounded_detached_and_never_recompute_admission(self):
        with trainer_fixture(chunk_tokens=2) as (trainer, _):
            loss = trainer.compute_loss(trainer.model, collate([rows()[1]]))
            before = deepcopy(trainer.last_execution_metadata)
            loss.backward()
            self.assertEqual(trainer.last_execution_metadata, before)
            self.assertTrue(no_tensor_tree(before))
            self.assertEqual(before["scope"], module.OBSERVATION_SCOPE)
            self.assertFalse(before["backward_recompute_or_native_dtype_admission_established"])
            self.assertTrue(before["original_student_is_trainer_model"])
            self.assertEqual(before["causal_shifts"], 1)
            self.assertFalse(before["full_sequence_logits_requested"])
            self.assertLessEqual(before["student_head"]["maximum_projection_rows"], 2)
            self.assertEqual(before["student_head"]["active_rows_projected"], 4)
            self.assertEqual(len(trainer.model.lm_head._forward_hooks), 0)

    def _tail_oracle(self, mode):
        initial, reference = tiny(71), None
        actual, expected = deepcopy(initial), deepcopy(initial)
        if mode != "sft":
            reference = tiny(73).eval().requires_grad_(False)
        reference_before = None if reference is None else state(reference)
        data, rate = rows(9), .025
        oracle_optimizer = torch.optim.SGD(expected.parameters(), lr=rate)
        for group in (data[:8], data[8:]):
            oracle_optimizer.zero_grad()
            for row in group:
                (dense_loss(expected, collate([row]), mode=mode, reference=reference) / len(group)).backward()
            oracle_optimizer.step()
        optimizer = torch.optim.SGD(actual.parameters(), lr=rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        observed = []
        hook = actual.model.register_forward_pre_hook(lambda _, __, kwargs: observed.append(kwargs["input_ids"].detach().clone()), with_kwargs=True)
        try:
            with trainer_fixture(actual, reference, mode=mode, data=data, optimizers=(optimizer, scheduler),
                    changes={"gradient_accumulation_steps": 8, "max_steps": 2}) as (trainer, _):
                trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                self.assertEqual(trainer.training_forward_examples, 9)
                self.assertEqual(len(observed), 9)
                self.assertEqual([value.tolist()[0] for value in observed], [row["input_ids"] for row in data])
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                self.assertIs(trainer.model, actual)
                for (name, value), (_, target) in zip(actual.named_parameters(), expected.named_parameters()):
                    with self.subTest(mode=mode, parameter=name):
                        torch.testing.assert_close(value, target, atol=4e-6, rtol=4e-5)
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertTrue(bool(torch.isfinite(value).all()))
                with self.assertRaisesRegex(ValueError, "fresh trainer"):
                    trainer.train()
        finally:
            hook.remove()
        if reference is not None:
            self.assertTrue(all(value.grad is None and not value.requires_grad for value in reference.parameters()))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, reference_before[name], atol=0, rtol=0)

    def test_real_hf_ga8_nine_examples_and_partial_tail_match_independent_sft_updates(self):
        self._tail_oracle("sft")

    def test_real_hf_ga8_nine_examples_and_partial_tail_match_independent_cekl_updates(self):
        self._tail_oracle("ce_kl")

    def test_resume_repeated_train_and_hf_evaluation_prediction_remain_explicitly_unsupported(self):
        with trainer_fixture() as (trainer, _):
            with self.assertRaisesRegex(ValueError, "resume"):
                trainer.train(resume_from_checkpoint="synthetic-checkpoint")
            self.assertFalse(trainer._prototype_train_started)
            trainer._prototype_train_started = True
            with self.assertRaisesRegex(ValueError, "fresh trainer"):
                trainer.train()
            with self.assertRaises(NotImplementedError):
                trainer.evaluate()
            with self.assertRaises(NotImplementedError):
                trainer.predict([])


if __name__ == "__main__":
    unittest.main()
