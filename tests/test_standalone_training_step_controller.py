"""Prospective32 actual-tiny-CPU fixtures, not native-model admission.

The frozen paired-test MODULE is reused only for construction and its independent
dense formula. Its TestCase class is never imported into this discovery module.
Future QA must use unittest discovery with tests/ as top_level_dir, as the frozen
QA stages do. The explicit small Parameter count, CPU hooks and tiny clones are
fixture seams, not original434/CUDA/BF16/recompute, shadow, long-memory or pilot
proof. No test/framework/model workload was run while authoring this source.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_paired_training_step_controller as fixtures
from extension import standalone_training_step_controller as module


class ThreeWitnessCPUObserver(fixtures.CPUObserver):
    """New CPU fixture validator; the frozen nine-witness observer is untouched."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clean_starts = []

    @contextmanager
    def phase(self, name, *, presentation_index):
        if name == "backward":
            if any(value.grad is not None for value in self.student.parameters()):
                raise AssertionError("A standalone witness inherited accumulated gradients")
            self.clean_starts.append(presentation_index)
        with super().phase(name, presentation_index=presentation_index):
            yield self

    def validate(self, *, require_backward_recompute, require_all_layers, require_reference):
        names = set(dict(self.student.named_parameters()))
        if self.stack or set(self.gradients) != names or any(count != 3 for count in self.gradients.values()):
            raise AssertionError("Every actual tiny Parameter needs exactly three backwards")
        if self.clean_starts != [0, 1, 2]:
            raise AssertionError("Every standalone witness must begin with cleared gradients")
        if not require_all_layers or not all(self.layers.get((index, "forward"), 0) >= 3
                for index in range(len(self.student.model.layers))):
            raise AssertionError("Missing actual tiny decoder forward coverage")
        if require_backward_recompute and (not self.calls["backward_head"] or not any(
                self.layers.get((index, "backward"), 0) for index in range(len(self.student.model.layers)))):
            raise AssertionError("Missing actual tiny CPU candidate recomputation")
        if require_reference and self.calls["reference"] < 3:
            raise AssertionError("Anchored objective omitted actual frozen-reference forwards")
        if not require_reference and self.calls["reference"]:
            raise AssertionError("SFT unexpectedly forwarded its resident reference")
        return {**self.snapshot(), "standalone_backwards": 3, "native_admission_established": False,
                "required_all_layers_checked": True, "clean_starts": list(self.clean_starts)}


def group_metadata(trainer):
    lookup = {id(value): name for name, value in trainer.model.named_parameters()}
    return [{"names": [lookup[id(value)] for value in group["params"]],
             "lr": group["lr"], "weight_decay": group["weight_decay"],
             "betas": tuple(group["betas"]), "eps": group["eps"]}
            for group in trainer.optimizer.param_groups]


class StandaloneFixtureHooks:
    """Independent mean1/autograd/once-clip and unchanged-original tiny oracles."""
    def __init__(self, value, mode):
        self.value, self.mode = value, mode
        self.trainers, self.reference = value["trainers"], value["reference"]
        self.initial = fixtures.clone_parameters(self.reference)
        self.manual = deepcopy(self.reference).train().requires_grad_(True)
        self.manual.gradient_checkpointing_disable()
        self.scheduler_initial, self.groups_initial = {}, {}
        self.events, self.callback_calls, self.zero_calls, self.presentations = [], [], [], []
        self.raw, self.clipped, self.norms = {}, {}, {}
        self.prepared_action = None

    def receipt(self, stage, *, count=False):
        return {"complete": True, "passed": True, "stage": stage, "cpu_fixture_only": True,
                "native_provenance_established": False,
                **({"compared_tensor_count": len(self.initial)} if count else {})}

    def verify_invariants(self, *, stage, **_):
        self.events.append(stage)
        if self.reference.training or any(value.requires_grad or value.grad is not None for value in self.reference.parameters()):
            raise AssertionError("Frozen reference policy changed")
        fixtures.close_maps(dict(self.reference.named_parameters()), self.initial)
        storages = []
        for branch, trainer in self.trainers.items():
            if trainer.model is not self.value["models"][branch] or trainer.model.lm_head.weight is not trainer.model.model.embed_tokens.weight:
                raise AssertionError("The original tiny model or its tie was rebound")
            if any(value.dtype != torch.float32 or not value.requires_grad for value in trainer.model.parameters()):
                raise AssertionError("The original trainable FP32 storage policy changed")
            raw = fixtures.preserving.check_optimizer(trainer.model, trainer.optimizer, self.reference)
            if raw.state or trainer.state.global_step != 0:
                raise AssertionError("Standalone diagnostics must never initialize Adam or update global_step")
            if any(group["lr"] != 0. for group in raw.param_groups):
                raise AssertionError("The ordinary fresh scheduler must retain its initial LR0")
            storages.append({value.untyped_storage().data_ptr() for value in trainer.model.parameters()})
            if stage == "standalone_prepared":
                self.scheduler_initial[branch] = deepcopy(trainer.lr_scheduler.state_dict())
                self.groups_initial[branch] = group_metadata(trainer)
                callback = trainer.callback_handler.callbacks[-1]
                original_clip, original_zero = callback.on_pre_optimizer_step, trainer.model.zero_grad

                def clip(*args, _original=original_clip, _branch=branch, _trainer=trainer, **kwargs):
                    index = _trainer.paired_presentation_index
                    self.callback_calls.append((_branch, index))
                    self.events.append("clip:" + _branch + ":" + str(index))
                    return _original(*args, **kwargs)

                def zero(*args, _original=original_zero, _branch=branch, _trainer=trainer, **kwargs):
                    answer = _original(*args, **kwargs)
                    if any(value.grad is not None for value in _trainer.model.parameters()):
                        raise AssertionError("Original zero_grad failed to clear every Parameter to None")
                    self.zero_calls.append((_branch, _trainer.paired_presentation_index))
                    self.events.append("zero:" + _branch + ":" + str(_trainer.paired_presentation_index))
                    return answer

                self.value["stack"].enter_context(patch.object(callback, "on_pre_optimizer_step", side_effect=clip))
                self.value["stack"].enter_context(patch.object(trainer.model, "zero_grad", side_effect=zero))
            elif trainer.lr_scheduler.state_dict() != self.scheduler_initial[branch] or group_metadata(trainer) != self.groups_initial[branch]:
                raise AssertionError("Scheduler/default groups changed without an admitted update")
        if storages[0] & storages[1]:
            raise AssertionError("Students share actual storage")
        reference_storage = {value.untyped_storage().data_ptr() for value in self.reference.parameters()}
        if any(values & reference_storage for values in storages):
            raise AssertionError("The resident reference shares a student Parameter")
        if stage == "standalone_prepared" and self.prepared_action is not None:
            self.prepared_action(self)
        if stage == "standalone_final" and any(value.grad is not None
                for trainer in self.trainers.values() for value in trainer.model.parameters()):
            raise AssertionError("The third standalone backward retained a gradient")
        return self.receipt(stage)

    def verify_identity(self, *, stage, **_):
        self.events.append(stage)
        for model in [self.reference] + [trainer.model for trainer in self.trainers.values()]:
            for name, value in model.named_parameters():
                if not torch.equal(value.detach(), self.initial[name]):
                    raise AssertionError("A standalone phase changed an original FP32 weight")
        return self.receipt(stage, count=True)

    def compare_scalars(self, *, stage, dense, candidate, **_):
        self.events.append(stage)
        index = int(stage.rsplit("_", 1)[1])
        if stage.startswith("standalone_presentation_"):
            batch = self.value["batches"][index]
            with torch.no_grad():
                expected = float(fixtures.dense_loss(self.reference, batch, self.reference, self.mode))
            active = int((batch["labels"] != -100).sum())
            eos = int((batch["labels"] == fixtures.EOS).sum())
            for actual in (dense, candidate):
                if actual["group_size"] != 1 or not math.isclose(actual["group1_loss"], expected, rel_tol=3e-5, abs_tol=2e-6):
                    raise AssertionError("Actual inherited training_step must divide by1, not8 or3")
                if not math.isclose(actual["metrics"]["loss"], expected, rel_tol=3e-5, abs_tol=2e-6):
                    raise AssertionError("Standalone compute_loss changed the independent objective")
                if actual["metrics"]["supervised_tokens"] != active or actual["metrics"]["eos_tokens"] != eos:
                    raise AssertionError("Standalone consumption omitted an original target/EOS")
            self.presentations.append({"index": index, "active": active, "eos": eos, "expected": expected,
                                       "dense": dense["group1_loss"], "candidate": candidate["group1_loss"]})
        elif stage.startswith("standalone_clip_norm_"):
            for actual in (dense, candidate):
                if not math.isclose(actual["clip_norm"], self.norms[index], rel_tol=5e-5, abs_tol=3e-6):
                    raise AssertionError("Owned clip norm disagrees with independent dense gradients")
        else:
            raise AssertionError("Missing independent scalar oracle for requested stage")
        return self.receipt(stage)

    def compare_tensors(self, *, stage, **_):
        self.events.append(stage)
        index = int(stage.rsplit("_", 1)[1])
        if stage.startswith("standalone_raw_gradients_"):
            self.manual.zero_grad(set_to_none=True)
            fixtures.dense_loss(self.manual, self.value["batches"][index], self.reference, self.mode).backward()
            expected = {name: value.grad for name, value in self.manual.named_parameters()}
            target = self.raw
        elif stage.startswith("standalone_clipped_gradients_"):
            self.norms[index] = float(torch.nn.utils.clip_grad_norm_(self.manual.parameters(), 1., error_if_nonfinite=True))
            expected = {name: value.grad for name, value in self.manual.named_parameters()}
            target = self.clipped
        else:
            raise AssertionError("Missing independent all-Parameter gradient oracle")
        for branch, trainer in self.trainers.items():
            actual = {name: value.grad for name, value in trainer.model.named_parameters()}
            fixtures.close_maps(actual, expected, gradients=True)
            target[(branch, index)] = {name: value.detach().clone() for name, value in actual.items()}
        return self.receipt(stage, count=True)

    def api(self):
        return module.StandaloneStepHooks(self.verify_invariants, self.verify_identity,
                                          self.compare_scalars, self.compare_tensors)


@contextmanager
def standalone_fixture(mode="sft", *, stochastic=False):
    # Importing the old module never places its TestCase class in this module's
    # discovery namespace. Its constructed nine-row driver is never executed.
    with fixtures.pair_fixture(mode) as value:
        value["batches"] = value["batches"][:3]
        value["observers"] = {branch: ThreeWitnessCPUObserver(value["models"][branch], value["reference"],
            branch, stochastic=stochastic) for branch in module.BRANCHES}
        for branch, trainer in value["trainers"].items():
            trainer.paired_observer = value["observers"][branch]
        value["standalone_hooks"] = StandaloneFixtureHooks(value, mode)
        yield value


def run(value, *, hooks=None, observers=None, expected=None):
    return module.run_standalone_training_steps(value["trainers"]["dense"], value["trainers"]["candidate"],
        value["batches"], reference_model=value["reference"],
        hooks=value["standalone_hooks"].api() if hooks is None else hooks,
        observers=value["observers"] if observers is None else observers,
        expected_parameters=value["count"] if expected is None else expected)


class StandaloneTrainingStepControllerTests(unittest.TestCase):
    def failure(self, value, **kwargs):
        with self.assertRaises(module.StandaloneControllerFailure) as caught:
            run(value, **kwargs)
        result = caught.exception.partial_result
        self.assertFalse(result["complete"])
        self.assertTrue(result["caller_must_preserve_private_partial_result_and_abort"])
        self.assertFalse(result["training_or_pilot_admitted"])
        self.assertEqual((result["controller_optimizer_step_calls"], result["controller_scheduler_step_calls"]), (0, 0))
        json.dumps(result, allow_nan=False)
        return result, caught.exception

    def assert_closed(self, value):
        for observer in value["observers"].values():
            if observer.entered:
                self.assertTrue(observer.exited)
                self.assertFalse(observer.handles)
                self.assertFalse(observer.stack)

    def test_01_api_source_parent_and_discovery_do_not_replay_the_imported_testcase(self):
        self.assertEqual(module.PAIRED_SOURCE_SHA256, "9e446719e34f2370a00967066813b28acb6bf0a072cfade473eda3ef5e59e726")
        self.assertEqual(module.ROLES, ("short0", "short1", "short2"))
        self.assertEqual(tuple(module.StandaloneStepHooks.__dataclass_fields__),
                         ("verify_invariants", "verify_identity", "compare_scalars", "compare_tensors"))
        discovered = [name for name, value in globals().items()
                      if inspect.isclass(value) and issubclass(value, unittest.TestCase)]
        self.assertEqual(discovered, ["StandaloneTrainingStepControllerTests"])
        self.assertEqual(unittest.defaultTestLoader.loadTestsFromTestCase(type(self)).countTestCases(), 32)

    def test_02_tiny_count_is_explicit_and_cannot_satisfy_original434(self):
        with standalone_fixture() as value:
            self.assertNotEqual(value["count"], 434)
            result, _ = self.failure(value, expected=434)
            self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})
            self.assertFalse(result["native_provenance_established"])
        self.assertEqual(fixtures.head.EXPECTED_PARAMETER_TENSORS, 434)

    def test_03_actual_inherited_training_step_and_original_model_ownership_are_preserved(self):
        with standalone_fixture() as value:
            for trainer in value["trainers"].values():
                self.assertIs(trainer.training_step.__func__, fixtures.Trainer.training_step)
            run(value)
            for branch, trainer in value["trainers"].items():
                self.assertIs(trainer.model, value["models"][branch])
                self.assertIs(trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=True), trainer.model)
                self.assertEqual(trainer.accelerator._models, [trainer.model])
                self.assertFalse(trainer.accelerator._schedulers)
                self.assertEqual(trainer.accelerator.gradient_accumulation_steps, 1)

    def test_04_success_counts_only_three_group1_backwards_and_denies_native_or_pilot_claims(self):
        with standalone_fixture() as value:
            result = run(value)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_backwards"], {"dense": 3, "candidate": 3})
            self.assertEqual(result["training_step_attempts"], {"dense": 3, "candidate": 3})
            self.assertEqual(result["finite_clip_boundaries"], {"dense": 3, "candidate": 3})
            self.assertEqual(result["actual_group_sizes"], [1, 1, 1])
            self.assertTrue(result["fixture_count_is_not_native_proof"])
            for key in ("trainer_train_called", "full_trainer_train_loop_established", "unique_row_driver_established",
                        "long_context_memory_established", "native_provenance_established", "training_or_pilot_admitted"):
                self.assertFalse(result[key], key)
            self.assert_closed(value)

    def test_05_each_raw_gradient_matches_independent_mean1_autograd_without_accumulation(self):
        with standalone_fixture() as value:
            run(value)
            hooks = value["standalone_hooks"]
            self.assertEqual(set(hooks.raw), {(branch, index) for branch in module.BRANCHES for index in range(3)})
            for index in range(3):
                fixtures.close_maps(hooks.raw[("dense", index)], hooks.raw[("candidate", index)], gradients=True)
            self.assertTrue(any(not torch.equal(hooks.raw[("dense", 0)][name], hooks.raw[("dense", 1)][name])
                                for name in hooks.initial))
            # A divided-by8 gradient cannot pass merely by matching branches.
            norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in hooks.raw[("dense", 0)].values()]))
            self.assertGreater(float(norm), 0.)
            self.assertFalse(torch.isclose(norm, norm / 8, atol=1e-6, rtol=1e-5))

    def test_06_all_anchored_modes_match_independent_raw_clipped_and_scalar_oracles(self):
        for mode in ("ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode), standalone_fixture(mode) as value:
                run(value)
                hooks = value["standalone_hooks"]
                self.assertEqual(set(hooks.norms), {0, 1, 2})
                self.assertEqual(len(hooks.clipped), 6)
                self.assertTrue(all(observer.calls["reference"] >= 3 for observer in value["observers"].values()))
                self.assertTrue(all(parameter.grad is None for parameter in value["reference"].parameters()))

    def test_07_sole_causal_shift_and_all_original_eos_tail_targets_keep_mean1_loss(self):
        with standalone_fixture() as value:
            run(value)
            records = value["standalone_hooks"].presentations
            self.assertEqual([row["active"] for row in records], [2, 4, 3])
            self.assertEqual([row["eos"] for row in records], [1, 1, 1])
            self.assertEqual([row["index"] for row in records], [0, 1, 2])
            for row in records:
                self.assertTrue(math.isclose(row["dense"], row["expected"], abs_tol=2e-6, rel_tol=3e-5))
                self.assertFalse(math.isclose(row["dense"], row["expected"] / 8, abs_tol=1e-5))
                self.assertFalse(math.isclose(row["candidate"], row["expected"] / 3, abs_tol=1e-5))

    def test_08_original_weights_reference_empty_adam_and_scheduler_remain_exactly_unchanged(self):
        with standalone_fixture() as value:
            run(value)
            hooks = value["standalone_hooks"]
            for branch, trainer in value["trainers"].items():
                raw = fixtures.preserving._optimizer(trainer.optimizer)
                self.assertIs(type(raw), torch.optim.AdamW)
                self.assertFalse(raw.state)
                self.assertEqual(trainer.state.global_step, 0)
                self.assertEqual(trainer.lr_scheduler.state_dict(), hooks.scheduler_initial[branch])
                self.assertEqual(group_metadata(trainer), hooks.groups_initial[branch])
                self.assertIs(trainer.lr_scheduler.optimizer, raw)
                self.assertEqual([group["lr"] for group in raw.param_groups], [0., 0.])
                for name, parameter in trainer.model.named_parameters():
                    self.assertTrue(torch.equal(parameter, hooks.initial[name]))
                    self.assertEqual(parameter.dtype, torch.float32)
            self.assertFalse(value["reference"].training)
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in value["reference"].parameters()))

    def test_09_actual_optimizer_and_scheduler_step_entrypoints_are_never_called_after_preparation(self):
        with standalone_fixture() as value:
            spies = {}
            def prepared(hooks):
                # LambdaLR installs an instance optimizer.step wrapper, so spy
                # that actual entrypoint rather than only the AdamW class.
                spies["optimizer"] = [value["stack"].enter_context(patch.object(
                    fixtures.preserving._optimizer(trainer.optimizer), "step",
                    side_effect=AssertionError("Standalone attempted AdamW.step")))
                    for trainer in value["trainers"].values()]
                # Patch the scheduler CLASS after both constructors ran. An
                # instance step attribute would itself pollute state_dict().
                spies["scheduler"] = value["stack"].enter_context(patch.object(torch.optim.lr_scheduler.LambdaLR, "step",
                    autospec=True, side_effect=AssertionError("Standalone attempted scheduler.step")))
            value["standalone_hooks"].prepared_action = prepared
            result = run(value)
            self.assertTrue(result["complete"])
            self.assertEqual([spy.call_count for spy in spies["optimizer"]], [0, 0])
            self.assertEqual(spies["scheduler"].call_count, 0)

    def test_10_raw_comparison_precedes_once_owned_clip_then_clipped_norm_identity_and_zero(self):
        with standalone_fixture() as value:
            run(value)
            hooks = value["standalone_hooks"]
            self.assertEqual(hooks.callback_calls, [(branch, index) for index in range(3) for branch in module.BRANCHES])
            for index in range(3):
                names = ["standalone_raw_gradients_" + str(index), "clip:dense:" + str(index),
                    "clip:candidate:" + str(index), "standalone_clipped_gradients_" + str(index),
                    "standalone_clip_norm_" + str(index), "standalone_unchanged_original_" + str(index),
                    "zero:dense:" + str(index), "zero:candidate:" + str(index)]
                positions = [hooks.events.index(name) for name in names]
                self.assertEqual(positions, sorted(positions))

    def test_11_all_gradients_clear_to_none_between_witnesses_and_after_the_third(self):
        with standalone_fixture() as value:
            run(value)
            self.assertEqual(value["standalone_hooks"].zero_calls,
                             [(branch, index) for index in range(3) for branch in module.BRANCHES])
            for branch, trainer in value["trainers"].items():
                self.assertEqual(value["observers"][branch].clean_starts, [0, 1, 2])
                self.assertEqual(trainer.training_forward_examples, 3)
                self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))

    def test_12_no_train_loop_dataloader_prefetch_or_more_than_three_presentations_occurs(self):
        with standalone_fixture() as value:
            for trainer in value["trainers"].values():
                value["stack"].enter_context(patch.object(trainer, "train", side_effect=AssertionError("No train loop")))
                value["stack"].enter_context(patch.object(trainer, "get_train_dataloader", side_effect=AssertionError("No prefetch")))
            result = run(value)
            beginnings = [event for event in result["events"] if event["event"] == "standalone_training_step_begin"]
            self.assertEqual([(event["branch"], event["presentation_index"]) for event in beginnings],
                             [(branch, index) for index in range(3) for branch in module.BRANCHES])
            self.assertTrue(all(event["group_size"] == 1 for event in beginnings))

    def test_13_python_numpy_torch_rng_progresses_once_for_each_paired_witness(self):
        with standalone_fixture(stochastic=True) as value:
            python = random.Random(); python.setstate(random.getstate())
            numpy = np.random.RandomState(); numpy.set_state(np.random.get_state())
            generator = torch.Generator(); generator.set_state(torch.get_rng_state())
            expected = [[(python.random(), float(numpy.random_sample()), float(torch.rand((), generator=generator)))]
                        for _ in range(3)]
            run(value)
            for observer in value["observers"].values():
                self.assertEqual(observer.draws, expected)
            self.assertEqual(random.getstate(), python.getstate())
            self.assertEqual(np.random.get_state()[1].tolist(), numpy.get_state()[1].tolist())
            self.assertEqual(np.random.get_state()[2:], numpy.get_state()[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), generator.get_state()))

    def test_14_rng_divergence_is_a_typed_partial_failure_and_removes_both_hook_sets(self):
        with standalone_fixture(stochastic=True) as value:
            value["observers"]["candidate"].draw_count = 2
            result, error = self.failure(value)
            self.assertEqual(result["completed_backwards"], {"dense": 1, "candidate": 1})
            self.assertEqual(result["finite_clip_boundaries"], {"dense": 0, "candidate": 0})
            self.assertIsInstance(error.__cause__, ValueError)
            self.assert_closed(value)

    def test_15_three_distinct_semantic_witnesses_cannot_be_fabricated_with_row_indices(self):
        with standalone_fixture() as value:
            duplicate = {key: tensor.clone() for key, tensor in value["batches"][0].items()}
            duplicate["row_index"] = torch.tensor([2], dtype=torch.int64)
            value["batches"][2] = duplicate
            result, _ = self.failure(value)
            self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})

    def test_16_exact_three_inputs_are_required_and_no_repetition_or_implicit_padding_is_added(self):
        for size in (0, 2, 4, 9):
            with self.subTest(size=size), standalone_fixture() as value:
                value["batches"] = (value["batches"] * 3)[:size]
                result, _ = self.failure(value)
                self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_17_batch_mutation_after_a_completed_witness_blocks_the_next_backward(self):
        with standalone_fixture() as value:
            original = value["standalone_hooks"].verify_identity
            def corrupt(*, stage, **kwargs):
                result = original(stage=stage, **kwargs)
                if stage == "standalone_unchanged_original_0":
                    value["batches"][1]["input_ids"][0, 0] += 1
                return result
            self.failure(value, hooks=replace(value["standalone_hooks"].api(), verify_identity=corrupt))
            self.assertEqual(value["trainers"]["dense"].training_forward_examples, 1)
            self.assert_closed(value)

    def test_18_batch_schema_dtype_shape_and_short_context_fail_before_forward(self):
        for change in ("float", "shape", "extra", "overlong"):
            with self.subTest(change=change), standalone_fixture() as value:
                batch = value["batches"][0]
                if change == "float": batch["input_ids"] = batch["input_ids"].float()
                elif change == "shape": batch["labels"] = batch["labels"][:, :-1].contiguous()
                elif change == "extra": batch["unbound"] = torch.tensor([1], dtype=torch.int64)
                else:
                    for key in ("input_ids", "labels", "attention_mask"):
                        batch[key] = torch.ones((1, 513), dtype=torch.int64)
                result, _ = self.failure(value)
                self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})

    def test_19_native_response_mask_eos_and_terminal_suffix_guards_are_not_bypassed(self):
        for change in ("no_active", "no_eos", "gap", "wrong_tail"):
            with self.subTest(change=change), standalone_fixture() as value:
                batch = value["batches"][0]
                if change == "no_active": batch["labels"].fill_(-100)
                elif change == "no_eos": batch["input_ids"][0, -2] = batch["labels"][0, -2] = 7
                elif change == "gap": batch["labels"][0, -1] = -100
                else: batch["input_ids"][0, -1] = batch["labels"][0, -1] = 7
                result, _ = self.failure(value)
                self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})
                self.assert_closed(value)

    def test_20_all_four_hooks_and_exact_source_reviewed_observer_bindings_are_mandatory(self):
        with self.assertRaises(ValueError): module.StandaloneStepHooks(None, None, None, None)
        for kind in ("wrong_hooks", "missing_observer", "wrong_binding"):
            with self.subTest(kind=kind), standalone_fixture() as value:
                kwargs = {}
                if kind == "wrong_hooks": kwargs["hooks"] = value["hooks"].api()  # Frozen paired ABI is not standalone ABI.
                elif kind == "missing_observer": kwargs["observers"] = {"dense": value["observers"]["dense"]}
                else: value["trainers"]["candidate"].paired_observer = value["observers"]["dense"]
                result, _ = self.failure(value, **kwargs)
                self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})

    def test_21_hook_receipts_require_exact_stage_true_flags_all_parameters_and_finite_json(self):
        for change in ("stage", "passed", "count", "bool_count", "tensor", "nonfinite"):
            with self.subTest(change=change), standalone_fixture() as value:
                original = value["standalone_hooks"].verify_identity
                def malformed(*, stage, _change=change, **kwargs):
                    answer = original(stage=stage, **kwargs)
                    if _change == "stage": answer["stage"] = "other"
                    elif _change == "passed": answer["passed"] = 1
                    elif _change == "count": answer["compared_tensor_count"] -= 1
                    elif _change == "bool_count": answer["compared_tensor_count"] = True
                    elif _change == "tensor": answer["unbounded_tensor"] = torch.ones(1)
                    else: answer["value"] = math.inf
                    return answer
                result, _ = self.failure(value, hooks=replace(value["standalone_hooks"].api(), verify_identity=malformed))
                self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_22_completed_standalone_trainers_are_consumed_and_cannot_enter_standalone_or_ga9_again(self):
        with standalone_fixture() as value:
            run(value)
            self.failure(value)
            with self.assertRaises(module.paired.PairedControllerFailure):
                module.paired.run_paired_training_steps(value["trainers"]["dense"], value["trainers"]["candidate"],
                    [fixtures.collate([row]) for row in fixtures.examples()], reference_model=value["reference"],
                    hooks=value["hooks"].api(), observers=value["observers"], expected_parameters=value["count"])
            self.assertTrue(all(trainer.training_forward_examples == 3 for trainer in value["trainers"].values()))

    def test_23_partial_preparation_failure_retains_consumed_flags_without_hidden_reinitialization(self):
        with standalone_fixture() as value:
            value["trainers"]["candidate"].beta = .1
            result, _ = self.failure(value)
            self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})
            self.assertTrue(value["trainers"]["dense"]._paired_controller_started)
            self.assertTrue(value["trainers"]["dense"]._standalone_controller_started)
            value["trainers"]["candidate"].beta = .05
            self.failure(value)
            self.assertIsNotNone(value["trainers"]["dense"].optimizer)

    def test_24_nonfinite_raw_gradient_aborts_before_any_owned_clip_or_state_update(self):
        with standalone_fixture() as value:
            original = value["standalone_hooks"].compare_scalars
            def corrupt(*, stage, **kwargs):
                answer = original(stage=stage, **kwargs)
                if stage == "standalone_presentation_0":
                    next(value["trainers"]["dense"].model.parameters()).grad.flatten()[0] = math.nan
                return answer
            result, _ = self.failure(value, hooks=replace(value["standalone_hooks"].api(), compare_scalars=corrupt))
            self.assertEqual(result["completed_backwards"], {"dense": 1, "candidate": 1})
            self.assertFalse(value["standalone_hooks"].callback_calls)
            self.assertTrue(all(not fixtures.preserving._optimizer(trainer.optimizer).state for trainer in value["trainers"].values()))
            self.assert_closed(value)

    def test_25_clip_callback_exception_preserves_cause_partial_gradients_and_cleans_observers(self):
        with standalone_fixture() as value:
            def prepared(_):
                callback = value["trainers"]["candidate"].callback_handler.callbacks[-1]
                value["stack"].enter_context(patch.object(callback, "on_pre_optimizer_step",
                    side_effect=RuntimeError("synthetic clip failure")))
            value["standalone_hooks"].prepared_action = prepared
            result, error = self.failure(value)
            self.assertEqual(result["finite_clip_boundaries"], {"dense": 1, "candidate": 0})
            self.assertIsInstance(error.__cause__, RuntimeError)
            self.assertEqual(str(error.__cause__), "synthetic clip failure")
            self.assertTrue(result["partial_gradients_may_be_present"])
            self.assert_closed(value)

    def test_26_optimizer_state_or_scheduler_mutation_is_detected_without_another_witness(self):
        for change in ("adam", "scheduler"):
            with self.subTest(change=change), standalone_fixture() as value:
                original = value["standalone_hooks"].verify_identity
                def corrupt(*, stage, _change=change, **kwargs):
                    answer = original(stage=stage, **kwargs)
                    if stage == "standalone_unchanged_original_0":
                        trainer = value["trainers"]["candidate"]
                        if _change == "adam":
                            raw = fixtures.preserving._optimizer(trainer.optimizer)
                            raw.state[next(trainer.model.parameters())] = {"step": torch.tensor(1.)}
                        else: trainer.lr_scheduler.step()
                    return answer
                result, _ = self.failure(value, hooks=replace(value["standalone_hooks"].api(), verify_identity=corrupt))
                self.assertEqual(result["training_step_attempts"], {"dense": 1, "candidate": 1})
                self.assert_closed(value)

    def test_27_student_or_reference_parameter_mutation_stops_unchanged_identity_proof(self):
        for role in ("dense", "reference"):
            with self.subTest(role=role), standalone_fixture() as value:
                original = value["standalone_hooks"].compare_scalars
                def corrupt(*, stage, **kwargs):
                    answer = original(stage=stage, **kwargs)
                    if stage == "standalone_presentation_0":
                        model = value["reference"] if role == "reference" else value["models"]["dense"]
                        with torch.no_grad(): next(model.parameters()).flatten()[0].add_(.125)
                    return answer
                result, _ = self.failure(value, hooks=replace(value["standalone_hooks"].api(), compare_scalars=corrupt))
                self.assertEqual(result["training_step_attempts"], {"dense": 1, "candidate": 1})
                self.assert_closed(value)

    def test_28_reference_must_be_resident_frozen_eval_and_gradient_free_including_sft(self):
        for change in ("train", "requires_grad", "gradient"):
            with self.subTest(change=change), standalone_fixture() as value:
                reference = value["reference"]
                if change == "train": reference.train()
                elif change == "requires_grad": next(reference.parameters()).requires_grad_(True)
                else: next(reference.parameters()).grad = torch.zeros_like(next(reference.parameters()))
                result, _ = self.failure(value)
                self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})

    def test_29_frozen_preparation_configuration_cannot_be_adapted_for_this_diagnostic(self):
        for change in ("args_ga", "accelerator", "beta", "chunk", "clip", "checkpoint"):
            with self.subTest(change=change), standalone_fixture() as value:
                trainer = value["trainers"]["dense"]
                if change == "args_ga": trainer.args.gradient_accumulation_steps = 1
                elif change == "accelerator": trainer.accelerator.gradient_accumulation_steps = 8
                elif change == "beta": trainer.beta = .1
                elif change == "chunk": trainer.chunk_tokens = 64
                elif change == "clip": trainer.clip_norm = .5
                else: trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": True}
                result, _ = self.failure(value)
                self.assertEqual(result["training_step_attempts"], {"dense": 0, "candidate": 0})

    def test_30_replaced_training_step_existing_optimizer_or_extra_callback_is_not_native_inherited_execution(self):
        for change in ("step", "optimizer", "callback"):
            with self.subTest(change=change), standalone_fixture() as value:
                trainer = value["trainers"]["dense"]
                if change == "step": trainer.training_step = lambda *args, **kwargs: torch.tensor(0.)
                elif change == "optimizer": trainer.create_optimizer()
                else: trainer.add_callback(fixtures.TrainerCallback())
                result, _ = self.failure(value)
                self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})

    def test_31_observer_validation_or_body_failures_leave_no_registered_hooks_and_no_success(self):
        for where in ("validation", "raw_hook"):
            with self.subTest(where=where), standalone_fixture() as value:
                if where == "validation":
                    value["stack"].enter_context(patch.object(value["observers"]["candidate"], "validate",
                        side_effect=RuntimeError("synthetic observation failure")))
                    result, error = self.failure(value)
                    self.assertEqual(result["completed_backwards"], {"dense": 3, "candidate": 3})
                else:
                    original = value["standalone_hooks"].compare_tensors
                    def fail(*, stage, **kwargs):
                        answer = original(stage=stage, **kwargs)
                        if stage == "standalone_raw_gradients_0": raise RuntimeError("synthetic observation failure")
                        return answer
                    result, error = self.failure(value, hooks=replace(value["standalone_hooks"].api(), compare_tensors=fail))
                    self.assertEqual(result["completed_backwards"], {"dense": 1, "candidate": 1})
                self.assertIsInstance(error.__cause__, RuntimeError)
                self.assert_closed(value)

    def test_32_results_and_typed_partial_journals_are_bounded_finite_json_without_tensor_or_native_claims(self):
        with standalone_fixture() as value:
            result = run(value)
            self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
            self.assertLessEqual(len(json.dumps(result).encode()), module.paired.MAX_JOURNAL_BYTES)
            self.assertTrue(result["source_file_verification_owned_by_caller"])
            self.assertFalse(result["seed_contract"]["seed_configuration_alone_proves_rng_state"])
            self.assertTrue(any(event["event"] == "standalone_observer_validated" and
                event["metadata"]["cpu_fixture_only"] is True for event in result["events"]))
        with standalone_fixture() as value:
            original = value["standalone_hooks"].verify_identity
            def oversized(*, stage, **kwargs):
                answer = original(stage=stage, **kwargs)
                answer["unbounded"] = "x" * module.paired.MAX_RECEIPT_BYTES
                return answer
            result, error = self.failure(value, hooks=replace(value["standalone_hooks"].api(), verify_identity=oversized))
            self.assertNotIn("unbounded", json.dumps(result))
            self.assertIsInstance(error.__cause__, ValueError)
            self.assertEqual(result["completed_backwards"], {"dense": 0, "candidate": 0})


if __name__ == "__main__":
    unittest.main()
