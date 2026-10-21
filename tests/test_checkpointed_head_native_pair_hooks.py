"""Prospective32 CPU fixtures; authoring is not execution or admission.

Disposable tensors, private shadows and isolated primitive source modules are
NONNATIVE. Source-verifier cases exercise the production code-binding rules
only, without constructing CUDA models or claiming original434 provenance.
The one actual tiny-HF controller case remains a CPU integration oracle, not
native BF16/projection/recompute/headroom/save/reload proof. No original model,
experiment records or private cache is read. Run only in an admitted CPU job.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_paired_training_step_controller as paired_fixtures
from extension import checkpointed_head_native_pair_hooks as module


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "extension/planning/checkpointed_head_native_numerical_protocol.json"
EOS, SUFFIX = 7, (7, 8)
ARGUMENTS = "*, stage, dense_trainer, candidate_trainer, reference_model, context"
PRIMITIVE_SOURCE = '''import hashlib
import json
STATE = {"sha256": None, "changes": {}, "calls": [], "actions": {}}
def _receipt(name, stage, context):
    STATE["calls"].append((name, stage, context["next_write_bytes"]))
    action = STATE["actions"].get((name, stage))
    if action is not None:
        action()
    result = {"complete": True, "passed": True, "stage": stage,
        "case_binding_sha256": context["case_binding_sha256"],
        "fixture_only": context["fixture_only"],
        "primitive_source_sha256": STATE["sha256"]}
    if name == "verify_phase_accounting":
        payload = json.dumps(context["budget"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()
        result.update(planned_remaining_bytes=context["next_write_bytes"],
            budget_snapshot_value_sha256=hashlib.sha256(payload).hexdigest(),
            crossprocess_accounting_owned_by_supervisor=True)
    if name == "verify_original_numerics":
        result.update(measurements={"absolute_ce_difference": 0., "absolute_kl": 0.,
            "maximum_absolute_logits": 0.}, student_branches=["dense", "candidate"],
            witness_count=3, numeric_protocol_file_sha256=PROTOCOL_SHA)
    result.update(STATE["changes"].get(name, {}))
    return result
def verify_native_context(''' + ARGUMENTS + '''):
    return _receipt("verify_native_context", stage, context)
def verify_original_numerics(''' + ARGUMENTS + '''):
    return _receipt("verify_original_numerics", stage, context)
def verify_phase_accounting(''' + ARGUMENTS + '''):
    return _receipt("verify_phase_accounting", stage, context)
'''


def value_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_record(path):
    payload = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def no_tensors(value):
    if isinstance(value, torch.Tensor):
        return False
    if type(value) is dict:
        return all(type(key) is str and no_tensors(item) for key, item in value.items())
    if type(value) is list:
        return all(no_tensors(item) for item in value)
    return value is None or type(value) in (bool, str, int, float)


class TinyTied(torch.nn.Module):
    """Two original unique FP32 Parameter tensors, explicitly fixture-only."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding.from_pretrained(torch.tensor([[.125, .25], [.5, 1.]]), freeze=False)
        self.body = torch.nn.Linear(2, 2, bias=False)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.body.weight.copy_(torch.tensor([[.25, .5], [.75, 1.]]))
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self): return self.embed
    def get_output_embeddings(self): return self.lm_head


@contextmanager
def fixture(*, subcase="ga9", objective="ce_kl", case_changes=None, source=PRIMITIVE_SOURCE,
            expected_parameters=2, suffix=SUFFIX):
    """Real bounded helpers, disposable actual tensors, synthetic context only."""
    saved_rng = torch.get_rng_state()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve(); root.chmod(0o700)
        protocol = root / PROTOCOL
        protocol.parent.mkdir(parents=True)
        protocol.write_bytes((ROOT / PROTOCOL).read_bytes())
        primitive_path = root / "primitives.py"
        primitive_path.write_text(source)
        primitive_record = file_record(primitive_path)
        name = "_pair_hook_cpu_fixture_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, primitive_path)
        worker = importlib.util.module_from_spec(spec)
        sys.modules[name] = worker
        try:
            spec.loader.exec_module(worker)
            worker.STATE["sha256"] = primitive_record["sha256"]
            worker.PROTOCOL_SHA = module.PROTOCOL_SHA256
            inventory = {"primitives.py": primitive_record, PROTOCOL: file_record(protocol)}
            case = {"schema": "extension-native-pair-hook-case-binding-v1", "objective": objective,
                "subcase": subcase, "expected_parameters": expected_parameters, "fixture_only": True, "device": "cpu",
                "source_root": str(root), "source_inventory": inventory,
                "source_inventory_value_sha256": value_sha(inventory),
                "numerical_protocol_file_sha256": module.PROTOCOL_SHA256,
                **{key: value_sha(key) for key in module.CASE_HASHES}}
            case["terminal_suffix_value_sha256"] = value_sha(list(suffix))
            case.update(case_changes or {})
            sources = {key: {"path": "primitives.py", "sha256": primitive_record["sha256"],
                "qualname": key} for key in module.PRIMITIVES}
            primitives = module.NativeHookPrimitives(**{key: getattr(worker, key) for key in module.PRIMITIVES})
            scratch = root / "private"; scratch.mkdir(mode=0o700)
            with patch.object(module.comparison.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)):
                budget = module.comparison.ScratchBudget(scratch)
                adapter = module.NativePairHookAdapter(private_directory=scratch / "reports", scratch_budget=budget,
                    protocol_path=protocol, case_binding=case, primitives=primitives,
                    primitive_sources=sources, fixture_only=True, expected_parameters=expected_parameters)
                original = TinyTied().eval().requires_grad_(False)
                models = [deepcopy(original).train().requires_grad_(True) for _ in range(2)]
                trainers = [SimpleNamespace(model=model, reference_model=original, objective=objective, beta=.05,
                    terminal_suffix=suffix, eos_token_id=suffix[0], state=SimpleNamespace(global_step=0),
                    optimizer=torch.optim.AdamW(model.parameters(), lr=0.)) for model in models]
                result = {"root": root, "worker": worker, "path": primitive_path, "sources": sources,
                    "primitives": primitives, "adapter": adapter, "case": case, "budget": budget,
                    "dense": trainers[0], "candidate": trainers[1], "reference": original}
                yield result
        finally:
            sys.modules.pop(name, None)
            torch.set_rng_state(saved_rng)


def context(value, *, updates=0):
    return {"expected_parameters": value["adapter"].expected_parameters, "objective": value["case"]["objective"],
        "completed_backwards": {"dense": 8 if updates else 0, "candidate": 8 if updates else 0},
        "optimizer_updates": {"dense": updates, "candidate": updates}}


def call(value, method, stage, *, updates=0, **extra):
    return getattr(value["adapter"], method)(stage=stage, dense_trainer=value["dense"],
        candidate_trainer=value["candidate"], reference_model=value["reference"],
        context=context(value, updates=updates), **extra)


def gradients(value, dense=1., candidate=None):
    for trainer, number in ((value["dense"], dense), (value["candidate"], dense if candidate is None else candidate)):
        for parameter in trainer.model.parameters(): parameter.grad = torch.full_like(parameter, number)


def populate_state(value, steps):
    for trainer in (value["dense"], value["candidate"]):
        for _ in range(steps):
            for parameter in trainer.model.parameters(): parameter.grad = torch.ones_like(parameter)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)


def shadow(value):
    populate_state(value, 1)
    call(value, "verify_identity", "post_lr0_original", updates=1)
    return call(value, "create_shadow", "post_lr0_pre_update2", updates=1)


def packet(*, objective="ce_kl", size=8, ce=2., weighted=None, kl=.25, weight=1., active=4, eos=1):
    weighted = ce if weighted is None else weighted
    if objective == "sft": kl = 0.
    if objective in ("sft", "ce_kl"): weight = 1.
    metrics = {"loss": weighted + .05 * kl, "ce": ce, "weighted_ce": weighted, "kl": kl,
        "effective_weight": weight, "active_tokens": active, "supervised_tokens": active,
        "eos_tokens": eos, "non_eos_tokens": active - eos, "examples": 1}
    for prefix in ("eos_", "non_eos_"):
        for key in ("ce", "weighted_ce", "kl", "effective_weight"):
            metrics[prefix + key] = metrics[key] if prefix == "eos_" or active > eos else 0.
    return {"scaled_loss": metrics["loss"] / size, "group_size": size, "metrics": metrics}


def source_verifier(value):
    """Production code verifier ONLY; no production adapter/model receipt."""
    return SimpleNamespace(root=value["root"], case=value["case"], primitives=value["primitives"],
        primitive_sources=deepcopy(value["sources"]), fixture_only=False)


def verify_source(value):
    module.NativePairHookAdapter._verify_primitive_sources(source_verifier(value))


class CheckpointedHeadNativePairHooksTests(unittest.TestCase):
    def failed(self, value, function):
        with self.assertRaises(module.PairHookFailure) as caught:
            function()
        result = caught.exception.partial_result
        self.assertFalse(result["complete"])
        self.assertFalse(result["native_admission_established"])
        self.assertTrue(result["caller_must_preserve_partial_reports_and_abort"])
        self.assertTrue(no_tensors(result))
        self.assertTrue(value["adapter"]._failed)
        return result

    def details(self, value, receipt, index=0):
        pin = receipt["private_reports"][index]
        path = value["adapter"].directory / pin["path"]
        self.assertEqual(file_record(path), {key: pin[key] for key in ("sha256", "bytes")})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        return json.loads(path.read_text())["details"]

    def test_01_literal_tensor_gates_are_the_independent_frozen_seven_field_specs(self):
        expected = {"identity": (0., 0., None, None, None, None, False),
            "gradients": (1e-6, .05, 1e-7, .05, .01, .9999, False),
            "adam_first_moments": (1e-7, .05, None, None, .01, None, False),
            "adam_second_moments": (1e-12, .1, None, None, .03, None, False),
            "updated_weights": (1e-7, 1e-6, None, None, None, None, False),
            "actual_update2_deltas": (2e-8, .1, None, None, .05, None, True)}
        names = ("elementwise_atol", "elementwise_rtol", "rms_atol", "rms_rtol",
            "group_relative_l2_maximum", "group_cosine_minimum", "require_positive_group_norms")
        for kind, values in expected.items():
            self.assertEqual(module.tensor_tolerances(kind), dict(zip(names, values)))
        with self.assertRaises(ValueError): module.tensor_tolerances("relaxed")
        changed = module.tensor_tolerances("identity"); changed["elementwise_atol"] = 9.
        self.assertEqual(module.tensor_tolerances("identity")["elementwise_atol"], 0.)

    def test_02_exact_protocol_file_pin_and_fixed_path_are_mandatory(self):
        self.assertEqual(file_record(ROOT / PROTOCOL)["sha256"], module.PROTOCOL_SHA256)
        with fixture() as value:
            protocol = value["root"] / PROTOCOL
            protocol.write_bytes(protocol.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "FILE SHA"):
                module.NativePairHookAdapter(private_directory=value["budget"].directory / "second",
                    scratch_budget=value["budget"], protocol_path=protocol, case_binding=value["case"],
                    primitives=value["primitives"], primitive_sources=value["sources"], fixture_only=True, expected_parameters=2)
        with fixture() as value:
            with self.assertRaisesRegex(ValueError, "exact root-frozen"):
                module.NativePairHookAdapter(private_directory=value["budget"].directory / "second",
                    scratch_budget=value["budget"], protocol_path=ROOT / PROTOCOL, case_binding=value["case"],
                    primitives=value["primitives"], primitive_sources=value["sources"], fixture_only=True, expected_parameters=2)

    def test_03_case_schema_hashes_types_and_explicit_small_nonnative_count_fail_closed(self):
        for change in ({"expected_parameters": True}, {"fixture_only": False}, {"device": "cuda:0"},
                {"objective": "unknown"}, {"subcase": "long"}, {"extra": 1},
                {"input_bundle_value_sha256": "A" * 64}, {"source_inventory_value_sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                with fixture(case_changes=change): pass
        with fixture() as value:
            self.assertTrue(value["adapter"].fixture_only)
            self.assertEqual(value["adapter"].expected_parameters, 2)
            self.assertNotEqual(value["adapter"].expected_parameters, 434)

    def test_04_no_missing_noncallable_or_duck_typed_native_primitive_default_exists(self):
        with self.assertRaises(TypeError): module.NativeHookPrimitives()
        with self.assertRaises(ValueError): module.NativeHookPrimitives(None, lambda: None, lambda: None)
        with fixture() as value:
            changed = deepcopy(value["sources"]); changed.pop("verify_native_context")
            with self.assertRaisesRegex(ValueError, "sources are mandatory"):
                module.NativePairHookAdapter(private_directory=value["budget"].directory / "second",
                    scratch_budget=value["budget"], protocol_path=value["root"] / PROTOCOL, case_binding=value["case"],
                    primitives=value["primitives"], primitive_sources=changed, fixture_only=True, expected_parameters=2)

    def test_05_source_verifier_checks_actual_file_qualname_inventory_and_module_attribute(self):
        with fixture() as value:
            verify_source(value)
            for key, changed in (("path", "wrong.py"), ("sha256", "0" * 64), ("qualname", "wrong")):
                verifier = source_verifier(value)
                verifier.primitive_sources["verify_native_context"][key] = changed
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.NativePairHookAdapter._verify_primitive_sources(verifier)
            original = value["worker"].verify_native_context
            value["worker"].verify_native_context = object()
            with self.assertRaisesRegex(ValueError, "actual source-module"):
                verify_source(value)
            value["worker"].verify_native_context = original

    def test_06_forged_code_with_correct_filename_and_runtime_metadata_is_rejected(self):
        with fixture() as value:
            fn = value["worker"].verify_native_context
            constants = tuple("forged_context" if item == "verify_native_context" else item for item in fn.__code__.co_consts)
            forged = FunctionType(fn.__code__.replace(co_consts=constants), fn.__globals__, fn.__name__)
            forged.__qualname__, forged.__module__ = fn.__qualname__, fn.__module__
            value["worker"].verify_native_context = forged
            value["adapter"].primitives = module.NativeHookPrimitives(forged,
                value["worker"].verify_original_numerics, value["worker"].verify_phase_accounting)
            value["primitives"] = value["adapter"].primitives
            with self.assertRaisesRegex(ValueError, "Runtime primitive code"):
                verify_source(value)

    def test_07_actual_globals_and_code_signature_cannot_be_spoofed_by_function_metadata(self):
        with fixture() as value:
            fn = value["worker"].verify_native_context
            forged = FunctionType(fn.__code__, dict(fn.__globals__), fn.__name__)
            forged.__qualname__, forged.__module__ = fn.__qualname__, fn.__module__
            value["worker"].verify_native_context = forged
            value["primitives"] = module.NativeHookPrimitives(forged,
                value["worker"].verify_original_numerics, value["worker"].verify_phase_accounting)
            with self.assertRaisesRegex(ValueError, "globals"):
                verify_source(value)
        for defaults in ((1,), {"stage": "forged"}):
            with fixture() as value:
                fn = value["worker"].verify_native_context
                if type(defaults) is tuple: fn.__defaults__ = defaults
                else: fn.__kwdefaults__ = defaults
                with self.assertRaisesRegex(ValueError, "mandatory keyword-only"):
                    verify_source(value)
        changed = PRIMITIVE_SOURCE.replace("def verify_native_context(" + ARGUMENTS,
            "def verify_native_context(stage, *, dense_trainer, candidate_trainer, reference_model, context")
        with fixture(source=changed) as value:
            value["worker"].verify_native_context.__signature__ = __import__("inspect").Signature()
            with self.assertRaisesRegex(ValueError, "mandatory keyword-only"):
                verify_source(value)

    def test_08_source_mutation_before_or_during_a_primitive_call_poisons_the_adapter(self):
        for during in (False, True):
            with self.subTest(during=during), fixture() as value:
                mutate = lambda: value["path"].write_bytes(value["path"].read_bytes() + b"\n")
                if during:
                    value["worker"].STATE["actions"][("verify_native_context", "prepared")] = mutate
                else: mutate()
                self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
                self.assertFalse(value["adapter"].reports)

    def test_09_primitive_receipt_stage_source_case_qualification_and_success_are_exact(self):
        for change in ({"passed": False}, {"complete": 1}, {"stage": "other"},
                {"case_binding_sha256": "0" * 64}, {"primitive_source_sha256": "0" * 64}, {"fixture_only": False}):
            with self.subTest(change=change), fixture() as value:
                value["worker"].STATE["changes"]["verify_native_context"] = change
                self.failed(value, lambda: call(value, "verify_invariants", "prepared"))

    def test_10_scalar_thresholds_use_independent_dense_relative_and_inclusive_boundaries(self):
        gates = {"ce": (2e-4, .002), "weighted_ce": (1e-6, .01), "kl": (1e-5, .01), "effective_weight": (1e-7, .01)}
        with fixture(objective="asft") as value:
            left = packet(objective="asft", weight=.5, weighted=1.)
            for field, (atol, rtol) in gates.items():
                for prefix in ("eos_", "non_eos_"):
                    key = prefix + field
                    boundary = left["metrics"][key] + atol + rtol * abs(left["metrics"][key])
                    for target, expected in ((math.nextafter(boundary, -math.inf), True), (boundary * 1.001, False)):
                        right = deepcopy(left); right["metrics"][key] = target
                        result = value["adapter"]._scalar("presentation_0", left, right, "asft")
                        self.assertEqual(result["checks"][key], expected)
            a, limit = 2., 1e-6 + .01 * 2.
            inside = math.nextafter(a + limit, a)
            self.assertTrue(value["adapter"]._scalar("clip_norm_1", {"clip_norm": a}, {"clip_norm": inside}, "asft")["passed"])
            self.assertFalse(value["adapter"]._scalar("clip_norm_1", {"clip_norm": a}, {"clip_norm": a + 2 * limit}, "asft")["passed"])
            right = deepcopy(left); right["metrics"]["weighted_ce"] += .006
            right["metrics"]["loss"] = right["metrics"]["weighted_ce"] + .05 * right["metrics"]["kl"]
            right["scaled_loss"] = right["metrics"]["loss"] / 8
            self.assertFalse(value["adapter"]._scalar("presentation_0", left, right, "asft")["checks"]["loss"])

    def test_11_loss_formula_and_actual_group8_tail1_standalone1_are_not_renormalized(self):
        for subcase, stage, size in (("ga9", "presentation_0", 8), ("ga9", "presentation_8", 1),
                ("standalone", "standalone_presentation_0", 1)):
            with self.subTest(subcase=subcase, stage=stage), fixture(subcase=subcase) as value:
                left = packet(size=size)
                if subcase == "standalone": left["group1_loss"] = left.pop("scaled_loss")
                self.assertTrue(value["adapter"]._scalar(stage, left, deepcopy(left), "ce_kl")["passed"])
                right = deepcopy(left); right["group_size"] = 8 if size == 1 else 1
                with self.assertRaises(ValueError): value["adapter"]._scalar(stage, left, right, "ce_kl")
                right = deepcopy(left); right["group1_loss" if subcase == "standalone" else "scaled_loss"] /= 8
                with self.assertRaises(ValueError): value["adapter"]._scalar(stage, left, right, "ce_kl")
                right = deepcopy(left); right["metrics"]["loss"] += .01
                with self.assertRaises(ValueError): value["adapter"]._scalar(stage, left, right, "ce_kl")

    def test_12_exact_counts_eos_non_eos_and_all_finite_metric_fields_are_required(self):
        for field, target in (("eos_tokens", 0), ("examples", 2), ("active_tokens", 3.5),
                ("non_eos_tokens", -1), ("supervised_tokens", 99), ("ce", float("nan"))):
            with self.subTest(field=field), fixture() as value:
                left, right = packet(), packet(); right["metrics"][field] = target
                with self.assertRaises(ValueError): value["adapter"]._scalar("presentation_0", left, right, "ce_kl")
        with fixture() as value:
            for change in ("missing", "extra"):
                right = packet()
                if change == "missing": right["metrics"].pop("eos_ce")
                else: right["metrics"]["unsealed"] = 1
                with self.assertRaises(ValueError): value["adapter"]._scalar("presentation_0", packet(), right, "ce_kl")
            changed = packet(active=5, eos=2)
            result = value["adapter"]._scalar("presentation_0", packet(), changed, "ce_kl")
            self.assertFalse(result["passed"])

    def test_13_unweighted_objectives_and_sft_zero_kl_cannot_be_relabelled(self):
        for objective in ("sft", "ce_kl", "asft", "prior_anchored"):
            with self.subTest(objective=objective), fixture(objective=objective) as value:
                left = packet(objective=objective, weight=.25,
                    weighted=None if objective in ("sft", "ce_kl") else .5)
                self.assertTrue(value["adapter"]._scalar("presentation_0", left, deepcopy(left), objective)["passed"])
                if objective in ("sft", "ce_kl"):
                    right = deepcopy(left); right["metrics"]["eos_effective_weight"] = .999
                    with self.assertRaises(ValueError): value["adapter"]._scalar("presentation_0", left, right, objective)
                if objective == "sft":
                    right = deepcopy(left); right["metrics"]["eos_kl"] = 1e-9
                    with self.assertRaises(ValueError): value["adapter"]._scalar("presentation_0", left, right, objective)

    def test_14_actual_raw_and_clipped_gradients_use_full_group_norm_rms_and_cosine(self):
        with fixture() as value:
            gradients(value, 1., 1.001)
            receipt = call(value, "compare_tensors", "raw_gradients_1")
            report = self.details(value, receipt)
            numbers = [float(parameter.grad.flatten()[0]) for parameter in (value["dense"].model.embed.weight,
                value["candidate"].model.embed.weight)]
            error = numbers[1] - numbers[0]
            group = report["groups"]["all"]
            self.assertAlmostEqual(group["difference_l2"], math.sqrt(8 * error * error), places=13)
            self.assertAlmostEqual(group["relative_l2"], abs(error / numbers[0]), places=13)
            self.assertAlmostEqual(group["rms_difference"], abs(error), places=13)
            self.assertAlmostEqual(group["cosine"], 1., places=13)
            self.assertEqual(report["tensor_count"], 2)
            self.assertLessEqual(report["streaming"]["peak_owned_host_bytes"], 128 << 20)
            self.assertTrue(call(value, "compare_tensors", "clipped_gradients_1")["passed"])
        with fixture() as value:
            gradients(value, 1., 1.02)
            result = self.failed(value, lambda: call(value, "compare_tensors", "raw_gradients_1"))
            report = json.loads((value["adapter"].directory / result["private_reports"][0]["path"]).read_text())["details"]
            self.assertEqual(report["groups"]["all"]["elementwise_failures"], 0)
            self.assertFalse(report["groups"]["all"]["passed"])
        with fixture() as value:
            gradients(value, 0.)
            value["candidate"].model.embed.weight.grad[0, 0] = 5e-7
            self.failed(value, lambda: call(value, "compare_tensors", "raw_gradients_1"))

    def test_15_every_actual_gradient_must_exist_be_finite_and_match_fp32_shape(self):
        for target in (None, "nan", "shape", "dtype"):
            with self.subTest(target=target), fixture() as value:
                gradients(value)
                parameter = value["candidate"].model.embed.weight
                if target is None: parameter.grad = None
                elif target == "nan": parameter.grad[0, 0] = float("nan")
                elif target == "shape": parameter.data = parameter.data.reshape(4)
                else: parameter.data = parameter.data.to(dtype=torch.float64)
                self.failed(value, lambda: call(value, "compare_tensors", "raw_gradients_1"))

    def test_16_adam_structure_step_and_nonnegative_second_moments_are_actual_not_inferred(self):
        for steps in (1, 2):
            with self.subTest(steps=steps), fixture() as value:
                populate_state(value, steps)
                for kind in ("adam_first_moments", "adam_second_moments"):
                    self.assertTrue(call(value, "compare_tensors", kind + "_" + str(steps), updates=steps)["passed"])
        for damage in ("missing", "extra", "step", "negative", "dtype"):
            with self.subTest(damage=damage), fixture() as value:
                populate_state(value, 1)
                optimizer = value["candidate"].optimizer
                parameter = next(value["candidate"].model.parameters()); state = optimizer.state[parameter]
                if damage == "missing": state.pop("exp_avg")
                elif damage == "extra": state["unreviewed"] = torch.tensor(0.)
                elif damage == "step": state["step"].fill_(2.)
                elif damage == "negative": state["exp_avg_sq"][0, 0] = -1.
                else: state["exp_avg"] = state["exp_avg"].double()
                self.failed(value, lambda: call(value, "compare_tensors", "adam_second_moments_1", updates=1))

    def test_17_distinct_first_second_moment_relative_gates_do_not_use_weight_tolerance(self):
        for kind, factor, expected in (("adam_first_moments", 1.005, True),
                ("adam_first_moments", 1.02, False), ("adam_second_moments", 1.02, True),
                ("adam_second_moments", 1.04, False)):
            with self.subTest(kind=kind, factor=factor), fixture() as value:
                populate_state(value, 1)
                key = "exp_avg" if "first" in kind else "exp_avg_sq"
                for state in value["candidate"].optimizer.state.values(): state[key].mul_(factor)
                if expected:
                    receipt = call(value, "compare_tensors", kind + "_1", updates=1)
                    report = self.details(value, receipt)
                    self.assertLess(report["groups"]["all"]["relative_l2"], .03)
                else: self.failed(value, lambda: call(value, "compare_tensors", kind + "_1", updates=1))

    def test_18_full_updated_weight_bounds_cover_both_tied_and_backbone_parameters(self):
        with fixture() as value:
            with torch.no_grad():
                for parameter in value["candidate"].model.parameters(): parameter.add_(1e-7)
            receipt = call(value, "compare_tensors", "updated_weights_2", updates=2)
            self.assertEqual(receipt["compared_tensor_count"], 2)
            self.assertEqual(set(receipt["comparisons"][0]["groups"]), {"all", "tied", "backbone"})
        for name in ("embed.weight", "body.weight"):
            with self.subTest(name=name), fixture() as value:
                with torch.no_grad(): dict(value["candidate"].model.named_parameters())[name].add_(.001)
                self.failed(value, lambda: call(value, "compare_tensors", "updated_weights_2", updates=2))

    def test_19_original_identity_requires_both_branch_byte_hashes_even_for_signed_zero(self):
        with fixture() as value:
            receipt = call(value, "verify_identity", "initial_original")
            self.assertEqual(receipt["compared_tensor_count"], 2)
            for index in (0, 1): self.assertTrue(self.details(value, receipt, index)["exact_stored_fp32_bytes_equal"])
        with fixture() as value:
            with torch.no_grad():
                value["reference"].embed.weight[0, 0] = 0.
                value["dense"].model.embed.weight[0, 0] = 0.
                value["candidate"].model.embed.weight[0, 0] = -0.
            self.assertEqual(float(value["candidate"].model.embed.weight[0, 0]), 0.)
            result = self.failed(value, lambda: call(value, "verify_identity", "initial_original"))
            reports = [json.loads((value["adapter"].directory / row["path"]).read_text())["details"]
                for row in result["private_reports"][:2]]
            self.assertTrue(reports[0]["passed"])
            self.assertFalse(reports[1]["exact_stored_fp32_bytes_equal"])

    def test_20_original_numeric_primitive_keeps_all_three_old_limits_and_witnesses(self):
        with fixture() as value:
            value["worker"].STATE["changes"]["verify_original_numerics"] = {"measurements": {
                "absolute_ce_difference": 1e-5, "absolute_kl": 1e-5, "maximum_absolute_logits": .001}}
            self.assertTrue(call(value, "verify_identity", "initial_original")["passed"])
        for change in ({"witness_count": 2}, {"student_branches": ["candidate"]},
                {"numeric_protocol_file_sha256": "0" * 64},
                {"measurements": {"absolute_ce_difference": 1.1e-5, "absolute_kl": 0., "maximum_absolute_logits": 0.}},
                {"measurements": {"absolute_ce_difference": 0., "absolute_kl": 1.1e-5, "maximum_absolute_logits": 0.}},
                {"measurements": {"absolute_ce_difference": 0., "absolute_kl": 0., "maximum_absolute_logits": .0011}}):
            with self.subTest(change=change), fixture() as value:
                value["worker"].STATE["changes"]["verify_original_numerics"] = change
                result = self.failed(value, lambda: call(value, "verify_identity", "initial_original"))
                self.assertGreaterEqual(len(result["private_reports"]), 2)

    def test_21_actual_ties_disjoint_storage_frozen_reference_and_model_identity_are_required(self):
        for change in ("untie", "shared", "reference_train", "reference_grad", "replace"):
            with self.subTest(change=change), fixture() as value:
                if change == "replace":
                    call(value, "verify_invariants", "prepared")
                    value["candidate"].model = deepcopy(value["candidate"].model)
                elif change == "untie": value["candidate"].model.lm_head.weight = torch.nn.Parameter(value["candidate"].model.lm_head.weight.detach().clone())
                elif change == "shared": value["candidate"].model = value["dense"].model
                elif change == "reference_train": value["reference"].train()
                else: next(value["reference"].parameters()).grad = torch.ones_like(next(value["reference"].parameters()))
                stage = "final" if change == "replace" else "prepared"
                self.failed(value, lambda: call(value, "verify_invariants", stage))

    def test_22_live_suffix_eos_and_case_hash_are_checked_without_trusting_primitive_success(self):
        for change in ("suffix", "suffix_type", "eos", "missing_eos"):
            with self.subTest(change=change), fixture() as value:
                if change == "suffix":
                    value["dense"].terminal_suffix = value["candidate"].terminal_suffix = (EOS, 9)
                elif change == "suffix_type": value["candidate"].terminal_suffix = list(SUFFIX)
                elif change == "eos": value["candidate"].eos_token_id = EOS + 1
                else: value["dense"].eos_token_id = value["candidate"].eos_token_id = 99
                self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
                self.assertFalse(value["worker"].STATE["calls"])

    def test_23_private_directory_and_report_files_are_exclusive_owner_only_and_no_alias(self):
        with fixture() as value:
            self.assertEqual(stat.S_IMODE(value["adapter"].directory.stat().st_mode), 0o700)
            call(value, "verify_invariants", "prepared")
            path = value["adapter"].directory / "report-0000.json"
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            os.link(path, value["root"] / "alias.json")
            with self.assertRaisesRegex(ValueError, "single-link"):
                module._file(path, maximum=module.MAX_REPORT_BYTES, private=True)
            with self.assertRaises(ValueError):
                module.NativePairHookAdapter(private_directory=value["adapter"].directory,
                    scratch_budget=value["budget"], protocol_path=value["root"] / PROTOCOL, case_binding=value["case"],
                    primitives=value["primitives"], primitive_sources=value["sources"], fixture_only=True, expected_parameters=2)
        with fixture() as value:
            value["adapter"].directory.chmod(0o750)
            self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
        with fixture() as value:
            link = value["budget"].directory / "link"; link.symlink_to(value["adapter"].directory, target_is_directory=True)
            with self.assertRaises(ValueError): module._path(link, directory=True)

    def test_24_closed_private_file_pins_preserve_detail_and_controller_receipts_are_bounded(self):
        with fixture() as value:
            gradients(value)
            receipt = call(value, "compare_tensors", "raw_gradients_1")
            report = self.details(value, receipt)
            self.assertEqual(set(report["tensors"]), {"embed.weight", "body.weight"})
            self.assertNotIn("tensors", receipt)
            self.assertNotIn("tensors", receipt["comparisons"][0])
            self.assertFalse(receipt["native_provenance_established"])
            self.assertFalse(receipt["crossprocess_budget_proven_here"])
            self.assertTrue(no_tensors(receipt))
            self.assertLess(len(json.dumps(receipt).encode()), module.MAX_RECEIPT_BYTES)
            intended = sum(row["bytes"] for row in value["adapter"].reports)
            self.assertEqual(value["budget"].bytes_written, intended)
            with self.assertRaises(ValueError):
                value["adapter"]._write_report("oversized", "x" * module.MAX_REPORT_BYTES,
                    native_args=(value["dense"], value["candidate"], value["reference"], context(value)))

    def test_25_partial_report_write_charges_actual_prefix_keeps_artifact_and_poison(self):
        with fixture() as value:
            original_write, calls = os.write, []
            def broken(fd, payload):
                calls.append(len(payload))
                if len(calls) == 1: return original_write(fd, payload[:17])
                raise OSError("synthetic private disk fault")
            with patch.object(module.os, "write", side_effect=broken):
                result = self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
            self.assertEqual(value["budget"].bytes_written, 17)
            self.assertEqual(result["partial_private_report"]["bytes_charged"], 17)
            self.assertFalse(result["private_reports"])
            self.assertEqual((value["adapter"].directory / "report-0000.json").stat().st_size, 17)
            self.failed(value, lambda: call(value, "verify_invariants", "final"))

    def test_26_phase_accounting_binds_actual_shared_budget_and_full_remaining_plan(self):
        for change in ({"planned_remaining_bytes": 0}, {"planned_remaining_bytes": True},
                {"budget_snapshot_value_sha256": "0" * 64}, {"crossprocess_accounting_owned_by_supervisor": False},
                {"planned_remaining_bytes": 113 << 30}):
            with self.subTest(change=change), fixture() as value:
                value["worker"].STATE["changes"]["verify_phase_accounting"] = change
                self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
        with fixture() as value:
            value["budget"].charge_written((112 << 30) - (1 << 20))
            self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
        with fixture() as value:
            with patch.object(module.comparison.shutil, "disk_usage", return_value=SimpleNamespace(free=32 << 30)):
                self.failed(value, lambda: call(value, "verify_invariants", "prepared"))

    def test_27_shadow_requires_post_lr0_identity_exact_phase_actual_state1_and_zero_lr(self):
        with fixture() as value:
            self.failed(value, lambda: call(value, "create_shadow", "post_lr0_pre_update2", updates=1))
        for change in ("counter", "version", "lr", "state", "data"):
            with self.subTest(change=change), fixture() as value:
                populate_state(value, 1)
                call(value, "verify_identity", "post_lr0_original", updates=1)
                if change == "version":
                    with torch.no_grad(): next(value["candidate"].model.parameters()).add_(0.)
                elif change == "lr": value["candidate"].optimizer.param_groups[0]["lr"] = 1e-6
                elif change == "state": value["candidate"].optimizer.state.clear()
                elif change == "data": next(value["candidate"].model.parameters()).data.add_(.01)
                self.failed(value, lambda: call(value, "create_shadow", "post_lr0_pre_update2", updates=2 if change == "counter" else 1))

    def test_28_one_common_shadow_is_actual_little_endian_fp32_and_physically_pinned(self):
        with fixture() as value:
            artifact = shadow(value)
            self.assertIsInstance(artifact, module.paired.ShadowArtifact)
            manifest = artifact.handle.reader.manifest
            self.assertEqual(manifest["tensor_count"], 2)
            self.assertEqual(artifact.receipt["shadow_manifest_file_sha256"], file_record(artifact.handle.reader.directory / "manifest.json")["sha256"])
            for name, row in manifest["tensors"].items():
                actual = dict(value["reference"].named_parameters())[name].detach().flatten().tolist()
                raw = (artifact.handle.reader.directory / row["filename"]).read_bytes()
                self.assertEqual(raw, struct.pack("<" + "f" * len(actual), *actual))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
            expected = sum(file_record(path)["bytes"] for path in value["adapter"].directory.iterdir()) + sum(
                file_record(path)["bytes"] for path in artifact.handle.reader.directory.iterdir())
            self.assertEqual(value["budget"].bytes_written, expected)
            self.assertEqual(len(list(value["budget"].directory.glob("*-initial-shadow"))), 1)
            self.assertFalse(artifact.receipt["native_provenance_established"])

    def test_29_actual_update2_deltas_use_same_initial_bytes_and_positive_tied_backbone_groups(self):
        with fixture() as value:
            artifact = shadow(value)
            for trainer in (value["dense"], value["candidate"]):
                for parameter in trainer.model.parameters(): parameter.grad = torch.ones_like(parameter)
                trainer.optimizer.param_groups[0]["lr"] = 2e-6 / 57
                trainer.optimizer.step()
            receipt = call(value, "compare_deltas", "actual_update2_deltas", updates=2, initial_shadow=artifact.handle)
            report = self.details(value, receipt)
            initial = dict(value["reference"].named_parameters())
            dense = dict(value["dense"].model.named_parameters())
            for group, names in (("tied", ("embed.weight",)), ("backbone", ("body.weight",))):
                differences = [float(b) - float(a) for name in names for a, b in zip(initial[name].detach().flatten(), dense[name].detach().flatten())]
                expected = math.sqrt(math.fsum(item * item for item in differences))
                self.assertGreater(expected, 0.)
                self.assertAlmostEqual(report["groups"][group]["dense_l2"], expected, places=15)
            self.assertEqual(receipt["positive_update_groups"], {branch: {"tied": True, "backbone": True} for branch in ("dense", "candidate")})
            self.assertFalse(report["optimizer_formula_used"])
        with fixture() as value:
            artifact = shadow(value)
            for trainer in (value["dense"], value["candidate"]):
                for parameter in trainer.model.parameters(): parameter.grad = torch.ones_like(parameter)
                trainer.optimizer.step()
            self.failed(value, lambda: call(value, "compare_deltas", "actual_update2_deltas", updates=2, initial_shadow=artifact.handle))

    def test_30_wrong_foreign_tampered_or_missing_shadow_cannot_produce_delta_success(self):
        for change in ("foreign", "raw", "manifest", "state"):
            with self.subTest(change=change), fixture() as value:
                artifact = shadow(value)
                for trainer in (value["dense"], value["candidate"]):
                    for parameter in trainer.model.parameters(): parameter.grad = torch.ones_like(parameter)
                    trainer.optimizer.step()
                handle = artifact.handle
                if change == "foreign": handle = object()
                elif change in ("raw", "manifest"):
                    path = artifact.handle.reader.directory / ("manifest.json" if change == "manifest" else "tensor-0000.f32")
                    path.write_bytes(path.read_bytes() + b"x")
                else: value["candidate"].optimizer.state.clear()
                self.failed(value, lambda: call(value, "compare_deltas", "actual_update2_deltas", updates=2, initial_shadow=handle))

    def test_31_unknown_repeated_wrong_subcase_and_failed_stages_never_resume(self):
        with fixture() as value:
            call(value, "verify_invariants", "prepared")
            self.failed(value, lambda: call(value, "verify_invariants", "prepared"))
            self.failed(value, lambda: call(value, "verify_identity", "initial_original"))
        for method, stage in (("verify_invariants", "unreviewed"), ("compare_scalars", "presentation_9"),
                ("compare_tensors", "raw_gradients_3"), ("verify_identity", "standalone_initial_original")):
            with self.subTest(stage=stage), fixture() as value:
                self.failed(value, lambda: call(value, method, stage))
        with fixture(subcase="standalone") as value:
            with self.assertRaises(ValueError): value["adapter"].paired_hooks()
            self.failed(value, lambda: call(value, "create_shadow", "post_lr0_pre_update2", updates=1))

    def test_32_actual_tiny_paired_controller_and_standalone_hook_types_remain_nonnative(self):
        with fixture(subcase="standalone") as value:
            self.assertIs(type(value["adapter"].standalone_hooks()), module.StandaloneStepHooks)
        with paired_fixtures.pair_fixture("sft") as tiny, fixture(objective="sft",
                expected_parameters=tiny["count"], suffix=paired_fixtures.SUFFIX) as value:
            # Tiny count/suffix are explicitly supplied through the real
            # constructor; no adapter, production guard or case is rebound.
            adapter = value["adapter"]
            hooks = adapter.paired_hooks()
            self.assertIs(type(hooks), module.paired.PairedStepHooks)
            result = paired_fixtures.run_pair(tiny, hooks=hooks)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_backwards"], {"dense": 9, "candidate": 9})
            self.assertEqual(result["optimizer_updates"], {"dense": 2, "candidate": 2})
            self.assertTrue(result["fixture_count_is_not_native_proof"])
            self.assertFalse(result["native_provenance_established"])
            self.assertFalse(result["full_trainer_train_loop_established"])
            self.assertIsNotNone(adapter._shadow)
            self.assertEqual({entry[0] for entry in adapter._entered}, {"invariants", "identity", "scalars", "tensors", "shadow", "deltas"})
            self.assertTrue(all(row["bytes"] <= module.MAX_REPORT_BYTES for row in adapter.reports))
            self.assertTrue(no_tensors(result))


if __name__ == "__main__":
    unittest.main()
