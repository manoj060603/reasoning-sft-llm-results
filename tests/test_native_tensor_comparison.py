"""Prospective synthetic CPU oracles; authored, never executed or admitted here.

All tensors, raw shadow bytes and paths in this module are disposable fixtures.
No original model, experiment cache, private records, GPU or scheduler is used.
Small configurable tensor counts explicitly do not establish native434 parity,
BF16 numerical acceptance, whole-process scratch bounds or training admission.
Run only in a separately approved native CPU allocation after source review.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from extension import native_tensor_comparison as module


GROUPS = {"tied": ("embedding.weight",), "backbone": ("body.weight", "body.bias")}


def tensors():
    dense = {
        "embedding.weight": torch.tensor([[1., -2., 3.], [4., 0., -1.]], dtype=torch.float32),
        "body.weight": torch.tensor([[.5, -1.5], [2.5, 3.5]], dtype=torch.float32),
        "body.bias": torch.tensor([-.25, .75, 1.25], dtype=torch.float32),
    }
    candidate = {
        "embedding.weight": dense["embedding.weight"] + torch.tensor([[.125, 0., -.25], [0., .5, -.125]]),
        "body.weight": dense["body.weight"] + torch.tensor([[0., .25], [-.5, .125]]),
        "body.bias": dense["body.bias"] + torch.tensor([.125, -.25, 0.]),
    }
    return candidate, dense


def tolerance(**changes):
    value = {"elementwise_atol": 1., "elementwise_rtol": 0.,
        "rms_atol": None, "rms_rtol": None,
        "group_relative_l2_maximum": None, "group_cosine_minimum": None,
        "require_positive_group_norms": False}
    value.update(changes)
    return value


def flattened(value, names=None):
    names = tuple(sorted(value)) if names is None else names
    return [float(item) for name in names for item in value[name].detach().reshape(-1).tolist()]


def dense_oracle(candidate, dense, *, atol=1., rtol=0., names=None):
    """Independent Python FP64 reductions over complete synthetic fixture data."""
    reference, actual = flattened(dense, names), flattened(candidate, names)
    if len(reference) != len(actual):
        raise AssertionError("The fixture oracle itself needs matching elements")
    errors = [b - a for a, b in zip(reference, actual)]
    dense_square = math.fsum(a * a for a in reference)
    candidate_square = math.fsum(b * b for b in actual)
    error_square = math.fsum(d * d for d in errors)
    dot = math.fsum(a * b for a, b in zip(reference, actual))
    count = len(reference)
    return {
        "elements": count,
        "dense_l2": math.sqrt(dense_square),
        "candidate_l2": math.sqrt(candidate_square),
        "difference_l2": math.sqrt(error_square),
        "max_absolute_difference": max((abs(d) for d in errors), default=0.),
        "dense_max_absolute": max(abs(a) for a in reference),
        "candidate_max_absolute": max(abs(b) for b in actual),
        "rms_difference": math.sqrt(error_square / count),
        "dense_rms": math.sqrt(dense_square / count),
        "candidate_rms": math.sqrt(candidate_square / count),
        "relative_l2": None if dense_square == 0 else math.sqrt(error_square / dense_square),
        "cosine": None if dense_square == 0 or candidate_square == 0 else dot / math.sqrt(dense_square * candidate_square),
        "elementwise_failures": sum(abs(d) > atol + rtol * abs(a) for a, d in zip(reference, errors)),
    }


def raw_bytes(value):
    items = [float(item) for item in value.detach().reshape(-1).tolist()]
    return struct.pack("<" + "f" * len(items), *items)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewrite_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    Path(path).chmod(0o600)


def no_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return False
    if type(value) is dict:
        return all(type(key) is str and no_tensor_tree(item) for key, item in value.items())
    if type(value) in (list, tuple):
        return all(no_tensor_tree(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


@contextmanager
def shadow_fixture(initial=None, *, chunk=2):
    initial = tensors()[1] if initial is None else initial
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        parent.chmod(0o700)
        with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)):
            budget = module.ScratchBudget(parent)
            path = parent / "initial"
            receipt = module.write_initial_shadow(initial, path, budget=budget,
                expected_count=len(initial), chunk_elements=chunk)
            yield parent, path, budget, receipt


def open_shadow(path, receipt, *, count=3, chunk=2):
    return module.open_initial_shadow(path, expected_manifest_sha256=receipt["manifest_sha256"],
        expected_count=count, chunk_elements=chunk)


class StreamingNumericalTests(unittest.TestCase):
    def compare(self, candidate, dense, *, groups=None, spec=None, chunk=2, count=None):
        return module.compare_named_tensors(candidate, dense,
            groups=GROUPS if groups is None else groups,
            tolerances=tolerance() if spec is None else spec,
            expected_count=len(dense) if count is None else count, chunk_elements=chunk)

    def assertStats(self, actual, expected):
        self.assertEqual(actual["elements"], expected["elements"])
        self.assertEqual(actual["elementwise_failures"], expected["elementwise_failures"])
        for key in ("dense_l2", "candidate_l2", "difference_l2", "max_absolute_difference",
                    "dense_max_absolute", "candidate_max_absolute", "rms_difference",
                    "dense_rms", "candidate_rms", "relative_l2", "cosine"):
            if expected[key] is None:
                self.assertIsNone(actual[key], key)
            else:
                self.assertTrue(math.isclose(actual[key], expected[key], rel_tol=2e-13, abs_tol=1e-14),
                    (key, actual[key], expected[key]))

    def test_declared_chunk_memory_and_scratch_bounds_are_finite_and_explicit(self):
        self.assertEqual(module.MAX_CHUNK_ELEMENTS, 1_048_576)
        self.assertEqual(module.MAX_HOST_SCRATCH_BYTES, 128 * 1024**2)
        for size in (1, 17, module.MAX_CHUNK_ELEMENTS):
            with self.subTest(size=size):
                bound = module.chunk_memory_bound(size)
                self.assertIs(type(bound), int)
                self.assertGreaterEqual(bound, size * 8)
                self.assertLessEqual(bound, module.MAX_HOST_SCRATCH_BYTES)
        for size in (0, -1, True, 1.5, module.MAX_CHUNK_ELEMENTS + 1):
            with self.subTest(invalid=size), self.assertRaises(ValueError):
                module.chunk_memory_bound(size)

    def test_unequal_tensor_sizes_match_independent_dense_fp64_and_exact_group_partition(self):
        candidate, dense = tensors()
        result = self.compare(candidate, dense)
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["tensor_count"], 3)
        self.assertEqual(set(result["tensors"]), set(dense))
        self.assertEqual(set(result["groups"]), {"all", "tied", "backbone"})
        for name in dense:
            self.assertStats(result["tensors"][name], dense_oracle(candidate, dense, names=(name,)))
        for name, members in {"all": tuple(sorted(dense)), **GROUPS}.items():
            self.assertStats(result["groups"][name], dense_oracle(candidate, dense, names=members))
        self.assertEqual(result["groups"]["all"]["elements"], 13)
        # An average of three per-tensor RMS values is not the13-element RMS.
        wrong = sum(result["tensors"][name]["rms_difference"] for name in dense) / 3
        self.assertFalse(math.isclose(wrong, result["groups"]["all"]["rms_difference"], abs_tol=1e-6))

    def test_fp64_reductions_do_not_overflow_on_finite_fp32_values(self):
        dense = {"a": torch.tensor([1e20, -1e20, 1e-20], dtype=torch.float32)}
        candidate = {"a": dense["a"] * 1.125}
        result = self.compare(candidate, dense, groups={"body": ("a",)}, spec=tolerance(elementwise_atol=0., elementwise_rtol=.2))
        self.assertStats(result["groups"]["all"], dense_oracle(candidate, dense, atol=0., rtol=.2))
        self.assertTrue(result["passed"])
        self.assertTrue(math.isfinite(result["groups"]["all"]["dense_l2"]))

    def test_elementwise_rule_is_inclusive_and_scales_only_from_dense_values(self):
        dense = {"a": torch.tensor([2., -4., 0.], dtype=torch.float32)}
        candidate = {"a": torch.tensor([2.75, -5.25, .25], dtype=torch.float32)}
        spec = tolerance(elementwise_atol=.25, elementwise_rtol=.25)
        result = self.compare(candidate, dense, groups={"body": ("a",)}, spec=spec)
        self.assertTrue(result["passed"])
        self.assertEqual(result["groups"]["all"]["elementwise_failures"], 0)
        candidate["a"][0] = 2.875
        failed = self.compare(candidate, dense, groups={"body": ("a",)}, spec=spec)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["tensors"]["a"]["elementwise_failures"], 1)
        self.assertEqual(failed["groups"]["all"]["elementwise_failures"], 1)

    def test_rms_rule_uses_full_tensor_denominator_and_inclusive_dyadic_boundary(self):
        dense = {"a": torch.ones(4, dtype=torch.float32)}
        candidate = {"a": torch.tensor([1.5, .5, 1.5, .5], dtype=torch.float32)}
        spec = tolerance(rms_atol=.25, rms_rtol=.25)
        self.assertTrue(self.compare(candidate, dense, groups={"body": ("a",)}, spec=spec)["passed"])
        failed = self.compare(candidate, dense, groups={"body": ("a",)}, spec=tolerance(rms_atol=.125, rms_rtol=.25))
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(failed["tensors"]["a"]["rms_difference"], .5)

    def test_group_relative_l2_and_cosine_gates_cannot_be_hidden_by_large_tied_tensor(self):
        dense = {"embedding.weight": torch.full((6,), 1000., dtype=torch.float32),
                 "body.weight": torch.tensor([1., 0.]), "body.bias": torch.tensor([0.])}
        candidate = {name: value.clone() for name, value in dense.items()}
        candidate["body.weight"] = torch.tensor([0., 1.])
        spec = tolerance(elementwise_atol=2., group_relative_l2_maximum=.1,
            group_cosine_minimum=.99, require_positive_group_norms=True)
        result = self.compare(candidate, dense, spec=spec)
        self.assertTrue(result["groups"]["all"]["passed"])
        self.assertFalse(result["groups"]["backbone"]["passed"])
        self.assertFalse(result["passed"])
        self.assertStats(result["groups"]["backbone"], dense_oracle(candidate, dense, atol=2., names=GROUPS["backbone"]))
        one, shifted = {"a": torch.tensor([1.])}, {"a": torch.tensor([1.5])}
        self.assertTrue(self.compare(shifted, one, groups={"body": ("a",)},
            spec=tolerance(group_relative_l2_maximum=.5))["passed"])
        self.assertFalse(self.compare(shifted, one, groups={"body": ("a",)},
            spec=tolerance(group_relative_l2_maximum=.499))["passed"])

    def test_zero_reference_norm_is_undefined_and_positive_group_rule_checks_both_branches(self):
        zeros = {"a": torch.zeros(4, dtype=torch.float32)}
        result = self.compare(zeros, zeros, groups={"body": ("a",)}, spec=tolerance(elementwise_atol=0.))
        self.assertTrue(result["passed"])
        self.assertIsNone(result["groups"]["all"]["relative_l2"])
        self.assertIsNone(result["groups"]["all"]["cosine"])
        nonzero = {"a": torch.ones(4, dtype=torch.float32)}
        undefined = self.compare(nonzero, zeros, groups={"body": ("a",)},
            spec=tolerance(elementwise_atol=1., group_relative_l2_maximum=0.))
        self.assertTrue(undefined["passed"])
        self.assertIsNone(undefined["groups"]["all"]["relative_l2"])
        for candidate, dense in ((zeros, zeros), (zeros, nonzero), (nonzero, zeros)):
            with self.subTest(candidate_zero=not bool(candidate["a"].any()), dense_zero=not bool(dense["a"].any())):
                failed = self.compare(candidate, dense, groups={"body": ("a",)},
                    spec=tolerance(elementwise_atol=2., require_positive_group_norms=True))
                self.assertFalse(failed["passed"])

    def test_zero_and_negative_cosine_are_not_clamped_into_agreement(self):
        dense = {"a": torch.tensor([1., 0.])}
        for values, cosine in (([0., 1.], 0.), ([-1., 0.], -1.)):
            with self.subTest(values=values):
                candidate = {"a": torch.tensor(values)}
                result = self.compare(candidate, dense, groups={"body": ("a",)},
                    spec=tolerance(elementwise_atol=3., group_cosine_minimum=.5))
                self.assertFalse(result["passed"])
                self.assertAlmostEqual(result["groups"]["all"]["cosine"], cosine)
        orthogonal = {"a": torch.tensor([0., 1.])}
        self.assertTrue(self.compare(orthogonal, dense, groups={"body": ("a",)},
            spec=tolerance(elementwise_atol=1., group_cosine_minimum=0.))["passed"])
        self.assertFalse(self.compare(orthogonal, dense, groups={"body": ("a",)},
            spec=tolerance(elementwise_atol=1., group_cosine_minimum=1e-8))["passed"])

    def test_name_shape_dtype_and_contiguity_mismatches_refuse_instead_of_zipping(self):
        candidate, dense = tensors()
        mutations = []
        value = deepcopy(candidate); value["renamed"] = value.pop("body.bias"); mutations.append(value)
        value = deepcopy(candidate); value.pop("body.bias"); mutations.append(value)
        value = deepcopy(candidate); value["body.weight"] = value["body.weight"].reshape(4); mutations.append(value)
        value = deepcopy(candidate); value["body.weight"] = value["body.weight"].double(); mutations.append(value)
        value = deepcopy(candidate); value["body.weight"] = value["body.weight"].t(); mutations.append(value)
        for index, value in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.compare(value, dense)
        with self.assertRaises(ValueError):
            self.compare(candidate, dense, count=434)
        for invalid in (0, True, 1.5, 435):
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                self.compare(candidate, dense, count=invalid)

    def test_nan_and_infinity_in_either_branch_fail_without_complete_receipt(self):
        candidate, dense = tensors()
        for side in ("candidate", "dense"):
            for number in (float("nan"), float("inf"), -float("inf")):
                left, right = deepcopy(candidate), deepcopy(dense)
                (left if side == "candidate" else right)["body.bias"][1] = number
                with self.subTest(side=side, number=number), self.assertRaises(ValueError):
                    self.compare(left, right)

    def test_groups_must_be_a_nonempty_exact_disjoint_partition_without_reserved_all(self):
        candidate, dense = tensors()
        bad = ({}, {"tied": ("embedding.weight",)},
            {"tied": ("embedding.weight",), "body": ("body.weight", "body.bias", "embedding.weight")},
            {"all": tuple(dense)}, {"body": tuple(dense), "empty": ()},
            {"body": tuple(dense) + ("unknown",)})
        for groups in bad:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.compare(candidate, dense, groups=groups)

    def test_tolerances_require_exact_fields_finite_numbers_and_consistent_optional_pairs(self):
        original = tolerance()
        self.assertEqual(module.validate_tolerances(original), original)
        bad = []
        value = deepcopy(original); value.pop("rms_atol"); bad.append(value)
        value = deepcopy(original); value["extra"] = 1; bad.append(value)
        for key, item in (("elementwise_atol", -1.), ("elementwise_atol", True),
                          ("elementwise_rtol", float("nan")), ("rms_atol", .1),
                          ("group_relative_l2_maximum", -1.), ("group_cosine_minimum", 1.01),
                          ("require_positive_group_norms", 1)):
            value = deepcopy(original); value[key] = item; bad.append(value)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_tolerances(value)

    def test_chunk_sizes_change_neither_statistics_nor_autograd_and_no_tensor_is_retained(self):
        candidate, dense = tensors()
        for item in tuple(candidate.values()) + tuple(dense.values()):
            item.requires_grad_(True)
        baseline = self.compare(candidate, dense, chunk=1)
        for chunk in (2, 4, 7, module.MAX_CHUNK_ELEMENTS):
            result = self.compare(candidate, dense, chunk=chunk)
            self.assertStats(result["groups"]["all"], dense_oracle(candidate, dense))
            self.assertEqual(result["tensors"], baseline["tensors"])
            self.assertTrue(no_tensor_tree(result))
            self.assertFalse(result["native_provenance_established"])
        self.assertTrue(all(value.grad is None for value in tuple(candidate.values()) + tuple(dense.values())))

    def test_each_copy_is_bounded_and_streaming_counter_covers_unequal_final_chunks(self):
        candidate, dense = tensors()
        original, observed = module._copy_chunk, []
        def tracked(tensor, start, count, tracker):
            self.assertLessEqual(count, 2)
            result = original(tensor, start, count, tracker)
            self.assertEqual(result.numel(), count)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "cpu")
            self.assertFalse(result.requires_grad)
            observed.append((start, count))
            return result
        with patch.object(module, "_copy_chunk", side_effect=tracked):
            result = self.compare(candidate, dense, chunk=2)
        self.assertEqual(len(observed), 2 * (3 + 2 + 2))
        self.assertTrue(any(count == 1 for _, count in observed))
        self.assertLessEqual(result["streaming"]["peak_owned_host_bytes"], module.MAX_HOST_SCRATCH_BYTES)

    def test_generic434_fixture_counts_every_name_but_never_claims_native_identity(self):
        dense = {"parameter.%03d" % index: torch.tensor([float(index + 1)]) for index in range(434)}
        candidate = {name: value.clone() for name, value in dense.items()}
        groups = {"tied": ("parameter.000",), "backbone": tuple(name for name in dense if name != "parameter.000")}
        result = self.compare(candidate, dense, groups=groups, count=434)
        self.assertTrue(result["passed"])
        self.assertEqual(result["groups"]["all"]["elements"], 434)
        self.assertEqual(result["expected_tensor_count"], 434)
        self.assertFalse(result["native_provenance_established"])


class PrivateShadowTests(unittest.TestCase):
    def test_private_roundtrip_has_exact_little_endian_bytes_shapes_hashes_and_no_native_claim(self):
        initial = tensors()[1]
        with shadow_fixture(initial) as (_, path, _, receipt):
            manifest = receipt["manifest"]
            self.assertEqual(manifest["schema"], module.SHADOW_SCHEMA)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["directory"], str(path))
            self.assertEqual(manifest["dtype"], "float32-le")
            self.assertEqual(manifest["tensor_count"], 3)
            self.assertEqual(manifest["total_bytes"], 13 * 4)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
            self.assertEqual(receipt["manifest_sha256"], digest(path / "manifest.json"))
            self.assertEqual(receipt["manifest_bytes"], (path / "manifest.json").stat().st_size)
            for index, name in enumerate(sorted(initial)):
                row = manifest["tensors"][name]
                self.assertEqual(row["filename"], "tensor-%04d.f32" % index)
                self.assertEqual(row["shape"], list(initial[name].shape))
                self.assertEqual(row["bytes"], initial[name].numel() * 4)
                self.assertEqual((path / row["filename"]).read_bytes(), raw_bytes(initial[name]))
                self.assertEqual(row["sha256"], digest(path / row["filename"]))
            self.assertTrue(all(stat.S_IMODE(value.stat().st_mode) == 0o600 for value in path.iterdir()))
            self.assertFalse(manifest["native_provenance_established"])
            self.assertFalse(manifest["training_phase_or_initialization_provenance_verified"])
            self.assertTrue(no_tensor_tree(receipt))
            json.dumps(receipt, allow_nan=False)

    def test_bounded_reader_slices_reconstruct_values_without_aliasing_source_or_later_reads(self):
        initial = tensors()[1]
        with shadow_fixture(initial) as (_, path, _, receipt):
            reader = open_shadow(path, receipt)
            for name, value in initial.items():
                pieces = [reader.read_slice(name, start, min(2, value.numel() - start))
                    for start in range(0, value.numel(), 2)]
                actual = torch.cat(pieces)
                torch.testing.assert_close(actual, value.flatten(), atol=0., rtol=0.)
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.dtype, torch.float32)
            first = reader.read_slice("body.bias", 0, 2)
            first.fill_(999.)
            torch.testing.assert_close(reader.read_slice("body.bias", 0, 2), initial["body.bias"][:2], atol=0., rtol=0.)
            untrusted_metadata = reader.manifest
            untrusted_metadata["tensors"]["body.bias"]["filename"] = "../caller-mutation.f32"
            self.assertEqual(reader.manifest["tensors"]["body.bias"]["filename"], "tensor-0000.f32")
            torch.testing.assert_close(reader.read_slice("body.bias", 0, 2), initial["body.bias"][:2], atol=0., rtol=0.)
            for name, start, count in (("unknown", 0, 1), ("body.bias", -1, 1),
                    ("body.bias", 0, 0), ("body.bias", 0, 3), ("body.bias", 2, 2), ("body.bias", True, 1)):
                with self.subTest(name=name, start=start, count=count), self.assertRaises(ValueError):
                    reader.read_slice(name, start, count)
            self.assertTrue(reader.verify_unchanged())

    def test_existing_or_symlink_shadow_and_symlink_parent_are_never_reused(self):
        initial = tensors()[1]
        with shadow_fixture(initial) as (parent, path, budget, receipt):
            before = {value.name: digest(value) for value in path.iterdir()}
            with self.assertRaises(ValueError):
                module.write_initial_shadow(initial, path, budget=budget, expected_count=3, chunk_elements=2)
            alias = parent / "alias"
            alias.symlink_to(path, target_is_directory=True)
            with self.assertRaises(ValueError):
                open_shadow(alias, receipt)
            with self.assertRaises(ValueError):
                module.write_initial_shadow(initial, alias / "another", budget=budget, expected_count=3, chunk_elements=2)
            self.assertEqual(before, {value.name: digest(value) for value in path.iterdir()})

    def test_owner_only_directory_manifest_and_raw_file_permissions_are_enforced(self):
        for target, mode in (("directory", 0o770), ("manifest.json", 0o640), ("tensor-0000.f32", 0o604)):
            with self.subTest(target=target), shadow_fixture() as (_, path, _, receipt):
                value = path if target == "directory" else path / target
                value.chmod(mode)
                with self.assertRaises(ValueError):
                    open_shadow(path, receipt)
                value.chmod(0o700 if target == "directory" else 0o600)
                self.assertTrue(open_shadow(path, receipt).verify_unchanged())

    def test_nonfinite_or_interrupted_writer_leaves_no_complete_manifest_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(); parent.chmod(0o700)
            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)):
                budget = module.ScratchBudget(parent)
                for number in (float("nan"), float("inf")):
                    path = parent / ("nan" if math.isnan(number) else "inf")
                    initial = tensors()[1]; initial["body.bias"][0] = number
                    with self.assertRaises(ValueError):
                        module.write_initial_shadow(initial, path, budget=budget, expected_count=3, chunk_elements=2)
                    self.assertFalse((path / "manifest.json").exists())
                    with self.assertRaises(ValueError):
                        module.write_initial_shadow(tensors()[1], path, budget=budget, expected_count=3, chunk_elements=2)
                path = parent / "interrupted"
                original, calls = module._copy_chunk, []
                def interrupted(tensor, start, count, tracker):
                    calls.append(count)
                    if len(calls) == 2:
                        raise RuntimeError("original synthetic transfer interruption")
                    return original(tensor, start, count, tracker)
                with patch.object(module, "_copy_chunk", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "original synthetic transfer"):
                    module.write_initial_shadow(tensors()[1], path, budget=budget, expected_count=3, chunk_elements=2)
                self.assertTrue(path.is_dir())
                self.assertFalse((path / "manifest.json").exists())
                self.assertEqual(len(calls), 2)

    def test_missing_extra_and_truncated_raw_inventory_is_refused_before_slice_access(self):
        for kind in ("missing", "extra", "truncated", "missing_manifest"):
            with self.subTest(kind=kind), shadow_fixture() as (_, path, _, receipt):
                raw = path / "tensor-0000.f32"
                if kind == "missing": raw.unlink()
                elif kind == "extra": (path / "unexpected.private").write_bytes(b"x")
                elif kind == "truncated": raw.write_bytes(raw.read_bytes()[:-4])
                else: (path / "manifest.json").unlink()
                with self.assertRaises((ValueError, OSError)):
                    open_shadow(path, receipt)

    def test_file_or_manifest_tampering_cannot_be_accepted_using_the_original_file_pin(self):
        for target in ("tensor-0000.f32", "manifest.json"):
            with self.subTest(target=target), shadow_fixture() as (_, path, _, receipt):
                file = path / target
                payload = bytearray(file.read_bytes())
                payload[len(payload) // 2] ^= 1
                file.write_bytes(payload)
                with self.assertRaises((ValueError, UnicodeError, json.JSONDecodeError)):
                    open_shadow(path, receipt)
        with shadow_fixture() as (_, path, _, receipt), self.assertRaises(ValueError):
            module.open_initial_shadow(path, expected_manifest_sha256="0" * 64, expected_count=3, chunk_elements=2)

    def test_resealed_malicious_manifest_paths_shapes_counts_or_native_claims_are_rejected(self):
        mutations = (
            lambda value: value.update(directory="/some/other/private/path"),
            lambda value: value.update(complete=False),
            lambda value: value.update(tensor_count=2),
            lambda value: value.update(tensor_count=True),
            lambda value: value.update(native_provenance_established=True),
            lambda value: value.update(training_phase_or_initialization_provenance_verified=True),
            lambda value: value["tensors"]["body.bias"].update(filename="../escape.f32"),
            lambda value: value["tensors"]["body.bias"].update(filename="/tmp/escape.f32"),
            lambda value: value["tensors"]["body.bias"].update(shape=[4]),
            lambda value: value["tensors"]["body.bias"].update(bytes=True),
            lambda value: value.update(total_bytes=value["total_bytes"] + 4),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), shadow_fixture() as (_, path, _, receipt):
                value = deepcopy(receipt["manifest"]); mutate(value)
                rewrite_json(path / "manifest.json", value)
                changed = {**receipt, "manifest_sha256": digest(path / "manifest.json")}
                with self.assertRaises(ValueError):
                    open_shadow(path, changed)

    def test_duplicate_nonfinite_and_nonobject_shadow_json_fail_even_with_matching_new_file_hash(self):
        for payload in (b'{"schema":"a","schema":"b"}\n', b'{"value":NaN}\n', b'[]\n', b'null\n'):
            with self.subTest(payload=payload), shadow_fixture() as (_, path, _, receipt):
                (path / "manifest.json").write_bytes(payload)
                changed = {**receipt, "manifest_sha256": digest(path / "manifest.json")}
                with self.assertRaises(ValueError):
                    open_shadow(path, changed)

    def test_open_reader_detects_later_raw_manifest_inventory_or_symlink_changes(self):
        for kind in ("raw", "manifest", "extra", "symlink"):
            with self.subTest(kind=kind), shadow_fixture() as (parent, path, _, receipt):
                reader = open_shadow(path, receipt)
                if kind == "raw":
                    raw = path / "tensor-0000.f32"; raw.write_bytes(b"\0" * raw.stat().st_size)
                elif kind == "manifest":
                    rewrite_json(path / "manifest.json", {**receipt["manifest"], "complete": False})
                elif kind == "extra":
                    (path / "extra").write_bytes(b"x")
                else:
                    raw = path / "tensor-0000.f32"
                    private = parent / "substitute.private"; private.write_bytes(raw.read_bytes()); private.chmod(0o600)
                    raw.unlink(); raw.symlink_to(private)
                with self.assertRaises(ValueError):
                    reader.verify_unchanged()

    def test_actual_stored_deltas_match_dense_subtraction_for_every_tensor_and_group(self):
        initial = tensors()[1]
        candidate = {name: value.clone() for name, value in initial.items()}
        dense = {name: value.clone() for name, value in initial.items()}
        for index, name in enumerate(sorted(initial)):
            dense[name].add_((index + 1) / 8)
            candidate[name].add_((index + 1) / 8 + 1 / 32)
        expected_candidate = {name: candidate[name].double() - value.double() for name, value in initial.items()}
        expected_dense = {name: dense[name].double() - value.double() for name, value in initial.items()}
        with shadow_fixture(initial) as (_, path, _, receipt):
            result = module.compare_stored_deltas(candidate, dense, open_shadow(path, receipt),
                groups=GROUPS, tolerances=tolerance(require_positive_group_norms=True), expected_count=3, chunk_elements=2)
            self.assertTrue(result["passed"])
            self.assertEqual(result["comparison"], "actual_stored_fp32_deltas")
            self.assertEqual(result["initial_shadow_manifest_sha256"], receipt["manifest_sha256"])
            self.assertFalse(result["optimizer_formula_used"])
            check = StreamingNumericalTests()
            for name in initial:
                check.assertStats(result["tensors"][name], dense_oracle(expected_candidate, expected_dense, names=(name,)))
            for name, members in {"all": tuple(sorted(initial)), **GROUPS}.items():
                check.assertStats(result["groups"][name], dense_oracle(expected_candidate, expected_dense, names=members))

    def test_delta_evidence_uses_real_fp32_rounding_and_rejects_zero_update_group_promotion(self):
        initial = {"embedding.weight": torch.tensor([1e8, -1e8]),
            "body.weight": torch.tensor([1.]), "body.bias": torch.tensor([0.])}
        dense = {name: value.clone() for name, value in initial.items()}
        candidate = {name: value.clone() for name, value in initial.items()}
        dense["embedding.weight"][0] += 16.
        candidate["embedding.weight"][0] += 8.
        dense["body.weight"][0] += .5
        candidate["body.weight"][0] += .25
        with shadow_fixture(initial) as (_, path, _, receipt):
            reader = open_shadow(path, receipt)
            result = module.compare_stored_deltas(candidate, dense, reader, groups=GROUPS,
                tolerances=tolerance(elementwise_atol=8.), expected_count=3, chunk_elements=2)
            self.assertEqual(result["groups"]["tied"]["dense_l2"], 16.)
            self.assertEqual(result["groups"]["tied"]["candidate_l2"], 8.)
            self.assertEqual(result["groups"]["tied"]["difference_l2"], 8.)
            frozen = module.compare_stored_deltas(initial, initial, reader, groups=GROUPS,
                tolerances=tolerance(require_positive_group_norms=True), expected_count=3, chunk_elements=2)
            self.assertFalse(frozen["passed"])
            self.assertIsNone(frozen["groups"]["all"]["relative_l2"])

    def test_shadow_wrong_count_name_shape_or_unpinned_plain_dict_cannot_compare_updates(self):
        candidate, dense = tensors()
        with shadow_fixture() as (_, path, _, receipt):
            with self.assertRaises(ValueError):
                open_shadow(path, receipt, count=434)
            reader = open_shadow(path, receipt)
            cases = []
            left, right = deepcopy(candidate), deepcopy(dense)
            left["renamed"] = left.pop("body.bias"); right["renamed"] = right.pop("body.bias")
            cases.append((left, right, {"tied": ("embedding.weight",), "body": ("body.weight", "renamed")}))
            left, right = deepcopy(candidate), deepcopy(dense)
            left["body.weight"] = left["body.weight"].reshape(4); right["body.weight"] = right["body.weight"].reshape(4)
            cases.append((left, right, GROUPS))
            for left, right, groups in cases:
                with self.assertRaises(ValueError):
                    module.compare_stored_deltas(left, right, reader, groups=groups,
                        tolerances=tolerance(), expected_count=3, chunk_elements=2)
            with self.assertRaises(ValueError):
                module.compare_stored_deltas(candidate, dense, receipt["manifest"], groups=GROUPS,
                    tolerances=tolerance(), expected_count=3, chunk_elements=2)

    def test_comparison_and_writer_refuse_source_mutation_during_chunk_transfer(self):
        candidate, dense = tensors()
        original, calls = module._copy_chunk, []
        def mutate(tensor, start, count, tracker):
            value = original(tensor, start, count, tracker)
            calls.append(count)
            if len(calls) == 1:
                with torch.no_grad(): tensor.add_(.5)
            return value
        with patch.object(module, "_copy_chunk", side_effect=mutate), self.assertRaises(ValueError):
            module.compare_named_tensors(candidate, dense, groups=GROUPS, tolerances=tolerance(), expected_count=3, chunk_elements=2)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(); parent.chmod(0o700)
            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)):
                budget = module.ScratchBudget(parent); calls.clear()
                with patch.object(module, "_copy_chunk", side_effect=mutate), self.assertRaises(ValueError):
                    module.write_initial_shadow(tensors()[1], parent / "partial", budget=budget, expected_count=3, chunk_elements=2)
                self.assertFalse((parent / "partial/manifest.json").exists())


class FiniteScratchTests(unittest.TestCase):
    def test_initial_free_scratch_boundary_and_parent_permissions_are_mandatory(self):
        self.assertEqual(module.INITIAL_FREE_BYTES, 160 << 30)
        self.assertEqual(module.STAGE_WRITE_CEILING_BYTES, 112 << 30)
        self.assertEqual(module.MINIMUM_RESERVE_BYTES, 32 << 30)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(); parent.chmod(0o700)
            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=(160 << 30) - 1)), self.assertRaises(ValueError):
                module.ScratchBudget(parent)
            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=160 << 30)):
                self.assertEqual(module.ScratchBudget(parent).evidence()["initial_free_bytes"], 160 << 30)
                parent.chmod(0o770)
                with self.assertRaises(ValueError): module.ScratchBudget(parent)
                parent.chmod(0o700)

    def test_planned_and_actual_byte_ceiling_and_free_reserve_boundaries_are_inclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(); parent.chmod(0o700)
            free = SimpleNamespace(free=200 << 30)
            with patch.object(module.shutil, "disk_usage", side_effect=lambda _: free):
                budget = module.ScratchBudget(parent)
                budget.charge_written((112 << 30) - 16)
                free.free = (32 << 30) + 16
                budget.check_planned_write(16)
                with self.assertRaises(ValueError): budget.check_planned_write(17)
                free.free -= 1
                with self.assertRaises(ValueError): budget.check_planned_write(16)
                free.free += 1
                budget.charge_written(16)
                budget.check_planned_write(0)
                self.assertEqual(budget.evidence()["accounted_bytes_written"], 112 << 30)
                for invalid in (1, True, -1, .5):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError): budget.charge_written(invalid)
                self.assertFalse(budget.evidence()["physical_disk_blocks_reserved"])
                self.assertFalse(budget.evidence()["unaccounted_external_writes_verified"])

    def test_writer_accounting_equals_real_raw_and_manifest_bytes_with_bounded_copies(self):
        initial = tensors()[1]
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(); parent.chmod(0o700)
            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)):
                budget = module.ScratchBudget(parent)
                observed, original = [], module._copy_chunk
                def traced(tensor, start, count, tracker):
                    self.assertLessEqual(count, 2); observed.append(count)
                    return original(tensor, start, count, tracker)
                with patch.object(module, "_copy_chunk", side_effect=traced):
                    receipt = module.write_initial_shadow(initial, parent / "initial", budget=budget, expected_count=3, chunk_elements=2)
                actual_bytes = sum(value.stat().st_size for value in (parent / "initial").iterdir())
                self.assertEqual(budget.evidence()["accounted_bytes_written"], actual_bytes)
                self.assertEqual(actual_bytes, 13 * 4 + receipt["manifest_bytes"])
                self.assertEqual(sum(observed), 13)
                self.assertEqual(len(observed), 7)
                self.assertLessEqual(receipt["manifest"]["streaming"]["peak_owned_host_bytes"], module.MAX_HOST_SCRATCH_BYTES)
                self.assertFalse(receipt["manifest"]["streaming"]["process_rss_or_allocator_peak_measured"])


if __name__ == "__main__":
    unittest.main()

