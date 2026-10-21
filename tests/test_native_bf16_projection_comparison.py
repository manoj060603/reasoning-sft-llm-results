"""Prospective24 independent synthetic CPU fixtures, authored but unexecuted.

No head/model/tokenizer, experiment cache, source record, CUDA device or scheduler
is used. BF16 CPU values and transfer seams do not establish native autocast,
original434 identity, backward/recompute, training or GPU/RSS/headroom admission.
The scalar oracle uses Python FP64/frexp/fsum, not the production tensor formula.
Run only in a separately approved native CPU allocation after source review.
"""
import gc
import json
import math
import unittest
import weakref
from unittest.mock import patch

import torch

from extension import native_bf16_projection_comparison as module


def bf16_ulp(magnitude):
    """Independent scalar exponent oracle, including zero/subnormal magnitudes."""
    if magnitude < math.ldexp(1., -126):
        return math.ldexp(1., -133)
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1., exponent - 8)


def oracle(dense, candidate):
    left = [float(value) for value in dense.detach().reshape(-1).tolist()]
    right = [float(value) for value in candidate.detach().reshape(-1).tolist()]
    if len(left) != len(right) or not left:
        raise AssertionError("The fixture oracle itself requires matching nonempty arrays")
    if not all(math.isfinite(value) for value in left + right):
        raise AssertionError("Nonfinite fixtures must exercise rejection instead of this oracle")
    errors = [a - b for a, b in zip(left, right)]
    ulps = [bf16_ulp(max(abs(a), abs(b))) for a, b in zip(left, right)]
    thresholds = [max(.001, ulp) for ulp in ulps]
    dense_square = math.fsum(value * value for value in left)
    candidate_square = math.fsum(value * value for value in right)
    difference_square = math.fsum(value * value for value in errors)
    dense_norm, candidate_norm, difference_norm = map(math.sqrt,
        (dense_square, candidate_square, difference_square))
    relative = 0. if dense_norm == difference_norm == 0 else None if dense_norm == 0 else difference_norm / dense_norm
    failures = sum(abs(value) > threshold for value, threshold in zip(errors, thresholds))
    return {"observed_elements": len(left), "elementwise_failures": failures,
        "max_absolute_error": max(abs(value) for value in errors),
        "dense_l2": dense_norm, "candidate_l2": candidate_norm, "difference_l2": difference_norm,
        "global_relative_l2": relative, "minimum_bf16_ulp": min(ulps), "maximum_bf16_ulp": max(ulps),
        "maximum_allowed_absolute_difference": max(thresholds),
        "passed": failures == 0 and relative is not None and relative <= .005}


def compare(dense, candidate, *, chunk=3, label="student"):
    result = module.ProjectionComparison(dense.numel(), max_chunk_elements=chunk, label=label)
    result.observe(dense, candidate)
    return result.finish()


def contains_tensor(value, seen=None):
    if isinstance(value, torch.Tensor):
        return True
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        return any(contains_tensor(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_tensor(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return contains_tensor(vars(value), seen)
    return False


class ProjectionComparisonTests(unittest.TestCase):
    def assertOracle(self, actual, expected):
        for key in ("observed_elements", "elementwise_failures", "passed"):
            self.assertEqual(actual[key], expected[key], key)
        for key in ("max_absolute_error", "dense_l2", "candidate_l2", "difference_l2", "global_relative_l2"):
            if expected[key] is None:
                self.assertIsNone(actual[key], key)
            elif expected[key] == 0:
                self.assertEqual(actual[key], 0., key)
            else:
                self.assertTrue(math.isclose(actual[key], expected[key], rel_tol=3e-13, abs_tol=0.),
                    (key, actual[key], expected[key]))
        for key in ("minimum_bf16_ulp", "maximum_bf16_ulp", "maximum_allowed_absolute_difference"):
            self.assertEqual(actual[key], expected[key], key)

    def test_fixed_policy_constructor_counts_labels_and_no_tolerance_overrides(self):
        self.assertEqual(module.MAX_CHUNK_ELEMENTS, 1_048_576)
        self.assertEqual(module.MAX_HOST_SCRATCH_BYTES, 128 * 1024**2)
        self.assertEqual(module.ABSOLUTE_FLOOR, .001)
        self.assertEqual(module.GLOBAL_RELATIVE_L2_MAXIMUM, .005)
        for expected in (0, -1, True, 1., (1 << 63)):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                module.ProjectionComparison(expected)
        for chunk in (0, -1, True, .5, module.MAX_CHUNK_ELEMENTS + 1):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                module.ProjectionComparison(1, max_chunk_elements=chunk)
        for label in ("candidate", "", None, 1):
            with self.subTest(label=label), self.assertRaises(ValueError):
                module.ProjectionComparison(1, label=label)
        with self.assertRaises(TypeError):
            module.ProjectionComparison(1, atol=100.)

    def test_fp32_dense_statistics_match_independent_fp64_scalar_oracle(self):
        dense = torch.tensor([[.1, -.3, .5, 1.1, 2., 200.], [0., 4., -8., .125, 32., -64.]], dtype=torch.float32)
        candidate = dense.to(torch.bfloat16)
        actual = compare(dense, candidate, chunk=5)
        self.assertOracle(actual, oracle(dense, candidate))
        self.assertEqual(actual["vocabulary"], 6)
        self.assertEqual(actual["blocks"], 1)
        self.assertEqual(actual["chunks"], 3)

    def test_dense_bf16_is_allowed_and_identical_values_have_zero_error(self):
        dense = torch.tensor([[1., -2., 3.5, .125]], dtype=torch.bfloat16)
        result = compare(dense, dense.clone(), chunk=2)
        self.assertOracle(result, oracle(dense, dense))
        self.assertTrue(result["passed"])
        self.assertEqual(result["dense_dtypes"], ["torch.bfloat16"])
        self.assertEqual(result["candidate_dtype"], "torch.bfloat16")

    def test_powers_of_two_and_adjacent_magnitudes_use_independent_frexp_ulp(self):
        values = (math.ldexp(1., -126), math.ldexp(1., -10), .125, .25, .5, 1., 1.9921875, 2., 4., 128., 256., -2.)
        for value in values:
            with self.subTest(value=value):
                dense = torch.tensor([[value]], dtype=torch.float32)
                candidate = dense.to(torch.bfloat16)
                result = compare(dense, candidate, chunk=1)
                self.assertOracle(result, oracle(dense, candidate))
                magnitude = max(abs(float(dense.item())), abs(float(candidate.item())))
                self.assertEqual(result["minimum_bf16_ulp"], bf16_ulp(magnitude))
                self.assertEqual(result["maximum_bf16_ulp"], bf16_ulp(magnitude))

    def test_zero_and_subnormal_ulp_reports_are_not_replaced_by_the_absolute_floor(self):
        for value in (0., math.ldexp(1., -149), math.ldexp(1., -140), math.ldexp(1., -133), math.ldexp(1., -127)):
            with self.subTest(value=value):
                dense = torch.tensor([[value]], dtype=torch.float32)
                candidate = dense.to(torch.bfloat16)
                result = compare(dense, candidate, chunk=1)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertEqual(result["minimum_bf16_ulp"], math.ldexp(1., -133))
                self.assertEqual(result["maximum_bf16_ulp"], math.ldexp(1., -133))
                self.assertEqual(result["maximum_allowed_absolute_difference"], .001)

    def test_absolute_floor_boundary_uses_actual_fp32_neighbors_not_decimal_assumptions(self):
        rounded = torch.tensor(.001, dtype=torch.float32)
        below = torch.nextafter(rounded, torch.tensor(0., dtype=torch.float32))
        candidate = torch.tensor([[0., 4.]], dtype=torch.bfloat16)
        for value, failures in ((below, 0), (rounded, 1)):
            with self.subTest(value=float(value)):
                dense = torch.tensor([[float(value), 4.]], dtype=torch.float32)
                result = compare(dense, candidate)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertEqual(result["elementwise_failures"], failures)
                self.assertEqual(result["passed"], failures == 0)

    def test_one_bf16_ulp_is_inclusive_but_two_ulps_fail_with_global_norm_still_small(self):
        dense = torch.tensor([[1., 4.]], dtype=torch.float32)
        for value, failures in ((1.0078125, 0), (1.015625, 1)):
            with self.subTest(value=value):
                candidate = torch.tensor([[value, 4.]], dtype=torch.bfloat16)
                result = compare(dense, candidate)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertLess(result["global_relative_l2"], .005)
                self.assertEqual(result["elementwise_failures"], failures)
                self.assertEqual(result["passed"], failures == 0)

    def test_ulp_scale_uses_maximum_of_both_magnitudes_at_an_exponent_boundary(self):
        for left, right in ((1.984375, 2.), (2., 1.984375)):
            with self.subTest(dense=left, candidate=right):
                dense = torch.tensor([[left, 4.]], dtype=torch.float32)
                candidate = torch.tensor([[right, 4.]], dtype=torch.bfloat16)
                result = compare(dense, candidate)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertEqual(result["elementwise_failures"], 0)
                self.assertTrue(result["passed"])

    def test_negative_values_sign_changes_and_large_finite_values_keep_the_scalar_rule(self):
        pairs = (
            ([-1., -2., -.25, 8.], [-1.0078125, -2.015625, -.25, 8.]),
            ([-1., 1., 8., 8.], [1., -1., 8., 8.]),
            ([torch.finfo(torch.float32).max], [torch.finfo(torch.bfloat16).max]),
        )
        for left, right in pairs:
            with self.subTest(dense=left):
                dense = torch.tensor([left], dtype=torch.float32)
                candidate = torch.tensor([right], dtype=torch.bfloat16)
                result = compare(dense, candidate)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertTrue(math.isfinite(result["dense_l2"]))
                self.assertTrue(math.isfinite(result["candidate_l2"]))

    def test_global_relative_l2_exact_boundary_is_inclusive_and_above_it_fails(self):
        for dense_value, candidate_value, passed in ((200., 201., True), (199., 200., False), (201., 202., True)):
            with self.subTest(dense=dense_value):
                dense = torch.tensor([[dense_value]], dtype=torch.float32)
                candidate = torch.tensor([[candidate_value]], dtype=torch.bfloat16)
                result = compare(dense, candidate)
                self.assertOracle(result, oracle(dense, candidate))
                self.assertEqual(result["elementwise_failures"], 0)
                self.assertEqual(result["passed"], passed)
                if dense_value == 200.:
                    self.assertEqual(result["global_relative_l2"], .005)

    def test_a_single_bad_element_cannot_be_averaged_away_or_excluded_from_statistics(self):
        dense = torch.ones((1, 101), dtype=torch.float32)
        candidate = torch.ones((1, 101), dtype=torch.bfloat16)
        candidate[0, 50] = 1.015625
        result = compare(dense, candidate, chunk=7)
        self.assertOracle(result, oracle(dense, candidate))
        self.assertLess(result["global_relative_l2"], .005)
        self.assertEqual(result["observed_elements"], 101)
        self.assertEqual(result["elementwise_failures"], 1)
        self.assertFalse(result["passed"])

    def test_zero_dense_norm_has_explicit_both_zero_success_and_nonzero_candidate_failure(self):
        zeros = torch.zeros((1, 3), dtype=torch.float32)
        equal = compare(zeros, zeros.to(torch.bfloat16))
        self.assertTrue(equal["passed"])
        self.assertEqual(equal["global_relative_l2"], 0.)
        candidate = torch.tensor([[math.ldexp(1., -10), 0., 0.]], dtype=torch.bfloat16)
        failed = compare(zeros, candidate)
        self.assertOracle(failed, oracle(zeros, candidate))
        self.assertEqual(failed["elementwise_failures"], 0)
        self.assertIsNone(failed["global_relative_l2"])
        self.assertFalse(failed["passed"])

    def test_nan_and_infinity_in_either_branch_raise_and_never_yield_a_complete_receipt(self):
        for side in ("dense", "candidate"):
            for value in (float("nan"), float("inf"), -float("inf")):
                dense = torch.ones((1, 4), dtype=torch.float32)
                candidate = torch.ones((1, 4), dtype=torch.bfloat16)
                (dense if side == "dense" else candidate)[0, 3] = value
                instance = module.ProjectionComparison(4, max_chunk_elements=2)
                with self.subTest(side=side, value=value), self.assertRaises(ValueError):
                    instance.observe(dense, candidate)
                with self.assertRaises(ValueError): instance.finish()
                # The valid first chunk must not leak into a later successful
                # count4 receipt containing statistics for6 values.
                with self.assertRaises(ValueError):
                    instance.observe(torch.ones((1, 4)), torch.ones((1, 4), dtype=torch.bfloat16))

    def test_dense_and_candidate_dtype_guards_never_silently_cast_unsupported_inputs(self):
        valid_dense = torch.ones((1, 3), dtype=torch.float32)
        valid_candidate = torch.ones((1, 3), dtype=torch.bfloat16)
        for dtype in (torch.float16, torch.float64, torch.int64):
            with self.subTest(dense_dtype=dtype), self.assertRaises(ValueError):
                compare(valid_dense.to(dtype), valid_candidate)
        for dtype in (torch.float16, torch.float32, torch.float64, torch.int64):
            with self.subTest(candidate_dtype=dtype), self.assertRaises(ValueError):
                compare(valid_dense, valid_candidate.to(dtype))

    def test_projection_shape_row_limit_layout_and_normal_tensor_guards(self):
        invalid_shapes = ((3,), (1, 1, 3), (0, 3), (129, 1), (1, 0))
        for shape in invalid_shapes:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                instance = module.ProjectionComparison(max(1, math.prod(shape)))
                instance.observe(torch.ones(shape), torch.ones(shape, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            module.ProjectionComparison(6).observe(torch.ones((2, 3)), torch.ones((1, 6), dtype=torch.bfloat16))
        dense = torch.ones((2, 3)); candidate = torch.ones((2, 3), dtype=torch.bfloat16)
        for left, right in ((dense.t(), candidate.t()), (dense.to_sparse(), candidate)):
            with self.assertRaises(ValueError): module.ProjectionComparison(6).observe(left, right)
        with torch.inference_mode():
            inference = torch.ones((2, 3), dtype=torch.bfloat16)
        with self.assertRaises(ValueError): module.ProjectionComparison(6).observe(dense, inference)

    def test_exact_device_and_cpu_cuda_only_contract_rejects_metadata_device_fixtures(self):
        # Meta is a cheap malformed-input fixture, not a simulated CUDA success.
        dense = torch.ones((1, 2), dtype=torch.float32)
        candidate = torch.ones((1, 2), dtype=torch.bfloat16)
        for left, right in ((dense, torch.empty((1, 2), dtype=torch.bfloat16, device="meta")),
                (torch.empty((1, 2), device="meta"), candidate)):
            with self.assertRaises(ValueError): module.ProjectionComparison(2).observe(left, right)
        result = compare(dense, candidate)
        self.assertEqual(result["observed_devices"], ["cpu"])
        self.assertFalse(result["actual_autocast_or_gpu_identity_established"])

    def test_first_observed_vocabulary_width_is_bound_for_every_later_block(self):
        instance = module.ProjectionComparison(5)
        instance.observe(torch.ones((1, 3)), torch.ones((1, 3), dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            instance.observe(torch.ones((1, 2)), torch.ones((1, 2), dtype=torch.bfloat16))
        with self.assertRaises(ValueError): instance.finish()

    def test_incomplete_and_oversized_coverage_are_errors_not_partial_success(self):
        empty = module.ProjectionComparison(1)
        with self.assertRaises(ValueError): empty.finish()
        with self.assertRaises(ValueError):
            empty.observe(torch.zeros((1, 1)), torch.zeros((1, 1), dtype=torch.bfloat16))
        partial = module.ProjectionComparison(6)
        partial.observe(torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bfloat16))
        with self.assertRaises(ValueError): partial.finish()
        with self.assertRaises(ValueError):
            partial.observe(torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bfloat16))
        excessive = module.ProjectionComparison(2)
        with self.assertRaises(ValueError):
            excessive.observe(torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bfloat16))
        with self.assertRaises(ValueError): excessive.finish()
        with self.assertRaises(ValueError):
            excessive.observe(torch.zeros((1, 2)), torch.zeros((1, 2), dtype=torch.bfloat16))

    def test_successful_finalization_is_single_use_and_returned_metadata_is_not_live_state(self):
        instance = module.ProjectionComparison(2)
        dense, candidate = torch.ones((1, 2)), torch.ones((1, 2), dtype=torch.bfloat16)
        instance.observe(dense, candidate)
        result = instance.finish()
        original = json.loads(json.dumps(result))
        with self.assertRaises(ValueError): instance.finish()
        with self.assertRaises(ValueError): instance.observe(dense, candidate)
        self.assertEqual(result, original)

    def test_cross_chunk_and_block_aggregation_matches_one_global_scalar_norm(self):
        dense = torch.tensor([[.1, 1., 4., 200., -.5], [.3, 2., -8., 16., 32.], [0., -1., 3., .125, -64.]])
        candidate = dense.to(torch.bfloat16)
        expected = oracle(dense, candidate)
        for chunk in (1, 2, 4, 7, module.MAX_CHUNK_ELEMENTS):
            with self.subTest(chunk=chunk):
                instance = module.ProjectionComparison(15, max_chunk_elements=chunk)
                instance.observe(dense[:2].contiguous(), candidate[:2].contiguous())
                instance.observe(dense[2:].contiguous(), candidate[2:].contiguous())
                result = instance.finish()
                self.assertOracle(result, expected)
                self.assertEqual(result["blocks"], 2)
                self.assertEqual(result["vocabulary"], 5)

    def test_actual_copy_seam_never_exceeds_one_million_and_accounts_final_short_slice(self):
        count = module.MAX_CHUNK_ELEMENTS + 3
        dense = torch.zeros((1, count), dtype=torch.float32)
        candidate = torch.zeros((1, count), dtype=torch.bfloat16)
        original, observed = module.ProjectionComparison._copy_chunk, []
        def traced(instance, tensor, start, size):
            self.assertLessEqual(size, module.MAX_CHUNK_ELEMENTS)
            value = original(instance, tensor, start, size)
            self.assertEqual(value.numel(), size)
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            observed.append((start, size))
            return value
        with patch.object(module.ProjectionComparison, "_copy_chunk", new=traced):
            result = compare(dense, candidate, chunk=module.MAX_CHUNK_ELEMENTS)
        self.assertEqual(observed, [(0, module.MAX_CHUNK_ELEMENTS)] * 2 + [(module.MAX_CHUNK_ELEMENTS, 3)] * 2)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["observed_elements"], count)
        self.assertEqual(result["maximum_chunk_elements"], module.MAX_CHUNK_ELEMENTS)
        self.assertLessEqual(result["declared_owned_host_scratch_bytes"], 128 * 1024**2)
        self.assertFalse(result["process_rss_or_cuda_headroom_measured"])

    def test_inputs_are_detached_no_tensor_or_graph_is_retained_and_metadata_is_finite_json(self):
        dense = torch.tensor([[1., 2., 4.]], dtype=torch.float32, requires_grad=True)
        candidate = torch.tensor([[1., 2., 4.]], dtype=torch.bfloat16, requires_grad=True)
        instance = module.ProjectionComparison(3, max_chunk_elements=2)
        left, right = weakref.ref(dense), weakref.ref(candidate)
        instance.observe(dense, candidate)
        self.assertIsNone(dense.grad); self.assertIsNone(candidate.grad)
        self.assertFalse(contains_tensor(vars(instance)))
        del dense, candidate
        gc.collect()
        self.assertIsNone(left()); self.assertIsNone(right())
        result = instance.finish()
        self.assertFalse(contains_tensor(result))
        json.dumps(result, allow_nan=False)
        self.assertFalse(result["tensor_values_retained_in_receipt"])
        self.assertFalse(result["actual_autocast_or_gpu_identity_established"])

    def test_input_version_shape_and_storage_changes_during_copy_are_detected(self):
        original = module.ProjectionComparison._copy_chunk
        for change in ("version", "shape", "storage"):
            dense = torch.ones((1, 4), dtype=torch.float32)
            candidate = torch.ones((1, 4), dtype=torch.bfloat16)
            observed = []
            def changed(instance, tensor, start, size):
                value = original(instance, tensor, start, size)
                observed.append(size)
                if len(observed) == 1:
                    with torch.no_grad():
                        if change == "version": dense.add_(1.)
                        elif change == "shape": dense.resize_(2, 2)
                        else: dense.set_(dense.clone())
                return value
            instance = module.ProjectionComparison(4, max_chunk_elements=2)
            with self.subTest(change=change), patch.object(module.ProjectionComparison, "_copy_chunk", new=changed), self.assertRaises(ValueError):
                instance.observe(dense, candidate)
            with self.assertRaises(ValueError): instance.finish()
            with self.assertRaises(ValueError):
                instance.observe(torch.ones((1, 4)), torch.ones((1, 4), dtype=torch.bfloat16))

    def test_reference_role_mixed_allowed_dense_dtypes_and_qualifications_are_explicit(self):
        instance = module.ProjectionComparison(4, label="reference")
        instance.observe(torch.tensor([[1., 2.]], dtype=torch.float32), torch.tensor([[1., 2.]], dtype=torch.bfloat16))
        instance.observe(torch.tensor([[4., 8.]], dtype=torch.bfloat16), torch.tensor([[4., 8.]], dtype=torch.bfloat16))
        result = instance.finish()
        self.assertEqual(result["schema"], module.SCHEMA)
        self.assertEqual(result["label"], "reference")
        self.assertEqual(result["expected_elements"], result["observed_elements"])
        self.assertEqual(result["dense_dtypes"], ["torch.bfloat16", "torch.float32"])
        self.assertEqual(result["vocabulary"], 2)
        self.assertTrue(result["passed"])
        self.assertTrue(result["conservative_application_buffer_accounting"])
        self.assertFalse(result["actual_autocast_or_gpu_identity_established"])
        self.assertFalse(result["process_rss_or_cuda_headroom_measured"])
        self.assertFalse(result["tensor_values_retained_in_receipt"])


if __name__ == "__main__":
    unittest.main()
