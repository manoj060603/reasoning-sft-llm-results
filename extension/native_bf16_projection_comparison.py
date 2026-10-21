"""Prospective bounded dense/chunk projection comparator; no workload/admission.

This pure helper compares already-computed short projection blocks under the
root-frozen BF16 ULP rule. It does not project a head, load a model, establish
actual autocast/B200 identity, retain an output tensor, perform a causal shift or
change an objective. Caller-owned dense short logits can remain on the device;
only detached <=1M-element slices enter host storage. Native CPU QA and a
separate reviewed real-model worker are required before use.

The conservative96N+8MiB application-buffer allowance is below128MiB and covers
the explicit FP32 transfers, FP64 values/difference/threshold and temporary
reductions. It is not measured process RSS, Torch cache or CUDA headroom.
"""
from __future__ import annotations

import math
import torch

SCHEMA = "extension-native-bf16-projection-comparison-v1"
MAX_CHUNK_ELEMENTS = 1048576
MAX_HOST_SCRATCH_BYTES = 134217728
ABSOLUTE_FLOOR = 0.001
GLOBAL_RELATIVE_L2_MAXIMUM = 0.005


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _Sum:
    def __init__(self):
        self.value, self.correction = 0.0, 0.0

    def add(self, value):
        _require(math.isfinite(value) and value >= 0, "Nonfinite FP64 projection statistic")
        combined = self.value + value
        self.correction += ((self.value - combined) + value if abs(self.value) >= abs(value)
                            else (value - combined) + self.value)
        self.value = combined

    def total(self):
        value = self.value + self.correction
        _require(math.isfinite(value) and value >= 0, "Nonfinite accumulated projection statistic")
        return value


def _identity(value):
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            str(value.dtype), str(value.device), value.untyped_storage().data_ptr())


def _block(value, name, allowed_dtypes):
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided and
             value.dtype in allowed_dtypes and value.device.type in ("cpu", "cuda") and
             value.ndim == 2 and 1 <= value.shape[0] <= 128 and value.shape[1] > 0 and
             value.is_contiguous() and not torch.is_inference(value),
             name + " must be a normal contiguous <=128-row FP32/BF16 projection block")


class ProjectionComparison:
    """Aggregate exact-shape projection blocks without retaining tensor values.

    ``dense`` is FP32 or BF16; ``candidate`` must be an actual BF16 output block.
    Both are on the same CPU/CUDA device. CPU is an explicit fixture seam, not
    native AMP proof. Numerical misses return passed=False at finish; malformed,
    nonfinite, incomplete or reused comparisons raise ValueError. The caller
    must abort its attempt on any exception; the instance is then poisoned and
    refuses observation/finalization. There is no retry/reset interface.
    Exact source positions/order and head/checkpoint identity remain caller-owned;
    this helper verifies counts and numerical values, not content provenance.
    Version checks detect ordinary mutation, not every out-of-band .data write.
    """
    def __init__(self, expected_elements, *, max_chunk_elements=MAX_CHUNK_ELEMENTS,
                 label="student"):
        _require(type(expected_elements) is int and 0 < expected_elements <= (1 << 63) - 1,
                 "Positive exact expected element count required")
        _require(type(max_chunk_elements) is int and 1 <= max_chunk_elements <= MAX_CHUNK_ELEMENTS,
                 "Projection transfers must contain at most1048576 elements")
        _require(type(label) is str and label in ("student", "reference"),
                 "Explicit student or reference projection role required")
        self.expected_elements, self.max_chunk_elements, self.label = expected_elements, max_chunk_elements, label
        self.observed_elements, self.blocks, self.chunks, self.maximum_chunk = 0, 0, 0, 0
        self.elementwise_failures, self.max_absolute_error = 0, 0.0
        self.minimum_bf16_ulp, self.maximum_bf16_ulp = None, 0.0
        self._dense_square, self._candidate_square, self._difference_square = _Sum(), _Sum(), _Sum()
        self._devices, self._dense_dtypes, self._finished = set(), set(), False
        self._vocabulary, self._device, self._failed = None, None, False

    def _copy_chunk(self, tensor, start, count):
        _require(0 < count <= self.max_chunk_elements, "Projection copy exceeds declared bound")
        return tensor.detach().view(-1).narrow(0, start, count).to(
            device="cpu", dtype=torch.float32, copy=True)

    def observe(self, dense, candidate):
        _require(not self._failed, "A failed projection comparison cannot continue")
        try:
            return self._observe(dense, candidate)
        except BaseException:
            self._failed = True
            raise

    def _observe(self, dense, candidate):
        _require(not self._finished, "Projection comparison is already finalized")
        _block(dense, "Dense", (torch.float32, torch.bfloat16))
        _block(candidate, "Candidate", (torch.bfloat16,))
        _require(dense.shape == candidate.shape and dense.device == candidate.device,
                 "Matching projection shape and exact device required")
        _require(self._vocabulary is None or (dense.shape[1] == self._vocabulary and
                 str(dense.device) == self._device), "Projection vocabulary/device changed across blocks")
        count = dense.numel()
        _require(self.observed_elements + count <= self.expected_elements,
                 "Projection coverage exceeds the prospective exact count")
        dense_before, candidate_before = _identity(dense), _identity(candidate)
        for start in range(0, count, self.max_chunk_elements):
            size = min(self.max_chunk_elements, count - start)
            a32, b32 = self._copy_chunk(dense, start, size), self._copy_chunk(candidate, start, size)
            a, b = a32.to(torch.float64), b32.to(torch.float64)
            del a32, b32
            _require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                     "Every compared projection value must be finite")
            difference = a - b
            maximum = max(float(difference.abs().max()), self.max_absolute_error)
            # The floor .001 dominates the subnormal2^-133 branch. Clamping m
            # before the logarithm implements the exact declared piecewise rule,
            # including m=0, without log2(0) or unbounded extra arrays.
            threshold = torch.maximum(a.abs(), b.abs())
            threshold.clamp_min_(2.0 ** -126)
            torch.log2(threshold, out=threshold)
            threshold.floor_().sub_(7).exp2_()
            minimum_ulp, maximum_ulp = float(threshold.min()), float(threshold.max())
            self.minimum_bf16_ulp = minimum_ulp if self.minimum_bf16_ulp is None else min(self.minimum_bf16_ulp, minimum_ulp)
            self.maximum_bf16_ulp = max(self.maximum_bf16_ulp, maximum_ulp)
            threshold.clamp_min_(ABSOLUTE_FLOOR)
            failed = int((difference.abs() > threshold).sum().item())
            dense_square = float(a.square().sum(dtype=torch.float64).item())
            candidate_square = float(b.square().sum(dtype=torch.float64).item())
            difference_square = float(difference.square().sum(dtype=torch.float64).item())
            self._dense_square.add(dense_square)
            self._candidate_square.add(candidate_square)
            self._difference_square.add(difference_square)
            self.max_absolute_error = maximum
            self.elementwise_failures += failed
            self.chunks += 1
            self.maximum_chunk = max(self.maximum_chunk, size)
            del a, b, difference, threshold
        _require(_identity(dense) == dense_before and _identity(candidate) == candidate_before,
                 "Projection tensor membership/storage/version changed during comparison")
        self.observed_elements += count
        self.blocks += 1
        self._vocabulary, self._device = dense.shape[1], str(dense.device)
        self._devices.add(str(dense.device))
        self._dense_dtypes.add(str(dense.dtype))

    def finish(self):
        _require(not self._failed, "A failed projection comparison cannot finalize")
        try:
            return self._finish()
        except BaseException:
            self._failed = True
            raise

    def _finish(self):
        _require(not self._finished, "Projection comparison is already finalized")
        _require(self.observed_elements == self.expected_elements and self.blocks > 0,
                 "Complete exact projection coverage required before finalization")
        dense_l2 = math.sqrt(self._dense_square.total())
        candidate_l2 = math.sqrt(self._candidate_square.total())
        difference_l2 = math.sqrt(self._difference_square.total())
        relative = 0.0 if dense_l2 == difference_l2 == 0 else (None if dense_l2 == 0 else difference_l2 / dense_l2)
        passed = self.elementwise_failures == 0 and relative is not None and relative <= GLOBAL_RELATIVE_L2_MAXIMUM
        self._finished = True
        scratch = 96 * self.maximum_chunk + (8 << 20)
        _require(scratch <= MAX_HOST_SCRATCH_BYTES, "Declared host comparison scratch exceeded")
        return {"schema": SCHEMA, "label": self.label, "passed": passed,
                "expected_elements": self.expected_elements, "observed_elements": self.observed_elements,
                "blocks": self.blocks, "chunks": self.chunks, "vocabulary": self._vocabulary,
                "elementwise_failures": self.elementwise_failures,
                "max_absolute_error": self.max_absolute_error,
                "dense_l2": dense_l2, "candidate_l2": candidate_l2,
                "difference_l2": difference_l2, "global_relative_l2": relative,
                "elementwise_rule": "abs(a-b)<=max(.001,bf16_ulp(max(abs(a),abs(b))))",
                "absolute_floor": ABSOLUTE_FLOOR,
                "minimum_bf16_ulp": self.minimum_bf16_ulp,
                "maximum_bf16_ulp": self.maximum_bf16_ulp,
                "maximum_allowed_absolute_difference": max(ABSOLUTE_FLOOR, self.maximum_bf16_ulp),
                "global_relative_l2_maximum": GLOBAL_RELATIVE_L2_MAXIMUM,
                "observed_devices": sorted(self._devices), "dense_dtypes": sorted(self._dense_dtypes),
                "candidate_dtype": "torch.bfloat16", "maximum_chunk_elements": self.maximum_chunk,
                "declared_owned_host_scratch_bytes": scratch,
                "host_scratch_limit_bytes": MAX_HOST_SCRATCH_BYTES,
                "conservative_application_buffer_accounting": True,
                "actual_autocast_or_gpu_identity_established": False,
                "process_rss_or_cuda_headroom_measured": False,
                "tensor_values_retained_in_receipt": False}
