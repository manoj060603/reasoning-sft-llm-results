"""Unadmitted bounded FP32 tensor comparison and private initial-shadow helpers.

No model, data, optimizer, CLI, job or admission is provided. The caller owns
original434 provenance, stable in-memory tensors, phase/LR identity, current
compute allocation and native RSS/CUDA headroom. A generic explicit fixture
count never proves native434. Only bounded detached slices enter host FP32/FP64
storage; no full state.cpu/double/clone or list of retained tensors is created.

The96-bytes/element scratch reservation covers the explicit transfer, FP64,
difference, threshold, mask and reduction buffers, with8MiB of fixed
allowance, below128MiB for<=1048576 elements. It is conservative application
buffer accounting, NOT an OS RSS, Torch allocator/cache, BLAS workspace or CUDA
headroom measurement. Those remain the future native worker's separate gates.

Shadows contain actual stored FP32 bytes, not an Adam formula. The writer emits
an exclusive private directory; manifest.json is written LAST after all raw
files are fsynced and verified. Failed/partial directories are retained and may
never be reused. Reader pins the manifest FILE hash/path, verifies every file,
and yields only bounded slices internally. No shadow proves when it was taken;
the caller must bind post-LR0/pre-update2 identity and unchanged initial weights.
Do not mutate input tensors or shadow files during a call. Tensor version checks
detect ordinary in-place changes, not every possible out-of-band .data write.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys

import torch


MAX_CHUNK_ELEMENTS = 1048576
MAX_HOST_SCRATCH_BYTES = 134217728
STAGE_WRITE_CEILING_BYTES = 112 << 30
INITIAL_FREE_BYTES = 160 << 30
MINIMUM_RESERVE_BYTES = 32 << 30
MAX_MANIFEST_BYTES = 2 << 20
COMPARISON_SCHEMA = "extension-native-tensor-comparison-v1"
SHADOW_SCHEMA = "extension-native-fp32-initial-shadow-v1"
TOLERANCE_KEYS = frozenset({"elementwise_atol", "elementwise_rtol", "rms_atol", "rms_rtol",
    "group_relative_l2_maximum", "group_cosine_minimum", "require_positive_group_norms"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def _number(value, *, low=0., high=None):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= low and (high is None or value <= high)
    except OverflowError:
        return False


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def chunk_memory_bound(chunk_elements):
    """Conservative explicit host-buffer bytes, not observed process memory."""
    _require(_integer(chunk_elements, 1, MAX_CHUNK_ELEMENTS), "Chunk must be a plain integer in[1,1048576]")
    value = 96 * chunk_elements + (8 << 20)
    _require(value <= MAX_HOST_SCRATCH_BYTES, "Explicit host comparison scratch exceeds128MiB")
    return value


def validate_tolerances(value):
    _require(type(value) is dict and set(value) == TOLERANCE_KEYS, "Supply every explicit tolerance key, without extras")
    _require(all(_number(value[name]) for name in ("elementwise_atol", "elementwise_rtol")), "Elementwise tolerances must be finite nonnegative numbers")
    _require((value["rms_atol"] is None and value["rms_rtol"] is None) or
        (_number(value["rms_atol"]) and _number(value["rms_rtol"])), "RMS tolerances must both be absent or finite nonnegative")
    _require(value["group_relative_l2_maximum"] is None or _number(value["group_relative_l2_maximum"]), "Invalid group relative-L2 limit")
    _require(value["group_cosine_minimum"] is None or _number(value["group_cosine_minimum"], low=-1., high=1.), "Invalid group cosine minimum")
    _require(type(value["require_positive_group_norms"]) is bool, "Positive-group-norm policy must be explicit bool")
    return dict(value)


def _tensor(value, name):
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided and
        value.dtype == torch.float32 and value.device.type in ("cpu", "cuda") and
        value.is_contiguous() and value.numel() > 0 and not torch.is_inference(value),
        name + " must be a nonempty contiguous normal FP32 CPU/CUDA tensor")


def _names(values, expected_count):
    _require(_integer(expected_count, 1, 434) and isinstance(values, Mapping) and len(values) == expected_count,
        "Exact explicit tensor count required; only the default434 is a native-size expectation")
    _require(all(type(name) is str and name and len(name.encode("utf-8")) <= 1024 and "\x00" not in name for name in values), "Invalid original tensor FQN")
    names = sorted(values)
    for name in names:
        _tensor(values[name], name)
    return names


def _groups(groups, names):
    _require(type(groups) is dict and bool(groups) and "all" not in groups and
        all(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) for name in groups), "Supply named groups without reserved'all'")
    result, used = {}, set()
    for name, members in groups.items():
        _require(type(members) in (tuple, list) and members and all(type(item) is str for item in members) and
            len(members) == len(set(members)) and not set(members) & used, "Groups must be nonempty disjoint FQN sequences")
        result[name] = tuple(members)
        used.update(members)
    _require(used == set(names), "Groups must partition every compared tensor exactly once")
    return result


def _identities(values, names):
    # Metadata only, never retained Tensor/state copies. Ordinary in-place or
    # storage/object/shape changes anywhere during the entire call are refused.
    return {name: (id(values[name]), values[name]._version, tuple(values[name].shape),
        tuple(values[name].stride()), str(values[name].device), str(values[name].dtype),
        values[name].untyped_storage().data_ptr()) for name in names}


def _unchanged_tensors(values, identities):
    _require(set(values) == set(identities) and _identities(values, identities) == identities,
        "Tensor membership/object/storage/shape/version changed during streaming")


class _Tracker:
    def __init__(self, chunk_elements):
        chunk_memory_bound(chunk_elements)
        self.limit, self.maximum, self.copies, self.chunks = chunk_elements, 0, 0, 0

    def observe(self, count):
        _require(_integer(count, 1, self.limit), "An operation exceeded the bounded transfer chunk")
        self.maximum = max(self.maximum, count)
        self.copies += 1

    def result(self):
        return {"chunk_elements": self.limit, "maximum_chunk_elements_observed": self.maximum,
            "chunks_compared": self.chunks, "bounded_fp32_copies": self.copies,
            "peak_owned_host_bytes": 0 if not self.maximum else chunk_memory_bound(self.maximum),
            "host_scratch_limit_bytes": MAX_HOST_SCRATCH_BYTES,
            "conservative_explicit_buffer_accounting": True, "process_rss_or_allocator_peak_measured": False}


def _copy_chunk(tensor, start, count, tracker):
    """Instrumentation seam: never copy/cast an entire tensor implicitly."""
    tracker.observe(count)
    _require(_integer(start, 0) and start + count <= tensor.numel(), "Invalid bounded tensor slice")
    return tensor.detach().view(-1).narrow(0, start, count).to(device="cpu", dtype=torch.float32, copy=True)


def _raw_view(value):
    _require(value.device.type == "cpu" and value.dtype == torch.float32 and value.is_contiguous(), "Expected a bounded host FP32 chunk")
    return memoryview(value.numpy()).cast("B")


class _Sum:
    def __init__(self):
        self.value, self.correction = 0., 0.

    def add(self, value):
        _require(math.isfinite(value), "Nonfinite FP64 streamed statistic")
        combined = self.value + value
        self.correction += ((self.value - combined) + value) if abs(self.value) >= abs(value) else ((value - combined) + self.value)
        self.value = combined

    def total(self):
        result = self.value + self.correction
        _require(math.isfinite(result), "Nonfinite aggregate FP64 statistic")
        return result


class _Statistics:
    def __init__(self):
        self.elements, self.failures = 0, 0
        self.dense_square, self.candidate_square, self.difference_square, self.dot = (_Sum() for _ in range(4))
        self.maximum, self.dense_maximum, self.candidate_maximum = 0., 0., 0.

    def merge(self, row):
        self.elements += row.elements
        self.failures += row.failures
        for name in ("dense_square", "candidate_square", "difference_square", "dot"):
            getattr(self, name).add(getattr(row, name).total())
        self.maximum = max(self.maximum, row.maximum)
        self.dense_maximum = max(self.dense_maximum, row.dense_maximum)
        self.candidate_maximum = max(self.candidate_maximum, row.candidate_maximum)

    def result(self, tolerances, *, group=False):
        _require(self.elements > 0, "No elements were compared")
        dense = math.sqrt(max(0., self.dense_square.total()))
        candidate = math.sqrt(max(0., self.candidate_square.total()))
        difference = math.sqrt(max(0., self.difference_square.total()))
        scale = math.sqrt(self.elements)
        relative = None if dense == 0 else difference / dense
        cosine = None if dense == 0 or candidate == 0 else max(-1., min(1., self.dot.total() / dense / candidate))
        rms, dense_rms = difference / scale, dense / scale
        passed = self.failures == 0
        if tolerances["rms_atol"] is not None:
            passed = passed and rms <= tolerances["rms_atol"] + tolerances["rms_rtol"] * dense_rms
        if group:
            if tolerances["require_positive_group_norms"]:
                passed = passed and dense > 0 and candidate > 0
            if relative is not None and tolerances["group_relative_l2_maximum"] is not None:
                passed = passed and relative <= tolerances["group_relative_l2_maximum"]
            if cosine is not None and tolerances["group_cosine_minimum"] is not None:
                passed = passed and cosine >= tolerances["group_cosine_minimum"]
        return {"elements": self.elements, "dense_l2": dense, "candidate_l2": candidate,
            "difference_l2": difference, "max_absolute_difference": self.maximum,
            "dense_max_absolute": self.dense_maximum, "candidate_max_absolute": self.candidate_maximum,
            "rms_difference": rms, "dense_rms": dense_rms, "candidate_rms": candidate / scale,
            "relative_l2": relative, "cosine": cosine, "elementwise_failures": self.failures,
            "positive_norms": dense > 0 and candidate > 0, "passed": bool(passed)}


def _accumulate(result, candidate32, dense32, tolerances, initial32=None):
    # Explicit peak: three FP32 buffers12N; three FP64 inputs24N; difference,
    # threshold and work24N; boolean/reduction allowance<=4N, total<=64N.
    # The declared96N+8MiB reservation adds transfer/metadata safety allowance.
    with torch.no_grad():
        _require(bool(torch.isfinite(candidate32).all()) and bool(torch.isfinite(dense32).all()), "Nonfinite input tensor chunk")
        candidate, dense = candidate32.to(dtype=torch.float64), dense32.to(dtype=torch.float64)
        if initial32 is not None:
            _require(bool(torch.isfinite(initial32).all()), "Nonfinite initial shadow chunk")
            initial = initial32.to(dtype=torch.float64)
            candidate.sub_(initial)
            dense.sub_(initial)
            del initial
        difference, work, threshold = torch.empty_like(dense), torch.empty_like(dense), torch.empty_like(dense)
        mask = torch.empty(dense.shape, dtype=torch.bool, device="cpu")
        torch.sub(candidate, dense, out=difference)
        result.elements += dense.numel()
        result.dense_square.add(float(torch.dot(dense, dense).item()))
        result.candidate_square.add(float(torch.dot(candidate, candidate).item()))
        result.difference_square.add(float(torch.dot(difference, difference).item()))
        result.dot.add(float(torch.dot(candidate, dense).item()))
        work.copy_(difference).abs_()
        result.maximum = max(result.maximum, float(work.max().item()))
        threshold.copy_(dense).abs_().mul_(tolerances["elementwise_rtol"]).add_(tolerances["elementwise_atol"])
        torch.gt(work, threshold, out=mask)
        result.failures += int(torch.count_nonzero(mask).item())
        work.copy_(dense).abs_()
        result.dense_maximum = max(result.dense_maximum, float(work.max().item()))
        work.copy_(candidate).abs_()
        result.candidate_maximum = max(result.candidate_maximum, float(work.max().item()))


def _compare(candidate, dense, *, groups, tolerances, expected_count, chunk_elements, shadow=None):
    _require(sys.byteorder == "little", "Canonical raw FP32 evidence requires a little-endian host")
    tolerance = validate_tolerances(tolerances)
    names = _names(dense, expected_count)
    _require(_names(candidate, expected_count) == names, "Candidate and dense FQNs differ")
    partitions = _groups(groups, names)
    candidate_identities, dense_identities = _identities(candidate, names), _identities(dense, names)
    tracker, rows = _Tracker(chunk_elements), {}
    totals = {name: _Statistics() for name in ("all", *partitions)}
    memberships = {member: name for name, members in partitions.items() for member in members}
    shadow_manifest = None
    if shadow is not None:
        _require(isinstance(shadow, ShadowReader), "Initial shadow must be the pinned bounded reader")
        shadow_manifest = shadow.manifest
        _require(shadow_manifest["tensor_count"] == expected_count and
            set(shadow_manifest["tensors"]) == set(names), "Initial shadow must bind every exact tensor name")
        _require(_integer(shadow.chunk_elements, 1, MAX_CHUNK_ELEMENTS) and chunk_elements <= shadow.chunk_elements,
            "Comparison chunk cannot exceed the reader's separately bound slice limit")
        shadow.verify_unchanged()
    for name in names:
        left, right = candidate[name], dense[name]
        _require(left.shape == right.shape, "Tensor shapes differ: " + name)
        if shadow is not None:
            _require(list(left.shape) == shadow_manifest["tensors"][name]["shape"], "Shadow tensor shape differs: " + name)
        versions = (left._version, right._version)
        statistics, candidate_hash, dense_hash = _Statistics(), hashlib.sha256(), hashlib.sha256()
        for start in range(0, right.numel(), chunk_elements):
            count = min(chunk_elements, right.numel() - start)
            a, b = _copy_chunk(left, start, count, tracker), _copy_chunk(right, start, count, tracker)
            candidate_hash.update(_raw_view(a)); dense_hash.update(_raw_view(b))
            initial = None if shadow is None else shadow.read_slice(name, start, count)
            _accumulate(statistics, a, b, tolerance, initial)
            del a, b, initial
            tracker.chunks += 1
        _require((left._version, right._version) == versions, "Input tensor changed during comparison: " + name)
        row = statistics.result(tolerance)
        row.update(shape=list(right.shape), candidate_sha256=candidate_hash.hexdigest(), dense_sha256=dense_hash.hexdigest())
        rows[name] = row
        totals["all"].merge(statistics)
        totals[memberships[name]].merge(statistics)
    if shadow is not None:
        shadow.verify_unchanged()
    _unchanged_tensors(candidate, candidate_identities)
    _unchanged_tensors(dense, dense_identities)
    aggregated = {name: value.result(tolerance, group=True) for name, value in totals.items()}
    result = {"schema": COMPARISON_SCHEMA, "complete": True,
        "passed": all(value["passed"] for value in rows.values()) and all(value["passed"] for value in aggregated.values()),
        "expected_tensor_count": expected_count, "tensor_count": len(rows), "tolerances": tolerance,
        "tensors": rows, "groups": aggregated, "streaming": tracker.result(),
        "comparison": "actual_stored_fp32_deltas" if shadow is not None else "stored_fp32_tensor_values",
        "initial_shadow_manifest_sha256": None if shadow is None else shadow.manifest_sha256,
        "native_provenance_established": False, "optimizer_formula_used": False}
    json.dumps(result, allow_nan=False)
    return result


def compare_named_tensors(candidate, dense, *, groups, tolerances, expected_count=434, chunk_elements=MAX_CHUNK_ELEMENTS):
    return _compare(candidate, dense, groups=groups, tolerances=tolerances,
        expected_count=expected_count, chunk_elements=chunk_elements)


def compare_stored_deltas(candidate, dense, initial_shadow, *, groups, tolerances,
                          expected_count=434, chunk_elements=MAX_CHUNK_ELEMENTS):
    """Compare actual after-initial FP32 stored differences, never Adam math."""
    return _compare(candidate, dense, groups=groups, tolerances=tolerances,
        expected_count=expected_count, chunk_elements=chunk_elements, shadow=initial_shadow)


def _no_alias(path):
    _require(path.is_absolute() and ".." not in path.parts, "Explicit absolute private path required")
    for part in (path, *path.parents):
        _require(not part.is_symlink(), "Private path cannot contain a symlink component")
    return path


def _private(path, *, directory=False):
    _no_alias(path)
    details = path.stat()
    _require((stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)) and
        details.st_uid == os.getuid() and stat.S_IMODE(details.st_mode) & 0o077 == 0,
        "Expected owner-only private directory/file")
    return details


class ScratchBudget:
    """Finite shared stage-byte budget; no student/reference offload mechanism.

    Instantiate once for the stage's existing private parent. Callers must also
    account their separately measured checkpoint/output writes with charge_written;
    this helper cannot discover writes made behind its back or reserve disk blocks.
    """
    def __init__(self, directory):
        self.directory = _no_alias(Path(directory).absolute())
        _private(self.directory, directory=True)
        self.initial_free_bytes = shutil.disk_usage(self.directory).free
        _require(_integer(self.initial_free_bytes, INITIAL_FREE_BYTES), "Require160GiB initial free scratch")
        self.bytes_written = 0

    def check_planned_write(self, count):
        _require(_integer(count, 0) and self.bytes_written + count <= STAGE_WRITE_CEILING_BYTES, "Stage output exceeds112GiB ceiling")
        free = shutil.disk_usage(self.directory).free
        _require(_integer(free, 0) and free >= count + MINIMUM_RESERVE_BYTES, "Keep32GiB reserve plus planned bytes")

    def charge_written(self, count):
        _require(_integer(count, 0) and self.bytes_written + count <= STAGE_WRITE_CEILING_BYTES, "Stage output exceeds112GiB ceiling")
        self.bytes_written += count

    def evidence(self):
        return {"initial_free_bytes": self.initial_free_bytes, "accounted_bytes_written": self.bytes_written,
            "stage_write_ceiling_bytes": STAGE_WRITE_CEILING_BYTES, "minimum_reserve_bytes": MINIMUM_RESERVE_BYTES,
            "physical_disk_blocks_reserved": False, "unaccounted_external_writes_verified": False}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_all(descriptor, payload, budget):
    view = memoryview(payload)
    while view:
        budget.check_planned_write(len(view))
        count = os.write(descriptor, view)
        _require(count > 0, "Short private write made no progress")
        budget.charge_written(count)
        view = view[count:]


def _fingerprint(details):
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def _hash_private(path, expected_bytes=None):
    before = _private(path)
    _require(expected_bytes is None or before.st_size == expected_bytes, "Private raw file size differs")
    checksum, count = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        _require(_fingerprint(os.fstat(stream.fileno())) == _fingerprint(before), "Private file changed before open")
        for payload in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(payload); count += len(payload)
    after = _private(path)
    _require(_fingerprint(before) == _fingerprint(after) and count == before.st_size, "Private file changed while hashing")
    return checksum.hexdigest(), _fingerprint(after)


def write_initial_shadow(tensors, directory, *, budget, expected_count=434, chunk_elements=MAX_CHUNK_ELEMENTS):
    """Exclusive owner-only raw FP32 writer; no successful partial manifest."""
    _require(sys.byteorder == "little", "Raw FP32 shadows require little endian")
    names, tracker = _names(tensors, expected_count), _Tracker(chunk_elements)
    path = _no_alias(Path(directory).absolute())
    _require(isinstance(budget, ScratchBudget) and path.parent == budget.directory, "Shadow must be a direct child of the stage scratch budget")
    _private(path.parent, directory=True)
    _require(not path.exists(), "Never reuse or overwrite a shadow, including a failed partial directory")
    total = sum(tensors[name].numel() * 4 for name in names)
    budget.check_planned_write(total + MAX_MANIFEST_BYTES)
    path.mkdir(mode=0o700, exist_ok=False)
    records, identities = {}, _identities(tensors, names)
    for index, name in enumerate(names):
        tensor = tensors[name]
        version, checksum = tensor._version, hashlib.sha256()
        filename = "tensor-" + str(index).zfill(4) + ".f32"
        descriptor = os.open(path / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            for start in range(0, tensor.numel(), chunk_elements):
                count = min(chunk_elements, tensor.numel() - start)
                value = _copy_chunk(tensor, start, count, tracker)
                _require(bool(torch.isfinite(value).all()), "Cannot seal nonfinite initial weights")
                payload = _raw_view(value)
                checksum.update(payload)
                _write_all(descriptor, payload, budget)
                del payload, value
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _require(tensor._version == version, "Initial tensor changed while writing: " + name)
        expected = tensor.numel() * 4
        actual, _ = _hash_private(path / filename, expected)
        _require(actual == checksum.hexdigest(), "Raw shadow bytes changed after write")
        records[name] = {"filename": filename, "shape": list(tensor.shape), "dtype": "float32-le",
            "bytes": expected, "sha256": actual}
    _unchanged_tensors(tensors, identities)
    manifest = {"schema": SHADOW_SCHEMA, "complete": True, "directory": str(path), "dtype": "float32-le",
        "expected_tensor_count": expected_count, "tensor_count": len(names), "total_bytes": total,
        "chunk_elements": chunk_elements, "tensors": records, "streaming": tracker.result(),
        "training_phase_or_initialization_provenance_verified": False, "native_provenance_established": False}
    payload = _json_bytes(manifest)
    _require(len(payload) <= MAX_MANIFEST_BYTES and {item.name for item in path.iterdir()} == {item["filename"] for item in records.values()},
        "Unexpected shadow file or oversized manifest before completion")
    descriptor = os.open(path / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        _write_all(descriptor, payload, budget)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    actual, _ = _hash_private(path / "manifest.json", len(payload))
    _require(actual == hashlib.sha256(payload).hexdigest(), "Completed shadow manifest bytes changed")
    return {"manifest": manifest, "manifest_sha256": actual, "manifest_bytes": len(payload), "directory": str(path)}


class ShadowReader:
    def __init__(self, directory, *, expected_manifest_sha256, expected_count=434, chunk_elements=MAX_CHUNK_ELEMENTS):
        _require(sys.byteorder == "little" and _sha(expected_manifest_sha256), "Pinned little-endian shadow manifest required")
        self.directory = _no_alias(Path(directory).absolute())
        _private(self.directory, directory=True)
        self.chunk_elements, self.manifest_sha256 = chunk_elements, expected_manifest_sha256
        chunk_memory_bound(chunk_elements)
        path = self.directory / "manifest.json"
        details = _private(path)
        _require(0 < details.st_size <= MAX_MANIFEST_BYTES, "Expected bounded completed shadow manifest")
        actual, fingerprint = _hash_private(path)
        _require(actual == expected_manifest_sha256, "Shadow manifest FILE hash differs")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read(MAX_MANIFEST_BYTES + 1)
        def pairs(items):
            result = {}
            for name, value in items:
                _require(name not in result, "Duplicate shadow manifest key")
                result[name] = value
            return result
        self._manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite shadow manifest")))
        _require(hashlib.sha256(payload).hexdigest() == actual and _fingerprint(_private(path)) == fingerprint,
            "Shadow manifest changed during parsing")
        manifest = self._manifest
        _require(type(manifest) is dict and set(manifest) == {"schema", "complete", "directory", "dtype",
            "expected_tensor_count", "tensor_count", "total_bytes", "chunk_elements", "tensors", "streaming",
            "training_phase_or_initialization_provenance_verified", "native_provenance_established"} and
            manifest.get("schema") == SHADOW_SCHEMA and manifest.get("complete") is True and
            manifest.get("directory") == str(self.directory) and manifest.get("dtype") == "float32-le" and
            _integer(expected_count, 1, 434) and _integer(manifest.get("tensor_count"), 1, 434) and
            _integer(manifest.get("expected_tensor_count"), 1, 434) and
            manifest["tensor_count"] == manifest["expected_tensor_count"] == expected_count and
            _integer(manifest.get("total_bytes"), 4, STAGE_WRITE_CEILING_BYTES) and
            _integer(manifest.get("chunk_elements"), 1, MAX_CHUNK_ELEMENTS) and
            type(manifest.get("tensors")) is dict and len(manifest["tensors"]) == expected_count and
            manifest.get("native_provenance_established") is False and
            manifest.get("training_phase_or_initialization_provenance_verified") is False,
            "Completed shadow path/count/type/provenance differs")
        self.fingerprints = {"manifest.json": fingerprint}
        total, filenames = 0, set()
        for index, (name, row) in enumerate(sorted(manifest["tensors"].items())):
            _require(type(name) is str and name and len(name.encode("utf-8")) <= 1024 and "\x00" not in name and
                type(row) is dict and set(row) == {"filename", "shape", "dtype", "bytes", "sha256"} and
                row["filename"] == "tensor-" + str(index).zfill(4) + ".f32" and row["dtype"] == "float32-le" and
                type(row["shape"]) is list and all(_integer(size, 1) for size in row["shape"]) and
                _integer(row["bytes"], 4) and row["bytes"] == math.prod(row["shape"]) * 4 and _sha(row["sha256"]),
                "Malformed bounded raw FP32 shadow record")
            checksum, fingerprint = _hash_private(self.directory / row["filename"], row["bytes"])
            _require(checksum == row["sha256"], "Raw shadow file hash differs")
            self.fingerprints[row["filename"]] = fingerprint
            total += row["bytes"]; filenames.add(row["filename"])
        _require(manifest.get("total_bytes") == total and total <= STAGE_WRITE_CEILING_BYTES and
            {item.name for item in self.directory.iterdir()} == filenames | {"manifest.json"}, "Unexpected/partial/oversized shadow inventory")

    @property
    def manifest(self):
        """Callable API metadata copy; callers cannot mutate the pinned reader."""
        return deepcopy(self._manifest)

    def read_slice(self, name, start, count):
        _private(self.directory, directory=True)
        _require(_integer(self.chunk_elements, 1, MAX_CHUNK_ELEMENTS) and
            name in self._manifest["tensors"] and _integer(start, 0) and
            _integer(count, 1, self.chunk_elements), "Invalid bounded shadow slice")
        row = self._manifest["tensors"][name]
        _require(start + count <= row["bytes"] // 4, "Shadow slice outside stored tensor")
        path = self.directory / row["filename"]
        before = _private(path)
        _require(_fingerprint(before) == self.fingerprints[row["filename"]], "Shadow file changed before slice")
        buffer = bytearray(count * 4)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            _require(_fingerprint(os.fstat(stream.fileno())) == _fingerprint(before), "Shadow slice open changed")
            stream.seek(start * 4)
            view, copied = memoryview(buffer), 0
            while copied < len(view):
                amount = stream.readinto(view[copied:])
                _require(amount is not None and amount > 0, "Truncated shadow slice")
                copied += amount
        _require(_fingerprint(_private(path)) == _fingerprint(before), "Shadow file changed during slice")
        # The Tensor owns a reference to only this bounded writable bytearray.
        result = torch.frombuffer(buffer, dtype=torch.float32, count=count)
        _require(bool(torch.isfinite(result).all()), "Nonfinite stored FP32 shadow slice")
        return result

    def verify_unchanged(self):
        expected = {row["filename"] for row in self._manifest["tensors"].values()} | {"manifest.json"}
        _private(self.directory, directory=True)
        _require({item.name for item in self.directory.iterdir()} == expected, "Shadow inventory changed")
        checksum, fingerprint = _hash_private(self.directory / "manifest.json")
        _require(checksum == self.manifest_sha256 and fingerprint == self.fingerprints["manifest.json"], "Shadow manifest changed")
        for row in self._manifest["tensors"].values():
            checksum, fingerprint = _hash_private(self.directory / row["filename"], row["bytes"])
            _require(checksum == row["sha256"] and fingerprint == self.fingerprints[row["filename"]], "Shadow raw file changed")
        return True


def open_initial_shadow(directory, *, expected_manifest_sha256, expected_count=434, chunk_elements=MAX_CHUNK_ELEMENTS):
    return ShadowReader(directory, expected_manifest_sha256=expected_manifest_sha256,
        expected_count=expected_count, chunk_elements=chunk_elements)
