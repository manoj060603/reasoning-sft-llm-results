"""Unadmitted, versioned identity adapter for an already guarded CUDA job.

PyTorch2.9's public get_device_properties(0).uuid is a _CUuuid object;
str(uuid) is an unprefixed dashed UUID, unlike nvidia-smi's GPU-prefixed
identifier. Normalize that representation without substituting a host ordinal.
Callers still own allocation/source/runtime admission before importing torch.
This module imports no torch, initializes no CUDA, and performs no work at import.
"""
from __future__ import annotations

import csv
import re
import subprocess
import uuid as uuid_module


SCHEMA = "extension-assigned-gpu-identity-v2"
_ABSENT = object()
UUID_PATTERN = r"(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
SOURCE_EVIDENCE = {
    "torch_binding": "https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/cuda/Module.cpp#L962-L1008",
    "torch_string_format": "https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/utils.cpp#L226-L251",
    "torch_public_properties": "https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/__init__.py#L560-L574",
    "nvidia_target_uuid": "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_uuid(value):
    """Accept only a complete bare/GPU-prefixed dashed UUID representation.

    The object's __str__ is intentional: pinned Torch returns _CUuuid even
    though its typing stub says str. No UUID prefix abbreviation, visibility
    environment, PCI address, MIG identity or numeric ordinal is a fallback.
    """
    try:
        text = str(value)
    except BaseException as error:
        raise ValueError("Assigned CUDA UUID has no valid string representation") from error
    match = re.fullmatch(UUID_PATTERN, text)
    _require(match is not None, "Assigned CUDA UUID must be a complete bare or GPU-prefixed dashed UUID")
    body = match.group(1).lower()
    _require(body != "00000000-0000-0000-0000-000000000000", "Assigned CUDA UUID cannot be nil")
    return "GPU-" + body


def assigned_gpu_identity(torch, *, runner=None):
    """Bind CUDA logical0 to one exact nvidia-smi UUID/driver observation.

    Execute only after the caller's current single-B200 allocation and source
    guards. The injected runner is solely a subprocess.run-compatible CPU-test
    seam. No broader GPU enumeration or ordinal query is attempted on failure.
    """
    cuda = torch.cuda
    _require(cuda.is_available() is True, "An available CUDA device is required")
    count = cuda.device_count()
    _require(type(count) is int and count == 1, "Exactly one visible CUDA device is required")
    current = cuda.current_device()
    _require(type(current) is int and current == 0, "Assigned CUDA device must map to logical0")
    _require(cuda.is_bf16_supported() is True, "Assigned CUDA device must support BF16")
    properties = cuda.get_device_properties(0)
    name = cuda.get_device_name(0)
    _require(type(name) is str and "B200" in name and getattr(properties, "name", None) == name,
        "Assigned CUDA logical0 must be the matching B200 device")
    raw_uuid = getattr(properties, "uuid", None)
    try:
        raw_text = str(raw_uuid)
    except BaseException as error:
        raise ValueError("Assigned CUDA UUID has no valid string representation") from error
    canonical = normalize_uuid(raw_text)
    raw_bytes = getattr(raw_uuid, "bytes", _ABSENT)
    if raw_bytes is not _ABSENT:
        _require(type(raw_bytes) in (list, tuple, bytes) and len(raw_bytes) == 16 and
            all(type(item) is int and 0 <= item <= 255 for item in raw_bytes),
            "Assigned Torch UUID bytes must be exactly16 plain byte values")
        raw_bytes = list(raw_bytes)
        _require(normalize_uuid(str(uuid_module.UUID(bytes=bytes(raw_bytes)))) == canonical,
            "Assigned Torch UUID byte and string identities disagree")
    else:
        raw_bytes = None
    command = ["nvidia-smi", "-i", canonical, "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"]
    result = (subprocess.run if runner is None else runner)(command, check=True, capture_output=True, text=True, timeout=15)
    _require(type(getattr(result, "returncode", None)) is int and result.returncode == 0 and
        type(getattr(result, "stdout", None)) is str and type(getattr(result, "stderr", None)) is str,
        "Assigned-device query failed or returned an unsupported response")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(len(lines) == 1, "Assigned-device query must return exactly one GPU row")
    try:
        fields = next(csv.reader(lines, strict=True))
    except (csv.Error, StopIteration) as error:
        raise ValueError("Assigned-device query returned malformed CSV") from error
    _require(len(fields) == 2, "Assigned-device query must return only UUID and driver")
    queried_uuid, driver = (field.strip() for field in fields)
    _require(queried_uuid.startswith("GPU-") and normalize_uuid(queried_uuid) == canonical and
        re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver) is not None,
        "Assigned-device UUID or driver proof does not match CUDA logical0")
    return {"schema": SCHEMA, "uuid": canonical, "driver_version": driver, "name": name,
        "properties": str(properties), "query_scope": "CUDA logical0 assigned device UUID only; no ordinal fallback",
        "logical_cuda_device": 0, "visible_cuda_device_count": 1, "bf16_supported": True,
        "torch_uuid_raw": raw_text,
        "torch_uuid_type": type(raw_uuid).__module__ + "." + type(raw_uuid).__qualname__,
        "normalization": "GPU-prefixed full dashed UUID" if raw_text.startswith("GPU-") else "bare full dashed UUID",
        "torch_uuid_bytes": raw_bytes, "source_evidence": dict(SOURCE_EVIDENCE),
        "query": {"command": command, "stdout": result.stdout, "stderr": result.stderr, "timeout_seconds": 15}}
