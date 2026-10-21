"""Prospective CPU64:32 topic fixtures, then32 native-pair-hook fixtures.

Pre-test control imports only stdlib. Topics run FIRST in a clean stdlib-only
parent; actual tiny64-row fitting occurs only in up to five fresh sequential
children, each bounded to60seconds. Pair fixtures are discovered only after
topics32 passes, and may then import Torch/NumPy/tiny HF classes. Read-only
Torch/backend observations bound that later pair-fixture phase only. No real
population/cache/model/tokenizer, native434/BF16 CUDA, score or draw is admitted.
HistoricalCPU80 and standaloneCPU32 are verified, never replayed.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys
import sysconfig
import time
import unittest


ROLE = "native_pair_topics_qa_20261021_v1"
NAME = "sft-extension-native-pair-topics-qa"
PLAN = "extension/native_pair_topics_qa_plan.json"
SCRIPT = "extension/native_pair_topics_qa.slurm"
BOOTSTRAP = "extension/operations/submit-native-pair-topics-qa-20261021.py"
SCHEMA = "extension-native-pair-topics-native-qa-v1"
LEDGER_SCHEMA = "reasoning-sft-extension-job-ledger-v1"
PATTERNS = {"test_full_pool_topics.py": 32, "test_checkpointed_head_native_pair_hooks.py": 32}
TEST_ORDER = ("test_full_pool_topics.py", "test_checkpointed_head_native_pair_hooks.py")
EXPECTED_TESTS = 64
WALL_SECONDS, RESERVE_SECONDS = 1800, 172800
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
PROTOCOL_PINS = {
    "extension/planning/checkpointed_head_native_numerical_protocol.json": "27cedace388a17f3b97a0b1466e119c59a4d63913085263c689c8cfbbb003cb3",
    "extension/planning/full_pool_selector_prospective_protocol.json": "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29",
    "extension/planning/full_pool_selector_root_approval.json": "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6"}
PINS = {
    "extension/checkpointed_head_native_pair_hooks.py": "6295f4144f82bc009195aa04cd0c2a384ffa0a83909d01c7aeb59c0390edf862",
    "tests/test_checkpointed_head_native_pair_hooks.py": "f65ee07fd0f1f9bcb29d87fcb9d5f096ce009728e6b10179235839e1df64976a",
    "extension/full_pool_topics.py": "55d5ae814ef9df94f972400c6ddccda7abecf42950a767d457aedc60d3f9db1c",
    "tests/test_full_pool_topics.py": "5d7ed0e1ee162411538c4a8ba0f48d5654a4cfb98ae1371800ef2cf2bb68fbd1"}
SOURCES = frozenset(PINS) | {PLAN, SCRIPT, "extension/native_pair_topics_qa.py"}
VERSIONS = {"python": "3.10.19", "implementation": "cpython", "torch": "2.9.0", "numpy": "1.26.4",
    "transformers": "4.57.3", "accelerate": "1.12.0", "safetensors": "0.7.0",
    "scipy": "1.15.3", "scikit-learn": "1.7.2", "threadpoolctl": "3.6.0", "joblib": "1.5.2"}
PRIOR = {"path": "extension/report/standalone-training-step-qa-20261021-1230332/job_ledger_snapshot.json",
    "sha256": "6c82cb0c6ad1241d1f89add0ed270d7cf0fb808c0864bf050b8d8692914670eb", "entries": 35,
    "entries_sha256": "e4355d016fecb811bdacca205f921bf57ca8683e95734cb6fc63c60163f5108c"}
PARENT = {"path": "extension/report/standalone-training-step-qa-20261021-1230332/evidence_manifest.json",
    "sha256": "c76f477ff2c2298d6d3934e89038e8f5f8c97ef198d931f71d8bfbcefa4cfd78",
    "schema": "extension-standalone-training-step-qa-evidence-v1", "inventory_key": "protected_source_inventory", "expected_files": 204}
HISTORY = {"path": "deadline/job_ledger.json", "sha256": "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d"}
HISTORICAL = {"head_native_tools80": {
    "directory": "extension/report/head-native-tools-qa-20261021-1230042", "job_id": "1230042",
    "name": "sft-extension-head-native-tools-qa", "tests": 80, "row_elapsed": [50, 50, 50], "source_files": 11,
    "test_counts": {"test_native_bf16_projection_comparison.py": 24,
        "test_checkpointed_head_native_observation.py": 24, "test_paired_training_step_controller.py": 32},
    "plan_path": "extension/head_native_tools_qa_plan.json", "qa_schema": "extension-head-native-tools-native-qa-v1",
    "complete_schema": "extension-head-native-tools-qa-complete-v1",
    "native_qa_sha256": "8db43ab6633e3986f442869e784a46c763a51359d4393f7d2f129ce3704e602e",
    "complete_sha256": "bd5f2187db80e1e561283e507f218b8bbcd858bd967503cbcf4c10055beb01e0",
    "accounting_sha256": "ea6b2c341453532d845654606973980b19d37f3fb70c9ea58121817fcbaad261"},
    "standalone32": {"directory": "extension/report/standalone-training-step-qa-20261021-1230332",
        "job_id": "1230332", "name": "sft-extension-standalone-training-step-qa", "tests": 32,
        "row_elapsed": [45, 45, 45], "source_files": 5,
        "test_counts": {"test_standalone_training_step_controller.py": 32},
        "plan_path": "extension/standalone_training_step_qa_plan.json",
        "qa_schema": "extension-standalone-training-step-native-qa-v1",
        "complete_schema": "extension-standalone-training-step-qa-complete-v1",
        "native_qa_sha256": "33ea7e0f51d36b494084f60e5ac60c62ccb0c0fc308f1f89c7a0c0de8fb5dfd9",
        "complete_sha256": "8981e7ebc7a21df64961d38ae0a577b69525390f74111bbee188a2d8ec3e4946",
        "accounting_sha256": "c614357250626101b7e4e182c448ac88c1e9e1e5403feac9ad17a363b21b8690"}}
CLAIMS = {"synthetic_cpu_fixture_scope_only": True, "small_hook_parameter_count_is_not_native434_proof": True,
    "intended_phase_order": "topics32_then_pair_hooks32", "pair_runtime_getters_scope": "pair_fixture_phase_only",
    "combined_parent_is_stdlib_only": False, "topic_fit_scope": "fresh_child_synthetic64_questions_only",
    "older_tests_replayed": False, "actual_source_population_fitted": False, "actual_R15000_drawn": False,
    "actual_original_3b_weights_loaded": False, "private_cache_or_source_data_opened": False,
    "tokenizer_loaded": False, "native_bf16_cuda_admission_established": False,
    "backward_cuda_recompute_established": False, "native_original_model_or_memory_admission": False,
    "original_model_full_train_loop_established": False, "actual_unique17_pilot_driver_established": False,
    "scoring_or_accuracy_established": False, "gpu_or_training_admitted": False,
    "automatic_retry": False, "automatic_downstream": False}
THREAD_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
TOPIC_CHILDREN = {"maximum": 5, "timeout_seconds": 60, "sequential": True, "positive_attempts": 1,
    "parent_scientific_modules_absent": True, "real_source_population": False}
RUNTIME = {"path": "extension/operations/full-pool-runtime-source-metadata-20261021.json",
    "sha256": "5788c73cff4f26607c9af604a11634334cdec15a0b7b687c48968fac06b73436",
    "schema": "extension-source-only-selection-runtime-metadata-v1"}
RUNTIME_SOURCES = {
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/random.py": {"bytes": 33221, "sha256": "ce80a2471965e64ae93caea14490f16850432e65dce87f15edfe25b6f562f8d1"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/numpy/version.py": {"bytes": 216, "sha256": "3932e74a1d0d19f5b22fc56b9c88f435db7f29939397567e903f427fe8d13266"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/scipy/version.py": {"bytes": 318, "sha256": "a271fdebc4f15ae566cd3c8ca9686bcf52e123dd5d2254b1618acbd9d4292ebc"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/base.py": {"bytes": 47777, "sha256": "a2ad39c5ff6491d04c35738e942860def0e80d6800aeb1690cfd68bf46be8d91"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py": {"bytes": 81743, "sha256": "2e0da803f41cc870df45c59a24cd2da93c46d064785fef3e8d5c2e4912d9c803"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/decomposition/_truncated_svd.py": {"bytes": 11708, "sha256": "9fe23f1ebc98fc0bf27050d95adf30296119f2719433c05962e6011ed6faaa3d"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/feature_extraction/text.py": {"bytes": 77416, "sha256": "876f0d334735f77174822fabff287fab5602ccff8973a5430ebc70548916da62"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/preprocessing/_data.py": {"bytes": 127903, "sha256": "1255040a0eb9c99f58775d55cf34f675cf61539f7c0a0dd0d6a36cb1f0da1e15"}}
RUNTIME_GETTERS = {"float32_matmul_precision": "get_float32_matmul_precision",
    "cuda_matmul_allow_tf32": "backends.cuda.matmul.allow_tf32",
    "cuda_matmul_allow_fp16_reduced_precision_reduction": "backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction": "backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
    "cudnn_allow_tf32": "backends.cudnn.allow_tf32", "flash_sdp_enabled": "backends.cuda.flash_sdp_enabled",
    "mem_efficient_sdp_enabled": "backends.cuda.mem_efficient_sdp_enabled", "math_sdp_enabled": "backends.cuda.math_sdp_enabled",
    "cudnn_sdp_enabled": "backends.cuda.cudnn_sdp_enabled"}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def inside(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Expected explicit project-relative source/receipt metadata")
    path, current = root / name, root
    for part in Path(name).parts:
        current = current / part
        require(not current.is_symlink(), "Symlink metadata/source path refused")
    require(path.resolve().is_relative_to(root), "Metadata path escapes the project")
    return path


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_bytes(root, name):
    path = inside(root, name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 2 << 20, "Expected bounded ordinary source/receipt metadata")
        pieces, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            require(bool(chunk), "Truncated source/receipt metadata")
            pieces.append(chunk)
            remaining -= len(chunk)
        require(not os.read(fd, 1) and _stat_identity(before) == _stat_identity(os.fstat(fd)) == _stat_identity(path.stat()),
            "Metadata/source changed while reading")
        return b"".join(pieces)
    finally:
        os.close(fd)


def record(root, name):
    payload = _read_bytes(root, name)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def read_json(root, name, expected=None):
    before = record(root, name)
    require(expected is None or before["sha256"] == expected, "Pinned metadata FILE SHA differs")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON metadata key")
            result[key] = value
        return result
    def finite_float(raw):
        value = float(raw)
        require(math.isfinite(value), "Nonfinite JSON metadata")
        return value
    payload = _read_bytes(root, name)
    require({"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)} == before, "Metadata changed before parsing")
    value = json.loads(payload, object_pairs_hook=pairs, parse_float=finite_float,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON metadata")))
    require(record(root, name) == before, "Metadata changed during parsing")
    return value


def write_new_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        observed = os.fstat(fd)
        require(stat.S_ISREG(observed.st_mode) and observed.st_uid == os.getuid() and
            stat.S_IMODE(observed.st_mode) == 0o600 and observed.st_nlink == 1, "Owner0600 fresh receipt required")
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd is not None:
            os.close(fd)


def source_inventory(root):
    result = {name: record(root, name) for name in sorted(SOURCES)}
    require(len(result) == 7 and all(result[name]["sha256"] == sha for name, sha in PINS.items()),
        "Reviewed new pair-hook/topic helper or fixture bytes changed")
    return result


def runtime_source_evidence(root):
    """Only public metadata and eight small installed files; no library import."""
    receipt = read_json(root, RUNTIME["path"], RUNTIME["sha256"])
    topic_versions = {name: VERSIONS[name] for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib")}
    require(type(receipt) is dict and receipt.get("schema") == RUNTIME["schema"] and
        receipt.get("actual_small_source_file_inventory") == RUNTIME_SOURCES and receipt.get("versions") == topic_versions and
        receipt.get("native_runtime_qa_established") is False and
        receipt.get("python_executable") == str(root / ".venv/bin/python") and
        type(receipt.get("python_version")) is str and receipt["python_version"].startswith("3.10.19 "),
        "Exact source-only root runtime observation required")
    files = {RUNTIME["path"]: record(root, RUNTIME["path"])}
    for name, expected in RUNTIME_SOURCES.items():
        path = Path(name)
        require(path.is_absolute() and ".." not in path.parts and
            all(not item.is_symlink() for item in (path, *path.parents)), "Installed source path cannot be aliased")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and type(expected["bytes"]) is int and
                0 < before.st_size == expected["bytes"] <= 2 << 20, "Exact small installed source size required")
            checksum, remaining = hashlib.sha256(), before.st_size
            while remaining:
                block = os.read(fd, min(remaining, 1 << 20))
                require(bool(block), "Truncated installed runtime source")
                checksum.update(block); remaining -= len(block)
            require(not os.read(fd, 1) and _stat_identity(before) == _stat_identity(os.fstat(fd)) == _stat_identity(path.stat()) and
                checksum.hexdigest() == expected["sha256"], "Installed runtime source changed or differs from its exact FILE pin")
            files[name] = {"sha256": checksum.hexdigest(), "bytes": before.st_size}
        finally:
            os.close(fd)
    require(len(files) == 9, "Runtime receipt plus exactly eight source pins required")
    return {"receipt": {"path": RUNTIME["path"], **files[RUNTIME["path"]]},
        "input_files": files, "native_runtime_or_population_proven": False}


def validate_plan(value):
    require(type(value) is dict and value.get("schema") == "extension-native-pair-topics-qa-plan-v1" and
        value.get("role") == ROLE and value.get("test_patterns") == PATTERNS and
        type(value.get("expected_tests")) is int and value["expected_tests"] == EXPECTED_TESTS and
        value.get("source_files") == sorted(SOURCES) and value.get("reviewed_input_sha256") == PINS and
        value.get("runtime_versions") == VERSIONS and
        value.get("resources") == {"cpus": 2, "ram_gib": 8, "gpus": 0, "wall_time_seconds": WALL_SECONDS,
                                  "reserve_seconds": RESERVE_SECONDS} and
        value.get("deadline_utc") == DEADLINE.isoformat() and value.get("claims") == CLAIMS and
        value.get("native_qa_schema") == SCHEMA and
        value.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True} and
        value.get("protocol_sha256") == PROTOCOL_PINS and value.get("historical_native_qa") == HISTORICAL and
        value.get("prior_ledger") == PRIOR and value.get("parent_sources") == PARENT and value.get("historical_ledger") == HISTORY and
        value.get("test_order") == list(TEST_ORDER) and value.get("topic_children") == TOPIC_CHILDREN and
        value.get("thread_environment") == THREAD_ENVIRONMENT and value.get("topic_runtime_receipt") == RUNTIME and
        value.get("installed_runtime_sources") == RUNTIME_SOURCES and value.get("pair_runtime_getters") == RUNTIME_GETTERS and
        value.get("assertions_owner") == "Exact pinned new topics32 first, then new pair-hooks32; no old-scope discovery or stage override" and
        len(SOURCES) == 7 and len(PINS) == 4 and sum(PATTERNS.values()) == EXPECTED_TESTS,
        "Only the reviewed ordered new CPU64/source7 scope and sealed history are permitted")
    return value


def deadline_guard():
    now = datetime.now(timezone.utc)
    require((DEADLINE - now).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
        "Keep the complete CPU30-minute stage and48-hour closeout reserve")
    return {"checked_at_utc": now.isoformat(), "guard_seconds": WALL_SECONDS + RESERVE_SECONDS,
        "deadline_utc": DEADLINE.isoformat()}


def require_cpu_allocation():
    """Require the actual assigned running CPU allocation before scoped imports."""
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) is not None and node and
        socket.gethostname().split(".")[0] == node.split(".")[0] and
        node.split(".")[0].lower() not in {"owl1", "tc2"} and "login" not in node.lower(),
        "An actual assigned CPU compute process is required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key, "") in ("", "0", "(null)", "N/A")
        for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")), "GPU visibility/request forbidden")
    command = ["scontrol", "--clusters=owl", "show", "job", "--oneliner", job]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    require(not result.stderr.strip() and 0 < len(result.stdout.encode()) <= 128 << 10, "Incomplete allocation proof")
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result.stdout)
    fields = dict(pairs)
    require(len(fields) == len(pairs), "Duplicate allocation metadata fields")
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    tres = fields.get("AllocTRES", "")
    memory = re.search(r"(?:^|,)mem=([0-9]+)([KMGT])(?:,|$)", tres)
    mem_kib = None if memory is None else int(memory.group(1)) * {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}[memory.group(2)]
    require(fields.get("JobId") == job and fields.get("JobName") == NAME and fields.get("JobState") == "RUNNING" and
        user is not None and int(user.group(1)) == os.getuid() and fields.get("NumNodes") == "1" and fields.get("NodeList") == node and
        fields.get("NumCPUs") == "2" and re.search(r"(?:^|,)cpu=2(?:,|$)", tres) is not None and mem_kib == 8 * 1024**2 and
        fields.get("Partition") == "normal_q" and fields.get("Account") == "tml_2026" and
        fields.get("QOS") == "owl_normal_short" and fields.get("TimeLimit") == "00:30:00" and fields.get("Requeue") == "0" and
        all("gpu" not in text.lower() for key, text in fields.items() if "tres" in key.lower() or "gres" in key.lower()),
        "Exact CPU2/8GiB/30-minute single-node zero-GPU no-requeue allocation required")
    return {"job_id": job, "node": node, "hostname": socket.gethostname(), "user_id": os.getuid(), "state": "RUNNING",
        "allocated_cpus": 2, "allocated_ram_gib": 8, "allocated_tres": tres, "gpu_requested_or_allocated": False,
        "step_id": os.environ.get("SLURM_STEP_ID", "batch"), "cuda_visible_devices": "", "command": command,
        "scontrol": result.stdout.strip(), "scontrol_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def validate_wrapper(root):
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "normal_q", "qos": "owl_normal_short",
        "nodes": "1", "ntasks": "1", "cpus-per-task": "2", "mem": "8G", "time": "00:30:00", "no-requeue": ""}
    found, header = {}, True
    for line in inside(root, SCRIPT).read_text().splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"): header = False
            continue
        require(header, "Late scheduler directive")
        parts = shlex.split(line.lstrip()[7:].strip())
        require(bool(parts), "Empty scheduler directive")
        option, separator, arg = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in set(expected) | {"output", "error"} and option[2:] not in found and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/ambiguous/duplicate scheduler directive")
        found[option[2:]] = arg if separator else parts[1] if len(parts) == 2 else ""
    require(set(found) == set(expected) | {"output", "error"} and all(found[key] == item for key, item in expected.items()),
        "Only the exact finite CPU wrapper is permitted")
    for name in ("output", "error"):
        path = Path(found[name])
        require(path.is_absolute() and ".." not in path.parts and path.parent == root / "extension/logs",
            "Bounded extension logs required")
    return found


def historical_qa_evidence(root, protected):
    results, files = {}, {}
    for label, spec in HISTORICAL.items():
        values = {}
        for name, pin in (("native_qa.json", "native_qa_sha256"), ("complete.json", "complete_sha256"),
                          ("accounting.json", "accounting_sha256")):
            path = spec["directory"] + "/" + name
            values[name] = read_json(root, path, spec[pin])
            files[path] = record(root, path)
        qa, complete, accounting = (values[name] for name in ("native_qa.json", "complete.json", "accounting.json"))
        require(qa.get("schema") == spec["qa_schema"] and qa.get("success") is True and qa.get("job_id") == spec["job_id"] and
            type(qa.get("tests_run")) is int and qa["tests_run"] == spec["tests"] and qa.get("test_counts") == spec["test_counts"] and
            qa.get("gpu_disabled") is True and qa.get("input_snapshot_unchanged") is True and
            all(type(qa.get(key)) is int and qa[key] == 0 for key in
                ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Historical native CPU80 did not pass its exact scope")
        sources = qa.get("source_inventory", {})
        require(type(sources) is dict and len(sources) == spec["source_files"] and
            all(protected.get(name) == item for name, item in sources.items()) and
            qa.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
            spec["plan_path"] in sources and qa.get("plan_sha256") == sources[spec["plan_path"]]["sha256"],
            "Historical passed sources/plan differ from protected204 bytes")
        require(complete.get("schema") == spec["complete_schema"] and complete.get("complete") is True and
            complete.get("job_id") == spec["job_id"] and complete.get("tests_run") == spec["tests"] and
            complete.get("cpu_only") is True and complete.get("gpu_seconds") == 0 and complete.get("input_snapshot_unchanged") is True and
            complete.get("source_inventory") == sources and complete.get("plan_sha256") == qa["plan_sha256"] and
            complete.get("native_qa_sha256") == spec["native_qa_sha256"], "Historical CPU80 completion/QA/source bindings differ")
        command = ["sacct", "--clusters=owl", "-j", spec["job_id"], "--parsable2", "--noheader",
            "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,AllocTRES"]
        require(accounting.get("schema") == "extension-terminal-cpu-accounting-v1" and accounting.get("complete") is True and
            accounting.get("job_id") == spec["job_id"] and accounting.get("state") == "COMPLETED" and accounting.get("exit_code") == "0:0" and
            type(accounting.get("elapsed_seconds")) is int and accounting["elapsed_seconds"] == spec["row_elapsed"][0] and
            accounting.get("allocated_gpus") == 0 and accounting.get("gpu_seconds") == 0 and accounting.get("no_gpu_in_alloc_tres") is True and
            accounting.get("timezone") == "UTC" and accounting.get("command") == command and accounting.get("stderr") == "" and
            type(accounting.get("stdout")) is str, "Historical exact terminal CPU80 accounting differs")
        rows = [line.split("|") for line in accounting["stdout"].splitlines()]
        require(len(rows) == 3 and all(len(row) == 8 for row in rows), "Historical allocation/batch/extern rows required")
        for index, (suffix, name) in enumerate((("", spec["name"]), (".batch", "batch"), (".extern", "extern"))):
            row = rows[index]
            require(row[:5] == [spec["job_id"] + suffix, name, "COMPLETED", "0:0", str(spec["row_elapsed"][index])] and
                "gpu" not in row[7].lower() and re.search(r"(?:^|,)cpu=2(?:,|$)", row[7]) is not None and
                re.search(r"(?:^|,)mem=8G(?:,|$)", row[7]) is not None and re.search(r"(?:^|,)node=1(?:,|$)", row[7]) is not None and
                (datetime.fromisoformat(row[6]) - datetime.fromisoformat(row[5])).total_seconds() == spec["row_elapsed"][index],
                "Historical exact CPU80 raw name/step/duration/TRES differs")
        results[label] = {"job_id": spec["job_id"], "tests_previously_passed": spec["tests"], "tests_replayed": False,
            "native_qa_sha256": spec["native_qa_sha256"], "complete_sha256": spec["complete_sha256"],
            "accounting_sha256": spec["accounting_sha256"], "actual_allocation_batch_extern_seconds": spec["row_elapsed"]}
    return {"scopes": results, "input_files": files}


def load_context(root, plan_path, allocation):
    sources = source_inventory(root)
    require(plan_path == PLAN, "The exact reviewed plan path is required")
    plan = validate_plan(read_json(root, plan_path))
    prior = read_json(root, PRIOR["path"], PRIOR["sha256"])
    require(prior.get("schema") == LEDGER_SCHEMA and prior.get("gpu_hour_cap", "missing") is None and
        prior.get("historical_ledger_unchanged") == HISTORY["path"] and type(prior.get("jobs")) is dict and
        len(prior["jobs"]) == PRIOR["entries"] and value_digest(prior["jobs"]) == PRIOR["entries_sha256"] and ROLE not in prior["jobs"] and
        all(type(item) is dict for item in prior["jobs"].values()), "Exact sealed prior35 history required")
    parent = read_json(root, PARENT["path"], PARENT["sha256"])
    protected = parent.get(PARENT["inventory_key"], {})
    require(parent.get("schema") == PARENT["schema"] and parent.get("raw_private_records_included") is False and
        type(protected) is dict and len(protected) == PARENT["expected_files"] and not set(protected) & SOURCES,
        "Exact terminal protected204 parent source inventory required")
    for name, item in protected.items():
        require(type(item) is dict and set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and 0 < item["bytes"] <= 1 << 20 and
            type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None and record(root, name) == item,
            "Protected source changed before fixture imports: " + name)
    historical = record(root, HISTORY["path"])
    require(historical["sha256"] == HISTORY["sha256"], "Historical completed-experiment ledger changed")
    verified_scopes = historical_qa_evidence(root, protected)
    runtime_sources = runtime_source_evidence(root)
    for name, expected in PROTOCOL_PINS.items():
        require(record(root, name)["sha256"] == expected and protected.get(name, {}).get("sha256") == expected,
            "Frozen protocol differs from protected parent bytes")
    bootstrap = record(root, BOOTSTRAP)
    ledger = read_json(root, "extension/job_ledger.json")
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == HISTORY["path"] and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) == set(prior["jobs"]) | {ROLE} and
        all(ledger["jobs"].get(name) == item for name, item in prior["jobs"].items()), "Only this CPU role may extend sealed35 history")
    entry = ledger["jobs"][ROLE]
    require(type(entry) is dict and type(entry.get("binding")) is dict, "Current CPU admission binding missing")
    binding = entry["binding"]
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    require(entry.get("status") == "submitted" and entry.get("job_id") == allocation["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and entry.get("source_sha256") == sources and
        entry.get("command") == command == binding.get("command") and entry.get("binding_sha256") == value_digest(binding) and
        binding.get("role") == ROLE and binding.get("source_sha256") == sources and binding.get("plan_sha256") == sources[PLAN]["sha256"] and
        binding.get("prior_entries_sha256") == PRIOR["entries_sha256"] and binding.get("expected_native_tests") == EXPECTED_TESTS and
        binding.get("wall_time_seconds") == WALL_SECONDS and binding.get("closeout_reserve_seconds") == RESERVE_SECONDS and
        binding.get("requested_gpu_seconds") == 0 and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP) == bootstrap and binding.get("parent_source_manifest_sha256") == PARENT["sha256"] and
        binding.get("protocol_sha256") == PROTOCOL_PINS and
        binding.get("historical_qa_evidence_sha256") == {name: item["sha256"] for name, item in verified_scopes["input_files"].items()} and
        binding.get("runtime_source_evidence_sha256") == {name: item["sha256"] for name, item in runtime_sources["input_files"].items()} and
        binding.get("test_order") == list(TEST_ORDER) and
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False,
        "Exact current-job CPU64/source7/protected204/plan/protocol/history/bootstrap admission required")
    require(not any(allocation["job_id"] == item.get(key) for item in prior["jobs"].values()
        for key in ("job_id", "possible_job_id")), "Current scheduler ID collides with prior history")
    validate_wrapper(root)
    return {"sources": sources, "plan": plan, "ledger": ledger, "admission": entry, "bootstrap": bootstrap,
        "prior_snapshot": record(root, PRIOR["path"]), "historical_ledger": historical, "parent_manifest": record(root, PARENT["path"]),
        "protected_source_inventory": protected, "historical_qa_verified_only": verified_scopes,
        "runtime_source_evidence": runtime_sources}


def pre_import_policy():
    require(sys.implementation.name == VERSIONS["implementation"] and
        ".".join(map(str, sys.version_info[:3])) == VERSIONS["python"] and sys.dont_write_bytecode is True and
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Pinned CPython3.10.19/no-bytecode process required")
    require({name: os.environ.get(name) for name in THREAD_ENVIRONMENT} == THREAD_ENVIRONMENT,
        "Observe the unchanged root-sealed thread1 policy; do not set it inside Python")
    require(os.environ.get("TOPIC_QA_RUNTIME_RECEIPT") == str(Path(__file__).resolve().parents[1] / RUNTIME["path"]),
        "Exact fixed TOPIC_QA_RUNTIME_RECEIPT required")
    require({name: importlib.metadata.version(name) for name in VERSIONS if name not in ("python", "implementation")} ==
        {name: value for name, value in VERSIONS.items() if name not in ("python", "implementation")},
        "Pinned installed metadata versions differ before fixture imports")


def require_fresh_topics_parent():
    forbidden = ("torch", "numpy", "scipy", "sklearn", "threadpoolctl", "joblib", "transformers", "accelerate")
    require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden),
        "Topics32 must complete before parent scientific/Torch/pair imports")


def stdlib_runtime_metadata(root):
    pre_import_policy()
    require_fresh_topics_parent()
    return {"scope": "topics_parent_stdlib_only_no_scientific_import", "python": sys.version,
        "implementation": sys.implementation.name, "dont_write_bytecode": True,
        "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
        "runtime_source_evidence": runtime_source_evidence(root), "native_population_or_gpu_proven": False}


def runtime_metadata():
    pre_import_policy()
    import torch
    versions = {name: importlib.metadata.version(name) for name in VERSIONS if name not in ("python", "implementation")}
    require(versions == {name: value for name, value in VERSIONS.items() if name not in ("python", "implementation")} and
        str(torch.__version__).split("+")[0] == VERSIONS["torch"] and torch.get_default_dtype() == torch.float32 and
        torch.get_default_device().type == "cpu" and not torch.cuda.is_initialized(),
        "Pinned unchanged FP32/default-CPU fixture runtime required")
    backend = {}
    for key, name in RUNTIME_GETTERS.items():
        value = torch
        for component in name.split("."): value = getattr(value, component)
        value = value() if callable(value) else value
        require((key == "float32_matmul_precision" and type(value) is str and value in {"highest", "high", "medium"}) or
            (key != "float32_matmul_precision" and type(value) is bool), "Unexpected read-only backend getter value")
        backend[key] = value
    require(not torch.cuda.is_initialized(), "Read-only backend getters must not initialize CUDA")
    return {"scope": "pair_fixture_phase_only_after_topics32", "python": sys.version, "versions": versions, "implementation": sys.implementation.name,
        "default_dtype": str(torch.get_default_dtype()), "default_device": str(torch.get_default_device()),
        "cuda_initialized": False, "dont_write_bytecode": True, "actual_original_model_or_cuda_loaded": False,
        "backend_getters": backend, "runtime_flags_set_or_swept": False}


def run(root, plan_path, output):
    root = Path(root).resolve()
    require(root == Path(__file__).resolve().parents[1], "Run only the installed source project")
    allocation = require_cpu_allocation()
    guard = deadline_guard()
    context = load_context(root, plan_path, allocation)
    output = Path(output)
    require(output == root / "extension/runs" / ("native-pair-topics-qa-" + allocation["job_id"]) and
        not output.exists() and not output.is_symlink(), "Exclusive exact current-job output required; no alias/resume")
    inside(root, str(output.relative_to(root)))
    parent_stat = output.parent.stat()
    require(not output.parent.is_symlink() and stat.S_ISDIR(parent_stat.st_mode) and parent_stat.st_uid == os.getuid(),
        "Assigned-user ordinary runs directory required")
    pre_import_policy()
    subprocess.run(["bash", "-n", str(root / SCRIPT)], check=True)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    observed = output.stat()
    require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid() and stat.S_IMODE(observed.st_mode) == 0o700,
        "Owner0700 fresh output required")
    base = {"schema": SCHEMA, "job_id": allocation["job_id"], "node": allocation["node"], "compute_proof": allocation,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_inventory": context["sources"],
        "code_sha256": {name: item["sha256"] for name, item in context["sources"].items()},
        "plan_sha256": context["sources"][PLAN]["sha256"], "prior_entries_sha256": PRIOR["entries_sha256"],
        "prior_snapshot": context["prior_snapshot"], "historical_ledger": context["historical_ledger"],
        "parent_source_manifest": context["parent_manifest"], "protected_source_inventory": context["protected_source_inventory"],
        "protocol_sha256": PROTOCOL_PINS, "historical_qa_verified_only": context["historical_qa_verified_only"],
        "admission_binding_sha256": context["admission"]["binding_sha256"], "cpu_bootstrap_sha256": context["bootstrap"]["sha256"],
        "runtime_policy": VERSIONS, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "test_patterns": PATTERNS, "test_order": list(TEST_ORDER), "expected_tests": EXPECTED_TESTS,
        "topic_children_policy": TOPIC_CHILDREN, "runtime_source_evidence": context["runtime_source_evidence"],
        "deadline_guard": guard, **CLAIMS}
    write_new_json(output / "started.json", base)
    metrics = {"tests_run": 0, "test_counts": {pattern: 0 for pattern in PATTERNS}, "failures": 0, "errors": 0,
        "skipped": 0, "expected_failures": 0, "unexpected_successes": 0}
    begun = time.monotonic()
    phase_results, discovered_counts = [], {}
    try:
        # Topic test28 requires this parent to remain free of scientific
        # imports. Do not construct a combined suite or import Torch here.
        topics_before = stdlib_runtime_metadata(root)
        write_new_json(output / "topics_runtime_before.json", {"schema": "extension-native-pair-topics-stdlib-runtime-v1",
            "job_id": allocation["job_id"], "runtime": topics_before})
        for pattern in TEST_ORDER:
            expected = PATTERNS[pattern]
            if pattern == TEST_ORDER[0]:
                require_fresh_topics_parent()
            else:
                require(metrics["test_counts"][TEST_ORDER[0]] == 32 and len(phase_results) == 1 and phase_results[0]["success"] is True,
                    "No pair fixture imports before all topics32 passes")
                require(load_context(root, plan_path, allocation) == context, "Sources/history/runtime/ledger changed between fixture phases")
                before_runtime = runtime_metadata()
                write_new_json(output / "runtime_before.json", {"schema": "extension-native-pair-topics-runtime-v1",
                    "job_id": allocation["job_id"], "runtime": before_runtime, "native_execution_or_population_proof": False})
            subset = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern, top_level_dir=str(root / "tests"))
            discovered_counts[pattern] = subset.countTestCases()
            require(discovered_counts[pattern] == expected, "Exact new authored count required; no older scope discovery/replay")
            result = unittest.TextTestRunner(verbosity=2).run(subset)
            current = {"pattern": pattern, "tests_run": result.testsRun, "expected_tests": expected,
                "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
                "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
                "success": result.wasSuccessful()}
            phase_results.append(current)
            metrics["tests_run"] += result.testsRun
            metrics["test_counts"][pattern] = result.testsRun
            for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"):
                metrics[key] += current[key]
            write_new_json(output / ("phase-" + str(len(phase_results)) + ".json"), current)
            require(result.testsRun == expected and result.wasSuccessful() and all(current[key] == 0 for key in
                ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")),
                "This exact fixture phase failed; do not continue, refill or retry")
            if pattern == TEST_ORDER[0]:
                topics_after = stdlib_runtime_metadata(root)
                require(topics_after == topics_before, "Topic parent/source/bytecode/thread metadata changed")
                tested = sys.modules.get("test_full_pool_topics")
                require(tested is not None and getattr(tested, "MAX_CHILDREN", None) == 5 and
                    getattr(tested, "CHILD_TIMEOUT", None) == 60 and type(getattr(tested, "_CHILD_COUNT", None)) is int and
                    tested._CHILD_COUNT == 5 and getattr(tested, "_POSITIVE_ATTEMPTED", None) is True and
                    type(getattr(tested, "_POSITIVE", None)) is dict,
                    "Exactly the sealed five sequential topic fixture children and one positive attempt required")
                write_new_json(output / "topics_runtime_after.json", {"schema": "extension-native-pair-topics-stdlib-runtime-v1",
                    "job_id": allocation["job_id"], "runtime": topics_after, "unchanged_from_before": True,
                    "actual_fixture_children": 5, "positive_attempts": 1, "no_real_population_fit": True})
        after_runtime = runtime_metadata()
        write_new_json(output / "runtime_after.json", {"schema": "extension-native-pair-topics-runtime-v1",
            "job_id": allocation["job_id"], "runtime": after_runtime, "unchanged_from_before": after_runtime == before_runtime,
            "native_execution_or_population_proof": False})
        require(metrics["tests_run"] == EXPECTED_TESTS and metrics["test_counts"] == PATTERNS and
            discovered_counts == PATTERNS and len(phase_results) == 2 and all(metrics[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Ordered new CPU64 fixtures did not all pass")
        require(after_runtime == before_runtime, "Fixtures changed the pinned CPU/default precision runtime")
        require(load_context(root, plan_path, allocation) == context, "Source/history/current-ledger/bootstrap inputs changed during QA")
        ending = require_cpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "CPU allocation changed")
        receipt = {**base, **metrics, "success": True, "compute_end": ending, "phase_results": phase_results,
            "discovered_test_counts": discovered_counts, "topics_parent_runtime_before": topics_before,
            "topics_parent_runtime_after": topics_after, "topics_parent_runtime_unchanged": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_elapsed_seconds": time.monotonic() - begun,
            "runtime_before": before_runtime, "runtime_after": after_runtime, "runtime_unchanged": True,
            "runtime_before_sha256": record(root, str((output / "runtime_before.json").relative_to(root)))["sha256"],
            "runtime_after_sha256": record(root, str((output / "runtime_after.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "started_sha256": record(root, str((output / "started.json").relative_to(root)))["sha256"]}
        write_new_json(output / "native_qa.json", receipt)
        write_new_json(output / "complete.json", {"schema": "extension-native-pair-topics-qa-complete-v1", "complete": True,
            "job_id": allocation["job_id"], "tests_run": EXPECTED_TESTS, "gpu_seconds": 0, "cpu_only": True,
            "source_inventory": context["sources"], "plan_sha256": base["plan_sha256"],
            "parent_source_manifest": context["parent_manifest"], "protected_source_inventory": context["protected_source_inventory"],
            "protocol_sha256": PROTOCOL_PINS, "historical_qa_verified_only": context["historical_qa_verified_only"],
            "runtime_source_evidence": context["runtime_source_evidence"], "test_order": list(TEST_ORDER),
            "test_counts": metrics["test_counts"], "phase_results": phase_results,
            "runtime_before": before_runtime, "runtime_after": after_runtime, "runtime_unchanged": True,
            "native_qa_sha256": record(root, str((output / "native_qa.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "created_at_utc": datetime.now(timezone.utc).isoformat(), **CLAIMS})
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {**base, **metrics, "success": False, "error_type": type(error).__name__,
            "message": "Ordered native-pair/topics CPU64 fixture stage failed; inspect its native log without retry",
            "phase_results": phase_results, "discovered_test_counts": discovered_counts,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begun,
            "input_snapshot_unchanged": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.plan, args.output)
    print(json.dumps({"native_pair_topics_cpu_fixture_qa_complete": True, "tests_run": EXPECTED_TESTS,
        "gpu_seconds": 0, "actual_original_model_or_memory_admission": False}), flush=True)


if __name__ == "__main__":
    main()
