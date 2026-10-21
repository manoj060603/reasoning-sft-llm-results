"""Prospective CPU32 standalone training_step diagnostic fixture stage.

All pre-test imports use only stdlib metadata/control. The32 new tests construct
random tiny CPU Qwen classes and explicitly small nonnative comparison hooks.
No original3B model, tokenizer, private corpus, CUDA or actual434 gradients are
loaded. Exact historical CPU80 receipts are verified only, never replayed.
A pass does not admit original-model numerical, memory, full.train or pilots.
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


ROLE = "standalone_training_step_qa_20261021_v1"
NAME = "sft-extension-standalone-training-step-qa"
PLAN = "extension/standalone_training_step_qa_plan.json"
SCRIPT = "extension/standalone_training_step_qa.slurm"
BOOTSTRAP = "extension/operations/submit-standalone-training-step-qa-20261021.py"
SCHEMA = "extension-standalone-training-step-native-qa-v1"
LEDGER_SCHEMA = "reasoning-sft-extension-job-ledger-v1"
PATTERNS = {"test_standalone_training_step_controller.py": 32}
EXPECTED_TESTS = 32
WALL_SECONDS, RESERVE_SECONDS = 1800, 172800
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
PROTOCOL_PINS = {}
PINS = {
    "extension/standalone_training_step_controller.py": "3565bf08c76594682102f3f8ddc12551a2380df78d3980d7f8d25f8cae6894f0",
    "tests/test_standalone_training_step_controller.py": "d530499f13cad30e62032dca2fe4264fc85442b212aac0586db07751fd6970fa"}
SOURCES = frozenset(PINS) | {PLAN, SCRIPT, "extension/standalone_training_step_qa.py"}
VERSIONS = {"python": "3.10.19", "implementation": "cpython", "torch": "2.9.0", "numpy": "1.26.4",
    "transformers": "4.57.3", "accelerate": "1.12.0", "safetensors": "0.7.0"}
PRIOR = {"path": "extension/report/full-pool-tools-qa-20261021-1230197/job_ledger_snapshot.json",
    "sha256": "12b92a578a9f19c6ac2841e1e9242eb6963bd5885291392415c92cb98875bd0e", "entries": 34,
    "entries_sha256": "882703da0a4106b77b2450f02cc787e249eae8f0a4e46321c6a12dbdf5be8416"}
PARENT = {"path": "extension/report/full-pool-tools-qa-20261021-1230197/evidence_manifest.json",
    "sha256": "a241bdcf4ed8fd9cb3c83ccd89158753a9e6459f7f1c37c187ff494e5f54c661",
    "schema": "extension-full-pool-tools-qa-evidence-v1", "inventory_key": "protected_source_inventory", "expected_files": 198}
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
    "accounting_sha256": "ea6b2c341453532d845654606973980b19d37f3fb70c9ea58121817fcbaad261"}}
CLAIMS = {'tiny_random_cpu_qwen_standalone_fixture_scope_only': True, 'small_hook_parameter_count_is_not_native434_proof': True, 'intended_inherited_cpu_training_step_fixture_scope': True, 'older_tests_replayed': False, 'actual_original_3b_weights_loaded': False, 'private_cache_or_source_data_opened': False, 'tokenizer_loaded': False, 'native_bf16_cuda_admission_established': False, 'backward_cuda_recompute_established': False, 'native_original_model_or_memory_admission': False, 'original_model_full_train_loop_established': False, 'actual_unique17_pilot_driver_established': False, 'gpu_or_training_admitted': False, 'automatic_retry': False, 'automatic_downstream': False}

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
    require(len(result) == 5 and all(result[name]["sha256"] == sha for name, sha in PINS.items()),
        "Reviewed standalone controller/test bytes changed")
    return result


def validate_plan(value):
    require(type(value) is dict and value.get("schema") == "extension-standalone-training-step-qa-plan-v1" and
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
        value.get("assertions_owner") == "Exact assertions in the pinned32 standalone-controller tiny CPU test module; no stage override" and
        len(SOURCES) == 5 and len(PINS) == 2 and sum(PATTERNS.values()) == EXPECTED_TESTS,
        "Only the reviewed standalone CPU32/source5 scope and sealed history are permitted")
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
            "Historical passed sources/plan differ from protected198 bytes")
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
        all(type(item) is dict for item in prior["jobs"].values()), "Exact sealed prior34 history required")
    parent = read_json(root, PARENT["path"], PARENT["sha256"])
    protected = parent.get(PARENT["inventory_key"], {})
    require(parent.get("schema") == PARENT["schema"] and parent.get("raw_private_records_included") is False and
        type(protected) is dict and len(protected) == PARENT["expected_files"] and not set(protected) & SOURCES,
        "Exact terminal protected198 parent source inventory required")
    for name, item in protected.items():
        require(type(item) is dict and set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and 0 < item["bytes"] <= 1 << 20 and
            type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None and record(root, name) == item,
            "Protected source changed before fixture imports: " + name)
    historical = record(root, HISTORY["path"])
    require(historical["sha256"] == HISTORY["sha256"], "Historical completed-experiment ledger changed")
    verified_scopes = historical_qa_evidence(root, protected)
    bootstrap = record(root, BOOTSTRAP)
    ledger = read_json(root, "extension/job_ledger.json")
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == HISTORY["path"] and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) == set(prior["jobs"]) | {ROLE} and
        all(ledger["jobs"].get(name) == item for name, item in prior["jobs"].items()), "Only this CPU role may extend sealed34 history")
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
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False,
        "Exact current-job CPU32/source5/protected198/plan/protocol/history/bootstrap admission required")
    require(not any(allocation["job_id"] == item.get(key) for item in prior["jobs"].values()
        for key in ("job_id", "possible_job_id")), "Current scheduler ID collides with prior history")
    validate_wrapper(root)
    return {"sources": sources, "plan": plan, "ledger": ledger, "admission": entry, "bootstrap": bootstrap,
        "prior_snapshot": record(root, PRIOR["path"]), "historical_ledger": historical, "parent_manifest": record(root, PARENT["path"]),
        "protected_source_inventory": protected, "historical_qa_verified_only": verified_scopes}


def pre_import_policy():
    require(sys.implementation.name == VERSIONS["implementation"] and
        ".".join(map(str, sys.version_info[:3])) == VERSIONS["python"] and sys.dont_write_bytecode is True and
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Pinned CPython3.10.19/no-bytecode process required")


def runtime_metadata():
    pre_import_policy()
    import torch
    versions = {name: importlib.metadata.version(name) for name in VERSIONS if name not in ("python", "implementation")}
    require(versions == {name: value for name, value in VERSIONS.items() if name not in ("python", "implementation")} and
        str(torch.__version__).split("+")[0] == VERSIONS["torch"] and torch.get_default_dtype() == torch.float32 and
        torch.get_default_device().type == "cpu" and not torch.cuda.is_initialized(),
        "Pinned unchanged FP32/default-CPU fixture runtime required")
    return {"python": sys.version, "versions": versions, "implementation": sys.implementation.name,
        "default_dtype": str(torch.get_default_dtype()), "default_device": str(torch.get_default_device()),
        "cuda_initialized": False, "dont_write_bytecode": True, "actual_original_model_or_cuda_loaded": False}


def run(root, plan_path, output):
    root = Path(root).resolve()
    require(root == Path(__file__).resolve().parents[1], "Run only the installed source project")
    allocation = require_cpu_allocation()
    guard = deadline_guard()
    context = load_context(root, plan_path, allocation)
    output = Path(output)
    require(output == root / "extension/runs" / ("standalone-training-step-qa-" + allocation["job_id"]) and
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
        "test_patterns": PATTERNS, "expected_tests": EXPECTED_TESTS, "deadline_guard": guard, **CLAIMS}
    write_new_json(output / "started.json", base)
    metrics = {"tests_run": 0, "test_counts": {pattern: 0 for pattern in PATTERNS}, "failures": 0, "errors": 0,
        "skipped": 0, "expected_failures": 0, "unexpected_successes": 0}
    begun = time.monotonic()
    try:
        # No target helper/test or Torch is imported before the guards above.
        before_runtime = runtime_metadata()
        write_new_json(output / "runtime_before.json", {"schema": "extension-standalone-training-step-runtime-v1",
            "job_id": allocation["job_id"], "runtime": before_runtime, "native_execution_or_population_proof": False})
        suite, counts = unittest.TestSuite(), {}
        for pattern, expected in PATTERNS.items():
            subset = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern, top_level_dir=str(root / "tests"))
            counts[pattern] = subset.countTestCases()
            require(counts[pattern] == expected, "Exact new authored counts required; no older scope replay")
            suite.addTest(subset)
        require(sum(counts.values()) == EXPECTED_TESTS, "Exact32 standalone tiny CPU scope required")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        metrics = {"tests_run": result.testsRun, "test_counts": counts, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses)}
        after_runtime = runtime_metadata()
        write_new_json(output / "runtime_after.json", {"schema": "extension-standalone-training-step-runtime-v1",
            "job_id": allocation["job_id"], "runtime": after_runtime, "unchanged_from_before": after_runtime == before_runtime,
            "native_execution_or_population_proof": False})
        require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and all(metrics[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Standalone CPU32 fixtures did not all pass")
        require(after_runtime == before_runtime, "Fixtures changed the pinned CPU/default precision runtime")
        require(load_context(root, plan_path, allocation) == context, "Source/history/current-ledger/bootstrap inputs changed during QA")
        ending = require_cpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "CPU allocation changed")
        receipt = {**base, **metrics, "success": True, "compute_end": ending,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_elapsed_seconds": time.monotonic() - begun,
            "runtime_before": before_runtime, "runtime_after": after_runtime, "runtime_unchanged": True,
            "runtime_before_sha256": record(root, str((output / "runtime_before.json").relative_to(root)))["sha256"],
            "runtime_after_sha256": record(root, str((output / "runtime_after.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "started_sha256": record(root, str((output / "started.json").relative_to(root)))["sha256"]}
        write_new_json(output / "native_qa.json", receipt)
        write_new_json(output / "complete.json", {"schema": "extension-standalone-training-step-qa-complete-v1", "complete": True,
            "job_id": allocation["job_id"], "tests_run": EXPECTED_TESTS, "gpu_seconds": 0, "cpu_only": True,
            "source_inventory": context["sources"], "plan_sha256": base["plan_sha256"],
            "parent_source_manifest": context["parent_manifest"], "protected_source_inventory": context["protected_source_inventory"],
            "protocol_sha256": PROTOCOL_PINS, "historical_qa_verified_only": context["historical_qa_verified_only"],
            "runtime_before": before_runtime, "runtime_after": after_runtime, "runtime_unchanged": True,
            "native_qa_sha256": record(root, str((output / "native_qa.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "created_at_utc": datetime.now(timezone.utc).isoformat(), **CLAIMS})
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {**base, **metrics, "success": False, "error_type": type(error).__name__,
            "message": "Standalone CPU32 fixture stage failed; inspect its native log without retry",
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
    print(json.dumps({"standalone_training_step_cpu_fixture_qa_complete": True, "tests_run": EXPECTED_TESTS,
        "gpu_seconds": 0, "actual_original_model_or_memory_admission": False}), flush=True)


if __name__ == "__main__":
    main()
