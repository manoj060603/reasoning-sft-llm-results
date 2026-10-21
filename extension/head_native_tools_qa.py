"""Unadmitted CPU-only projection, observation and paired-controller fixtures.

Imports above the guarded run are standard-library only. This controller never
opens a corpus, private cache, original-model checkpoint, tokenizer or benchmark result.
The exact80 tests use synthetic CPU tensors, tiny Trainers and explicit fake
observers. They do NOT admit original3B/434, native BF16/CUDA/recomputation,
RSS/headroom, actual shadow phase identity or a full train loop. Historical
36/24/32 receipts are verified only and are NOT replayed or aliased. A separate
metadata bootstrap must admit this fixed role after the exact prior32 closeout.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import unittest


ROLE = "head_native_tools_qa_20261021_v1"
NAME = "sft-extension-head-native-tools-qa"
PLAN = "extension/head_native_tools_qa_plan.json"
SCRIPT = "extension/head_native_tools_qa.slurm"
BOOTSTRAP = "extension/operations/submit-head-native-tools-qa-20261021.py"
SCHEMA = "extension-head-native-tools-native-qa-v1"
LEDGER_SCHEMA = "reasoning-sft-extension-job-ledger-v1"
PATTERNS = {"test_native_bf16_projection_comparison.py": 24,
    "test_checkpointed_head_native_observation.py": 24,
    "test_paired_training_step_controller.py": 32}
EXPECTED_TESTS = 80
WALL_SECONDS, RESERVE_SECONDS = 1800, 172800
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
PROTOCOL_PINS = {
    "extension/planning/checkpointed_head_native_numerical_protocol.json": "27cedace388a17f3b97a0b1466e119c59a4d63913085263c689c8cfbbb003cb3",
    "extension/planning/paired_training_step_controller_protocol.json": "1c9c0aa078b1c31b7b589ca51dd36651f69b5cef2efbaec44af271ac17148a2b"}
PINS = {
    "extension/native_bf16_projection_comparison.py": "a1fcf8637450c5b15fa990dfcd26cb267b54b87b59e10a4735bf25324da4fb22",
    "tests/test_native_bf16_projection_comparison.py": "08c187824458b47fc6eec2e0847b122930995a0e50a8e7dcf4a8defe86ae3e2b",
    "extension/checkpointed_head_native_observation.py": "555e40b513a1034753b7055123207bc6ae2aa4b855b06de8ee82e878a49b486a",
    "tests/test_checkpointed_head_native_observation.py": "90c6df32ce892d197173744a00cbda671a4acbc8379f6210c418ce00a7a53a20",
    "extension/paired_training_step_controller.py": "9e446719e34f2370a00967066813b28acb6bf0a072cfade473eda3ef5e59e726",
    "tests/test_paired_training_step_controller.py": "806fd6da4fc25b62126d2909c4e8bf5b1148d427ed4fa924e12b2f43c636e852",
    **PROTOCOL_PINS}
SOURCES = frozenset(PINS) | {PLAN, SCRIPT, "extension/head_native_tools_qa.py"}
VERSIONS = {"torch": "2.9.0", "transformers": "4.57.3", "accelerate": "1.12.0", "safetensors": "0.7.0", "numpy": "1.26.4"}
CLAIMS = {"synthetic_cpu_projection_observation_and_controller_scope_only": True,
    "tiny_cpu_trainer_fixture_scope": True, "fake_cuda_observation_fixture_scope": True,
    "runtime_precision_backend_getter_metadata_only": True, "runtime_flags_set_or_swept": False,
    "runtime_getters_establish_kernel_or_gpu_behavior": False,
    "fixture_tensor_count_override_is_not_native_434_proof": True, "older_tests_replayed": False,
    "actual_original_3b_weights_loaded": False, "actual_434_numeric_comparison_established": False,
    "private_experiment_inputs_opened": False, "tokenizer_loaded": False,
    "native_bf16_cuda_or_recompute_admission_established": False, "actual_process_rss_or_cuda_headroom_measured": False,
    "initial_shadow_phase_identity_established": False, "full_train_loop_or_long_context_admission_established": False,
    "training_or_pilot_admitted": False,
    "automatic_retry": False, "automatic_downstream": False}
HISTORICAL = {
    "head36": {"directory": "extension/report/head-objective-qa-20261021-1228995", "job_id": "1228995",
        "name": "sft-extension-head-objective-qa", "tests": 36, "row_elapsed": [23, 23, 23],
        "qa_schema": "extension-head-objective-native-qa-v1", "complete_schema": "extension-head-objective-qa-complete-v1",
        "native_qa_sha256": "7e160735bcc8e3de3b9e34b8c9819e36a61e8e704ab19ca5bf8f7972d8095cc1",
        "complete_sha256": "90ed243cfc47e592089bab63c48546d1577e1fd23428aac0c028422377ca10bc",
        "accounting_sha256": "784e872caf436d98ac7c30058db76a0aa1eb9aec267e7bb0a7e7e6529d3c5414"},
    "trainer24": {"directory": "extension/report/checkpointed-head-trainer-qa-20261021-1229383", "job_id": "1229383",
        "name": "sft-extension-checkpointed-head-trainer-qa", "tests": 24, "row_elapsed": [40, 40, 40],
        "qa_schema": "extension-checkpointed-head-trainer-native-qa-v1", "complete_schema": "extension-checkpointed-head-trainer-qa-complete-v1",
        "native_qa_sha256": "5932df2f7e6519eda903ea854b26de7a2dd563243980ceac46b749f7f6ab68c1",
        "complete_sha256": "298a80f83949cbd3e575d9a4406479ac60b43684c9386999f9616246e50f267d",
        "accounting_sha256": "c184cbaf0511f388f7f61951b5d6dcd026b18403dd6bd9bbd1522fc67f45e0c9"},
    "comparison32": {"directory": "extension/report/native-tensor-comparison-qa-20261021-1229923", "job_id": "1229923",
        "name": "sft-extension-native-tensor-comparison-qa", "tests": 32, "row_elapsed": [16, 16, 17],
        "qa_schema": "extension-native-tensor-comparison-native-qa-v1", "complete_schema": "extension-native-tensor-comparison-qa-complete-v1",
        "native_qa_sha256": "c909b3ccd8c423982fd711baf2131b1872f6f3f7920b7ecd700432b488a67e12",
        "complete_sha256": "fb559baac116a62ae9588523f1e8cefd8b93879c366e443fd5ad53790a48b755",
        "accounting_sha256": "ac34db9eed874059eadcc62c3ead59e4644356878799e8c5fcebe6d949c545f7"}}
RUNTIME_GETTERS = {"float32_matmul_precision": "get_float32_matmul_precision",
    "cuda_matmul_allow_tf32": "backends.cuda.matmul.allow_tf32",
    "cuda_matmul_allow_fp16_reduced_precision_reduction": "backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction": "backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
    "cudnn_allow_tf32": "backends.cudnn.allow_tf32",
    "flash_sdp_enabled": "backends.cuda.flash_sdp_enabled",
    "mem_efficient_sdp_enabled": "backends.cuda.mem_efficient_sdp_enabled",
    "math_sdp_enabled": "backends.cuda.math_sdp_enabled",
    "cudnn_sdp_enabled": "backends.cuda.cudnn_sdp_enabled"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def inside(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Expected an explicit project-relative metadata path")
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Metadata path escapes the project")
    return path


def record(root, name):
    path = inside(root, name)
    before = path.stat()
    require(path.is_file() and 0 < before.st_size <= 2 << 20, "Expected bounded source/receipt metadata")
    payload = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Input changed while reading")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def read_json(root, name, expected=None):
    before = record(root, name)
    require(expected is None or before["sha256"] == expected, "Pinned metadata SHA differs")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON metadata key")
            result[key] = value
        return result
    value = json.loads(inside(root, name).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON metadata")))
    require(record(root, name) == before, "Metadata changed during parsing")
    return value


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_inventory(root):
    require(all(type(sha) is str and re.fullmatch(r"[0-9a-f]{64}", sha) for sha in PINS.values()),
        "Root must freeze every new helper/test/protocol pin before admission")
    result = {name: record(root, name) for name in sorted(SOURCES)}
    require(all(result[name]["sha256"] == sha for name, sha in PINS.items()), "Reviewed helper/test bytes changed")
    return result


def validate_plan(value):
    require(len(SOURCES) == 11 and len(PINS) == 8 and sum(PATTERNS.values()) == EXPECTED_TESTS and
        type(value) is dict and value.get("schema") == "extension-head-native-tools-qa-plan-v1" and
        value.get("role") == ROLE and value.get("test_patterns") == PATTERNS and
        value.get("expected_tests") == EXPECTED_TESTS and value.get("source_files") == sorted(SOURCES) and
        value.get("reviewed_input_sha256") == PINS and value.get("runtime_versions") == VERSIONS and
        value.get("resources") == {"cpus": 2, "ram_gib": 8, "gpus": 0, "wall_time_seconds": WALL_SECONDS,
                                  "reserve_seconds": RESERVE_SECONDS} and
        value.get("deadline_utc") == DEADLINE.isoformat() and value.get("claims") == CLAIMS and
        value.get("native_qa_schema") == SCHEMA and
        value.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True} and
        value.get("protocol_sha256") == PROTOCOL_PINS and value.get("historical_native_qa") == HISTORICAL and
        value.get("runtime_getters") == RUNTIME_GETTERS and
        value.get("numerical_assertions_owner") == "Exact prospective assertions in the pinned24+24+32 test modules and frozen protocols; no stage tolerance override",
        "Only the reviewed synthetic-CPU-tools80/source11 scope is permitted")
    prior, historical = value.get("prior_ledger", {}), value.get("historical_ledger", {})
    require(type(prior) is dict and set(prior) == {"path", "sha256", "entries", "entries_sha256"} and
        type(prior["path"]) is str and prior["path"].startswith("extension/report/") and
        type(prior["entries"]) is int and prior["entries"] == 32 and
        all(type(prior[key]) is str and re.fullmatch(r"[0-9a-f]{64}", prior[key]) for key in ("sha256", "entries_sha256")) and
        historical == {"path": "deadline/job_ledger.json", "sha256": "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d"},
        "Root must seal the exact prior32 history before admission")
    require(not Path(prior["path"]).is_absolute() and ".." not in Path(prior["path"]).parts,
        "Prior snapshot must be explicit public project metadata")
    parent = value.get("parent_sources", {})
    require(type(parent) is dict and set(parent) == {"path", "sha256", "schema", "inventory_key", "expected_files"} and
        type(parent["path"]) is str and parent["path"].startswith("extension/report/") and
        not Path(parent["path"]).is_absolute() and ".." not in Path(parent["path"]).parts and
        type(parent["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", parent["sha256"]) is not None and
        type(parent["schema"]) is str and parent["schema"].startswith("extension-") and len(parent["schema"]) <= 128 and
        parent["inventory_key"] == "protected_source_inventory" and type(parent["expected_files"]) is int and parent["expected_files"] == 176,
        "Root must seal the exact terminal public protected176 parent manifest before admission")
    return value


def deadline_guard():
    now = datetime.now(timezone.utc)
    require((DEADLINE - now).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
        "Keep the complete CPU30-minute stage and48-hour closeout reserve")
    return {"checked_at_utc": now.isoformat(), "guard_seconds": WALL_SECONDS + RESERVE_SECONDS,
            "deadline_utc": DEADLINE.isoformat()}


def require_cpu_allocation():
    """Verify this running zero-GPU allocation before torch/test imports."""
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) is not None and node and
        socket.gethostname().split(".")[0] == node.split(".")[0] and
        node.split(".")[0].lower() not in {"owl1", "tc2"} and "login" not in node.lower(),
        "A real assigned CPU compute process is required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key, "") in ("", "0", "(null)", "N/A")
        for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")), "GPU visibility/request forbidden")
    command = ["scontrol", "--clusters=owl", "show", "job", "--oneliner", job]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    require(not result.stderr.strip() and len(result.stdout.encode()) <= 128 << 10, "Incomplete allocation proof")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result.stdout))
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    tres = fields.get("AllocTRES", "")
    memory = re.search(r"(?:^|,)mem=([0-9]+)([KMGT])(?:,|$)", tres)
    mem_kib = None if memory is None else int(memory.group(1)) * {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}[memory.group(2)]
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING" and user is not None and
        int(user.group(1)) == os.getuid() and fields.get("NumNodes") == "1" and fields.get("NodeList") == node and
        fields.get("NumCPUs") == "2" and re.search(r"(?:^|,)cpu=2(?:,|$)", tres) is not None and mem_kib == 8 * 1024**2 and
        fields.get("Partition") == "normal_q" and fields.get("Account") == "tml_2026" and
        fields.get("QOS") == "owl_normal_short" and fields.get("TimeLimit") == "00:30:00" and
        all("gpu" not in value.lower() for key, value in fields.items() if "tres" in key.lower() or "gres" in key.lower()),
        "Exact CPU2/8GiB/30-minute single-node zero-GPU allocation required")
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
    require(set(found) == set(expected) | {"output", "error"} and all(found[name] == value for name, value in expected.items()),
        "Only the exact finite CPU wrapper is permitted")
    for name in ("output", "error"):
        path = Path(found[name])
        require(path.is_absolute() and ".." not in path.parts and path.parent == root / "extension/logs", "Bounded extension logs required")
    return found


def historical_qa_evidence(root, protected):
    results, files = {}, {}
    for label, spec in HISTORICAL.items():
        directory = spec["directory"]
        values = {}
        for name, pin in (("native_qa.json", "native_qa_sha256"), ("complete.json", "complete_sha256"),
                          ("accounting.json", "accounting_sha256")):
            path = directory + "/" + name
            values[name] = read_json(root, path, spec[pin])
            files[path] = record(root, path)
        qa, complete, accounting = (values[name] for name in ("native_qa.json", "complete.json", "accounting.json"))
        require(qa.get("schema") == spec["qa_schema"] and qa.get("success") is True and qa.get("job_id") == spec["job_id"] and
            qa.get("tests_run") == spec["tests"] and qa.get("gpu_disabled") is True and qa.get("input_snapshot_unchanged") is True and
            all(type(qa.get(key)) is int and qa[key] == 0 for key in
                ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Historical native QA did not pass its exact scope")
        sources = qa.get("source_inventory", {})
        require(type(sources) is dict and len(sources) == 5 and all(protected.get(name) == item for name, item in sources.items()) and
            qa.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
            qa.get("plan_sha256") in {item["sha256"] for name, item in sources.items() if name.endswith("_plan.json")},
            "Historical passed sources/plan are not the unchanged protected parent bytes")
        require(complete.get("schema") == spec["complete_schema"] and complete.get("complete") is True and
            complete.get("job_id") == spec["job_id"] and complete.get("tests_run") == spec["tests"] and
            complete.get("cpu_only") is True and complete.get("gpu_seconds") == 0 and complete.get("input_snapshot_unchanged") is True and
            complete.get("source_inventory") == sources and complete.get("plan_sha256") == qa["plan_sha256"] and
            complete.get("native_qa_sha256") == spec["native_qa_sha256"], "Historical completion/QA/source bindings differ")
        command = ["sacct", "--clusters=owl", "-j", spec["job_id"], "--parsable2", "--noheader",
            "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,AllocTRES"]
        require(accounting.get("schema") == "extension-terminal-cpu-accounting-v1" and accounting.get("complete") is True and
            accounting.get("job_id") == spec["job_id"] and accounting.get("state") == "COMPLETED" and accounting.get("exit_code") == "0:0" and
            accounting.get("elapsed_seconds") == spec["row_elapsed"][0] and accounting.get("allocated_gpus") == 0 and
            accounting.get("gpu_seconds") == 0 and accounting.get("no_gpu_in_alloc_tres") is True and accounting.get("timezone") == "UTC" and
            accounting.get("command") == command and accounting.get("stderr") == "" and type(accounting.get("stdout")) is str,
            "Historical exact terminal CPU accounting differs")
        rows = [line.split("|") for line in accounting["stdout"].splitlines()]
        require(len(rows) == 3 and all(len(row) == 8 for row in rows), "Historical allocation/batch/extern accounting required")
        for index, (suffix, name) in enumerate((("", spec["name"]), (".batch", "batch"), (".extern", "extern"))):
            row = rows[index]
            require(row[:5] == [spec["job_id"] + suffix, name, "COMPLETED", "0:0", str(spec["row_elapsed"][index])] and
                "gpu" not in row[7].lower() and re.search(r"(?:^|,)cpu=2(?:,|$)", row[7]) is not None and
                re.search(r"(?:^|,)mem=8G(?:,|$)", row[7]) is not None and re.search(r"(?:^|,)node=1(?:,|$)", row[7]) is not None and
                (datetime.fromisoformat(row[6]) - datetime.fromisoformat(row[5])).total_seconds() == spec["row_elapsed"][index],
                "Historical exact raw terminal step/name/elapsed/TRES differs")
        results[label] = {"job_id": spec["job_id"], "tests_previously_passed": spec["tests"], "tests_replayed": False,
            "native_qa_sha256": spec["native_qa_sha256"], "complete_sha256": spec["complete_sha256"],
            "accounting_sha256": spec["accounting_sha256"], "actual_allocation_batch_extern_seconds": spec["row_elapsed"]}
    return {"scopes": results, "input_files": files}


def runtime_metadata(torch):
    require(not torch.cuda.is_initialized(), "Runtime metadata getters must not initialize CUDA")
    values = {}
    for key, path in RUNTIME_GETTERS.items():
        target = torch
        for part in path.split("."):
            require(hasattr(target, part), "Pinned pure runtime getter is unavailable: " + path)
            target = getattr(target, part)
        value = target() if callable(target) else target
        require((key == "float32_matmul_precision" and type(value) is str and value in {"highest", "high", "medium"}) or
            (key != "float32_matmul_precision" and type(value) is bool), "Unexpected runtime getter metadata type/value")
        values[key] = value
    require(not torch.cuda.is_initialized(), "Runtime metadata getter unexpectedly initialized CUDA")
    return values


def load_context(root, plan_path, allocation):
    sources = source_inventory(root)
    require(plan_path == PLAN, "The exact reviewed plan path is required")
    plan = validate_plan(read_json(root, plan_path))
    pin = plan["prior_ledger"]
    prior = read_json(root, pin["path"], pin["sha256"])
    require(prior.get("schema") == LEDGER_SCHEMA and prior.get("gpu_hour_cap", "missing") is None and
        prior.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(prior.get("jobs")) is dict and
        len(prior["jobs"]) == pin["entries"] and value_digest(prior["jobs"]) == pin["entries_sha256"] and ROLE not in prior["jobs"],
        "Exact sealed prior history required")
    require(all(type(item) is dict for item in prior["jobs"].values()), "Malformed sealed prior admission")
    parent_pin = plan["parent_sources"]
    parent = read_json(root, parent_pin["path"], parent_pin["sha256"])
    protected = parent.get(parent_pin["inventory_key"], {})
    require(parent.get("schema") == parent_pin["schema"] and parent.get("raw_private_records_included") is False and
        type(protected) is dict and len(protected) == 176 and not set(protected) & SOURCES,
        "Exact terminal protected176 parent source inventory required")
    parent_records = {}
    for name, item in protected.items():
        require(type(item) is dict and set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and 0 < item["bytes"] <= 1 << 20 and
            type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
            "Malformed protected-parent source record")
        parent_records[name] = record(root, name)
        require(parent_records[name] == item, "Protected source changed before fixture imports: " + name)
    historical = record(root, plan["historical_ledger"]["path"])
    require(historical["sha256"] == plan["historical_ledger"]["sha256"], "Historical completed-experiment ledger changed")
    verified_scopes = historical_qa_evidence(root, parent_records)
    bootstrap = record(root, BOOTSTRAP)
    ledger = read_json(root, "extension/job_ledger.json")
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) == set(prior["jobs"]) | {ROLE} and
        all(ledger["jobs"].get(name) == value for name, value in prior["jobs"].items()), "Only this new CPU role may extend sealed history")
    entry = ledger["jobs"][ROLE]
    require(type(entry) is dict and type(entry.get("binding")) is dict, "Current CPU admission binding missing")
    binding = entry["binding"]
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    require(entry.get("status") == "submitted" and entry.get("job_id") == allocation["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and entry.get("source_sha256") == sources and
        entry.get("command") == command == binding.get("command") and entry.get("binding_sha256") == value_digest(binding) and
        binding.get("role") == ROLE and binding.get("source_sha256") == sources and binding.get("plan_sha256") == sources[PLAN]["sha256"] and
        binding.get("prior_entries_sha256") == pin["entries_sha256"] and binding.get("expected_native_tests") == EXPECTED_TESTS and
        binding.get("wall_time_seconds") == WALL_SECONDS and binding.get("closeout_reserve_seconds") == RESERVE_SECONDS and
        binding.get("requested_gpu_seconds") == 0 and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP) == bootstrap and
        binding.get("parent_source_manifest_sha256") == parent_pin["sha256"] and
        binding.get("protocol_sha256") == PROTOCOL_PINS and
        binding.get("historical_qa_evidence_sha256") == {name: item["sha256"] for name, item in verified_scopes["input_files"].items()} and
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False,
        "Exact current-job CPU80/source11/protected176/protocol/plan/history/bootstrap admission required")
    require(not any(allocation["job_id"] == item.get(key) for item in prior["jobs"].values()
        for key in ("job_id", "possible_job_id")), "Current CPU scheduler identity collides with prior history")
    validate_wrapper(root)
    return {"sources": sources, "plan": plan, "ledger": ledger, "admission": entry, "bootstrap": bootstrap,
        "prior_snapshot": record(root, pin["path"]), "historical_ledger": historical,
        "parent_manifest": record(root, parent_pin["path"]), "protected_source_inventory": parent_records,
        "historical_qa_verified_only": verified_scopes}


def run(root, plan_path, output):
    root = Path(root).resolve()
    require(root == Path(__file__).resolve().parents[1], "Run only the installed source project")
    allocation = require_cpu_allocation()
    guard = deadline_guard()
    context = load_context(root, plan_path, allocation)
    output = Path(output)
    require(not output.is_symlink(), "No output alias")
    output = output.resolve()
    require(output.parent == root / "extension/runs" and output.name == "head-native-tools-qa-" + allocation["job_id"] and
        not output.exists(), "Exclusive current-job output required; no resume")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "Pinned native fixture versions differ")
    subprocess.run(["bash", "-n", str(root / SCRIPT)], check=True)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    base = {"schema": SCHEMA, "job_id": allocation["job_id"], "node": allocation["node"], "compute_proof": allocation,
        "started_at_utc": started, "source_inventory": context["sources"],
        "code_sha256": {name: item["sha256"] for name, item in context["sources"].items()},
        "plan_sha256": context["sources"][PLAN]["sha256"], "prior_entries_sha256": context["plan"]["prior_ledger"]["entries_sha256"],
        "prior_snapshot": context["prior_snapshot"], "historical_ledger": context["historical_ledger"],
        "parent_source_manifest": context["parent_manifest"], "protected_source_inventory": context["protected_source_inventory"],
        "protocol_sha256": PROTOCOL_PINS, "historical_qa_verified_only": context["historical_qa_verified_only"],
        "admission_binding_sha256": context["admission"]["binding_sha256"], "cpu_bootstrap_sha256": context["bootstrap"]["sha256"],
        "versions": {"python": sys.version, **versions}, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "test_patterns": PATTERNS, "expected_tests": EXPECTED_TESTS, "deadline_guard": guard, **CLAIMS}
    write_new_json(output / "started.json", base)
    metrics = {"tests_run": 0, "test_counts": {pattern: 0 for pattern in PATTERNS}, "failures": 0, "errors": 0,
               "skipped": 0, "expected_failures": 0, "unexpected_successes": 0}
    begun = time.monotonic()
    try:
        # No torch, helper or test module is imported before all guards above.
        import torch
        require(str(torch.__version__).split("+")[0] == VERSIONS["torch"] and
            torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == "cpu" and
            not torch.cuda.is_initialized(), "Expected untouched FP32 CPU-only fixture runtime")
        before_runtime = runtime_metadata(torch)
        write_new_json(output / "runtime_before.json", {"schema": "extension-head-native-tools-runtime-getters-v1",
            "job_id": allocation["job_id"], "getters": RUNTIME_GETTERS, "values": before_runtime,
            "cuda_initialized": False, "kernel_or_native_bf16_admission_established": False})
        suite, counts = unittest.TestSuite(), {}
        for pattern, expected in PATTERNS.items():
            subset = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern, top_level_dir=str(root / "tests"))
            counts[pattern] = subset.countTestCases()
            require(counts[pattern] == expected, "Exact new authored fixture counts required; no older scope replay")
            suite.addTest(subset)
        require(sum(counts.values()) == EXPECTED_TESTS, "Exact80 combined fixture scope required")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        metrics = {"tests_run": result.testsRun, "test_counts": counts, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses)}
        after_runtime = runtime_metadata(torch)
        write_new_json(output / "runtime_after.json", {"schema": "extension-head-native-tools-runtime-getters-v1",
            "job_id": allocation["job_id"], "getters": RUNTIME_GETTERS, "values": after_runtime,
            "cuda_initialized": False, "unchanged_from_before": after_runtime == before_runtime,
            "kernel_or_native_bf16_admission_established": False})
        require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and all(metrics[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Synthetic CPU-tools80 fixtures did not all pass")
        require(torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == "cpu" and
            not torch.cuda.is_initialized(), "Fixture runtime mutated CPU/default-precision policy")
        require(after_runtime == before_runtime, "Fixtures mutated pinned runtime precision/backend defaults")
        require(load_context(root, plan_path, allocation) == context, "Sources, prior/current/historical ledger or bootstrap changed during QA")
        ending = require_cpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "CPU allocation changed")
        receipt = {**base, **metrics, "success": True, "compute_end": ending,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_elapsed_seconds": time.monotonic() - begun,
            "runtime_getters_before": before_runtime, "runtime_getters_after": after_runtime, "runtime_getters_unchanged": True,
            "runtime_before_sha256": record(root, str((output / "runtime_before.json").relative_to(root)))["sha256"],
            "runtime_after_sha256": record(root, str((output / "runtime_after.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "started_sha256": record(root, str((output / "started.json").relative_to(root)))["sha256"]}
        write_new_json(output / "native_qa.json", receipt)
        write_new_json(output / "complete.json", {"schema": "extension-head-native-tools-qa-complete-v1", "complete": True,
            "job_id": allocation["job_id"], "tests_run": EXPECTED_TESTS, "gpu_seconds": 0, "cpu_only": True,
            "source_inventory": context["sources"], "plan_sha256": base["plan_sha256"],
            "parent_source_manifest": context["parent_manifest"], "protected_source_inventory": context["protected_source_inventory"],
            "protocol_sha256": PROTOCOL_PINS, "historical_qa_verified_only": context["historical_qa_verified_only"],
            "runtime_getters_before": before_runtime, "runtime_getters_after": after_runtime, "runtime_getters_unchanged": True,
            "native_qa_sha256": record(root, str((output / "native_qa.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "created_at_utc": datetime.now(timezone.utc).isoformat(), **CLAIMS})
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {**base, **metrics, "success": False,
            "error_type": type(error).__name__, "message": "CPU fixture stage failed; inspect its native log without retry",
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
    print(json.dumps({"head_native_tools_cpu_fixture_qa_complete": True, "tests_run": EXPECTED_TESTS,
                      "gpu_seconds": 0, "actual_model_or_memory_admission": False}), flush=True)


if __name__ == "__main__":
    main()
