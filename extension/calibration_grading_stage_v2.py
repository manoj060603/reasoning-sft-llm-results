"""UNADMITTED operational v2 after preserved permission-only failure1229195.

Standard-library metadata guards precede tests, private input reads and worker
imports. Exactly32 synthetic tests run first. Only their complete success permits
the separately pinned original32x8 calibration inputs and frozen v2 workers.
No model/tokenizer, generation, ranking, selection, training or retry is admitted.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import traceback
import unittest


ROLE = "calibration_grading_20261021_v2"
NAME = "sft-extension-calibration-grading-v2"
PLAN = "extension/calibration_grading_v2_plan.json"
SCRIPT = "extension/calibration_grading_v2.slurm"
BOOTSTRAP = "extension/operations/submit-calibration-grading-v2-20261021.py"
SCHEMA = "extension-calibration-grading-stage-v2"
QA_SCHEMA = "extension-calibration-grading-native-qa-v2"
LEDGER_SCHEMA = "reasoning-sft-extension-job-ledger-v1"
PATTERN, EXPECTED_TESTS = "test_calibration_grading.py", 32
WALL_SECONDS, RESERVE_SECONDS = 3600, 172800
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
OLD_SOURCES = {
    "extension/calibration_grading.py": {"sha256": "ec20d82bdc8ca7c4528253fe18d01ae35b9f0d74ae9bd347eb30e14bfe9012a4", "bytes": 23701},
    "tests/test_calibration_grading.py": {"sha256": "721c7eae4006429062e08e04335d8d1705693b89e07711671d47752719c46c66", "bytes": 31207},
    "extension/calibration_grading_stage.py": {"sha256": "c500f79eb46d6e703127f3dde4a2c70a4ad2fd2e73a1d32a81c08791ed96c0f7", "bytes": 32947},
    "extension/calibration_grading_plan.json": {"sha256": "cc4e77e18020bc7af07feb6fa323bdf5baec912d2dd5338326f36d0b6bcecd94", "bytes": 6651},
    "extension/calibration_grading.slurm": {"sha256": "79bd8a8f1ffd3e317a629bf016d8e0912e60e5686b8bcc5eae30cdb3de316e2e", "bytes": 3407}}
SOURCES = frozenset(OLD_SOURCES) | frozenset({PLAN, SCRIPT, "extension/calibration_grading_stage_v2.py"})
PINS = {name: item["sha256"] for name, item in OLD_SOURCES.items()}
FAILED_DIRECTORY = "extension/report/calibration-grading-failed-20261021-1229195"
FAILED_ROLE = "calibration_grading_20261021_v1"
FAILURE_PINS = {"failure.json": "208179f66368f159b7e331bf5f88fb8c7bc4a205a0e08576ce2dd5658bfeaeef",
    "native_qa.json": "554a1749893ed34f515935784f1047f1ccdaab8731077ff6af9572d39802112b",
    "accounting.json": "c2972809093004323ed6b9b8d05688f7e2ae63bc188cfb4af659d7afa08190c7"}
REPAIR = {"path": "extension/report/calibration-permissions-repair-20261021/repair.json",
    "sha256": "b11e0b7951b69c79cb4c875a98a5737c30286f7b28eddb95d919cdc32aed8055"}
HISTORY = {"path": FAILED_DIRECTORY + "/job_ledger_snapshot.json",
    "sha256": "e0dfe0d97245e8c49972a234c18ad988354824be20a3be4a3575ea41dc9ed1c4", "entries": 29,
    "entries_sha256": "791fa7c9715eff12d8d65b95cc5c855b1930b6f473c00fd06a88abc817c2453b"}
HISTORICAL = {"path": "deadline/job_ledger.json", "sha256": "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d"}
PARENT = {"path": FAILED_DIRECTORY + "/evidence_manifest.json",
    "sha256": "3e77bdb0d2bbfe641e0df0961a1a9d393b7e97e58748acd97e524af7eb69dd5f",
    "inventory_key": "protected_source_inventory", "expected_files": 160}
GENERATION_DIRECTORY = "extension/report/source-generation-v3-complete-20261021-1228872"
PUBLIC_PINS = {"complete.json": "cc46a0b2629c3c1f94f8018cefe73be359c6d91153c1cebf44e363dd5e0f6381",
    "accounting.json": "a5da4eb91d9717351b900fb59f460f93491fd25759fd71ab5b44cbc8c880e530",
    "private_hash_inventory.json": "3910bd706f9d353efb0226885b8ec282931405bad869dcd78c90abb1f299dfb3",
    "evidence_manifest.json": "0cd7ebd074aaca4035d27070cdd3db776cb8d5479078fcab3dbc299e5a6f1a5d",
    "provenance.json": "42f774923a210ec3ed6f368c6742972ef29a3a0dc3cfaa2715b730721c49cc92"}
RAW_DIRECTORY = "extension/runs/source-generation-original-v3-1228872"
RAW_COMPLETE_SHA256 = "2d2869b76b14af823c377d98ca524922b2ed1ed8d50e4282a4e826842de2c119"
CACHE = {"path": "extension/runs/source-readiness-v2-retry-1226240/calibration_prompt_cache.json",
    "sha256": "6b9e9ff88a23aa73b9ea34cb4618d5858d0a23ef606e2892639fb953bbfede65"}
PROTOCOL = {"path": "extension/planning/source_grader_v2_protocol.json",
    "sha256": "c2cd03ac8281116810da029eed76116b21dfe5a7d78772765d9773c2ff216e2a"}
VERSIONS = {"accelerate": "1.12.0", "antlr4-python3-runtime": "4.7.2", "latex2sympy2": "1.9.1",
    "regex": "2025.11.3", "safetensors": "0.7.0", "sympy": "1.14.0", "torch": "2.9.0",
    "transformers": "4.57.3", "vllm": "0.12.0", "word2number": "1.1"}
CLAIMS = {"calibration_engineering_only": True, "method_ranking_permitted": False,
    "accuracy_driven_retuning_permitted": False, "intermediate_reasoning_verified": False,
    "model_or_tokenizer_loaded": False, "generation_performed": False, "selection_performed": False,
    "training_performed": False, "benchmark_accuracy_established": False,
    "bf16_training_or_memory_admission_established": False, "automatic_retry": False, "automatic_downstream": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Explicit project-relative input required")
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Input alias escapes the project")
    return path


def decode(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate input metadata key")
            result[key] = value
        return result
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite input metadata")))


class Files:
    """Actual byte hashes and filesystem fingerprints, never caller SHA claims."""
    def __init__(self, root):
        self.root, self.records, self.fingerprints, self.limits, self.private = root, {}, {}, {}, set()

    def read(self, name, expected=None, *, maximum=2 << 20, private=False, expected_bytes=None, keep=True):
        path = inside(self.root, name)
        if private:
            for parent in (path, *path.parents):
                if parent == self.root: break
                require(not parent.is_symlink(), "Private input contains a symlink component")
            folder = path.parent if path.parent.name != "engine-config" else path.parent.parent
            directory = folder.stat()
            require(stat.S_ISDIR(directory.st_mode) and directory.st_uid == os.getuid() and
                stat.S_IMODE(directory.st_mode) & 0o077 == 0, "Private input folder must remain owner-only")
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum and
            (expected_bytes is None or before.st_size == expected_bytes), "Input length/type differs from its bound")
        if private:
            require(before.st_uid == os.getuid() and stat.S_IMODE(before.st_mode) & 0o077 == 0,
                "Private input file must remain owner-only")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        checksum, blocks, count = hashlib.sha256(), [], 0
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            require((opened.st_dev, opened.st_ino, opened.st_size) == (before.st_dev, before.st_ino, before.st_size), "Input changed before opening")
            for block in iter(lambda: stream.read(1 << 20), b""):
                count += len(block); require(count <= maximum, "Input grew past its bound")
                checksum.update(block)
                if keep: blocks.append(block)
        after = path.stat()
        fingerprint = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        require(fingerprint(before) == fingerprint(after) and count == before.st_size, "Input changed while reading")
        actual = {"sha256": checksum.hexdigest(), "bytes": count}
        require(expected is None or actual["sha256"] == expected, "Actual input FILE SHA differs")
        require(name not in self.records or self.records[name] == actual and self.fingerprints[name] == fingerprint(after), "An earlier bound input changed")
        self.records[name], self.fingerprints[name], self.limits[name] = actual, fingerprint(after), maximum
        if private: self.private.add(name)
        return b"".join(blocks) if keep else actual

    def json(self, name, expected=None):
        return decode(self.read(name, expected))

    def unchanged(self):
        for name, item in list(self.records.items()):
            self.read(name, item["sha256"], maximum=self.limits[name], private=name in self.private,
                expected_bytes=item["bytes"], keep=False)


def write_new(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def source_inventory(root):
    files = Files(root)
    for name in sorted(SOURCES): files.read(name, PINS.get(name), maximum=1 << 20, keep=False)
    require(len(files.records) == 8 and all(files.records.get(name) == item for name, item in OLD_SOURCES.items()),
        "All five failed-parent sources remain byte-identical inside the new eight-source inventory")
    return files.records


def validate_plan(value):
    require(type(value) is dict and value.get("schema") == "extension-calibration-grading-plan-v2" and value.get("role") == ROLE and
        value.get("source_files") == sorted(SOURCES) and value.get("reviewed_input_sha256") == PINS and
        value.get("native_qa") == {"schema": QA_SCHEMA, "test_pattern": PATTERN, "expected_tests": EXPECTED_TESTS,
            "same32_regression_replayed": True, "new_test_methods": 0, "parent_native_qa_sha256": FAILURE_PINS["native_qa.json"]} and
        value.get("resources") == {"cpus": 2, "ram_gib": 8, "gpus": 0, "wall_time_seconds": WALL_SECONDS, "reserve_seconds": RESERVE_SECONDS} and
        value.get("prior_ledger") == HISTORY and value.get("historical_ledger") == HISTORICAL and value.get("parent_sources") == PARENT and
        value.get("generation_public_evidence") == {"directory": GENERATION_DIRECTORY, "pins": PUBLIC_PINS} and
        value.get("private_inputs") == {"generation_directory": RAW_DIRECTORY, "raw_complete_sha256": RAW_COMPLETE_SHA256, "cache": CACHE} and
        value.get("worker_protocol") == PROTOCOL and value.get("runtime_versions") == VERSIONS and
        value.get("failed_parent") == {"directory": FAILED_DIRECTORY, "role": FAILED_ROLE, "job_id": "1229195", "pins": FAILURE_PINS} and
        value.get("permission_repair") == REPAIR and value.get("old_source_records") == OLD_SOURCES and
        value.get("deadline_utc") == DEADLINE.isoformat() and value.get("claims") == CLAIMS and
        value.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True},
        "Only the fixed CPU32-then-native32x8 calibration engineering stage is supported")
    return value


def failure_and_repair_evidence(files):
    """Only sealed public metadata: preserve failed7s/QA32 and mode-only repair."""
    values = {name: files.json(FAILED_DIRECTORY + "/" + name, sha) for name, sha in FAILURE_PINS.items()}
    manifest = files.json(PARENT["path"], PARENT["sha256"])
    failure, qa, account = values["failure.json"], values["native_qa.json"], values["accounting.json"]
    require(failure.get("schema") == "extension-calibration-grading-stage-v1" and failure.get("job_id") == "1229195" and
        failure.get("complete") is False and failure.get("native_qa_passed") is True and failure.get("error_type") == "ValueError" and
        failure.get("tests_run") == 32 and failure.get("source_inventory") == OLD_SOURCES and
        failure.get("automatic_retry") is False and failure.get("automatic_downstream") is False,
        "Failed parent remains failed after its passed32 regression scope")
    require(qa.get("schema") == "extension-calibration-grading-native-qa-v1" and qa.get("success") is True and qa.get("job_id") == "1229195" and
        qa.get("tests_run") == 32 and qa.get("test_counts") == {PATTERN: 32} and qa.get("source_inventory") == OLD_SOURCES and
        qa.get("code_sha256") == PINS and qa.get("private_inputs_opened") is False and qa.get("native_calibration_grading_performed") is False and
        qa.get("gpu_disabled") is True and qa.get("compute_proof", {}).get("job_id") == "1229195" and
        qa["compute_proof"].get("allocated_cpus") == 2 and qa["compute_proof"].get("allocated_ram_gib") == 8 and
        qa["compute_proof"].get("gpu_requested_or_allocated") is False and
        all(type(qa.get(key)) is int and qa[key] == 0 and failure.get(key) == 0
            for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")),
        "The exact first32 passed CPU fixtures are preserved, never relabeled as completed grading")
    require(account.get("schema") == "extension-terminal-cpu-accounting-v1" and account.get("complete") is True and
        account.get("job_id") == "1229195" and account.get("state") == "FAILED" and account.get("exit_code") == "1:0" and
        account.get("elapsed_seconds") == 7 and account.get("allocated_gpus") == account.get("gpu_seconds") == 0 and
        account.get("no_gpu_in_alloc_tres") is True and account.get("stderr") == "" and account.get("timezone") == "UTC",
        "Exact failed7-second zero-GPU parent accounting required")
    rows = [line.split("|") for line in account.get("stdout", "").splitlines() if line]
    states = {"1229195": ["FAILED", "1:0"], "1229195.batch": ["FAILED", "1:0"], "1229195.extern": ["COMPLETED", "0:0"]}
    require(len(rows) == 3 and {row[0] for row in rows} == set(states) and all(len(row) == 8 and row[2:4] == states[row[0]] and row[4] == "7" and
        "cpu=2" in row[7].split(",") and "mem=8G" in row[7].split(",") and "gpu" not in row[7].lower() for row in rows),
        "Preserve allocation/batch FAILED and extern COMPLETED exactly")
    for name, sha in FAILURE_PINS.items():
        require(manifest.get("files", {}).get(name, {}).get("sha256") == sha, "Failed parent manifest hashes differ")
    repair = files.json(REPAIR["path"], REPAIR["sha256"])
    require(repair.get("schema") == "extension-calibration-private-permissions-repair-v1" and repair.get("complete") is True and
        repair.get("failed_job_id") == "1229195" and repair.get("cache_path") == CACHE["path"] and
        repair.get("folder_path") == str(Path(CACHE["path"]).parent) and repair.get("ledger_sha256") == HISTORY["sha256"] and
        repair.get("failed_public_manifest") == {"path": PARENT["path"], **files.records[PARENT["path"]]} and
        repair.get("protected_source_files") == 160 and
        all(repair.get(key) is True for key in ("input_bytes_unchanged", "only_permission_modes_changed", "extension_ledger_unchanged",
            "historical_ledger_unchanged", "protected_sources_unchanged")) and
        all(repair.get(key) is False for key in ("grading_performed", "model_tokenizer_or_generation_performed", "raw_private_records_included", "automatic_retry", "automatic_downstream")),
        "Repair must only tighten privacy modes without source/content/ledger or scientific changes")
    before, after = repair.get("before", {}), repair.get("after", {})
    require(before.get("cache_file") == after.get("cache_file") == {"bytes": 248071, "sha256": CACHE["sha256"]}, "Repair changed native cache bytes")
    for name, old, new in (("folder", "0o770", "0o700"), ("cache", "0o660", "0o600")):
        original, repaired = before.get(name, {}), after.get(name, {})
        require(type(original) is dict and type(repaired) is dict and set(original) == set(repaired) == {"device", "gid", "inode", "mode", "symlink", "uid"} and
            original.get("mode") == old and repaired.get("mode") == new and original.get("symlink") is repaired.get("symlink") is False and
            all(original.get(key) == repaired.get(key) for key in ("device", "inode", "uid", "gid")), "Only the explicit folder/cache modes may differ")
    return {"failure": failure, "native_qa": qa, "accounting": account, "repair": repair, "manifest": manifest}


def deadline_guard():
    now = datetime.now(timezone.utc)
    require((DEADLINE - now).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS, "Keep the complete CPU hour and48-hour reserve")
    return {"checked_at_utc": now.isoformat(), "guard_seconds": WALL_SECONDS + RESERVE_SECONDS, "deadline_utc": DEADLINE.isoformat()}


def require_cpu_allocation():
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) is not None and node and socket.gethostname().split(".")[0] == node.split(".")[0] and
        node.split(".")[0].lower() not in {"owl1", "tc2"} and "login" not in node.lower(), "Real current CPU compute allocation required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key, "") in ("", "0", "(null)", "N/A")
        for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")), "GPU environment forbidden")
    command = ["scontrol", "--clusters=owl", "show", "job", "--oneliner", job]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    require(not result.stderr.strip() and len(result.stdout.encode()) <= 128 << 10, "Incomplete allocation proof")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result.stdout))
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", "")); tres = fields.get("AllocTRES", "")
    memory = re.search(r"(?:^|,)mem=([0-9]+)([KMGT])(?:,|$)", tres)
    mem_kib = None if memory is None else int(memory.group(1)) * {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}[memory.group(2)]
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING" and user is not None and int(user.group(1)) == os.getuid() and
        fields.get("NumNodes") == "1" and fields.get("NodeList") == node and fields.get("NumCPUs") == "2" and
        re.search(r"(?:^|,)cpu=2(?:,|$)", tres) is not None and mem_kib == 8 * 1024**2 and fields.get("Partition") == "normal_q" and
        fields.get("Account") == "tml_2026" and fields.get("QOS") == "owl_normal_short" and fields.get("TimeLimit") == "01:00:00" and
        all("gpu" not in value.lower() for key, value in fields.items() if "tres" in key.lower() or "gres" in key.lower()),
        "Exact CPU2/8GiB/one-hour/zero-GPU allocation required")
    return {"job_id": job, "node": node, "hostname": socket.gethostname(), "user_id": os.getuid(), "state": "RUNNING",
        "allocated_cpus": 2, "allocated_ram_gib": 8, "allocated_tres": tres, "gpu_requested_or_allocated": False,
        "cuda_visible_devices": "", "step_id": os.environ.get("SLURM_STEP_ID", "batch"), "command": command,
        "scontrol": result.stdout.strip(), "scontrol_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def validate_wrapper(root):
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "normal_q", "qos": "owl_normal_short", "nodes": "1", "ntasks": "1",
        "cpus-per-task": "2", "mem": "8G", "time": "01:00:00", "no-requeue": ""}
    found, header = {}, True
    for line in inside(root, SCRIPT).read_text().splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"): header = False
            continue
        require(header, "Late scheduler directive")
        parts = shlex.split(line.lstrip()[7:].strip()); require(bool(parts), "Empty directive")
        option, separator, argument = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in set(expected) | {"output", "error"} and option[2:] not in found and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/duplicate/ambiguous directive")
        found[option[2:]] = argument if separator else parts[1] if len(parts) == 2 else ""
    require(set(found) == set(expected) | {"output", "error"} and all(found[key] == value for key, value in expected.items()), "Fixed finite CPU wrapper required")
    require(all(Path(found[key]).is_absolute() and ".." not in Path(found[key]).parts and Path(found[key]).parent == root / "extension/logs"
        for key in ("output", "error")), "Bounded extension logs required")
    return found


def public_generation_evidence(files):
    values = {name: files.json(GENERATION_DIRECTORY + "/" + name, sha) for name, sha in PUBLIC_PINS.items()}
    summary, account, inventory = values["complete.json"], values["accounting.json"], values["private_hash_inventory.json"]
    require(summary.get("complete") is True and summary.get("job_id") == "1228872" and summary.get("native_interface_established") is True and
        summary.get("fixture_factory_injected") is False and summary.get("model_role") == "original" and
        summary.get("request_count") == 32 and summary.get("sample_count") == 256 and summary.get("cache_sha256") == CACHE["sha256"] and
        summary.get("input_snapshot_unchanged") is True and summary.get("raw_generated_records_exported") is False and
        all(summary.get(key) is False for key in ("grading", "ranking", "training", "automatic_retry", "automatic_downstream")), "Terminal original generation proof differs")
    raw = summary.get("raw_native_complete", {})
    require(raw.get("path") == RAW_DIRECTORY + "/complete.json" and raw.get("sha256") == RAW_COMPLETE_SHA256 and
        type(raw.get("bytes")) is int and 0 < raw["bytes"] <= 2 << 20 and inventory.get("raw_complete") == raw,
        "Sanitized and raw native complete bytes must remain distinct and explicitly linked")
    require(account.get("schema") == "extension-terminal-gpu-accounting-v1" and account.get("complete") is True and
        account.get("job_id") == "1228872" and account.get("state") == "COMPLETED" and account.get("exit_code") == "0:0" and
        account.get("elapsed_seconds") == account.get("gpu_seconds") == 117 and account.get("allocated_gpus") == 1 and
        account.get("original_requested_gpu_seconds") == 7200 and account.get("stderr") == "" and account.get("timezone") == "UTC",
        "Exact completed117-second one-GPU terminal accounting required")
    rows = [line.split("|") for line in account.get("stdout", "").splitlines() if line]
    require(len(rows) == 3 and {row[0] for row in rows} == {"1228872", "1228872.batch", "1228872.extern"} and
        all(len(row) == 8 and row[2:5] == ["COMPLETED", "0:0", "117"] and "cpu=8" in row[7].split(",") and
            "mem=128G" in row[7].split(",") and "gres/gpu=1" in row[7].split(",") for row in rows), "Raw terminal allocation/step accounting differs")
    closed = inventory.get("closed_json_files", {})
    expected = {"record-" + str(index).zfill(2) + ".private.json" for index in range(32)} | {
        "engine-config/generation_config.json", "engine_outputs.private.json", "engine_proof.private.json", "requests.private.json", "started.json"}
    require(inventory.get("schema") == "extension-source-generation-terminal-private-hashes-v1" and set(closed) == expected and
        inventory.get("private_outputs_hashed_without_parsing") is True and inventory.get("worker_logs_sealed_only_after_verified_terminal_accounting") is True and
        inventory.get("generated_texts_token_records_request_ids_or_private_logs_exported") is False,
        "The authoritative37 closed JSON files and terminal-log hash receipt are required")
    for name, item in closed.items():
        require(type(item) is dict and set(item) == {"bytes", "sha256"} and type(item["bytes"]) is int and 0 < item["bytes"] <= 16 << 20 and
            type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "Malformed terminal private file hash")
    for name, item in PUBLIC_PINS.items():
        if name != "evidence_manifest.json":
            require(values["evidence_manifest.json"].get("files", {}).get(name, {}).get("sha256") == item, "Public generation evidence manifest differs")
    return values


def load_context(root, plan_path, allocation, files):
    require(plan_path == PLAN, "Exact grading stage plan required")
    sources = source_inventory(root)
    for name, item in sources.items(): files.read(name, item["sha256"], maximum=1 << 20, expected_bytes=item["bytes"], keep=False)
    plan = validate_plan(files.json(PLAN))
    parent = files.json(PARENT["path"], PARENT["sha256"])
    protected = parent.get(PARENT["inventory_key"], {})
    require(parent.get("schema") == "extension-calibration-grading-failure-evidence-v1" and parent.get("raw_private_records_included") is False and
        type(protected) is dict and len(protected) == 160 and set(protected) & SOURCES == set(OLD_SOURCES) and
        all(protected.get(name) == item for name, item in OLD_SOURCES.items()),
        "Exact unchanged protected160, including all five old source records, required")
    for name, item in protected.items():
        require(type(item) is dict and set(item) == {"sha256", "bytes"}, "Malformed protected source record")
        files.read(name, item["sha256"], maximum=1 << 20, expected_bytes=item["bytes"], keep=False)
    prior = files.json(HISTORY["path"], HISTORY["sha256"])
    require(prior.get("schema") == LEDGER_SCHEMA and prior.get("gpu_hour_cap", "missing") is None and
        prior.get("historical_ledger_unchanged") == HISTORICAL["path"] and type(prior.get("jobs")) is dict and
        len(prior["jobs"]) == 29 and value_digest(prior["jobs"]) == HISTORY["entries_sha256"] and ROLE not in prior["jobs"], "Exact prior29 ledger required")
    failed = failure_and_repair_evidence(files)
    old_entry = prior["jobs"].get(FAILED_ROLE, {})
    require(old_entry.get("job_id") == "1229195" and old_entry.get("source_sha256") == OLD_SOURCES and
        old_entry.get("binding", {}).get("plan_sha256") == OLD_SOURCES["extension/calibration_grading_plan.json"]["sha256"] and
        old_entry.get("binding_sha256") == failed["native_qa"].get("admission_binding_sha256") == failed["failure"].get("admission_binding_sha256"),
        "Keep the old failed role and its exact executed source/admission identity")
    files.read(HISTORICAL["path"], HISTORICAL["sha256"], keep=False)
    files.read(BOOTSTRAP, maximum=1 << 20, keep=False)
    ledger = files.json("extension/job_ledger.json")
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == HISTORICAL["path"] and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) == set(prior["jobs"]) | {ROLE} and all(ledger["jobs"].get(name) == value for name, value in prior["jobs"].items()),
        "Only this new CPU role may extend unchanged prior29")
    entry = ledger["jobs"][ROLE]; binding = entry.get("binding", {}) if type(entry) is dict else {}
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    require(entry.get("status") == "submitted" and entry.get("job_id") == allocation["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and entry.get("source_sha256") == sources and
        entry.get("command") == command == binding.get("command") and entry.get("binding_sha256") == value_digest(binding) and
        binding.get("role") == ROLE and binding.get("source_sha256") == sources and binding.get("plan_sha256") == sources[PLAN]["sha256"] and
        binding.get("prior_entries_sha256") == HISTORY["entries_sha256"] and binding.get("expected_native_tests") == EXPECTED_TESTS and
        binding.get("wall_time_seconds") == WALL_SECONDS and binding.get("closeout_reserve_seconds") == RESERVE_SECONDS and
        binding.get("requested_gpu_seconds") == 0 and binding.get("admission_code_sha256") == files.records[BOOTSTRAP]["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP) == files.records[BOOTSTRAP] and
        binding.get("failed_parent_manifest_sha256") == PARENT["sha256"] and binding.get("permission_repair_sha256") == REPAIR["sha256"] and
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False,
        "Current CPU32-then-grading admission/source/plan/history/bootstrap binding required")
    require(not any(allocation["job_id"] == item.get(key) for item in prior["jobs"].values() for key in ("job_id", "possible_job_id")), "CPU identity collides with prior history")
    generation = public_generation_evidence(files)
    protocol = files.json(PROTOCOL["path"], PROTOCOL["sha256"])
    require(protocol.get("historical_and_final_benchmark_grading") == "unchanged" and protocol.get("runtime_versions") == VERSIONS and
        all(protocol.get("worker_protocol", {}).get(key) == value for key, value in {"setup_seconds": 10, "work_seconds": 5, "cleanup_seconds": .55,
            "fresh_worker_per_math_response": True, "worker_reuse": False, "automatic_retry": False,
            "cleanup_error_blocks_further_grading_and_ranking": True}.items()), "Frozen v2 worker phase budgets/semantics differ")
    comparator = {"loaded_modules": {}, "source_sha256": {}, "runtime_versions": {key: VERSIONS[key]
        for key in ("sympy", "antlr4-python3-runtime", "latex2sympy2", "regex", "word2number")}}
    for filename, item in protocol["comparator_source"].items():
        name = "utils." + Path(filename).stem
        require(protected.get(item["path"], {}).get("sha256") == item["sha256"], "Comparator bytes not bound to protected sources")
        comparator["loaded_modules"][name] = str(inside(root, item["path"]))
        comparator["source_sha256"][name] = item["sha256"]
    validate_wrapper(root)
    return {"sources": sources, "protected": protected, "plan": plan, "ledger": ledger, "admission": entry,
        "generation": generation, "comparator": comparator, "failed_parent_and_repair": failed}


@contextmanager
def private_logs(output):
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(index) for index in (1, 2)]
    streams = [os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        for name in ("worker.stdout.private.log", "worker.stderr.private.log")]
    try:
        for target, stream in zip((1, 2), streams): os.dup2(stream, target)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        for target, descriptor in zip((1, 2), saved): os.dup2(descriptor, target); os.close(descriptor)
        for stream in streams: os.close(stream)


def private_inputs(files, evidence):
    """Called only AFTER native synthetic32 passed and was sealed."""
    inventory = evidence["private_hash_inventory.json"]
    for name, item in inventory["closed_json_files"].items():
        files.read(RAW_DIRECTORY + "/" + name, item["sha256"], maximum=16 << 20, private=True, expected_bytes=item["bytes"], keep=False)
    for name, item in inventory["terminal_worker_log_hashes"].items():
        files.read(RAW_DIRECTORY + "/" + name, item["sha256"], maximum=16 << 20, private=True, expected_bytes=item["bytes"], keep=False)
    for name, item in inventory["terminal_slurm_log_hashes"].items():
        files.read(name, item["sha256"], maximum=16 << 20, expected_bytes=item["bytes"], keep=False)
    cache = files.read(CACHE["path"], CACHE["sha256"], maximum=4 << 20, private=True)
    raw = inventory["raw_complete"]
    complete = files.read(raw["path"], RAW_COMPLETE_SHA256, private=True, expected_bytes=raw["bytes"])
    parsed = decode(complete)
    require(parsed.get("private_output_sha256") == {name: item["sha256"] for name, item in inventory["closed_json_files"].items()},
        "Raw native complete must match every terminally sealed closed JSON file")
    records = {"record-" + str(index).zfill(2) + ".private.json": files.read(RAW_DIRECTORY + "/record-" + str(index).zfill(2) + ".private.json",
        inventory["closed_json_files"]["record-" + str(index).zfill(2) + ".private.json"]["sha256"], maximum=8 << 20, private=True) for index in range(32)}
    files.unchanged()
    return cache, complete, records


def run(root, plan_path, output):
    root = Path(root).resolve(); require(root == Path(__file__).resolve().parents[1], "Installed source root required")
    allocation, guard, files = require_cpu_allocation(), deadline_guard(), Files(root)
    context = load_context(root, plan_path, allocation, files)
    output = Path(output); require(not output.is_symlink(), "No output alias")
    output = output.resolve()
    require(output.parent == root / "extension/runs" and output.name == "calibration-grading-v2-" + allocation["job_id"] and not output.exists(), "Exclusive current-job private output required; no resume")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}; require(versions == VERSIONS, "Pinned native runtime differs")
    subprocess.run(["bash", "-n", str(root / SCRIPT)], check=True)
    old_umask = os.umask(0o077); output.mkdir(mode=0o700, exist_ok=False)
    base = {"schema": SCHEMA, "job_id": allocation["job_id"], "node": allocation["node"], "compute_proof": allocation,
        "source_inventory": context["sources"], "protected_source_inventory": context["protected"],
        "plan_sha256": context["sources"][PLAN]["sha256"], "prior_entries_sha256": HISTORY["entries_sha256"],
        "admission_binding_sha256": context["admission"]["binding_sha256"], "cpu_bootstrap_sha256": files.records[BOOTSTRAP]["sha256"],
        "versions": {"python": sys.version, **versions}, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "failed_parent_job_id": "1229195", "failed_parent_manifest_sha256": PARENT["sha256"], "permission_repair_sha256": REPAIR["sha256"],
        "same32_regression_replayed": True, "new_test_methods": 0, "regression_parent_native_qa_sha256": FAILURE_PINS["native_qa.json"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "deadline_guard": guard, **CLAIMS}
    qa, result = None, None
    metrics = {"tests_run": 0, "test_counts": {PATTERN: 0}, "failures": 0, "errors": 0,
        "skipped": 0, "expected_failures": 0, "unexpected_successes": 0}
    try:
        write_new(output / "started.json", {**base, "private_inputs_opened": False, "expected_tests": EXPECTED_TESTS})
        suites = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=PATTERN, top_level_dir=str(root / "tests"))
        count = suites.countTestCases(); require(count == EXPECTED_TESTS, "Exactly32 synthetic fixtures required")
        begun = time.monotonic(); tested = unittest.TextTestRunner(verbosity=2).run(suites)
        metrics = {"tests_run": tested.testsRun, "test_counts": {PATTERN: count}, "failures": len(tested.failures), "errors": len(tested.errors),
            "skipped": len(tested.skipped), "expected_failures": len(tested.expectedFailures), "unexpected_successes": len(tested.unexpectedSuccesses)}
        require(tested.testsRun == EXPECTED_TESTS and tested.wasSuccessful() and all(metrics[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Synthetic grading fixtures did not all pass")
        files.unchanged()
        require(load_context(root, plan_path, allocation, Files(root)) == context, "Source/plan/history/current-admission changed during QA")
        qa = {**base, **metrics, "schema": QA_SCHEMA, "success": True, "private_inputs_opened": False,
            "native_calibration_grading_performed": False, "older_tests_replayed": True, "elapsed_seconds": time.monotonic() - begun,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": {name: item["sha256"] for name, item in context["sources"].items()}}
        write_new(output / "native_qa.json", qa)
        # Revalidate metadata/native QA BEFORE any real private byte or worker.
        qa_name = str((output / "native_qa.json").relative_to(root))
        require(files.json(qa_name) == qa, "Native QA receipt changed before private grading")
        deadline_guard(); require_cpu_allocation()
        with private_logs(output):
            from extension import calibration_grading as grading
            cache, complete, records = private_inputs(files, context["generation"])
            files.unchanged()
            require(load_context(root, plan_path, allocation, Files(root)) == context, "Admission changed before first native grade")
            result = grading.grade_calibration(cache, complete, records,
                expected_file_pins={"cache_json_sha256": CACHE["sha256"], "generation_complete_json_sha256": RAW_COMPLETE_SHA256},
                expected_comparator=context["comparator"])
            write_new(output / "grades.private.json", result["private"])
        files.unchanged()
        require(load_context(root, plan_path, allocation, Files(root)) == context, "Bound inputs/history/current-admission changed during grading")
        ending = require_cpu_allocation(); require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "CPU allocation changed")
        require(result["public"].get("complete") is True and result["public"].get("observed_grade_count") == 256 and
            all(row.get("receipt_validated") is True for row in result["private"]["observed_grades"]), "Complete validated32x8 grading required")
        write_new(output / "public_summary.json", {**result["public"], "stage_plan_sha256": base["plan_sha256"],
            "job_id": allocation["job_id"], "source_inventory": context["sources"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
        output_files = {name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), "bytes": (output / name).stat().st_size}
            for name in ("started.json", "native_qa.json", "grades.private.json", "public_summary.json")}
        complete_receipt = {**base, "complete": True, "native_qa_passed": True, "tests_run": EXPECTED_TESTS,
            "native_calibration_grading_performed": True, "record_count": 32, "sample_count": 256,
            "input_snapshot_unchanged": True, "compute_end": ending, "gpu_seconds": 0, "output_sha256": output_files,
            "input_file_sha256": files.records, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "private_worker_logs_unsealed_until_terminal_closeout": True}
        write_new(output / "complete.json", complete_receipt)
        return complete_receipt
    except BaseException as error:
        if hasattr(error, "result"):
            write_new(output / "grading-abort.private.json", error.result)
        elif result is not None and not (output / "grades.private.json").exists():
            write_new(output / "grades.partial.private.json", result["private"])
        write_new(output / "failure.private.json", {"error_type": type(error).__name__, "traceback": traceback.format_exc(),
            "native_qa_passed": qa is not None, "private_failure_evidence": True})
        write_new(output / "failure.json", {**base, **metrics, "complete": False, "native_qa_passed": qa is not None,
            "error_type": type(error).__name__, "message": "Calibration grading stage failed; preserve private evidence without retry",
            "private_worker_logs_unsealed_until_terminal_closeout": True, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        raise RuntimeError("Calibration grading failed. See private failure evidence; no retry.") from None
    finally:
        os.umask(old_umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", default="."); parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    try: receipt = run(args.root, args.plan, args.output)
    except BaseException: raise RuntimeError("Calibration grading stage failed. See private evidence.") from None
    print(json.dumps({"calibration_grading_complete": receipt["complete"], "tests_run": EXPECTED_TESTS,
        "record_count": 32, "sample_count": 256, "gpu_seconds": 0, "method_ranking_permitted": False}), flush=True)


if __name__ == "__main__":
    main()
