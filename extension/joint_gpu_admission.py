"""Fixed metadata-only admission of the separately tested joint-memory diagnostic.

This is not a training or experiment-selection entry point. It opens bounded
source/config and CPU-QA/accounting JSON only, never private witnesses, weights,
tokenizers, source questions, grader outputs or benchmark results. The scientific
79-file/410-check gate and this adapter's later native parser QA are distinct.
Any ambiguous/failed submission is preserved and requires manual reconciliation;
there is no automatic retry, downstream admission or resource override.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import os
from pathlib import Path
import re
import shlex
import subprocess

from extension import submission as metadata
from extension import joint_memory_qa as qa
from extension import source_readiness_retry_submission as previous
from extension.cpu_feasibility import validate_allocation, value_digest

ROLE = "joint_memory_admission_20261021_v1"
METADATA_ROLE = "joint_gpu_metadata_qa_20261021_v1"
NAME = "sft-extension-joint-memory-admission"
CPU_NAME = "sft-extension-joint-memory-qa"
CPU_JOB = "1225557"
SCRIPT = "extension/joint_memory_admission.slurm"
PLAN = qa.PLAN
PLAN_SHA256 = "b926be22779d59af501afddd10e7129f105b68b723b9bc8386be0aa397ddfc24"
WALL_SECONDS, RESERVE_SECONDS, MAX_GPU_ALLOCATIONS = 7200, 172800, 2
SCHEMA = "extension-joint-memory-admission-job-v1"
METADATA_QA_SCHEMA = "extension-joint-gpu-admission-native-qa-v1"
PARSER_SOURCES = frozenset({"extension/joint_gpu_admission.py", "tests/test_joint_gpu_admission.py"})
# Preserved, reconciled pre-scheduler DNS rejection. It is not an allocation
# identity and cannot be retried through this adapter. The whole immutable entry
# is also present in the separately hard-pinned scientific native-QA receipt.
RECONCILED_ROLE = "development_level_diagnostic_20261021_v1"
RECONCILED_ENTRY_SHA256 = "18a2fa351ad3e4ba1fe19af4999d2d3c4bd467211d33647fcc6f868cce1d0022"
QA_PINS = {
    "complete.json": "66a6ba25cea01a70ee3a945d1dce9cddf474915ab6b32e506de3058d23a8f51f",
    "unit_qa.json": "27d923622700acb2805f2fc15a924121f9bce74e061fda0f1a69ff9452a2eb3f",
    "native_qa.json": "00af45d78452c1d0a6d6aacf24518e9e77b749db39a2cbbd94b724a4f2cca4ee",
    "accounting.json": "58c20b9a3c6ab0484236eda3cdf7bd06fb0bd7acca65ee096016ce7b07ed9d52"}
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"})
NONTERMINAL = frozenset({"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING",
    "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING", "STAGE_OUT", "STOPPED", "SPECIAL_EXIT"})
SubmissionError, require = metadata.SubmissionError, metadata.require
_sha, _canonical, _read = metadata._sha, metadata._canonical, metadata._read


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _numeric_job(value):
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inside(root, proposed, *, runs=False, directory=False):
    root, proposed = Path(root).resolve(), Path(proposed)
    require(".." not in proposed.parts, "Parent traversal is not a reviewed metadata path")
    path = proposed if proposed.is_absolute() else root / proposed
    boundary = root / "extension/runs" if runs else root
    require(not path.is_symlink() and path.resolve().is_relative_to(boundary) and path.resolve() != boundary,
        "Metadata path escapes its fixed project/run namespace")
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        require(not cursor.is_symlink(), "Symlinked metadata paths are forbidden")
    require(path.is_dir() if directory else path.is_file(), "Missing expected metadata path")
    return path.resolve()


def _record(root, path):
    path = _inside(root, path)
    raw = metadata._small_bytes(path, Path(root).resolve())
    return {"sha256": _sha(raw), "bytes": len(raw)}


def gpu_count(value):
    """Parse Slurm %b, not AllocTRES. Unknown GRES must never count as zero.

    Accept only one whole GPU resource, optionally typed. IDX annotations and
    lists/mixed resources are deliberately unsupported and fail closed. The
    actual Owl forms ``gres/gpu:1`` and ``gpu:1`` are both supported.
    """
    require(isinstance(value, str), "Missing scheduler GRES string")
    if value in {"N/A", "(null)", "null", "None", "none", "0"}:
        return 0
    match = re.fullmatch(r"(?:gres/)?gpu:(?:[A-Za-z][A-Za-z0-9_.-]*:)?([0-9]+)", value)
    require(match is not None, "Unsupported/unknown scheduler GRES; do not infer zero GPUs")
    count = int(match.group(1))
    require(count <= 1024, "Implausible GPU count in scheduler metadata")
    return count


def parse_queue(stdout):
    require(isinstance(stdout, str) and len(stdout.encode()) <= 256 << 10, "Oversized/missing scheduler snapshot")
    jobs, seen = [], set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line == "CLUSTER: owl":
            continue
        parts = [part.strip() for part in line.split("|")]
        require(len(parts) == 4 and _numeric_job(parts[0]) and parts[0] not in seen and
            parts[1] in TERMINAL | NONTERMINAL and bool(parts[2]),
            "Ambiguous, duplicate, unknown-state or array scheduler row")
        require(len(parts[2]) <= 128 and len(parts[3]) <= 128, "Truncated/oversized scheduler identity")
        seen.add(parts[0])
        jobs.append({"job_id": parts[0], "state": parts[1], "name": parts[2], "gres": parts[3],
            "gpus": gpu_count(parts[3])})
    return jobs


def queue_snapshot(runner):
    user = getpass.getuser()
    require(isinstance(user, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", user), "Unknown scheduler user")
    command = ["squeue", "--clusters=owl", "--array", "--states=all", "--user=" + user,
        "--noheader", "--format=%64i|%64T|%128j|%128b"]
    try:
        response = runner(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise SubmissionError("Cannot establish a complete live scheduler snapshot") from error
    require(response.returncode == 0 and isinstance(response.stderr, str) and not response.stderr.strip(),
        "Scheduler snapshot failed or was incomplete")
    return {"command": command, "user": user, "stdout": response.stdout, "stderr": response.stderr,
        "stdout_sha256": _sha(response.stdout.encode()), "jobs": parse_queue(response.stdout),
        "observed_at_utc": datetime.now(timezone.utc).isoformat()}


def check_capacity(snapshot, *, already_submitted=False):
    """Pending requests count, as do unrelated user GPU allocations of any type."""
    allocated = sum(row["gpus"] > 0 and row["state"] in NONTERMINAL for row in snapshot["jobs"])
    require(allocated + (0 if already_submitted else 1) <= MAX_GPU_ALLOCATIONS,
        "At most two initial user GPU allocations, including pending requests, are permitted")
    return allocated


def _cpu_tres(value):
    require(isinstance(value, str) and value and "gpu" not in value.lower(), "Missing/non-CPU allocated TRES")
    fields = value.split(",")
    require(all(re.fullmatch(r"(?:billing|cpu|node)=[0-9]+|mem=[0-9]+(?:\.[0-9]+)?[KMGT]?", part) for part in fields),
        "Unknown accounting resources cannot establish a zero-GPU allocation")
    keys = [part.split("=", 1)[0] for part in fields]
    require(len(keys) == len(set(keys)) and "cpu" in keys and "node" in keys and
        int(dict(part.split("=", 1) for part in fields)["cpu"]) > 0 and
        dict(part.split("=", 1) for part in fields)["node"] == "1", "Incomplete single-node CPU accounting")


def validate_accounting(value, job, *, name=None):
    """Validate exact eight-column terminal sacct evidence, including all steps."""
    require(isinstance(value, dict) and value.get("schema") == "extension-terminal-cpu-accounting-v1" and
        value.get("complete") is True and value.get("job_id") == job and _numeric_job(job) and
        _integer(value.get("elapsed_seconds"), 1) and value["elapsed_seconds"] <= 7200 and
        type(value.get("gpu_seconds")) is int and value["gpu_seconds"] == 0 and
        value.get("no_gpu_in_alloc_tres") is True and value.get("timezone") == "UTC" and value.get("stderr") == "",
        "Terminal successful zero-GPU CPU accounting is required")
    expected = ["sacct", "--clusters=owl", "-j", job, "--parsable2", "--noheader",
        "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,AllocTRES"]
    require(value.get("command") == expected and isinstance(value.get("stdout"), str) and
        len(value["stdout"].encode()) <= 256 << 10, "Exact bounded terminal sacct command/output is required")
    if "state" in value:
        require(value["state"] == "COMPLETED", "Non-successful terminal state")
    if "exit_code" in value:
        require(value["exit_code"] == "0:0", "Nonzero CPU exit code")
    if "allocated_gpus" in value:
        require(type(value["allocated_gpus"]) is int and value["allocated_gpus"] == 0, "CPU accounting allocated GPUs")
    rows, ids = [], set()
    for line in value["stdout"].splitlines():
        if not line:
            continue
        row = line.split("|")
        require(len(row) == 8 and row[0] not in ids and
            (row[0] == job or re.fullmatch(re.escape(job) + r"\.(?:batch|extern|[0-9]+)", row[0])) and
            row[2:4] == ["COMPLETED", "0:0"] and re.fullmatch(r"[0-9]+", row[4]),
            "Every allocation/batch/extern/step must have complete successful CPU accounting")
        ids.add(row[0])
        require(int(row[4]) <= value["elapsed_seconds"], "CPU step elapsed time exceeds its allocation")
        try:
            start, end = (datetime.fromisoformat(item).replace(tzinfo=timezone.utc) for item in row[5:7])
        except (ValueError, TypeError) as error:
            raise SubmissionError("Unknown CPU accounting timestamps") from error
        require(0 <= (end - start).total_seconds() <= 7200 and
            abs((end - start).total_seconds() - int(row[4])) <= 1, "CPU elapsed/timestamp evidence disagrees")
        _cpu_tres(row[7])
        rows.append(row)
    require({job, job + ".batch", job + ".extern"} <= ids and
        next(row for row in rows if row[0] == job)[4] == str(value["elapsed_seconds"]),
        "Allocation, batch and extern accounting rows are all mandatory")
    allocation = next(row for row in rows if row[0] == job)
    require(bool(allocation[1]) and (name is None or allocation[1] == name), "Unexpected CPU allocation name")
    require(next(row for row in rows if row[0] == job + ".batch")[1] == "batch" and
        next(row for row in rows if row[0] == job + ".extern")[1] == "extern", "CPU step identities changed")
    return allocation[1]


def _validate_cpu_proof(value):
    proof = value.get("compute_proof", {})
    require(_numeric_job(value.get("job_id")) and isinstance(value.get("node"), str) and
        isinstance(proof, dict) and proof.get("job_id") == value["job_id"] and proof.get("node") == value["node"] and
        _integer(proof.get("user_id")) and proof["user_id"] == os.getuid() and
        isinstance(proof.get("scontrol"), str) and _checksum(proof.get("scontrol_sha256")),
        "Actual same-owner/same-job CPU compute proof is required")
    short = value["node"].split(".", 1)[0].lower()
    require(short not in {"owl1", "tc2", "tc1"} and "login" not in short, "QA was not on a compute host")
    text = proof["scontrol"]
    # The preserved CPU helper hashes subprocess stdout before stripping its
    # terminal newline for storage. Replay precisely either recorded form.
    candidates = [raw for raw in (text, text + "\n") if _sha(raw.encode()) == proof["scontrol_sha256"]]
    require(len(candidates) == 1, "CPU control-text hash changed")
    text = candidates[0]
    try:
        checked = validate_allocation(text, value["job_id"], value["node"], os.getuid())
    except (ValueError, RuntimeError) as error:
        raise SubmissionError("Invalid original CPU allocation proof") from error
    require(all(proof.get(key) == val for key, val in checked.items()), "CPU allocation proof contradicts raw scontrol")


def _passed_qa(value, schema):
    require(isinstance(value, dict) and value.get("schema") == schema and value.get("success") is True and
        value.get("gpu_disabled") is True and value.get("wrapper_syntax_passed") is True and
        _integer(value.get("tests_run"), 1) and all(type(value.get(key)) is int and value[key] == 0
        for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")),
        "Successful, unskipped native CPU QA is required")
    _validate_cpu_proof(value)


def _entry(value, role, *, gpu_seconds, sources=None, job=None):
    binding = value.get("binding", {}) if isinstance(value, dict) else {}
    require(isinstance(value, dict) and value.get("status") == "submitted" and _numeric_job(value.get("job_id")) and
        (job is None or value["job_id"] == job) and value.get("requested_gpu_seconds") == gpu_seconds and
        value.get("requested_gpu_hours") == gpu_seconds / 3600 and isinstance(binding, dict) and
        binding.get("role") == role and binding.get("automatic_downstream") is False and
        value.get("binding_sha256") == _sha(_canonical(binding)), "Prior/current CPU admission is uncertain or changed")
    if sources is not None:
        require(value.get("source_sha256") == sources == binding.get("source_sha256"), "CPU admission source bytes differ")
    return binding


def _scientific(root, qa_run, ledger):
    directory = _inside(root, qa_run, runs=True, directory=True)
    require(directory.name == "joint-memory-qa-" + CPU_JOB and directory.parent == root / "extension/runs",
        "Only the actual joint CPU QA run is supported")
    assets, values = {}, {}
    for name, checksum in QA_PINS.items():
        path = directory / name
        assets[str(path.relative_to(root))] = _record(root, path)
        require(assets[str(path.relative_to(root))]["sha256"] == checksum, "Pinned actual 410-check receipt changed: " + name)
        values[name] = _read(path)
    sources = qa.source_inventory(root)
    native, unit, complete = (values[name] for name in ("native_qa.json", "unit_qa.json", "complete.json"))
    codes = {name: record["sha256"] for name, record in sources.items()}
    for value, schema in ((native, "extension-joint-memory-admission-native-qa-v1"), (unit, qa.SCHEMA)):
        _passed_qa(value, schema)
        require(value["job_id"] == CPU_JOB and value["tests_run"] == 410 and value.get("source_inventory") == sources and
            value.get("code_sha256") == codes and value.get("plan_sha256") == PLAN_SHA256 and
            value.get("test_counts") == dict(zip(qa.PATTERNS, qa.COUNTS)) and value.get("automatic_downstream") is False and
            value.get("source_grader_executed") is False and value.get("source_grader_admitted") is False and
            value.get("full_source_readiness_465_scope_replayed") is False,
            "The complete scientific 79-source/410-check scope must remain exact")
    require(native["node"] == unit["node"] and native["compute_proof"] == unit["compute_proof"] and
        native.get("unit_qa_sha256") == QA_PINS["unit_qa.json"] and
        native.get("prepared_sha256") == _read(root / PLAN)["inputs"]["prepared"]["sha256"],
        "Native/scoped QA receipt identity changed")
    require(complete.get("schema") == "extension-joint-memory-qa-complete-v1" and complete.get("complete") is True and
        complete.get("cpu_only") is True and complete.get("job_id") == CPU_JOB and complete.get("gpu_seconds") == 0 and
        complete.get("tests_run") == 410 and complete.get("plan_sha256") == PLAN_SHA256 and
        complete.get("native_qa_sha256") == QA_PINS["native_qa.json"] and
        complete.get("unit_qa_sha256") == QA_PINS["unit_qa.json"] and complete.get("source_inventory") == sources and
        complete.get("input_snapshot_unchanged") is True and complete.get("automatic_downstream") is False,
        "Scientific QA did not complete unchanged")
    validate_accounting(values["accounting.json"], CPU_JOB, name=CPU_NAME)
    prior = native.get("prior_admissions", {})
    require(isinstance(prior, dict) and len(prior) == 11 and prior == unit.get("prior_admissions") and
        qa.ROLE not in prior and ROLE not in prior and METADATA_ROLE not in prior and
        all(ledger["jobs"].get(name) == entry for name, entry in prior.items()), "Preserve the exact eleven earlier admissions")
    cpu = ledger["jobs"].get(qa.ROLE, {})
    binding = _entry(cpu, qa.ROLE, gpu_seconds=0, sources=sources, job=CPU_JOB)
    require(binding.get("plan_sha256") == PLAN_SHA256 and binding.get("wall_time_seconds") == 7200 and
        binding.get("closeout_reserve_seconds") == RESERVE_SECONDS and
        binding.get("prior_binding_sha256") == {name: entry["binding_sha256"] for name, entry in prior.items()} and
        cpu["binding_sha256"] == native.get("admission_binding_sha256") == unit.get("admission_binding_sha256"),
        "Native QA is not bound to the actual completed CPU admission")
    return sources, {**prior, qa.ROLE: cpu}, assets, directory


def _parser_inventory(root, inventory):
    require(isinstance(inventory, dict) and PARSER_SOURCES <= set(inventory) and len(inventory) <= 100,
        "Parser QA must bind its actual source and test")
    actual, total = {}, 0
    forbidden = {"runs", "report", "artifacts", "logs", "hf", "results", "models", "weights"}
    for name, record in sorted(inventory.items()):
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts and
            not (set(Path(name).parts) & forbidden) and
            (name.startswith("extension/") and Path(name).suffix in {".py", ".slurm", ".json"} or
             name.startswith("tests/") and Path(name).suffix == ".py" or name == "reasoning/data.py"),
            "QA inventory may contain bounded source/config/tests only")
        require(isinstance(record, dict) and set(record) == {"sha256", "bytes"} and _checksum(record["sha256"]) and
            _integer(record["bytes"], 1), "Invalid source inventory record")
        actual[name] = _record(root, name)
        require(actual[name] == record, "Parser-QA source changed: " + name)
        total += record["bytes"]
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Parser source inventory exceeds metadata bound")
    return actual


def _parser_qa(root, proposed, checksum, ledger, prior):
    require(_checksum(checksum), "The parser native-QA file needs a caller-pinned SHA256")
    path = _inside(root, proposed, runs=True)
    require(path.suffix == ".json", "Expected a bounded native parser QA JSON file")
    assets = {str(path.relative_to(root)): _record(root, path)}
    require(assets[str(path.relative_to(root))]["sha256"] == checksum, "Pinned parser native QA bytes changed")
    value = _read(path)
    _passed_qa(value, METADATA_QA_SCHEMA)
    sources = _parser_inventory(root, value.get("source_inventory"))
    require(value.get("code_sha256") == {name: record["sha256"] for name, record in sources.items()} and
        value.get("automatic_downstream") is False,
        "Parser QA code hashes differ from its actual inventory")
    if "prior_admissions" in value:
        require(value["prior_admissions"] == prior, "Parser QA changed the twelve earlier admissions")
    entry = ledger["jobs"].get(METADATA_ROLE, {})
    binding = _entry(entry, METADATA_ROLE, gpu_seconds=0, sources=sources, job=value["job_id"])
    require(entry["binding_sha256"] == value.get("admission_binding_sha256") and
        _integer(binding.get("wall_time_seconds"), 1) and binding["wall_time_seconds"] <= 7200 and
        binding.get("expected_native_tests") == value["tests_run"] and
        binding.get("prior_entries_sha256") == value_digest(prior),
        "Parser native QA is not bound to its separately admitted CPU stage")
    accounting = path.parent / "accounting.json"
    assets[str(accounting.relative_to(root))] = _record(root, accounting)
    terminal = _read(accounting)
    cpu_name = validate_accounting(terminal, value["job_id"])
    require(terminal["elapsed_seconds"] <= binding["wall_time_seconds"], "Metadata QA exceeds its admitted CPU wall time")
    require(cpu_name != CPU_NAME and cpu_name != NAME, "Separate parser CPU allocation identity required")
    return {**prior, METADATA_ROLE: entry}, assets, sources, value, cpu_name


def validate_wrapper(root):
    text = metadata._small_bytes(root / SCRIPT, root).decode("utf-8")
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "b200_normal_q", "qos": "owl_b200_normal_short",
        "nodes": "1", "ntasks": "1", "gres": "gpu:1", "cpus-per-task": "8", "mem": "128G",
        "time": "02:00:00", "no-requeue": ""}
    allowed, directives, header = set(expected) | {"output", "error"}, {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"):
                header = False
            continue
        require(header, "SBATCH directive after executable code")
        parts = shlex.split(line.lstrip()[7:].strip(), comments=True)
        require(bool(parts), "Empty SBATCH directive")
        key, separator, argument = parts[0].partition("=")
        require(key.startswith("--") and key[2:] in allowed and key[2:] not in directives and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/duplicate/ambiguous SBATCH option")
        directives[key[2:]] = argument if separator else parts[1] if len(parts) == 2 else ""
    require(set(directives) == allowed and all(directives[key] == value for key, value in expected.items()),
        "Only the fixed single-B200/2h/8CPU/128G/no-requeue wrapper is permitted")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and ".." not in target.parts and target.parent == root / "extension/logs",
            "GPU logs must stay in the fixed extension/logs directory")
    return directives


def _ledger(value):
    require(isinstance(value, dict) and value.get("schema") == metadata.SCHEMA and
        value.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
        value.get("gpu_hour_cap", "missing") is None and isinstance(value.get("jobs"), dict),
        "Unknown extension ledger schema/history")
    return value


def scheduler_identities(prior):
    """Only one exact, already-sealed no-ID rejection is not an allocation.

    Called only after the current prior entries equal the hard-pinned native
    history. Unknown/altered uncertain entries remain a reconciliation blocker.
    """
    identities = {}
    for role, value in prior.items():
        if role == RECONCILED_ROLE:
            require(isinstance(value, dict) and value.get("status") == "uncertain" and
                "job_id" not in value and "possible_job_id" not in value and
                value.get("scheduler_returncode") == 1 and value.get("scheduler_stdout") == "" and
                value.get("requested_gpu_seconds") == 0 and
                _sha(_canonical(value)) == RECONCILED_ENTRY_SHA256,
                "The sole reconciled pre-scheduler no-ID rejection changed")
            continue
        require(isinstance(value, dict) and value.get("status") == "submitted" and
            _numeric_job(value.get("job_id")) and value["job_id"] not in identities,
            "Unknown uncertain/no-ID or duplicate prior scheduler identity")
        identities[value["job_id"]] = value
    return identities


def _known_queue(snapshot, prior, metadata_value, metadata_name):
    identities = scheduler_identities(prior)
    for row in snapshot["jobs"]:
        known = identities.get(row["job_id"])
        if known is not None:
            if known.get("requested_gpu_seconds") == 0:
                require(row["gpus"] == 0, "Earlier CPU identity unexpectedly requests GPUs")
            elif row["state"] in NONTERMINAL:
                require(row["gpus"] == 1, "Earlier live GPU allocation lacks exact whole-GPU resource evidence")
    previous.check_queue(snapshot, {name: value for name, value in prior.items()
        if name not in {previous.ROLE, qa.PRIOR_DIAGNOSTIC_ROLE, qa.ROLE, METADATA_ROLE}})
    fixed = {
        "1225360": ("FAILED", previous.NAME),
        qa.PRIOR_DIAGNOSTIC_JOB: ("COMPLETED", "sft-extension-source-grader-diagnostic"),
        CPU_JOB: ("COMPLETED", CPU_NAME),
        metadata_value["job_id"]: ("COMPLETED", metadata_name)}
    require(len(fixed) == 4, "Parser QA must have its own separate CPU allocation")
    for row in snapshot["jobs"]:
        if row["job_id"] in fixed:
            require((row["state"], row["name"]) == fixed[row["job_id"]] and row["gpus"] == 0,
                "Earlier CPU stage is active, renamed or has unexpected GPU resources")
        elif row["name"] in {value[1] for value in fixed.values()}:
            raise SubmissionError("Untracked duplicate prior CPU stage")


def load_evidence(root, qa_run, metadata_qa, metadata_qa_sha256, ledger):
    """Small metadata only. In particular, never follow plan.inputs.prepared."""
    root = Path(root).resolve()
    _ledger(ledger)
    plan_record = _record(root, PLAN)
    require(plan_record["sha256"] == PLAN_SHA256, "The frozen diagnostic plan changed")
    plan = qa.joint.validate_plan(_read(root / PLAN))
    sources, twelve, assets, directory = _scientific(root, qa_run, ledger)
    require(sources[PLAN] == plan_record, "Scientific plan/source identity differs")
    prior, parser_assets, parser_sources, parser_value, parser_name = _parser_qa(
        root, metadata_qa, metadata_qa_sha256, ledger, twelve)
    require(len(prior) == 13 and set(ledger["jobs"]) <= set(prior) | {ROLE} and
        all(ledger["jobs"].get(name) == value for name, value in prior.items()),
        "Preserve exactly thirteen established admissions; unknown/new roles require a new review")
    assets.update(parser_assets)
    assets[PLAN] = plan_record
    assets["deadline/job_ledger.json"] = _record(root, "deadline/job_ledger.json")
    require(assets["deadline/job_ledger.json"]["sha256"] == qa.HISTORICAL_LEDGER_SHA256,
        "Historical completed-experiment GPU ledger changed")
    directives = validate_wrapper(root)
    metadata.validate_campaign(_read(root / "extension/campaign.json"))
    return {"sources": sources, "parser_sources": parser_sources, "assets": assets, "prior": prior,
        "qa_run": str(directory.relative_to(root)), "prepared_sha256": plan["inputs"]["prepared"]["sha256"],
        "metadata_qa": str(_inside(root, metadata_qa, runs=True).relative_to(root)),
        "metadata_qa_sha256": metadata_qa_sha256, "metadata_value": parser_value,
        "metadata_name": parser_name, "directives": directives}


def _admission_receipt(root, entry, evidence):
    job = entry["job_id"]
    path = root / "extension/operations" / ("joint-memory-admission-job-" + job + ".json")
    receipt = {"schema": SCHEMA, "status": "admitted", "job_id": job, "plan_sha256": PLAN_SHA256,
        "prepared_sha256": evidence["prepared_sha256"], "qa_sha256": QA_PINS["native_qa.json"],
        "code_sha256": {name: item["sha256"] for name, item in evidence["sources"].items()},
        "metadata_qa": {"path": evidence["metadata_qa"], "sha256": evidence["metadata_qa_sha256"]},
        "metadata_source_sha256": evidence["parser_sources"], "binding_sha256": entry["binding_sha256"],
        "automatic_downstream": False}
    return path, receipt


def submit(root, qa_run, metadata_qa, metadata_qa_sha256, *, now=None, runner=None):
    """Submit at most once, after BOTH distinct native CPU gates terminate.

    A matching already-submitted entry returns its ID only when its immutable
    current-job receipt still matches. Uncertain/rejected jobs are never retried.
    ``runner`` is solely the offline fixture seam; production uses subprocess.run.
    """
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
        "Real isolated extension directory required")
    runner = subprocess.run if runner is None else runner
    def guard():
        when = datetime.now(timezone.utc) if now is None else now
        require(isinstance(when, datetime) and when.tzinfo is not None and
            (metadata.DEADLINE - when).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
            "Keep the full two-hour allocation and 48-hour deadline reserve")
        return when
    guard()
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "No ledger/lock aliases")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _ledger(_read(ledger_path))
        evidence = load_evidence(root, qa_run, metadata_qa, metadata_qa_sha256, ledger)
        command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT), evidence["qa_run"]]
        binding = {"role": ROLE, "command": command, "source_sha256": evidence["sources"],
            "metadata_source_sha256": evidence["parser_sources"], "plan_sha256": PLAN_SHA256,
            "prepared_sha256": evidence["prepared_sha256"], "initial_evidence": evidence["assets"],
            "qa_run": evidence["qa_run"], "metadata_qa": evidence["metadata_qa"],
            "metadata_qa_sha256": evidence["metadata_qa_sha256"],
            "prior_binding_sha256": {name: entry["binding_sha256"] for name, entry in evidence["prior"].items()},
            "requested_gpu_seconds": WALL_SECONDS, "requested_gpu_hours": 2,
            "wall_time_seconds": WALL_SECONDS, "closeout_reserve_seconds": RESERVE_SECONDS,
            "maximum_initial_concurrent_gpu_allocations": MAX_GPU_ALLOCATIONS,
            "deadline_utc": metadata.DEADLINE.isoformat(), "automatic_downstream": False}
        checksum = _sha(_canonical(binding))
        snapshot = queue_snapshot(runner)
        _known_queue(snapshot, evidence["prior"], evidence["metadata_value"], evidence["metadata_name"])
        require(load_evidence(root, qa_run, metadata_qa, metadata_qa_sha256, ledger) == evidence and
            _read(ledger_path) == ledger, "Sources, receipts or ledger changed while observing the live queue")
        existing = ledger["jobs"].get(ROLE)
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        if existing is not None:
            _entry(existing, ROLE, gpu_seconds=WALL_SECONDS, sources=evidence["sources"])
            require(existing.get("binding") == binding and existing.get("binding_sha256") == checksum and
                existing.get("command") == command and not same - {existing["job_id"]},
                "Existing admission is changed/uncertain/failed; never resubmit")
            for row in snapshot["jobs"]:
                if row["job_id"] == existing["job_id"]:
                    require(row["name"] == NAME and row["gpus"] == 1, "Current GPU job identity/resources changed")
            path, receipt = _admission_receipt(root, existing, evidence)
            require(_inside(root, path) == path and _read(path) == receipt,
                "Existing admitted job lacks its exact current-job receipt; reconcile manually")
            check_capacity(snapshot, already_submitted=True)
            return existing["job_id"]
        require(not same, "Untracked matching GPU job requires reconciliation")
        check_capacity(snapshot)
        when = guard()
        environment, removed = metadata._submission_environment()
        for key in ("output", "error"):
            Path(evidence["directives"][key]).parent.mkdir(parents=True, exist_ok=True)
        operations = root / "extension/operations"
        require(operations.is_dir() and not operations.is_symlink(), "Reviewed operations directory required")
        entry = {"status": "submitting", "binding": binding, "binding_sha256": checksum,
            "source_sha256": evidence["sources"], "metadata_source_sha256": evidence["parser_sources"],
            "plan_sha256": PLAN_SHA256, "command": command, "requested_gpu_seconds": WALL_SECONDS,
            "requested_gpu_hours": 2, "requested_at_utc": when.isoformat(), "scheduler_snapshot": snapshot,
            "removed_scheduler_environment_keys": removed}
        ledger["jobs"][ROLE] = entry
        metadata._write(ledger_path, ledger)
        try:
            response = runner(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status="uncertain", error_type=type(error).__name__)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Submission outcome uncertain; manual reconciliation required, no retry") from error
        entry.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:],
            scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip()) if isinstance(response.stdout, str) else None
        if response.returncode or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            entry["status"] = "submission_failed" if response.returncode else "uncertain"
            if match is not None:
                entry["possible_job_id"] = match.group(1)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Rejected/ambiguous submission preserved; never retry automatically")
        job = match.group(1)
        require(job not in scheduler_identities(evidence["prior"]) and
            job not in {row["job_id"] for row in snapshot["jobs"]}, "Scheduler returned an already known identity")
        # Journal acceptance before the current-job handoff. A crash here remains
        # uncertain, rather than falsely claiming a submitted job with no receipt.
        entry.update(status="uncertain", job_id=job, accepted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        path, receipt = _admission_receipt(root, entry, evidence)
        require(not path.exists() and not path.is_symlink(), "Existing current-job receipt requires manual reconciliation")
        metadata._write(path, receipt)
        entry.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            admission_receipt=str(path.relative_to(root)), admission_receipt_sha256=_record(root, path)["sha256"])
        metadata._write(ledger_path, ledger)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--qa-run", required=True)
    parser.add_argument("--metadata-qa", required=True)
    parser.add_argument("--metadata-qa-sha256", required=True)
    args = parser.parse_args()
    print(submit(args.root, args.qa_run, args.metadata_qa, args.metadata_qa_sha256), flush=True)


if __name__ == "__main__":
    main()
