"""Fixed metadata-only CPU preparation and manually admitted B200 diagnostic.

The historical ledger and six archived extension entries remain unchanged. This
adapter never imports model/data libraries or opens the private prepared cache,
pool, features, tokenizer or weights. Native QA is performed in the CPU stage;
the GPU stage requires its terminal accounting and independently bound metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess

from extension import development_retry_submission as previous
from extension import submission as metadata

CPU_ROLE = "smoke_preparation_20261021_v1"
GPU_ROLE = "native_model_admission_20261021_v1"
SCHEMA, DEADLINE = metadata.SCHEMA, metadata.DEADLINE
WALL_SECONDS, RESERVE_SECONDS = 7200, 172800
SubmissionError, require = metadata.SubmissionError, metadata.require
_canonical, _sha, _read = metadata._canonical, metadata._sha, metadata._read
SEAL_ROLE, SEAL_JOB_ID = previous.ROLE, "1223943"
EVIDENCE_DIRECTORY = "extension/report/development-seal-v2-20261021-1223943"
RECEIPT_SHA256 = {
    "accounting.json": "bfc7614d82c34246f35b285148121521ae8397e2ebd148914fbe042c8afbe846",
    "complete.json": "354859dd44e2fe3aa7800ec991b60649f00a99ce4768d8075af42c007289987d",
    "native_qa.json": "186e7574e26fcd1c1ed5ad3d840a590f4a4417aaa0b0880295b2a7b35f1d6359",
    "job_ledger_snapshot.json": "0fd2f4a3e3bc752158587d5843421306b0c91b28069c90fdd302fb63dda26041",
}
SOURCE_PATHS = frozenset(previous.SOURCE_PATHS | {
    "extension/preserving_data.py", "tests/test_preservation_data.py",
    "extension/native_model_admission.py", "tests/test_native_model_admission.py",
    "extension/smoke_preparation.py", "tests/test_smoke_preparation.py",
    "extension/smoke_submission.py", "tests/test_smoke_submission.py",
    "extension/smoke_preparation.slurm", "extension/native_model_admission.slurm",
    "extension/smoke_preparation_plan.json", "extension/native_model_admission_plan.json",
    "reasoning/features.py"})
CPU_PLAN = "extension/smoke_preparation_plan.json"
GPU_PLAN = "extension/native_model_admission_plan.json"
CPU_SCRIPT = "extension/smoke_preparation.slurm"
GPU_SCRIPT = "extension/native_model_admission.slurm"
CPU_NAME = "sft-extension-smoke-preparation"
GPU_NAME = "sft-extension-native-model-admission"
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"})
ZERO_FIELDS = ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")
_SHA = re.compile(r"[0-9a-f]{64}")


def source_inventory(root):
    """Bind exactly 51 reviewed small sources, never mutable runtime artifacts."""
    root = Path(root).resolve()
    result, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        require((root / name).is_file(), "Missing reviewed smoke source: " + name)
        raw = metadata._small_bytes(root / name, root)
        total += len(raw)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Smoke source inventory exceeds metadata bound")
        result[name] = {"sha256": _sha(raw), "bytes": len(raw)}
    return result


def _accounting(value, job_id, *, elapsed=None):
    require(isinstance(value, dict) and value.get("schema") == "extension-terminal-cpu-accounting-v1" and
        value.get("complete") is True and value.get("job_id") == job_id and value.get("gpu_seconds") == 0 and
        value.get("no_gpu_in_alloc_tres") is True and type(value.get("elapsed_seconds")) is int and
        0 < value["elapsed_seconds"] <= WALL_SECONDS and (elapsed is None or value["elapsed_seconds"] == elapsed) and
        value.get("stderr") == "" and isinstance(value.get("stdout"), str),
        "Terminal successful zero-GPU CPU accounting is required")
    rows = [line.split("|") for line in value["stdout"].splitlines() if line]
    allowed = {job_id, job_id + ".batch", job_id + ".extern"}
    require(rows and len({row[0] for row in rows}) == len(rows) and
        sum(row[0] == job_id for row in rows) == 1 and all(len(row) == 8 and row[0] in allowed and
            row[2:4] == ["COMPLETED", "0:0"] and row[4] == str(value["elapsed_seconds"]) and
            "gpu" not in row[7].lower() for row in rows), "CPU allocation accounting is partial/ambiguous/nonterminal")


def initial_evidence(root):
    """Return 19 pinned small receipts and six exact immutable ledger entries."""
    root = Path(root).resolve()
    inventory, five = previous.initial_evidence(root)
    inventory = dict(inventory)
    values = previous._receipts(root, EVIDENCE_DIRECTORY, RECEIPT_SHA256, inventory)
    six = previous._ledger(values["job_ledger_snapshot.json"], set(five) | {SEAL_ROLE})
    require(all(six.get(role) == entry for role, entry in five.items()), "Six-entry seal archive changed prior history")
    previous._job(six[SEAL_ROLE], SEAL_ROLE, SEAL_JOB_ID)
    sources = previous.source_inventory(root)
    qa, complete, entry = values["native_qa.json"], values["complete.json"], six[SEAL_ROLE]
    require(qa.get("schema") == "extension-development-seal-native-qa-v2" and qa.get("success") is True and
        qa.get("tests_run") == 230 and qa.get("job_id") == SEAL_JOB_ID and qa.get("gpu_disabled") is True and
        qa.get("wrapper_syntax_passed") is True and all(type(qa.get(key)) is int and qa[key] == 0 for key in ZERO_FIELDS) and
        qa.get("source_inventory") == sources == entry.get("source_sha256") and
        qa.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        qa.get("admission_binding_sha256") == entry["binding_sha256"] and
        qa.get("compute_proof", {}).get("gpu_requested_or_allocated") is False,
        "All 38 sealed sources and successful native QA230 must remain unchanged")
    require(complete.get("schema") == "extension-development-seal-v2" and complete.get("complete") is True and
        complete.get("input_stats_unchanged_at_closeout") is True and
        complete.get("native_qa", {}).get("sha256") == RECEIPT_SHA256["native_qa.json"] and
        complete.get("native_qa", {}).get("tests_run") == 230 and
        complete.get("compute_start", {}).get("job_id") == SEAL_JOB_ID and
        complete.get("compute_end", {}).get("job_id") == SEAL_JOB_ID and
        complete.get("actions") == {"splits_chosen": True, "generation": False, "training": False,
            "automatic_downstream": False} and isinstance(complete.get("output_sha256"), dict) and
        len(complete["output_sha256"]) == 12 and all(_SHA.fullmatch(item) for item in complete["output_sha256"].values()),
        "Completed reserved development split metadata is required")
    _accounting(values["accounting.json"], SEAL_JOB_ID, elapsed=99)
    return inventory, six


def validate_plans(cpu, gpu, sources):
    require(isinstance(cpu, dict) and cpu.get("schema") == "extension-native-smoke-preparation-plan-v1" and
        cpu.get("cpu_only") is True and cpu.get("choose_smoke_witnesses") is True and
        all(cpu.get(name) is False for name in ("model_inference", "training", "automatic_downstream")) and
        cpu.get("deadline_utc") == DEADLINE.isoformat() and cpu.get("wall_time_seconds") == WALL_SECONDS and
        cpu.get("reserve_seconds") == RESERVE_SECONDS and cpu.get("criterion") == previous.CRITERION and
        cpu.get("native_plan") == {"path": GPU_PLAN, "sha256": sources[GPU_PLAN]["sha256"]} and
        cpu.get("development_seal") == {"path": "extension/runs/development-seal-v2-20261021-1223943/complete.json",
            "sha256": RECEIPT_SHA256["complete.json"]} and
        cpu.get("pool", {}).get("path") == "deadline/artifacts/pool.json" and
        cpu.get("pool", {}).get("sha256") == previous.previous.POOL_SHA256,
        "Only the separately frozen CPU preparation scope/input pins are permitted")
    modes = ["sft", "ce_kl", "asft", "prior_anchored"]
    require(isinstance(gpu, dict) and gpu.get("schema") == "extension-native-model-admission-plan-v1" and
        gpu.get("diagnostic_only") is True and gpu.get("training_pilot") is False and
        gpu.get("automatic_downstream") is False and gpu.get("deadline_utc") == DEADLINE.isoformat() and
        gpu.get("base_revision") == "aa8e72537993ba99e69dfaafa59ed015b17504d1" and
        gpu.get("objectives") == modes and gpu.get("case_order") == [mode + "-" + kind for mode in modes for kind in ("short", "long")],
        "Only the frozen eight-case original-base diagnostic is permitted")
    resources = gpu.get("resources", {})
    require(all(resources.get(key) == value for key, value in {"gpus": 1, "gpu_type": "B200", "host_cpus": 8,
        "host_ram_gib": 128, "wall_time_seconds": WALL_SECONDS, "reserve_seconds": RESERVE_SECONDS,
        "maximum_initial_concurrent_gpu_allocations": 2}.items()), "GPU diagnostic resource/concurrency bounds differ")


def validate_wrapper(path, root, *, gpu=False):
    text = metadata._small_bytes(path, root).decode("utf-8")
    expected = {"job-name": GPU_NAME if gpu else CPU_NAME, "account": "tml_2026",
        "partition": "b200_normal_q" if gpu else "normal_q", "qos": "owl_b200_normal_short" if gpu else "owl_normal_short",
        "nodes": "1", "ntasks": "1", "cpus-per-task": "8", "mem": "128G" if gpu else "64G",
        "time": "02:00:00", "no-requeue": ""}
    if gpu:
        expected["gres"] = "gpu:1"
    allowed, directives, header = set(expected) | {"output", "error"}, {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"):
                header = False
            continue
        require(header, "SBATCH directive after executable code would be ignored")
        parts = shlex.split(line.lstrip()[7:].strip(), comments=True)
        require(bool(parts), "Empty SBATCH directive")
        option, separator, argument = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in allowed and option[2:] not in directives,
            "Unknown, duplicated or unauthorized SBATCH option")
        require(len(parts) == 1 if separator else len(parts) in (1, 2), "Ambiguous SBATCH directive")
        directives[option[2:]] = argument if separator else (parts[1] if len(parts) == 2 else "")
    require(set(directives) == allowed and all(directives.get(key) == item for key, item in expected.items()),
        "Wrapper must declare only its exact reviewed resources")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and target.resolve().is_relative_to(Path(root) / "extension") and
            ".." not in target.parts, "Runtime logs must remain below extension")
    return directives


def _inside_metadata(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and
        ".." not in Path(relative).parts and relative.startswith("extension/"), "Expected bounded extension metadata path")
    path = root / relative
    metadata._small_bytes(path, root)
    return path


def _qa(value, schema, sources, job_id, *, native_plan_sha256, cpu_plan_sha256=None):
    require(isinstance(value, dict) and value.get("schema") == schema and value.get("success") is True and
        type(value.get("tests_run")) is int and value["tests_run"] > 230 and value.get("gpu_disabled") is True and
        all(type(value.get(key)) is int and value[key] == 0 for key in ZERO_FIELDS) and
        value.get("source_inventory") == sources and
        value.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        value.get("compute_proof", {}).get("job_id") == job_id and
        value.get("compute_proof", {}).get("gpu_requested_or_allocated") is False,
        "Current CPU-native zero-error QA/code/allocation binding is required")
    key = "native_plan_sha256" if schema.endswith("unit-qa-v1") else "plan_sha256"
    require(value.get(key) == native_plan_sha256 and (cpu_plan_sha256 is None or
        value.get("cpu_preparation_plan_sha256") == cpu_plan_sha256), "Native QA plans differ")


def _binding(root, sources, evidence, six, *, gpu=False, cpu_run=None):
    return {"role": GPU_ROLE if gpu else CPU_ROLE,
        "command": ["sbatch", "--clusters=owl", "--parsable", str(root / (GPU_SCRIPT if gpu else CPU_SCRIPT))] +
            ([cpu_run] if gpu else []),
        "source_sha256": sources, "campaign_sha256": sources["extension/campaign.json"]["sha256"],
        "plan_sha256": sources[GPU_PLAN if gpu else CPU_PLAN]["sha256"],
        "native_plan_sha256": sources[GPU_PLAN]["sha256"], "initial_evidence": evidence,
        "prior_binding_sha256": {name: entry["binding_sha256"] for name, entry in six.items()},
        "requested_gpu_hours": 2 if gpu else 0, "requested_gpu_seconds": WALL_SECONDS if gpu else 0,
        "wall_time_seconds": WALL_SECONDS, "closeout_reserve_seconds": RESERVE_SECONDS,
        "deadline_utc": DEADLINE.isoformat(), "automatic_downstream": False}


def cpu_readiness(root, cpu_run, sources, ledger):
    """Validate only small proof files. Never read/hash the private prepared cache."""
    entry = ledger["jobs"].get(CPU_ROLE, {})
    job_id = entry.get("job_id")
    evidence, six = initial_evidence(root)
    expected_binding = _binding(root, sources, evidence, six)
    require(entry.get("status") == "submitted" and isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id) and
        entry.get("source_sha256") == sources and entry.get("requested_gpu_seconds") == 0 and
        entry.get("requested_gpu_hours") == 0 and entry.get("binding") == expected_binding and
        entry.get("command") == expected_binding["command"] and
        entry.get("binding_sha256") == _sha(_canonical(expected_binding)), "Exact submitted CPU preparation admission required")
    expected = "extension/runs/native-smoke-preparation-" + job_id
    require(cpu_run == expected, "CPU run must be exactly derived from its admitted scheduler ID")
    inventory, values = {}, {}
    for name in ("complete.json", "native_qa.json", "unit_qa.json", "preparation.json", "accounting.json"):
        relative = expected + "/" + name
        path = _inside_metadata(root, relative)
        raw = metadata._small_bytes(path, root)
        inventory[relative] = {"sha256": _sha(raw), "bytes": len(raw)}
        values[name] = _read(path)
    complete = values["complete.json"]
    require(complete.get("schema") == "extension-native-smoke-preparation-complete-v1" and complete.get("complete") is True and
        complete.get("cpu_only") is True and complete.get("job_id") == job_id and complete.get("gpu_seconds") == 0 and
        complete.get("input_snapshot_unchanged") is True and complete.get("source_inventory") == sources and
        complete.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        complete.get("plan_sha256") == sources[CPU_PLAN]["sha256"] and
        complete.get("native_plan_sha256") == sources[GPU_PLAN]["sha256"] and
        all(complete.get(key, {}).get("job_id") == job_id and
            complete[key].get("gpu_requested_or_allocated") is False for key in ("compute_start", "compute_end")),
        "Complete CPU preparation and matching source/input/allocation proofs are required")
    for key, filename in (("native_qa", "native_qa.json"), ("unit_qa", "unit_qa.json"), ("preparation", "preparation.json")):
        require(complete.get(key) == {"path": expected + "/" + filename,
            "sha256": inventory[expected + "/" + filename]["sha256"]}, "CPU completion receipt binding differs: " + key)
    prepared = complete.get("prepared", {})
    require(isinstance(prepared, dict) and set(prepared) == {"path", "sha256"} and
        prepared["path"] == expected + "/prepared.json" and isinstance(prepared["sha256"], str) and
        _SHA.fullmatch(prepared["sha256"]), "Prepared-cache hash must be supplied by completed CPU proof")
    outputs = complete.get("output_sha256", {})
    require(isinstance(outputs, dict) and outputs.get("prepared.json") == prepared["sha256"] and
        all(outputs.get(name) == inventory[expected + "/" + name]["sha256"] for name in
            ("native_qa.json", "unit_qa.json", "preparation.json")), "CPU output inventory differs")
    _qa(values["native_qa.json"], "extension-native-model-admission-native-qa-v1", sources, job_id,
        native_plan_sha256=sources[GPU_PLAN]["sha256"], cpu_plan_sha256=sources[CPU_PLAN]["sha256"])
    require(values["native_qa.json"].get("prepared_sha256") == prepared["sha256"], "Native QA cache hash differs")
    _qa(values["unit_qa.json"], "extension-native-smoke-preparation-unit-qa-v1", sources, job_id,
        native_plan_sha256=sources[GPU_PLAN]["sha256"])
    preparation = values["preparation.json"]
    require(preparation.get("schema") == "extension-native-smoke-preparation-v1" and preparation.get("complete") is True and
        preparation.get("cpu_only") is True and preparation.get("gpu_seconds") == 0 and
        preparation.get("plan_sha256") == sources[GPU_PLAN]["sha256"] and
        preparation.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        preparation.get("native_qa", {}).get("path") in (expected + "/unit_qa.json", str(root / expected / "unit_qa.json")) and
        preparation.get("native_qa", {}).get("sha256") == inventory[expected + "/unit_qa.json"]["sha256"] and
        preparation.get("input_snapshot_unchanged") is True and preparation.get("no_model_or_benchmark_outcomes") is True,
        "CPU preparation/source-only proof differs")
    _accounting(values["accounting.json"], job_id)
    return {"cpu_run": expected, "job_id": job_id, "receipts": inventory, "prepared": prepared,
        "native_qa": complete["native_qa"], "accounting_sha256": inventory[expected + "/accounting.json"]["sha256"]}


def _check_prior_queue(snapshot, six):
    previous._check_prior_queue(snapshot, {key: value for key, value in six.items() if key != SEAL_ROLE})
    for row in snapshot["jobs"]:
        if row["job_id"] == SEAL_JOB_ID:
            require((row["state"], row["name"]) == ("COMPLETED", "sft-extension-development-seal-v2"),
                "Archived seal has active/unknown scheduler state")
        elif row["name"] == "sft-extension-development-seal-v2" or row["job_id"].split("_")[0] == SEAL_JOB_ID:
            raise SubmissionError("Untracked duplicate sealed development job")


def gpu_queue_snapshot(runner):
    """Count expanded nonterminal GPU allocations; unknown GRES fails closed."""
    command = ["squeue", "--clusters=owl", "--array", "--states=all", "--user=" + metadata.getpass.getuser(),
        "--noheader", "--format=%64i|%64T|%128j|%128b"]
    result = runner(command, capture_output=True, text=True, timeout=30)
    require(result.returncode == 0 and not result.stderr.strip() and len(result.stdout.encode()) <= 256 << 10,
        "Cannot establish GPU concurrency snapshot")
    rows, seen, active = [], set(), 0
    for line in result.stdout.splitlines():
        if not line.strip() or line.strip() == "CLUSTER: owl":
            continue
        parts = [item.strip() for item in line.split("|")]
        require(len(parts) == 4 and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", parts[0]) and
            re.fullmatch(r"[A-Z_]+", parts[1]) and parts[2] and parts[0] not in seen, "Ambiguous GPU queue row")
        seen.add(parts[0])
        gres = parts[3]
        tokens = [] if gres in ("", "N/A", "(null)", "None") else gres.split(",")
        gpu = False
        for token in tokens:
            match = re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:([1-9][0-9]*)(?:\(S:[0-9,-]+\))?", token)
            require(match is not None, "Unknown scheduler GRES cannot establish GPU concurrency")
            gpu = True
        if gpu and parts[1] not in TERMINAL:
            active += 1
        rows.append({"job_id": parts[0], "state": parts[1], "name": parts[2], "gres": gres, "gpu": gpu})
    return {"command": command, "stdout": result.stdout, "stdout_sha256": _sha(result.stdout.encode()),
        "jobs": rows, "active_gpu_allocations": active}


def _admission_path(root, job_id):
    path = root / "extension/operations" / ("native-model-admission-job-" + job_id + ".json")
    require(not path.is_symlink() and path.parent.resolve().is_relative_to(root / "extension"), "Unsafe admission receipt path")
    return path


def _gpu_receipt(root, entry):
    proof = entry["binding"]["cpu_preparation"]
    value = {"schema": "extension-native-model-admission-job-v1", "status": "admitted", "job_id": entry["job_id"],
        "role": GPU_ROLE, "plan_sha256": entry["binding"]["native_plan_sha256"], "prepared_sha256": proof["prepared"]["sha256"],
        "qa_sha256": proof["native_qa"]["sha256"], "code_sha256": {name: item["sha256"] for name, item in entry["source_sha256"].items()},
        "source_inventory": entry["source_sha256"], "binding_sha256": entry["binding_sha256"],
        "cpu_preparation": proof, "requested_gpu_hours": 2, "requested_gpu_seconds": WALL_SECONDS,
        "automatic_downstream": False}
    path = _admission_path(root, entry["job_id"])
    if path.exists():
        require(_read(path) == value, "Conflicting existing current-job admission receipt")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata._write(path, value)


def _submit(root, *, gpu, cpu_run=None, now=None, runner=None):
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
        "Real isolated extension directory required")
    require(cpu_run is None if not gpu else isinstance(cpu_run, str), "Only GPU admission takes the fixed CPU run argument")
    runner = subprocess.run if runner is None else runner
    def guard():
        checked = datetime.now(timezone.utc) if now is None else now
        require(isinstance(checked, datetime) and checked.tzinfo is not None and
            (DEADLINE - checked).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS, "Insufficient diagnostic time and 48-hour reserve")
        return checked
    guard()
    sources = source_inventory(root)
    metadata.validate_campaign(_read(extension / "campaign.json"))
    validate_plans(_read(root / CPU_PLAN), _read(root / GPU_PLAN), sources)
    script = root / (GPU_SCRIPT if gpu else CPU_SCRIPT)
    directives = validate_wrapper(script, root, gpu=gpu)
    evidence, six = initial_evidence(root)
    role = GPU_ROLE if gpu else CPU_ROLE
    command = ["sbatch", "--clusters=owl", "--parsable", str(script)] + ([cpu_run] if gpu else [])
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "Ledger/lock symlinks forbidden")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(ledger_path)
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
            ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and ledger.get("gpu_hour_cap", "missing") is None and
            isinstance(ledger.get("jobs"), dict) and set(ledger["jobs"]) <= set(six) | {CPU_ROLE, GPU_ROLE} and
            all(ledger["jobs"].get(key) == value for key, value in six.items()), "Changed six-entry history or unknown role")
        require(gpu or GPU_ROLE not in ledger["jobs"] or CPU_ROLE in ledger["jobs"], "GPU admission without CPU role")
        proof = cpu_readiness(root, cpu_run, sources, ledger) if gpu else None
        binding = _binding(root, sources, evidence, six, gpu=gpu, cpu_run=cpu_run)
        if gpu:
            binding["cpu_preparation"] = proof
            binding["cpu_admission_binding_sha256"] = ledger["jobs"][CPU_ROLE]["binding_sha256"]
        checksum = _sha(_canonical(binding))
        snapshot = metadata._queue_snapshot(runner)
        _check_prior_queue(snapshot, six)
        require(source_inventory(root) == sources and validate_wrapper(script, root, gpu=gpu) == directives and
            initial_evidence(root) == (evidence, six), "Source/evidence changed during admission")
        if gpu:
            require(cpu_readiness(root, cpu_run, sources, ledger) == proof, "CPU proof changed during admission")
            cpu_id = ledger["jobs"][CPU_ROLE]["job_id"]
            for row in snapshot["jobs"]:
                if row["job_id"] == cpu_id:
                    require((row["state"], row["name"]) == ("COMPLETED", CPU_NAME), "CPU preparation not terminal in live scheduler")
                elif row["name"] == CPU_NAME or row["job_id"].split("_")[0] == cpu_id:
                    raise SubmissionError("Untracked duplicate CPU preparation")
        same_name = {row["job_id"] for row in snapshot["jobs"] if row["name"] == directives["job-name"]}
        existing = ledger["jobs"].get(role)
        if existing is not None:
            require(isinstance(existing, dict) and existing.get("status") == "submitted" and
                existing.get("binding") == binding and existing.get("binding_sha256") == checksum and
                existing.get("command") == command and existing.get("source_sha256") == sources and
                existing.get("requested_gpu_seconds") == binding["requested_gpu_seconds"] and
                isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", existing["job_id"]),
                "Existing role changed, failed or uncertain; never resubmit")
            require(not same_name - {existing["job_id"]}, "Untracked duplicate current role")
            if gpu:
                _gpu_receipt(root, existing)
            return existing["job_id"]
        require(not same_name, "Untracked current role; reconcile before admission")
        gpu_snapshot = gpu_queue_snapshot(runner) if gpu else None
        require(not gpu or gpu_snapshot["active_gpu_allocations"] < 2, "Two initial GPU allocations are already admitted")
        checked = guard()
        for key in ("output", "error"):
            Path(directives[key]).parent.mkdir(parents=True, exist_ok=True)
        environment, removed = metadata._submission_environment()
        entry = {"status": "submitting", "binding": binding, "binding_sha256": checksum, "command": command,
            "source_sha256": sources, "campaign_sha256": binding["campaign_sha256"], "plan_sha256": binding["plan_sha256"],
            "initial_evidence": evidence, "requested_gpu_hours": binding["requested_gpu_hours"],
            "requested_gpu_seconds": binding["requested_gpu_seconds"], "requested_at_utc": checked.isoformat(),
            "scheduler_snapshot": snapshot, "removed_scheduler_environment_keys": removed}
        if gpu:
            entry["gpu_concurrency_snapshot"] = gpu_snapshot
        ledger["jobs"][role] = entry
        metadata._write(ledger_path, ledger)
        try:
            completed = runner(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status="uncertain", error_type=type(error).__name__)
            for name in ("stdout", "stderr"):
                value = getattr(error, name, None)
                if value:
                    entry["scheduler_" + name] = (value.decode(errors="replace") if isinstance(value, bytes) else str(value))[-2000:]
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Submission uncertain; no automatic retry") from error
        entry.update(scheduler_stdout=completed.stdout[-2000:], scheduler_stderr=completed.stderr[-2000:], scheduler_returncode=completed.returncode)
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", completed.stdout.strip())
        if completed.returncode != 0 or parsed is None:
            entry["status"] = "submission_failed" if completed.returncode else "uncertain"
            if parsed:
                entry["possible_job_id"] = parsed.group(1)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Failed/ambiguous submission; never resubmit without reconciliation")
        entry.update(status="submitted", job_id=parsed.group(1), submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        if gpu:
            _gpu_receipt(root, entry)
        return entry["job_id"]


def submit_cpu_preparation(root=Path("."), *, now=None, runner=None):
    return _submit(root, gpu=False, now=now, runner=runner)


def submit_gpu_diagnostic(root=Path("."), *, cpu_run, now=None, runner=None):
    return _submit(root, gpu=True, cpu_run=cpu_run, now=now, runner=runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("cpu")
    commands.add_parser("gpu").add_argument("--cpu-run", required=True)
    args = parser.parse_args()
    print(submit_cpu_preparation(args.root) if args.stage == "cpu" else
        submit_gpu_diagnostic(args.root, cpu_run=args.cpu_run))


if __name__ == "__main__":
    main()
