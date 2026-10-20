"""Admit only the fixed, metadata-bound labeled-source CPU feasibility stage.

No model/data dependencies are imported. The earlier admission and its evidence
are preserved byte-for-byte in meaning; this module never opens the historical
deadline ledger. Failed or ambiguous submissions require reconciliation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import shlex
import subprocess

from extension import submission as metadata

ROLE = "labeled_feasibility_20261020_v1"
INITIAL_ROLE = metadata.ROLE
INITIAL_JOB_ID = "1222225"
SCHEMA = metadata.SCHEMA
DEADLINE = metadata.DEADLINE
WALL_SECONDS = 7200
RESERVE_SECONDS = 172800
SubmissionError = metadata.SubmissionError
require = metadata.require
_canonical, _sha, _read = metadata._canonical, metadata._sha, metadata._read
EVIDENCE_DIRECTORY = "extension/report/feasibility-20261020-1222225"
INITIAL_RECEIPT_SHA256 = {
    "accounting.json": "d864ce6d17a8ee2afc7c421bf4af14a421920334a8c25798b4a5e3330ef41969",
    "complete.json": "8cee12034e7c9f9f6f3334cdac2d3edf5504658ef8000c664761a16887440505",
    "native_qa.json": "231aa7156bd5b2ae9622f464c2c501ee5aba4de25ecb1bc4775026e65a33f1e0",
    "job_ledger_snapshot.json": "fc8823b440c38a3880bd39cc4b1d18403500f437bcbe346c6bddba025babe56b",
}
SOURCE_PATHS = frozenset(metadata.REQUIRED | {
    "extension/objectives.py", "tests/test_extension_objectives.py",
    "extension/labeled_submission.py", "extension/labeled_feasibility.py",
    "extension/labeled_feasibility.slurm", "extension/labeled_plan.json",
    "tests/test_labeled_submission.py", "tests/test_labeled_feasibility.py",
})
SOURCE_PINS = {
    "EleutherAI/hendrycks_math": {
        "revision": "fe39eaeb3783a7dd5d7e305da9a8db9b0c4e6b93",
        "files": frozenset(subject + "/train/0000.parquet" for subject in (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")),
    },
    "allenai/ai2_arc": {
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "files": frozenset({"ARC-Challenge/train-00000-of-00001.parquet"}),
    },
}


def source_inventory(root):
    """Explicit small static sources only; mutable ledger/receipts are separate."""
    root = Path(root).resolve()
    inventory, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        path = root / name
        require(path.is_file(), "Missing required labeled stage source/test file: " + name)
        value = metadata._small_bytes(path, root)
        total += len(value)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Labeled source inventory exceeds metadata bound")
        inventory[name] = dict(sha256=_sha(value), bytes=len(value))
    return inventory


def validate_plan(value):
    require(isinstance(value, dict) and value.get("schema") == "extension-labeled-feasibility-plan-v1",
            "Expected the fixed labeled feasibility plan schema")
    require(value.get("cpu_only") is True and value.get("model_inference") is False and
            value.get("choose_splits") is False and value.get("automatic_downstream") is False,
            "Plan must admit CPU feasibility only, without inference, splits or downstream work")
    require(value.get("deadline_utc") == DEADLINE.isoformat() and
            type(value.get("wall_time_seconds")) is int and value["wall_time_seconds"] == WALL_SECONDS and
            type(value.get("reserve_seconds")) is int and value["reserve_seconds"] == RESERVE_SECONDS and
            value.get("source_metadata") == "extension/development_source_metadata.json",
            "Plan deadline, fixed metadata path or two-hour/48-hour bounds differ")
    sources = value.get("sources")
    require(isinstance(sources, list) and len(sources) == len(SOURCE_PINS), "Exactly the two pinned train sources are required")
    seen = set()
    for source in sources:
        require(isinstance(source, dict) and isinstance(source.get("repo_id"), str), "Invalid pinned source")
        name = source["repo_id"]
        require(name in SOURCE_PINS and name not in seen, "Unknown or duplicate labeled source")
        seen.add(name)
        files = source.get("files")
        require(source.get("revision") == SOURCE_PINS[name]["revision"] and
                isinstance(files, list) and all(isinstance(item, str) for item in files) and
                len(files) == len(set(files)) and set(files) == SOURCE_PINS[name]["files"],
                "Only the exact pinned MATH/ARC-Challenge training files are permitted")


def validate_wrapper(path, root):
    text = metadata._small_bytes(path, root).decode("utf-8")
    allowed = {"job-name", "account", "partition", "qos", "nodes", "ntasks", "cpus-per-task",
               "mem", "time", "no-requeue", "output", "error"}
    directives, header = {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"):
                header = False
            continue
        require(header, "SBATCH directives after executable code would be ignored")
        parts = shlex.split(line.lstrip()[7:].strip(), comments=True)
        require(bool(parts), "Empty SBATCH directive")
        option, separator, argument = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in allowed, "GPU/unknown SBATCH option is forbidden")
        key = option[2:]
        require(key not in directives, "Duplicate SBATCH option")
        if separator:
            require(len(parts) == 1, "Ambiguous SBATCH directive")
        else:
            require(len(parts) in (1, 2), "Ambiguous SBATCH directive")
            argument = parts[1] if len(parts) == 2 else ""
        directives[key] = argument
    expected = {"job-name": "sft-extension-labeled-feasibility", "account": "tml_2026",
                "partition": "normal_q", "qos": "owl_normal_short", "nodes": "1", "ntasks": "1",
                "cpus-per-task": "8", "mem": "64G", "time": "02:00:00", "no-requeue": ""}
    require(set(directives) == allowed and all(directives.get(key) == value for key, value in expected.items()),
            "Only the fixed 8-CPU/64G/zero-GPU/two-hour labeled wrapper is permitted")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and target.resolve().is_relative_to(Path(root) / "extension") and
                ".." not in target.parts, "Runtime logs must stay below extension")
    return directives


def initial_evidence(root):
    """Read only four pinned small public native receipts, never source/model data."""
    root = Path(root).resolve()
    values, inventory = {}, {}
    for name, expected in INITIAL_RECEIPT_SHA256.items():
        relative = EVIDENCE_DIRECTORY + "/" + name
        value = metadata._small_bytes(root / relative, root)
        require(_sha(value) == expected, "Initial native evidence digest mismatch: " + name)
        inventory[relative] = dict(sha256=expected, bytes=len(value))
        values[name] = _read(root / relative)
    account, complete = values["accounting.json"], values["complete.json"]
    qa, snapshot = values["native_qa.json"], values["job_ledger_snapshot.json"]
    require(account.get("schema") == "extension-terminal-cpu-accounting-v1" and account.get("complete") is True and
            account.get("job_id") == INITIAL_JOB_ID and account.get("gpu_seconds") == 0 and
            account.get("no_gpu_in_alloc_tres") is True and type(account.get("elapsed_seconds")) is int and
            account["elapsed_seconds"] > 0 and account.get("stderr") == "", "Initial terminal zero-GPU accounting is incomplete")
    require(isinstance(account.get("stdout"), str), "Initial accounting raw evidence missing")
    rows = [line.split("|") for line in account["stdout"].splitlines() if line]
    allocations = [row for row in rows if row[0] == INITIAL_JOB_ID]
    require(len(allocations) == 1 and all(len(row) == 8 and row[0] in
            {INITIAL_JOB_ID, INITIAL_JOB_ID + ".batch", INITIAL_JOB_ID + ".extern"} for row in rows) and
            len({row[0] for row in rows}) == len(rows), "Initial scheduler allocation evidence is ambiguous")
    allocation = allocations[0]
    require(allocation[2:4] == ["COMPLETED", "0:0"] and allocation[4] == str(account["elapsed_seconds"]) and
            "gpu" not in allocation[7].lower() and allocation[5] and allocation[6],
            "Initial CPU allocation must have completed successfully with no GPU allocation")
    require(qa.get("schema") == "extension-feasibility-native-qa-v1" and qa.get("job_id") == INITIAL_JOB_ID and
            qa.get("success") is True and qa.get("gpu_disabled") is True and qa.get("wrapper_syntax_passed") is True and
            qa.get("tests_run") == 64 and all(qa.get(key) == 0 for key in
                ("errors", "failures", "skipped", "expected_failures", "unexpected_successes")) and
            qa.get("compute_proof", {}).get("gpu_requested_or_allocated") is False,
            "Initial native QA must have all 64 tests pass without skips or GPUs")
    require(complete.get("schema") == "extension-cpu-feasibility-v1" and complete.get("complete") is True and
            complete.get("gpu_seconds_requested_or_used") == 0 and
            complete.get("native_qa", {}).get("tests_run") == 64 and
            complete.get("native_qa", {}).get("sha256") == INITIAL_RECEIPT_SHA256["native_qa.json"] and
            complete.get("input_stats_unchanged_at_closeout") is True,
            "Initial CPU feasibility completion does not bind the passing QA")
    require(snapshot.get("schema") == SCHEMA and snapshot.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
            snapshot.get("gpu_hour_cap", "missing") is None and isinstance(snapshot.get("jobs"), dict) and
            set(snapshot["jobs"]) == {INITIAL_ROLE}, "Initial ledger archive has unexpected roles")
    entry = snapshot["jobs"][INITIAL_ROLE]
    require(entry.get("status") == "submitted" and entry.get("job_id") == INITIAL_JOB_ID and
            entry.get("requested_gpu_hours") == 0 and entry.get("requested_gpu_seconds") == 0 and
            isinstance(entry.get("binding"), dict) and entry.get("binding_sha256") == _sha(_canonical(entry["binding"])) and
            entry.get("binding", {}).get("role") == INITIAL_ROLE and
            entry.get("binding", {}).get("requested_gpu_seconds") == 0,
            "Initial archived submission binding is invalid")
    return inventory, entry


def submit_labeled_feasibility(root=Path("."), *, now=None, runner=None):
    """Submit only this fixed CPU role; exact successful retries return its ID."""
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
            "A real isolated extension directory is required")
    runner = subprocess.run if runner is None else runner
    clock = (lambda: datetime.now(timezone.utc)) if now is None else (lambda: now)

    def guard():
        checked = clock()
        require(isinstance(checked, datetime) and checked.tzinfo is not None and
                (DEADLINE - checked).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
                "Insufficient time for two-hour CPU stage and 48-hour deadline reserve")
        return checked

    checked = guard()
    sources = source_inventory(root)
    campaign = _read(extension / "campaign.json")
    metadata.validate_campaign(campaign)
    validate_plan(_read(extension / "labeled_plan.json"))
    script = extension / "labeled_feasibility.slurm"
    directives = validate_wrapper(script, root)
    evidence, initial_entry = initial_evidence(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(script)]
    binding = dict(role=ROLE, command=command, source_sha256=sources,
                   campaign_sha256=sources["extension/campaign.json"]["sha256"],
                   plan_sha256=sources["extension/labeled_plan.json"]["sha256"], initial_evidence=evidence,
                   initial_role_binding_sha256=initial_entry["binding_sha256"],
                   requested_gpu_hours=0, requested_gpu_seconds=0, wall_time_seconds=WALL_SECONDS,
                   deadline_utc=DEADLINE.isoformat(), closeout_reserve_seconds=RESERVE_SECONDS)
    binding_sha256 = _sha(_canonical(binding))
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "Extension ledger/lock cannot be symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(ledger_path.exists(), "Existing initial-stage extension ledger is required")
        ledger = _read(ledger_path)
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
                ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
                ledger.get("gpu_hour_cap", "missing") is None and isinstance(ledger.get("jobs"), dict) and
                set(ledger["jobs"]) <= {INITIAL_ROLE, ROLE} and
                ledger["jobs"].get(INITIAL_ROLE) == initial_entry,
                "Unknown extension role or changed initial entry; reconcile manually")
        snapshot = metadata._queue_snapshot(runner)
        require(not any(row["job_id"].split("_")[0] == INITIAL_JOB_ID or
                        row["name"] == "sft-extension-feasibility" for row in snapshot["jobs"]),
                "Initial CPU role is still visible or duplicated in scheduler; reconcile before admission")
        require(source_inventory(root) == sources and validate_wrapper(script, root) == directives and
                initial_evidence(root) == (evidence, initial_entry), "Source/evidence changed during admission")
        existing = ledger["jobs"].get(ROLE)
        same_name = {row["job_id"] for row in snapshot["jobs"] if row["name"] == directives["job-name"]}
        if existing is not None:
            require(isinstance(existing, dict) and existing.get("status") == "submitted" and
                    existing.get("binding_sha256") == binding_sha256 and existing.get("binding") == binding and
                    existing.get("command") == command and existing.get("source_sha256") == sources and
                    existing.get("campaign_sha256") == binding["campaign_sha256"] and
                    existing.get("plan_sha256") == binding["plan_sha256"] and
                    existing.get("requested_gpu_hours") == 0 and existing.get("requested_gpu_seconds") == 0 and
                    isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", existing["job_id"]),
                    "Existing labeled submission changed, uncertain or failed; never resubmit")
            require(not (same_name - {existing["job_id"]}), "Untracked duplicate labeled job requires reconciliation")
            return existing["job_id"]
        require(not same_name, "Untracked live labeled feasibility job requires reconciliation")
        checked = guard()
        for key in ("output", "error"):
            Path(directives[key]).parent.mkdir(parents=True, exist_ok=True)
        environment, removed = metadata._submission_environment()
        entry = dict(status="submitting", binding=binding, binding_sha256=binding_sha256,
                     command=command, source_sha256=sources, campaign_sha256=binding["campaign_sha256"],
                     plan_sha256=binding["plan_sha256"], initial_evidence=evidence,
                     requested_gpu_hours=0, requested_gpu_seconds=0, requested_at_utc=checked.isoformat(),
                     scheduler_snapshot=snapshot, removed_scheduler_environment_keys=removed)
        ledger["jobs"][ROLE] = entry
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
            raise SubmissionError("Scheduler submission uncertain; reconcile manually before retry") from error
        entry.update(scheduler_stdout=completed.stdout[-2000:], scheduler_stderr=completed.stderr[-2000:],
                     scheduler_returncode=completed.returncode)
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", completed.stdout.strip())
        if completed.returncode != 0:
            entry["status"] = "submission_failed"
            if parsed:
                entry["possible_job_id"] = parsed.group(1)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Scheduler rejected/failed submission; reconcile, never resubmit")
        if parsed is None:
            entry["status"] = "uncertain"
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Ambiguous scheduler response; do not submit a duplicate")
        entry.update(status="submitted", job_id=parsed.group(1), submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        return entry["job_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(submit_labeled_feasibility(args.root))


if __name__ == "__main__":
    main()
