"""Metadata-only admission of one CPU development seal and joint native QA.

This admits no model, inference, training, GPU or downstream stage. Raw labeled
records and the old source pool are never opened here. The two existing ledger
entries must exactly match their archived native snapshot and remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess

from extension import labeled_submission as labeled
from extension import submission as metadata

ROLE = "development_seal_20261021_v1"
SCHEMA, DEADLINE = metadata.SCHEMA, metadata.DEADLINE
WALL_SECONDS, RESERVE_SECONDS = 7200, 172800
SubmissionError, require = metadata.SubmissionError, metadata.require
_canonical, _sha, _read = metadata._canonical, metadata._sha, metadata._read
LABELED_JOB_ID = "1222926"
EVIDENCE_DIRECTORY = "extension/report/labeled-feasibility-20261020-1222926"
LABELED_RUN_DIRECTORY = "extension/runs/labeled-feasibility-20261020-1222926"
LABELED_RECEIPT_SHA256 = {
    "accounting.json": "a93a348efb28bf6b43a0dabb154fe085eeeb98809c3adb4afb1574f024e38632",
    "complete.json": "6c225e606b0086649fa6d578d6a846f37594b893eefcee117ead22e8fb2c3458",
    "native_qa.json": "c5d12f7d0c9bf40ac80f5c685a5db289ff88e1699c6b1628afa1fab120db5945",
    "job_ledger_snapshot.json": "436b393022f8548360d70919debf7d67083c4882b76448f823c0436a65872e2f",
}
LABELED_OUTPUT_SHA256 = {
    "excluded.jsonl": "3bd9d05421e5eb221759a6347795513026ed3f0c3b4037db7b1baa071017144f",
    "groups.json": "a333abab42471a1f990e4ac670c1f0e0b56a48fe7aaee0930332cd1f525d4c0d",
    "quarantined.jsonl": "25c78446728aff2df00a44fc0825147d45cc1b135829b44aa4d98f3cb9861d4a",
    "surviving.jsonl": "f1304725d60087666fa2c3995fe7b4b344067ad28d1f2af35e8dbc5dab4e0551",
    "surviving_ids.json": "d2d6e77998f1cadfc79f418fa315eec88d8a6c97c9cb7120bb5c3de976cc5098",
}
POOL_SHA256 = "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0"
TARGETS = {"calibration": {"math": 16, "arc_challenge": 16},
           "development": {"math": 512, "arc_challenge": 256},
           "confirmation": {"math": 1024, "arc_challenge": 512}}
SOURCE_PATHS = frozenset(labeled.SOURCE_PATHS | {
    "extension/preserving_objectives.py", "extension/preserving_trainer.py",
    "tests/test_preserving_objectives.py", "tests/test_preserving_trainer.py",
    "extension/development_submission.py", "extension/development_seal.py",
    "extension/development_seal.slurm", "extension/development_plan.json",
    "tests/test_development_submission.py", "tests/test_development_seal.py",
})


def source_inventory(root):
    """Bind exactly 28 static files; never glob receipts, models or datasets."""
    root = Path(root).resolve()
    inventory, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        path = root / name
        require(path.is_file(), "Missing reviewed development source/test: " + name)
        value = metadata._small_bytes(path, root)
        total += len(value)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Development source metadata bound exceeded")
        inventory[name] = dict(sha256=_sha(value), bytes=len(value))
    return inventory


def validate_plan(value):
    require(isinstance(value, dict) and value.get("schema") == "extension-development-seal-plan-v1",
            "Expected the fixed development seal plan schema")
    require(value.get("cpu_only") is True and value.get("model_inference") is False and
            value.get("training") is False and value.get("automatic_downstream") is False and
            value.get("choose_splits") is True,
            "Only CPU split sealing and joint native QA are permitted")
    require(value.get("deadline_utc") == DEADLINE.isoformat() and
            type(value.get("wall_time_seconds")) is int and value["wall_time_seconds"] == WALL_SECONDS and
            type(value.get("reserve_seconds")) is int and value["reserve_seconds"] == RESERVE_SECONDS and
            type(value.get("seed")) is int and value["seed"] == 20261021 and value.get("targets") == TARGETS,
            "Unsealed development deadline, runtime/reserve, seed or target counts")
    audit = value.get("labeled_audit", {})
    require(isinstance(audit, dict) and audit.get("directory") == LABELED_RUN_DIRECTORY and
            audit.get("complete_sha256") == LABELED_RECEIPT_SHA256["complete.json"] and
            audit.get("qa_sha256") == LABELED_RECEIPT_SHA256["native_qa.json"],
            "Development must bind the exact completed labeled-data audit")
    require(value.get("pool") == {"path": "deadline/artifacts/pool.json", "sha256": POOL_SHA256},
            "Only the pinned historical source pool may be read by the CPU seal")


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
        require(option.startswith("--") and option[2:] in allowed, "GPU/unknown SBATCH option forbidden")
        key = option[2:]
        require(key not in directives, "Duplicate SBATCH option")
        if separator:
            require(len(parts) == 1, "Ambiguous SBATCH directive")
        else:
            require(len(parts) in (1, 2), "Ambiguous SBATCH directive")
            argument = parts[1] if len(parts) == 2 else ""
        directives[key] = argument
    expected = {"job-name": "sft-extension-development-seal", "account": "tml_2026",
                "partition": "normal_q", "qos": "owl_normal_short", "nodes": "1", "ntasks": "1",
                "cpus-per-task": "8", "mem": "64G", "time": "02:00:00", "no-requeue": ""}
    require(set(directives) == allowed and all(directives.get(key) == item for key, item in expected.items()),
            "Only the fixed 8-CPU/64G/two-hour/zero-GPU development seal is permitted")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and target.resolve().is_relative_to(Path(root) / "extension") and
                ".." not in target.parts, "Runtime logs must stay below extension")
    return directives


def initial_evidence(root):
    """Return eight small native receipt bindings and the two immutable jobs.

    Output-data hashes are validated from the pinned complete receipt. No output
    data file is opened or hashed; the compute-only sealer must verify its bytes.
    """
    root = Path(root).resolve()
    inventory, initial_job = labeled.initial_evidence(root)
    inventory, values = dict(inventory), {}
    for name, expected in LABELED_RECEIPT_SHA256.items():
        relative = EVIDENCE_DIRECTORY + "/" + name
        raw = metadata._small_bytes(root / relative, root)
        require(_sha(raw) == expected, "Labeled native evidence digest mismatch: " + name)
        inventory[relative] = dict(sha256=expected, bytes=len(raw))
        values[name] = _read(root / relative)
    account, complete = values["accounting.json"], values["complete.json"]
    qa, snapshot = values["native_qa.json"], values["job_ledger_snapshot.json"]
    require(account.get("schema") == "extension-terminal-cpu-accounting-v1" and account.get("complete") is True and
            account.get("job_id") == LABELED_JOB_ID and account.get("gpu_seconds") == 0 and
            account.get("no_gpu_in_alloc_tres") is True and account.get("elapsed_seconds") == 54 and
            account.get("stderr") == "" and isinstance(account.get("stdout"), str),
            "Labeled terminal 54-second zero-GPU accounting is required")
    rows = [line.split("|") for line in account["stdout"].splitlines() if line]
    require(rows and len({row[0] for row in rows}) == len(rows) and
            len([row for row in rows if row[0] == LABELED_JOB_ID]) == 1 and
            all(len(row) == 8 and row[0] in {LABELED_JOB_ID, LABELED_JOB_ID + ".batch", LABELED_JOB_ID + ".extern"} and
                row[2:5] == ["COMPLETED", "0:0", "54"] and row[5] and row[6] and
                "gpu" not in row[7].lower() for row in rows),
            "Labeled scheduler accounting is partial, failed or GPU allocated")
    require(snapshot.get("schema") == SCHEMA and snapshot.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
            snapshot.get("gpu_hour_cap", "missing") is None and isinstance(snapshot.get("jobs"), dict) and
            set(snapshot["jobs"]) == {metadata.ROLE, labeled.ROLE} and snapshot["jobs"][metadata.ROLE] == initial_job,
            "Archived two-stage ledger changed the original entry or roles")
    jobs = snapshot["jobs"]
    job = jobs[labeled.ROLE]
    require(job.get("status") == "submitted" and job.get("job_id") == LABELED_JOB_ID and
            job.get("requested_gpu_hours") == 0 and job.get("requested_gpu_seconds") == 0 and
            isinstance(job.get("binding"), dict) and job.get("binding_sha256") == _sha(_canonical(job["binding"])) and
            job["binding"].get("role") == labeled.ROLE and job["binding"].get("requested_gpu_seconds") == 0,
            "Archived labeled admission is invalid")
    require(qa.get("schema") == "extension-labeled-feasibility-native-qa-v1" and qa.get("job_id") == LABELED_JOB_ID and
            qa.get("success") is True and qa.get("gpu_disabled") is True and qa.get("wrapper_syntax_passed") is True and
            qa.get("tests_run") == 108 and qa.get("regression_tests") == 64 and qa.get("new_labeled_tests") == 44 and
            all(qa.get(key) == 0 for key in ("errors", "failures", "skipped", "expected_failures", "unexpected_successes")) and
            qa.get("compute_proof", {}).get("gpu_requested_or_allocated") is False and
            qa.get("admission_binding_sha256") == job["binding_sha256"] and
            qa.get("source_inventory") == job.get("source_sha256") and
            qa.get("code_sha256") == {name: item["sha256"] for name, item in qa.get("source_inventory", {}).items()} and
            labeled.source_inventory(root) == qa.get("source_inventory"),
            "Labeled QA108 must pass, bind its admission and preserve all 18 tested sources")
    require(complete.get("schema") == "extension-labeled-feasibility-v1" and complete.get("complete") is True and
            complete.get("gpu_seconds_requested_or_used") == 0 and complete.get("input_stats_unchanged_at_closeout") is True and
            complete.get("native_qa", {}).get("sha256") == LABELED_RECEIPT_SHA256["native_qa.json"] and
            complete.get("native_qa", {}).get("tests_run") == 108 and
            complete.get("native_qa", {}).get("code_sha256") == qa.get("code_sha256") and
            complete.get("output_sha256") == LABELED_OUTPUT_SHA256 and
            complete.get("actions") == {"splits_chosen": False, "generation": False, "training": False, "automatic_downstream": False} and
            all(complete.get(key, {}).get("job_id") == LABELED_JOB_ID and
                complete.get(key, {}).get("gpu_requested_or_allocated") is False for key in ("compute_start", "compute_end")),
            "Labeled completion/data hashes/CPU provenance are incomplete")
    return inventory, jobs


def submit_development_seal(root=Path("."), *, now=None, runner=None):
    """Admit only the fixed CPU wrapper; exact successful repeats return its ID."""
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
            "Real isolated extension directory required")
    runner = subprocess.run if runner is None else runner
    clock = (lambda: datetime.now(timezone.utc)) if now is None else (lambda: now)

    def guard():
        checked = clock()
        require(isinstance(checked, datetime) and checked.tzinfo is not None and
                (DEADLINE - checked).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
                "Insufficient time for two-hour CPU seal and 48-hour reserve")
        return checked

    checked = guard()
    sources = source_inventory(root)
    metadata.validate_campaign(_read(extension / "campaign.json"))
    validate_plan(_read(extension / "development_plan.json"))
    script = extension / "development_seal.slurm"
    directives = validate_wrapper(script, root)
    evidence, prior_jobs = initial_evidence(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(script)]
    binding = dict(role=ROLE, command=command, source_sha256=sources,
        campaign_sha256=sources["extension/campaign.json"]["sha256"],
        plan_sha256=sources["extension/development_plan.json"]["sha256"], initial_evidence=evidence,
        prior_binding_sha256={name: job["binding_sha256"] for name, job in prior_jobs.items()},
        labeled_output_sha256=LABELED_OUTPUT_SHA256, requested_gpu_hours=0, requested_gpu_seconds=0,
        wall_time_seconds=WALL_SECONDS, closeout_reserve_seconds=RESERVE_SECONDS, deadline_utc=DEADLINE.isoformat())
    checksum = _sha(_canonical(binding))
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "Extension ledger/lock cannot be symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(ledger_path.exists(), "Prior two-stage extension ledger required")
        ledger = _read(ledger_path)
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
                ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
                ledger.get("gpu_hour_cap", "missing") is None and isinstance(ledger.get("jobs"), dict) and
                set(ledger["jobs"]) <= set(prior_jobs) | {ROLE} and
                all(ledger["jobs"].get(name) == entry for name, entry in prior_jobs.items()),
                "Unknown role or changed prior entries; reconcile manually")
        snapshot = metadata._queue_snapshot(runner)
        require(not any(row["job_id"].split("_")[0] in {labeled.INITIAL_JOB_ID, LABELED_JOB_ID} or
                        row["name"] in {"sft-extension-feasibility", "sft-extension-labeled-feasibility"}
                        for row in snapshot["jobs"]), "Prior CPU stage is still visible or duplicated; reconcile")
        require(source_inventory(root) == sources and validate_wrapper(script, root) == directives and
                initial_evidence(root) == (evidence, prior_jobs), "Source/evidence changed during admission")
        existing = ledger["jobs"].get(ROLE)
        same_name = {row["job_id"] for row in snapshot["jobs"] if row["name"] == directives["job-name"]}
        if existing is not None:
            require(isinstance(existing, dict) and existing.get("status") == "submitted" and
                    existing.get("binding_sha256") == checksum and existing.get("binding") == binding and
                    existing.get("command") == command and existing.get("source_sha256") == sources and
                    existing.get("campaign_sha256") == binding["campaign_sha256"] and
                    existing.get("plan_sha256") == binding["plan_sha256"] and
                    existing.get("requested_gpu_hours") == 0 and existing.get("requested_gpu_seconds") == 0 and
                    isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", existing["job_id"]),
                    "Existing development submission changed, uncertain or failed; never resubmit")
            require(not (same_name - {existing["job_id"]}), "Untracked duplicate development job; reconcile")
            return existing["job_id"]
        require(not same_name, "Untracked live development seal job; reconcile")
        checked = guard()
        for key in ("output", "error"):
            Path(directives[key]).parent.mkdir(parents=True, exist_ok=True)
        environment, removed = metadata._submission_environment()
        entry = dict(status="submitting", binding=binding, binding_sha256=checksum, command=command,
            source_sha256=sources, campaign_sha256=binding["campaign_sha256"], plan_sha256=binding["plan_sha256"],
            initial_evidence=evidence, requested_gpu_hours=0, requested_gpu_seconds=0,
            requested_at_utc=checked.isoformat(), scheduler_snapshot=snapshot,
            removed_scheduler_environment_keys=removed)
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
            raise SubmissionError("Scheduler submission uncertain; reconcile before retry") from error
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
            raise SubmissionError("Ambiguous scheduler response; never submit a duplicate")
        entry.update(status="submitted", job_id=parsed.group(1), submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        return entry["job_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(submit_development_seal(args.root))


if __name__ == "__main__":
    main()
