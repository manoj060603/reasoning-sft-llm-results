"""One explicit CPU-only retry of the preserved source-readiness fixture failure.

All nine earlier ledger entries stay byte-for-byte equivalent. Exactly one
executed test fixture may change to its pinned repaired bytes; the other 66
executed sources, original plan, scorer and data/model scope are immutable.
Only small source and native metadata are read. No cache, dataset, model or GPU
stage is opened or admitted, and uncertain submissions are never resubmitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess

from extension import source_readiness_submission as previous
from extension import submission as metadata

ROLE, NAME = "source_readiness_20261021_v2", "sft-extension-source-readiness-v2"
PLAN, SCRIPT = previous.PLAN, "extension/source_readiness_retry.slurm"
SCHEMA, DEADLINE = metadata.SCHEMA, metadata.DEADLINE
WALL_SECONDS, RESERVE_SECONDS = 7200, 172800
SubmissionError, require = metadata.SubmissionError, metadata.require
_read, _sha, _canonical = metadata._read, metadata._sha, metadata._canonical
FAILED_ROLE, FAILED_JOB_ID = previous.ROLE, "1225252"
DIRECTORY = "extension/report/source-readiness-failed-20261021-1225252"
PLAN_SHA256 = "70c6823627d8bc93ce927665b87b943e1bf22f1ddc99eb2fd511a778696b590d"
FIXTURE = "tests/test_source_evaluation.py"
OLD_FIXTURE_SHA256 = "e1e2bcaa4b29dbe3abe57e8e55525caece7bbc900752df72d352a0ad04d0a33a"
NEW_FIXTURE_SHA256 = "3b4a9689a0b44312b81e898ef6722f96acd8bb752093134034743d26255ce1af"
RECEIPTS = {
    "unit_qa_failure.json": "ab1f54696de38a7d890a30094cd66d9cb574b5a0db3b0589ab07a84b8482debd",
    "accounting.json": "62337f57dae3e86f070198f458cdfc00fd8f3e4f2dc828848dfa418d7a56666b",
    "source_install_receipt.json": "516ab670d8a2241e4aaaa4fc466b9f62ea4fee6b78d1b612d836161ed0328209",
    "job_ledger_snapshot.json": "69ad79601473ced278fb8650db21cc9529f7cbec534216a794ca41fa943fc7c3",
    "executed_test_source_evaluation.py": OLD_FIXTURE_SHA256,
}
SOURCE_PATHS = frozenset(previous.SOURCE_PATHS | {
    "extension/source_readiness_retry_submission.py", "tests/test_source_readiness_retry_submission.py",
    "extension/source_readiness_retry_qa.py", SCRIPT})


def source_inventory(root):
    root = Path(root).resolve()
    result, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        raw = metadata._small_bytes(root / name, root)
        total += len(raw)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Retry source metadata exceeds its bound")
        result[name] = {"sha256": _sha(raw), "bytes": len(raw)}
    return result


def initial_evidence(root):
    """Return exactly 30 pinned public metadata/source assets and nine priors."""
    root = Path(root).resolve()
    evidence, eight = previous.initial_evidence(root)
    evidence, values = dict(evidence), {}
    for name, checksum in RECEIPTS.items():
        relative = DIRECTORY + "/" + name
        raw = metadata._small_bytes(root / relative, root)
        require(_sha(raw) == checksum, "Pinned failed-stage evidence changed: " + relative)
        evidence[relative] = {"sha256": checksum, "bytes": len(raw)}
        if name.endswith(".json"):
            values[name] = _read(root / relative)
    nine = previous.previous.previous._ledger(values["job_ledger_snapshot.json"], set(eight) | {FAILED_ROLE})
    require(all(nine.get(role) == entry for role, entry in eight.items()), "Failure archive changed earlier eight entries")
    entry = nine[FAILED_ROLE]
    require(isinstance(entry, dict) and entry.get("status") == "submitted" and entry.get("job_id") == FAILED_JOB_ID and
        entry.get("requested_gpu_hours") == 0 and entry.get("requested_gpu_seconds") == 0 and
        isinstance(entry.get("binding"), dict) and entry["binding"].get("role") == FAILED_ROLE and
        entry["binding"].get("requested_gpu_seconds") == 0 and
        entry.get("binding_sha256") == _sha(_canonical(entry["binding"])), "Invalid archived failed CPU admission")
    installed = values["source_install_receipt.json"]
    old = installed.get("source_inventory", {})
    require(installed.get("schema") == "extension-source-readiness-source-install-v1" and
        installed.get("installed_on_arc") is True and installed.get("expected_native_test_count") == 445 and
        isinstance(old, dict) and set(old) == set(previous.SOURCE_PATHS) and len(old) == 67 and
        old == entry.get("source_sha256") == entry["binding"].get("source_sha256") and
        old.get(FIXTURE, {}).get("sha256") == OLD_FIXTURE_SHA256 and
        old.get(PLAN, {}).get("sha256") == PLAN_SHA256,
        "Failed installation must bind the exact executed 67 sources and original plan")
    current = previous.source_inventory(root)
    require(set(current) == set(old) and all(current[name] == old[name] for name in old if name != FIXTURE) and
        current[FIXTURE]["sha256"] == NEW_FIXTURE_SHA256,
        "Only the explicitly pinned empty-gold fixture repair may change; all other 66 sources are immutable")
    qa = values["unit_qa_failure.json"]
    require(qa.get("schema") == "extension-source-readiness-native-qa-v1" and qa.get("success") is False and
        qa.get("job_id") == FAILED_JOB_ID and qa.get("tests_run") == 445 and qa.get("failures") == 1 and
        type(qa.get("errors")) is int and qa["errors"] == 0 and type(qa.get("skipped")) is int and qa["skipped"] == 0 and
        qa.get("no_automatic_retry") is True and qa.get("code_sha256") ==
            {name: record["sha256"] for name, record in old.items()}, "Preserve the exact 445-check single-fixture failure")
    accounting = values["accounting.json"]
    require(accounting.get("schema") == "extension-terminal-source-readiness-failure-v1" and
        accounting.get("job_id") == FAILED_JOB_ID and accounting.get("complete") is False and
        accounting.get("terminal") is True and accounting.get("elapsed_seconds") == 46 and accounting.get("gpu_seconds") == 0 and
        accounting.get("tests_run") == 445 and accounting.get("failures") == 1 and accounting.get("errors") == 0 and
        accounting.get("skipped") == 0 and accounting.get("calibration_prepared") is False and
        accounting.get("model_or_benchmark_outcomes_viewed") is False and accounting.get("fixture_only_repair_required") is True and
        accounting.get("failure_kind") == "fixture_constructed_already_quarantined_empty_gold" and
        accounting.get("stderr") == "" and isinstance(accounting.get("stdout"), str),
        "Terminal zero-GPU failure without prepared calibration/model outcomes is required")
    rows = [line.split("|") for line in accounting["stdout"].splitlines() if line]
    require(len(rows) == 3 and {row[0] for row in rows} == {FAILED_JOB_ID, FAILED_JOB_ID + ".batch", FAILED_JOB_ID + ".extern"} and
        all(len(row) == 8 and row[4] == "46" and "gpu" not in row[7].lower() and
            row[2:4] == (["COMPLETED", "0:0"] if row[0].endswith(".extern") else ["FAILED", "1:0"])
            for row in rows), "Exact FAILED1:0 allocation/batch and completed extern accounting required")
    return evidence, nine


def validate_wrapper(path, root):
    text = metadata._small_bytes(path, root).decode("utf-8")
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "normal_q", "qos": "owl_normal_short",
        "nodes": "1", "ntasks": "1", "cpus-per-task": "8", "mem": "64G", "time": "02:00:00", "no-requeue": ""}
    allowed, directives, header = set(expected) | {"output", "error"}, {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"):
                header = False
            continue
        require(header, "SBATCH directive after executable code")
        parts = shlex.split(line.lstrip()[7:].strip(), comments=True)
        require(bool(parts), "Empty SBATCH directive")
        option, separator, argument = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in allowed and option[2:] not in directives and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/duplicated/ambiguous SBATCH option")
        directives[option[2:]] = argument if separator else (parts[1] if len(parts) == 2 else "")
    require(set(directives) == allowed and all(directives.get(key) == value for key, value in expected.items()),
        "Only the exact two-hour zero-GPU 8CPU/64G retry wrapper is permitted")
    for key in ("output", "error"):
        path = Path(directives[key])
        require(path.is_absolute() and path.resolve().is_relative_to(Path(root) / "extension") and
            ".." not in path.parts, "Runtime logs must stay below extension")
    return directives


def check_queue(snapshot, nine):
    previous.check_queue(snapshot, {role: entry for role, entry in nine.items() if role != FAILED_ROLE})
    for row in snapshot["jobs"]:
        if row["job_id"] == FAILED_JOB_ID:
            require((row["state"], row["name"]) == ("FAILED", previous.NAME), "Failed prior CPU job is active/unknown or renamed")
        elif row["name"] == previous.NAME or row["job_id"].split("_")[0] == FAILED_JOB_ID:
            raise SubmissionError("Untracked duplicate failed source-readiness job")


def submit(root=Path("."), *, now=None, runner=None):
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
        "Real isolated extension directory required")
    runner = subprocess.run if runner is None else runner
    def guard():
        when = datetime.now(timezone.utc) if now is None else now
        require(isinstance(when, datetime) and when.tzinfo is not None and
            (DEADLINE - when).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
            "Keep full retry work time plus 48-hour deadline reserve")
        return when
    guard()
    sources = source_inventory(root)
    require(sources[PLAN]["sha256"] == PLAN_SHA256, "The original CPU preparation plan must remain byte-identical")
    metadata.validate_campaign(_read(extension / "campaign.json"))
    previous.validate_plan(_read(root / PLAN), sources)
    directives = validate_wrapper(root / SCRIPT, root)
    evidence, nine = initial_evidence(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    repair = {"path": FIXTURE, "before_sha256": OLD_FIXTURE_SHA256, "after_sha256": NEW_FIXTURE_SHA256}
    binding = {"role": ROLE, "command": command, "source_sha256": sources, "plan_sha256": PLAN_SHA256,
        "campaign_sha256": sources["extension/campaign.json"]["sha256"], "initial_evidence": evidence,
        "prior_binding_sha256": {role: entry["binding_sha256"] for role, entry in nine.items()},
        "failed_job_id": FAILED_JOB_ID, "fixture_only_repair": repair,
        "requested_gpu_hours": 0, "requested_gpu_seconds": 0, "wall_time_seconds": WALL_SECONDS,
        "closeout_reserve_seconds": RESERVE_SECONDS, "deadline_utc": DEADLINE.isoformat(), "automatic_downstream": False}
    checksum = _sha(_canonical(binding))
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "No ledger/lock symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(ledger_path)
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
            ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and ledger.get("gpu_hour_cap", "missing") is None and
            isinstance(ledger.get("jobs"), dict) and set(ledger["jobs"]) <= set(nine) | {ROLE} and
            all(ledger["jobs"].get(role) == entry for role, entry in nine.items()),
            "Preserve all nine entries exactly; unknown or changed jobs block retry admission")
        snapshot = metadata._queue_snapshot(runner)
        check_queue(snapshot, nine)
        require(source_inventory(root) == sources and initial_evidence(root) == (evidence, nine) and
            validate_wrapper(root / SCRIPT, root) == directives, "Source/evidence changed during retry admission")
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        existing = ledger["jobs"].get(ROLE)
        if existing is not None:
            require(isinstance(existing, dict) and existing.get("status") == "submitted" and
                existing.get("binding") == binding and existing.get("binding_sha256") == checksum and
                existing.get("source_sha256") == sources and existing.get("command") == command and
                existing.get("requested_gpu_hours") == 0 and existing.get("requested_gpu_seconds") == 0 and
                isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", existing["job_id"]) and
                not same - {existing["job_id"]}, "Existing retry uncertain/failed/changed; never resubmit")
            return existing["job_id"]
        require(not same, "Untracked current retry; reconcile before admission")
        when = guard()
        for key in ("output", "error"):
            Path(directives[key]).parent.mkdir(parents=True, exist_ok=True)
        environment, removed = metadata._submission_environment()
        entry = {"status": "submitting", "binding": binding, "binding_sha256": checksum, "command": command,
            "source_sha256": sources, "plan_sha256": PLAN_SHA256, "campaign_sha256": binding["campaign_sha256"],
            "initial_evidence": evidence, "requested_gpu_hours": 0, "requested_gpu_seconds": 0,
            "requested_at_utc": when.isoformat(), "scheduler_snapshot": snapshot,
            "removed_scheduler_environment_keys": removed}
        ledger["jobs"][ROLE] = entry
        metadata._write(ledger_path, ledger)
        try:
            response = runner(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status="uncertain", error_type=type(error).__name__)
            for name in ("stdout", "stderr"):
                value = getattr(error, name, None)
                if value:
                    entry["scheduler_" + name] = (value.decode(errors="replace") if isinstance(value, bytes) else str(value))[-2000:]
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Retry submission uncertain; no automatic resubmission") from error
        entry.update(scheduler_returncode=response.returncode, scheduler_stdout=response.stdout[-2000:], scheduler_stderr=response.stderr[-2000:])
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip())
        if response.returncode != 0 or parsed is None:
            entry["status"] = "submission_failed" if response.returncode else "uncertain"
            if parsed:
                entry["possible_job_id"] = parsed.group(1)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Failed/ambiguous retry submission; preserve it without resubmission")
        entry.update(status="submitted", job_id=parsed.group(1), submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        return entry["job_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    print(submit(parser.parse_args().root))


if __name__ == "__main__":
    main()
