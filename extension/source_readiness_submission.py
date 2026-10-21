"""One finite metadata-only CPU source-readiness admission, never a GPU adapter.

The prior eight journal entries, 51 tested sources and 25 parent metadata files
are immutable prerequisites. No raw cohort, prompt cache, tokenizer or model is
opened here. The already-admitted B200 diagnostic may stay queued or running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess

from extension import smoke_submission as previous
from extension import submission as metadata

ROLE, NAME = "source_readiness_20261021_v1", "sft-extension-source-readiness"
PLAN, SCRIPT = "extension/source_readiness_plan.json", "extension/source_readiness.slurm"
SCHEMA, DEADLINE = metadata.SCHEMA, metadata.DEADLINE
WALL_SECONDS, RESERVE_SECONDS = 7200, 172800
SubmissionError, require = metadata.SubmissionError, metadata.require
_read, _sha, _canonical = metadata._read, metadata._sha, metadata._canonical
DIRECTORY = "extension/report/native-smoke-preparation-20261021-1224497"
RECEIPTS = {
    "complete.json": "b7d8dc41b6fd90befc6a85130b3904245aea6dbd2cafc3209941d647a3ab8df5",
    "native_qa.json": "351368511546e4c8d7b897c6340c057e7595a2dde0afc5b26b655c62d57a3b6e",
    "unit_qa.json": "4d900d4ef8b5b706cfc89e2e3fc4017eab3338bb4efd4e7e83e15836efaa3497",
    "accounting.json": "4d670f1fc86ab88c10da7270c94262d3bdd023061320d868b0a61852850aa1a2",
    "benchmark_audit.json": "2671a58d0abf9f52de5445649aecf9a389363f0fe54d8f05dbf5bca243c6fa6f",
    "job_ledger_snapshot.json": "9e8caec6473d6ec8cce39f581c4084ee9f866403a3e1fee404156a6dd4624d00"}
SOURCE_PATHS = frozenset(previous.SOURCE_PATHS | {
    "extension/source_evaluation.py", "tests/test_source_evaluation.py", "extension/pilot_driver.py", "tests/test_pilot_driver.py",
    "extension/source_readiness.py", "tests/test_source_readiness.py", "extension/source_readiness_submission.py",
    "tests/test_source_readiness_submission.py", "extension/source_readiness.slurm", "extension/source_readiness_plan.json",
    "extension/source_readiness_qa.py",
    "eval/utils/grader.py", "eval/utils/parser.py", "eval/utils/math_normalization.py", "eval/utils/utils.py", "eval/utils/examples.py"})


def source_inventory(root):
    root = Path(root).resolve()
    inventory, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        raw = metadata._small_bytes(root / name, root)
        total += len(raw)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Source readiness metadata inventory exceeds its bound")
        inventory[name] = {"sha256": _sha(raw), "bytes": len(raw)}
    return inventory


def initial_evidence(root):
    root = Path(root).resolve()
    evidence, six = previous.initial_evidence(root)
    evidence = dict(evidence)
    values = previous.previous._receipts(root, DIRECTORY, RECEIPTS, evidence)
    eight = previous.previous._ledger(values["job_ledger_snapshot.json"], set(six) | {previous.CPU_ROLE, previous.GPU_ROLE})
    require(all(eight.get(role) == entry for role, entry in six.items()), "The eight-entry archive changed earlier history")
    old = previous.source_inventory(root)
    for role, job, seconds in ((previous.CPU_ROLE, "1224497", 0), (previous.GPU_ROLE, "1224577", 7200)):
        entry = eight[role]
        require(entry.get("status") == "submitted" and entry.get("job_id") == job and
            entry.get("source_sha256") == old and entry.get("requested_gpu_seconds") == seconds and
            entry.get("requested_gpu_hours") == seconds / 3600 and isinstance(entry.get("binding"), dict) and
            entry["binding"].get("source_sha256") == old and entry["binding"].get("role") == role and
            entry.get("binding_sha256") == _sha(_canonical(entry["binding"])), "Changed archived native-stage admission")
    require(eight[previous.GPU_ROLE]["binding"].get("cpu_admission_binding_sha256") ==
        eight[previous.CPU_ROLE]["binding_sha256"] and
        eight[previous.GPU_ROLE]["binding"].get("cpu_preparation", {}).get("job_id") == "1224497",
        "B200 admission no longer binds its complete CPU predecessor")
    qa, complete, unit = (values[name] for name in ("native_qa.json", "complete.json", "unit_qa.json"))
    for value, schema in ((qa, "extension-native-model-admission-native-qa-v1"),
                         (unit, "extension-native-smoke-preparation-unit-qa-v1")):
        require(value.get("schema") == schema and value.get("success") is True and value.get("tests_run") == 341 and
            value.get("source_inventory") == old and value.get("code_sha256") ==
            {name: entry["sha256"] for name, entry in old.items()} and value.get("gpu_disabled") is True and
            value.get("job_id") == "1224497" and all(type(value.get(name)) is int and value[name] == 0 for name in previous.ZERO_FIELDS) and
            value.get("admission_binding_sha256") == eight[previous.CPU_ROLE]["binding_sha256"] and
            value.get("compute_proof", {}).get("gpu_requested_or_allocated") is False,
            "All 51 prior sources and complete native QA341 must remain unchanged")
    require(complete.get("schema") == "extension-native-smoke-preparation-complete-v1" and complete.get("complete") is True and
        complete.get("job_id") == "1224497" and complete.get("cpu_only") is True and complete.get("gpu_seconds") == 0 and
        complete.get("input_snapshot_unchanged") is True and complete.get("source_inventory") == old and
        complete.get("native_qa", {}).get("sha256") == RECEIPTS["native_qa.json"] and
        complete.get("unit_qa", {}).get("sha256") == RECEIPTS["unit_qa.json"], "Completed native CPU source audit/cache proof is required")
    benchmark = values["benchmark_audit.json"]
    require(benchmark.get("schema") == "extension-final-benchmark-pool-exclusions-v1" and benchmark.get("complete") is True and
        benchmark.get("scope") == "all_nine_final_benchmarks" and benchmark.get("pool_rows_scanned") == 83029 and
        benchmark.get("criterion") == previous.previous.CRITERION and benchmark.get("answers_or_outcomes_used") is False,
        "The completed full-pool all-nine question-only audit is required")
    previous._accounting(values["accounting.json"], "1224497", elapsed=215)
    return evidence, eight


def validate_plan(value, sources):
    require(isinstance(value, dict) and value.get("schema") == "extension-source-readiness-plan-v1" and
        value.get("cpu_only") is True and value.get("prepare_calibration_prompts") is True and
        all(value.get(name) is False for name in ("training", "model_inference", "automatic_downstream")) and
        value.get("wall_time_seconds") == WALL_SECONDS and value.get("reserve_seconds") == RESERVE_SECONDS and
        value.get("deadline_utc") == DEADLINE.isoformat(), "Only the fixed finite CPU source-readiness stage is permitted")
    require(value.get("prior_cpu") == {name: {"path": DIRECTORY + "/" + name, "sha256": checksum}
        for name, checksum in RECEIPTS.items()}, "Bind the six exact native CPU predecessor receipts")
    grader = value.get("grader_source", {})
    require(set(grader) == {"grader.py", "parser.py", "math_normalization.py", "utils.py", "examples.py"} and
        all(grader[name] == {"path": "eval/utils/" + name, "sha256": sources["eval/utils/" + name]["sha256"]}
            for name in grader), "Pin all actual local grader dependencies")


def validate_wrapper(path, root):
    # This separate fixed role preserves the reviewed two-hour CPU resources.
    text = metadata._small_bytes(path, root).decode("utf-8")
    import shlex
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
    require(set(directives) == allowed and all(directives.get(key) == item for key, item in expected.items()),
        "Only reviewed zero-GPU 8CPU/64G/two-hour directives are permitted")
    for name in ("output", "error"):
        path = Path(directives[name])
        require(path.is_absolute() and path.resolve().is_relative_to(Path(root) / "extension") and
            ".." not in path.parts, "Runtime logs must stay inside extension")
    return directives


def check_queue(snapshot, eight):
    six = {role: entry for role, entry in eight.items() if role not in (previous.CPU_ROLE, previous.GPU_ROLE)}
    previous._check_prior_queue(snapshot, six)
    for row in snapshot["jobs"]:
        if row["job_id"] == "1224497":
            require((row["state"], row["name"]) == ("COMPLETED", previous.CPU_NAME), "Prior CPU stage must remain terminal")
        elif row["name"] == previous.CPU_NAME or row["job_id"].split("_")[0] == "1224497":
            raise SubmissionError("Untracked duplicate prior CPU stage")
        if row["job_id"] == "1224577":
            require(row["name"] == previous.GPU_NAME, "Admitted diagnostic scheduler identity changed")
        elif row["name"] == previous.GPU_NAME or row["job_id"].split("_")[0] == "1224577":
            raise SubmissionError("Untracked duplicate B200 diagnostic")


def submit(root=Path("."), *, now=None, runner=None):
    root = Path(root).resolve()
    extension = root / "extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension, "Real isolated extension directory required")
    runner = subprocess.run if runner is None else runner
    def guard():
        when = datetime.now(timezone.utc) if now is None else now
        require(isinstance(when, datetime) and when.tzinfo is not None and
            (DEADLINE - when).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS, "Keep full work time plus 48-hour reserve")
        return when
    guard()
    sources = source_inventory(root)
    metadata.validate_campaign(_read(extension / "campaign.json"))
    validate_plan(_read(root / PLAN), sources)
    directives = validate_wrapper(root / SCRIPT, root)
    evidence, eight = initial_evidence(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    binding = {"role": ROLE, "command": command, "source_sha256": sources, "plan_sha256": sources[PLAN]["sha256"],
        "campaign_sha256": sources["extension/campaign.json"]["sha256"], "initial_evidence": evidence,
        "prior_binding_sha256": {role: entry["binding_sha256"] for role, entry in eight.items()},
        "requested_gpu_hours": 0, "requested_gpu_seconds": 0, "wall_time_seconds": WALL_SECONDS,
        "closeout_reserve_seconds": RESERVE_SECONDS, "deadline_utc": DEADLINE.isoformat(), "automatic_downstream": False}
    checksum = _sha(_canonical(binding))
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "No ledger/lock symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(ledger_path)
        require(ledger.get("schema") == SCHEMA and ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
            ledger.get("gpu_hour_cap", "missing") is None and isinstance(ledger.get("jobs"), dict) and
            set(ledger["jobs"]) <= set(eight) | {ROLE} and all(ledger["jobs"].get(role) == entry for role, entry in eight.items()),
            "Preserve all eight entries; unknown or changed jobs block admission")
        snapshot = metadata._queue_snapshot(runner)
        check_queue(snapshot, eight)
        require(source_inventory(root) == sources and initial_evidence(root) == (evidence, eight) and
            validate_wrapper(root / SCRIPT, root) == directives, "Source/evidence changed during CPU admission")
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        existing = ledger["jobs"].get(ROLE)
        if existing is not None:
            require(existing.get("status") == "submitted" and existing.get("binding") == binding and
                existing.get("binding_sha256") == checksum and existing.get("source_sha256") == sources and
                isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", existing["job_id"]) and
                not same - {existing["job_id"]}, "Existing role uncertain/changed; never resubmit")
            return existing["job_id"]
        require(not same, "Untracked current CPU job; reconcile before admission")
        when = guard()
        for key in ("output", "error"):
            Path(directives[key]).parent.mkdir(parents=True, exist_ok=True)
        environment, removed = metadata._submission_environment()
        entry = {"status": "submitting", "binding": binding, "binding_sha256": checksum, "command": command,
            "source_sha256": sources, "plan_sha256": binding["plan_sha256"], "campaign_sha256": binding["campaign_sha256"],
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
            raise SubmissionError("Submission uncertain; no automatic retry") from error
        entry.update(scheduler_returncode=response.returncode, scheduler_stdout=response.stdout[-2000:], scheduler_stderr=response.stderr[-2000:])
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip())
        if response.returncode != 0 or parsed is None:
            entry["status"] = "submission_failed" if response.returncode else "uncertain"
            if parsed:
                entry["possible_job_id"] = parsed.group(1)
            metadata._write(ledger_path, ledger)
            raise SubmissionError("Failed/ambiguous CPU submission; preserve it, no automatic retry")
        entry.update(status="submitted", job_id=parsed.group(1), submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        return entry["job_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    print(submit(parser.parse_args().root))


if __name__ == "__main__":
    main()
