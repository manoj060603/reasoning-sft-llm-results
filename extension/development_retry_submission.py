"""Fixed metadata-only CPU retry after the evidenced development seal failure.

The original 28 sources and five prior ledger entries are immutable. This module
cannot admit anything until exact diagnostic evidence is pinned below. It never
opens raw records, pool data, model files or the historical deadline ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess

from extension import development_submission as previous
from extension import submission as metadata

ROLE = "development_seal_20261021_v2"
REJECTED_ROLE = "development_level_diagnostic_20261021_v1"
DIAGNOSTIC_ROLE = "development_level_diagnostic_20261021_v2"
FAILED_ROLE, FAILED_JOB_ID = previous.ROLE, "1223566"
SCHEMA, DEADLINE = metadata.SCHEMA, metadata.DEADLINE
WALL_SECONDS, RESERVE_SECONDS = 7200, 172800
SubmissionError, require = metadata.SubmissionError, metadata.require
_canonical, _sha, _read = metadata._canonical, metadata._sha, metadata._read
FAILED_DIRECTORY = "extension/report/development-seal-failed-20261021-1223566"
FAILED_RECEIPT_SHA256 = {
    "native_qa.json": "cc64b1a5bc9a2f2778adab28d0460a645a6a67b821a14326fe008edc07f4c9cb",
    "development-seal-1223566-failure.json": "62072e4644341ccf315525646ebc3c9a9025425586c3a22c1226f6a281aab8ac",
    "job_ledger_snapshot.json": "efb72a659abc654eddbb79f2096738f83ec835435fa423fe8a08eb2ae8436525",
}
# Exact metadata from the delivered finite CPU diagnostic, never model outcomes.
DIAGNOSTIC_JOB_ID = "1223713"
DIAGNOSTIC_DIRECTORY = "extension/report/development-level-diagnostic-20261021-1223713"
DIAGNOSTIC_RUN_DIRECTORY = "extension/runs/development-level-diagnostic-20261021-1223713"
DIAGNOSTIC_RECEIPT_SHA256 = {
    "complete.json": "7944215f48b8bd12ba543fe5cf4b543ef1c21cfc674fe35ea4b17c49698cf96c",
    "accounting.json": "6f8316eb7aabe984e4b934f40712a5dd876812f8d4ebf43e65598300d841a266",
    "job_ledger_snapshot.json": "905d68cbeccfb05dc77a3b30970379396f46a0609c3f359ee32e84a200a81d35",
    "development-diagnostic-dns-rejection-20261021.json": "691be4b92f20a78d135a733124de3cb9a7a5201da51c5fa8f3a83217c93fa81d",
}
OBSERVED_UNKNOWN_LEVELS = {"Level ?": 1}
CRITERION = {"normalization": "NFKC casefold whitespace; preserve numbers/operators",
    "tokenization": "word-or-single-nonwhitespace-character", "shingle_tokens": 3,
    "jaccard_numerator": 4, "jaccard_denominator": 5, "search": "exact inverted index",
    "short_question_policy": "exact normalized equality when fewer than3tokens"}
SOURCE_PATHS = frozenset(previous.SOURCE_PATHS | {
    "extension/operations/development_metadata_diagnostic.slurm",
    "extension/operations/submit_development_diagnostic.py",
    "extension/operations/development_metadata_diagnostic_v2.slurm",
    "extension/operations/submit_development_diagnostic_v2.py",
    "extension/development_retry_submission.py", "tests/test_preservation_retry_submission.py",
    "extension/development_seal_v2.py", "tests/test_preservation_development_seal_v2.py",
    "extension/development_retry.slurm", "extension/development_retry_plan.json",
})


def _ready():
    require(isinstance(DIAGNOSTIC_JOB_ID, str) and re.fullmatch(r"[1-9][0-9]*", DIAGNOSTIC_JOB_ID) and
            DIAGNOSTIC_JOB_ID not in {"1222225", "1222926", FAILED_JOB_ID} and
            isinstance(DIAGNOSTIC_DIRECTORY, str) and DIAGNOSTIC_DIRECTORY.startswith("extension/report/") and
            set(DIAGNOSTIC_RECEIPT_SHA256) == {"complete.json", "accounting.json", "job_ledger_snapshot.json",
                "development-diagnostic-dns-rejection-20261021.json"} and
            all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in DIAGNOSTIC_RECEIPT_SHA256.values()) and
            isinstance(OBSERVED_UNKNOWN_LEVELS, dict) and bool(OBSERVED_UNKNOWN_LEVELS) and
            all(isinstance(level, str) and level and type(count) is int and count > 0
                for level, count in OBSERVED_UNKNOWN_LEVELS.items()),
            "Retry is unready: exact native diagnostic pins/job/observed levels must be sealed first")


def source_inventory(root):
    root = Path(root).resolve()
    inventory, total = {}, 0
    for name in sorted(SOURCE_PATHS):
        path = root / name
        require(path.is_file(), "Missing reviewed retry source/test: " + name)
        value = metadata._small_bytes(path, root)
        total += len(value)
        require(total <= metadata.MAX_TOTAL_SOURCE_BYTES, "Retry source metadata bound exceeded")
        inventory[name] = dict(sha256=_sha(value), bytes=len(value))
    return inventory


def validate_plan(value):
    _ready()
    require(isinstance(value, dict) and value.get("schema") == "extension-development-seal-plan-v2",
            "Expected the fixed development retry plan schema")
    require(value.get("cpu_only") is True and value.get("model_inference") is False and
            value.get("training") is False and value.get("automatic_downstream") is False and
            value.get("choose_splits") is True, "Only the fixed CPU development retry is permitted")
    require(value.get("deadline_utc") == DEADLINE.isoformat() and
            type(value.get("wall_time_seconds")) is int and value["wall_time_seconds"] == WALL_SECONDS and
            type(value.get("reserve_seconds")) is int and value["reserve_seconds"] == RESERVE_SECONDS and
            type(value.get("seed")) is int and value["seed"] == 20261021 and value.get("targets") == previous.TARGETS and
            value.get("near_duplicate") == CRITERION,
            "Retry must preserve exact targets/seed/criterion/deadline/runtime/reserve")
    labeled = value.get("labeled_audit", {})
    require(isinstance(labeled, dict) and labeled.get("directory") == previous.LABELED_RUN_DIRECTORY and
            labeled.get("complete_sha256") == previous.LABELED_RECEIPT_SHA256["complete.json"] and
            labeled.get("qa_sha256") == previous.LABELED_RECEIPT_SHA256["native_qa.json"] and
            value.get("pool") == {"path": "deadline/artifacts/pool.json", "sha256": previous.POOL_SHA256},
            "Retry cannot change its labeled population or training pool")
    diagnostic = value.get("level_diagnostic", {})
    require(isinstance(diagnostic, dict) and diagnostic.get("job_id") == DIAGNOSTIC_JOB_ID and
            diagnostic.get("directory") == DIAGNOSTIC_RUN_DIRECTORY and
            diagnostic.get("complete_sha256") == DIAGNOSTIC_RECEIPT_SHA256["complete.json"] and
            value.get("math_unknown_levels") == OBSERVED_UNKNOWN_LEVELS,
            "Retry must bind the exact metadata diagnostic and observed unknown-level values")


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
    expected = {"job-name": "sft-extension-development-seal-v2", "account": "tml_2026",
                "partition": "normal_q", "qos": "owl_normal_short", "nodes": "1", "ntasks": "1",
                "cpus-per-task": "8", "mem": "64G", "time": "02:00:00", "no-requeue": ""}
    require(set(directives) == allowed and all(directives.get(key) == item for key, item in expected.items()),
            "Only the fixed 8-CPU/64G/two-hour/zero-GPU retry wrapper is permitted")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and target.resolve().is_relative_to(Path(root) / "extension") and
                ".." not in target.parts, "Runtime logs must stay below extension")
    return directives


def _receipts(root, directory, pins, inventory):
    values = {}
    for name, expected in pins.items():
        relative = directory + "/" + name
        raw = metadata._small_bytes(root / relative, root)
        require(_sha(raw) == expected, "Retry parent evidence digest mismatch: " + relative)
        inventory[relative] = dict(sha256=expected, bytes=len(raw))
        values[name] = _read(root / relative)
    return values


def _ledger(value, expected_roles):
    require(isinstance(value, dict) and value.get("schema") == SCHEMA and
            value.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
            value.get("gpu_hour_cap", "missing") is None and isinstance(value.get("jobs"), dict) and
            set(value["jobs"]) == set(expected_roles), "Unexpected archived retry-parent ledger")
    return value["jobs"]


def _job(entry, role, identifier):
    require(isinstance(entry, dict) and entry.get("status") == "submitted" and entry.get("job_id") == identifier and
            entry.get("requested_gpu_hours") == 0 and entry.get("requested_gpu_seconds") == 0 and
            isinstance(entry.get("binding"), dict) and entry["binding"].get("role") == role and
            entry["binding"].get("requested_gpu_seconds") == 0 and
            entry.get("binding_sha256") == _sha(_canonical(entry["binding"])), "Invalid archived CPU admission")


def initial_evidence(root):
    """Return pinned receipts and all five exact immutable parent entries."""
    _ready()
    root = Path(root).resolve()
    inventory, first_two = previous.initial_evidence(root)
    inventory = dict(inventory)
    failed = _receipts(root, FAILED_DIRECTORY, FAILED_RECEIPT_SHA256, inventory)
    qa, failure = failed["native_qa.json"], failed["development-seal-1223566-failure.json"]
    three = _ledger(failed["job_ledger_snapshot.json"], set(first_two) | {FAILED_ROLE})
    require(all(three[name] == entry for name, entry in first_two.items()), "Failure archive changed earlier entries")
    _job(three[FAILED_ROLE], FAILED_ROLE, FAILED_JOB_ID)
    old_sources = previous.source_inventory(root)
    require(qa.get("schema") == "extension-development-seal-native-qa-v1" and qa.get("job_id") == FAILED_JOB_ID and
            qa.get("success") is True and qa.get("tests_run") == 192 and qa.get("gpu_disabled") is True and
            qa.get("wrapper_syntax_passed") is True and qa.get("compute_proof", {}).get("gpu_requested_or_allocated") is False and
            all(qa.get(key) == 0 for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
            qa.get("source_inventory") == old_sources == three[FAILED_ROLE].get("source_sha256") and
            qa.get("code_sha256") == {name: item["sha256"] for name, item in old_sources.items()} and
            qa.get("admission_binding_sha256") == three[FAILED_ROLE]["binding_sha256"],
            "Failed stage must preserve all 28 sources and its successful native QA192")
    require(failure.get("schema") == "extension-terminal-cpu-failure-v1" and failure.get("job_id") == FAILED_JOB_ID and
            failure.get("complete") is False and failure.get("terminal") is True and failure.get("elapsed_seconds") == 45 and
            failure.get("gpu_seconds") == 0 and failure.get("native_qa_sha256") == FAILED_RECEIPT_SHA256["native_qa.json"] and
            failure.get("tests_passed") == 192 and failure.get("splits_chosen") is False and
            failure.get("scientific_outcomes_viewed") is False and failure.get("stderr") == "" and
            isinstance(failure.get("stdout"), str), "The exact terminal failure evidence is required")
    rows = [line.split("|") for line in failure["stdout"].splitlines() if line]
    require(len(rows) == 3 and {row[0] for row in rows} ==
            {FAILED_JOB_ID, FAILED_JOB_ID + ".batch", FAILED_JOB_ID + ".extern"} and
            all(len(row) == 8 and row[4] == "45" and "gpu" not in row[7].lower() and
                row[2:4] == (["COMPLETED", "0:0"] if row[0].endswith(".extern") else ["FAILED", "1:0"])
                for row in rows), "Failed stage allocation proof is ambiguous or not zero-GPU FAILED1:0")
    diagnostic = _receipts(root, DIAGNOSTIC_DIRECTORY, DIAGNOSTIC_RECEIPT_SHA256, inventory)
    five = _ledger(diagnostic["job_ledger_snapshot.json"], set(three) | {REJECTED_ROLE, DIAGNOSTIC_ROLE})
    require(all(five[name] == entry for name, entry in three.items()), "Diagnostic archive changed earlier entries")
    rejected = five[REJECTED_ROLE]
    rejection = diagnostic["development-diagnostic-dns-rejection-20261021.json"]
    expected_error = ("sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host\n"
        "sbatch: error: fetch_config: DNS SRV lookup failed\n"
        "sbatch: error: _establish_config_source: failed to fetch config\n"
        "sbatch: fatal: Could not establish a configuration source\n")
    require(rejected.get("status") == "uncertain" and rejected.get("scheduler_returncode") == 1 and
            rejected.get("scheduler_stdout") == "" and rejected.get("scheduler_stderr") == expected_error and
            "job_id" not in rejected and "possible_job_id" not in rejected and
            rejected.get("requested_gpu_hours") == 0 and rejected.get("requested_gpu_seconds") == 0 and
            isinstance(rejected.get("binding"), dict) and rejected["binding"].get("role") == REJECTED_ROLE and
            rejected.get("binding_sha256") == _sha(_canonical(rejected["binding"])) and
            rejected["binding"].get("prior_jobs_sha256") == _sha(_canonical(three)) and
            rejected["binding"].get("source_sha256") == old_sources,
            "Only the exact no-ID DNS rejection may remain an uncertain parent entry")
    require(rejection.get("schema") == "extension-diagnostic-pre-scheduler-rejection-v1" and
            rejection.get("complete") is True and rejection.get("role") == REJECTED_ROLE and
            rejection.get("original_entry") == rejected and
            rejection.get("original_entry_sha256") == _sha(_canonical(rejected)) and
            rejection.get("gpu_seconds") == 0 and rejection.get("scheduler_snapshot", {}).get("jobs") == [],
            "Pinned DNS reconciliation must preserve the exact rejected entry and empty queue evidence")
    for field, relative in (("wrapper_sha256", "extension/operations/development_metadata_diagnostic.slurm"),
                            ("submit_source_sha256", "extension/operations/submit_development_diagnostic.py")):
        require(rejected["binding"].get(field) == _sha(metadata._small_bytes(root / relative, root)),
                "Rejected diagnostic source changed")
    _job(five[DIAGNOSTIC_ROLE], DIAGNOSTIC_ROLE, DIAGNOSTIC_JOB_ID)
    entry, complete, accounting = five[DIAGNOSTIC_ROLE], diagnostic["complete.json"], diagnostic["accounting.json"]
    bind = entry["binding"]
    require(bind.get("failed_job") == FAILED_JOB_ID and bind.get("failed_binding_sha256") == three[FAILED_ROLE]["binding_sha256"] and
            bind.get("failure_receipt_sha256") == FAILED_RECEIPT_SHA256["development-seal-1223566-failure.json"] and
            bind.get("source_sha256") == old_sources and
            bind.get("prior_jobs_sha256") == _sha(_canonical({name: entry for name, entry in five.items() if name != DIAGNOSTIC_ROLE})) and
            bind.get("rejected_diagnostic_receipt_sha256") ==
                DIAGNOSTIC_RECEIPT_SHA256["development-diagnostic-dns-rejection-20261021.json"] and
            bind.get("wall_seconds") == 600 and bind.get("cpu") == 1 and bind.get("mem_gb") == 4,
            "Diagnostic admission must bind the exact failed stage and immutable prior history")
    for field, relative in (("wrapper_sha256", "extension/operations/development_metadata_diagnostic_v2.slurm"),
                            ("submit_source_sha256", "extension/operations/submit_development_diagnostic_v2.py")):
        require(bind.get(field) == _sha(metadata._small_bytes(root / relative, root)), "Diagnostic source changed")
    require(complete.get("schema") == "extension-development-level-diagnostic-v1" and complete.get("complete") is True and
            complete.get("compute", {}).get("job_id") == DIAGNOSTIC_JOB_ID and
            complete.get("compute", {}).get("gpu_requested_or_allocated") is False and complete.get("gpu_seconds") == 0 and
            complete.get("admission_binding_sha256") == entry["binding_sha256"] and
            complete.get("input_sha256") == previous.LABELED_OUTPUT_SHA256["surviving.jsonl"] and
            complete.get("rows") == 7132 and complete.get("math_rows") == 6013 and
            complete.get("non_1_to_5_levels") == OBSERVED_UNKNOWN_LEVELS and
            complete.get("no_labels_or_solutions_inspected") is True and
            complete.get("no_split_assignment_or_model_outcomes") is True,
            "Diagnostic must establish only the exact observed source-level schema")
    require(accounting.get("schema") == "extension-terminal-cpu-accounting-v1" and accounting.get("complete") is True and
            accounting.get("job_id") == DIAGNOSTIC_JOB_ID and accounting.get("gpu_seconds") == 0 and
            accounting.get("no_gpu_in_alloc_tres") is True and type(accounting.get("elapsed_seconds")) is int and
            accounting["elapsed_seconds"] == 2 and accounting.get("stderr") == "" and
            isinstance(accounting.get("stdout"), str), "Terminal successful zero-GPU diagnostic accounting is required")
    rows = [line.split("|") for line in accounting["stdout"].splitlines() if line]
    require(rows and len({row[0] for row in rows}) == len(rows) and
            len([row for row in rows if row[0] == DIAGNOSTIC_JOB_ID]) == 1 and
            all(len(row) == 8 and row[0] in {DIAGNOSTIC_JOB_ID, DIAGNOSTIC_JOB_ID + ".batch", DIAGNOSTIC_JOB_ID + ".extern"} and
                row[2:4] == ["COMPLETED", "0:0"] and row[4] == str(accounting["elapsed_seconds"]) and
                "gpu" not in row[7].lower() for row in rows), "Diagnostic terminal allocation rows do not reconcile")
    return inventory, five


def _check_prior_queue(snapshot, prior_jobs):
    expected = {
        "cpu_feasibility_20261020_v1": ("COMPLETED", "sft-extension-feasibility"),
        "labeled_feasibility_20261020_v1": ("COMPLETED", "sft-extension-labeled-feasibility"),
        FAILED_ROLE: ("FAILED", "sft-extension-development-seal"),
        DIAGNOSTIC_ROLE: ("COMPLETED", "sft-extension-level-diagnostic"),
    }
    by_id = {entry["job_id"]: expected[role] for role, entry in prior_jobs.items() if role != REJECTED_ROLE}
    names = {name for _, name in expected.values()}
    for row in snapshot["jobs"]:
        identifier = row["job_id"]
        if identifier in by_id:
            require((row["state"], row["name"]) == by_id[identifier], "Prior job has active/unknown or mismatched scheduler state")
        elif row["name"] in names or identifier.split("_")[0] in by_id:
            raise SubmissionError("Untracked duplicate prior role requires reconciliation")


def submit_development_retry(root=Path("."), *, now=None, runner=None):
    _ready()
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
                "Insufficient time for two-hour CPU retry and 48-hour reserve")
        return checked

    checked = guard()
    sources = source_inventory(root)
    metadata.validate_campaign(_read(extension / "campaign.json"))
    validate_plan(_read(extension / "development_retry_plan.json"))
    script = extension / "development_retry.slurm"
    directives = validate_wrapper(script, root)
    evidence, prior_jobs = initial_evidence(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(script)]
    binding = dict(role=ROLE, command=command, source_sha256=sources,
        campaign_sha256=sources["extension/campaign.json"]["sha256"],
        plan_sha256=sources["extension/development_retry_plan.json"]["sha256"], initial_evidence=evidence,
        prior_binding_sha256={name: job["binding_sha256"] for name, job in prior_jobs.items()},
        labeled_output_sha256=previous.LABELED_OUTPUT_SHA256, observed_unknown_levels=OBSERVED_UNKNOWN_LEVELS,
        requested_gpu_hours=0, requested_gpu_seconds=0, wall_time_seconds=WALL_SECONDS,
        closeout_reserve_seconds=RESERVE_SECONDS, deadline_utc=DEADLINE.isoformat())
    checksum = _sha(_canonical(binding))
    ledger_path, lock_path = extension / "job_ledger.json", extension / "job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "Extension ledger/lock cannot be symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(ledger_path.exists(), "The five-entry extension ledger is required")
        ledger = _read(ledger_path)
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
                ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
                ledger.get("gpu_hour_cap", "missing") is None and isinstance(ledger.get("jobs"), dict) and
                set(ledger["jobs"]) <= set(prior_jobs) | {ROLE} and
                all(ledger["jobs"].get(name) == entry for name, entry in prior_jobs.items()),
                "Unknown role or changed five-entry history; reconcile manually")
        snapshot = metadata._queue_snapshot(runner)
        _check_prior_queue(snapshot, prior_jobs)
        require(source_inventory(root) == sources and validate_wrapper(script, root) == directives and
                initial_evidence(root) == (evidence, prior_jobs), "Source/evidence changed during retry admission")
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
                    "Existing retry changed, uncertain or failed; never resubmit")
            require(not (same_name - {existing["job_id"]}), "Untracked duplicate retry role")
            return existing["job_id"]
        require(not same_name, "Untracked live retry role; reconcile")
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
            raise SubmissionError("Scheduler rejected/failed retry; reconcile, never resubmit")
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
    print(submit_development_retry(args.root))


if __name__ == "__main__":
    main()
