"""One fixed metadata-only CPU32-regression-then-calibration retry admission; no model or grading here."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import calibration_grading_stage_v2 as metadata
from extension import joint_gpu_admission as shared
from extension import source_generation_submission as queue_owner
from extension import submission as io

SELF = "extension/operations/submit-calibration-grading-v2-20261021.py"
INSTALL = "extension/calibration_grading_v2_source_receipt.json"
INSTALL_SHA256 = "772746c8296b29b9958f1cf576de15b5e06372b72bcd657fbb0af9dbe65de8a6"
PLAN_SHA256 = "73481660869027a932bcfe9a23b4b9ecdad11f49dc8160150170dca2bb40987a"
PRIOR_SHA256 = "791fa7c9715eff12d8d65b95cc5c855b1930b6f473c00fd06a88abc817c2453b"


def prerequisites(root):
    shared.require(shared._checksum(INSTALL_SHA256) and shared._checksum(PLAN_SHA256),
        "Root must seal the exact installation and prospective plan")
    files = metadata.Files(root)
    plan = metadata.validate_plan(files.json(metadata.PLAN, PLAN_SHA256))
    sources = metadata.source_inventory(root)
    metadata.validate_wrapper(root)
    prior = files.json(metadata.HISTORY["path"], metadata.HISTORY["sha256"])
    shared._ledger(prior)
    shared.require(len(prior["jobs"]) == 29 and metadata.value_digest(prior["jobs"]) == PRIOR_SHA256 and
        metadata.ROLE not in prior["jobs"], "Only the exact29 prior entries may precede this fixed CPU stage")
    parent = files.json(metadata.PARENT["path"], metadata.PARENT["sha256"])
    protected = parent.get("protected_source_inventory", {})
    shared.require(len(protected) == 160 and set(protected) & metadata.SOURCES == set(metadata.OLD_SOURCES) and
        all(protected.get(name) == item for name, item in metadata.OLD_SOURCES.items()),
        "Protect exact160 unchanged parent sources")
    installation = files.json(INSTALL, INSTALL_SHA256)
    shared.require(installation.get("installed_on_arc") is True and installation.get("source_inventory") == sources and
        installation.get("protected_parent_sources") == protected and installation.get("historical_ledger_unchanged") is True,
        "Exact source8 installation and unchanged protected160 required")
    for name, item in protected.items():
        files.read(name, item["sha256"], maximum=1 << 20, expected_bytes=item["bytes"], keep=False)
    for name, item in sources.items():
        files.read(name, item["sha256"], maximum=1 << 20, expected_bytes=item["bytes"], keep=False)
    files.read(metadata.HISTORICAL["path"], metadata.HISTORICAL["sha256"], keep=False)
    files.read(SELF, maximum=1 << 20, keep=False)
    metadata.public_generation_evidence(files)
    metadata.failure_and_repair_evidence(files)
    files.read(metadata.PROTOCOL["path"], metadata.PROTOCOL["sha256"], keep=False)
    files.unchanged()
    return {"plan": plan, "sources": sources, "prior": prior["jobs"], "assets": files.records}


def main():
    root = Path.cwd().resolve()
    shared.require(Path(__file__).resolve() == root / SELF, "Run only the installed fixed bootstrap")
    metadata.deadline_guard()
    path, lockpath = root / "extension/job_ledger.json", root / "extension/job_ledger.lock"
    shared.require(not path.is_symlink() and not lockpath.is_symlink(), "No ledger aliases")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proof = prerequisites(root)
        ledger = io._read(path)
        shared._ledger(ledger)
        before = {name: value for name, value in ledger["jobs"].items() if name != metadata.ROLE}
        shared.require(before == proof["prior"], "Preserve exact29 entries; reconcile unknown or uncertain admissions")
        command = ["sbatch", "--clusters=owl", "--parsable", str(root / metadata.SCRIPT)]
        binding = {"role": metadata.ROLE, "command": command, "source_sha256": proof["sources"],
            "plan_sha256": PLAN_SHA256, "prior_entries_sha256": PRIOR_SHA256,
            "failed_parent_manifest_sha256": metadata.PARENT["sha256"],
            "permission_repair_sha256": metadata.REPAIR["sha256"],
            "admission_code_sha256": proof["assets"][SELF]["sha256"], "initial_evidence": proof["assets"],
            "requested_gpu_seconds": 0, "requested_gpu_hours": 0, "expected_native_tests": 32,
            "wall_time_seconds": 3600, "closeout_reserve_seconds": 172800,
            "automatic_retry": False, "automatic_downstream": False}
        checksum = metadata.value_digest(binding)
        snapshot = shared.queue_snapshot(subprocess.run)
        queue_owner.known_queue(root, snapshot, ledger["jobs"])
        shared.check_capacity(snapshot, already_submitted=True)
        shared.require(io._read(path) == ledger and prerequisites(root) == proof,
            "Source/self/plan/history/ledger changed during the live queue query")
        current = ledger["jobs"].get(metadata.ROLE)
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == metadata.NAME}
        if current is not None:
            shared._entry(current, metadata.ROLE, gpu_seconds=0, sources=proof["sources"])
            shared.require(current.get("binding") == binding and current.get("binding_sha256") == checksum and
                not same - {current["job_id"]}, "Existing or uncertain stage must be reconciled, never retried")
            print(current["job_id"], flush=True)
            return
        shared.require(not same, "Untracked duplicate CPU calibration grading")
        metadata.deadline_guard()
        environment, removed = io._submission_environment()
        current = {"status": "submitting", "binding": binding, "binding_sha256": checksum,
            "source_sha256": proof["sources"], "command": command, "requested_gpu_seconds": 0,
            "requested_gpu_hours": 0, "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "scheduler_snapshot": snapshot, "removed_scheduler_environment_keys": removed}
        ledger["jobs"][metadata.ROLE] = current
        io._write(path, ledger)
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            current.update(status="uncertain", error_type=type(error).__name__)
            io._write(path, ledger)
            raise
        current.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:],
            scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip()) if isinstance(response.stdout, str) else None
        if type(response.returncode) is not int or response.returncode != 0 or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            current["status"] = "uncertain"
            if match:
                current["possible_job_id"] = match.group(1)
            io._write(path, ledger)
            raise io.SubmissionError("Rejected or ambiguous CPU admission preserved without retry")
        job = match.group(1)
        current.update(status="uncertain", job_id=job)
        io._write(path, ledger)
        shared.require(job not in shared.scheduler_identities(before) and job not in {row["job_id"] for row in snapshot["jobs"]},
            "Scheduler identity collision")
        shared.require(io._read(path) == ledger and prerequisites(root) == proof,
            "Source/history changed after acceptance; preserve uncertain identity")
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat())
        io._write(path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
