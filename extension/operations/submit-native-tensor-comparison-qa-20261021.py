"""One fixed metadata-only CPU32 synthetic tensor-comparison admission, never model or fixture execution."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import native_tensor_comparison_qa as metadata
from extension import joint_gpu_admission as shared
from extension import source_generation_submission as queue_owner
from extension import submission as io

SELF = "extension/operations/submit-native-tensor-comparison-qa-20261021.py"
INSTALL = "extension/native_tensor_comparison_source_receipt.json"
INSTALL_SHA256 = "7c7fa62c89e8a5cb1b22970aca247bd571db97d808eaf14d293ab5f14fa7b3df"
PLAN_SHA256 = "6b81927173d7ebd8d20b5e640c82fdaa466d6d570ade7271acb27a47a20cc5e0"
PRIOR_SHA256 = "6ae9ce8c4be049d237b0e628f32d94c51343e1a5631d478669ae0acebf72cd00"


def prerequisites(root):
    shared.require(shared._checksum(INSTALL_SHA256) and shared._checksum(PLAN_SHA256), "Root must seal exact installation and prospective plan")
    plan = metadata.validate_plan(metadata.read_json(root, metadata.PLAN, PLAN_SHA256))
    sources = metadata.source_inventory(root)
    metadata.validate_wrapper(root)
    pin = plan["prior_ledger"]
    prior = metadata.read_json(root, pin["path"], pin["sha256"])
    shared._ledger(prior)
    shared.require(len(prior["jobs"]) == pin["entries"] == 31 and
        metadata.value_digest(prior["jobs"]) == pin["entries_sha256"] == PRIOR_SHA256 and metadata.ROLE not in prior["jobs"],
        "Only exact31 prior entries may precede this fixed CPU stage")
    installation = metadata.read_json(root, INSTALL, INSTALL_SHA256)
    parent_pin = plan["parent_sources"]
    parent = metadata.read_json(root, parent_pin["path"], parent_pin["sha256"])
    protected = parent.get(parent_pin["inventory_key"], {})
    shared.require(parent.get("schema") == parent_pin["schema"] and parent.get("raw_private_records_included") is False and
        isinstance(protected, dict) and len(protected) == 170 and not set(protected) & metadata.SOURCES,
        "Exact pinned protected170 parent manifest required before scheduler admission")
    shared.require(installation.get("installed_on_arc") is True and installation.get("source_inventory") == sources and
        installation.get("protected_parent_sources") == protected, "Exact source5 and unchanged protected170 required")
    assets = {name: metadata.record(root, name) for name in (SELF, INSTALL, pin["path"], parent_pin["path"], "deadline/job_ledger.json")}
    for name, record in installation["protected_parent_sources"].items():
        shared.require(metadata.record(root, name) == record, "A protected parent changed: " + name)
        assets[name] = record
    shared.require(assets["deadline/job_ledger.json"]["sha256"] == plan["historical_ledger"]["sha256"] ==
        "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d", "Historical ARC ledger changed")
    return {"plan": plan, "sources": sources, "prior": prior["jobs"], "assets": assets}


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
        shared.require(before == proof["prior"], "Preserve all exact31 entries; reconcile any unknown or uncertain admission")
        command = ["sbatch", "--clusters=owl", "--parsable", str(root / metadata.SCRIPT)]
        binding = {"role": metadata.ROLE, "command": command, "source_sha256": proof["sources"],
            "plan_sha256": PLAN_SHA256, "prior_entries_sha256": PRIOR_SHA256,
            "parent_source_manifest_sha256": proof["plan"]["parent_sources"]["sha256"],
            "admission_code_sha256": proof["assets"][SELF]["sha256"], "initial_evidence": proof["assets"],
            "requested_gpu_seconds": 0, "requested_gpu_hours": 0, "expected_native_tests": 32,
            "wall_time_seconds": 1200, "closeout_reserve_seconds": 172800,
            "automatic_retry": False, "automatic_downstream": False}
        checksum = metadata.value_digest(binding)
        snapshot = shared.queue_snapshot(subprocess.run)
        queue_owner.known_queue(root, snapshot, ledger["jobs"])
        shared.check_capacity(snapshot, already_submitted=True)
        shared.require(io._read(path) == ledger and prerequisites(root) == proof, "Source/self/plan/history/ledger changed during queue query")
        current = ledger["jobs"].get(metadata.ROLE)
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == metadata.NAME}
        if current is not None:
            shared._entry(current, metadata.ROLE, gpu_seconds=0, sources=proof["sources"])
            shared.require(current.get("binding") == binding and current.get("binding_sha256") == checksum and
                not same - {current["job_id"]}, "Existing or uncertain CPU stage must be reconciled, never retried")
            print(current["job_id"], flush=True)
            return
        shared.require(not same, "Untracked duplicate CPU stage")
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
        current.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:], scheduler_stderr=str(response.stderr)[-2000:])
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
        shared.require(job not in shared.scheduler_identities(before) and job not in {row["job_id"] for row in snapshot["jobs"]}, "Scheduler identity collision")
        shared.require(io._read(path) == ledger and prerequisites(root) == proof, "Source/history changed after acceptance; preserve uncertain identity")
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat())
        io._write(path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
