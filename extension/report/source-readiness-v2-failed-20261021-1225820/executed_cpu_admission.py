"""One fixed metadata-only CPU admission; no private cache/model/result opens.

Preserve the exact fourteen existing admissions, including both already-admitted
GPU diagnostics and the one sealed pre-scheduler DNS rejection. This adapter
only admits scoped CPU QA and unchanged-cohort calibration-prompt preparation.
Failure/uncertainty requires manual reconciliation, never automatic resubmission.
"""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess

from extension import submission as metadata
from extension import joint_gpu_admission as gpu
from extension import source_readiness_v2 as stage
from extension.cpu_feasibility import value_digest


ROLE = stage.ROLE
NAME = "sft-extension-source-readiness-v2-calibration"
SCRIPT = "extension/source_readiness_v2.slurm"
SELF = "extension/operations/submit-source-readiness-v2-20261021.py"
PLAN_SHA256 = "40479982fe3b43e27f51ec27f28a2cf39f76a464ceffdc576e1f38268f0a6a3b"
PRIOR_ENTRIES_SHA256 = "8118d230dbb888031226b4bd5f1df7d47727ffa4bbc18c7821c8ccfecb3856d5"
QA_RUN = "extension/runs/joint-memory-qa-1225557"
METADATA_QA = "extension/runs/joint-gpu-metadata-qa-1225720/native_qa.json"
METADATA_QA_SHA256 = "d15f2a5de33c4fdb20a1a4b20b91e55ac3fb6d247f998a9712c061484deef39b"
GPU_JOB = "1225727"
GPU_RECEIPT = "extension/operations/joint-memory-admission-job-1225727.json"
GPU_RECEIPT_SHA256 = "13fe2386f951daa3b2dc4eb4133d5d5b65df7fc63eee2af8ec1725d65d77c7e2"
PARSER_COMPLETE = "extension/report/joint-gpu-metadata-qa-20261021-1225720/complete.json"
PARSER_COMPLETE_SHA256 = "55fcd29d2dee64780a7e867b1fd7c40722edbedadfc2beffa55486152f2cfb89"
PARSER_ACCOUNTING_SHA256 = "76f0fc0983bab7ab9152c3daf8f3743c7216a5eea38f833c323c9dffb3af1981"
FROZEN_NEW = {
    "extension/source_evaluation_v2.py": "639e3964a750f3af77d90b4aa4aa6f5e166fd20d494c07abeb2eb08a29e226c4",
    "tests/test_source_evaluation_v2.py": "57c2b630f2d9b77da0dab279e4f3ba00faac8d2f6a4edc6344b82ef7c9c5010a",
    "extension/source_readiness_v2.py": "4aa35f9c70c1663877fe52fede56e0d93f73a5d267d82c9c1b8770f471dc3e00",
    "tests/test_source_readiness_v2.py": "5971cfc15b30255e1b7528ad276de92293510d30980ad91935c6c730b63c008a",
    "extension/source_readiness_v2_plan.json": PLAN_SHA256,
    "extension/source_readiness_v2_qa.py": "9e73c53c7412c9416a9e48c8aa2e33ec97a60537f99f61f6107f15d843a91291",
    SCRIPT: "2a8f586af0ed5cea76ff00a67346880bbcee33a736edc3377df99fc6d85f4b8d"}
require = metadata.require


def record(root, relative):
    return gpu._record(root, relative)


def guard():
    now = datetime.now(timezone.utc)
    require((metadata.DEADLINE - now).total_seconds() >= 180000,
        "The full two-hour CPU stage and 48-hour closeout reserve are required")
    return now


def prior_jobs(ledger):
    gpu._ledger(ledger)
    prior = {key: value for key, value in ledger["jobs"].items() if key != ROLE}
    require(len(prior) == 14 and value_digest(prior) == PRIOR_ENTRIES_SHA256 and
        set(ledger["jobs"]) <= set(prior) | {ROLE}, "Preserve the exact fourteen actual prior entries")
    # This is the only allowed uncertain/no-ID entry, pinned in full by the
    # native scientific history and by PRIOR_ENTRIES_SHA256 above.
    identities = gpu.scheduler_identities(prior)
    require(len(identities) == 13 and prior[gpu.ROLE].get("job_id") == GPU_JOB and
        prior[gpu.METADATA_ROLE].get("job_id") == "1225720", "Actual tested CPU/GPU identities changed")
    return prior


def prerequisites(root, prior):
    """Small source and receipt metadata only; never follow prepared/source data."""
    parent = {"schema": metadata.SCHEMA, "historical_ledger_unchanged": "deadline/job_ledger.json",
        "gpu_hour_cap": None, "jobs": prior}
    inherited = gpu.load_evidence(root, QA_RUN, METADATA_QA, METADATA_QA_SHA256, parent)
    require(set(prior) == set(inherited["prior"]) | {gpu.ROLE} and
        all(prior[key] == value for key, value in inherited["prior"].items()), "All thirteen pre-GPU admissions must remain exact")
    current_gpu = prior[gpu.ROLE]
    gpu._entry(current_gpu, gpu.ROLE, gpu_seconds=7200, sources=inherited["sources"], job=GPU_JOB)
    receipt_path, expected_receipt = gpu._admission_receipt(root, current_gpu, inherited)
    require(str(receipt_path.relative_to(root)) == GPU_RECEIPT and
        record(root, GPU_RECEIPT)["sha256"] == GPU_RECEIPT_SHA256 and
        metadata._read(receipt_path) == expected_receipt and
        current_gpu.get("admission_receipt_sha256") == GPU_RECEIPT_SHA256,
        "Preserve the exact successful already-admitted joint GPU handoff")
    sources = stage.source_inventory(root)
    require(len(sources) == 86 and set(sources) == set(inherited["sources"]) | set(FROZEN_NEW) and
        all(sources[name] == value for name, value in inherited["sources"].items()) and
        all(sources[name]["sha256"] == checksum for name, checksum in FROZEN_NEW.items()),
        "Only the frozen eighty-six source/config/test files are permitted")
    plan = stage.validate_plan(metadata._read(root / stage.PLAN))
    assets = dict(inherited["assets"])
    for relative, checksum in ((stage.PLAN, PLAN_SHA256),
            (stage.LEGACY_PLAN["path"], stage.LEGACY_PLAN["sha256"]),
            (stage.PROTOCOL_PIN["path"], stage.PROTOCOL_PIN["sha256"]),
            (GPU_RECEIPT, GPU_RECEIPT_SHA256), (PARSER_COMPLETE, PARSER_COMPLETE_SHA256)):
        assets[relative] = record(root, relative)
        require(assets[relative]["sha256"] == checksum, "Pinned prerequisite metadata changed: " + relative)
    old = stage.legacy.validate_plan(metadata._read(root / stage.LEGACY_PLAN["path"]))
    stage.validate_protocol(metadata._read(root / stage.PROTOCOL_PIN["path"]), old)
    parser_complete = metadata._read(root / PARSER_COMPLETE)
    require(parser_complete.get("schema") == "extension-joint-gpu-metadata-qa-complete-v1" and
        parser_complete.get("complete") is True and parser_complete.get("cpu_only") is True and
        parser_complete.get("job_id") == "1225720" and parser_complete.get("tests_run") == 46 and
        parser_complete.get("native_qa_sha256") == METADATA_QA_SHA256 and
        parser_complete.get("gpu_seconds") == 0 and parser_complete.get("automatic_downstream") is False and
        inherited["metadata_value"].get("tests_run") == 46,
        "The independent forty-six-check native metadata QA must be terminal and complete")
    accounting_path = str((Path(METADATA_QA).parent / "accounting.json"))
    require(inherited["assets"][accounting_path]["sha256"] == PARSER_ACCOUNTING_SHA256,
        "The exact terminal zero-GPU parser-QA accounting changed")
    require(plan["native_qa"]["new_readiness_tests"] == 24, "Freeze the exact scoped four-hundred-seventy-four checks")
    expected = ["#SBATCH --job-name=" + NAME, "#SBATCH --account=tml_2026", "#SBATCH --partition=normal_q",
        "#SBATCH --qos=owl_normal_short", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(root / "extension/logs/extension-source-readiness-v2-calibration-%j.out"),
        "#SBATCH --error=" + str(root / "extension/logs/extension-source-readiness-v2-calibration-%j.err")]
    text = metadata._small_bytes(root / SCRIPT, root).decode("utf-8")
    require([line.strip() for line in text.splitlines() if line.lstrip().startswith("#SBATCH")] == expected,
        "Only the exact single-node CPU8/64G/2h/zero-GPU/no-requeue wrapper is permitted")
    header = True
    for line in text.splitlines():
        if line.lstrip().startswith("#SBATCH"):
            require(header, "SBATCH directive after executable code")
        elif line.strip() and not line.lstrip().startswith("#"):
            header = False
    assets[SELF] = record(root, SELF)
    return {"sources": sources, "inherited": inherited, "assets": assets}


def check_queue(snapshot, prior, proof):
    inherited = proof["inherited"]
    gpu._known_queue(snapshot, inherited["prior"], inherited["metadata_value"], inherited["metadata_name"])
    # Both known GPU allocations may still be pending/running or become terminal;
    # neither is cancelled, altered or resubmitted by this CPU-only admission.
    for row in snapshot["jobs"]:
        if row["job_id"] == GPU_JOB:
            require(row["name"] == gpu.NAME and row["gpus"] in {0, 1} and
                (row["state"] in gpu.TERMINAL or row["gpus"] == 1),
                "Already admitted joint GPU identity/resources changed")
        elif row["name"] == gpu.NAME:
            raise metadata.SubmissionError("Untracked duplicate joint GPU diagnostic")
    require(len(gpu.scheduler_identities(prior)) == 13, "Known numeric/no-ID history changed")


def main():
    root = Path.cwd().resolve()
    require(Path(__file__).resolve() == root / SELF, "Run only this installed one-use adapter from the project root")
    guard()
    ledger_path, lock_path = root / "extension/job_ledger.json", root / "extension/job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "No ledger/lock aliases")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = metadata._read(ledger_path)
        prior = prior_jobs(ledger)
        proof = prerequisites(root, prior)
        command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
        binding = {"role": ROLE, "command": command, "source_sha256": proof["sources"],
            "admission_code_sha256": proof["assets"][SELF]["sha256"], "plan_sha256": PLAN_SHA256,
            "legacy_plan_sha256": stage.LEGACY_PLAN["sha256"], "worker_protocol_sha256": stage.PROTOCOL_PIN["sha256"],
            "prior_entries_sha256": value_digest(prior), "initial_evidence": proof["assets"],
            "metadata_source_sha256": proof["inherited"]["parser_sources"], "expected_native_tests": 474,
            "requested_gpu_hours": 0, "requested_gpu_seconds": 0, "wall_time_seconds": 7200,
            "closeout_reserve_seconds": 172800, "deadline_utc": metadata.DEADLINE.isoformat(),
            "old465_scope_replayed": False, "automatic_downstream": False}
        checksum = value_digest(binding)
        snapshot = gpu.queue_snapshot(subprocess.run)
        check_queue(snapshot, prior, proof)
        require(metadata._read(ledger_path) == ledger and prior_jobs(ledger) == prior and
            prerequisites(root, prior) == proof, "Sources, prerequisites, self bytes or ledger changed during live snapshot")
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        current = ledger["jobs"].get(ROLE)
        if current is not None:
            require(current.get("status") == "submitted" and current.get("binding") == binding and
                current.get("binding_sha256") == checksum and current.get("source_sha256") == proof["sources"] and
                current.get("requested_gpu_seconds") == 0 and current.get("requested_gpu_hours") == 0 and
                gpu._numeric_job(current.get("job_id")) and not same - {current["job_id"]},
                "Existing/failed/uncertain CPU admission must be preserved, never retried")
            for row in snapshot["jobs"]:
                if row["job_id"] == current["job_id"]:
                    require(row["name"] == NAME and row["gpus"] == 0, "Current CPU scheduler identity/resources changed")
            print(current["job_id"], flush=True)
            return
        require(not same, "Untracked current CPU preparation requires reconciliation")
        when = guard()
        require((root / "extension/logs").is_dir() and not (root / "extension/logs").is_symlink(),
            "Reviewed extension log directory must exist before admission")
        environment, removed = metadata._submission_environment()
        current = {"status": "submitting", "binding": binding, "binding_sha256": checksum,
            "source_sha256": proof["sources"], "plan_sha256": PLAN_SHA256, "command": command,
            "requested_gpu_hours": 0, "requested_gpu_seconds": 0, "requested_at_utc": when.isoformat(),
            "scheduler_snapshot": snapshot, "removed_scheduler_environment_keys": removed}
        ledger["jobs"][ROLE] = current
        metadata._write(ledger_path, ledger)
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            current.update(status="uncertain", error_type=type(error).__name__)
            metadata._write(ledger_path, ledger)
            raise metadata.SubmissionError("Submission outcome uncertain; preserve for reconciliation, no retry") from error
        current.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:],
            scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip()) if isinstance(response.stdout, str) else None
        if response.returncode or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            current["status"] = "submission_failed" if response.returncode else "uncertain"
            if match is not None:
                current["possible_job_id"] = match.group(1)
            metadata._write(ledger_path, ledger)
            raise metadata.SubmissionError("Rejected/ambiguous CPU admission preserved; never resubmit automatically")
        job = match.group(1)
        current.update(status="uncertain", job_id=job)
        metadata._write(ledger_path, ledger)
        require(job not in gpu.scheduler_identities(prior) and job not in {row["job_id"] for row in snapshot["jobs"]},
            "Scheduler returned an already known identity; reconcile manually")
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat())
        metadata._write(ledger_path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
