"""One fixed metadata-only GPU admission for the allocator-only v4 replay.

INTENTIONALLY BLOCKED until root seals every CPU97 closeout pin below. Separate
review and admission are required. No private witnesses, weights/tokenizers or model outcomes are opened.
An ambiguous/rejected submission is journaled and cannot be retried here.
"""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import subprocess
import uuid

from extension import source_generation_submission as parent
from extension import joint_allocator_v4_qa as qa
from extension.cpu_feasibility import value_digest


SELF = "extension/operations/submit-joint-allocator-v4-20261021.py"
DIRECTORY = "extension/report/joint-allocator-v4-qa-20261021-1228551"
QA_RUN = "extension/runs/joint-allocator-v4-qa-1228551"
CPU_JOB = "1228551"
PLAN_SHA256 = "c779c8e3ebbfb128fba0fd01547e8728a5b0c6d1c2ea333361e95b570a381a79"
INSTALL_SHA256 = "f4217f032612b3f86e75ad29b20044536250974bf9d5e52867a71e48d5bdaca3"
PRIOR_SHA256 = "938285e75ab09aacd5ffc1248baa0c3ef9bcb766d1c42e9b41ce327302c2ed8a"
CPU_ELAPSED_SECONDS = 52
PINS = {
    "unit_qa.json": "dab9595d0bac0fe1dd4abbce72005fb4d07ed855017b30793d95b11a7b546c66",
    "native_qa.json": "c4207f2fb3bf28a67d7028465931c49415d29dafba032d025640b581fff613d1",
    "complete.json": "c9b6e44594f43cceedc270c8d8ba8ad1b5fbeb9c8ceb9ef5e09fc955df28049a",
    "accounting.json": "311ddc8a3678b0a1f392ef50bfc1604bf29ab631c9e1be99990da59e8ab1f6ff",
    "job_ledger_snapshot.json": "0a73e09fec48dddb3bbeeb2b81cdff622be74206669ada0c6d464fd418ac08f0",
}
ROLE = "joint_memory_allocator_20261021_v4"
NAME = "sft-extension-joint-memory-allocator-v4"
SCRIPT = "extension/joint_memory_allocator_v4.slurm"
HANDOFF_PREFIX = "joint-memory-allocator-v4-job-"
GPU_SECONDS, RESERVE_SECONDS = 7200, 172800


def guard():
    now = datetime.now(timezone.utc)
    parent.shared.require((parent.m.DEADLINE - now).total_seconds() >= GPU_SECONDS + RESERVE_SECONDS,
        "Keep the full two-hour allocator replay and48-hour closeout reserve")
    return now


def validate_wrapper(root):
    text = parent.m._small_bytes(root / SCRIPT, root).decode()
    directives = [line.strip() for line in text.splitlines() if line.lstrip().startswith("#SBATCH")]
    expected = ["#SBATCH --job-name=" + NAME, "#SBATCH --account=tml_2026", "#SBATCH --partition=b200_normal_q",
        "#SBATCH --qos=owl_b200_normal_short", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8", "#SBATCH --mem=128G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(root / "extension/logs/extension-joint-allocator-v4-%j.out"),
        "#SBATCH --error=" + str(root / "extension/logs/extension-joint-allocator-v4-%j.err")]
    parent.shared.require(directives == expected,
        "Exact reviewed oneB200/8CPU/128G/two-hour/no-requeue allocator-v4 wrapper required")


def prerequisites(root):
    """Bounded source/public metadata only; exact future pins fail closed."""
    s, m = parent.shared, parent.m
    s.require(all(s._checksum(sha) for sha in (*PINS.values(), PRIOR_SHA256, PLAN_SHA256, INSTALL_SHA256)) and
        type(CPU_ELAPSED_SECONDS) is int and 0 < CPU_ELAPSED_SECONDS <= 1200,
        "Root must first seal terminal CPU97 receipts, source128 install, exact elapsed seconds and fixed23-entry snapshot")
    assets, values = {}, {}
    for name, sha in PINS.items():
        path = DIRECTORY + "/" + name
        assets[path] = s._record(root, path)
        s.require(assets[path]["sha256"] == sha, "Pinned CPU97 closeout changed: " + name)
        values[name] = m._read(root / path)
    unit, complete, snapshot = (values[name] for name in ("unit_qa.json", "complete.json", "job_ledger_snapshot.json"))
    s._ledger(snapshot)
    prior = snapshot["jobs"]
    s.require(len(prior) == 23 and value_digest(prior) == PRIOR_SHA256 and ROLE not in prior,
        "Preserve the exact23 prior admissions; allocator-v4GPU role must be new")
    # Only the already hard-pinned reconciled pre-scheduler no-ID entry is
    # exempt. All other prior entries require unique validated numeric IDs.
    s.require(len(s.scheduler_identities(prior)) == 22, "Unknown uncertain/no-ID/duplicate prior admission")
    sources = qa.source_inventory(root)
    codes = {name: item["sha256"] for name, item in sources.items()}
    s._passed_qa(unit, qa.SCHEMA)
    s.require(len(sources) == 128 and unit.get("job_id") == CPU_JOB and unit.get("tests_run") == 97 and
        unit.get("test_counts") == dict(zip(qa.PATTERNS, qa.COUNTS)) and unit.get("source_inventory") == sources and
        unit.get("code_sha256") == codes and unit.get("plan_sha256") == PLAN_SHA256 and
        unit.get("previous_focused65_regression_checks_rerun") is True and
        unit.get("historical450_receipts_verified_not_replayed") is True and unit.get("historical65_receipts_verified") is True and
        unit.get("allocator_policy") == qa.worker.ALLOCATOR_POLICY and unit.get("effective_expandable_segments_interface_established") is False and
        unit.get("failed_parent_remains_failed") is True and unit.get("actual_gpu_or_memory_admission_established") is False and
        unit.get("private_prepared_file_opened") is False and unit.get("experiment_model_or_tokenizer_loaded") is False and
        unit.get("automatic_downstream") is False and unit.get("compute_proof", {}).get("allocated_cpus") == 2,
        "The exact97 CPU fixtures and source128 proof must precede the separate allocator-only GPU replay")
    old22 = {role: entry for role, entry in prior.items() if role != qa.ROLE}
    cpu = prior.get(qa.ROLE, {})
    s._entry(cpu, qa.ROLE, gpu_seconds=0, sources=sources, job=CPU_JOB)
    s.require(len(old22) == 22 and value_digest(old22) == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("prior_admissions") == old22 and unit.get("prior_entries_sha256") == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("admission_binding_sha256") == cpu.get("binding_sha256") and
        cpu["binding"].get("expected_native_tests") == 97 and cpu["binding"].get("wall_time_seconds") == 1200 and
        cpu["binding"].get("closeout_reserve_seconds") == RESERVE_SECONDS and cpu["binding"].get("plan_sha256") == PLAN_SHA256,
        "All22 earlier entries and the successful23rd CPU admission must remain exact")
    s.validate_accounting(values["accounting.json"], CPU_JOB, name=qa.NAME)
    account = values["accounting.json"]
    s.require(account.get("elapsed_seconds") == CPU_ELAPSED_SECONDS and account.get("allocated_gpus") == 0 and
        all("cpu=2" in row.split("|")[7].split(",") and "mem=8G" in row.split("|")[7].split(",")
            for row in account["stdout"].splitlines() if row), "Exact terminal CPU2/8G/elapsed/zeroGPU proof required")
    s.require(complete.get("schema") == "extension-joint-allocator-v4-qa-complete-v1" and complete.get("complete") is True and
        complete.get("job_id") == CPU_JOB and complete.get("tests_run") == 97 and complete.get("gpu_seconds") == 0 and
        complete.get("source_inventory") == sources and complete.get("input_snapshot_unchanged") is True and
        complete.get("plan_sha256") == PLAN_SHA256 and complete.get("actual_gpu_or_memory_admission_established") is False and
        complete.get("automatic_downstream") is False and complete.get("output_sha256") ==
            {name: PINS[name] for name in ("unit_qa.json", "native_qa.json")}, "CPU97 did not complete unchanged")
    # The CPU proof's allowed input namespace contains source/config and public
    # failure/preparation metadata. Never follow arbitrary private run pins.
    old_assets = unit.get("initial_evidence", {})
    allowed = set(sources) | set(qa.PINS) | {qa.INSTALL, "deadline/job_ledger.json"}
    s.require(isinstance(old_assets, dict) and set(old_assets) <= allowed and set(qa.PINS) <= set(old_assets) and
        qa.INSTALL in old_assets, "CPU proof must bind only known public source/failure metadata")
    for name, expected in old_assets.items():
        actual = s._record(root, name)
        s.require(actual == expected, "A source/failed-parent/preparation metadata input changed: " + name)
        assets[name] = actual
    assets[qa.INSTALL] = s._record(root, qa.INSTALL)
    installed = m._read(root / qa.INSTALL)
    s.require(assets[qa.INSTALL]["sha256"] == INSTALL_SHA256 and installed.get("installed_on_arc") is True and
        installed.get("source_inventory") == sources and len(installed.get("parent_source_inventory", {})) == 122 and
        installed.get("parent_source_inventory") == unit.get("parent_source_inventory"), "Installed128/unchanged122 proof differs")
    expected_alias = {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "plan_sha256": qa.JOINT_PLAN["sha256"], "prepared_sha256": qa.PREPARED["sha256"],
        "operational_plan_sha256": PLAN_SHA256, "unit_qa_sha256": PINS["unit_qa.json"],
        "cpu_qa_alias_of_same97_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True}
    s.require(values["native_qa.json"] == expected_alias, "The handoff alias must describe the SAME97 CPU checks without a GPU claim")
    private_alias = QA_RUN + "/native_qa.json"
    assets[private_alias] = s._record(root, private_alias)
    s.require(assets[private_alias]["sha256"] == PINS["native_qa.json"], "Actual handoff QA alias differs from its public closeout")
    validate_wrapper(root)
    for name, sha in ((qa.JOINT_PLAN["path"], qa.JOINT_PLAN["sha256"]), (qa.NATIVE_PLAN["path"], qa.NATIVE_PLAN["sha256"]),
            (qa.PLAN, PLAN_SHA256), ("deadline/job_ledger.json", qa.HISTORICAL_LEDGER_SHA256)):
        assets[name] = s._record(root, name)
        s.require(assets[name]["sha256"] == sha, "A pinned scientific/operational plan or historical ledger changed")
    assets[SELF] = s._record(root, SELF)
    return {"sources": sources, "codes": codes, "assets": assets, "prior": prior}


def binding_for(root, proof):
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT), QA_RUN]
    return {"role": ROLE, "command": command, "source_sha256": proof["sources"], "plan_sha256": qa.JOINT_PLAN["sha256"],
        "prepared_sha256": qa.PREPARED["sha256"], "qa_sha256": PINS["native_qa.json"], "operational_plan_sha256": PLAN_SHA256,
        "allocator_policy": qa.worker.ALLOCATOR_POLICY,
        "cpu_unit_qa_sha256": PINS["unit_qa.json"], "cpu_qa_job_id": CPU_JOB, "prior_entries_sha256": PRIOR_SHA256,
        "initial_evidence": proof["assets"], "admission_code_sha256": proof["assets"][SELF]["sha256"],
        "requested_gpu_seconds": GPU_SECONDS, "requested_gpu_hours": 2, "wall_time_seconds": GPU_SECONDS,
        "closeout_reserve_seconds": RESERVE_SECONDS, "same_original_scientific_plan_and_prepared_cache": True,
        "automatic_retry": False, "automatic_downstream": False}


def handoff(root, entry, proof):
    # This is the scheduler/submission digest. The worker independently binds
    # its six runtime inputs, including this handoff file's SHA and op-plan SHA.
    path = root / "extension/operations" / (HANDOFF_PREFIX + entry["job_id"] + ".json")
    value = {"schema": "extension-joint-memory-admission-job-v1", "status": "admitted", "job_id": entry["job_id"], "role": ROLE,
        "plan_sha256": qa.JOINT_PLAN["sha256"], "prepared_sha256": qa.PREPARED["sha256"], "qa_sha256": PINS["native_qa.json"],
        "code_sha256": proof["codes"], "binding_sha256": entry["binding_sha256"], "operational_plan_sha256": PLAN_SHA256,
        "allocator_policy": qa.worker.ALLOCATOR_POLICY,
        "cpu_unit_qa_sha256": PINS["unit_qa.json"], "cpu_qa_job_id": CPU_JOB, "admission_code_sha256": proof["assets"][SELF]["sha256"],
        "automatic_retry": False, "automatic_downstream": False}
    return path, value


def validate_history(root, ledger, proof):
    s = parent.shared
    s._ledger(ledger)
    s.require(set(ledger["jobs"]) <= set(proof["prior"]) | {ROLE} and
        all(ledger["jobs"].get(role) == value for role, value in proof["prior"].items()), "Unknown/rewritten prior admission")
    entry = ledger["jobs"].get(ROLE)
    if entry is not None:
        s._entry(entry, ROLE, gpu_seconds=GPU_SECONDS, sources=proof["sources"])
        binding = binding_for(root, proof)
        s.require(entry.get("binding") == binding and entry.get("binding_sha256") == value_digest(binding),
            "Existing/uncertain/rejected replay requires manual reconciliation, never retry")
        target, expected = handoff(root, entry, proof)
        s.require(target.is_file() and not target.is_symlink() and parent.m._read(target) == expected and
            entry.get("admission_receipt_sha256") == s._record(root, target)["sha256"], "Existing submitted handoff must remain immutable")
    s.scheduler_identities(ledger["jobs"])


def write_exclusive(path, value):
    """Fully written atomic handoff publication, refusing every existing path."""
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(parent.m._canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    root, s, m = Path.cwd().resolve(), parent.shared, parent.m
    s.require(Path(__file__).resolve() == root / SELF, "Run the exact installed one-use bootstrap from project root")
    guard()
    path, lockpath = root / "extension/job_ledger.json", root / "extension/job_ledger.lock"
    s.require(not path.is_symlink() and not lockpath.is_symlink(), "No ledger aliases")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proof, ledger = prerequisites(root), m._read(path)
        validate_history(root, ledger, proof)
        binding = binding_for(root, proof)
        checksum = value_digest(binding)
        snapshot = s.queue_snapshot(subprocess.run)
        parent.known_queue(root, snapshot, ledger["jobs"])
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        current = ledger["jobs"].get(ROLE)
        s.require(not same if current is None else not same - {current["job_id"]}, "Untracked/duplicate allocator v4 replay")
        s.require(m._read(path) == ledger and prerequisites(root) == proof,
            "Source128/self/public evidence/ledger changed during the live scheduler snapshot")
        if current is not None:
            s.check_capacity(snapshot, already_submitted=True)
            print(current["job_id"], flush=True)
            return
        s.check_capacity(snapshot)
        when = guard()
        environment, removed = m._submission_environment()
        current = {"status": "submitting", "binding": binding, "binding_sha256": checksum, "source_sha256": proof["sources"],
            "command": binding["command"], "requested_gpu_seconds": GPU_SECONDS, "requested_gpu_hours": 2,
            "requested_at_utc": when.isoformat(), "scheduler_snapshot": snapshot, "removed_scheduler_environment_keys": removed}
        before = dict(ledger["jobs"])
        ledger["jobs"][ROLE] = current
        m._write(path, ledger)
        try:
            response = subprocess.run(binding["command"], capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            current.update(status="uncertain", error_type=type(error).__name__)
            m._write(path, ledger)
            raise
        current.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:],
            scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip()) if isinstance(response.stdout, str) else None
        if type(response.returncode) is not int or response.returncode != 0 or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            current["status"] = "uncertain"
            if match:
                current["possible_job_id"] = match.group(1)
            m._write(path, ledger)
            raise m.SubmissionError("Rejected/ambiguous replay is preserved without automatic retry")
        job = match.group(1)
        current.update(status="uncertain", job_id=job)
        m._write(path, ledger)
        s.require(job not in s.scheduler_identities(before) and job not in {row["job_id"] for row in snapshot["jobs"]},
            "Returned scheduler identity collides with a validated prior/current allocation")
        s.require(m._read(path) == ledger and prerequisites(root) == proof,
            "Source/self/evidence/ledger changed after scheduler acceptance")
        target, value = handoff(root, current, proof)
        s.require(not target.exists() and not target.is_symlink(), "Never overwrite an existing allocator-v4GPU handoff")
        write_exclusive(target, value)
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            admission_receipt_sha256=s._record(root, target)["sha256"])
        m._write(path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
