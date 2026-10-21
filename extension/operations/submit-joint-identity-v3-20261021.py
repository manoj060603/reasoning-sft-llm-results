"""One fixed metadata-only GPU admission for the reviewed joint v3 repair.

The exact terminalCPU65 closeout is sealed below; separate root review and
admission are required. This adapter never opens private witnesses, weights/tokenizers or model outcomes.
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
from extension import joint_identity_replay_v3_qa as qa
from extension.cpu_feasibility import value_digest


SELF = "extension/operations/submit-joint-identity-v3-20261021.py"
DIRECTORY = "extension/report/joint-identity-v3-qa-20261021-1228170"
QA_RUN = "extension/runs/joint-identity-v3-qa-1228170"
CPU_JOB = "1228170"
PLAN_SHA256 = "88e57bc1399d180a30645b07e578d9edde7630c1dcf734e95bcc667d7f015fb8"
INSTALL_SHA256 = "4d88d39475bd1309780aec6e4760447cefed68e98e8ac8f7924ab3373a638139"
PRIOR_SHA256 = "cdefeccbda017dcf7ddfb0023b2b620eabaf888a43097b92d5fe3b0bcd8b381e"
CPU_ELAPSED_SECONDS = 32
PINS = {
    "unit_qa.json": "8e0f61aabe1ba4d285007c92d680542762cfae1271905c2cec63f908dc760874",
    "native_qa.json": "cf05eb73d2a2a44e548b4b354254af6544cdee23b8cf691cf30191414d2045a5",
    "complete.json": "00835a892b571eaba14a6c0f65c7ede751d6c050472d43146460f5df0f9dd94e",
    "accounting.json": "455cda55ba852be2ef7bb0d60e393b94e51708947e7f1738409b12fcc011a96f",
    "job_ledger_snapshot.json": "7bd5d37f72860ae5b9e5a7743d561ca1c62286da706d3a5500313e192c787cd6",
}
ROLE = "joint_memory_admission_20261021_v3"
NAME = "sft-extension-joint-memory-admission-v3"
SCRIPT = "extension/joint_memory_admission_v3.slurm"
HANDOFF_PREFIX = "joint-memory-admission-v3-job-"
GPU_SECONDS, RESERVE_SECONDS = 7200, 172800


def guard():
    now = datetime.now(timezone.utc)
    parent.shared.require((parent.m.DEADLINE - now).total_seconds() >= GPU_SECONDS + RESERVE_SECONDS,
        "Keep the full two-hour joint replay and48-hour closeout reserve")
    return now


def validate_wrapper(root):
    text = parent.m._small_bytes(root / SCRIPT, root).decode()
    directives = [line.strip() for line in text.splitlines() if line.lstrip().startswith("#SBATCH")]
    expected = ["#SBATCH --job-name=" + NAME, "#SBATCH --account=tml_2026", "#SBATCH --partition=b200_normal_q",
        "#SBATCH --qos=owl_b200_normal_short", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8", "#SBATCH --mem=128G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(root / "extension/logs/extension-joint-memory-v3-%j.out"),
        "#SBATCH --error=" + str(root / "extension/logs/extension-joint-memory-v3-%j.err")]
    parent.shared.require(directives == expected,
        "Exact reviewed oneB200/8CPU/128G/two-hour/no-requeue v3 wrapper required")


def prerequisites(root):
    """Bounded source/public metadata only; exact future pins fail closed."""
    s, m = parent.shared, parent.m
    s.require(all(s._checksum(sha) for sha in (*PINS.values(), PRIOR_SHA256, PLAN_SHA256, INSTALL_SHA256)) and
        type(CPU_ELAPSED_SECONDS) is int and 0 < CPU_ELAPSED_SECONDS <= 1200,
        "Root must first seal terminal CPU65 receipts, exact elapsed seconds and the fixed21-entry snapshot")
    assets, values = {}, {}
    for name, sha in PINS.items():
        path = DIRECTORY + "/" + name
        assets[path] = s._record(root, path)
        s.require(assets[path]["sha256"] == sha, "Pinned CPU65 closeout changed: " + name)
        values[name] = m._read(root / path)
    unit, complete, snapshot = (values[name] for name in ("unit_qa.json", "complete.json", "job_ledger_snapshot.json"))
    s._ledger(snapshot)
    prior = snapshot["jobs"]
    s.require(len(prior) == 21 and value_digest(prior) == PRIOR_SHA256 and ROLE not in prior,
        "Preserve the exact21 prior admissions; the v3GPU role must be new")
    # Only the already hard-pinned reconciled pre-scheduler no-ID entry is
    # exempt. All other prior entries require unique validated numeric IDs.
    s.require(len(s.scheduler_identities(prior)) == 20, "Unknown uncertain/no-ID/duplicate prior admission")
    sources = qa.source_inventory(root)
    codes = {name: item["sha256"] for name, item in sources.items()}
    s._passed_qa(unit, qa.SCHEMA)
    s.require(len(sources) == 120 and unit.get("job_id") == CPU_JOB and unit.get("tests_run") == 65 and
        unit.get("test_counts") == dict(zip(qa.PATTERNS, qa.COUNTS)) and unit.get("source_inventory") == sources and
        unit.get("code_sha256") == codes and unit.get("plan_sha256") == PLAN_SHA256 and
        unit.get("previous_tests_previously_passed") == 450 and unit.get("previous_tests_replayed") is False and
        unit.get("failed_parent_remains_failed") is True and unit.get("actual_gpu_or_memory_admission_established") is False and
        unit.get("private_prepared_file_opened") is False and unit.get("experiment_model_or_tokenizer_loaded") is False and
        unit.get("automatic_downstream") is False and unit.get("compute_proof", {}).get("allocated_cpus") == 2,
        "The exact65 CPU fixtures and source120 proof must precede this separate GPU replay")
    old20 = {role: entry for role, entry in prior.items() if role != qa.ROLE}
    cpu = prior.get(qa.ROLE, {})
    s._entry(cpu, qa.ROLE, gpu_seconds=0, sources=sources, job=CPU_JOB)
    s.require(len(old20) == 20 and value_digest(old20) == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("prior_admissions") == old20 and unit.get("prior_entries_sha256") == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("admission_binding_sha256") == cpu.get("binding_sha256") and
        cpu["binding"].get("expected_native_tests") == 65 and cpu["binding"].get("wall_time_seconds") == 1200 and
        cpu["binding"].get("closeout_reserve_seconds") == RESERVE_SECONDS and cpu["binding"].get("plan_sha256") == PLAN_SHA256,
        "All20 earlier entries and the successful21st CPU admission must remain exact")
    s.validate_accounting(values["accounting.json"], CPU_JOB, name=qa.NAME)
    account = values["accounting.json"]
    s.require(account.get("elapsed_seconds") == CPU_ELAPSED_SECONDS and account.get("allocated_gpus") == 0 and
        all("cpu=2" in row.split("|")[7].split(",") and "mem=8G" in row.split("|")[7].split(",")
            for row in account["stdout"].splitlines() if row), "Exact terminal CPU2/8G/elapsed/zeroGPU proof required")
    s.require(complete.get("schema") == "extension-joint-identity-replay-v3-qa-complete-v1" and complete.get("complete") is True and
        complete.get("job_id") == CPU_JOB and complete.get("tests_run") == 65 and complete.get("gpu_seconds") == 0 and
        complete.get("source_inventory") == sources and complete.get("input_snapshot_unchanged") is True and
        complete.get("plan_sha256") == PLAN_SHA256 and complete.get("actual_gpu_or_memory_admission_established") is False and
        complete.get("automatic_downstream") is False and complete.get("output_sha256") ==
            {name: PINS[name] for name in ("unit_qa.json", "native_qa.json")}, "CPU65 did not complete unchanged")
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
        installed.get("source_inventory") == sources and len(installed.get("parent_source_inventory", {})) == 114 and
        installed.get("parent_source_inventory") == unit.get("parent_source_inventory"), "Installed120/unchanged114 proof differs")
    expected_alias = {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "plan_sha256": qa.JOINT_PLAN["sha256"], "prepared_sha256": qa.PREPARED["sha256"],
        "operational_plan_sha256": PLAN_SHA256, "unit_qa_sha256": PINS["unit_qa.json"],
        "cpu_qa_alias_of_same65_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True}
    s.require(values["native_qa.json"] == expected_alias, "The handoff alias must describe the SAME65 CPU checks without a GPU claim")
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
        "cpu_unit_qa_sha256": PINS["unit_qa.json"], "cpu_qa_job_id": CPU_JOB, "prior_entries_sha256": PRIOR_SHA256,
        "initial_evidence": proof["assets"], "admission_code_sha256": proof["assets"][SELF]["sha256"],
        "requested_gpu_seconds": GPU_SECONDS, "requested_gpu_hours": 2, "wall_time_seconds": GPU_SECONDS,
        "closeout_reserve_seconds": RESERVE_SECONDS, "same_original_scientific_plan_and_prepared_cache": True,
        "automatic_retry": False, "automatic_downstream": False}


def handoff(root, entry, proof):
    path = root / "extension/operations" / (HANDOFF_PREFIX + entry["job_id"] + ".json")
    value = {"schema": "extension-joint-memory-admission-job-v1", "status": "admitted", "job_id": entry["job_id"], "role": ROLE,
        "plan_sha256": qa.JOINT_PLAN["sha256"], "prepared_sha256": qa.PREPARED["sha256"], "qa_sha256": PINS["native_qa.json"],
        "code_sha256": proof["codes"], "binding_sha256": entry["binding_sha256"], "operational_plan_sha256": PLAN_SHA256,
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
        s.require(not same if current is None else not same - {current["job_id"]}, "Untracked/duplicate joint v3 replay")
        s.require(m._read(path) == ledger and prerequisites(root) == proof,
            "Source120/self/public evidence/ledger changed during the live scheduler snapshot")
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
        s.require(not target.exists() and not target.is_symlink(), "Never overwrite an existing v3GPU handoff")
        write_exclusive(target, value)
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            admission_receipt_sha256=s._record(root, target)["sha256"])
        m._write(path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
