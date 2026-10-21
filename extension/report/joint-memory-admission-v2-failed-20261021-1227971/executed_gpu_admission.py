"""Fixed metadata-only admissions for the two reviewed identity-repair replays.

Run only after independent review, from the installed project root. This script
never opens private prepared witnesses, model/tokenizer files or outcomes. Each
kind has one immutable role; any ambiguous submission blocks both kinds until
manual reconciliation. No automatic retry, cancellation or downstream work.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import uuid

from extension import source_generation_submission as parent
from extension import identity_replay_qa as qa
from extension.cpu_feasibility import value_digest


SELF = "extension/operations/submit-identity-replays-20261021.py"
DIRECTORY = "extension/report/identity-replay-qa-20261021-1227308"
QA_RUN = "extension/runs/identity-replay-qa-1227308"
PLAN_SHA256 = "54e33200b3f8a3259bcb6aa6ba99c9d571e215e96b6b70a4cf47ba2cf5263dff"
PRIOR_SHA256 = "80181be7d9dea76e8083b66a509c70b4eef9a87415e3cf115d37dbaf62d172ba"
PINS = {
    "unit_qa.json": "fd790d109f379ce617b111f10062369d221a59db553d597f6cb5bb5ce485a6f6",
    "native_qa.json": "5010058bef28306fb66b046b4e26d949a90eb51df4b4dfd9651d2056bf000dbd",
    "joint_qa.json": "81d2f8c11c2335e087ece8b9107f01f88aed38ca84509af49cc78605e53a28c3",
    "complete.json": "23b1345e23041d80c39271640a822a3c33baa4ba8a28ea10b2f05e9137b4bc5c",
    "accounting.json": "6d3f2b23d9b767ab0f8fc8da2692703aa5da0f2eb383353b0b5a6ff41e18465b",
    "job_ledger_snapshot.json": "f454eb4d4b35f94e8b26938c5538ae708a692569ef67c469276c92058a67099f",
}
KINDS = {
    "native": {"role": "native_model_admission_20261021_v2", "name": "sft-extension-native-model-admission-v2",
        "script": "extension/native_model_admission_v2.slurm", "plan": qa.NATIVE_PLAN,
        "alias": "native_qa.json", "schema": "extension-native-model-admission-job-v1",
        "handoff_prefix": "native-model-admission-v2-job-", "log_prefix": "extension-native-model-v2-"},
    "joint": {"role": "joint_memory_admission_20261021_v2", "name": "sft-extension-joint-memory-admission-v2",
        "script": "extension/joint_memory_admission_v2.slurm", "plan": qa.JOINT_PLAN,
        "alias": "joint_qa.json", "schema": "extension-joint-memory-admission-job-v1",
        "handoff_prefix": "joint-memory-admission-v2-job-", "log_prefix": "extension-joint-memory-v2-"},
}
GPU_SECONDS, RESERVE_SECONDS = 7200, 172800


def guard():
    now = datetime.now(timezone.utc)
    parent.shared.require((parent.m.DEADLINE - now).total_seconds() >= GPU_SECONDS + RESERVE_SECONDS,
        "Keep the full two-hour replay and48-hour closeout reserve")
    return now


def validate_wrapper(root, spec):
    text = parent.m._small_bytes(root / spec["script"], root).decode()
    directives = [line.strip() for line in text.splitlines() if line.lstrip().startswith("#SBATCH")]
    expected = ["#SBATCH --job-name=" + spec["name"], "#SBATCH --account=tml_2026", "#SBATCH --partition=b200_normal_q",
        "#SBATCH --qos=owl_b200_normal_short", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8", "#SBATCH --mem=128G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(root / "extension/logs" / (spec["log_prefix"] + "%j.out")),
        "#SBATCH --error=" + str(root / "extension/logs" / (spec["log_prefix"] + "%j.err"))]
    parent.shared.require(directives == expected, "Exact reviewed oneB200/8CPU/128G/two-hour/no-requeue wrapper required")


def prerequisites(root):
    """All reads here are bounded source/public metadata, never private cache."""
    s, m = parent.shared, parent.m
    assets, values = {}, {}
    for name, sha in PINS.items():
        path = DIRECTORY + "/" + name
        assets[path] = s._record(root, path)
        s.require(assets[path]["sha256"] == sha, "Pinned CPU450 closeout changed: " + name)
        values[name] = m._read(root / path)
    unit, complete, snapshot = (values[name] for name in ("unit_qa.json", "complete.json", "job_ledger_snapshot.json"))
    s._ledger(snapshot)
    prior = snapshot["jobs"]
    s.require(len(prior) == 18 and value_digest(prior) == PRIOR_SHA256 and
        not {spec["role"] for spec in KINDS.values()} & set(prior), "Preserve the exact18 prior admissions")
    # This tested function permits only the one already hard-pinned no-ID
    # pre-scheduler rejection. Every other entry needs a unique numeric ID.
    s.require(len(s.scheduler_identities(prior)) == 17, "Unknown uncertain/no-ID/duplicate prior admission")
    sources = qa.source_inventory(root)
    codes = {name: item["sha256"] for name, item in sources.items()}
    s._passed_qa(unit, qa.SCHEMA)
    s.require(len(sources) == 113 and unit.get("job_id") == "1227308" and unit.get("tests_run") == 450 and
        unit.get("test_counts") == dict(zip(qa.PATTERNS, qa.COUNTS)) and unit.get("source_inventory") == sources and
        unit.get("code_sha256") == codes and unit.get("plan_sha256") == PLAN_SHA256 and
        unit.get("actual_gpu_or_memory_admission_established") is False and unit.get("private_prepared_file_opened") is False and
        unit.get("experiment_model_or_tokenizer_loaded") is False and unit.get("automatic_downstream") is False and
        unit.get("compute_proof", {}).get("allocated_cpus") == 2,
        "The exact450 CPU fixtures and source113 proof must precede any replay")
    old17 = {role: entry for role, entry in prior.items() if role != qa.ROLE}
    cpu = prior.get(qa.ROLE, {})
    s._entry(cpu, qa.ROLE, gpu_seconds=0, sources=sources, job="1227308")
    s.require(len(old17) == 17 and value_digest(old17) == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("prior_admissions") == old17 and unit.get("prior_entries_sha256") == qa.PRIOR_ENTRIES_SHA256 and
        unit.get("admission_binding_sha256") == cpu.get("binding_sha256") and
        cpu["binding"].get("expected_native_tests") == 450 and cpu["binding"].get("wall_time_seconds") == 1200,
        "All17 earlier entries and the successful18th CPU admission must remain exact")
    s.validate_accounting(values["accounting.json"], "1227308", name=qa.NAME)
    account = values["accounting.json"]
    s.require(account.get("elapsed_seconds") == 49 and account.get("allocated_gpus") == 0 and
        all("cpu=2" in row.split("|")[7].split(",") and "mem=8G" in row.split("|")[7].split(",")
            for row in account["stdout"].splitlines() if row), "Actual terminal CPU2/8G/49second proof required")
    s.require(complete.get("schema") == "extension-identity-replay-qa-complete-v1" and complete.get("complete") is True and
        complete.get("job_id") == "1227308" and complete.get("tests_run") == 450 and complete.get("gpu_seconds") == 0 and
        complete.get("source_inventory") == sources and complete.get("input_snapshot_unchanged") is True and
        complete.get("plan_sha256") == PLAN_SHA256 and complete.get("actual_gpu_or_memory_admission_established") is False and
        complete.get("automatic_downstream") is False and complete.get("output_sha256") ==
            {name: PINS[name] for name in ("unit_qa.json", "native_qa.json", "joint_qa.json")}, "CPU450 did not complete unchanged")
    # Recheck the exact public input inventory sealed by CPU450. The allowed
    # namespace excludes all private run caches and model/tokenizer locations.
    old_assets = unit.get("initial_evidence", {})
    allowed = set(sources) | set(qa.PINS) | {qa.INSTALL, "deadline/job_ledger.json"}
    s.require(isinstance(old_assets, dict) and set(old_assets) <= allowed and set(qa.PINS) <= set(old_assets),
        "CPU proof must bind only the known public source/failure metadata")
    for name, expected in old_assets.items():
        actual = s._record(root, name)
        s.require(actual == expected, "A source/failed-parent/preparation metadata input changed: " + name)
        assets[name] = actual
    for spec in KINDS.values():
        alias = values[spec["alias"]]
        schema = "extension-native-model-admission-native-qa-v1" if spec is KINDS["native"] else "extension-joint-memory-admission-native-qa-v1"
        expected = {**unit, "schema": schema, "plan_sha256": spec["plan"]["sha256"], "prepared_sha256": qa.PREPARED["sha256"],
            "operational_plan_sha256": PLAN_SHA256, "unit_qa_sha256": PINS["unit_qa.json"],
            "cpu_qa_alias_of_same450_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True}
        s.require(alias == expected, "Each native/joint alias must describe the SAME450 CPU checks without a GPU claim")
        private_alias = QA_RUN + "/" + spec["alias"]
        assets[private_alias] = s._record(root, private_alias)
        s.require(assets[private_alias]["sha256"] == PINS[spec["alias"]], "Actual handoff QA alias differs from its public closeout")
        validate_wrapper(root, spec)
        assets[spec["plan"]["path"]] = s._record(root, spec["plan"]["path"])
        s.require(assets[spec["plan"]["path"]]["sha256"] == spec["plan"]["sha256"], "Original scientific plan changed")
    assets[qa.PLAN] = s._record(root, qa.PLAN)
    s.require(assets[qa.PLAN]["sha256"] == PLAN_SHA256, "Operational repair plan changed")
    assets["deadline/job_ledger.json"] = s._record(root, "deadline/job_ledger.json")
    s.require(assets["deadline/job_ledger.json"]["sha256"] == qa.HISTORICAL_LEDGER_SHA256, "Historical GPU ledger changed")
    assets[SELF] = s._record(root, SELF)
    return {"sources": sources, "codes": codes, "assets": assets, "prior": prior}


def binding_for(root, kind, proof):
    spec = KINDS[kind]
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / spec["script"]), QA_RUN]
    return {"role": spec["role"], "kind": kind, "command": command, "source_sha256": proof["sources"],
        "plan_sha256": spec["plan"]["sha256"], "prepared_sha256": qa.PREPARED["sha256"], "qa_sha256": PINS[spec["alias"]],
        "operational_plan_sha256": PLAN_SHA256, "cpu_unit_qa_sha256": PINS["unit_qa.json"], "cpu_qa_job_id": "1227308",
        "prior_entries_sha256": PRIOR_SHA256, "initial_evidence": proof["assets"],
        "admission_code_sha256": proof["assets"][SELF]["sha256"], "requested_gpu_seconds": GPU_SECONDS, "requested_gpu_hours": 2,
        "wall_time_seconds": GPU_SECONDS, "closeout_reserve_seconds": RESERVE_SECONDS,
        "same_original_scientific_plan_and_prepared_cache": True, "automatic_retry": False, "automatic_downstream": False}


def handoff(root, kind, entry, proof):
    spec = KINDS[kind]
    path = root / "extension/operations" / (spec["handoff_prefix"] + entry["job_id"] + ".json")
    value = {"schema": spec["schema"], "status": "admitted", "job_id": entry["job_id"], "role": spec["role"],
        "plan_sha256": spec["plan"]["sha256"], "prepared_sha256": qa.PREPARED["sha256"], "qa_sha256": PINS[spec["alias"]],
        "code_sha256": proof["codes"], "binding_sha256": entry["binding_sha256"], "operational_plan_sha256": PLAN_SHA256,
        "cpu_unit_qa_sha256": PINS["unit_qa.json"], "cpu_qa_job_id": "1227308",
        "admission_code_sha256": proof["assets"][SELF]["sha256"], "automatic_retry": False, "automatic_downstream": False}
    return path, value


def validate_history(root, ledger, proof):
    s = parent.shared
    s._ledger(ledger)
    allowed = {spec["role"] for spec in KINDS.values()}
    s.require(set(ledger["jobs"]) <= set(proof["prior"]) | allowed and
        all(ledger["jobs"].get(role) == value for role, value in proof["prior"].items()), "Unknown/rewritten prior admission")
    for kind, spec in KINDS.items():
        entry = ledger["jobs"].get(spec["role"])
        if entry is None:
            continue
        s._entry(entry, spec["role"], gpu_seconds=GPU_SECONDS, sources=proof["sources"])
        binding = binding_for(root, kind, proof)
        s.require(entry.get("binding") == binding and entry.get("binding_sha256") == value_digest(binding),
            "Existing/uncertain/rejected replay requires manual reconciliation, never retry")
        target, expected = handoff(root, kind, entry, proof)
        s.require(target.is_file() and not target.is_symlink() and parent.m._read(target) == expected and
            entry.get("admission_receipt_sha256") == s._record(root, target)["sha256"], "An existing submitted handoff must remain immutable")
    s.scheduler_identities(ledger["jobs"])


def write_exclusive(path, value):
    """Publish a fully written handoff atomically without replacing any path."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=tuple(KINDS))
    args = parser.parse_args()
    root, s, m = Path.cwd().resolve(), parent.shared, parent.m
    s.require(Path(__file__).resolve() == root / SELF, "Run the exact installed one-use bootstrap from project root")
    guard()
    path, lockpath = root / "extension/job_ledger.json", root / "extension/job_ledger.lock"
    s.require(not path.is_symlink() and not lockpath.is_symlink(), "No ledger aliases")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proof, ledger = prerequisites(root), m._read(path)
        validate_history(root, ledger, proof)
        spec = KINDS[args.kind]
        binding = binding_for(root, args.kind, proof)
        checksum = value_digest(binding)
        snapshot = s.queue_snapshot(subprocess.run)
        parent.known_queue(root, snapshot, ledger["jobs"])
        for kind, candidate in KINDS.items():
            same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == candidate["name"]}
            existing = ledger["jobs"].get(candidate["role"])
            s.require(not same if existing is None else not same - {existing["job_id"]}, "Untracked/duplicate native or joint v2 replay")
        s.require(m._read(path) == ledger and prerequisites(root) == proof,
            "Source113/self/public evidence/ledger changed during the live scheduler snapshot")
        current = ledger["jobs"].get(spec["role"])
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
        ledger["jobs"][spec["role"]] = current
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
        s.require(m._read(path) == ledger and prerequisites(root) == proof, "Source/self/evidence/ledger changed after scheduler acceptance")
        target, value = handoff(root, args.kind, current, proof)
        s.require(not target.exists() and not target.is_symlink(), "Never overwrite an existing GPU handoff")
        write_exclusive(target, value)
        current.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            admission_receipt_sha256=s._record(root, target)["sha256"])
        m._write(path, ledger)
        print(job, flush=True)


if __name__ == "__main__":
    main()
