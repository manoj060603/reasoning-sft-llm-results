"""Unadmitted metadata-only admission for the versioned original collector.

Unresolved seals deliberately block admission until native allocator97 is
terminal and the manually admitted v4 GPU entry belongs to one exact full
snapshot. No private prompt cache, model, labels, inference, or grader is read.
"""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

from extension import submission as m
from extension import joint_gpu_admission as shared
from extension import source_generation_submission as predecessor
from extension import source_generation_v2 as generation
from extension.cpu_feasibility import Evidence, value_digest

ROLE = generation.ROLE
QA_ROLE = "source_generation_v2_qa_20261021_v1"
NAME = "sft-extension-original-source-calibration-v2"
QA_NAME = "sft-extension-source-generation-v2-qa"
SCRIPT = "extension/source_generation_v2.slurm"
QA_SCRIPT = "extension/source_generation_v2_qa.slurm"
BOOTSTRAP = "extension/operations/submit-source-generation-v2-qa-20261021.py"
METADATA_TESTS, COLLECTOR_TESTS = 20, 40
CPU_SECONDS, GPU_SECONDS, RESERVE_SECONDS = 1200, 7200, 172800
ALLOCATOR_BOOTSTRAPS = frozenset({"extension/operations/submit-joint-allocator-v4-qa-20261021.py",
    "extension/operations/submit-joint-allocator-v4-20261021.py"})
known_queue = predecessor.known_queue


def metadata_inventory(root):
    return {name: shared._record(root, name) for name in sorted(generation.METADATA_SOURCES)}


def guard(now=None):
    now = datetime.now(timezone.utc) if now is None else now
    shared.require(isinstance(now, datetime) and now.tzinfo is not None and
        (m.DEADLINE - now).total_seconds() >= GPU_SECONDS + RESERVE_SECONDS,
        "Keep the full finite two-hour job and48-hour closeout reserve")
    return now


def _public_path(value):
    shared.require(type(value) is str and value.startswith("extension/report/") and
        not Path(value).is_absolute() and ".." not in Path(value).parts, "Only sealed public report metadata is allowed")
    return value


def validate_seals(value):
    """Fail closed on every unresolved value; never infer or extend history."""
    shared.require(type(value) is dict and set(value) == {"prior_snapshot", "prior_entries", "prior_entries_sha256",
        "allocator97", "allocator_v4", "executed_bootstraps"}, "Exact admission seals are required")
    snapshot, cpu, gpu = value["prior_snapshot"], value["allocator97"], value["allocator_v4"]
    shared.require(type(snapshot) is dict and set(snapshot) == {"path", "sha256"} and
        shared._checksum(snapshot["sha256"]) and type(value["prior_entries"]) is int and value["prior_entries"] >= 24 and
        shared._checksum(value["prior_entries_sha256"]), "Unresolved final full snapshot; root must seal actual completed97 plus manualv4 history")
    _public_path(snapshot["path"])
    shared.require(type(cpu) is dict and set(cpu) == {"directory", "job_id", "elapsed_seconds", "pins"} and
        shared._numeric_job(cpu["job_id"]) and type(cpu["elapsed_seconds"]) is int and 0 < cpu["elapsed_seconds"] <= CPU_SECONDS and
        type(cpu["pins"]) is dict and set(cpu["pins"]) == {"unit_qa.json", "native_qa.json", "complete.json", "accounting.json", "source_install_receipt.json"} and
        all(shared._checksum(sha) for sha in cpu["pins"].values()), "Unresolved native97 terminal evidence")
    _public_path(cpu["directory"])
    shared.require(type(gpu) is dict and set(gpu) == {"role", "job_id", "admission"} and
        type(gpu["role"]) is str and gpu["role"] and gpu["role"] not in {ROLE, QA_ROLE} and
        shared._numeric_job(gpu["job_id"]) and gpu["job_id"] != cpu["job_id"] and
        type(gpu["admission"]) is dict and set(gpu["admission"]) == {"path", "sha256"} and
        shared._checksum(gpu["admission"]["sha256"]), "Unresolved manually admitted allocatorv4 identity")
    path = gpu["admission"]["path"]
    shared.require(type(path) is str and path.startswith(("extension/report/", "extension/operations/")) and
        not Path(path).is_absolute() and ".." not in Path(path).parts, "Allocator handoff must remain public metadata")
    bootstraps = value["executed_bootstraps"]
    shared.require(type(bootstraps) is dict and set(bootstraps) == ALLOCATOR_BOOTSTRAPS and
        all(shared._checksum(sha) for sha in bootstraps.values()), "Unresolved actually executed allocator bootstrap identities")
    return value


def validate_metadata_plan(plan):
    generation.validate_plan(plan)
    shared.require(plan.get("metadata_expected_tests") == METADATA_TESTS and plan.get("expected_total_tests") == 60 and
        plan.get("cpu_role") == QA_ROLE and plan.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True},
        "Only the prospective40+20 CPU scope and fixed outside-inventory bootstrap are allowed")
    return validate_seals(plan.get("admission_seals"))


def _allocator_evidence(root, seals, read, prior):
    from extension import joint_allocator_v4_qa as allocator
    cpu = seals["allocator97"]
    values = {name: read(cpu["directory"] + "/" + name, sha) for name, sha in cpu["pins"].items()}
    unit, complete = values["unit_qa.json"], values["complete.json"]
    sources = allocator.source_inventory(root)
    installed = values["source_install_receipt.json"]
    shared.require(installed.get("installed_on_arc") is True and installed.get("source_inventory") == sources and
        installed.get("parent_source_inventory") == unit.get("parent_source_inventory"), "Actual installed128 source proof differs")
    shared._passed_qa(unit, allocator.SCHEMA)
    shared.require(len(sources) == 128 and unit.get("job_id") == cpu["job_id"] and unit.get("tests_run") == 97 and
        unit.get("test_counts") == dict(zip(allocator.PATTERNS, allocator.COUNTS)) and
        unit.get("source_inventory") == sources and unit.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        unit.get("plan_sha256") == sources[allocator.PLAN]["sha256"] and
        unit.get("actual_gpu_or_memory_admission_established") is False and unit.get("private_prepared_file_opened") is False and
        unit.get("experiment_model_or_tokenizer_loaded") is False and unit.get("effective_expandable_segments_interface_established") is False and
        complete.get("complete") is True and complete.get("job_id") == cpu["job_id"] and complete.get("tests_run") == 97 and
        complete.get("gpu_seconds") == 0 and complete.get("source_inventory") == sources and
        complete.get("plan_sha256") == unit.get("plan_sha256") and
        complete.get("input_snapshot_unchanged") is True and complete.get("actual_gpu_or_memory_admission_established") is False and
        complete.get("output_sha256") == {name: cpu["pins"][name] for name in ("unit_qa.json", "native_qa.json")},
        "Exact terminal native97/source128 proof is required without a GPU claim")
    shared.validate_accounting(values["accounting.json"], cpu["job_id"], name=allocator.NAME)
    shared.require(values["accounting.json"].get("elapsed_seconds") == cpu["elapsed_seconds"], "Native97 accounting elapsed differs")
    cpu_entry = prior.get(allocator.ROLE, {})
    shared._entry(cpu_entry, allocator.ROLE, gpu_seconds=0, sources=sources, job=cpu["job_id"])
    earlier = unit.get("prior_admissions", {})
    shared.require(type(earlier) is dict and value_digest(earlier) == allocator.PRIOR_ENTRIES_SHA256 and
        all(prior.get(name) == entry for name, entry in earlier.items()) and
        unit.get("admission_binding_sha256") == cpu_entry.get("binding_sha256") and
        unit.get("prior_entries_sha256") == value_digest(earlier), "Native97 admission/history binding differs")
    alias = {**unit, "schema": "extension-joint-memory-admission-native-qa-v1", "plan_sha256": allocator.JOINT_PLAN["sha256"],
        "prepared_sha256": allocator.PREPARED["sha256"], "operational_plan_sha256": unit["plan_sha256"],
        "unit_qa_sha256": cpu["pins"]["unit_qa.json"], "cpu_qa_alias_of_same97_tests": True,
        "alias_is_not_independent_gpu_or_numerical_evidence": True}
    shared.require(values["native_qa.json"] == alias, "Allocator alias must describe the same97 CPU checks")
    gpu = seals["allocator_v4"]
    gpu_entry = prior.get(gpu["role"], {})
    shared._entry(gpu_entry, gpu["role"], gpu_seconds=GPU_SECONDS, sources=sources, job=gpu["job_id"])
    for entry, script in ((cpu_entry, "extension/joint_allocator_v4_qa.slurm"), (gpu_entry, "extension/joint_memory_allocator_v4.slurm")):
        wrappers = [part for part in entry.get("command", []) if isinstance(part, str) and part.endswith(".slurm")]
        shared.require(len(wrappers) == 1 and (root / wrappers[0]).resolve() == root / script, "Allocator admission used an unexpected wrapper")
    handoff = read(gpu["admission"]["path"], gpu["admission"]["sha256"])
    shared.require(handoff.get("status") == "admitted" and handoff.get("job_id") == gpu["job_id"] and
        handoff.get("binding_sha256") == gpu_entry.get("binding_sha256") and
        gpu_entry.get("admission_receipt_sha256") == gpu["admission"]["sha256"] and
        handoff.get("code_sha256") == unit["code_sha256"] and handoff.get("qa_sha256") == cpu["pins"]["native_qa.json"] and
        handoff.get("allocator_policy") == allocator.worker.ALLOCATOR_POLICY and
        handoff.get("operational_plan_sha256") == unit["plan_sha256"] and
        handoff.get("automatic_downstream") is False, "Manual allocatorv4 handoff must bind the actual97 proof")
    for path, sha in seals["executed_bootstraps"].items():
        read(path, sha, False)
    shared.require(cpu_entry["binding"].get("admission_code_sha256") == seals["executed_bootstraps"][allocator.BOOTSTRAP] and
        gpu_entry["binding"].get("admission_code_sha256") == seals["executed_bootstraps"]["extension/operations/submit-joint-allocator-v4-20261021.py"],
        "Actually executed outside-inventory bootstrap identities differ")
    return sources


def initial_evidence(root):
    root, assets = Path(root).resolve(), {}
    def read(name, checksum=None, json_value=True):
        assets[name] = shared._record(root, name)
        shared.require(checksum is None or assets[name]["sha256"] == checksum, "Pinned metadata changed: " + name)
        return m._read(root / name) if json_value else None
    plan = read(generation.PLAN)
    seals = validate_metadata_plan(plan)
    snapshot = read(seals["prior_snapshot"]["path"], seals["prior_snapshot"]["sha256"])
    shared._ledger(snapshot)
    initial = snapshot["jobs"]
    shared.require(len(initial) == seals["prior_entries"] and value_digest(initial) == seals["prior_entries_sha256"] and
        ROLE not in initial and QA_ROLE not in initial and len(shared.scheduler_identities(initial)) == len(initial) - 1,
        "The one exact final full snapshot, including native97 and manualv4, must remain unchanged")
    core, metadata = generation.source_inventory(root), metadata_inventory(root)
    shared.require(len(core) == 124 and len(metadata) == 4 and not set(core) & set(metadata), "Exact124+4 namespaces required")
    science = read(generation.SCIENTIFIC_PARENT["path"], generation.SCIENTIFIC_PARENT["sha256"])
    generation.scientific_parent.validate_plan(science)
    identity_evidence = Evidence()
    generation._identity_parent_evidence(root, identity_evidence, core)
    assets.update({str(Path(path).relative_to(root)): value for path, value in identity_evidence.files.items()})
    parents = {name: read(generation.PARENT_DIRECTORY + "/" + name, sha) for name, sha in generation.PARENT_PINS.items()}
    complete = parents["complete.json"]
    shared._passed_qa(parents["unit_qa.json"], generation.prepared_stage.QA_SCHEMA)
    shared.require(complete.get("complete") is True and complete.get("job_id") == "1226240" and complete.get("tests_run") == 478 and
        complete.get("gold_self_comparisons_passed") == 32 and complete.get("native_gold_mutation_invariance_passed") is True and
        complete.get("input_snapshot_unchanged") is True and complete.get("native_qa_sha256") == generation.PARENT_PINS["unit_qa.json"] and
        complete.get("output_sha256", {}).get("calibration_prompt_cache.json") == generation.CACHE["sha256"], "Exact source-only parent478/cache binding required")
    shared.validate_accounting(parents["accounting.json"], "1226240", name="sft-extension-source-readiness-v2-calibration-retry")
    allocator_sources = _allocator_evidence(root, seals, read, initial)
    for name, record in {**core, **metadata, **allocator_sources}.items():
        shared.require(shared._record(root, name) == record, "Bounded source changed: " + name)
    for name in (BOOTSTRAP, "deadline/job_ledger.json"):
        assets[name] = shared._record(root, name)
    shared.require(assets["deadline/job_ledger.json"]["sha256"] == shared.qa.HISTORICAL_LEDGER_SHA256, "Historical ledger changed")
    return {"initial": initial, "initial_sha256": seals["prior_entries_sha256"], "assets": assets, "sources": core,
        "metadata": metadata, "allocator_sources": allocator_sources, "plan": plan, "plan_sha256": assets[generation.PLAN]["sha256"]}


def validate_wrapper(root):
    text = m._small_bytes(Path(root) / SCRIPT, Path(root)).decode()
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "b200_normal_q", "qos": "owl_b200_normal_short",
        "nodes": "1", "ntasks": "1", "gres": "gpu:1", "cpus-per-task": "8", "mem": "128G", "time": "02:00:00", "no-requeue": ""}
    allowed, found, header = set(expected) | {"output", "error"}, {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"): header = False
            continue
        shared.require(header, "Late SBATCH directive")
        parts = shlex.split(line.lstrip()[7:].strip())
        shared.require(bool(parts), "Empty directive")
        key, separator, arg = parts[0].partition("=")
        shared.require(key.startswith("--") and key[2:] in allowed and key[2:] not in found and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/duplicate/ambiguous directive")
        found[key[2:]] = arg if separator else parts[1] if len(parts) == 2 else ""
    shared.require(set(found) == allowed and all(found[key] == value for key, value in expected.items()), "Exact finite singleB200 wrapper required")
    for key in ("output", "error"):
        path = Path(found[key])
        shared.require(path.is_absolute() and ".." not in path.parts and path.parent == Path(root) / "extension/logs", "Bounded extension logs required")
    return found


def validate_history(ledger, proof, cpu_job):
    shared._ledger(ledger)
    initial = proof["initial"]
    shared.require(set(ledger["jobs"]) <= set(initial) | {QA_ROLE, ROLE} and all(ledger["jobs"].get(key) == value for key, value in initial.items()),
        "Unknown or rewritten fixed snapshot admission")
    current = ledger["jobs"].get(QA_ROLE, {})
    shared._entry(current, QA_ROLE, gpu_seconds=0, sources=proof["sources"], job=cpu_job)
    shared.require(current["binding"].get("prior_entries_sha256") == proof["initial_sha256"] and
        current["binding"].get("metadata_source_inventory") == proof["metadata"] and
        current["binding"].get("plan_sha256") == proof["plan_sha256"] and
        current["binding"].get("admission_code_sha256") == proof["assets"][BOOTSTRAP]["sha256"] and
        current["binding"].get("initial_evidence", {}).get(BOOTSTRAP) == proof["assets"][BOOTSTRAP] and
        current["binding"].get("closeout_reserve_seconds") == RESERVE_SECONDS and current["binding"].get("automatic_downstream") is False and
        current["binding"].get("wall_time_seconds") == CPU_SECONDS and
        current["binding"].get("expected_native_tests") == 60, "CPU60 must bind the exact final snapshot and sources")
    shared.require(cpu_job not in shared.scheduler_identities(initial), "CPU job ID reused an earlier allocation")
    return {**initial, QA_ROLE: current}


def current_cpu_admission(root, proof, compute):
    ledger = m._read(Path(root) / "extension/job_ledger.json")
    prior = validate_history(ledger, proof, compute["job_id"])
    shared.require(set(ledger["jobs"]) == set(proof["initial"]) | {QA_ROLE}, "Only the newly admitted CPU role may extend the fixed snapshot")
    return ledger, prior[QA_ROLE]


def validate_native_qa(qa, proof, prior):
    counts = {"test_source_generation_identity_v2.py": 40, "test_source_generation_v2_submission.py": 20}
    current = prior[QA_ROLE]
    shared.require(qa.get("tests_run") == 60 and qa.get("test_counts") == counts and qa.get("parent_tests_previously_passed") == 478 and
        qa.get("parent_tests_replayed") is False and qa.get("identity_parent_tests_previously_passed") == 65 and
        qa.get("identity_parent_tests_replayed") is False and qa.get("allocator_tests_previously_passed") == 97 and
        qa.get("allocator_tests_replayed") is False and qa.get("collector_tests") == 40 and qa.get("metadata_tests") == 20 and
        qa.get("source_inventory") == proof["sources"] and qa.get("code_sha256") == {name: value["sha256"] for name, value in proof["sources"].items()} and
        qa.get("metadata_source_inventory") == proof["metadata"] and qa.get("allocator_source_inventory") == proof["allocator_sources"] and
        qa.get("plan_sha256") == proof["plan_sha256"] and qa.get("cache_sha256") == generation.CACHE["sha256"] and
        qa.get("parent_complete_sha256") == generation.PARENT_PINS["complete.json"] and
        qa.get("scientific_parent_sha256") == generation.SCIENTIFIC_PARENT["sha256"] and
        qa.get("identity_parent_complete_sha256") == generation.IDENTITY_PINS["complete.json"] and
        qa.get("identity_helper_sha256") == generation.IDENTITY_HELPER["sha256"] and
        qa.get("actual_vllm_interface_established") is False and qa.get("wrapper_syntax_passed") is True and
        qa.get("admission_binding_sha256") == current["binding_sha256"] and
        qa.get("cpu_bootstrap_sha256") == proof["assets"][BOOTSTRAP]["sha256"], "Exact native60/source/parent/current-admission bindings required")
    shared.require(current.get("source_sha256") == proof["sources"] == current["binding"].get("source_sha256") and
        current["binding"].get("metadata_source_inventory") == proof["metadata"], "Current CPU source binding changed")
    return qa


def load_evidence(root, qa_run, qa_sha256, ledger):
    root, proof = Path(root).resolve(), initial_evidence(root)
    directory = shared._inside(root, qa_run, runs=True, directory=True)
    path = directory / "native_qa.json"
    shared.require(shared._checksum(qa_sha256) and shared._record(root, path)["sha256"] == qa_sha256, "Caller-pinned native60 QA required")
    qa = m._read(path)
    shared._passed_qa(qa, generation.QA_SCHEMA)
    shared.require(directory.name == "source-generation-v2-qa-" + qa["job_id"] and directory.parent == root / "extension/runs", "Exact CPU run namespace required")
    prior = validate_history(ledger, proof, qa["job_id"])
    validate_native_qa(qa, proof, prior)
    account_path = directory / "accounting.json"
    account = m._read(account_path)
    shared.validate_accounting(account, qa["job_id"], name=QA_NAME)
    shared.require(account["elapsed_seconds"] <= CPU_SECONDS, "CPU60 exceeded finite ceiling")
    proof["assets"].update({str(item.relative_to(root)): shared._record(root, item) for item in (path, account_path)})
    validate_wrapper(root)
    return {**proof, "prior": prior, "qa_run": str(directory.relative_to(root)), "qa_sha256": qa_sha256, "qa_value": qa}


def binding_for(root, proof):
    return {"role": ROLE, "command": ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT), proof["qa_run"] + "/native_qa.json"],
        "source_sha256": proof["sources"], "metadata_source_inventory": proof["metadata"], "plan_sha256": proof["plan_sha256"],
        "qa_sha256": proof["qa_sha256"], "cache_sha256": generation.CACHE["sha256"], "profile_sha256": generation.PROFILE_SHA256,
        "parent_complete_sha256": generation.PARENT_PINS["complete.json"], "scientific_parent_sha256": generation.SCIENTIFIC_PARENT["sha256"],
        "identity_parent_complete_sha256": generation.IDENTITY_PINS["complete.json"], "identity_helper_sha256": generation.IDENTITY_HELPER["sha256"],
        "model_role": "original", "authorize_first_interface_checking_generation": True, "requested_gpu_seconds": GPU_SECONDS,
        "requested_gpu_hours": 2, "wall_time_seconds": GPU_SECONDS, "closeout_reserve_seconds": RESERVE_SECONDS,
        "prior_entries_sha256": value_digest(proof["prior"]), "initial_evidence": proof["assets"], "automatic_retry": False, "automatic_downstream": False}


def handoff(root, entry, proof):
    path = root / "extension/operations" / ("source-generation-original-v2-job-" + entry["job_id"] + ".json")
    value = {"schema": generation.ADMISSION_SCHEMA, "status": "admitted", "job_id": entry["job_id"], "plan_sha256": proof["plan_sha256"],
        "qa_sha256": proof["qa_sha256"], "code_sha256": {name: item["sha256"] for name, item in proof["sources"].items()},
        "binding_sha256": entry["binding_sha256"], "cache_sha256": generation.CACHE["sha256"],
        "scientific_parent_sha256": generation.SCIENTIFIC_PARENT["sha256"], "identity_parent_complete_sha256": generation.IDENTITY_PINS["complete.json"],
        "identity_helper_sha256": generation.IDENTITY_HELPER["sha256"], "authorize_first_interface_checking_generation": True,
        "automatic_retry": False, "automatic_downstream": False}
    return path, value


def write_exclusive(path, value):
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(m._canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def submit(root, qa_run, qa_sha256, *, now=None, runner=None):
    root, runner = Path(root).resolve(), subprocess.run if runner is None else runner
    guard(now)
    path, lockpath = root / "extension/job_ledger.json", root / "extension/job_ledger.lock"
    shared.require(not path.is_symlink() and not lockpath.is_symlink(), "No ledger aliases")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = m._read(path)
        proof = load_evidence(root, qa_run, qa_sha256, ledger)
        binding = binding_for(root, proof)
        checksum = value_digest(binding)
        snapshot = shared.queue_snapshot(runner)
        existing = ledger["jobs"].get(ROLE)
        known_queue(root, snapshot, {**proof["prior"], **({ROLE: existing} if existing is not None else {})})
        same = {row["job_id"] for row in snapshot["jobs"] if row["name"] == NAME}
        shared.require(m._read(path) == ledger and load_evidence(root, qa_run, qa_sha256, ledger) == proof, "Source/evidence/ledger changed during queue snapshot")
        if existing is not None:
            shared._entry(existing, ROLE, gpu_seconds=GPU_SECONDS, sources=proof["sources"])
            shared.require(existing.get("binding") == binding and existing.get("binding_sha256") == checksum and not same - {existing["job_id"]}, "Existing/uncertain/rejected generation requires manual reconciliation, never retry")
            target, expected = handoff(root, existing, proof)
            shared.require(target.is_file() and not target.is_symlink() and m._read(target) == expected and
                existing.get("admission_receipt_sha256") == shared._record(root, target)["sha256"], "Existing handoff must remain exact")
            shared.check_capacity(snapshot, already_submitted=True)
            return existing["job_id"]
        shared.require(not same, "Untracked generation requires reconciliation")
        shared.check_capacity(snapshot)
        when = guard(now)
        environment, removed = m._submission_environment()
        entry = {"status": "submitting", "binding": binding, "binding_sha256": checksum, "source_sha256": proof["sources"],
            "command": binding["command"], "requested_gpu_seconds": GPU_SECONDS, "requested_gpu_hours": 2,
            "requested_at_utc": when.isoformat(), "scheduler_snapshot": snapshot, "removed_scheduler_environment_keys": removed}
        ledger["jobs"][ROLE] = entry
        m._write(path, ledger)
        try: response = runner(binding["command"], capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status="uncertain", error_type=type(error).__name__)
            m._write(path, ledger)
            raise
        entry.update(scheduler_returncode=response.returncode, scheduler_stdout=str(response.stdout)[-2000:], scheduler_stderr=str(response.stderr)[-2000:])
        match = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?", response.stdout.strip()) if isinstance(response.stdout, str) else None
        if type(response.returncode) is not int or response.returncode != 0 or match is None or not isinstance(response.stderr, str) or response.stderr.strip():
            entry["status"] = "uncertain"
            if match: entry["possible_job_id"] = match.group(1)
            m._write(path, ledger)
            raise m.SubmissionError("Preserve rejected/ambiguous generation without retry")
        job = match.group(1)
        entry.update(status="uncertain", job_id=job)
        m._write(path, ledger)
        shared.require(job not in shared.scheduler_identities(proof["prior"]) and job not in {row["job_id"] for row in snapshot["jobs"]}, "Scheduler identity collision")
        shared.require(m._read(path) == ledger and load_evidence(root, qa_run, qa_sha256, ledger) == proof, "Source/evidence/ledger changed after scheduler acceptance")
        target, value = handoff(root, entry, proof)
        shared.require(not target.exists() and not target.is_symlink(), "Never overwrite an existing GPU handoff")
        write_exclusive(target, value)
        entry.update(status="submitted", submitted_at_utc=datetime.now(timezone.utc).isoformat(), admission_receipt_sha256=shared._record(root, target)["sha256"])
        m._write(path, ledger)
        return job


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--qa-run", required=True)
    parser.add_argument("--qa-sha256", required=True)
    args = parser.parse_args()
    print(submit(args.root, args.qa_run, args.qa_sha256), flush=True)


if __name__ == "__main__": main()
