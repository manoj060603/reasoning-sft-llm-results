"""Prospective CPU-only450 checks for an explicit GPU-identity repair.

This stage opens source/config and public receipt metadata only. It never opens
the private prepared witnesses or model/tokenizer bytes, queries a GPU, submits
a job, or admits downstream work. Both QA aliases describe the SAME CPU tests;
neither alias establishes a real CUDA/model/memory envelope.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import inside
from extension import native_model_admission_v2 as native
from extension import joint_memory_admission_v2 as joint


ROLE = "identity_replay_qa_20261021_v1"
NAME = "sft-extension-identity-replay-qa"
SCHEMA = "extension-identity-replay-unit-qa-v1"
PLAN = "extension/identity_replay_plan.json"
PLAN_SCHEMA = "extension-identity-replay-operational-plan-v1"
INSTALL = "extension/identity_replay_source_receipt.json"
BOOTSTRAP = "extension/operations/submit-identity-replay-qa-20261021.py"
HISTORICAL_LEDGER_SHA256 = "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d"
PRIOR_ENTRIES_SHA256 = "6f2091d2b86d3866a1527e01b51f9fe7850ee0bf1c3d82aef71f9fce980cbff1"
NATIVE_PLAN = {"path": "extension/native_model_admission_plan.json", "sha256": "94b6d9bc5e46acc0d5a9e3f409d88544f7e091873c2150af705b92c6313b758e"}
JOINT_PLAN = {"path": "extension/joint_memory_admission_plan.json", "sha256": "b926be22779d59af501afddd10e7129f105b68b723b9bc8386be0aa397ddfc24"}
PREPARED = {"path": "extension/runs/native-smoke-preparation-1224497/prepared.json", "sha256": "d1bb1875a293c71f2a108df4b287b3bb2ab35b3168cd6505b5bd91a98fd66457"}
PARENT = "extension/report/source-generation-qa-20261021-1227150"
PINS = {
    "extension/operations/native-model-admission-job-1224577.json": "82be324fee430aa6bf781d4ffa0fc1c9bc70a4d5534c0f4d25574064ea8a8dd0",
    "extension/operations/joint-memory-admission-job-1225727.json": "13fe2386f951daa3b2dc4eb4133d5d5b65df7fc63eee2af8ec1725d65d77c7e2",
    PARENT + "/native_qa.json": "bc960fb04d9309e8a6b836f45741582674a62603ae8ed8f04d10728c241393eb",
    PARENT + "/complete.json": "e5a3e3b8257ef3e3d5de4ab5824fd2f643d97a9ab7b970cd64364bccfe5d4e1f",
    PARENT + "/accounting.json": "65d65b207bb3228b510c4ad055e954a311b777268548f7a0b9f5de56c5680205",
    PARENT + "/job_ledger_snapshot.json": "414192e5298a83351af02b05f413482ce025d698c0a7aa342840e489adfd9857",
    "extension/report/native-smoke-preparation-20261021-1224497/complete.json": "b7d8dc41b6fd90befc6a85130b3904245aea6dbd2cafc3209941d647a3ab8df5",
    "extension/report/native-model-admission-failed-20261021-1224577/accounting.json": "ac2ceb1b4e8b821976bf6960835f5476a17b62681228a2f8b5783cf4edf1ff7f",
    "extension/report/native-model-admission-failed-20261021-1224577/failure.json": "1ecd15e8016e2dec09710003bf7608e9b78a0ad801460f0e079869ce4e0acb10",
    "extension/report/native-model-admission-failed-20261021-1224577/first_case_failure.json": "d929e2117ff9bdaef1e08429f7fc278cbb2af10d2f2e217dc813281d6235ace3",
    "extension/report/joint-memory-admission-failed-20261021-1225727/accounting.json": "9094575d6b9428c31c3f58bc7960c5b4121c2d3b35403a66b492b70019fb78e6",
    "extension/report/joint-memory-admission-failed-20261021-1225727/failure.json": "9318ce3ffcb4220ac9dea91d7dbfb1e6ae5228998d0ae1c627a721f9e616b265",
    "extension/report/joint-memory-admission-failed-20261021-1225727/first_case_failure.json": "921d3bc7abe327b1a54252d450b6ee8bc01bed9343df63dba48b3185c6a598c6",
}
IMPLEMENTATION = {
    "extension/gpu_identity_v2.py": "9fcaea73bb8c77a0f88a53ee3c7615f953a6480f085a041b50cc6df3658280d1",
    "tests/test_gpu_identity_v2.py": "b487956a696bf1d8afff7c7c61a11f129b2002925d255771e6fe52a3395eab88",
    "extension/native_model_admission_v2.py": "93e5a7ebdb009c086e0a4675fcc5b5d2539a5028176f9560aa3dc499b7c8656a",
    "extension/joint_memory_admission_v2.py": "382c661b7cfa5eb2ba18ee4bd91c553efe5d5a9349b1d6d3322b362795472abe",
    "tests/test_native_identity_replay.py": "b5c2e15a82717daafcde2dad4896a684cd30be224304463e5dfa720436323fe9",
}
NEW = frozenset(IMPLEMENTATION) | frozenset({PLAN, "extension/identity_replay_qa.py", "extension/identity_replay_qa.slurm",
    "extension/native_model_admission_v2.slurm", "extension/joint_memory_admission_v2.slurm"})
PATTERNS = ("test_extension_*.py", "test_labeled_*.py", "test_preserving_*.py", "test_development_*.py",
    "test_preservation_development_seal_v2.py", "test_preservation_retry_submission.py", "test_preservation_data.py",
    "test_native_model_admission.py", "test_smoke_preparation.py", "test_smoke_submission.py", "test_pilot_driver.py",
    "test_joint_memory_admission.py", "test_gpu_identity_v2.py", "test_native_identity_replay.py")
COUNTS = (64, 44, 36, 48, 17, 21, 30, 36, 21, 24, 44, 25, 24, 16)
NOT_EXECUTED = ("source-grader/readiness tests", "collector tests", "actual private prepared cache", "original weights/tokenizer", "GPU/model/memory diagnostics")


def _record(root, name):
    path = inside(Path(root).resolve(), name)
    require(path.is_file() and path.suffix in {".py", ".json", ".slurm"} and 0 < path.stat().st_size <= 2 << 20,
        "Only explicit bounded source or public metadata files are permitted")
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def _parent_sources(root):
    path = inside(root, PARENT + "/native_qa.json")
    require(digest(path) == PINS[PARENT + "/native_qa.json"], "Pinned parent CPU48 receipt changed")
    qa = read_json(path)
    core, metadata = qa.get("source_inventory", {}), qa.get("metadata_source_inventory", {})
    require(type(core) is dict and len(core) == 95 and type(metadata) is dict and len(metadata) == 8 and
        not set(core) & set(metadata) and qa.get("code_sha256") == {name: item["sha256"] for name, item in core.items()},
        "Preserve the exact tested parent core95 plus metadata8 namespace")
    return {**core, **metadata}


def source_inventory(root):
    root = Path(root).resolve()
    parent = _parent_sources(root)
    require(not set(parent) & NEW and len(NEW) == 10, "Operational repair must not replace any parent103 source")
    result = {name: _record(root, name) for name in sorted(set(parent) | NEW)}
    require(len(result) == 113 and all(result[name] == item for name, item in parent.items()) and
        all(result[name]["sha256"] == sha for name, sha in IMPLEMENTATION.items()),
        "Exact old103 plus independently reviewed10 new files are required")
    require(sum(item["bytes"] for item in result.values()) <= 8 << 20, "Bounded source inventory exceeded")
    return result


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA and plan.get("role") == ROLE and
        plan.get("cpu_job_name") == NAME and plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        plan.get("cpu_resources") == {"cpus": 2, "ram_gib": 8, "wall_time_seconds": 1200, "reserve_seconds": 172800, "gpus": 0} and
        plan.get("source_inventory_files") == 113 and plan.get("parent_source_inventory_files") == 103 and
        plan.get("prior_entries") == 17 and plan.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        plan.get("prior_snapshot") == {"path": PARENT + "/job_ledger_snapshot.json", "sha256": PINS[PARENT + "/job_ledger_snapshot.json"]} and
        plan.get("scientific_plans") == {"native": NATIVE_PLAN, "joint": JOINT_PLAN} and plan.get("prepared") == PREPARED and
        plan.get("test_patterns") == list(PATTERNS) and plan.get("expected_counts") == list(COUNTS) and
        plan.get("expected_total_tests") == 450 and plan.get("prepared_bytes_opened_by_cpu_qa") is False and
        plan.get("models_or_tokenizers_opened_by_cpu_qa") is False and
        all(plan.get(key) is False for key in ("automatic_retry", "automatic_downstream", "training_pilot", "generation", "grading")),
        "Only the fixed450-test,113-source CPU identity-replay proposal is permitted")
    return plan


def initial_evidence(root):
    """Pure metadata proof for the one-use bootstrap and the compute-only QA."""
    root, evidence = Path(root).resolve(), Evidence()
    sources = source_inventory(root)
    values = {name: evidence.read(inside(root, name), sha) for name, sha in PINS.items()}
    plan = validate_plan(evidence.read(inside(root, PLAN)))
    old_native = native.validate_plan(evidence.read(inside(root, NATIVE_PLAN["path"]), NATIVE_PLAN["sha256"]))
    old_joint = joint.validate_plan(evidence.read(inside(root, JOINT_PLAN["path"]), JOINT_PLAN["sha256"]))
    require(old_joint["inputs"] == {"native_plan": NATIVE_PLAN, "prepared": PREPARED}, "Scientific plan/prepared metadata pins differ")
    prepared_metadata = values["extension/report/native-smoke-preparation-20261021-1224497/complete.json"]
    require(prepared_metadata.get("complete") is True and prepared_metadata.get("gpu_seconds") == 0 and
        prepared_metadata.get("output_sha256", {}).get("prepared.json") == PREPARED["sha256"] and
        prepared_metadata.get("prepared") == PREPARED,
        "Pinned original preparation receipt must bind the private file without opening it")
    parent = values[PARENT + "/native_qa.json"]
    complete = values[PARENT + "/complete.json"]
    require(parent.get("success") is True and parent.get("tests_run") == 48 and parent.get("gpu_disabled") is True and
        all(parent.get(key) == 0 for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        parent.get("parent_tests_previously_passed") == 478 and parent.get("parent_tests_replayed") is False and
        parent.get("actual_vllm_interface_established") is False and complete.get("complete") is True and
        complete.get("job_id") == "1227150" and complete.get("gpu_seconds") == 0 and complete.get("tests_run") == 48 and
        complete.get("native_qa_sha256") == PINS[PARENT + "/native_qa.json"], "The exact previous CPU48 evidence is required")
    from extension.joint_gpu_admission import validate_accounting
    validate_accounting(values[PARENT + "/accounting.json"], "1227150", name="sft-extension-source-generation-qa")
    snapshot = values[PARENT + "/job_ledger_snapshot.json"]
    require(snapshot.get("schema") == "reasoning-sft-extension-job-ledger-v1" and snapshot.get("gpu_hour_cap", "missing") is None and
        snapshot.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(snapshot.get("jobs")) is dict and
        len(snapshot["jobs"]) == 17 and value_digest(snapshot["jobs"]) == PRIOR_ENTRIES_SHA256,
        "Preserve all17 exact prior entries, including the pinned reconciled no-ID entry")
    for job, role, stem, seconds in (("1224577", "native_model_admission_20261021_v1", "native-model-admission", 64),
            ("1225727", "joint_memory_admission_20261021_v1", "joint-memory-admission", 43)):
        prefix = "extension/report/" + stem + "-failed-20261021-" + job
        account, failure, first = (values[prefix + "/" + name] for name in ("accounting.json", "failure.json", "first_case_failure.json"))
        entry = snapshot["jobs"].get(role, {})
        handoff_path = "extension/operations/" + stem + "-job-" + job + ".json"
        handoff = values[handoff_path]
        require(handoff.get("status") == "admitted" and handoff.get("job_id") == job and
            handoff.get("binding_sha256") == entry.get("binding_sha256") and
            handoff.get("code_sha256") == {name: item["sha256"] for name, item in entry["source_sha256"].items()},
            "Preserve the original ledger admission and its exact handoff")
        # Runtime bindings hash actual plan/cache/QA/handoff/code; the submission
        # journal hashes its command/resources/parents. These are distinct digests.
        runtime_binding = value_digest({"plan_sha256": handoff["plan_sha256"],
            "prepared_sha256": handoff["prepared_sha256"], "qa_sha256": handoff["qa_sha256"],
            "admission_sha256": PINS[handoff_path], "code_sha256": handoff["code_sha256"]})
        require(account.get("schema") == "extension-terminal-gpu-accounting-v1" and account.get("complete") is True and
            account.get("job_id") == job and account.get("state") == "FAILED" and account.get("exit_code") == "1:0" and
            account.get("allocated_gpus") == 1 and account.get("elapsed_seconds") == seconds and account.get("gpu_seconds") == seconds and
            account.get("open_gpu_seconds") == 0 and account.get("requested_gpu_seconds") == 7200 and account.get("timezone") == "UTC" and
            entry.get("job_id") == job and failure.get("complete") is False and failure.get("completed_cases") == {} and
            failure.get("retry") is False and failure.get("automatic_downstream") is False and
            first.get("message") == "Assigned GPU UUID is unavailable" and first.get("complete") is False and
            first.get("allocator_peak_allocated_seen") == first.get("allocator_peak_reserved_seen") == 0 and
            first.get("memory_stages_observed") == [] and failure.get("binding_sha256") == first.get("binding_sha256") == runtime_binding,
            "Exact terminal pre-model identity failures must remain failed; no scientific result is inferred")
        if job == "1225727":
            require(first.get("completed_presentations") == [] and first.get("completed_update_boundaries") == [], "The failed joint run completed no presentations")
    installed = evidence.read(inside(root, INSTALL))
    require(installed.get("installed_on_arc") is True and installed.get("source_inventory") == sources and
        installed.get("parent_source_inventory") == _parent_sources(root), "Exact installed113 and unchanged parent103 receipt required")
    evidence.check(root / "deadline/job_ledger.json", HISTORICAL_LEDGER_SHA256)
    for name, item in sources.items():
        evidence.check(inside(root, name), item["sha256"])
    evidence.unchanged()
    return {"sources": sources, "parent_sources": _parent_sources(root), "assets": {str(Path(name).relative_to(root)): item
        for name, item in evidence.files.items()}, "prior": snapshot["jobs"], "plan": plan,
        "plan_sha256": evidence.check(inside(root, PLAN))}


def current_admission(root, proof, compute):
    root = Path(root).resolve()
    ledger = read_json(root / "extension/job_ledger.json")
    jobs = ledger.get("jobs", {})
    require(ledger.get("schema") == "reasoning-sft-extension-job-ledger-v1" and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(jobs) is dict and
        set(jobs) == set(proof["prior"]) | {ROLE} and all(jobs.get(name) == item for name, item in proof["prior"].items()),
        "All17 prior admissions must remain exact; no unrelated new role is admitted")
    entry = jobs[ROLE]
    binding = entry.get("binding", {})
    bootstrap = _record(root, BOOTSTRAP)
    require(entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == 0 and entry.get("requested_gpu_hours") == 0 and
        entry.get("source_sha256") == proof["sources"] == binding.get("source_sha256") and binding.get("role") == ROLE and
        binding.get("plan_sha256") == proof["plan_sha256"] and binding.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        binding.get("expected_native_tests") == 450 and binding.get("requested_gpu_seconds") == 0 and
        binding.get("wall_time_seconds") == 1200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_downstream") is False and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP, {}).get("sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(INSTALL, {}).get("sha256") == proof["assets"][INSTALL]["sha256"] and
        entry.get("binding_sha256") == value_digest(binding), "Exact source-bound current CPU admission/bootstrap is required")
    return ledger, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    compute = require_cpu_allocation()
    deadline_guard(174000)
    native._host_resources(compute, {"host_cpus": 2, "host_ram_gib": 8})
    root, proposed = Path(args.root).resolve(), Path(args.output)
    require(root == Path(__file__).resolve().parents[1] and compute["allocated_cpus"] == 2 and not proposed.is_symlink(),
        "The exact installed two-CPU allocation is required")
    output = proposed.resolve()
    require(output.parent == root / "extension/runs" and output.name == "identity-replay-qa-" + compute["job_id"] and
        not output.exists(), "Exclusive current-job CPU output required; no automatic retry")
    proof = initial_evidence(root)
    ledger, admission = current_admission(root, proof, compute)
    for name, expected in native.VERSIONS.items():
        require(importlib.metadata.version(name) == expected, "Pinned native CPU fixture runtime differs: " + name)
    for name in ("extension/identity_replay_qa.slurm", "extension/native_model_admission_v2.slurm", "extension/joint_memory_admission_v2.slurm"):
        subprocess.run(["bash", "-n", str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
        "source_inventory": proof["sources"], "expected_tests": 450, "explicitly_not_executed": list(NOT_EXECUTED)})
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests")) for name in PATTERNS]
    counts = [suite.countTestCases() for suite in suites]
    require(tuple(counts) == COUNTS and sum(counts) == 450, "Only the exact410+24+16 fixture scope is authorized")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {"tests_run": result.testsRun, "test_counts": dict(zip(PATTERNS, counts)), "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses)}
    if result.testsRun != 450 or not result.wasSuccessful() or any(metrics[name] for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")):
        write_new_json(output / "unit_qa_failure.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
            **metrics, "source_inventory": proof["sources"], "compute_proof": compute,
            "automatic_retry": False, "automatic_downstream": False})
        raise RuntimeError("Identity replay CPU fixtures failed; preserve this attempt without retry")
    require(initial_evidence(root) == proof and current_admission(root, proof, compute) == (ledger, admission),
        "Source113, parent evidence or any exact prior/current admission changed during QA")
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    unit = {"schema": SCHEMA, "success": True, "job_id": compute["job_id"], "node": compute["node"],
        "compute_proof": compute, "compute_end": ending, "started_at_utc": started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), **metrics, "gpu_disabled": True,
        "wrapper_syntax_passed": True, "source_inventory": proof["sources"], "parent_source_inventory": proof["parent_sources"],
        "code_sha256": {name: item["sha256"] for name, item in proof["sources"].items()}, "plan_sha256": proof["plan_sha256"],
        "initial_evidence": proof["assets"], "prior_admissions": proof["prior"], "prior_entries_sha256": PRIOR_ENTRIES_SHA256,
        "admission_binding_sha256": admission["binding_sha256"], "synthetic_fixture_models_only": True,
        "actual_gpu_or_memory_admission_established": False, "private_prepared_file_opened": False,
        "experiment_model_or_tokenizer_loaded": False, "source_grader_executed": False, "collector_tests_replayed": False,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in native.VERSIONS}},
        "automatic_retry": False, "automatic_downstream": False}
    write_new_json(output / "unit_qa.json", unit)
    for filename, schema, science in (("native_qa.json", "extension-native-model-admission-native-qa-v1", NATIVE_PLAN),
            ("joint_qa.json", "extension-joint-memory-admission-native-qa-v1", JOINT_PLAN)):
        write_new_json(output / filename, {**unit, "schema": schema, "plan_sha256": science["sha256"],
            "prepared_sha256": PREPARED["sha256"], "operational_plan_sha256": proof["plan_sha256"],
            "unit_qa_sha256": digest(output / "unit_qa.json"), "cpu_qa_alias_of_same450_tests": True,
            "alias_is_not_independent_gpu_or_numerical_evidence": True})
    write_new_json(output / "complete.json", {"schema": "extension-identity-replay-qa-complete-v1", "complete": True,
        "job_id": compute["job_id"], "tests_run": 450, "gpu_seconds": 0, "plan_sha256": proof["plan_sha256"],
        "source_inventory": proof["sources"], "input_snapshot_unchanged": True,
        "output_sha256": {name: digest(output / name) for name in ("unit_qa.json", "native_qa.json", "joint_qa.json")},
        "actual_gpu_or_memory_admission_established": False, "automatic_downstream": False})
    print({"identity_replay_cpu_qa_complete": True, "tests_run": 450, "gpu_seconds": 0, "automatic_downstream": False}, flush=True)


if __name__ == "__main__":
    main()
