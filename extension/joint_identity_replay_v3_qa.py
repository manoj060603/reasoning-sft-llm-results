"""Scoped CPU verification of the explicit joint-worker initializer repair.

Only source/config and public receipt metadata are opened before synthetic
fixtures. The old450 result is verified, not replayed. The joint-schema alias
describes the SAME65 CPU checks; it is not real model or memory evidence.
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
from extension import identity_replay_qa as parent
from extension import joint_memory_admission_v3 as joint
from extension import native_model_admission_v2 as native


ROLE = "joint_identity_replay_v3_qa_20261021_v1"
NAME = "sft-extension-joint-identity-v3-qa"
SCHEMA = "extension-joint-identity-replay-v3-unit-qa-v1"
PLAN = "extension/joint_identity_replay_v3_plan.json"
PLAN_SCHEMA = "extension-joint-identity-replay-v3-operational-plan-v1"
INSTALL = "extension/joint_identity_replay_v3_source_receipt.json"
BOOTSTRAP = "extension/operations/submit-joint-identity-v3-qa-20261021.py"
PRIOR_ENTRIES_SHA256 = "7e03f2b63becabd011e2e9b6cdd87a2697597ea910b8264127dc7b349608ae7c"
HISTORICAL_LEDGER_SHA256 = parent.HISTORICAL_LEDGER_SHA256
NATIVE_PLAN, JOINT_PLAN, PREPARED = parent.NATIVE_PLAN, parent.JOINT_PLAN, parent.PREPARED
PREVIOUS = "extension/report/identity-replay-qa-20261021-1227308"
FAILED = "extension/report/joint-memory-admission-v2-failed-20261021-1227971"
OLD_BOOTSTRAP = "extension/operations/submit-identity-replays-20261021.py"
OLD_BOOTSTRAP_SHA256 = "a783e3986f0f5f3054f44659573193583a0bebc0620f84e65c7cbdc7a98d844a"
IMPLEMENTATION = {
    "extension/joint_memory_admission_v3.py": "be1d4aa53b5e01755bcddfc380cf132af3ea0dca90f32b5418d1ecd28325dafa",
    "tests/test_joint_identity_replay_v3.py": "551f7e1e3421b841b2f0af2eedcfc948f7622985bac066455e043bcdc3291810",
}
NEW = frozenset(IMPLEMENTATION) | frozenset({PLAN, "extension/joint_identity_replay_v3_qa.py",
    "extension/joint_identity_replay_v3_qa.slurm", "extension/joint_memory_admission_v3.slurm"})
PATTERNS = ("test_joint_memory_admission.py", "test_gpu_identity_v2.py", "test_joint_identity_replay_v3.py")
COUNTS = (25, 24, 16)
PINS = {
    PREVIOUS + "/unit_qa.json": "fd790d109f379ce617b111f10062369d221a59db553d597f6cb5bb5ce485a6f6",
    PREVIOUS + "/native_qa.json": "5010058bef28306fb66b046b4e26d949a90eb51df4b4dfd9651d2056bf000dbd",
    PREVIOUS + "/joint_qa.json": "81d2f8c11c2335e087ece8b9107f01f88aed38ca84509af49cc78605e53a28c3",
    PREVIOUS + "/complete.json": "23b1345e23041d80c39271640a822a3c33baa4ba8a28ea10b2f05e9137b4bc5c",
    PREVIOUS + "/accounting.json": "6d3f2b23d9b767ab0f8fc8da2692703aa5da0f2eb383353b0b5a6ff41e18465b",
    PREVIOUS + "/job_ledger_snapshot.json": "f454eb4d4b35f94e8b26938c5538ae708a692569ef67c469276c92058a67099f",
    FAILED + "/accounting.json": "e6e9bcf021c6f7414683f65e49f391b6291896e514624afec4786fd61c130404",
    FAILED + "/failure.json": "b979b82150fd61a8917e50269057885bf0dc6e846038554b4f813e66b5e08536",
    FAILED + "/first_case_failure.json": "fd1f8ffab66b0d0db59b9fa871a2d79f915bddc6a0cbfb500727d97e2ec3a9f2",
    FAILED + "/job_ledger_snapshot.json": "7de19dc305699fe87af6a5ce9f10835142608e5a0a780442abd134498c6e90a2",
    FAILED + "/joint_admission.json": "eda8590c48bbfddf3e385c42f09e059315613ec1c5782ec94ef0ef8eda14456a",
    FAILED + "/native_admission.json": "71d45220bbda8f78b8b9478168d17f85fd8d7351d254fca09ce1cf768b615986",
    "extension/report/native-smoke-preparation-20261021-1224497/complete.json": "b7d8dc41b6fd90befc6a85130b3904245aea6dbd2cafc3209941d647a3ab8df5",
}
NOT_EXECUTED = ("previous450-test suite", "source-grader/readiness/collector tests", "private prepared witnesses",
    "original weights/tokenizer", "actual GPU/model/memory diagnostics")


def _record(root, name):
    return parent._record(root, name)


def source_inventory(root):
    root = Path(root).resolve()
    old = parent.source_inventory(root)
    require(len(old) == 113 and OLD_BOOTSTRAP not in old and not (set(old) | {OLD_BOOTSTRAP}) & NEW and len(NEW) == 6,
        "Only six new files may extend the unchanged114 parent namespace")
    old = {**old, OLD_BOOTSTRAP: _record(root, OLD_BOOTSTRAP)}
    require(old[OLD_BOOTSTRAP]["sha256"] == OLD_BOOTSTRAP_SHA256, "Preserve the exact executed GPU bootstrap")
    result = {name: _record(root, name) for name in sorted(set(old) | NEW)}
    require(len(result) == 120 and all(result[name] == item for name, item in old.items()) and
        all(result[name]["sha256"] == sha for name, sha in IMPLEMENTATION.items()) and
        sum(item["bytes"] for item in result.values()) <= 8 << 20, "Exact reviewed120-source inventory required")
    return result


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA and plan.get("role") == ROLE and
        plan.get("cpu_job_name") == NAME and plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        plan.get("cpu_resources") == {"cpus": 2, "ram_gib": 8, "wall_time_seconds": 1200, "reserve_seconds": 172800, "gpus": 0} and
        plan.get("source_inventory_files") == 120 and plan.get("parent_source_inventory_files") == 114 and
        plan.get("prior_entries") == 20 and plan.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        plan.get("prior_snapshot") == {"path": FAILED + "/job_ledger_snapshot.json", "sha256": PINS[FAILED + "/job_ledger_snapshot.json"]} and
        plan.get("scientific_plans") == {"native": NATIVE_PLAN, "joint": JOINT_PLAN} and plan.get("prepared") == PREPARED and
        plan.get("test_patterns") == list(PATTERNS) and plan.get("expected_counts") == list(COUNTS) and
        plan.get("expected_total_tests") == 65 and plan.get("previous_tests_previously_passed") == 450 and
        plan.get("previous_tests_replayed") is False and
        plan.get("failed_parent") == {"job_id": "1227971", "directory": FAILED, "elapsed_seconds": 85,
            "state": "FAILED", "presentations_completed": 0, "updates_completed": 0} and
        plan.get("source_install_receipt") == INSTALL and plan.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True} and
        all(plan.get(key) is False for key in ("prepared_bytes_opened_by_cpu_qa", "models_or_tokenizers_opened_by_cpu_qa",
            "automatic_retry", "automatic_downstream", "training_pilot", "generation", "grading")),
        "Only the fixed65-test120-source initializer-repair CPU proposal is permitted")
    return plan


def initial_evidence(root):
    """Bounded metadata-only proof; no private witness/model opens or GPU query."""
    from extension.joint_gpu_admission import _passed_qa, validate_accounting
    root, evidence = Path(root).resolve(), Evidence()
    sources = source_inventory(root)
    values = {name: evidence.read(inside(root, name), sha) for name, sha in PINS.items()}
    plan = validate_plan(evidence.read(inside(root, PLAN)))
    native.validate_plan(evidence.read(inside(root, NATIVE_PLAN["path"]), NATIVE_PLAN["sha256"]))
    science = joint.validate_plan(evidence.read(inside(root, JOINT_PLAN["path"]), JOINT_PLAN["sha256"]))
    require(science["inputs"] == {"native_plan": NATIVE_PLAN, "prepared": PREPARED}, "Original scientific inputs changed")
    prepared_metadata = values["extension/report/native-smoke-preparation-20261021-1224497/complete.json"]
    require(prepared_metadata.get("complete") is True and prepared_metadata.get("gpu_seconds") == 0 and
        prepared_metadata.get("prepared") == PREPARED and prepared_metadata.get("output_sha256", {}).get("prepared.json") == PREPARED["sha256"],
        "Public preparation metadata must bind the private cache without opening it")
    old_sources = {name: item for name, item in sources.items() if name not in NEW and name != OLD_BOOTSTRAP}
    codes = {name: item["sha256"] for name, item in old_sources.items()}
    unit, complete = values[PREVIOUS + "/unit_qa.json"], values[PREVIOUS + "/complete.json"]
    _passed_qa(unit, parent.SCHEMA)
    require(unit.get("tests_run") == 450 and unit.get("job_id") == "1227308" and unit.get("source_inventory") == old_sources and
        unit.get("code_sha256") == codes and unit.get("test_counts") == dict(zip(parent.PATTERNS, parent.COUNTS)) and
        complete.get("complete") is True and complete.get("job_id") == "1227308" and complete.get("gpu_seconds") == 0 and
        complete.get("tests_run") == 450 and complete.get("source_inventory") == old_sources and
        complete.get("output_sha256") == {name: PINS[PREVIOUS + "/" + name] for name in ("unit_qa.json", "native_qa.json", "joint_qa.json")},
        "The previous450 native checks are verified by exact receipts, not rerun")
    account = values[PREVIOUS + "/accounting.json"]
    validate_accounting(account, "1227308", name=parent.NAME)
    native._host_resources(unit["compute_proof"], {"host_cpus": 2, "host_ram_gib": 8})
    require(account.get("elapsed_seconds") == 49 and unit["compute_proof"].get("allocated_cpus") == 2,
        "Exact terminal CPU450 allocation metadata required")
    for filename, schema, scientific in (("native_qa.json", "extension-native-model-admission-native-qa-v1", NATIVE_PLAN),
            ("joint_qa.json", "extension-joint-memory-admission-native-qa-v1", JOINT_PLAN)):
        require(values[PREVIOUS + "/" + filename] == {**unit, "schema": schema, "plan_sha256": scientific["sha256"],
            "prepared_sha256": PREPARED["sha256"], "operational_plan_sha256": unit["plan_sha256"],
            "unit_qa_sha256": PINS[PREVIOUS + "/unit_qa.json"], "cpu_qa_alias_of_same450_tests": True,
            "alias_is_not_independent_gpu_or_numerical_evidence": True}, "Previous aliases must describe exactly the same450 checks")
    snapshot, old_snapshot = values[FAILED + "/job_ledger_snapshot.json"], values[PREVIOUS + "/job_ledger_snapshot.json"]
    prior, old_prior = snapshot.get("jobs", {}), old_snapshot.get("jobs", {})
    require(snapshot.get("schema") == old_snapshot.get("schema") == "reasoning-sft-extension-job-ledger-v1" and
        snapshot.get("gpu_hour_cap", "missing") is None and snapshot.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
        type(prior) is dict and len(prior) == 20 and value_digest(prior) == PRIOR_ENTRIES_SHA256 and len(old_prior) == 18 and
        all(prior.get(name) == entry for name, entry in old_prior.items()) and
        set(prior) - set(old_prior) == {"native_model_admission_20261021_v2", "joint_memory_admission_20261021_v2"},
        "Preserve all exact20 admissions, including known reconciled no-ID history and both v2 GPU entries")
    for kind, job, scientific, qa_name in (("native", "1227970", NATIVE_PLAN, "native_qa.json"),
            ("joint", "1227971", JOINT_PLAN, "joint_qa.json")):
        role = ("native_model" if kind == "native" else "joint_memory") + "_admission_20261021_v2"
        entry, handoff = prior[role], values[FAILED + "/" + kind + "_admission.json"]
        require(entry.get("status") == "submitted" and entry.get("job_id") == job and entry.get("source_sha256") == old_sources and
            entry.get("binding_sha256") == value_digest(entry.get("binding")) and
            entry.get("admission_receipt_sha256") == PINS[FAILED + "/" + kind + "_admission.json"] and
            handoff.get("status") == "admitted" and handoff.get("job_id") == job and handoff.get("role") == role and
            handoff.get("binding_sha256") == entry["binding_sha256"] and handoff.get("code_sha256") == codes and
            handoff.get("admission_code_sha256") == OLD_BOOTSTRAP_SHA256 and handoff.get("plan_sha256") == scientific["sha256"] and
            handoff.get("prepared_sha256") == PREPARED["sha256"] and handoff.get("qa_sha256") == PINS[PREVIOUS + "/" + qa_name] and
            handoff.get("cpu_unit_qa_sha256") == PINS[PREVIOUS + "/unit_qa.json"], "Exact separately admitted v2 GPU provenance required")
    account, failure, first = (values[FAILED + "/" + name] for name in ("accounting.json", "failure.json", "first_case_failure.json"))
    handoff = values[FAILED + "/joint_admission.json"]
    runtime_binding = value_digest({"plan_sha256": JOINT_PLAN["sha256"], "prepared_sha256": PREPARED["sha256"],
        "qa_sha256": PINS[PREVIOUS + "/joint_qa.json"], "admission_sha256": PINS[FAILED + "/joint_admission.json"], "code_sha256": codes})
    require(account.get("schema") == "extension-terminal-gpu-accounting-v1" and account.get("complete") is True and
        account.get("job_id") == "1227971" and account.get("state") == "FAILED" and account.get("exit_code") == "1:0" and
        account.get("elapsed_seconds") == account.get("gpu_seconds") == 85 and account.get("allocated_gpus") == 1 and
        account.get("requested_gpu_seconds") == 7200 and account.get("open_gpu_seconds") == 0 and account.get("timezone") == "UTC" and
        account.get("stderr") == "" and failure.get("complete") is False and failure.get("completed_cases") == {} and
        failure.get("retry") is False and failure.get("automatic_downstream") is False and
        first.get("complete") is False and first.get("message") == "too many values to unpack (expected 5)" and
        first.get("completed_presentations") == first.get("completed_update_boundaries") == first.get("actual_pre_forward_shape_observations") == [] and
        first.get("pending_presentation") is None and failure.get("binding_sha256") == first.get("binding_sha256") == runtime_binding and
        [item.get("stage") for item in first.get("memory_stages_observed", [])] == ["load_original_fp32_student_and_reference"],
        "Preserve the actual85-second post-model-load initializer failure; no completed presentations or updates")
    parent_sources = {name: item for name, item in sources.items() if name not in NEW}
    installed = evidence.read(inside(root, INSTALL))
    require(installed.get("installed_on_arc") is True and installed.get("source_inventory") == sources and
        installed.get("parent_source_inventory") == parent_sources, "Exact installed120 and unchanged parent114 required")
    evidence.check(root / "deadline/job_ledger.json", HISTORICAL_LEDGER_SHA256)
    for name, item in sources.items():
        evidence.check(inside(root, name), item["sha256"])
    evidence.unchanged()
    return {"sources": sources, "parent_sources": parent_sources, "assets": {str(Path(name).relative_to(root)): item
        for name, item in evidence.files.items()}, "prior": prior, "plan": plan, "plan_sha256": evidence.check(inside(root, PLAN))}


def current_admission(root, proof, compute):
    root = Path(root).resolve()
    ledger = read_json(root / "extension/job_ledger.json")
    jobs = ledger.get("jobs", {})
    require(ledger.get("schema") == "reasoning-sft-extension-job-ledger-v1" and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(jobs) is dict and
        set(jobs) == set(proof["prior"]) | {ROLE} and all(jobs.get(name) == entry for name, entry in proof["prior"].items()),
        "All20 prior admissions must remain unchanged; only this fixed new CPU role is allowed")
    entry, bootstrap = jobs[ROLE], _record(root, BOOTSTRAP)
    binding = entry.get("binding", {})
    require(entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and
        entry.get("source_sha256") == proof["sources"] == binding.get("source_sha256") and binding.get("role") == ROLE and
        binding.get("plan_sha256") == proof["plan_sha256"] and binding.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        binding.get("expected_native_tests") == 65 and binding.get("requested_gpu_seconds") == 0 and
        binding.get("wall_time_seconds") == 1200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_downstream") is False and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP, {}).get("sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(INSTALL, {}).get("sha256") == proof["assets"][INSTALL]["sha256"] and
        entry.get("binding_sha256") == value_digest(binding), "Exact source-bound current CPU admission/bootstrap required")
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
        "Exact installed two-CPU allocation required")
    output = proposed.resolve()
    require(output.parent == root / "extension/runs" and output.name == "joint-identity-v3-qa-" + compute["job_id"] and
        not output.exists(), "Exclusive current-job CPU output required; no automatic retry")
    proof = initial_evidence(root)
    ledger, admission = current_admission(root, proof, compute)
    for name, expected in native.VERSIONS.items():
        require(importlib.metadata.version(name) == expected, "Pinned native CPU fixture runtime differs: " + name)
    for name in ("extension/joint_identity_replay_v3_qa.slurm", "extension/joint_memory_admission_v3.slurm"):
        subprocess.run(["bash", "-n", str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
        "source_inventory": proof["sources"], "expected_tests": 65, "explicitly_not_executed": list(NOT_EXECUTED)})
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests")) for name in PATTERNS]
    counts = [suite.countTestCases() for suite in suites]
    require(tuple(counts) == COUNTS and sum(counts) == 65, "Only the exact25+24+16 fixture scope is authorized")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {"tests_run": result.testsRun, "test_counts": dict(zip(PATTERNS, counts)), "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses)}
    if result.testsRun != 65 or not result.wasSuccessful() or any(metrics[key] for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")):
        write_new_json(output / "unit_qa_failure.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
            **metrics, "source_inventory": proof["sources"], "compute_proof": compute,
            "automatic_retry": False, "automatic_downstream": False})
        raise RuntimeError("Joint initializer CPU fixtures failed; preserve this attempt without retry")
    require(initial_evidence(root) == proof and current_admission(root, proof, compute) == (ledger, admission),
        "Source120, metadata or any exact prior/current admission changed during QA")
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
        "experiment_model_or_tokenizer_loaded": False, "previous_tests_previously_passed": 450, "previous_tests_replayed": False,
        "failed_parent_remains_failed": True, "source_grader_executed": False, "collector_tests_replayed": False,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in native.VERSIONS}},
        "automatic_retry": False, "automatic_downstream": False}
    write_new_json(output / "unit_qa.json", unit)
    write_new_json(output / "native_qa.json", {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "plan_sha256": JOINT_PLAN["sha256"], "prepared_sha256": PREPARED["sha256"],
        "operational_plan_sha256": proof["plan_sha256"], "unit_qa_sha256": digest(output / "unit_qa.json"),
        "cpu_qa_alias_of_same65_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True})
    write_new_json(output / "complete.json", {"schema": "extension-joint-identity-replay-v3-qa-complete-v1", "complete": True,
        "job_id": compute["job_id"], "tests_run": 65, "gpu_seconds": 0, "plan_sha256": proof["plan_sha256"],
        "source_inventory": proof["sources"], "input_snapshot_unchanged": True,
        "output_sha256": {name: digest(output / name) for name in ("unit_qa.json", "native_qa.json")},
        "actual_gpu_or_memory_admission_established": False, "automatic_downstream": False})
    print({"joint_initializer_cpu_qa_complete": True, "tests_run": 65, "gpu_seconds": 0, "automatic_downstream": False}, flush=True)


if __name__ == "__main__":
    main()
