"""Prospective CPU97 checks for one allocator-only engineering replay.

The previously passed focused65 checks are rerun as regression fixtures along
with32 new allocator fixtures. Historical450/65 receipts are separately bound.
No private witnesses/model bytes, real allocator/GPU or downstream job is used.
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
from extension import joint_identity_replay_v3_qa as parent
from extension import joint_memory_allocator_v4 as worker


ROLE = "joint_allocator_v4_qa_20261021_v1"
NAME = "sft-extension-joint-allocator-v4-qa"
SCHEMA = "extension-joint-allocator-v4-unit-qa-v1"
PLAN = "extension/joint_allocator_v4_plan.json"
INSTALL = "extension/joint_allocator_v4_source_receipt.json"
BOOTSTRAP = "extension/operations/submit-joint-allocator-v4-qa-20261021.py"
PRIOR_ENTRIES_SHA256 = "455946c6102643a93bd3bb976cd69855f254a99d5ebd5814bcbeef98a1492c08"
HISTORICAL_LEDGER_SHA256 = parent.HISTORICAL_LEDGER_SHA256
NATIVE_PLAN, JOINT_PLAN, PREPARED = parent.NATIVE_PLAN, parent.JOINT_PLAN, parent.PREPARED
PREVIOUS = "extension/report/joint-identity-v3-qa-20261021-1228170"
FAILED = "extension/report/joint-memory-admission-v3-failed-20261021-1228339"
PARENT_BOOTSTRAPS = {
    "extension/operations/submit-joint-identity-v3-qa-20261021.py": "3aad1b2af0f50bf8cee4d5e857ed304966c9842ee0849747e38665632b5faec4",
    "extension/operations/submit-joint-identity-v3-20261021.py": "86e6a6fff11216e848f63602752da57052f0782e6d7965654ce97b72dc7ec853",
}
NEW = frozenset({PLAN, "extension/joint_memory_allocator_v4.py", "tests/test_joint_allocator_v4.py",
    "extension/joint_allocator_v4_qa.py", "extension/joint_allocator_v4_qa.slurm", "extension/joint_memory_allocator_v4.slurm"})
PATTERNS = ("test_joint_memory_admission.py", "test_gpu_identity_v2.py", "test_joint_identity_replay_v3.py", "test_joint_allocator_v4.py")
COUNTS = (25, 24, 16, 32)
PINS = {
    PREVIOUS + "/unit_qa.json": "8e0f61aabe1ba4d285007c92d680542762cfae1271905c2cec63f908dc760874",
    PREVIOUS + "/native_qa.json": "cf05eb73d2a2a44e548b4b354254af6544cdee23b8cf691cf30191414d2045a5",
    PREVIOUS + "/complete.json": "00835a892b571eaba14a6c0f65c7ede751d6c050472d43146460f5df0f9dd94e",
    PREVIOUS + "/accounting.json": "455cda55ba852be2ef7bb0d60e393b94e51708947e7f1738409b12fcc011a96f",
    PREVIOUS + "/job_ledger_snapshot.json": "7bd5d37f72860ae5b9e5a7743d561ca1c62286da706d3a5500313e192c787cd6",
    FAILED + "/accounting.json": "af737bdd63af55105dd9627b83d8921781d169373a4083806c989ab6c028dba7",
    FAILED + "/failure.json": "2262fa39abd807981c3146a61de0aaed2bc1a9cd23cbaf659f8096d21d982511",
    FAILED + "/first_case_failure.sanitized.json": "36de0aa2d11cc6e629f837a21efec396e6668cf71b1c2145a8730653d3b9faba",
    FAILED + "/admission.json": "bf17cbd756c1334b3172adda66df7b789b379a51f6a67c839660e6c59f818594",
    FAILED + "/job_ledger_snapshot.json": "d1f9dfe9c90acd6de567f33e7346877cccd61fde99df0c22e122ea91884d8254",
    "extension/report/native-smoke-preparation-20261021-1224497/complete.json": "b7d8dc41b6fd90befc6a85130b3904245aea6dbd2cafc3209941d647a3ab8df5",
}
NOT_EXECUTED = ("historical450 suite", "source-grader/readiness/collector tests", "private prepared witnesses",
    "original weights/tokenizer", "actual expandable-segments allocator interface", "GPU/model/memory diagnostics")


def _record(root, name):
    return parent._record(root, name)


def source_inventory(root):
    root = Path(root).resolve()
    old = parent.source_inventory(root)
    require(len(old) == 120 and not set(old) & (NEW | set(PARENT_BOOTSTRAPS)) and len(NEW) == 6,
        "Only two executed operational adapters plus six new files may extend the frozen120 namespace")
    old = {**old, **{name: _record(root, name) for name in PARENT_BOOTSTRAPS}}
    require(all(old[name]["sha256"] == sha for name, sha in PARENT_BOOTSTRAPS.items()), "The two executed operational adapters must remain exact")
    result = {name: _record(root, name) for name in sorted(set(old) | NEW)}
    require(len(result) == 128 and all(result[name] == value for name, value in old.items()) and
        sum(value["bytes"] for value in result.values()) <= 8 << 20, "Exact reviewed128-source inventory required")
    return result


def validate_plan(plan):
    worker.validate_allocator_plan(plan)
    require(plan.get("role") == ROLE and plan.get("cpu_job_name") == NAME and
        plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        plan.get("cpu_resources") == {"cpus": 2, "ram_gib": 8, "wall_time_seconds": 1200, "reserve_seconds": 172800, "gpus": 0} and
        plan.get("source_inventory_files") == 128 and plan.get("parent_source_inventory_files") == 122 and
        plan.get("prior_entries") == 22 and plan.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        plan.get("prior_snapshot") == {"path": FAILED + "/job_ledger_snapshot.json", "sha256": PINS[FAILED + "/job_ledger_snapshot.json"]} and
        plan.get("test_patterns") == list(PATTERNS) and plan.get("expected_counts") == list(COUNTS) and
        plan.get("expected_total_tests") == 97 and plan.get("previous_focused65_regression_checks_rerun") is True and
        plan.get("historical450_receipts_verified_not_replayed") is True and plan.get("historical65_receipts_verified") is True and
        plan.get("source_install_receipt") == INSTALL and plan.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True} and
        all(plan.get(key) is False for key in ("prepared_bytes_opened_by_cpu_qa", "models_or_tokenizers_opened_by_cpu_qa",
            "automatic_retry", "automatic_downstream", "training_pilot", "generation", "grading")),
        "Only the fixed97-test128-source prospective allocator proposal is permitted")
    return plan


def initial_evidence(root):
    """Pure public metadata proof, without private witness/model/allocator work."""
    from extension.joint_gpu_admission import _passed_qa, validate_accounting
    root, evidence = Path(root).resolve(), Evidence()
    sources = source_inventory(root)
    values = {name: evidence.read(inside(root, name), sha) for name, sha in PINS.items()}
    plan = validate_plan(evidence.read(inside(root, PLAN)))
    science = worker.validate_plan(evidence.read(inside(root, JOINT_PLAN["path"]), JOINT_PLAN["sha256"]))
    require(science["settings"] == plan["settings"] and science["tolerances"] == plan["tolerances"] and
        science["case_order"] == plan["case_order"] and science["presentation_roles"] == plan["presentation_roles"] and
        science["inputs"] == {"native_plan": NATIVE_PLAN, "prepared": PREPARED}, "Original scientific workload or thresholds changed")
    evidence.check(inside(root, NATIVE_PLAN["path"]), NATIVE_PLAN["sha256"])
    prepared_metadata = values["extension/report/native-smoke-preparation-20261021-1224497/complete.json"]
    require(prepared_metadata.get("complete") is True and prepared_metadata.get("gpu_seconds") == 0 and
        prepared_metadata.get("prepared") == PREPARED and prepared_metadata.get("output_sha256", {}).get("prepared.json") == PREPARED["sha256"],
        "Public preparation metadata must bind the unchanged private cache without opening it")
    old_sources = {name: item for name, item in sources.items() if name not in NEW and name not in PARENT_BOOTSTRAPS}
    codes = {name: item["sha256"] for name, item in old_sources.items()}
    unit, complete = values[PREVIOUS + "/unit_qa.json"], values[PREVIOUS + "/complete.json"]
    _passed_qa(unit, parent.SCHEMA)
    require(unit.get("job_id") == "1228170" and unit.get("tests_run") == 65 and unit.get("source_inventory") == old_sources and
        unit.get("code_sha256") == codes and unit.get("test_counts") == dict(zip(parent.PATTERNS, parent.COUNTS)) and
        unit.get("previous_tests_previously_passed") == 450 and unit.get("previous_tests_replayed") is False and
        complete.get("complete") is True and complete.get("job_id") == "1228170" and complete.get("gpu_seconds") == 0 and
        complete.get("tests_run") == 65 and complete.get("source_inventory") == old_sources and
        complete.get("output_sha256") == {name: PINS[PREVIOUS + "/" + name] for name in ("unit_qa.json", "native_qa.json")},
        "Exact historical65 and its previously verified450 metadata must remain intact")
    account = values[PREVIOUS + "/accounting.json"]
    validate_accounting(account, "1228170", name=parent.NAME)
    worker.native._host_resources(unit["compute_proof"], {"host_cpus": 2, "host_ram_gib": 8})
    require(account.get("elapsed_seconds") == 32 and unit["compute_proof"].get("allocated_cpus") == 2,
        "Exact terminal CPU65 allocation required")
    require(values[PREVIOUS + "/native_qa.json"] == {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "plan_sha256": JOINT_PLAN["sha256"], "prepared_sha256": PREPARED["sha256"],
        "operational_plan_sha256": unit["plan_sha256"], "unit_qa_sha256": PINS[PREVIOUS + "/unit_qa.json"],
        "cpu_qa_alias_of_same65_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True},
        "Historical joint alias must describe exactly the same65 CPU checks")
    snapshot, old_snapshot = values[FAILED + "/job_ledger_snapshot.json"], values[PREVIOUS + "/job_ledger_snapshot.json"]
    prior, old_prior = snapshot.get("jobs", {}), old_snapshot.get("jobs", {})
    require(snapshot.get("schema") == "reasoning-sft-extension-job-ledger-v1" and snapshot.get("gpu_hour_cap", "missing") is None and
        snapshot.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(prior) is dict and len(prior) == 22 and
        value_digest(prior) == PRIOR_ENTRIES_SHA256 and len(old_prior) == 21 and
        all(prior.get(name) == entry for name, entry in old_prior.items()) and
        set(prior) - set(old_prior) == {"joint_memory_admission_20261021_v3"}, "Preserve every exact22 prior admission")
    entry, admission = prior["joint_memory_admission_20261021_v3"], values[FAILED + "/admission.json"]
    require(entry.get("status") == "submitted" and entry.get("job_id") == "1228339" and entry.get("source_sha256") == old_sources and
        entry.get("binding_sha256") == value_digest(entry.get("binding")) and
        entry.get("admission_receipt_sha256") == PINS[FAILED + "/admission.json"] and admission.get("status") == "admitted" and
        admission.get("job_id") == "1228339" and admission.get("role") == "joint_memory_admission_20261021_v3" and
        admission.get("binding_sha256") == entry["binding_sha256"] and admission.get("code_sha256") == codes and
        admission.get("admission_code_sha256") == PARENT_BOOTSTRAPS["extension/operations/submit-joint-identity-v3-20261021.py"] and
        admission.get("plan_sha256") == JOINT_PLAN["sha256"] and admission.get("prepared_sha256") == PREPARED["sha256"] and
        admission.get("qa_sha256") == PINS[PREVIOUS + "/native_qa.json"], "Exact failed v3 admission lineage is required")
    account, failure, first = (values[FAILED + "/" + name] for name in ("accounting.json", "failure.json", "first_case_failure.sanitized.json"))
    runtime = value_digest({"plan_sha256": JOINT_PLAN["sha256"], "prepared_sha256": PREPARED["sha256"],
        "qa_sha256": PINS[PREVIOUS + "/native_qa.json"], "admission_sha256": PINS[FAILED + "/admission.json"], "code_sha256": codes})
    require(account.get("schema") == "extension-terminal-gpu-accounting-v1" and account.get("complete") is True and
        account.get("job_id") == "1228339" and account.get("state") == "FAILED" and account.get("exit_code") == "1:0" and
        account.get("elapsed_seconds") == account.get("gpu_seconds") == 197 and account.get("allocated_gpus") == 1 and
        account.get("original_requested_gpu_seconds") == 7200 and account.get("timezone") == "UTC" and
        failure.get("complete") is False and failure.get("completed_cases") == {} and failure.get("retry") is False and
        first.get("complete") is False and first.get("message") == "Joint actual memory headroom failed" and first.get("mode") == "sft" and
        first.get("private_presentation_count") == 17 and len(first.get("completed_update_boundaries", [])) == 3 and
        len(first.get("actual_pre_forward_shape_observations", [])) == 17 and first.get("pending_presentation") is None and
        first.get("raw_witnesses_exported") is False and first.get("raw_private_receipt", {}).get("sha256") ==
            "204c3f2604ffa9063b53b7c9028bd7df1342a4c9d8792dfcd7c0cee8527ecb95" and
        failure.get("binding_sha256") == first.get("binding_sha256") == runtime,
        "Keep the197-second completed-work/headroom-failed case failed; never open or export its private raw receipt")
    parent_sources = {name: item for name, item in sources.items() if name not in NEW}
    installed = evidence.read(inside(root, INSTALL))
    require(installed.get("installed_on_arc") is True and installed.get("source_inventory") == sources and
        installed.get("parent_source_inventory") == parent_sources, "Exact installed128 and unchanged122 receipt required")
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
        set(jobs) == set(proof["prior"]) | {ROLE} and all(jobs.get(name) == value for name, value in proof["prior"].items()),
        "All22 prior admissions must remain unchanged; only this new fixed CPU role is allowed")
    entry, bootstrap = jobs[ROLE], _record(root, BOOTSTRAP)
    binding = entry.get("binding", {})
    require(entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and
        entry.get("source_sha256") == proof["sources"] == binding.get("source_sha256") and binding.get("role") == ROLE and
        binding.get("plan_sha256") == proof["plan_sha256"] and binding.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256 and
        binding.get("expected_native_tests") == 97 and binding.get("requested_gpu_seconds") == 0 and
        binding.get("wall_time_seconds") == 1200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_downstream") is False and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP, {}).get("sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(INSTALL, {}).get("sha256") == proof["assets"][INSTALL]["sha256"] and
        entry.get("binding_sha256") == value_digest(binding), "Exact source-bound current CPU admission required")
    return ledger, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    compute = require_cpu_allocation()
    deadline_guard(174000)
    worker.native._host_resources(compute, {"host_cpus": 2, "host_ram_gib": 8})
    root, proposed = Path(args.root).resolve(), Path(args.output)
    require(root == Path(__file__).resolve().parents[1] and compute["allocated_cpus"] == 2 and not proposed.is_symlink(), "Exact installed two-CPU allocation required")
    output = proposed.resolve()
    require(output.parent == root / "extension/runs" and output.name == "joint-allocator-v4-qa-" + compute["job_id"] and
        not output.exists(), "Exclusive current-job CPU output required; no retry")
    proof = initial_evidence(root)
    ledger, admission = current_admission(root, proof, compute)
    for name, expected in worker.native.VERSIONS.items():
        require(importlib.metadata.version(name) == expected, "Pinned native CPU fixture runtime differs: " + name)
    for name in ("extension/joint_allocator_v4_qa.slurm", "extension/joint_memory_allocator_v4.slurm"):
        subprocess.run(["bash", "-n", str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
        "source_inventory": proof["sources"], "expected_tests": 97, "explicitly_not_executed": list(NOT_EXECUTED)})
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests")) for name in PATTERNS]
    counts = [suite.countTestCases() for suite in suites]
    require(tuple(counts) == COUNTS and sum(counts) == 97, "Only the exact65 regression plus32 new fixtures are authorized")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {"tests_run": result.testsRun, "test_counts": dict(zip(PATTERNS, counts)), "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses)}
    if result.testsRun != 97 or not result.wasSuccessful() or any(metrics[key] for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")):
        write_new_json(output / "unit_qa_failure.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"], **metrics,
            "source_inventory": proof["sources"], "compute_proof": compute, "automatic_retry": False, "automatic_downstream": False})
        raise RuntimeError("Allocator CPU fixtures failed; preserve this attempt without retry")
    require(initial_evidence(root) == proof and current_admission(root, proof, compute) == (ledger, admission), "Source128, evidence or prior/current admission changed")
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    unit = {"schema": SCHEMA, "success": True, "job_id": compute["job_id"], "node": compute["node"], "compute_proof": compute,
        "compute_end": ending, "started_at_utc": started, "created_at_utc": datetime.now(timezone.utc).isoformat(), **metrics,
        "gpu_disabled": True, "wrapper_syntax_passed": True, "source_inventory": proof["sources"], "parent_source_inventory": proof["parent_sources"],
        "code_sha256": {name: item["sha256"] for name, item in proof["sources"].items()}, "plan_sha256": proof["plan_sha256"],
        "initial_evidence": proof["assets"], "prior_admissions": proof["prior"], "prior_entries_sha256": PRIOR_ENTRIES_SHA256,
        "admission_binding_sha256": admission["binding_sha256"], "allocator_policy": worker.ALLOCATOR_POLICY,
        "previous_focused65_regression_checks_rerun": True, "historical450_receipts_verified_not_replayed": True, "historical65_receipts_verified": True,
        "synthetic_fixture_models_only": True, "effective_expandable_segments_interface_established": False,
        "actual_gpu_or_memory_admission_established": False, "private_prepared_file_opened": False,
        "experiment_model_or_tokenizer_loaded": False, "failed_parent_remains_failed": True,
        "source_grader_executed": False, "collector_tests_replayed": False,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in worker.native.VERSIONS}},
        "automatic_retry": False, "automatic_downstream": False}
    write_new_json(output / "unit_qa.json", unit)
    write_new_json(output / "native_qa.json", {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "plan_sha256": JOINT_PLAN["sha256"], "prepared_sha256": PREPARED["sha256"], "operational_plan_sha256": proof["plan_sha256"],
        "unit_qa_sha256": digest(output / "unit_qa.json"), "cpu_qa_alias_of_same97_tests": True, "alias_is_not_independent_gpu_or_numerical_evidence": True})
    write_new_json(output / "complete.json", {"schema": "extension-joint-allocator-v4-qa-complete-v1", "complete": True, "job_id": compute["job_id"],
        "tests_run": 97, "gpu_seconds": 0, "plan_sha256": proof["plan_sha256"], "source_inventory": proof["sources"], "input_snapshot_unchanged": True,
        "output_sha256": {name: digest(output / name) for name in ("unit_qa.json", "native_qa.json")},
        "actual_gpu_or_memory_admission_established": False, "automatic_downstream": False})
    print({"joint_allocator_cpu_qa_complete": True, "tests_run": 97, "gpu_seconds": 0, "automatic_downstream": False}, flush=True)


if __name__ == "__main__":
    main()
