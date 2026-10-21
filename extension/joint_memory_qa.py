"""Prospective, compute-only410-check QA for the separate joint diagnostic.

The73 earlier sources are hash-bound but this execution scope is deliberately
legacy341 + pilot44 + joint25. Source-grader/readiness tests are NOT executed or
admitted here. This is no successful replay of their465-test scope and changes
none of their code or deadlines. All models used by tests are tiny CPU fixtures;
no original weights, private witnesses, tokenizer or model outcomes are opened.

A separate one-use source-bound CPU admission adapter is still required. This
module neither submits that stage nor admits/submits any GPU work afterward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import (deadline_guard, digest, read_json, require,
    require_cpu_allocation, write_new_json)
from extension import joint_memory_admission as joint
from extension import source_readiness_retry_submission as previous

ROLE = "joint_memory_qa_20261021_v1"
SCHEMA = "extension-joint-memory-unit-qa-v1"
PLAN = "extension/joint_memory_admission_plan.json"
PRIOR_DIAGNOSTIC_ROLE = "source_grader_diagnostic_20261021_v1"
PRIOR_DIAGNOSTIC_JOB = "1225426"
DIRECTORY = "extension/report/source-readiness-v2-failed-20261021-1225360"
PARENT_PINS = {
    "accounting.json": "1e725f1fd3bf40e29fff1004b36228695cf219f66ac50ed158b060f2c6bf7d2b",
    "unit_qa_failure.json": "8cdb5a256023a64623616f43c99f19f5deaf20cad8b540e1ae0edaf8b078d8b2",
    "job_ledger_snapshot.json": "8ddea4b35c0b8f06b77632378c418feb891b0281192c1b8386e83d2eec9c99bc",
    "source_install_receipt.json": "fcfbdfcfade0a4889db061c02f8c0ccc91b953f6ad7e86965dff1eac65b55717"}
HISTORICAL_LEDGER_SHA256 = "9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d"
FROZEN_PROTOTYPE = {
    "extension/joint_memory_admission.py": "4fcce77d633ea13baab8630c8485fccade52a5026eb51c4a5a75806c712f95fe",
    "tests/test_joint_memory_admission.py": "f996b0f6c9ecf39c42656c4120335ee53182e6a6a0d721d5c361058fc9b410b6"}
PRIOR_DIAGNOSTIC_SOURCES = frozenset({"extension/source_grader_diagnostic.py", "extension/source_grader_diagnostic.slurm"})
NEW_SOURCES = frozenset(FROZEN_PROTOTYPE) | frozenset({PLAN, "extension/joint_memory_admission.slurm",
    "extension/joint_memory_qa.py", "extension/joint_memory_qa.slurm"})
PATTERNS = ("test_extension_*.py", "test_labeled_*.py", "test_preserving_*.py", "test_development_*.py",
    "test_preservation_development_seal_v2.py", "test_preservation_retry_submission.py",
    "test_preservation_data.py", "test_native_model_admission.py", "test_smoke_preparation.py",
    "test_smoke_submission.py", "test_pilot_driver.py", "test_joint_memory_admission.py")
COUNTS = (64, 44, 36, 48, 17, 21, 30, 36, 21, 24, 44, 25)
NOT_EXECUTED = ("tests/test_source_evaluation.py", "tests/test_source_readiness.py",
    "tests/test_source_readiness_submission.py", "tests/test_source_readiness_retry_submission.py",
    "extension/source_grader_diagnostic.py")


def _record(root, relative):
    data = previous.metadata._small_bytes(Path(root) / relative, Path(root))
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def source_inventory(root):
    """Exactly73 untouched sources plus these6 new files; no mutable runs/logs."""
    root = Path(root).resolve()
    result = previous.source_inventory(root)
    require(len(result) == 71, "Preserve the exact previous71-source namespace")
    for name in sorted(PRIOR_DIAGNOSTIC_SOURCES | NEW_SOURCES):
        result[name] = _record(root, name)
    require(len(result) == 79 and all(result[name]["sha256"] == checksum for name, checksum in FROZEN_PROTOTYPE.items()),
        "The independently reviewed diagnostic prototype/test bytes changed")
    require(sum(value["bytes"] for value in result.values()) <= previous.metadata.MAX_TOTAL_SOURCE_BYTES,
        "QA source inventory exceeds its bounded metadata allowance")
    return result


def initial_evidence(root, sources):
    """Bind immutable prior metadata/11 entries, not grading results or labels."""
    root = Path(root).resolve()
    assets, values = {}, {}
    for name, expected in PARENT_PINS.items():
        relative = DIRECTORY + "/" + name
        actual = _record(root, relative)
        require(actual["sha256"] == expected, "Pinned pre-model failure metadata changed: " + name)
        assets[relative], values[name] = actual, read_json(root / relative)
    old = values["source_install_receipt.json"].get("source_inventory", {})
    prior_names = set(previous.SOURCE_PATHS)
    require(len(old) == 71 and {name: sources[name] for name in prior_names} == old,
        "The prior71-source bytes no longer equal their native installation receipt")
    archived = values["job_ledger_snapshot.json"]
    require(archived.get("schema") == previous.SCHEMA and type(archived.get("jobs")) is dict and len(archived["jobs"]) == 10,
        "The exact ten-entry native failure snapshot is required")
    ledger = read_json(root / "extension/job_ledger.json")
    require(ledger.get("schema") == previous.SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) <= set(archived["jobs"]) | {PRIOR_DIAGNOSTIC_ROLE, ROLE} and
        all(ledger["jobs"].get(name) == value for name, value in archived["jobs"].items()),
        "All ten earlier admissions must remain equivalent; unknown entries block QA")
    prior_entry = ledger["jobs"].get(PRIOR_DIAGNOSTIC_ROLE, {})
    old73 = {name: value for name, value in sources.items() if name not in NEW_SOURCES}
    binding = prior_entry.get("binding", {})
    require(prior_entry.get("status") == "submitted" and prior_entry.get("job_id") == PRIOR_DIAGNOSTIC_JOB and
        prior_entry.get("source_sha256") == old73 and prior_entry.get("requested_gpu_seconds") == 0 and
        prior_entry.get("requested_gpu_hours") == 0 and binding.get("role") == PRIOR_DIAGNOSTIC_ROLE and
        binding.get("source_sha256") == old73 and binding.get("wall_time_seconds") == 600 and
        binding.get("automatic_downstream") is False and prior_entry.get("binding_sha256") ==
            hashlib.sha256(previous.metadata._canonical(binding)).hexdigest(),
        "The earlier73-source diagnostic admission changed; its grading success is not a QA prerequisite")
    require(digest(root / "deadline/job_ledger.json") == HISTORICAL_LEDGER_SHA256,
        "Historical GPU ledger must remain unchanged")
    priors = dict(archived["jobs"])
    priors[PRIOR_DIAGNOSTIC_ROLE] = prior_entry
    return assets, priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compute = require_cpu_allocation()
    deadline_guard(180000)
    joint.native._host_resources(compute, {"host_cpus": 8, "host_ram_gib": 64})
    root, proposed = Path(args.root).resolve(), Path(args.output)
    require(root == Path(__file__).resolve().parents[1] and not proposed.is_symlink(), "Use this installed project and a fresh QA output")
    output = proposed.resolve()
    require(output.parent == root / "extension/runs" and output.name == "joint-memory-qa-" + compute["job_id"] and
        not output.exists(), "The exact current-allocation exclusive QA output is required")
    before = source_inventory(root)
    parents = initial_evidence(root, before)
    plan_path = root / PLAN
    plan = joint.validate_plan(read_json(plan_path))
    require(plan.get("cpu_qa", {}).get("test_patterns") == list(PATTERNS) and
        plan["cpu_qa"].get("expected_counts") == list(COUNTS) and plan["cpu_qa"].get("expected_total_tests") == 410 and
        plan["cpu_qa"].get("inventory_files") == 79 and
        plan["cpu_qa"].get("explicitly_not_executed_or_admitted") == list(NOT_EXECUTED),
        "New scoped QA is exactly410 checks; grader exclusions cannot be hidden")
    ledger = read_json(root / "extension/job_ledger.json")
    admission = ledger.get("jobs", {}).get(ROLE, {})
    binding = admission.get("binding", {})
    require(admission.get("status") == "submitted" and admission.get("job_id") == compute["job_id"] and
        admission.get("source_sha256") == before and admission.get("requested_gpu_seconds") == 0 and
        admission.get("requested_gpu_hours") == 0 and binding.get("role") == ROLE and
        binding.get("source_sha256") == before and binding.get("plan_sha256") == digest(plan_path) and
        binding.get("wall_time_seconds") == 7200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_downstream") is False and
        admission.get("binding_sha256") == hashlib.sha256(previous.metadata._canonical(binding)).hexdigest(),
        "A future separate exact-source one-use CPU admission is still required")
    for package, expected in joint.native.VERSIONS.items():
        require(importlib.metadata.version(package) == expected, "Unadmitted native version: " + package)
    for name in ("extension/joint_memory_qa.slurm", "extension/joint_memory_admission.slurm"):
        subprocess.run(["bash", "-n", str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
        "source_inventory": before, "expected_tests": 410, "explicitly_not_executed_or_admitted": list(NOT_EXECUTED)})
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern, top_level_dir=str(root / "tests"))
        for pattern in PATTERNS]
    counts = [suite.countTestCases() for suite in suites]
    require(tuple(counts) == COUNTS and sum(counts) == 410, "Run only the exact legacy341+pilot44+joint25 regression scope")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    if result.testsRun != 410 or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
        write_new_json(output / "unit_qa_failure.json", {"schema": SCHEMA, "success": False, "job_id": compute["job_id"],
            "tests_run": result.testsRun, "test_counts": dict(zip(PATTERNS, counts)), "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses), "source_inventory": before,
            "code_sha256": {name: value["sha256"] for name, value in before.items()},
            "compute_proof": compute, "no_automatic_retry": True, "explicitly_not_executed_or_admitted": list(NOT_EXECUTED)})
        raise RuntimeError("Scoped joint diagnostic CPU QA failed; preserve this attempt")
    require(source_inventory(root) == before and initial_evidence(root, before) == parents and
        read_json(root / "extension/job_ledger.json")["jobs"].get(ROLE) == admission,
        "Sources or the exact earlier/current admissions changed during QA")
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    unit = {"schema": SCHEMA, "success": True, "job_id": compute["job_id"], "node": compute["node"],
        "compute_proof": compute, "compute_end": ending, "started_at_utc": started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
        "test_counts": dict(zip(PATTERNS, counts)), "failures": 0, "errors": 0, "skipped": 0,
        "expected_failures": 0, "unexpected_successes": 0, "gpu_disabled": True,
        "wrapper_syntax_passed": True, "source_inventory": before,
        "code_sha256": {name: value["sha256"] for name, value in before.items()},
        "plan_sha256": digest(plan_path), "initial_evidence": parents[0], "prior_admissions": parents[1],
        "admission_binding_sha256": admission["binding_sha256"], "synthetic_fixture_models_only": True,
        "experiment_model_loaded": False, "source_grader_executed": False,
        "source_grader_admitted": False, "full_source_readiness_465_scope_replayed": False,
        "explicitly_not_executed_or_admitted": list(NOT_EXECUTED), "original_prepared_file_read_or_replayed_here": False,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in joint.native.VERSIONS}},
        "automatic_downstream": False}
    write_new_json(output / "unit_qa.json", unit)
    native = {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
        "prepared_sha256": plan["inputs"]["prepared"]["sha256"], "unit_qa_sha256": digest(output / "unit_qa.json"),
        "actual_model_or_memory_admission_established": False}
    write_new_json(output / "native_qa.json", native)
    write_new_json(output / "complete.json", {"schema": "extension-joint-memory-qa-complete-v1", "complete": True,
        "cpu_only": True, "job_id": compute["job_id"], "gpu_seconds": 0, "tests_run": 410,
        "plan_sha256": digest(plan_path), "native_qa_sha256": digest(output / "native_qa.json"),
        "unit_qa_sha256": digest(output / "unit_qa.json"), "source_inventory": before,
        "input_snapshot_unchanged": True, "source_grader_admitted": False,
        "actual_model_or_memory_admission_established": False, "automatic_downstream": False})
    print(json.dumps({"scoped_qa_complete": True, "tests_run": 410, "gpu_seconds": 0,
        "source_grader_admitted": False, "automatic_downstream": False}), flush=True)


if __name__ == "__main__":
    main()
