"""Scoped native CPU60 fixtures; no private prompts, models or generation.

The parent478, identity65 and allocator97 are bound by exact receipts without
replaying their tests or claiming their CPU checks establish native inference.
The prior CPU60 remains failed; this new corrected40+versioned20 scope is not
an alias or retrospective pass for the failed v2 test file.
"""
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import unittest

from extension.cpu_feasibility import require_cpu_allocation, deadline_guard, write_new_json, value_digest, digest
from extension import source_generation_v3_submission as metadata


def main():
    root, proposed = Path.cwd().resolve(), Path(sys.argv[1])
    compute = require_cpu_allocation()
    deadline_guard(174000)
    metadata.generation.native._host_resources(compute, {"host_cpus": 2, "host_ram_gib": 8})
    metadata.shared.require(root == Path(__file__).resolve().parents[1] and not proposed.is_symlink() and
        compute["allocated_cpus"] == 2, "Exact current two-CPU native60 allocation required")
    output = proposed.resolve()
    metadata.shared.require(output.parent == root / "extension/runs" and output.name == "source-generation-v3-qa-" + compute["job_id"] and
        not output.exists(), "Exclusive current CPU run required")
    proof = metadata.initial_evidence(root)
    ledger, admission = metadata.current_cpu_admission(root, proof, compute)
    metadata.validate_wrapper(root)
    for package, expected in metadata.generation.native.VERSIONS.items():
        metadata.shared.require(importlib.metadata.version(package) == expected, "Pinned native CPU fixture runtime differs: " + package)
    for name in (metadata.SCRIPT, metadata.QA_SCRIPT):
        subprocess.run(["bash", "-n", str(root / name)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": metadata.generation.QA_SCHEMA, "job_id": compute["job_id"],
        "source_inventory": proof["sources"], "metadata_source_inventory": proof["metadata"], "expected_tests": 60,
        **metadata.retry_bindings(), "old_failed_cpu60_replayed": False, "old_failed_cpu60_passed": False,
        "old_fixture_failure_preserved": True,
        "parent_tests_replayed": False, "identity_parent_tests_replayed": False, "allocator_tests_replayed": False})
    patterns = ["test_source_generation_identity_v3.py", "test_source_generation_v3_submission.py"]
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests")) for name in patterns]
    counts = [suite.countTestCases() for suite in suites]
    metadata.shared.require(counts == [40, 20], "Exactly40 versioned collector plus20 metadata checks are allowed")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {"tests_run": result.testsRun, "test_counts": dict(zip(patterns, counts)), "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses)}
    if result.testsRun != 60 or not result.wasSuccessful() or any(metrics[name] for name in
        ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")):
        write_new_json(output / "failure.json", {"schema": metadata.generation.QA_SCHEMA, "success": False,
            "job_id": compute["job_id"], **metrics, "source_inventory": proof["sources"], "metadata_source_inventory": proof["metadata"],
            "compute_proof": compute, "automatic_retry": False, "automatic_downstream": False,
            **metadata.retry_bindings(), "old_failed_cpu60_replayed": False, "old_failed_cpu60_passed": False,
            "old_fixture_failure_preserved": True,
            "parent_tests_replayed": False, "identity_parent_tests_replayed": False, "allocator_tests_replayed": False})
        raise RuntimeError("Native collector-v3 fixtures failed; preserve without automatic retry")
    metadata.shared.require(metadata.initial_evidence(root) == proof and metadata.current_cpu_admission(root, proof, compute) == (ledger, admission),
        "Core128/metadata4/protected139/failed CPU60/external sources/history/current admission changed during QA")
    ending = require_cpu_allocation()
    metadata.shared.require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    receipt = {"schema": metadata.generation.QA_SCHEMA, "success": True, "job_id": compute["job_id"], "node": compute["node"],
        "compute_proof": compute, "compute_end": ending, "started_at_utc": started, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **metrics, "source_inventory": proof["sources"], "metadata_source_inventory": proof["metadata"],
        "allocator_source_inventory": proof["allocator_sources"], "code_sha256": {name: item["sha256"] for name, item in proof["sources"].items()},
        "plan_sha256": proof["plan_sha256"], "cache_sha256": metadata.generation.CACHE["sha256"],
        "parent_complete_sha256": metadata.generation.PARENT_PINS["complete.json"],
        "scientific_parent_sha256": metadata.generation.SCIENTIFIC_PARENT["sha256"],
        "identity_parent_complete_sha256": metadata.generation.IDENTITY_PINS["complete.json"],
        "identity_helper_sha256": metadata.generation.IDENTITY_HELPER["sha256"],
        **metadata.retry_bindings(), "old_failed_cpu60_replayed": False, "old_failed_cpu60_passed": False,
        "old_fixture_failure_preserved": True,
        "parent_tests_previously_passed": 478, "parent_tests_replayed": False,
        "identity_parent_tests_previously_passed": 65, "identity_parent_tests_replayed": False,
        "allocator_tests_previously_passed": 97, "allocator_tests_replayed": False,
        "collector_tests": 40, "metadata_tests": 20, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "actual_vllm_interface_established": False, "experiment_model_loaded": False, "private_cache_opened": False,
        "automatic_downstream": False, "admission_binding_sha256": admission["binding_sha256"],
        "prior_entries_sha256": proof["initial_sha256"], "cpu_bootstrap_sha256": proof["assets"][metadata.BOOTSTRAP]["sha256"],
        "started_sha256": digest(output / "started.json")}
    metadata.validate_native_qa(receipt, proof, {metadata.QA_ROLE: admission})
    write_new_json(output / "native_qa.json", receipt)
    write_new_json(output / "complete.json", {"schema": "extension-source-generation-v3-qa-complete-v1", "complete": True,
        "job_id": compute["job_id"], "tests_run": 60, "cpu_only": True, "gpu_seconds": 0,
        "native_qa_sha256": digest(output / "native_qa.json"), "actual_vllm_interface_established": False,
        **metadata.retry_bindings(), "old_failed_cpu60_replayed": False, "old_failed_cpu60_passed": False,
        "old_fixture_failure_preserved": True,
        "automatic_retry": False, "automatic_downstream": False})
    print(json.dumps({"native_collector_v3_qa_complete": True, "tests_run": 60, "gpu_seconds": 0,
        "parent478_identity65_allocator97_replayed": False}), flush=True)


if __name__ == "__main__": main()
