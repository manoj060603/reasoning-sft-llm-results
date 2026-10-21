"""Fresh CPU-only QA for the archived, fixture-only source-readiness retry.

The failed 1225252 attempt and original QA/wrapper remain immutable. This
spawn-safe module repeats every one of its 445 tests, including the corrected
empty-gold fixture, then the separately bound retry-admission tests. It performs
no generation, scientific selection, model-weight loading or GPU submission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import (deadline_guard, digest, read_json, require,
    require_cpu_allocation, write_new_json)
from extension.source_readiness_retry_submission import ROLE, initial_evidence, source_inventory


PATTERNS = ("test_extension_*.py", "test_labeled_*.py", "test_preserving_*.py", "test_development_*.py",
    "test_preservation_development_seal_v2.py", "test_preservation_retry_submission.py", "test_preservation_data.py",
    "test_native_model_admission.py", "test_smoke_preparation.py", "test_smoke_submission.py",
    "test_source_evaluation.py", "test_pilot_driver.py", "test_source_readiness.py", "test_source_readiness_submission.py",
    "test_source_readiness_retry_submission.py")
PRIOR_COUNTS = (64, 44, 36, 48, 17, 21, 30, 36, 21, 24, 37, 44, 9, 14)


def main():
    compute = require_cpu_allocation()
    deadline_guard(180000)
    root, output = Path.cwd(), Path(sys.argv[1]).resolve()
    require(output.parent == root / "extension/runs" and output.name == "source-readiness-" + compute["job_id"],
        "Exclusive current-allocation retry QA output required")
    before, parents = source_inventory(root), initial_evidence(root)
    require(len(before) == 71, "Bind exactly the 67 original plus four retry sources")
    admission = read_json(root / "extension/job_ledger.json")["jobs"][ROLE]
    require(admission.get("job_id") == compute["job_id"] and admission.get("status") == "submitted" and
        admission.get("source_sha256") == before and admission.get("requested_gpu_seconds") == 0,
        "CPU retry differs from its current source admission")
    require(len(parents[1]) == 9, "Preserve the exact nine prior ledger entries, including failed 1225252")
    for name, entry in parents[1].items():
        require(read_json(root / "extension/job_ledger.json")["jobs"].get(name) == entry, "Earlier entry changed")
    subprocess.run(["bash", "-n", str(root / "extension/source_readiness_retry.slurm")], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests"))
        for name in PATTERNS]
    counts = [suite.countTestCases() for suite in suites]
    require(tuple(counts[:-1]) == PRIOR_COUNTS and sum(counts[:-1]) == 445 and counts[-1] == 20,
        "Repeat the exact 445 prior tests and the complete new retry-admission module")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    if result.testsRun != sum(counts) or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
        write_new_json(output / "unit_qa_failure.json", {"schema": "extension-source-readiness-native-qa-v1", "success": False,
            "job_id": compute["job_id"], "retry_of_job_id": "1225252", "tests_run": result.testsRun,
            "test_counts": dict(zip(PATTERNS, counts)), "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses), "source_inventory": before,
            "code_sha256": {name: meta["sha256"] for name, meta in before.items()},
            "compute_proof": compute, "no_automatic_retry": True})
        raise RuntimeError("Fresh native CPU retry checks failed; preserve this attempt")
    require(source_inventory(root) == before and initial_evidence(root) == parents,
        "Source or archived parent evidence changed during retry QA")
    plan_path = root / "extension/source_readiness_plan.json"
    plan = read_json(plan_path)
    receipt = {"schema": "extension-source-readiness-native-qa-v1", "success": True, "job_id": compute["job_id"],
        "retry_of_job_id": "1225252", "node": compute["node"], "compute_proof": compute, "started_at_utc": started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
        "test_counts": dict(zip(PATTERNS, counts)), "failures": 0, "errors": 0, "skipped": 0,
        "expected_failures": 0, "unexpected_successes": 0, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "source_inventory": before, "code_sha256": {name: meta["sha256"] for name, meta in before.items()},
        "plan_sha256": digest(plan_path), "initial_evidence": parents[0], "admission_binding_sha256": admission["binding_sha256"],
        "synthetic_fixture_models_only": True, "experiment_model_loaded": False, "spawn_safe_module_entrypoint": True,
        "original_failed_attempt_preserved": True, "scorer_protocol_and_source_cohorts_unchanged": True,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in plan["runtime_versions"]}}}
    write_new_json(output / "unit_qa.json", receipt)
    print(json.dumps({"unit_qa_complete": True, "retry_of_job_id": "1225252", "job_id": compute["job_id"],
        "tests_run": result.testsRun}), flush=True)


if __name__ == "__main__":
    main()
