"""Compute-only QA entry point with a spawn-safe, importable main module."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import (deadline_guard, digest, read_json, require,
    require_cpu_allocation, write_new_json)
from extension.source_readiness_submission import ROLE, initial_evidence, source_inventory


def main():
    compute = require_cpu_allocation()
    deadline_guard(180000)
    root, output = Path.cwd(), Path(sys.argv[1]).resolve()
    require(output.parent == root / "extension/runs" and output.name == "source-readiness-" + compute["job_id"],
        "Exclusive current-allocation QA output required")
    before, parents = source_inventory(root), initial_evidence(root)
    require(len(before) == 67, "Bind exactly the 51 prior plus 16 new/dependency sources")
    admission = read_json(root / "extension/job_ledger.json")["jobs"][ROLE]
    require(admission.get("job_id") == compute["job_id"] and admission.get("status") == "submitted" and
        admission.get("source_sha256") == before and admission.get("requested_gpu_seconds") == 0,
        "CPU job differs from current source admission")
    for name, entry in parents[1].items():
        require(read_json(root / "extension/job_ledger.json")["jobs"].get(name) == entry, "Earlier entry changed")
    subprocess.run(["bash", "-n", str(root / "extension/source_readiness.slurm")], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    patterns = ("test_extension_*.py", "test_labeled_*.py", "test_preserving_*.py", "test_development_*.py",
        "test_preservation_development_seal_v2.py", "test_preservation_retry_submission.py", "test_preservation_data.py",
        "test_native_model_admission.py", "test_smoke_preparation.py", "test_smoke_submission.py",
        "test_source_evaluation.py", "test_pilot_driver.py", "test_source_readiness.py", "test_source_readiness_submission.py")
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name, top_level_dir=str(root / "tests"))
        for name in patterns]
    counts = [suite.countTestCases() for suite in suites]
    require(counts[:10] == [64, 44, 36, 48, 17, 21, 30, 36, 21, 24] and all(count > 0 for count in counts[10:]),
        "Run all 341 prior checks and every exact new module")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    if result.testsRun != sum(counts) or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
        write_new_json(output / "unit_qa_failure.json", {"schema": "extension-source-readiness-native-qa-v1", "success": False,
            "job_id": compute["job_id"], "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "code_sha256":
            {name: meta["sha256"] for name, meta in before.items()}, "no_automatic_retry": True})
        raise RuntimeError("Native CPU checks failed; preserve this fresh attempt")
    require(source_inventory(root) == before and initial_evidence(root) == parents, "Source or parent evidence changed during QA")
    plan_path = root / "extension/source_readiness_plan.json"
    plan = read_json(plan_path)
    receipt = {"schema": "extension-source-readiness-native-qa-v1", "success": True, "job_id": compute["job_id"],
        "node": compute["node"], "compute_proof": compute, "started_at_utc": started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
        "test_counts": dict(zip(patterns, counts)), "failures": 0, "errors": 0, "skipped": 0,
        "expected_failures": 0, "unexpected_successes": 0, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "source_inventory": before, "code_sha256": {name: meta["sha256"] for name, meta in before.items()},
        "plan_sha256": digest(plan_path), "initial_evidence": parents[0], "admission_binding_sha256": admission["binding_sha256"],
        "synthetic_fixture_models_only": True, "experiment_model_loaded": False, "spawn_safe_module_entrypoint": True,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in plan["runtime_versions"]}}}
    write_new_json(output / "unit_qa.json", receipt)
    print(json.dumps({"unit_qa_complete": True, "job_id": compute["job_id"], "tests_run": result.testsRun}), flush=True)


if __name__ == "__main__":
    main()
