"""CPU-only478-check QA for the separately identified fixture/controller retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, require,
    require_cpu_allocation, write_new_json)
from extension import source_readiness_v2_retry as stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compute = require_cpu_allocation()
    stage.legacy.native._host_resources(compute, {"host_cpus": 8, "host_ram_gib": 64})
    deadline_guard(180000)
    root, proposed = Path(args.root).resolve(), Path(args.output)
    require(root == Path(__file__).resolve().parents[1] and not proposed.is_symlink(), "Use this installed retry project and exclusive output")
    output = proposed.resolve()
    require(output.parent == root / "extension/runs" and output.name == "source-readiness-v2-retry-" + compute["job_id"] and
        not output.exists(), "Exclusive current retry QA output required")
    evidence = Evidence()
    plan_path = root / stage.PLAN
    plan, old, protocol = stage._read_plans(root, plan_path, evidence)
    before = stage.source_inventory(root)
    admission = stage.current_admission(root, compute, before, evidence.check(plan_path))
    patterns = list(stage.joint.PATTERNS) + plan["native_qa"]["new_patterns"]
    expected = list(stage.joint.COUNTS) + [40, 28]
    for package, version in old["runtime_versions"].items():
        require(importlib.metadata.version(package) == version, "Unadmitted retry runtime: " + package)
    subprocess.run(["bash", "-n", str(root / "extension/source_readiness_v2_retry.slurm")], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    begun = datetime.now(timezone.utc).isoformat()
    write_new_json(output / "started.json", {"schema": stage.QA_SCHEMA, "success": False, "job_id": compute["job_id"],
        "expected_tests": 478, "source_inventory": before, "old465_scope_replayed": False, "old_readiness24_executed": False})
    suites = [unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern, top_level_dir=str(root / "tests")) for pattern in patterns]
    counts = [suite.countTestCases() for suite in suites]
    require(counts == expected and sum(counts) == 478, "Exactly410 old regressions,40 frozen worker tests,28 new retry tests")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    metrics = {"tests_run": result.testsRun, "test_counts": dict(zip(patterns, counts)), "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses)}
    if result.testsRun != 478 or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
        write_new_json(output / "unit_qa_failure.json", {"schema": stage.QA_SCHEMA, "success": False, "job_id": compute["job_id"], **metrics,
            "source_inventory": before, "code_sha256": {key: value["sha256"] for key, value in before.items()}, "compute_proof": compute,
            "old465_scope_replayed": False, "old_readiness24_executed": False, "no_automatic_retry": True})
        raise RuntimeError("New scoped retry QA failed; preserve this attempt")
    require(stage.source_inventory(root) == before and stage.current_admission(root, compute, before, evidence.check(plan_path)) == admission,
        "Current/prior admissions or sources changed during retry QA")
    evidence.unchanged()
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    receipt = {"schema": stage.QA_SCHEMA, "success": True, "job_id": compute["job_id"], "node": compute["node"], "compute_proof": compute,
        "compute_end": ending, "started_at_utc": begun, "created_at_utc": datetime.now(timezone.utc).isoformat(), **metrics,
        "gpu_disabled": True, "wrapper_syntax_passed": True, "source_inventory": before, "code_sha256": {key: value["sha256"] for key, value in before.items()},
        "plan_sha256": evidence.check(plan_path), "scientific_plan_sha256": stage.SCIENTIFIC_PLAN["sha256"],
        "legacy_plan_sha256": stage.LEGACY_PLAN["sha256"], "worker_protocol_sha256": stage.PROTOCOL_PIN["sha256"], "worker_protocol": stage.source.PROTOCOL,
        "retry_of": {"role": stage.FAILED_ROLE, "job_id": stage.FAILED_JOB}, "admission_binding_sha256": admission[0]["binding_sha256"],
        "prior_entries_sha256": admission[1], "started_sha256": digest(output / "started.json"), "old465_scope_replayed": False,
        "old_readiness24_executed": False, "old_five_second_end_to_end_contract_validated": False, "synthetic_fixture_models_only": True,
        "experiment_model_loaded": False, "source_generated_answers_used": False, "automatic_downstream": False,
        "versions": {"python": sys.version, **{name: importlib.metadata.version(name) for name in old["runtime_versions"]}}}
    write_new_json(output / "unit_qa.json", receipt)
    print(json.dumps({"scoped_retry_qa_complete": True, "tests_run": 478, "gpu_seconds": 0, "automatic_downstream": False}), flush=True)


if __name__ == "__main__":
    main()
