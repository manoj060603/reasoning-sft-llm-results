"""Separately versioned CPU-only orchestration after a test-fixture path error.

All86 failed-stage sources and its failed474-check receipt remain unchanged.
Scientific operations delegate to their frozen functions: same32 calibration
records, tokenizer, prompts, generation profile and v2 comparator. This module
owns only retry admission, QA/provenance/output orchestration. It neither calls
the old prepare/current_admission nor changes module globals. No generation,
training, confirmation-output inspection or downstream admission occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import time
import traceback

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json, require,
    require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import inside
from extension import source_readiness_v2 as parent

ROLE, FAILED_ROLE, FAILED_JOB = "source_readiness_v2_20261021_v2", parent.ROLE, "1225820"
PLAN = "extension/source_readiness_v2_retry_plan.json"
PLAN_SCHEMA = "extension-source-readiness-v2-retry-plan-v1"
QA_SCHEMA = "extension-source-readiness-v2-retry-native-qa-v1"
SCHEMA = "extension-source-readiness-v2-retry-complete-v1"
CACHE_SCHEMA = parent.CACHE_SCHEMA
SCIENTIFIC_PLAN = {"path": parent.PLAN, "sha256": "40479982fe3b43e27f51ec27f28a2cf39f76a464ceffdc576e1f38268f0a6a3b"}
LEGACY_PLAN, PROTOCOL_PIN, PRIOR_QA = parent.LEGACY_PLAN, parent.PROTOCOL_PIN, parent.PRIOR_QA
FAILED_DIRECTORY = "extension/report/source-readiness-v2-failed-20261021-1225820"
FAILURE_PINS = {"accounting.json": "f126490b505da796081cd750b17bd6868c45b9955f45763e9e20ab4d20876151",
    "unit_qa_failure.json": "6cd6a0f4a039bb20f8aaf7697846f25e4b3b0473d2177017b1e4152f707ba09f",
    "job_ledger_snapshot.json": "48dafdc5a9bc8e5664b56ba6c3d94af35e82d731c92f7cf9ce465743b2034503",
    "source_install_receipt.json": "7f0d0dc304ced0559c09a626a14527a761bb67184bd0d47979eb4f51cba644c4"}
PRIOR_ENTRIES_SHA256 = "3c97abcf6cd6a01838b54c30c91ce37dc80156c2f8ced93a247fe20874ea79ce"
NEW_SOURCES = frozenset({PLAN, "extension/source_readiness_v2_retry.py", "extension/source_readiness_v2_retry_qa.py",
    "extension/source_readiness_v2_retry.slurm", "tests/test_source_readiness_v2_retry.py"})
legacy, source, joint = parent.legacy, parent.source, parent.joint
CALIBRATION_COUNTS = parent.CALIBRATION_COUNTS
# Exact scientific callable objects, not copied implementations or runtime patches.
build_calibration_cache = parent.build_calibration_cache
gold_mutation = parent.gold_mutation
comparator_identity = parent.comparator_identity
validate_comparator_result = parent.validate_comparator_result
validate_protocol = parent.validate_protocol


def _scientific_projection(plan):
    value = deepcopy(plan)
    for key in ("scientific_plan", "retry_of", "failure_evidence", "prior_entries_sha256", "fixture_correction", "retry_qualification"):
        value.pop(key, None)
    value.update(schema=parent.PLAN_SCHEMA, status="prospective_unadmitted")
    value["native_qa"].update(new_readiness_tests=24, source_inventory_files=86,
        new_patterns=["test_source_evaluation_v2.py", "test_source_readiness_v2.py"])
    return value


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA and plan.get("scientific_plan") == SCIENTIFIC_PLAN and
        plan.get("retry_of") == {"role": FAILED_ROLE, "job_id": FAILED_JOB} and
        plan.get("failure_evidence") == {name: {"path": FAILED_DIRECTORY + "/" + name, "sha256": checksum}
            for name, checksum in FAILURE_PINS.items()} and plan.get("prior_entries_sha256") == PRIOR_ENTRIES_SHA256,
        "Exact failed-attempt lineage and fifteen unchanged prior admissions are required")
    correction = plan.get("fixture_correction", {})
    require(correction == {"original_test": "tests/test_source_readiness_v2.py",
        "original_sha256": "5971cfc15b30255e1b7528ad276de92293510d30980ad91935c6c730b63c008a",
        "replacement_test": "tests/test_source_readiness_v2_retry.py", "copied_tests": 24, "new_metadata_tests": 4,
        "relative_plan_argument": True, "original_scope_executed": False}, "Only the separately identified fixture/controller adaptation is allowed")
    qa = plan.get("native_qa", {})
    require(type(qa) is dict and qa.get("new_readiness_tests") == 28 and qa.get("source_inventory_files") == 91 and
        qa.get("new_patterns") == ["test_source_evaluation_v2.py", "test_source_readiness_v2_retry.py"],
        "Execute the explicit478-check scope with the new28-test replacement module")
    parent.validate_plan(_scientific_projection(plan))
    return plan


def failure_evidence(root, evidence):
    values = {name: evidence.read(inside(root, FAILED_DIRECTORY + "/" + name), checksum)
        for name, checksum in FAILURE_PINS.items()}
    qa, accounting, archive = (values[name] for name in ("unit_qa_failure.json", "accounting.json", "job_ledger_snapshot.json"))
    require(qa.get("schema") == parent.QA_SCHEMA and qa.get("success") is False and qa.get("job_id") == FAILED_JOB and
        qa.get("tests_run") == 474 and qa.get("failures") == 0 and qa.get("errors") == 2 and qa.get("skipped") == 0 and
        qa.get("no_automatic_retry") is True and qa.get("old465_scope_replayed") is False and
        qa.get("test_counts") == dict(zip(list(joint.PATTERNS) + ["test_source_evaluation_v2.py", "test_source_readiness_v2.py"],
            list(joint.COUNTS) + [40, 24])), "Keep the exact472-pass/two-fixture-error failed receipt")
    require(accounting.get("schema") == "extension-terminal-cpu-accounting-v1" and accounting.get("complete") is True and
        accounting.get("job_id") == FAILED_JOB and accounting.get("state") == "FAILED" and accounting.get("exit_code") == "1:0" and
        accounting.get("elapsed_seconds") == 71 and accounting.get("gpu_seconds") == 0 and accounting.get("stderr") == "" and
        accounting.get("no_gpu_in_alloc_tres") is True, "The failed attempt must remain terminal71s/zeroGPU")
    rows = [line.split("|") for line in accounting.get("stdout", "").splitlines() if line]
    require(len(rows) == 3 and {row[0] for row in rows} == {FAILED_JOB, FAILED_JOB + ".batch", FAILED_JOB + ".extern"} and
        all(len(row) == 8 and row[4] == "71" and "gpu" not in row[7].lower() and row[2:4] ==
            (["COMPLETED", "0:0"] if row[0].endswith(".extern") else ["FAILED", "1:0"]) for row in rows),
        "Exact failed allocation/batch and successful extern accounting rows are required")
    require(type(archive.get("jobs")) is dict and len(archive["jobs"]) == 15 and
        value_digest(archive["jobs"]) == PRIOR_ENTRIES_SHA256, "Preserve the fifteen-entry native failure snapshot")
    return values


def source_inventory(root):
    root, evidence = Path(root).resolve(), Evidence()
    values = failure_evidence(root, evidence)
    result = parent.source_inventory(root)
    require(len(result) == 86 and result == values["unit_qa_failure.json"].get("source_inventory") ==
        values["source_install_receipt.json"].get("source_inventory"), "All eighty-six executed sources must remain byte-identical")
    for name in sorted(NEW_SOURCES):
        result[name] = joint._record(root, name)
    require(len(result) == 91 and sum(item["bytes"] for item in result.values()) <= joint.previous.metadata.MAX_TOTAL_SOURCE_BYTES,
        "Exactly ninety-one bounded source/config/test files are required")
    evidence.unchanged()
    return result


def current_admission(root, compute, sources, plan_sha256):
    ledger = read_json(Path(root) / "extension/job_ledger.json")
    require(type(ledger) is dict and ledger.get("schema") == joint.previous.SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict,
        "Preserve the isolated extension ledger/history")
    entry, prior = ledger["jobs"].get(ROLE, {}), {key: value for key, value in ledger["jobs"].items() if key != ROLE}
    require(type(entry) is dict, "A structured current retry admission is required")
    binding = entry.get("binding", {})
    require(type(binding) is dict and entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == 0 and entry.get("requested_gpu_hours") == 0 and entry.get("source_sha256") == sources == binding.get("source_sha256") and
        binding.get("role") == ROLE and binding.get("plan_sha256") == plan_sha256 and binding.get("scientific_plan_sha256") == SCIENTIFIC_PLAN["sha256"] and
        binding.get("legacy_plan_sha256") == LEGACY_PLAN["sha256"] and binding.get("worker_protocol_sha256") == PROTOCOL_PIN["sha256"] and
        len(prior) == 15 and value_digest(prior) == PRIOR_ENTRIES_SHA256 == binding.get("prior_entries_sha256") and
        binding.get("wall_time_seconds") == 7200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_downstream") is False and entry.get("binding_sha256") == value_digest(binding),
        "A separate exact91-source current retry admission preserving all15 prior entries is required")
    return entry, value_digest(prior)


def _read_plans(root, plan_path, evidence):
    plan = validate_plan(evidence.read(plan_path))
    scientific, old, protocol = parent._read_plans(root, inside(root, SCIENTIFIC_PLAN["path"]), evidence)
    require(evidence.check(inside(root, SCIENTIFIC_PLAN["path"]), SCIENTIFIC_PLAN["sha256"]) == SCIENTIFIC_PLAN["sha256"] and
        _scientific_projection(plan) == scientific, "All scientific plan/cohort/profile fields remain exact")
    failure_evidence(root, evidence)
    return plan, old, protocol


def prepare(root, plan_path, output, qa_receipt):
    compute, begun = require_cpu_allocation(), time.monotonic()
    legacy.native._host_resources(compute, {"host_cpus": 8, "host_ram_gib": 64})
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root == Path(__file__).resolve().parents[1] and output.parent == root / "extension/runs" and
        output.name == "source-readiness-v2-retry-" + compute["job_id"] and output.is_dir() and
        {path.name for path in output.iterdir()} == {"started.json", "unit_qa.json"} and Path(qa_receipt).resolve() == output / "unit_qa.json",
        "Use an exclusive current retry output containing only this allocation's QA")
    evidence, observations = Evidence(), []
    plan_path = inside(root, str(plan_path))
    plan, old, protocol = _read_plans(root, plan_path, evidence)
    deadline_guard(180000)
    sources = source_inventory(root)
    admission = current_admission(root, compute, sources, evidence.check(plan_path))
    qa = evidence.read(output / "unit_qa.json")
    patterns, counts = list(joint.PATTERNS) + plan["native_qa"]["new_patterns"], list(joint.COUNTS) + [40, 28]
    require(qa.get("schema") == QA_SCHEMA and qa.get("success") is True and qa.get("job_id") == compute["job_id"] and
        qa.get("node") == compute["node"] and qa.get("gpu_disabled") is True and qa.get("tests_run") == 478 and
        qa.get("test_counts") == dict(zip(patterns, counts)) and all(type(qa.get(key)) is int and qa[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and qa.get("source_inventory") == sources and
        qa.get("code_sha256") == {key: value["sha256"] for key, value in sources.items()} and qa.get("plan_sha256") == evidence.check(plan_path) and
        qa.get("scientific_plan_sha256") == SCIENTIFIC_PLAN["sha256"] and qa.get("legacy_plan_sha256") == LEGACY_PLAN["sha256"] and
        qa.get("worker_protocol_sha256") == PROTOCOL_PIN["sha256"] and qa.get("old465_scope_replayed") is False and
        qa.get("old_readiness24_executed") is False and qa.get("admission_binding_sha256") == admission[0]["binding_sha256"] and
        qa.get("compute_proof", {}).get("job_id") == compute["job_id"] and qa["compute_proof"].get("gpu_requested_or_allocated") is False,
        "New same-allocation478-check retry QA and all91 exact source bindings are required")
    started = evidence.read(output / "started.json", qa.get("started_sha256"))
    require(started.get("schema") == QA_SCHEMA and started.get("job_id") == compute["job_id"] and
        started.get("source_inventory") == sources and started.get("old465_scope_replayed") is False and
        started.get("old_readiness24_executed") is False, "Preserve the new same-allocation retry QA start")
    for name, meta in sources.items():
        evidence.check(inside(root, name), meta["sha256"])
    try:
        for package, expected in old["runtime_versions"].items():
            require(importlib.metadata.version(package) == expected, "Unpinned retry runtime: " + package)
        for pin in list(old["prior_cpu"].values()) + list(old["grader_source"].values()):
            evidence.check(inside(root, pin["path"]), pin["sha256"])
        prior = evidence.read(inside(root, old["prior_cpu"]["complete.json"]["path"]))
        require(prior.get("complete") is True and prior.get("cpu_only") is True and prior.get("job_id") == "1224497" and
            prior.get("input_snapshot_unchanged") is True, "The prior question-only audit/cache remains required")
        seal, _, _, cohort = legacy.prepared_source.load_development(root, old["development_seal"], evidence)
        directory = inside(root, old["development_seal"]["path"]).parent
        records = legacy.prepared_source._jsonl(directory / "calibration.jsonl", evidence)
        base = legacy.prepared_source._validate_base(root, old["base"], evidence)
        tokenizer = legacy.prepared_source._load_tokenizer(base)
        identity = {"repository": legacy.native.REPOSITORY, "revision": legacy.native.REVISION, "files": legacy.native.TOKENIZER_FILES,
            "chat_template_sha256": legacy.prepared_source.text_sha256(tokenizer.chat_template), "eos_token_id": tokenizer.eos_token_id}
        require(tokenizer.eos_token_id == 151645, "Original native EOS differs")
        rows, prechecks = build_calibration_cache(records, tokenizer, identity, expected_comparator=comparator_identity(root, old),
            grading_observations=observations)
        cache = {"schema": CACHE_SCHEMA, "complete": True, "plan_sha256": SCIENTIFIC_PLAN["sha256"], "execution_plan_sha256": evidence.check(plan_path),
            "legacy_plan": LEGACY_PLAN, "worker_protocol_pin": PROTOCOL_PIN, "worker_protocol": source.PROTOCOL, "records": rows,
            "model": {"repository": legacy.native.REPOSITORY, "revision": legacy.native.REVISION, **old["base"]}, "tokenizer_identity": identity,
            "source_generation_profile": old["source_generation_profile"], "development_seal": old["development_seal"], "code_sha256": qa["code_sha256"],
            "calibration_ids_sha256": seal["splits"]["calibration"]["ids_sha256"], "raw_records_private": True, "generation_performed": False,
            "training_performed": False, "retry_of": {"role": FAILED_ROLE, "job_id": FAILED_JOB}}
        evidence.unchanged()
        require(source_inventory(root) == sources and current_admission(root, compute, sources, evidence.check(plan_path)) == admission,
            "Retry sources/current/prior admissions changed")
        ending = require_cpu_allocation()
        require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
        evidence.unchanged()
        require(current_admission(root, compute, sources, evidence.check(plan_path)) == admission,
            "Retry current/prior admissions changed while obtaining ending allocation proof")
        write_new_json(output / "calibration_gold_prechecks.json", prechecks)
        write_new_json(output / "calibration_prompt_cache.json", cache)
        counts = [len(row["prompt"]["prompt_token_ids"]) for row in rows]
        complete = {"schema": SCHEMA, "complete": True, "cpu_only": True, "job_id": compute["job_id"], "gpu_seconds": 0,
            "plan_sha256": evidence.check(plan_path), "scientific_plan_sha256": SCIENTIFIC_PLAN["sha256"], "legacy_plan_sha256": LEGACY_PLAN["sha256"],
            "worker_protocol_sha256": PROTOCOL_PIN["sha256"], "worker_protocol": source.PROTOCOL, "retry_of": {"role": FAILED_ROLE, "job_id": FAILED_JOB},
            "source_inventory": sources, "code_sha256": qa["code_sha256"], "tests_run": 478, "native_qa_sha256": digest(output / "unit_qa.json"),
            "record_count": 32, "domain_counts": CALIBRATION_COUNTS, "group_count": 32, "maximum_prompt_tokens": max(counts),
            "minimum_prompt_tokens": min(counts), "prompt_token_sum": sum(counts), "native_gold_mutation_invariance_passed": True,
            "gold_self_comparisons_passed": len(prechecks), "calibration_only_gold_validation": True,
            "source_generation_profile_sha256": value_digest(old["source_generation_profile"]), "source_cohorts": cohort, "input_files": evidence.files,
            "input_snapshot_unchanged": True, "output_sha256": {name: digest(output / name) for name in
                ("calibration_gold_prechecks.json", "calibration_prompt_cache.json", "unit_qa.json", "started.json")},
            "admission_binding_sha256": admission[0]["binding_sha256"], "prior_entries_sha256": admission[1], "compute_start": compute, "compute_end": ending,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begun, "old465_scope_replayed": False,
            "old_readiness24_executed": False, "old_five_second_end_to_end_contract_validated": False, "confirmation_outputs_opened": False,
            "generation": False, "training_pilot": False, "automatic_downstream": False, "raw_cache_private": True}
        write_new_json(output / "complete.json", complete)
        return complete
    except BaseException as error:
        write_new_json(output / "calibration_failure_private.json", {"exception": type(error).__name__, "original_exception_message": str(error),
            "original_traceback": traceback.format_exc(), "grading_results": observations, "raw_grades_private": True})
        write_new_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "job_id": compute["job_id"], "exception": type(error).__name__,
            "calibration_prepared": False, "grading_results_private": True, "records_attempted": len(observations), "no_drop_refill_retry": True,
            "automatic_downstream": False, "retry_of": {"role": FAILED_ROLE, "job_id": FAILED_JOB}})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.root, args.plan, args.output, args.qa_receipt)
    except BaseException:
        raise RuntimeError("Source readiness v2 retry failed. See private failure evidence.") from None
    print({key: result[key] for key in ("complete", "record_count", "tests_run", "maximum_prompt_tokens", "gpu_seconds")}, flush=True)


if __name__ == "__main__":
    main()
