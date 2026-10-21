"""UNADMITTED CPU-only calibration prompt preparation using worker protocol v2.

The immutable legacy plan supplies the exact same 32 calibration records,
original tokenizer/base, prompts and prospective generation profile. Only the
separately identified source MATH worker setup/work lifecycle changes. No model
generation, training, confirmation-label format development or downstream job
admission occurs. Raw records, gold self-checks and prompts remain private.
The old465-test grader scope is neither replayed nor declared repaired.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import time
import traceback

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import inside
from extension import joint_memory_qa as joint
from extension import source_readiness as legacy
from extension import source_evaluation_v2 as source


ROLE = "source_readiness_v2_20261021_v1"
PLAN = "extension/source_readiness_v2_plan.json"
PLAN_SCHEMA = "extension-source-readiness-v2-plan-v1"
QA_SCHEMA = "extension-source-readiness-v2-native-qa-v1"
SCHEMA = "extension-source-readiness-v2-complete-v1"
CACHE_SCHEMA = "extension-source-calibration-prompt-cache-v2"
LEGACY_PLAN = {"path": "extension/source_readiness_plan.json",
    "sha256": "70c6823627d8bc93ce927665b87b943e1bf22f1ddc99eb2fd511a778696b590d"}
PROTOCOL_PIN = {"path": "extension/planning/source_grader_v2_protocol.json",
    "sha256": "c2cd03ac8281116810da029eed76116b21dfe5a7d78772765d9773c2ff216e2a"}
PRIOR_QA = {"path": "extension/report/joint-memory-qa-20261021-1225557/complete.json",
    "sha256": "66a6ba25cea01a70ee3a945d1dce9cddf474915ab6b32e506de3058d23a8f51f"}
FROZEN_V2 = {"extension/source_evaluation_v2.py":
    "639e3964a750f3af77d90b4aa4aa6f5e166fd20d494c07abeb2eb08a29e226c4",
    "tests/test_source_evaluation_v2.py": "57c2b630f2d9b77da0dab279e4f3ba00faac8d2f6a4edc6344b82ef7c9c5010a"}
NEW_SOURCES = frozenset(FROZEN_V2) | frozenset({"extension/source_readiness_v2.py",
    "tests/test_source_readiness_v2.py", PLAN, "extension/source_readiness_v2_qa.py", "extension/source_readiness_v2.slurm"})
CALIBRATION_COUNTS = legacy.CALIBRATION_COUNTS
gold_mutation = legacy.gold_mutation


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA and plan.get("cpu_only") is True and
        plan.get("prepare_calibration_prompts") is True and all(plan.get(key) is False for key in
            ("training", "model_inference", "automatic_downstream")), "Only separate CPU source-readiness v2 is allowed")
    require(plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and plan.get("wall_time_seconds") == 7200 and
        plan.get("reserve_seconds") == 172800 and plan.get("resources") == {"cpus": 8, "memory_gib": 64, "gpus": 0},
        "Preserve exact CPU8/64G/2h/zeroGPU and the48-hour reserve")
    require(plan.get("legacy_plan") == LEGACY_PLAN and plan.get("worker_protocol") == PROTOCOL_PIN and
        plan.get("predecessor_joint_qa") == PRIOR_QA and plan.get("new_frozen_sources") == FROZEN_V2,
        "All immutable legacy plan/protocol/source prerequisites must remain exact")
    require(plan.get("calibration") == {"split": "calibration", "record_count": 32, "domain_counts": CALIBRATION_COUNTS,
        "group_count": 32, "maximum_prompt_tokens": 8192, "validate_gold_only_calibration": True,
        "gold_self_comparison": True, "drop_refill_or_replace": False}, "Use exactly the same sealed calibration population")
    qa = plan.get("native_qa", {})
    require(qa.get("legacy_patterns") == list(joint.PATTERNS) and qa.get("legacy_counts") == list(joint.COUNTS) and
        qa.get("legacy_tests") == 410 and qa.get("v2_helper_tests") == 40 and
        type(qa.get("new_readiness_tests")) is int and qa["new_readiness_tests"] > 0 and
        qa.get("source_inventory_files") == 86 and qa.get("old465_scope_replayed") is False and
        qa.get("new_patterns") == ["test_source_evaluation_v2.py", "test_source_readiness_v2.py"],
        "Explicitly scope legacy410 plus newv2/helper preparation checks, never the failed465 scope")
    require(plan.get("generation_profile") == "unchanged_legacy_plan.source_generation_profile" and
        plan.get("source_generated_answers_used") is False and plan.get("confirmation_outputs_opened") is False,
        "CPU preparation cannot change source generation or inspect model/source-generated outcomes")
    return plan


def source_inventory(root):
    """Exactly79 historically bound sources plus these7 separately new sources."""
    root = Path(root).resolve()
    result = joint.source_inventory(root)
    archived = read_json(inside(root, PRIOR_QA["path"]))
    require(digest(inside(root, PRIOR_QA["path"])) == PRIOR_QA["sha256"] and archived.get("complete") is True and
        archived.get("cpu_only") is True and archived.get("tests_run") == 410 and archived.get("gpu_seconds") == 0 and
        archived.get("source_inventory") == result, "All79 executed joint-QA source bytes remain immutable")
    for name in sorted(NEW_SOURCES):
        result[name] = joint._record(root, name)
    require(len(result) == 86 and all(result[name]["sha256"] == checksum for name, checksum in FROZEN_V2.items()) and
        sum(value["bytes"] for value in result.values()) <= joint.previous.metadata.MAX_TOTAL_SOURCE_BYTES,
        "The exact frozen86-source v2 inventory is required")
    return result


def validate_protocol(protocol, legacy_plan):
    require(type(protocol) is dict and protocol.get("schema") == "extension-source-grader-v2-prospective-protocol-v1" and
        protocol.get("comparator_source") == legacy_plan["grader_source"] and
        protocol.get("runtime_versions") == legacy_plan["runtime_versions"] and
        protocol.get("historical_and_final_benchmark_grading") == "unchanged",
        "The prospective v2 protocol must preserve pinned comparator/runtime and historical benchmark grading")
    expected = {"setup_seconds": 10, "work_seconds": 5, "cleanup_seconds": .55, "maximum_total_seconds": 15.55,
        "setup_clock_starts_before_spawn": True, "work_clock_starts_immediately_before_go_send": True,
        "go_messages": 1, "fresh_worker_per_math_response": True, "worker_reuse": False, "automatic_retry": False,
        "late_results_accepted": False, "cleanup_error_blocks_further_grading_and_ranking": True}
    require(all(type(protocol.get("worker_protocol", {}).get(key)) is type(value) and
        protocol["worker_protocol"][key] == value for key, value in expected.items()),
        "Do not change fixed10/5/.55 phase/deadline/error rules")
    return protocol


def comparator_identity(root, legacy_plan):
    root = Path(root).resolve()
    return {"loaded_modules": {"utils." + Path(name).stem: str(inside(root, pin["path"]))
        for name, pin in legacy_plan["grader_source"].items()},
        "source_sha256": {"utils." + Path(name).stem: pin["sha256"] for name, pin in legacy_plan["grader_source"].items()},
        "runtime_versions": {name: legacy_plan["runtime_versions"][name] for name in source.PACKAGE_NAMES}}


def validate_comparator_result(result, expected):
    require(type(result) is dict and result.get("status") == "graded" and type(result.get("correct")) is bool and
        result.get("rankable") is True and result.get("denominator") == 1 and result.get("error") is None and
        result.get("worker_protocol") == source.PROTOCOL and result.get("caller_must_abort") is False,
        "A successful separately versioned source MATH result is required")
    proof, worker = result.get("comparator_provenance"), result.get("worker", {})
    source._validate_proof(proof, expected)
    require(type(worker) is dict and worker.get("status") == "graded" and worker.get("correct") is result["correct"] and
        worker.get("go_sent") is True and worker.get("worker_protocol") == source.PROTOCOL and
        worker.get("comparator_provenance") == proof and worker.get("cleanup", {}).get("successful") is True,
        "The result must bind its verified READY, oneGO and successful cleanup")
    timing = worker.get("timings", {})
    require(all(timing.get(key) == value for key, value in {"declared_setup_deadline_seconds": 10.,
        "declared_comparison_deadline_seconds": 5., "declared_cleanup_join_budget_seconds": .55,
        "declared_total_budget_seconds": 15.55}.items()) and
        all(type(timing.get(key)) in (int, float) and math.isfinite(timing[key]) and timing[key] >= 0 for key in
            ("setup_elapsed_seconds", "comparison_elapsed_seconds", "total_elapsed_including_cleanup_seconds")),
        "Verified phase budgets and observed timings are required; no OS timing guarantee is inferred")


def build_calibration_cache(records, tokenizer, tokenizer_identity, *, expected_comparator, grader=None, grading_observations=None):
    """Reuse unchanged cohort/prompt checks without calling the v1 proof validator."""
    source._expected_identity(expected_comparator)
    require(grading_observations is None or type(grading_observations) is list, "Private in-memory grading journal must be a list")
    grade = source.grade_response if grader is None else grader

    def checked_grade(row, text, timeout_seconds):
        try:
            result = grade(row, text, timeout_seconds=timeout_seconds, expected_comparator=expected_comparator)
        except source.SourceGraderCleanupError as error:
            if grading_observations is not None:
                grading_observations.append(deepcopy(error.result))
            raise
        if grading_observations is not None:
            grading_observations.append(deepcopy(result))
        require(type(result) is dict and result.get("record_id") == row["id"] and result.get("source_kind") == row["source_kind"] and
            result.get("denominator") == 1 and result.get("worker_protocol") == source.PROTOCOL and
            result.get("caller_must_abort") is False, "Source self-check result identity/protocol differs")
        if row["source_kind"] == "math":
            validate_comparator_result(result, expected_comparator)
        return result

    # None is intentional: the old helper's v1 timeout-owner assertion must not
    # be run on a new, explicitly different operational protocol.
    return legacy.build_calibration_cache(records, tokenizer, tokenizer_identity,
        grader=checked_grade, expected_comparator=None)


def current_admission(root, compute, sources, plan_sha256):
    ledger = read_json(Path(root) / "extension/job_ledger.json")
    require(type(ledger) is dict and ledger.get("schema") == joint.previous.SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict,
        "Keep the isolated extension and historical ledger identity")
    entry, priors = ledger["jobs"].get(ROLE, {}), {key: value for key, value in ledger["jobs"].items() if key != ROLE}
    require(type(entry) is dict and type(entry.get("binding")) is dict, "A current structured admission entry is required")
    binding = entry["binding"]
    require(entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == 0 and entry.get("requested_gpu_hours") == 0 and
        entry.get("source_sha256") == sources and binding.get("source_sha256") == sources and binding.get("role") == ROLE and
        binding.get("plan_sha256") == plan_sha256 and binding.get("legacy_plan_sha256") == LEGACY_PLAN["sha256"] and
        binding.get("worker_protocol_sha256") == PROTOCOL_PIN["sha256"] and
        binding.get("prior_entries_sha256") == value_digest(priors) and binding.get("wall_time_seconds") == 7200 and
        binding.get("closeout_reserve_seconds") == 172800 and binding.get("automatic_downstream") is False and
        entry.get("binding_sha256") == value_digest(binding),
        "A new exact86-source current CPU admission preserving every actual prior entry is required")
    return entry, value_digest(priors)


def _read_plans(root, plan_path, evidence):
    plan = validate_plan(evidence.read(plan_path))
    old = legacy.validate_plan(evidence.read(inside(root, LEGACY_PLAN["path"]), LEGACY_PLAN["sha256"]))
    protocol = validate_protocol(evidence.read(inside(root, PROTOCOL_PIN["path"]), PROTOCOL_PIN["sha256"]), old)
    evidence.check(inside(root, PRIOR_QA["path"]), PRIOR_QA["sha256"])
    return plan, old, protocol


def prepare(root, plan_path, output, qa_receipt):
    compute, begun = require_cpu_allocation(), time.monotonic()
    legacy.native._host_resources(compute, {"host_cpus": 8, "host_ram_gib": 64})
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root == Path(__file__).resolve().parents[1] and output.parent == root / "extension/runs" and
        output.name == "source-readiness-v2-" + compute["job_id"] and output.is_dir() and
        {path.name for path in output.iterdir()} == {"started.json", "unit_qa.json"} and
        Path(qa_receipt).resolve() == output / "unit_qa.json", "Exclusive current v2 CPU QA/preparation output required")
    evidence, observations = Evidence(), []
    plan_path = inside(root, str(plan_path))
    plan, old, protocol = _read_plans(root, plan_path, evidence)
    deadline_guard(180000)
    sources = source_inventory(root)
    admission = current_admission(root, compute, sources, evidence.check(plan_path))
    qa = evidence.read(output / "unit_qa.json")
    expected_patterns = list(joint.PATTERNS) + plan["native_qa"]["new_patterns"]
    expected_counts = list(joint.COUNTS) + [40, plan["native_qa"]["new_readiness_tests"]]
    require(qa.get("schema") == QA_SCHEMA and qa.get("success") is True and qa.get("job_id") == compute["job_id"] and
        qa.get("node") == compute["node"] and qa.get("gpu_disabled") is True and
        qa.get("tests_run") == sum(expected_counts) and qa.get("test_counts") == dict(zip(expected_patterns, expected_counts)) and
        all(type(qa.get(key)) is int and qa[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("source_inventory") == sources and qa.get("code_sha256") == {key: value["sha256"] for key, value in sources.items()} and
        qa.get("plan_sha256") == evidence.check(plan_path) and qa.get("legacy_plan_sha256") == LEGACY_PLAN["sha256"] and
        qa.get("worker_protocol_sha256") == PROTOCOL_PIN["sha256"] and qa.get("old465_scope_replayed") is False and
        qa.get("admission_binding_sha256") == admission[0]["binding_sha256"] and
        qa.get("compute_proof", {}).get("job_id") == compute["job_id"] and
        qa["compute_proof"].get("gpu_requested_or_allocated") is False, "New same-allocation scoped v2 native QA is required")
    started = evidence.read(output / "started.json", qa.get("started_sha256"))
    require(started.get("schema") == QA_SCHEMA and started.get("job_id") == compute["job_id"] and
        started.get("source_inventory") == sources and started.get("old465_scope_replayed") is False,
        "Preserve the exact same-allocation QA start evidence")
    for name, meta in sources.items():
        evidence.check(inside(root, name), meta["sha256"])
    try:
        for package, expected in old["runtime_versions"].items():
            require(importlib.metadata.version(package) == expected, "Unpinned source readiness runtime: " + package)
        for pin in list(old["prior_cpu"].values()) + list(old["grader_source"].values()):
            evidence.check(inside(root, pin["path"]), pin["sha256"])
        prior = evidence.read(inside(root, old["prior_cpu"]["complete.json"]["path"]))
        require(prior.get("complete") is True and prior.get("cpu_only") is True and prior.get("job_id") == "1224497" and
            prior.get("input_snapshot_unchanged") is True, "The prior all-nine question-only audit/cache is required")
        seal, _, _, cohort_proof = legacy.prepared_source.load_development(root, old["development_seal"], evidence)
        directory = inside(root, old["development_seal"]["path"]).parent
        records = legacy.prepared_source._jsonl(directory / "calibration.jsonl", evidence)
        base = legacy.prepared_source._validate_base(root, old["base"], evidence)
        tokenizer = legacy.prepared_source._load_tokenizer(base)
        identity = {"repository": legacy.native.REPOSITORY, "revision": legacy.native.REVISION,
            "files": legacy.native.TOKENIZER_FILES,
            "chat_template_sha256": legacy.prepared_source.text_sha256(tokenizer.chat_template), "eos_token_id": tokenizer.eos_token_id}
        require(tokenizer.eos_token_id == 151645, "Original native EOS differs")
        expected = comparator_identity(root, old)
        rows, prechecks = build_calibration_cache(records, tokenizer, identity,
            expected_comparator=expected, grading_observations=observations)
        cache = {"schema": CACHE_SCHEMA, "complete": True, "plan_sha256": evidence.check(plan_path),
            "legacy_plan": LEGACY_PLAN, "worker_protocol_pin": PROTOCOL_PIN, "worker_protocol": source.PROTOCOL,
            "records": rows, "model": {"repository": legacy.native.REPOSITORY, "revision": legacy.native.REVISION, **old["base"]},
            "tokenizer_identity": identity, "source_generation_profile": old["source_generation_profile"],
            "development_seal": old["development_seal"], "code_sha256": qa["code_sha256"],
            "calibration_ids_sha256": seal["splits"]["calibration"]["ids_sha256"], "raw_records_private": True,
            "generation_performed": False, "training_performed": False}
        evidence.unchanged()
        require(source_inventory(root) == sources and current_admission(root, compute, sources, evidence.check(plan_path)) == admission,
            "Source/admission changed during v2 CPU preparation")
        ending = require_cpu_allocation()
        require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
        write_new_json(output / "calibration_gold_prechecks.json", prechecks)
        write_new_json(output / "calibration_prompt_cache.json", cache)
        counts = [len(row["prompt"]["prompt_token_ids"]) for row in rows]
        complete = {"schema": SCHEMA, "complete": True, "cpu_only": True, "job_id": compute["job_id"], "gpu_seconds": 0,
            "plan_sha256": evidence.check(plan_path), "legacy_plan_sha256": LEGACY_PLAN["sha256"],
            "worker_protocol_sha256": PROTOCOL_PIN["sha256"], "worker_protocol": source.PROTOCOL,
            "source_inventory": sources, "code_sha256": qa["code_sha256"], "tests_run": qa["tests_run"],
            "native_qa_sha256": digest(output / "unit_qa.json"), "record_count": 32, "domain_counts": CALIBRATION_COUNTS,
            "group_count": 32, "maximum_prompt_tokens": max(counts), "minimum_prompt_tokens": min(counts),
            "prompt_token_sum": sum(counts), "native_gold_mutation_invariance_passed": True,
            "gold_self_comparisons_passed": len(prechecks), "calibration_only_gold_validation": True,
            "source_generation_profile_sha256": value_digest(old["source_generation_profile"]),
            "source_cohorts": cohort_proof, "input_files": evidence.files, "input_snapshot_unchanged": True,
            "output_sha256": {name: digest(output / name) for name in
                ("calibration_gold_prechecks.json", "calibration_prompt_cache.json", "unit_qa.json", "started.json")},
            "admission_binding_sha256": admission[0]["binding_sha256"], "prior_entries_sha256": admission[1],
            "compute_start": compute, "compute_end": ending, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - begun, "old465_scope_replayed": False,
            "old_five_second_end_to_end_contract_validated": False, "confirmation_outputs_opened": False,
            "generation": False, "training_pilot": False, "automatic_downstream": False, "raw_cache_private": True}
        write_new_json(output / "complete.json", complete)
        return complete
    except BaseException as error:
        write_new_json(output / "calibration_failure_private.json", {"exception": type(error).__name__,
            "original_exception_message": str(error), "original_traceback": traceback.format_exc(),
            "grading_results": observations, "raw_grades_private": True})
        write_new_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "job_id": compute["job_id"],
            "exception": type(error).__name__, "calibration_prepared": False, "grading_results_private": True,
            "records_attempted": len(observations), "no_drop_refill_retry": True, "automatic_downstream": False})
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
        # prepare retains its original exception semantics for callers/tests;
        # the CLI must not print a legacy error containing private record IDs.
        raise RuntimeError("Source readiness v2 failed. See private failure evidence.") from None
    print({key: result[key] for key in ("complete", "record_count", "tests_run", "maximum_prompt_tokens", "gpu_seconds")}, flush=True)


if __name__ == "__main__":
    main()
