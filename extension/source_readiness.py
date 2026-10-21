"""CPU-only source prompt preparation after the sealed question audits.

This stage prepares the fixed 32-record calibration cohort and exercises no
experiment model. It never generates source answers, opens confirmation outputs,
selects training pairs or admits a GPU stage. Raw labels and native prompts stay
private. The independent confirmation labels are not used for grader development.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import inside
from extension import native_model_admission as native
from extension import smoke_preparation as prepared_source
from extension import source_evaluation as source

PLAN_SCHEMA = "extension-source-readiness-plan-v1"
QA_SCHEMA = "extension-source-readiness-native-qa-v1"
SCHEMA = "extension-source-readiness-complete-v1"
CALIBRATION_COUNTS = {"math": 16, "arc_challenge": 16}
DEVELOPMENT_SEAL_SHA256 = "354859dd44e2fe3aa7800ec991b60649f00a99ce4768d8075af42c007289987d"
PRIOR_COMPLETE_SHA256 = "b7d8dc41b6fd90befc6a85130b3904245aea6dbd2cafc3209941d647a3ab8df5"
REQUIRED_CODE = frozenset({"extension/source_readiness.py", "tests/test_source_readiness.py",
    "extension/source_evaluation.py", "tests/test_source_evaluation.py", "extension/pilot_driver.py",
    "tests/test_pilot_driver.py", "extension/source_readiness_submission.py", "tests/test_source_readiness_submission.py",
    "extension/source_readiness_plan.json", "extension/source_readiness.slurm", "extension/source_readiness_qa.py",
    "eval/utils/grader.py", "eval/utils/parser.py", "eval/utils/math_normalization.py", "eval/utils/utils.py",
    "eval/utils/examples.py"})


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA and plan.get("cpu_only") is True and
        plan.get("prepare_calibration_prompts") is True and all(plan.get(name) is False for name in
        ("training", "model_inference", "automatic_downstream")), "Only fixed CPU source readiness is permitted")
    require(plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and plan.get("wall_time_seconds") == 7200 and
        plan.get("reserve_seconds") == 172800, "Preserve the finite CPU bound and 48-hour reserve")
    calibration = plan.get("calibration", {})
    require(calibration.get("split") == "calibration" and calibration.get("record_count") == 32 and
        calibration.get("domain_counts") == CALIBRATION_COUNTS and calibration.get("group_count") == 32 and
        calibration.get("maximum_prompt_tokens") == 8192 and calibration.get("timeout_seconds") == 5 and
        calibration.get("validate_gold_only_calibration") is True and calibration.get("gold_self_comparison") is True,
        "Do not change, replace or reuse the sealed source calibration population")
    require(plan.get("development_seal") == {"path":
        "extension/runs/development-seal-v2-20261021-1223943/complete.json", "sha256": DEVELOPMENT_SEAL_SHA256} and
        plan.get("prior_cpu", {}).get("complete.json") == {"path":
        "extension/report/native-smoke-preparation-20261021-1224497/complete.json", "sha256": PRIOR_COMPLETE_SHA256},
        "Exact completed cohort and all-nine exclusion evidence is required")
    generation = plan.get("source_generation_profile", {})
    expected = {"n": 8, "max_new_tokens": 8192, "max_model_len": 16384, "seed": 20261021,
        "native_eos_token_id": 151645, "ignore_eos": False, "generation_config": "vllm", "top_k": -1,
        "temperature": .6, "top_p": .95, "min_p": 0., "repetition_penalty": 1., "presence_penalty": 0.,
        "frequency_penalty": 0., "min_tokens": 0, "additional_stop_strings": [], "additional_stop_token_ids": [],
        "batch_size": 32, "max_num_seqs": 128, "gpu_memory_utilization": .85, "tensor_parallel_size": 1,
        "dtype": "bfloat16", "hardware": "B200",
        "per_record_seed": "(20261021+int(record_id[:8],16)) modulo2**31"}
    require(all(generation.get(key) == value for key, value in expected.items()),
        "Prospective calibration sampling and native context/EOS policy differ")
    require(isinstance(plan.get("runtime_versions"), dict) and plan["runtime_versions"] and
        isinstance(plan.get("grader_source"), dict) and set(plan["grader_source"]) ==
        {"grader.py", "parser.py", "math_normalization.py", "utils.py", "examples.py"}, "Pin grader runtime/transitive sources")
    require(plan.get("base", {}).get("files") == native.ORIGINAL_FILES and
        isinstance(plan["base"].get("path"), str) and Path(plan["base"]["path"]).is_absolute(), "Use the exact original base")
    return plan


def gold_mutation(row):
    """Change only gold/solution fields, retaining question and choice bytes."""
    changed = deepcopy(row)
    changed["label"] = {"answer": "source-gold-mutation", "answer_key": "source-gold-mutation"}
    changed["label_id"] = changed["label_sha256"] = "0" * 64
    raw = changed.get("original_record", {})
    for name in ("solution", "answerKey"):
        if name in raw:
            raw[name] = "source-gold-mutation"
    return changed


def validate_comparator_result(result, expected):
    provenance = result.get("comparator_provenance", {})
    require(isinstance(provenance, dict) and provenance.get("loaded_modules") == expected["loaded_modules"] and
        provenance.get("runtime_versions") == expected["runtime_versions"] and
        provenance.get("include_percentage") is False and provenance.get("is_close") is True and
        provenance.get("internal_timeout") is False and provenance.get("timeout_owner") == "external_spawn_process" and
        provenance.get("normalization_order") == "explicit_percentage_then_pinned_strip_string",
        "Math worker did not load the pinned source comparator and runtime")


def build_calibration_cache(records, tokenizer, tokenizer_identity, *, grader=None, expected_comparator=None):
    """Pure, fixed-cohort preparation; tests may supply a bounded fixture grader."""
    require(isinstance(records, list) and len(records) == 32 and
        dict(Counter(row.get("source_kind") for row in records)) == CALIBRATION_COUNTS and
        len({row.get("id") for row in records}) == len({row.get("group_id") for row in records}) == 32 and
        [row["id"] for row in records] == sorted(row["id"] for row in records), "Exact sorted 32-record calibration groups required")
    grade = source.grade_response if grader is None else grader
    rows, prechecks = [], []
    for record in records:
        source.validate_record(record)
        gold = source.validate_gold(record)
        require(gold.get("supported") is True, "Unsupported calibration gold; no refill: " + record["id"])
        prompt = source.serialize_prompt(record, tokenizer, tokenizer_identity, max_prompt_tokens=8192)
        mutated = source.serialize_prompt(gold_mutation(record), tokenizer, tokenizer_identity, max_prompt_tokens=8192)
        require(all(prompt[key] == mutated[key] for key in ("messages", "rendered_prompt", "prompt_token_ids")),
            "Changing gold/solution changed the native generation request")
        self_grade = grade(record, "\\boxed{" + gold["gold_text"] + "}", timeout_seconds=5)
        require(self_grade.get("correct") is True and self_grade.get("rankable") is True,
            "Calibration gold self-comparison failed or timed out: " + record["id"])
        if record["source_kind"] == "math" and expected_comparator is not None:
            validate_comparator_result(self_grade, expected_comparator)
        prechecks.append({"record_id": record["id"], "gold": gold, "self_comparison": self_grade})
        rows.append({"record": deepcopy(record), "prompt": prompt, "request_seed": source.sample_seed(record["id"], 20261021),
            "sample_indices": list(range(8))})
    return rows, prechecks


def _qa(path, root, output, compute, evidence, plan_path):
    require(Path(path).resolve() == output / "unit_qa.json", "Use the current exclusive CPU QA receipt")
    qa = evidence.read(Path(path))
    hashes, inventory = qa.get("code_sha256", {}), qa.get("source_inventory", {})
    require(qa.get("schema") == QA_SCHEMA and qa.get("success") is True and type(qa.get("tests_run")) is int and
        qa["tests_run"] > 341 and qa.get("gpu_disabled") is True and qa.get("job_id") == compute["job_id"] and
        qa.get("node") == compute["node"] and qa.get("plan_sha256") == evidence.check(plan_path) and
        all(type(qa.get(name)) is int and qa[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        isinstance(hashes, dict) and isinstance(inventory, dict) and set(hashes) == set(inventory) and
        REQUIRED_CODE <= set(hashes), "Require new successful same-allocation source/pilot-driver fixture QA")
    require(qa.get("compute_proof", {}).get("job_id") == compute["job_id"] and
        qa["compute_proof"].get("gpu_requested_or_allocated") is False, "Actual CPU allocation proof missing")
    for name, checksum in hashes.items():
        path = inside(root, name)
        require(path.stat().st_size <= 2 << 20 and inventory[name] == {"sha256": checksum, "bytes": path.stat().st_size},
            "Source QA inventory differs")
        evidence.check(path, checksum)
    return qa


def prepare(root, plan_path, output, qa_receipt):
    compute, begun = require_cpu_allocation(), time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = inside(root, str(plan_path))
    require(output.parent == root / "extension/runs" and output.name == "source-readiness-" + compute["job_id"] and
        output.is_dir() and {path.name for path in output.iterdir()} == {"unit_qa.json"}, "Exclusive current CPU output required")
    evidence = Evidence()
    plan = validate_plan(evidence.read(plan_path))
    deadline_guard(180000)
    qa = _qa(Path(qa_receipt), root, output, compute, evidence, plan_path)
    for package, version in plan["runtime_versions"].items():
        require(importlib.metadata.version(package) == version, "Unpinned source/pilot runtime: " + package)
    for spec in list(plan["prior_cpu"].values()) + list(plan["grader_source"].values()):
        native._file_pin(spec)
        evidence.check(inside(root, spec["path"]), spec["sha256"])
    prior = evidence.read(inside(root, plan["prior_cpu"]["complete.json"]["path"]))
    require(prior.get("complete") is True and prior.get("cpu_only") is True and prior.get("job_id") == "1224497" and
        prior.get("input_snapshot_unchanged") is True, "Completed all-nine exclusion and native-cache stage required")
    seal, _, _, cohort_proof = prepared_source.load_development(root, plan["development_seal"], evidence)
    seal_directory = inside(root, plan["development_seal"]["path"]).parent
    records = prepared_source._jsonl(seal_directory / "calibration.jsonl", evidence)
    base = prepared_source._validate_base(root, plan["base"], evidence)
    tokenizer = prepared_source._load_tokenizer(base)
    tokenizer_identity = {"repository": native.REPOSITORY, "revision": native.REVISION, "files": native.TOKENIZER_FILES,
        "chat_template_sha256": prepared_source.text_sha256(tokenizer.chat_template), "eos_token_id": tokenizer.eos_token_id}
    require(tokenizer.eos_token_id == 151645, "Original native EOS differs")
    expected_comparator = {"loaded_modules": {"utils." + Path(name).stem: str(inside(root, spec["path"]))
        for name, spec in plan["grader_source"].items()}, "runtime_versions": {name: plan["runtime_versions"][name]
        for name in ("sympy", "antlr4-python3-runtime", "latex2sympy2", "regex", "word2number")}}
    rows, prechecks = build_calibration_cache(records, tokenizer, tokenizer_identity, expected_comparator=expected_comparator)
    cache = {"schema": "extension-source-calibration-prompt-cache-v1", "complete": True, "plan_sha256": evidence.check(plan_path),
        "records": rows, "model": {"repository": native.REPOSITORY, "revision": native.REVISION, **plan["base"]},
        "tokenizer_identity": tokenizer_identity, "source_generation_profile": plan["source_generation_profile"],
        "development_seal": plan["development_seal"], "code_sha256": qa["code_sha256"],
        "calibration_ids_sha256": seal["splits"]["calibration"]["ids_sha256"], "raw_records_private": True,
        "generation_performed": False, "training_performed": False}
    evidence.unchanged()
    write_new_json(output / "calibration_gold_prechecks.json", prechecks)
    write_new_json(output / "calibration_prompt_cache.json", cache)
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    counts = [len(row["prompt"]["prompt_token_ids"]) for row in rows]
    result = {"schema": SCHEMA, "complete": True, "cpu_only": True, "job_id": compute["job_id"], "gpu_seconds": 0,
        "plan_sha256": evidence.check(plan_path), "native_qa_sha256": digest(output / "unit_qa.json"),
        "code_sha256": qa["code_sha256"], "source_inventory": qa["source_inventory"], "tests_run": qa["tests_run"],
        "record_count": len(rows), "domain_counts": CALIBRATION_COUNTS, "group_count": 32,
        "maximum_prompt_tokens": max(counts), "minimum_prompt_tokens": min(counts), "prompt_token_sum": sum(counts),
        "native_gold_mutation_invariance_passed": True, "gold_self_comparisons_passed": len(prechecks),
        "calibration_only_gold_validation": True, "confirmation_outputs_opened": False,
        "input_files": evidence.files, "input_snapshot_unchanged": True, "source_cohorts": cohort_proof,
        "output_sha256": {name: digest(output / name) for name in
            ("calibration_prompt_cache.json", "calibration_gold_prechecks.json", "unit_qa.json")},
        "compute_start": compute, "compute_end": ending, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - begun, "automatic_downstream": False, "generation": False,
        "training_pilot": False, "real_model_ga8_admitted": False, "raw_cache_private": True}
    write_new_json(output / "complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="extension/source_readiness_plan.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.plan, args.output, args.qa_receipt)
    print({key: result[key] for key in ("complete", "record_count", "tests_run", "maximum_prompt_tokens", "gpu_seconds")}, flush=True)


if __name__ == "__main__":
    main()
