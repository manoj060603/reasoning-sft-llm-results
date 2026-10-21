"""UNADMITTED, disposable long-after-Adam memory diagnostic.

This module is outside the earlier67-source freeze and proves nothing until a
new exact plan, private-input hashes, native CPU QA and current-job admission are
bound and the real compute-only diagnostic completes. It never selects rows or
submits a job. The four existing sealed witnesses supply exactly17 presentations
per objective:8short, long,long,6short, then1short tail. Repeated source IDs are
explicit diagnostic repeats; the frozen pilot unique-ID rule is not modified.

Before the SECOND long forward, all student Adam moments and an accumulated
long-example gradient must already exist. Each of four modes starts in a fresh
process from the original FP32 student and independent FP32 reference. No retry,
resume, change of context/precision, scientific promotion or downstream action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, value_digest, write_new_json)
from extension import native_model_admission as native
from extension import pilot_driver as pilot
from extension.labeled_feasibility import CRITERION

SCHEMA = "extension-joint-memory-admission-plan-v1"
COMPLETE_SCHEMA = "extension-joint-memory-admission-complete-v1"
CASE_SCHEMA = "extension-pilot-long-after-adam-admission-v1"
MODES = native.MODES
PRESENTATION_ROLES = ("short0", "short1", "short2", "short0", "short1", "short2", "short0", "short1",
    "long", "long", "short0", "short1", "short2", "short0", "short1", "short2", "short0")
REQUIRED_CODE = pilot.REQUIRED_CODE | frozenset({"extension/joint_memory_admission.py", "tests/test_joint_memory_admission.py"})
SETTINGS = {"accumulation": 8, "microbatch": 1, "student_dtype": "float32", "reference_dtype": "float32",
    "bf16": True, "attention": "sdpa", "gradient_checkpointing": True, "use_reentrant": False,
    "checkpoint_chunks": True, "chunk_tokens": 128, "optimizer": "adamw_torch", "learning_rate": 2e-6,
    "weight_decay": .01, "clip_norm": 1., "seed": 20261021, "beta": .05,
    "scheduler": "cosine_prefix", "warmup_steps": 57, "total_steps": 1875, "resume": False}


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == SCHEMA and plan.get("diagnostic_only") is True and
        plan.get("automatic_downstream") is False and plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00",
        "A new separately frozen compute-only diagnostic plan is required")
    require(plan.get("case_order") == list(MODES) and plan.get("presentation_roles") == list(PRESENTATION_ROLES) and
        plan.get("diagnostic_repeat_ids") is True and plan.get("optimizer_updates") == 3,
        "The fixed17 presentations/8+8+1/4fresh modes cannot change")
    settings = plan.get("settings", {})
    require(type(settings) is dict and set(settings) == set(SETTINGS) and
        all(type(settings[key]) is type(expected) and settings[key] == expected for key, expected in SETTINGS.items()),
        "Joint diagnostic settings differ from the prospective fixed path")
    inputs = plan.get("inputs", {})
    require(type(inputs) is dict and set(inputs) == {"native_plan", "prepared"}, "Pin the existing native plan and4-witness cache")
    for pin in inputs.values():
        native._file_pin(pin)
    resources, limits = plan.get("resources", {}), plan.get("tolerances", {})
    require(resources.get("gpu_type") == "B200" and resources.get("gpus") == 1 and
        all(type(resources.get(key)) is int and resources[key] >= lower for key, lower in
            (("wall_time_seconds", 1), ("host_cpus", 8), ("host_ram_gib", 128), ("scratch_free_gib", 160), ("reserve_seconds", 172800))) and
        resources["wall_time_seconds"] <= 7200, "One B200, bounded2h, host/scratch/deadline reserve are required")
    require(type(limits) is dict and set(limits) == {"logit_max_abs", "ce_abs", "kl_abs", "reload_logit_max_abs",
        "headroom_min_bytes", "headroom_fraction"}, "Explicit unchanged numerical/memory limits required")
    for key, upper in (("logit_max_abs", .001), ("ce_abs", 1e-5), ("kl_abs", 1e-5), ("reload_logit_max_abs", .001)):
        require(type(limits.get(key)) in (int, float) and math.isfinite(limits[key]) and 0 < limits[key] <= upper,
            "Do not weaken the admitted numerical limits")
    require(type(limits["headroom_min_bytes"]) is int and limits["headroom_min_bytes"] >= 16 << 30 and
        type(limits["headroom_fraction"]) in (int, float) and math.isfinite(limits["headroom_fraction"]) and
        .1 <= limits["headroom_fraction"] < 1, "Require at least16GiB/10percent headroom")
    return plan


def presentation_rows(prepared):
    rows = prepared.get("rows", {})
    require(type(rows.get("short")) is list and len(rows["short"]) == 3 and type(rows.get("long")) is dict,
        "Use the same existing sealed3short+1long witnesses")
    lookup = {"short" + str(index): row for index, row in enumerate(rows["short"])}
    lookup["long"] = rows["long"]
    require(len({row["id"] for row in lookup.values()}) == 4, "The4 sealed witness source IDs must remain distinct")
    require(all(len(row["input_ids"]) <= 512 for row in rows["short"]) and
        512 < len(rows["long"]["input_ids"]) <= 32768, "Actual short/long context envelope differs")
    return [lookup[role] for role in PRESENTATION_ROLES]


class DiagnosticLedger(pilot.ConsumptionLedger):
    """Only the fixed17-position disposable repeat schedule, never pilot data."""
    def __init__(self, rows, accumulation=8, *, event_sink=None):
        require(type(rows) is list and len(rows) == 17 and accumulation == 8, "Only17disposable presentations/GA8")
        representatives = {"short0": rows[0], "short1": rows[1], "short2": rows[2], "long": rows[8]}
        require(len({row["id"] for row in representatives.values()}) == 4 and
            all(value_digest(row) == value_digest(representatives[role])
                for row, role in zip(rows, PRESENTATION_ROLES)), "Diagnostic repeats must preserve the entire original/native example")
        for row in representatives.values():
            proof = row.get("provenance", {})
            require(type(row.get("original_row")) is dict and
                proof.get("row_sha256") == value_digest(row["original_row"]) and
                proof.get("full_token_ids_sha256") == value_digest(row["input_ids"]) and
                proof.get("labels_sha256") == value_digest(row["labels"]),
                "A repeated pair's source/tokens/labels changed even if all repeated references share the mutation")
        super().__init__(rows, accumulation, event_sink=event_sink)

    def finalize(self):
        require(self.pending is None and self.open_update is None and len(self.presentations) == 17 and
            len(self.updates) == 3 and [value["microbatches"] for value in self.updates] == [8, 8, 1],
            "Diagnostic failed to consume all17positions/three updates")
        ids = [value["id"] for value in self.presentations]
        require(ids == [row["id"] for row in self.rows] and
            [value["row_index"] for value in self.presentations] == list(range(17)), "Diagnostic position order differs")
        return {"complete": True, "presentation_ids": ids, "presentation_indices": list(range(17)),
            "presentations": self.presentations, "updates": self.updates, "examples": 17, "optimizer_updates": 3,
            "ordered_ids_sha256": value_digest(ids), "diagnostic_repeated_ids": True, "unique_source_examples": 4,
            "source_role_sequence": list(PRESENTATION_ROLES),
            **{key: sum(row[key] for row in self.presentations) for key in ("total_tokens", "active_tokens", "eos_tokens")}}


def make_trainer_class(trainer_base=None):
    return pilot.make_pilot_trainer_class(trainer_base)


def inspect_pre_forward(model, optimizer, *, completed_updates, backward_count, tokens,
                        row_index, source_role, expected_parameters=434):
    """Inspect real tensors BEFORE a forward, not post-hoc inferred state."""
    import torch
    from extension.preserving_trainer import check_optimizer
    for value in (completed_updates, backward_count, tokens, row_index):
        require(type(value) is int and value >= 0, "Integral actual presentation/update counters required")
    parameters = list(model.parameters())
    require(len(parameters) == expected_parameters and all(value.dtype == torch.float32 for value in parameters),
        "Full FP32 student tensor count differs")
    actual = check_optimizer(model, optimizer)
    moments = sum(all(name in actual.state.get(parameter, {}) for name in ("exp_avg", "exp_avg_sq")) for parameter in parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in gradients),
        "Resident accumulated gradients must already be finite FP32")
    return {"row_index": row_index, "source_role": source_role, "tokens": tokens,
        "optimizer_updates_before_forward": completed_updates,
        "student_parameters_with_adam_moments": moments,
        "resident_gradient_parameters_before_forward": len(gradients),
        "successful_backwards_in_current_update": backward_count,
        "forward_complete": False, "backward_complete": False, "optimizer_update_after_backward_complete": False}


def validate_observations(observations, prepared, expected_parameters=434):
    rows = presentation_rows(prepared)
    require(type(observations) is list and len(observations) == 17, "Every actual pre-forward state must be recorded")
    for index, (value, row, role) in enumerate(zip(observations, rows, PRESENTATION_ROLES)):
        require(type(value) is dict and all(type(value.get(key)) is int for key in
                ("row_index", "tokens", "optimizer_updates_before_forward", "successful_backwards_in_current_update",
                 "student_parameters_with_adam_moments", "resident_gradient_parameters_before_forward")) and
            value.get("row_index") == index and value.get("source_role") == role and
            value.get("tokens") == len(row["input_ids"]) and
            value.get("optimizer_updates_before_forward") == index // 8 and
            value.get("successful_backwards_in_current_update") == index % 8 and
            value.get("student_parameters_with_adam_moments") == (0 if index < 8 else expected_parameters) and
            value.get("resident_gradient_parameters_before_forward") == (0 if index % 8 == 0 else expected_parameters) and
            all(value.get(flag) is True for flag in ("forward_complete", "backward_complete", "optimizer_update_after_backward_complete")),
            "Actual forward/backward/update order is incomplete")
    first, second = observations[8], observations[9]
    require(first.get("student_parameters_with_adam_moments") == expected_parameters and
        first.get("resident_gradient_parameters_before_forward") == 0 and
        second.get("student_parameters_with_adam_moments") == expected_parameters and
        second.get("resident_gradient_parameters_before_forward") == expected_parameters and
        second["optimizer_updates_before_forward"] == 1 and second["successful_backwards_in_current_update"] == 1,
        "The second longest forward lacks already allocated Adam and one full accumulated gradient set")
    return {"scope": "long_after_adam_accumulation", "maximum_complete_tokens": max(len(row["input_ids"]) for row in rows),
        "long_context_joint_peak_proven": True, "observations": observations}


def _pin(root, evidence, value):
    native._file_pin(value)
    return evidence.read(native._project_path(root, value["path"]), value["sha256"])


def load_inputs(root, plan_path, qa_receipt, admission_receipt):
    """Reuse the existing4sealed witnesses; new QA/job bindings are mandatory."""
    root, evidence = Path(root).resolve(), Evidence()
    paths = {key: native._project_path(root, value) for key, value in
        (("plan", plan_path), ("qa", qa_receipt), ("admission", admission_receipt))}
    plan = validate_plan(evidence.read(paths["plan"]))
    plan_sha = evidence.check(paths["plan"])
    old_plan = native.validate_plan(_pin(root, evidence, plan["inputs"]["native_plan"]))
    prepared = native.validate_prepared(_pin(root, evidence, plan["inputs"]["prepared"]), old_plan,
        plan["inputs"]["native_plan"]["sha256"])
    presentation_rows(prepared)
    qa, admission = evidence.read(paths["qa"]), evidence.read(paths["admission"])
    codes = qa.get("code_sha256", {})
    require(qa.get("schema") == "extension-joint-memory-admission-native-qa-v1" and qa.get("success") is True and
        qa.get("plan_sha256") == plan_sha and qa.get("prepared_sha256") == plan["inputs"]["prepared"]["sha256"] and
        type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and qa.get("gpu_disabled") is True and
        all(type(qa.get(key)) is int and qa[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        type(codes) is dict and REQUIRED_CODE <= set(codes), "New joint-diagnostic CPU QA is required; the old67freeze is insufficient")
    for name, checksum in codes.items():
        path = native._project_path(root, name)
        require(path.suffix in (".py", ".json", ".slurm") and path.stat().st_size <= 2 << 20, "Bounded actual source inventory required")
        evidence.check(path, native._sha(checksum))
    require(admission.get("schema") == "extension-joint-memory-admission-job-v1" and admission.get("status") == "admitted" and
        admission.get("job_id") == os.environ.get("SLURM_JOB_ID") and admission.get("plan_sha256") == plan_sha and
        admission.get("prepared_sha256") == plan["inputs"]["prepared"]["sha256"] and
        admission.get("qa_sha256") == evidence.check(paths["qa"]) and admission.get("code_sha256") == codes and
        admission.get("automatic_downstream") is False, "A new exact current-job admission is required")
    base = Path(prepared["base"]["path"]).resolve()
    for name, checksum in native.ORIGINAL_FILES.items():
        evidence.check(base / name, checksum)
    identity = _pin(root, evidence, prepared["base"]["identity_receipt"])
    require(identity.get("path") == str(base) and identity.get("model_type") == "qwen2" and
        {name: record.get("sha256") for name, record in identity.get("files", {}).items()} == native.ORIGINAL_FILES and
        identity.get("identity_sha256") == value_digest({key: value for key, value in identity.items() if key != "identity_sha256"}),
        "Actual pinned original identity receipt differs")
    audit, values = prepared["source_audit"], {}
    for name, pin in audit.items():
        if name in ("pool", "features"):
            evidence.check(native._project_path(root, pin["path"]), pin["sha256"])
        else:
            values[name] = _pin(root, evidence, pin)
    seal, reserved, benchmark, blocked, union, preparation = (values[name] for name in
        ("development_seal", "reserved_pool_ids", "benchmark_audit", "benchmark_pool_ids", "exclusion_receipt", "preparation"))
    require(prepared["source_identity"].get("pool_sha256") == audit["pool"]["sha256"] and
        prepared["source_identity"].get("features_sha256") == audit["features"]["sha256"], "Source identity differs from actual pool/features")
    require(seal.get("schema") == "extension-development-seal-v2" and seal.get("complete") is True and
        seal.get("input_stats_unchanged_at_closeout") is True and
        seal.get("output_sha256", {}).get("reserved_training_pool_ids.json") == audit["reserved_pool_ids"]["sha256"] and
        seal.get("reservation", {}).get("reserved_training_pool_ids_sha256") == value_digest(reserved), "Repaired source reservation is unbound")
    require(benchmark.get("schema") == "extension-final-benchmark-pool-exclusions-v1" and benchmark.get("complete") is True and
        benchmark.get("pool_sha256") == audit["pool"]["sha256"] and benchmark.get("criterion") == CRITERION and
        benchmark.get("scope") == "all_nine_final_benchmarks" and
        set(benchmark.get("benchmark_question_sha256", {})) == {"aime", "aime25", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"} and
        benchmark.get("excluded_pool_ids_sha256") == value_digest(blocked) and
        benchmark.get("excluded_pool_ids_file_sha256") == audit["benchmark_pool_ids"]["sha256"] and
        benchmark.get("answers_or_outcomes_used") is False, "All-nine question-only exclusions are unbound")
    for name, checksum in benchmark["benchmark_question_sha256"].items():
        evidence.check(root / "deadline/artifacts/benchmarks" / name / "test.jsonl", checksum)
    require(union.get("schema") == "preserving-pool-exclusions-v1" and
        union.get("id_namespace") == "canonical_pool_prompt_sha256_v1" and
        set(union.get("components", {})) == {"reserved_source_evaluation", "final_benchmark_questions"}, "Exclusion union is incomplete")
    excluded = set()
    for purpose, ids, receipt_name, scope in (("reserved_source_evaluation", reserved, "development_seal", "all_reserved_source_splits"),
            ("final_benchmark_questions", blocked, "benchmark_audit", "all_nine_final_benchmarks")):
        require(type(ids) is list and ids == sorted(set(ids)) and all(native._sha(value) for value in ids), "Malformed canonical exclusions")
        component = union["components"][purpose]
        require(component.get("pool_ids") == ids and component.get("scope") == scope and
            component.get("audit_receipt_sha256") == audit[receipt_name]["sha256"] and
            component.get("criterion_sha256") == value_digest(CRITERION) and native._sha(component.get("question_inventory_sha256")),
            "Union component differs from the independently pinned audit")
        excluded.update(ids)
    require(not excluded.intersection(row["id"] for row in presentation_rows(prepared)), "A diagnostic witness is reserved or benchmark-overlapping")
    require(preparation.get("schema") == "extension-native-smoke-preparation-v1" and preparation.get("complete") is True and
        preparation.get("cpu_only") is True and preparation.get("gpu_seconds") == 0 and
        preparation.get("plan_sha256") == plan["inputs"]["native_plan"]["sha256"] and
        preparation.get("rows_sha256") == value_digest(prepared["rows"]) and
        preparation.get("exclusion_receipt_sha256") == audit["exclusion_receipt"]["sha256"] and
        preparation.get("pool_sha256") == audit["pool"]["sha256"] and preparation.get("features_sha256") == audit["features"]["sha256"] and
        preparation.get("source_identity") == prepared["source_identity"] and preparation.get("no_model_or_benchmark_outcomes") is True and
        preparation.get("input_snapshot_unchanged") is True, "Original CPU preparation receipt is incomplete")
    unit = _pin(root, evidence, {key: preparation.get("native_qa", {}).get(key) for key in ("path", "sha256")})
    require(unit.get("schema") == "extension-native-smoke-preparation-unit-qa-v1" and unit.get("success") is True and
        unit.get("native_plan_sha256") == plan["inputs"]["native_plan"]["sha256"] and unit.get("gpu_disabled") is True and
        type(unit.get("tests_run")) is int and unit["tests_run"] > 0 and
        all(type(unit.get(key)) is int and unit[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        unit.get("code_sha256") == preparation.get("code_sha256") and
        {"extension/smoke_preparation.py", "tests/test_smoke_preparation.py", "extension/preserving_data.py"} <= set(unit.get("code_sha256", {})),
        "Original compute-only CPU preparation unit proof is missing")
    require(unit.get("compute_proof", {}).get("gpu_requested_or_allocated") is False and
        unit["compute_proof"].get("job_id") == unit.get("job_id") and unit["compute_proof"].get("node") == unit.get("node"),
        "Original preparation QA is not bound to its native CPU allocation")
    for name, checksum in unit["code_sha256"].items():
        evidence.check(native._project_path(root, name), checksum)
    original_inputs = {(base / name).resolve() for name in native.ORIGINAL_FILES}
    require(type(preparation.get("input_files")) is dict and preparation["input_files"], "Missing actual preparation input snapshot")
    for name, record in preparation["input_files"].items():
        path = Path(name).resolve()
        require(Path(name).is_absolute() and (path.is_relative_to(root) or path in original_inputs) and
            type(record) is dict and type(record.get("bytes")) is int and record["bytes"] == path.stat().st_size,
            "Preparation input escaped exact base/source paths or changed size")
        evidence.check(path, record.get("sha256"))
    longest = preparation.get("longest", {})
    require(longest.get("id") == prepared["rows"]["long"]["id"] and
        longest.get("total_tokens") == longest.get("max_eligible_total_tokens") == len(prepared["rows"]["long"]["input_ids"]) and
        longest.get("after_reserved_exclusion") is True and longest.get("after_benchmark_exclusion") is True,
        "Exact existing longest complete eligible witness is unbound")
    evidence.unchanged()
    binding = {"plan_sha256": plan_sha, "prepared_sha256": plan["inputs"]["prepared"]["sha256"],
        "qa_sha256": evidence.check(paths["qa"]), "admission_sha256": evidence.check(paths["admission"]), "code_sha256": codes}
    return {"root": root, "paths": paths, "plan": plan, "prepared": prepared, "union": union,
        "admission": admission, "evidence": evidence, **binding, "binding_sha256": value_digest(binding)}


def validate_case_receipt(receipt, context, mode, output, *, verify_outputs=True, output_evidence=None):
    require(type(receipt) is dict and receipt.get("schema") == CASE_SCHEMA and receipt.get("mode") == mode and
        receipt.get("binding_sha256") == context["binding_sha256"] and
        receipt.get("plan_sha256") == context["plan_sha256"] and receipt.get("prepared_sha256") == context["prepared_sha256"] and
        receipt.get("code_sha256") == context["code_sha256"] and
        receipt.get("diagnostic_repeated_ids") is True and receipt.get("completed_optimizer_updates") == 3 and
        receipt.get("presentation_ids") == [row["id"] for row in presentation_rows(context["prepared"])],
        "Joint case is incomplete or differs from the exact diagnostic order/inputs")
    shape = validate_observations(receipt.get("shape_evidence", {}).get("observations"), context["prepared"])
    require(value_digest(shape) == value_digest(receipt["shape_evidence"]), "Joint shape summary differs from actual observation replay")
    settings = context["plan"]["settings"]
    pilot_plan = {"objective": mode, "beta": 0. if mode == "sft" else settings["beta"],
        "chunk_tokens": settings["chunk_tokens"], "checkpoint_chunks": True, "parameter_dtype": "float32",
        "reference_dtype": "float32", "compute_dtype": "bfloat16", "attention": "sdpa"}
    pilot.validate_joint_shape_admission(receipt, plan=pilot_plan, code_sha256=context["code_sha256"],
        maximum_tokens=len(context["prepared"]["rows"]["long"]["input_ids"]))
    require(set(receipt.get("checks", {})) == native.REQUIRED_CHECKS and
        all(value is True for value in receipt["checks"].values()) and
        receipt.get("consumption", {}).get("optimizer_updates") == 3 and
        receipt["consumption"].get("unique_source_examples") == 4 and receipt["consumption"].get("examples") == 17,
        "All numerical/EOS/reference/checkpoint/consumption checks are required")
    actual = receipt["consumption"]
    require(type(actual.get("updates")) is list and len(actual["updates"]) == 3 and
        type(receipt.get("lr_used")) is list and type(receipt.get("lr_after_scheduler")) is list and
        len(receipt["lr_used"]) == len(receipt["lr_after_scheduler"]) == 3,
        "All three actual LR/update observations are required")
    replay = DiagnosticLedger(presentation_rows(context["prepared"]))
    for update, indices in enumerate(replay.groups, 1):
        for index in indices:
            replay.begin(index)
            replay.backward_complete(index)
        entry, before, after = actual["updates"][update - 1], receipt["lr_used"][update - 1], receipt["lr_after_scheduler"][update - 1]
        require(type(before) is list and type(after) is list and before and len(before) == len(after) and
            entry.get("learning_rates_before") == entry.get("learning_rates_after") == before and
            all(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, settings["learning_rate"] * pilot.cosine_factor(update - 1), rel_tol=1e-12, abs_tol=1e-18) for value in before) and
            all(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, settings["learning_rate"] * pilot.cosine_factor(update), rel_tol=1e-12, abs_tol=1e-18) for value in after),
            "Actual optimizer/scheduler LR prefix differs from the independent oracle")
        replay.before_update(update, before)
        replay.after_update(update, before)
    require(value_digest(actual) == value_digest(replay.finalize()), "Diagnostic consumption differs from exact native tokens/EOS/order")
    memory = receipt.get("memory", {})
    peak = max(memory.get("peak_allocated_bytes", -1), memory.get("peak_reserved_bytes", -1))
    expected_headroom = native.headroom_report(peak, memory.get("total_bytes", 0),
        context["plan"]["tolerances"]["headroom_min_bytes"], context["plan"]["tolerances"]["headroom_fraction"])
    require(expected_headroom["passed"] is True and memory.get("headroom") == expected_headroom,
        "The measured joint headroom does not satisfy the prospective threshold")
    files = receipt.get("output_files", {})
    require(type(files) is dict and files, "No exact student checkpoint inventory")
    if verify_outputs:
        evidence = output_evidence or Evidence()
        for name, checksum in files.items():
            require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe diagnostic saved output")
            evidence.check(output / name, native._sha(checksum))
        expected = {str(path.relative_to(output)) for path in (output / "checkpoint").rglob("*") if path.is_file()}
        expected.update(("consumption.json", "shape_observations.json", "presentations.jsonl"))
        require(set(files) == expected, "Exact diagnostic output file inventory differs")
    return receipt


def run(root, plan_path, output, qa_receipt, admission_receipt, *, runner=None):
    allocation = native.require_gpu_allocation()
    context = load_inputs(root, plan_path, qa_receipt, admission_receipt)
    require(context["admission"]["job_id"] == allocation["job_id"], "Another job cannot consume this admission")
    resources = context["plan"]["resources"]
    deadline_guard(resources["wall_time_seconds"] + resources["reserve_seconds"])
    proposed = Path(output)
    require(not proposed.is_symlink(), "A new output cannot alias an earlier attempt")
    output = proposed.resolve()
    require(output.is_relative_to(context["root"] / "extension/runs"), "Use a unique extension/runs output")
    output = native.reserve_output(output)
    write_new_json(output / "started.json", {"schema": COMPLETE_SCHEMA, "complete": False,
        "binding_sha256": context["binding_sha256"], "case_order": list(MODES), "allocation": allocation})
    begun, receipts, execute, observed_files = time.monotonic(), {}, runner or subprocess.run, Evidence()
    try:
        require(shutil.disk_usage(output).free >= resources["scratch_free_gib"] * (1 << 30), "Insufficient declared free scratch")
        for mode in MODES:
            remaining = math.floor(resources["wall_time_seconds"] - (time.monotonic() - begun))
            require(remaining > 0, "Bounded diagnostic time exhausted")
            deadline_guard(remaining + resources["reserve_seconds"])
            command = [sys.executable, "-m", "extension.joint_memory_admission", "--root", str(context["root"]),
                "--plan", str(context["paths"]["plan"]), "--output", str(output), "--qa-receipt", str(context["paths"]["qa"]),
                "--admission-receipt", str(context["paths"]["admission"]), "--mode", mode]
            result = execute(command, cwd=str(context["root"]), check=False, timeout=remaining)
            require(result.returncode == 0, "Joint diagnostic failed without retry: " + mode)
            directory = output / "cases" / mode
            receipt = read_json(directory / "complete.json")
            validate_case_receipt(receipt, context, mode, directory, output_evidence=observed_files)
            pin = {"path": str(directory / "complete.json"), "sha256": digest(directory / "complete.json")}
            observed_files.check(directory / "complete.json", pin["sha256"])
            receipts[mode] = pin
            observed_files.unchanged()
            context["evidence"].unchanged()
        ending = native.require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Allocation changed")
        context["evidence"].unchanged()
        observed_files.unchanged()
        result = {"schema": COMPLETE_SCHEMA, "complete": True, "binding_sha256": context["binding_sha256"],
            "plan_sha256": context["plan_sha256"], "prepared_sha256": context["prepared_sha256"],
            "code_sha256": context["code_sha256"], "allocation": allocation, "allocation_end": ending,
            "case_receipts": receipts, "input_files": context["evidence"].files,
            "input_snapshot_unchanged": True, "output_snapshot_unchanged": True, "training_pilot": False,
            "diagnostic_repeat_ids": True, "automatic_downstream": False, "resume_supported": False,
            "elapsed_seconds": time.monotonic() - begun, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_new_json(output / "complete.json", result)
        return result
    except BaseException as error:
        write_new_json(output / "failure.json", {"schema": "extension-joint-memory-admission-failure-v1", "complete": False,
            "binding_sha256": context["binding_sha256"], "completed_cases": receipts, "exception": type(error).__name__,
            "message": str(error)[:4000], "retry": False, "resume_supported": False, "automatic_downstream": False})
        raise


def run_case(context, mode, output, allocation):
    """Fresh real-model work; never callable successfully from a login node."""
    require(mode in MODES, "Unknown joint diagnostic mode")
    actual = native.require_gpu_allocation()
    require(actual["job_id"] == allocation["job_id"] == context["admission"]["job_id"] and actual["node"] == allocation["node"],
        "Case is outside its admitted running compute allocation")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed
    from transformers.loss.loss_utils import LOSS_MAPPING
    from extension.preserving_data import verify_serialized_examples
    from extension.preserving_trainer import check_optimizer, disjoint_models
    for name, expected in native.VERSIONS.items():
        require(importlib.metadata.version(name) == expected, "Unvalidated native version: " + name)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported() and
        "B200" in torch.cuda.get_device_name(0), "One assigned BF16 B200 is required")
    plan, prepared = context["plan"], context["prepared"]
    settings, limits, serial = plan["settings"], plan["tolerances"], prepared["serialization"]
    native._host_resources(allocation, plan["resources"])
    initial_loss = LOSS_MAPPING["ForCausalLM"]
    require(initial_loss.__module__ == "transformers.loss.loss_utils", "Global CE was already patched")
    base, begun, memory = Path(prepared["base"]["path"]), time.monotonic(), native.MemoryTrace(torch)
    output = native.reserve_output(output)
    write_new_json(output / "started.json", {"schema": CASE_SCHEMA, "complete": False, "mode": mode,
        "binding_sha256": context["binding_sha256"], "allocation": allocation})
    ledger, journal, observations = None, None, []
    try:
        set_seed(settings["seed"])
        rng = {"seed": settings["seed"], "before_model_initialization": pilot.rng_fingerprint()}
        gpu_identity = native.assigned_gpu_identity(torch)
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
        replay = {}
        for cohort, values, cutoff in (("short", prepared["rows"]["short"], 512), ("long", [prepared["rows"]["long"]], 32768)):
            replay[cohort] = verify_serialized_examples(values, tokenizer, source_identity=prepared["source_identity"],
                tokenizer_identity=serial["tokenizer_identity"], exclusion_receipt_sha256=value_digest(context["union"]),
                cutoff=cutoff, context_role="diagnostic", eos_token_id=serial["eos_token_id"],
                terminal_suffix=tuple(serial["terminal_suffix"]), terminal_text=serial["terminal_text"], vocab_size=151936)

        def load_model(path):
            model, info = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32, attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False, output_loading_info=True)
            require(all(not info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "Partial model load")
            return model.to("cuda")

        with memory.phase("load_original_fp32_student_and_reference"):
            student, reference = load_model(base), load_model(base).eval().requires_grad_(False)
            student.config.use_cache = reference.config.use_cache = False
            require(student.config.model_type == "qwen2" and student.config.vocab_size == 151936 and
                student.config.max_position_embeddings == 32768 and student.config.attention_dropout == 0 and
                student.config.use_sliding_window is False, "Original architecture/context/dropout changed")
            disjoint_models(student, reference)
            original = native.tensor_inventory(reference)
            require(len(original) == 434 and native.tensor_inventory(student) == original, "Fresh original FP32 parameters differ")
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rows, metrics, lr_used, lr_after, gradient_norms = presentation_rows(prepared), [], [], [], [], []
        observed = {"student_logits_dtypes": set(), "reference_logits_dtypes": set(), "sdpa_operators": [], "sdpa_by_presentation": {}}
        journal = (output / "presentations.jsonl").open("x", encoding="utf-8")

        def event_sink(event):
            journal.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            if event["event"] in ("before_optimizer_update", "optimizer_update_complete"):
                os.fsync(journal.fileno())

        ledger = DiagnosticLedger(rows, event_sink=event_sink)

        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 17

            def __getitem__(self, index):
                return {**{key: rows[index][key] for key in ("input_ids", "labels", "attention_mask")}, "row_index": index}

        def collate(items):
            require(len(items) == 1, "Microbatch1 required")
            return {key: torch.tensor([value], dtype=torch.int64) for key, value in items[0].items()}

        def batch(row):
            return {key: torch.tensor([row[key]], dtype=torch.int64, device="cuda") for key in ("input_ids", "labels", "attention_mask")}

        def forward(row):
            return {key: value for key, value in batch(row).items() if key != "labels"}

        Base = make_trainer_class()

        class MeasuredTrainer(Base):
            def compute_loss(self, model, inputs, **kwargs):
                active = model.training and torch.is_grad_enabled()
                if active:
                    index = int(inputs["row_index"].item())
                    value = inspect_pre_forward(model, self.optimizer, completed_updates=len(ledger.updates),
                        backward_count=len(ledger.presentations) - 8 * len(ledger.updates), tokens=inputs["input_ids"].shape[1],
                        row_index=index, source_role=PRESENTATION_ROLES[index])
                    observations.append(value)
                    event_sink({"event": "actual_pre_forward_optimizer_and_gradient_state", **value})
                    require(value["student_parameters_with_adam_moments"] == (0 if index < 8 else 434) and
                        value["resident_gradient_parameters_before_forward"] == (0 if index % 8 == 0 else 434),
                        "Actual pre-forward Adam/accumulated-gradient occupancy differs; do not execute an invalid shape diagnostic")
                with memory.phase("external_objective_forward"):
                    result = super().compute_loss(model, inputs, **kwargs)
                if active:
                    observations[-1]["forward_complete"] = True
                return result

            def training_step(self, model, inputs, num_items_in_batch=None):
                index = int(inputs["row_index"].item())
                if index in (0, 9):
                    with memory.phase("actual_backward"), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False, profile_memory=False, with_stack=False) as profile:
                        value = super().training_step(model, inputs, num_items_in_batch)
                    operators = sorted({item.key for item in profile.key_averages() if "_scaled_dot_product_" in item.key and
                        any(name in item.key for name in ("flash", "efficient", "cudnn", "math"))})
                    require(operators, "Actual SDPA dispatch was not observed for the short or second-long presentation")
                    observed["sdpa_by_presentation"][str(index)] = operators
                    observed["sdpa_operators"] = sorted(set(observed["sdpa_operators"]) | set(operators))
                else:
                    with memory.phase("actual_backward"):
                        value = super().training_step(model, inputs, num_items_in_batch)
                observations[-1]["backward_complete"] = True
                metrics.append({"row_index": int(inputs["row_index"].item()), **self.last_metrics})
                return value

            def create_scheduler(self, num_training_steps, optimizer=None):
                require(num_training_steps == 3 and self.lr_scheduler is None, "Fresh three-update schedule required")
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, pilot.cosine_factor)
                return self.lr_scheduler

        args = TrainingArguments(output_dir=str(output / "trainer"), bf16=True, fp16=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=3, num_train_epochs=1,
            learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"], lr_scheduler_type="cosine", warmup_steps=57,
            max_grad_norm=0., optim="adamw_torch", gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False, label_smoothing_factor=0., eval_strategy="no", push_to_hub=False,
            save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True, dataloader_num_workers=0,
            dataloader_pin_memory=False, dataloader_drop_last=False, seed=settings["seed"], data_seed=settings["seed"])
        trainer = MeasuredTrainer(model=student, reference_model=reference, args=args, objective=mode,
            beta=0. if mode == "sft" else settings["beta"], chunk_tokens=128, checkpoint_chunks=True, clip_norm=1.,
            eos_token_id=serial["eos_token_id"], terminal_suffix=tuple(serial["terminal_suffix"]),
            train_dataset=Dataset(), data_collator=collate, consumption=ledger)
        short = prepared["rows"]["short"][0]

        class Observe(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                require(state.global_step == 0, "Original initialization at step0 required")
                self.previous_student, self.previous_reference = model.forward, reference.forward

                def wrap(previous, kind):
                    def measured(*items, **named):
                        output = previous(*items, **named)
                        logits = output["logits"] if isinstance(output, dict) else output.logits
                        observed[kind + "_logits_dtypes"].add(str(logits.dtype))
                        memory.record(kind + "_returned_logits")
                        return output
                    return measured
                model.forward, reference.forward = wrap(self.previous_student, "student"), wrap(self.previous_reference, "reference")
                with memory.phase("prepared_zero_update_parity"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.train()
                    p, q = model(**forward(short), use_cache=False).logits.float(), reference(**forward(short), use_cache=False).logits.float()
                    labels = batch(short)["labels"][:, 1:]
                    logp, logq = p[:, :-1].log_softmax(-1), q[:, :-1].log_softmax(-1)
                    ce = torch.nn.functional.cross_entropy(p[:, :-1].reshape(-1, p.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    qce = torch.nn.functional.cross_entropy(q[:, :-1].reshape(-1, q.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    kl = (logq.exp() * (logq - logp)).sum(-1)[labels != -100].mean()
                    observed["parity"] = native.check_parity({"max_abs_logits": float((p - q).abs().max()),
                        "ce_abs": abs(float(ce - qce)), "kl_abs": abs(float(kl))}, limits)
                    model.eval()
                    trainer.compute_loss(model, batch(short))
                    observed["independent_ce_abs"] = abs(float(ce) - trainer.last_metrics["ce"])
                    require(observed["independent_ce_abs"] <= limits["ce_abs"], "Independent shifted CE differs")
                    model.train()

            def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
                native.assert_finite_model(model, gradients=True)
                lrs = [float(group["lr"]) for group in optimizer.param_groups]
                require(all(math.isclose(value, settings["learning_rate"] * pilot.cosine_factor(state.global_step),
                    rel_tol=1e-12, abs_tol=1e-18) for value in lrs), "Used LR differs from the full1875 schedule prefix")
                lr_used.append(lrs)
                gradient_norms.append(trainer.last_gradient_norm)
                self.frame = memory.begin("adam_update_" + str(state.global_step + 1))

            def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
                check_optimizer(model, optimizer, reference)
                memory.end(self.frame)
                for index in ledger.groups[state.global_step]:
                    observations[index]["optimizer_update_after_backward_complete"] = True

            def on_step_end(self, args, state, control, optimizer=None, **kwargs):
                values = [float(group["lr"]) for group in optimizer.param_groups]
                require(all(math.isclose(value, settings["learning_rate"] * pilot.cosine_factor(state.global_step),
                    rel_tol=1e-12, abs_tol=1e-18) for value in values), "Scheduler boundary differs")
                lr_after.append(values)
                require(time.monotonic() - begun < plan["resources"]["wall_time_seconds"], "Case time bound exhausted")
                deadline_guard(plan["resources"]["reserve_seconds"])

            def on_train_end(self, args, state, control, model=None, **kwargs):
                model.forward, reference.forward = self.previous_student, self.previous_reference

        trainer.add_callback(Observe())
        rng["before_train"] = pilot.rng_fingerprint()
        with memory.phase("actual_seventeen_presentations_three_updates"):
            trainer.train()
        rng["after_train"] = pilot.rng_fingerprint()
        journal.close()
        consumption = ledger.finalize()
        require(trainer.state.global_step == 3 and trainer.training_forward_examples == 17, "Actual diagnostic exposure differs")
        shape = validate_observations(observations, prepared)
        with memory.phase("full_finite_weights_moments_and_reference_scan"):
            finite_count = native.assert_finite_model(student)
            optimizer = check_optimizer(student, trainer.optimizer, reference)
            require(optimizer.state and all(parameter in optimizer.state for parameter in student.parameters()), "Adam states missing")
            require(not reference.training and all(parameter.grad is None for parameter in reference.parameters()) and
                native.tensor_inventory(reference) == original, "Frozen original reference changed")
            inventory = native.tensor_inventory(student)
            require(any(inventory[name]["sha256"] != original[name]["sha256"] for name in inventory), "FP32 diagnostic updates changed no parameter")
        checkpoint = output / "checkpoint"
        with memory.phase("save_and_verify_student_checkpoint"):
            trainer.save_model(str(checkpoint))
            for name in native.TOKENIZER_FILES:
                shutil.copyfile(base / name, checkpoint / name)
            saved = native._checkpoint_inventory(checkpoint, inventory)
        with memory.phase("saved_reload_short_parity"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student.eval()
            expected = student(**forward(short), use_cache=False).logits.float().cpu()
            reference.to("cpu")
            reloaded = load_model(checkpoint).eval()
            actual = reloaded(**forward(short), use_cache=False).logits.float().cpu()
            reload_difference = float((expected - actual).abs().max())
            require(reload_difference <= limits["reload_logit_max_abs"], "Saved student reload differs")
            del reloaded, expected, actual
        require(LOSS_MAPPING["ForCausalLM"] is initial_loss and "torch.float32" in observed["student_logits_dtypes"],
            "Global loss or actual prepared external logits precision differs")
        memory_result = memory.result(limits)
        require(memory_result["headroom"]["passed"], "Joint actual memory headroom failed")
        context["evidence"].unchanged()
        ending = native.require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Case allocation changed")
        write_new_json(output / "consumption.json", consumption)
        write_new_json(output / "shape_observations.json", shape)
        files = {str(path.relative_to(output)): digest(path) for path in sorted(checkpoint.rglob("*")) if path.is_file()}
        files.update({name: digest(output / name) for name in ("consumption.json", "shape_observations.json", "presentations.jsonl")})
        receipt = {"schema": CASE_SCHEMA, "complete": True, "mode": mode, "binding_sha256": context["binding_sha256"],
            "plan_sha256": context["plan_sha256"], "prepared_sha256": context["prepared_sha256"], "code_sha256": context["code_sha256"],
            "base": {"repo_id": native.REPOSITORY, "revision": native.REVISION, "files": native.ORIGINAL_FILES},
            "versions": dict(native.VERSIONS), "gpu_identity": gpu_identity, "allocation": allocation, "allocation_end": ending,
            "runtime": {"python": sys.version, "torch_cuda": torch.version.cuda},
            "settings": {"objective": mode, "beta": 0. if mode == "sft" else .05, "chunk_tokens": 128,
                "checkpoint_chunks": True, "parameter_dtype": "float32", "reference_dtype": "float32", "compute_dtype": "bfloat16", "attention": "sdpa"},
            "accumulation": 8, "completed_optimizer_updates": 3, "diagnostic_repeated_ids": True,
            "presentation_ids": [row["id"] for row in rows], "shape_evidence": shape, "consumption": consumption,
            "source_native_serialization_verified": True, "global_loss_unchanged": True, "finite_weights": True,
            "finite_moments": True, "reference_unchanged": True, "memory": memory_result,
            "checks": {name: True for name in sorted(native.REQUIRED_CHECKS)}, "finite_parameter_count": finite_count,
            "reference_tensor_inventory_sha256": value_digest(original), "saved_student": saved,
            "parity": observed["parity"], "independent_ce_abs": observed["independent_ce_abs"], "reload_max_abs_logits": reload_difference,
            "student_logits_dtypes": sorted(observed["student_logits_dtypes"]), "reference_logits_dtypes": sorted(observed["reference_logits_dtypes"]),
            "actual_sdpa_operators": observed["sdpa_operators"], "actual_sdpa_by_presentation": observed["sdpa_by_presentation"],
            "gradient_norms": gradient_norms, "lr_used": lr_used,
            "lr_after_scheduler": lr_after, "presentation_metrics": metrics, "native_replay": replay, "rng": rng,
            "output_files": files, "input_files": context["evidence"].files, "resume_supported": False,
            "automatic_downstream": False, "training_pilot": False, "raw_pairs_exported": False,
            "elapsed_seconds": time.monotonic() - begun, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        validate_case_receipt(receipt, context, mode, output, verify_outputs=False)
        write_new_json(output / "complete.json", receipt)
        return receipt
    except BaseException as error:
        if journal is not None:
            try:
                journal.flush()
                journal.close()
            except BaseException:
                pass
        write_new_json(output / "failure.json", {"schema": "extension-joint-memory-admission-case-failure-v1", "complete": False,
            "mode": mode, "binding_sha256": context["binding_sha256"], "exception": type(error).__name__, "message": str(error)[:4000],
            "completed_presentations": [] if ledger is None else ledger.presentations,
            "pending_presentation": None if ledger is None else ledger.pending,
            "completed_update_boundaries": [] if ledger is None else ledger.updates,
            "actual_pre_forward_shape_observations": observations,
            "memory_stages_observed": memory.stages, "allocator_peak_allocated_seen": memory.whole_allocated,
            "allocator_peak_reserved_seen": memory.whole_reserved, "max_observed_nonallocator_bytes": memory.max_observed_nonallocator_bytes,
            "retry": False, "resume_supported": False, "automatic_downstream": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode is None:
        result = run(args.root, args.plan, args.output, args.qa_receipt, args.admission_receipt)
        print({"complete": result["complete"], "cases": len(result["case_receipts"]), "training_pilot": False})
    else:
        allocation = native.require_gpu_allocation()
        context = load_inputs(args.root, args.plan, args.qa_receipt, args.admission_receipt)
        parent = Path(args.output).resolve()
        require(parent.is_relative_to(context["root"] / "extension/runs"), "Unsafe worker parent output")
        started = read_json(parent / "started.json")
        require(started.get("binding_sha256") == context["binding_sha256"] and started.get("case_order") == list(MODES) and
            started.get("allocation", {}).get("job_id") == allocation["job_id"] and
            not (parent / "failure.json").exists() and not (parent / "complete.json").exists(), "Worker needs its active immutable parent")
        for mode in MODES[:MODES.index(args.mode)]:
            validate_case_receipt(read_json(parent / "cases" / mode / "complete.json"), context, mode, parent / "cases" / mode, verify_outputs=False)
        require(not any((parent / "cases" / later).exists() for later in MODES[MODES.index(args.mode) + 1:]), "Worker order differs")
        run_case(context, args.mode, parent / "cases" / args.mode, allocation)


if __name__ == "__main__":
    main()
