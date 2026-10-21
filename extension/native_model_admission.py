"""UNADMITTED eight-case, single-B200 diagnostic prototype.

No cache selection/preparation, pilot recipe, retry, resume, generation, scheduler
or upload is implemented. A future reviewed CPU preparation, native QA, frozen
plan and explicit job-admission receipt are all required. Run only on compute.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, value_digest, write_new_json)
from extension.labeled_feasibility import inside, CRITERION
from reasoning.data import prompt_id

REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
REPOSITORY = "Qwen/Qwen2.5-3B-Instruct"
MODES = ("sft", "ce_kl", "asft", "prior_anchored")
CASES = tuple(mode + "-" + kind for mode in MODES for kind in ("short", "long"))
VERSIONS = {"torch": "2.9.0", "transformers": "4.57.3", "accelerate": "1.12.0", "safetensors": "0.7.0"}
ORIGINAL_FILES = {
    "config.json": "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
    "generation_config.json": "ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
    "model.safetensors.index.json": "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}
TOKENIZER_FILES = {name: ORIGINAL_FILES[name] for name in
    ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")}
REQUIRED_CODE = frozenset({"extension/native_model_admission.py", "tests/test_native_model_admission.py",
    "extension/preserving_trainer.py", "extension/preserving_objectives.py", "extension/objectives.py",
    "extension/preserving_data.py", "reasoning/data.py", "reasoning/features.py"})
REQUIRED_CHECKS = frozenset({"fresh_original", "parity", "native_serialization", "independent_ce",
    "finite_loss", "finite_gradients", "finite_weights", "finite_moments", "reference_unchanged",
    "optimizer_isolation", "checkpoint_isolation", "checkpoint_reload_parity", "headroom",
    "exact_presentations", "fp32_parameter_change", "no_global_ce_patch"})
AUDIT_FIELDS = frozenset({"preparation", "development_seal", "reserved_pool_ids", "benchmark_audit",
    "benchmark_pool_ids", "exclusion_receipt", "pool", "features"})
_SHA = re.compile(r"[0-9a-f]{64}")


def _sha(value):
    require(type(value) is str and _SHA.fullmatch(value), "Expected lowercase SHA256")
    return value


def _number(value, name, minimum=0, strict=False):
    require(type(value) in (float, int) and math.isfinite(value) and
        (value > minimum if strict else value >= minimum), "Invalid finite configuration: " + name)
    return value


def validate_plan(value):
    require(isinstance(value, dict) and value.get("schema") == "extension-native-model-admission-plan-v1" and
        value.get("diagnostic_only") is True and value.get("automatic_downstream") is False and
        value.get("training_pilot") is False and value.get("base_revision") == REVISION and
        value.get("objectives") == list(MODES) and value.get("case_order") == list(CASES),
        "A separately frozen diagnostic-only eight-case plan is required")
    require(value.get("deadline_utc") == "2026-10-28T03:59:00+00:00", "Unexpected extension deadline")
    resources, settings, tolerances = (value.get(name, {}) for name in ("resources", "settings", "tolerances"))
    require(isinstance(resources, dict) and all(type(resources.get(name)) is int and resources[name] > 0
        for name in ("wall_time_seconds", "host_cpus", "host_ram_gib", "scratch_free_gib", "reserve_seconds")) and
        resources.get("gpus") == 1 and resources.get("gpu_type") == "B200" and
        resources["wall_time_seconds"] <= 7200, "Explicit bounded resources/reserve required")
    require(isinstance(settings, dict) and settings.get("student_dtype") == settings.get("reference_dtype") == "float32" and
        settings.get("bf16") is True and settings.get("attention") == "sdpa" and
        settings.get("gradient_checkpointing") is True and settings.get("checkpoint_use_reentrant") is False and
        settings.get("checkpoint_chunks") is True and settings.get("accumulation") == 2 and
        settings.get("optimizer") == "adamw_torch" and settings.get("scheduler") == "constant" and
        settings.get("resume") is False and settings.get("dropout_from_original_config_must_be_zero") is True and
        type(settings.get("chunk_tokens")) is int and settings["chunk_tokens"] > 0 and
        type(settings.get("seed")) is int, "Unsupported diagnostic precision/shape/path")
    for name in ("beta", "weight_decay"):
        _number(settings.get(name), name)
    for name in ("learning_rate", "clip_norm"):
        _number(settings.get(name), name, strict=True)
    require(isinstance(tolerances, dict) and type(tolerances.get("headroom_min_bytes")) is int and
        tolerances["headroom_min_bytes"] > 0, "Explicit memory headroom required")
    for name in ("logit_max_abs", "ce_abs", "kl_abs", "reload_logit_max_abs", "headroom_fraction"):
        _number(tolerances.get(name), name, strict=True)
    require(tolerances["headroom_fraction"] < 1, "Headroom fraction must be below one")
    return value


def _file_pin(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"} and
        isinstance(value["path"], str) and value["path"], "Explicit file path/hash pin required")
    _sha(value["sha256"])


def _example(row, serialization):
    require(isinstance(row, dict) and set(row) == {"id", "input_ids", "labels", "attention_mask", "original_row", "provenance"},
        "Use the complete preserving_data example schema")
    _sha(row["id"])
    original = row["original_row"]
    require(isinstance(original, dict) and set(original) == {"id", "instruction", "input", "output", "system"} and
        all(type(item) is str for item in original.values()) and original["input"] == "" and
        all(original[name].strip() for name in ("instruction", "output", "system")) and
        original["id"] == row["id"] == prompt_id(original["instruction"]), "Native canonical pair identity differs")
    tokens, labels, mask, proof = (row[name] for name in ("input_ids", "labels", "attention_mask", "provenance"))
    require(isinstance(tokens, list) and len(tokens) >= 2 and
        all(type(token) is int and 0 <= token < 151936 for token in tokens) and
        isinstance(labels, list) and len(labels) == len(tokens) and
        all(type(token) is int and (token == -100 or 0 <= token < 151936) for token in labels) and isinstance(mask, list) and
        mask == [1] * len(tokens) and isinstance(proof, dict), "Malformed complete native cache")
    start = next((index for index, target in enumerate(labels) if target != -100), len(labels))
    suffix = serialization["terminal_suffix"]
    require(0 < start < len(tokens) and labels == [-100] * start + tokens[start:] and
        len(tokens) - start > len(suffix) and tokens[-len(suffix):] == suffix,
        "Cache must preserve masked prompt and complete native response/EOS tail")
    require(proof.get("row_id") == row["id"] and proof.get("row_sha256") == value_digest(original) and
        proof.get("prompt_tokens") == start and proof.get("total_tokens") == len(tokens) and
        proof.get("active_tokens") == len(tokens) - start and
        proof.get("prompt_token_ids_sha256") == value_digest(tokens[:start]) and
        proof.get("full_token_ids_sha256") == value_digest(tokens) and proof.get("labels_sha256") == value_digest(labels) and
        proof.get("native_terminal_suffix") == suffix and proof.get("eos_token_id") == serialization["eos_token_id"] and
        proof.get("context_role") == "diagnostic" and proof.get("complete_original_target") is True and
        proof.get("truncated") is False and proof.get("target_rewritten") is False and
        proof.get("causal_shift_applied") is False, "Cache token/mask/provenance hash differs")
    return row


def validate_prepared(value, plan, plan_sha256):
    validate_plan(plan)
    require(isinstance(value, dict) and value.get("schema") == "extension-native-model-smoke-cache-v1" and
        value.get("complete") is True and value.get("plan_sha256") == _sha(plan_sha256), "Unbound prepared smoke cache")
    base, serial, rows = (value.get(name, {}) for name in ("base", "serialization", "rows"))
    require(isinstance(base, dict) and base.get("repository") == REPOSITORY and base.get("revision") == REVISION and
        isinstance(base.get("path"), str) and Path(base["path"]).is_absolute() and base.get("files") == ORIGINAL_FILES,
        "Only the exact pinned original base file inventory is allowed")
    _file_pin(base.get("identity_receipt"))
    require(isinstance(serial, dict) and serial.get("eos_token_id") == 151645 and
        isinstance(serial.get("terminal_suffix"), list) and serial["terminal_suffix"] and
        serial["terminal_suffix"][0] == 151645 and serial["terminal_suffix"].count(151645) == 1 and
        all(type(token) is int and 0 <= token < 151936 for token in serial["terminal_suffix"]) and
        isinstance(serial.get("terminal_text"), str) and serial["terminal_text"] and
        isinstance(serial.get("tokenizer_identity"), dict) and
        serial["tokenizer_identity"].get("files") == TOKENIZER_FILES,
        "Explicit pinned tokenizer/native EOS plus tail required")
    require(isinstance(rows, dict) and set(rows) == {"short", "long"} and
        isinstance(rows["short"], list) and len(rows["short"]) == 3, "Exactly three short and one longest witness required")
    all_rows = [_example(row, serial) for row in rows["short"] + [rows["long"]]]
    require(len({row["id"] for row in all_rows}) == 4 and all(len(row["input_ids"]) <= 512 for row in rows["short"]) and
        len({row["provenance"]["active_tokens"] for row in rows["short"]}) == 3 and
        all(row["provenance"].get("cutoff") == 512 for row in rows["short"]) and
        rows["long"]["provenance"].get("cutoff") == 32768 and
        max(len(row["input_ids"]) for row in rows["short"]) < len(rows["long"]["input_ids"]) <= 32768,
        "Witness IDs, short lengths or longest complete context differ")
    schedule = {"short": [row["id"] for row in rows["short"]], "long": [rows["long"]["id"]] * 2}
    require(value.get("presentation_schedule") == schedule, "Bind the exact before-forward presentation schedule")
    require(isinstance(value.get("source_identity"), dict) and bool(value["source_identity"]),
        "Prepared examples require the CPU-bound source identity")
    audit = value.get("source_audit", {})
    require(isinstance(audit, dict) and set(audit) == AUDIT_FIELDS, "Complete CPU source/reservation/benchmark audit pins required")
    for pin in audit.values():
        _file_pin(pin)
    return value


def case_schedule(prepared):
    return {case: list(prepared["presentation_schedule"][case.rsplit("-", 1)[1]]) for case in CASES}


def headroom_report(peak_bytes, total_bytes, min_bytes, fraction):
    require(all(type(value) is int and value >= 0 for value in (peak_bytes, total_bytes, min_bytes)) and
        total_bytes > 0 and peak_bytes <= total_bytes, "Invalid actual memory accounting")
    _number(fraction, "headroom_fraction", strict=True)
    require(fraction < 1, "Invalid headroom fraction")
    required = max(min_bytes, math.ceil(total_bytes * fraction))
    return {"peak_bytes": peak_bytes, "total_bytes": total_bytes, "free_after_peak_bytes": total_bytes - peak_bytes,
        "required_headroom_bytes": required, "passed": total_bytes - peak_bytes >= required}


def check_parity(metrics, tolerances):
    limits = {"max_abs_logits": "logit_max_abs", "ce_abs": "ce_abs", "kl_abs": "kl_abs"}
    require(isinstance(metrics, dict), "Numerical parity metrics required")
    for name, limit in limits.items():
        _number(metrics.get(name), name)
        _number(tolerances.get(limit), limit, strict=True)
        require(metrics[name] <= tolerances[limit], "Zero-update parity failed: " + name)
    return dict(metrics, passed=True)


def validate_gpu_allocation(text, job_id, node, uid):
    require(type(job_id) is str and re.fullmatch(r"[0-9]+", job_id), "Numeric allocated job required")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", text))
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    short_node = node.split(".", 1)[0].casefold() if isinstance(node, str) else ""
    require(fields.get("JobId") == job_id and fields.get("JobState") == "RUNNING" and user and int(user.group(1)) == uid and
        fields.get("NumNodes") == "1" and fields.get("NodeList") == node and node and
        short_node not in {"owl1", "tc2"} and "login" not in node.casefold(),
        "Not the actual user's one-node RUNNING allocation")
    allocated = fields.get("AllocTRES", "")
    require(re.search(r"(?:^|,)gres/gpu=1(?:,|$)", allocated) and
        re.search(r"(?:^|,)gres/gpu:b200=1(?:,|$)", allocated, re.I), "Exactly one allocated B200 TRES is required")
    return {"job_id": job_id, "node": node, "user_id": uid, "state": "RUNNING", "allocated_tres": allocated,
        "scontrol_sha256": hashlib.sha256(text.encode()).hexdigest(), "scontrol": text.strip()}


def require_gpu_allocation():
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(node and socket.gethostname().split(".")[0] == node.split(".")[0] and
        os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1", "NoDevFiles"), "Compute node and assigned GPU required")
    require(os.environ.get("WORLD_SIZE", "1") == "1", "One unsharded process only")
    result = subprocess.run(["scontrol", "--clusters=owl", "show", "job", "--oneliner", job],
        check=True, capture_output=True, text=True, timeout=30)
    proof = validate_gpu_allocation(result.stdout, job, node, os.getuid())
    proof.update(step_id=os.environ.get("SLURM_STEP_ID", "batch"), hostname=socket.gethostname(),
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], checked_at_utc=datetime.now(timezone.utc).isoformat())
    return proof


def reserve_output(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Output already exists; no reuse/resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path.resolve()


def _read_pin(root, evidence, pin):
    _file_pin(pin)
    return evidence.read(_project_path(root, pin["path"]), pin["sha256"])


def _project_path(root, value):
    """CLI and native receipts may use relative or exact in-project absolute paths."""
    path = Path(value)
    if path.is_absolute():
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Input path escapes the project")
        return path.resolve()
    return inside(root, str(value))


def load_inputs(root, plan_path, prepared_path, qa_receipt, admission_receipt):
    """Verify actual input bytes on compute; CPU preparation remains separately audited."""
    root = Path(root).resolve()
    plan_path, prepared_path = _project_path(root, plan_path), _project_path(root, prepared_path)
    evidence = Evidence()
    plan = validate_plan(evidence.read(plan_path))
    prepared = validate_prepared(evidence.read(prepared_path), plan, evidence.check(plan_path))
    qa_path, admission_path = _project_path(root, qa_receipt), _project_path(root, admission_receipt)
    qa, admission = evidence.read(qa_path), evidence.read(admission_path)
    hashes = qa.get("code_sha256", {})
    require(qa.get("schema") == "extension-native-model-admission-native-qa-v1" and qa.get("success") is True and
        type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and
        all(type(qa.get(name)) is int and qa[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("gpu_disabled") is True and qa.get("plan_sha256") == evidence.check(plan_path) and
        qa.get("prepared_sha256") == evidence.check(prepared_path) and isinstance(hashes, dict) and REQUIRED_CODE <= set(hashes),
        "New admission-specific native QA bound to this cache/plan/code is required; prior230 is insufficient")
    for name, checksum in hashes.items():
        path = inside(root, name)
        require(path.suffix in (".py", ".json", ".slurm") and path.stat().st_size <= 2 << 20,
            "Only explicit bounded source/config files enter code QA")
        evidence.check(path, _sha(checksum))
    require(admission.get("schema") == "extension-native-model-admission-job-v1" and admission.get("status") == "admitted" and
        admission.get("job_id") == os.environ.get("SLURM_JOB_ID") and
        admission.get("plan_sha256") == evidence.check(plan_path) and
        admission.get("prepared_sha256") == evidence.check(prepared_path) and admission.get("qa_sha256") == evidence.check(qa_path) and
        admission.get("code_sha256") == hashes and admission.get("automatic_downstream") is False,
        "Explicit current-job admission binding required")
    base = Path(prepared["base"]["path"]).resolve()
    for name, checksum in ORIGINAL_FILES.items():
        evidence.check(base / name, checksum)
    identity = _read_pin(root, evidence, prepared["base"]["identity_receipt"])
    require(identity.get("path") == str(base) and identity.get("model_type") == "qwen2" and
        {name: entry.get("sha256") for name, entry in identity.get("files", {}).items()} == ORIGINAL_FILES and
        identity.get("identity_sha256") == value_digest({key: value for key, value in identity.items() if key != "identity_sha256"}),
        "Actual pinned original identity receipt differs")
    audit = prepared["source_audit"]
    values = {}
    for name, pin in audit.items():
        path = _project_path(root, pin["path"])
        evidence.check(path, pin["sha256"])
        if name not in ("pool", "features"):
            values[name] = evidence.read(path)
    require(prepared["source_identity"].get("pool_sha256") == audit["pool"]["sha256"] and
        prepared["source_identity"].get("features_sha256") == audit["features"]["sha256"],
        "Caller source identity differs from actual pinned source/features")
    seal, reserved, benchmark, blocked = (values[name] for name in
        ("development_seal", "reserved_pool_ids", "benchmark_audit", "benchmark_pool_ids"))
    require(seal.get("schema") == "extension-development-seal-v2" and seal.get("complete") is True and
        seal.get("input_stats_unchanged_at_closeout") is True and
        seal.get("output_sha256", {}).get("reserved_training_pool_ids.json") == audit["reserved_pool_ids"]["sha256"] and
        seal.get("reservation", {}).get("reserved_training_pool_ids_sha256") == value_digest(reserved),
        "Completed repaired development reservation is missing")
    require(benchmark.get("schema") == "extension-final-benchmark-pool-exclusions-v1" and benchmark.get("complete") is True and
        benchmark.get("pool_sha256") == audit["pool"]["sha256"] and benchmark.get("criterion") == CRITERION and
        benchmark.get("scope") == "all_nine_final_benchmarks" and
        set(benchmark.get("benchmark_question_sha256", {})) == {"aime", "aime25", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"} and
        benchmark.get("excluded_pool_ids_sha256") == value_digest(blocked) and
        benchmark.get("excluded_pool_ids_file_sha256") == audit["benchmark_pool_ids"]["sha256"] and
        benchmark.get("answers_or_outcomes_used") is False, "Final all-nine benchmark lexical audit is missing")
    for name, checksum in benchmark["benchmark_question_sha256"].items():
        evidence.check(root / "deadline/artifacts/benchmarks" / name / "test.jsonl", _sha(checksum))
    for identifiers in (reserved, blocked):
        require(isinstance(identifiers, list) and identifiers == sorted(set(identifiers)) and
            all(type(item) is str and _SHA.fullmatch(item) for item in identifiers), "Invalid exclusion IDs")
    excluded = set(reserved) | set(blocked)
    require(not excluded.intersection(row["id"] for row in prepared["rows"]["short"] + [prepared["rows"]["long"]]),
        "Smoke witness overlaps reserved or benchmark questions")
    exclusion = values["exclusion_receipt"]
    require(exclusion.get("schema") == "preserving-pool-exclusions-v1" and
        exclusion.get("id_namespace") == "canonical_pool_prompt_sha256_v1" and
        _SHA.fullmatch(str(exclusion.get("population_ids_sha256", ""))), "Missing independently bound union receipt")
    components = exclusion.get("components", {})
    require(set(components) == {"reserved_source_evaluation", "final_benchmark_questions"}, "Incomplete union scope")
    for name, ids, audit_name in (("reserved_source_evaluation", reserved, "development_seal"),
            ("final_benchmark_questions", blocked, "benchmark_audit")):
        component = components[name]
        expected_scope = "all_reserved_source_splits" if name == "reserved_source_evaluation" else "all_nine_final_benchmarks"
        require(component.get("pool_ids") == ids and component.get("audit_receipt_sha256") == audit[audit_name]["sha256"] and
            component.get("criterion_sha256") == value_digest(CRITERION) and component.get("scope") == expected_scope and
            _SHA.fullmatch(str(component.get("question_inventory_sha256", ""))), "Union component differs from actual audit bytes")
    preparation = values["preparation"]
    row_binding = value_digest(prepared["rows"])
    require(preparation.get("schema") == "extension-native-smoke-preparation-v1" and preparation.get("complete") is True and
        preparation.get("cpu_only") is True and preparation.get("gpu_seconds") == 0 and
        preparation.get("plan_sha256") == evidence.check(plan_path) and preparation.get("rows_sha256") == row_binding and
        preparation.get("exclusion_receipt_sha256") == audit["exclusion_receipt"]["sha256"] and
        preparation.get("pool_sha256") == audit["pool"]["sha256"] and preparation.get("features_sha256") == audit["features"]["sha256"] and
        preparation.get("source_identity") == prepared["source_identity"] and
        preparation.get("no_model_or_benchmark_outcomes") is True and preparation.get("input_snapshot_unchanged") is True,
        "Reviewed CPU cache preparation receipt is missing or does not bind actual witnesses")
    native = preparation.get("native_qa", {})
    _file_pin({key: native.get(key) for key in ("path", "sha256")})
    unit = _read_pin(root, evidence, {key: native[key] for key in ("path", "sha256")})
    prep_sources = preparation.get("code_sha256", {})
    require(unit.get("schema") == "extension-native-smoke-preparation-unit-qa-v1" and unit.get("success") is True and
        type(unit.get("tests_run")) is int and unit["tests_run"] > 0 and
        unit.get("native_plan_sha256") == evidence.check(plan_path) and unit.get("gpu_disabled") is True and
        all(type(unit.get(name)) is int and unit[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        isinstance(prep_sources, dict) and prep_sources == unit.get("code_sha256") and
        {"extension/smoke_preparation.py", "tests/test_smoke_preparation.py", "extension/preserving_data.py"} <= set(prep_sources),
        "CPU preparation lacks its independently bound native source/unit-QA proof")
    compute = unit.get("compute_proof", {})
    require(isinstance(compute, dict) and compute.get("gpu_requested_or_allocated") is False and
        compute.get("job_id") == unit.get("job_id") and compute.get("node") == unit.get("node"),
        "CPU preparation QA is not bound to its CPU allocation")
    for name, checksum in prep_sources.items():
        evidence.check(inside(root, name), _sha(checksum))
    snapshot = preparation.get("input_files", {})
    require(isinstance(snapshot, dict) and bool(snapshot), "CPU preparation lacks its complete input snapshot")
    base_inputs = {(base / name).resolve() for name in ORIGINAL_FILES}
    for name, entry in snapshot.items():
        path = Path(name).resolve()
        require(Path(name).is_absolute() and (path.is_relative_to(root) or path in base_inputs) and
            isinstance(entry, dict) and type(entry.get("bytes")) is int and entry["bytes"] == path.stat().st_size,
            "CPU preparation input snapshot escapes pinned source/base files or changed size")
        evidence.check(path, _sha(entry.get("sha256")))
    longest = preparation.get("longest", {})
    require(longest.get("id") == prepared["rows"]["long"]["id"] and
        longest.get("total_tokens") == longest.get("max_eligible_total_tokens") == len(prepared["rows"]["long"]["input_ids"]) and
        longest.get("after_reserved_exclusion") is True and longest.get("after_benchmark_exclusion") is True,
        "Longest eligible complete-row proof is unbound")
    evidence.unchanged()
    context = dict(root=root, plan_path=plan_path, prepared_path=prepared_path, qa_path=qa_path,
        admission_path=admission_path, plan=plan, prepared=prepared, plan_sha256=evidence.check(plan_path),
        prepared_sha256=evidence.check(prepared_path), qa=qa, admission=admission, code_sha256=hashes,
        evidence=evidence, exclusion_receipt=exclusion, preparation=preparation)
    context["binding_sha256"] = value_digest({"plan_sha256": context["plan_sha256"],
        "prepared_sha256": context["prepared_sha256"], "qa_sha256": evidence.check(qa_path),
        "admission_sha256": evidence.check(admission_path), "code_sha256": hashes})
    return context


def validate_case_receipt(receipt, context, case, case_output, *, verify_outputs=True, output_evidence=None):
    mode, kind = case.rsplit("-", 1)
    require(isinstance(receipt, dict) and receipt.get("schema") == "extension-native-model-admission-case-v1" and
        receipt.get("complete") is True and receipt.get("case") == case and receipt.get("mode") == mode and receipt.get("kind") == kind and
        receipt.get("binding_sha256") == context["binding_sha256"] and receipt.get("plan_sha256") == context["plan_sha256"] and
        receipt.get("prepared_sha256") == context["prepared_sha256"] and receipt.get("code_sha256") == context["code_sha256"] and
        receipt.get("job_id") == context["admission"]["job_id"] and
        receipt.get("presentation_ids") == case_schedule(context["prepared"])[case] and
        receipt.get("optimizer_updates") == (2 if kind == "short" else 1) and
        set(receipt.get("checks", {})) == REQUIRED_CHECKS and all(item is True for item in receipt["checks"].values()),
        "Incomplete or mismatched native case receipt")
    memory, limits = receipt.get("memory", {}), context["plan"]["tolerances"]
    expected = headroom_report(max(memory.get("peak_allocated_bytes", -1), memory.get("peak_reserved_bytes", -1)),
        memory.get("total_bytes", 0), limits["headroom_min_bytes"], limits["headroom_fraction"])
    require(expected["passed"] and memory.get("headroom") == expected, "Case failed its measured memory headroom")
    files = receipt.get("output_files", {})
    require(isinstance(files, dict) and "checkpoint/config.json" in files and any(name.endswith(".safetensors") for name in files),
        "Case lacks a complete saved-student checkpoint inventory")
    checkpoint = case_output / "checkpoint"
    require(checkpoint.is_dir() and all(path.is_file() and not path.is_symlink() for path in checkpoint.iterdir()) and
        {str(path.relative_to(case_output)) for path in checkpoint.iterdir()} == set(files),
        "Saved checkpoint has unverified extra files or unsupported nested layout")
    for name, checksum in files.items():
        path = (case_output / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(case_output.resolve()) and path.is_file(), "Unsafe checkpoint inventory path")
        _sha(checksum)
        if verify_outputs:
            actual = output_evidence.check(path, checksum) if output_evidence is not None else digest(path)
            require(actual == checksum, "Case output changed after native verification")
    return receipt


def run(root, plan_path, prepared_path, output, qa_receipt, admission_receipt, *, runner=None):
    allocation = require_gpu_allocation()
    context = load_inputs(root, plan_path, prepared_path, qa_receipt, admission_receipt)
    limits = context["plan"]["resources"]
    deadline_guard(limits["wall_time_seconds"] + limits["reserve_seconds"])
    output = Path(output).resolve()
    require(output.is_relative_to(context["root"] / "extension/runs"), "Diagnostic outputs must remain in extension/runs")
    output = reserve_output(output)
    write_new_json(output / "started.json", {"schema": "extension-native-model-admission-v1", "complete": False,
        "binding_sha256": context["binding_sha256"], "allocation": allocation, "cases": list(CASES),
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    begun, execute, receipts = time.monotonic(), runner or subprocess.run, {}
    output_evidence = Evidence()
    try:
        require(shutil.disk_usage(output).free >= limits["scratch_free_gib"] * (1 << 30), "Explicit scratch capacity is unavailable")
        for case in CASES:
            remaining = math.floor(limits["wall_time_seconds"] - (time.monotonic() - begun))
            require(remaining > 0, "Diagnostic wall-time bound exhausted")
            deadline_guard(remaining + limits["reserve_seconds"])
            command = [sys.executable, "-m", "extension.native_model_admission", "--root", str(context["root"]),
                "--plan", str(context["plan_path"].relative_to(context["root"])),
                "--prepared", str(context["prepared_path"].relative_to(context["root"])), "--output", str(output),
                "--qa-receipt", str(context["qa_path"].relative_to(context["root"])),
                "--admission-receipt", str(context["admission_path"].relative_to(context["root"])), "--case", case]
            result = execute(command, cwd=str(context["root"]), check=False, timeout=remaining)
            require(result.returncode == 0, "Native diagnostic case failed without retry: " + case)
            case_output = output / "cases" / case
            receipt = read_json(case_output / "complete.json")
            validate_case_receipt(receipt, context, case, case_output, output_evidence=output_evidence)
            receipts[case] = {"path": str(case_output / "complete.json"), "sha256": digest(case_output / "complete.json")}
            output_evidence.check(case_output / "complete.json", receipts[case]["sha256"])
            output_evidence.unchanged()
            context["evidence"].unchanged()
        ending = require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Parent allocation changed")
        context["evidence"].unchanged()
        output_evidence.unchanged()
        receipt = {"schema": "extension-native-model-admission-v1", "complete": True,
            "binding_sha256": context["binding_sha256"], "plan_sha256": context["plan_sha256"],
            "prepared_sha256": context["prepared_sha256"], "code_sha256": context["code_sha256"],
            "allocation": allocation, "allocation_end": ending, "case_receipts": receipts, "input_files": context["evidence"].files,
            "input_snapshot_unchanged": True, "automatic_downstream": False, "training_pilot": False,
            "output_snapshot_unchanged": True,
            "resume_supported": False, "elapsed_seconds": time.monotonic() - begun,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_new_json(output / "complete.json", receipt)
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {"schema": "extension-native-model-admission-failure-v1", "complete": False,
            "binding_sha256": context["binding_sha256"], "completed_cases": receipts,
            "exception": type(error).__name__, "message": str(error)[:4000], "retry": False, "automatic_downstream": False})
        raise


def tensor_inventory(model):
    """Stream named parameter bytes; no complete model/gradient copy is retained."""
    import torch
    result = {}
    for name, parameter in model.named_parameters():
        require(parameter.dtype == torch.float32 and not parameter.is_sparse, "Expected dense FP32 parameters")
        checksum = hashlib.sha256()
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 1 << 20):
            checksum.update(flat[start:start + (1 << 20)].to("cpu").contiguous().numpy().tobytes())
        result[name] = {"sha256": checksum.hexdigest(), "shape": list(parameter.shape),
            "dtype": "float32", "elements": parameter.numel()}
    require(bool(result), "Model has no named parameters")
    return result


def assert_finite_model(model, *, gradients=False):
    import torch
    count = 0
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        require(value is not None and value.dtype == torch.float32, "Missing or non-FP32 tensor: " + name)
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), 1 << 22):
            require(bool(torch.isfinite(flat[start:start + (1 << 22)]).all()), "Nonfinite model tensor: " + name)
        count += 1
    return count


class MemoryTrace:
    """Nested stage peaks plus a retained whole-process maximum, never emptied."""
    def __init__(self, torch):
        self.torch, self.stages, self.stack = torch, [], []
        torch.cuda.synchronize()
        free, self.total = torch.cuda.mem_get_info()
        self.baseline_driver_bytes = self.total - free - torch.cuda.memory_reserved()
        self.max_observed_nonallocator_bytes = max(0, self.baseline_driver_bytes)
        self.whole_allocated = self.whole_reserved = 0
        torch.cuda.reset_peak_memory_stats()

    def _capture(self):
        cuda = self.torch.cuda
        cuda.synchronize()
        allocated, reserved = cuda.max_memory_allocated(), cuda.max_memory_reserved()
        self.whole_allocated, self.whole_reserved = max(self.whole_allocated, allocated), max(self.whole_reserved, reserved)
        for frame in self.stack:
            frame["peak_allocated_bytes"] = max(frame["peak_allocated_bytes"], allocated)
            frame["peak_reserved_bytes"] = max(frame["peak_reserved_bytes"], reserved)
        free, total = cuda.mem_get_info()
        self.max_observed_nonallocator_bytes = max(self.max_observed_nonallocator_bytes,
            total - free - cuda.memory_reserved())
        return {"allocated_bytes": cuda.memory_allocated(), "reserved_bytes": cuda.memory_reserved(),
            "free_device_bytes": free, "total_bytes": total}

    def begin(self, label):
        current = self._capture()
        frame = {"stage": label, "started": time.monotonic(), "peak_allocated_bytes": current["allocated_bytes"],
            "peak_reserved_bytes": current["reserved_bytes"], "start": current}
        self.stack.append(frame)
        self.torch.cuda.reset_peak_memory_stats()
        return frame

    def end(self, frame):
        current = self._capture()
        require(self.stack and self.stack[-1] is frame, "Memory phases must close in nested order")
        self.stack.pop()
        frame.update(end=current, elapsed_seconds=time.monotonic() - frame.pop("started"))
        self.stages.append(frame)

    def abort(self, frame):
        """Preserve the triggering error even if an optimizer left a phase open."""
        if not any(active is frame for active in self.stack):
            return
        capture_error = None
        try:
            self._capture()
        except BaseException as error:
            capture_error = type(error).__name__
        while self.stack:
            active = self.stack.pop()
            active.update(aborted=True, complete=False,
                elapsed_seconds=time.monotonic() - active.pop("started"), capture_error=capture_error)
            self.stages.append(active)
            if active is frame:
                break

    @contextmanager
    def phase(self, label):
        frame = self.begin(label)
        try:
            yield
        except BaseException:
            try:
                self.abort(frame)
            except BaseException:
                # Diagnostics must never replace the original CUDA/optimizer error.
                pass
            raise
        else:
            self.end(frame)

    def record(self, label):
        self.stages.append({"stage": label, "snapshot": self._capture()})

    def result(self, tolerances):
        self._capture()
        # Independently observed maxima are summed conservatively; nonallocator
        # memory is sampled, so no unobserved driver transient is claimed exact.
        allocated = self.whole_allocated + self.max_observed_nonallocator_bytes
        reserved = self.whole_reserved + self.max_observed_nonallocator_bytes
        return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "total_bytes": self.total,
            "allocator_peak_allocated_bytes": self.whole_allocated, "allocator_peak_reserved_bytes": self.whole_reserved,
            "baseline_driver_bytes": self.baseline_driver_bytes,
            "max_observed_nonallocator_bytes": self.max_observed_nonallocator_bytes,
            "nonallocator_qualification": "Sampled at synchronized captures; summed with allocator maxima conservatively. Unobserved driver transients are not measured exactly.",
            "stages": self.stages,
            "headroom": headroom_report(max(allocated, reserved), self.total,
                tolerances["headroom_min_bytes"], tolerances["headroom_fraction"])}


def _checkpoint_inventory(path, expected):
    import torch
    from safetensors import safe_open
    actual = {}
    files = sorted(path.glob("*.safetensors"))
    require(files, "No saved safetensors checkpoint")
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                require(name in expected and name not in actual, "Saved checkpoint has unexpected/reference/duplicate tensor")
                tensor = handle.get_tensor(name)
                require(tensor.dtype == torch.float32 and list(tensor.shape) == expected[name]["shape"] and
                    bool(torch.isfinite(tensor).all()), "Saved weights must be finite matching FP32 tensors")
                checksum = hashlib.sha256()
                flat = tensor.reshape(-1)
                for start in range(0, flat.numel(), 1 << 20):
                    checksum.update(flat[start:start + (1 << 20)].numpy().tobytes())
                require(checksum.hexdigest() == expected[name]["sha256"], "Saved checkpoint differs from post-update student")
                actual[name] = expected[name]
    require(set(actual) == set(expected), "Saved checkpoint lost student parameters")
    return {"tensors": len(actual), "tensor_inventory_sha256": value_digest(actual), "all_finite_fp32": True}


def _host_resources(allocation, resources):
    match = re.search(r"(?:^|,)mem=([0-9]+)([KMGTP]?)(?:,|$)", allocation["allocated_tres"], re.I)
    require(match is not None, "Allocated host memory is not recorded")
    factor = 1 << (10 * " KMGTP".index(match.group(2).upper())) if match.group(2) else 1
    require(int(match.group(1)) * factor >= resources["host_ram_gib"] * (1 << 30), "Allocated host memory below frozen requirement")
    cpus = re.search(r"(?:^|,)cpu=([0-9]+)(?:,|$)", allocation["allocated_tres"])
    require(cpus is not None and int(cpus.group(1)) >= resources["host_cpus"], "Allocated host CPUs below frozen requirement")


def assigned_gpu_identity(torch):
    """Query only the CUDA-assigned device, by its UUID rather than another ordinal."""
    properties = torch.cuda.get_device_properties(0)
    uuid = str(getattr(properties, "uuid", ""))
    require(re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid) is not None, "Assigned GPU UUID is unavailable")
    result = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-gpu=uuid,driver_version",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, "Assigned-device query returned ambiguous GPUs")
    fields = [field.strip() for field in lines[0].split(",")]
    require(len(fields) == 2 and fields[0].lower() == uuid.lower() and
        re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", fields[1]) is not None, "Assigned GPU identity/driver differs")
    return {"uuid": fields[0], "driver_version": fields[1], "name": torch.cuda.get_device_name(0),
        "properties": str(properties), "query_scope": "CUDA-assigned device UUID only"}


def run_case(context, case, output, allocation):
    """Actual model work: callable only after all input/admission guards passed."""
    require(case in CASES, "Unknown diagnostic case")
    actual_allocation = require_gpu_allocation()
    require(actual_allocation["job_id"] == allocation["job_id"] == context["admission"]["job_id"] and
        actual_allocation["node"] == allocation["node"], "Worker is not inside the bound current GPU allocation")
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed
    from transformers.loss.loss_utils import LOSS_MAPPING
    from extension.preserving_data import verify_serialized_examples
    from extension.preserving_trainer import make_trainer_class, check_optimizer, disjoint_models

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
        "Exactly one BF16-capable assigned CUDA GPU is required")
    require("B200" in torch.cuda.get_device_name(0), "Actual assigned device is not B200")
    for name, version in VERSIONS.items():
        require(importlib.metadata.version(name) == version, "Unvalidated native version: " + name)
    _host_resources(allocation, context["plan"]["resources"])
    mode, kind = case.rsplit("-", 1)
    prepared, plan = context["prepared"], context["plan"]
    settings, serial, limits = plan["settings"], prepared["serialization"], plan["tolerances"]
    base = Path(prepared["base"]["path"])
    initial_loss = LOSS_MAPPING["ForCausalLM"]
    require(initial_loss.__module__ == "transformers.loss.loss_utils", "Global causal loss mapping was patched")
    set_seed(settings["seed"])
    memory = MemoryTrace(torch)
    begun = time.monotonic()
    output = reserve_output(output)
    write_new_json(output / "started.json", {"schema": "extension-native-model-admission-case-v1", "complete": False,
        "case": case, "binding_sha256": context["binding_sha256"], "allocation": allocation})
    try:
        gpu_identity = assigned_gpu_identity(torch)
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
        replay = {}
        for cohort, examples, cutoff in (("short", prepared["rows"]["short"], 512), ("long", [prepared["rows"]["long"]], 32768)):
            replay[cohort] = verify_serialized_examples(examples, tokenizer, source_identity=prepared["source_identity"],
                tokenizer_identity=serial["tokenizer_identity"], exclusion_receipt_sha256=value_digest(context["exclusion_receipt"]),
                cutoff=cutoff, context_role="diagnostic", eos_token_id=serial["eos_token_id"],
                terminal_suffix=tuple(serial["terminal_suffix"]), terminal_text=serial["terminal_text"], vocab_size=151936)
        with memory.phase("load_fresh_fp32_student_and_reference"):
            def load_model(path):
                model, info = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32, attn_implementation="sdpa",
                    local_files_only=True, trust_remote_code=False, output_loading_info=True)
                require(all(not info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                    "Model load was partial or initialized additional parameters")
                return model.to("cuda")

            student = load_model(base)
            reference = load_model(base).eval().requires_grad_(False)
            student.config.use_cache = reference.config.use_cache = False
            require(student.config.model_type == "qwen2" and student.config.vocab_size == 151936 and
                student.config.max_position_embeddings == 32768 and student.config.attention_dropout == 0 and
                student.config.use_sliding_window is False, "Pinned architecture/dropout/context differs")
            require(all(parameter.dtype == torch.float32 and parameter.requires_grad for parameter in student.parameters()) and
                all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in reference.parameters()),
                "Fresh full-parameter/reference precision differs")
            disjoint_models(student, reference)
            student_items, reference_items = dict(student.named_parameters()), dict(reference.named_parameters())
            require(set(student_items) == set(reference_items) and len(student_items) == 434 and
                all(torch.equal(student_items[name], reference_items[name]) for name in student_items), "Fresh original parameters differ")
            reference_before = tensor_inventory(reference)
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        witnesses = prepared["rows"]["short"] if kind == "short" else [prepared["rows"]["long"]] * 2
        identifiers = [row["id"] for row in witnesses]
        short = prepared["rows"]["short"][0]
        observed = {"forward_ids": [], "backward_ids": [], "updates": 0, "gradient_norms": [],
            "logits": [], "parity": None, "independent_ce": None, "optimizer_phase": None}

        def batch(row, index=None):
            result = {name: torch.tensor([row[name]], dtype=torch.int64, device="cuda")
                for name in ("input_ids", "labels", "attention_mask")}
            if index is not None:
                result["row_index"] = torch.tensor([index], dtype=torch.int64, device="cuda")
            return result

        def forward_tokens(row):
            return {name: value for name, value in batch(row).items() if name != "labels"}

        def output_hook(label):
            def capture(model, inputs, outputs):
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                observed["logits"].append({"model": label, "dtype": str(logits.dtype), "shape": list(logits.shape),
                    "grad_enabled": torch.is_grad_enabled(), "training": model.training})
                memory.record(label + "_forward_returned")
            return capture

        handles = [student.register_forward_hook(output_hook("student")), reference.register_forward_hook(output_hook("reference"))]
        Base = make_trainer_class()

        class OrderedTrainer(Base):
            def _get_train_sampler(self, train_dataset=None):
                return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

            def compute_loss(self, model, inputs, **kwargs):
                if model.training and torch.is_grad_enabled():
                    index = int(inputs["row_index"].item())
                    observed["forward_ids"].append(identifiers[index])
                with memory.phase("external_loss_forward"):
                    return super().compute_loss(model, inputs, **kwargs)

            def training_step(self, model, inputs, num_items_in_batch=None):
                index = int(inputs["row_index"].item())
                with memory.phase("microbatch_forward_and_backward"):
                    result = super().training_step(model, inputs, num_items_in_batch)
                observed["backward_ids"].append(identifiers[index])
                return result

        class Witnesses(torch.utils.data.Dataset):
            def __len__(self):
                return len(witnesses)

            def __getitem__(self, index):
                row = witnesses[index]
                return {**{name: row[name] for name in ("input_ids", "labels", "attention_mask")}, "row_index": index}

        def collate(rows):
            require(len(rows) == 1, "Only microbatch1 is admitted")
            return {name: torch.tensor([value], dtype=torch.int64) for name, value in rows[0].items()}

        args = TrainingArguments(output_dir=str(output / "trainer"), bf16=True, fp16=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=2, num_train_epochs=1,
            learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
            lr_scheduler_type="constant", warmup_steps=0, max_grad_norm=0., optim="adamw_torch",
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False, label_smoothing_factor=0., eval_strategy="no", push_to_hub=False,
            save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True,
            dataloader_num_workers=0, dataloader_pin_memory=False, seed=settings["seed"], data_seed=settings["seed"])
        trainer = OrderedTrainer(model=student, reference_model=reference, args=args, train_dataset=Witnesses(),
            data_collator=collate, objective=mode, eos_token_id=serial["eos_token_id"], terminal_suffix=tuple(serial["terminal_suffix"]),
            beta=settings["beta"], chunk_tokens=settings["chunk_tokens"], checkpoint_chunks=True, clip_norm=settings["clip_norm"])

        class Observe(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                with memory.phase("prepared_zero_update_parity_and_independent_ce"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.train()
                    p = model(**forward_tokens(short), use_cache=False).logits.float()
                    q = reference(**forward_tokens(short), use_cache=False).logits.float()
                    labels = batch(short)["labels"][:, 1:]
                    active = labels != -100
                    logp, logq = p[:, :-1].log_softmax(-1), q[:, :-1].log_softmax(-1)
                    ce = torch.nn.functional.cross_entropy(p[:, :-1].reshape(-1, p.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    qce = torch.nn.functional.cross_entropy(q[:, :-1].reshape(-1, q.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    kl = (logq.exp() * (logq - logp)).sum(-1)[active].mean()
                    observed["parity"] = check_parity({"max_abs_logits": float((p - q).abs().max()),
                        "ce_abs": float((ce - qce).abs()), "kl_abs": abs(float(kl))}, limits)
                    model.eval()
                    trainer.compute_loss(model, batch(short))
                    observed["independent_ce"] = abs(trainer.last_metrics["ce"] - float(ce))
                    require(observed["independent_ce"] <= limits["ce_abs"], "Independent shifted short CE differs")
                    model.train()

            def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
                assert_finite_model(model, gradients=True)
                observed["gradient_norms"].append(trainer.last_gradient_norm)
                observed["optimizer_phase"] = memory.begin("first_or_subsequent_adam_step")

            def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
                check_optimizer(model, optimizer, reference)
                observed["updates"] += 1
                memory.end(observed.pop("optimizer_phase"))

        trainer.add_callback(Observe())
        with memory.phase("actual_trainer_train"), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False, profile_memory=False, with_stack=False) as profile:
            trainer.train()
        sdpa = sorted({item.key for item in profile.key_averages() if "_scaled_dot_product_" in item.key and
            any(name in item.key for name in ("flash", "efficient", "cudnn", "math"))})
        require(sdpa, "Actual SDPA operator dispatch was not observed")
        expected_updates = 2 if kind == "short" else 1
        require(observed["forward_ids"] == observed["backward_ids"] == identifiers and
            observed["updates"] == trainer.state.global_step == expected_updates and
            trainer.training_forward_examples == len(witnesses), "Actual presentations/update boundaries differ")
        with memory.phase("post_update_full_finite_and_reference_scan"):
            finite_count = assert_finite_model(student)
            optimizer = check_optimizer(student, trainer.optimizer, reference)
            require(optimizer.state and all(parameter in optimizer.state for parameter in student.parameters()), "Adam did not initialize every student state")
            require(all(parameter.grad is None for parameter in reference.parameters()) and not reference.training,
                "Reference acquired gradients or train mode")
            require(tensor_inventory(reference) == reference_before, "Frozen original reference changed")
            student_inventory = tensor_inventory(student)
            require(any(student_inventory[name]["sha256"] != reference_before[name]["sha256"] for name in student_inventory),
                "Nonzero FP32 diagnostic update changed no student parameters")
        checkpoint = output / "checkpoint"
        with memory.phase("save_and_verify_student_checkpoint"):
            trainer.save_model(str(checkpoint))
            for name in TOKENIZER_FILES:
                shutil.copyfile(base / name, checkpoint / name)
            saved = _checkpoint_inventory(checkpoint, student_inventory)
        with memory.phase("saved_student_reload_parity"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student.eval()
            expected_logits = student(**forward_tokens(short), use_cache=False).logits.float().cpu()
            reloaded = load_model(checkpoint).eval()
            actual_logits = reloaded(**forward_tokens(short), use_cache=False).logits.float().cpu()
            reload_difference = float((expected_logits - actual_logits).abs().max())
            require(reload_difference <= limits["reload_logit_max_abs"], "Reloaded student short forward differs")
            del reloaded, actual_logits, expected_logits
        require(LOSS_MAPPING["ForCausalLM"] is initial_loss, "Global causal loss mapping changed")
        require(any(item["model"] == "student" and item["dtype"] == "torch.float32" for item in observed["logits"]),
            "Actual Accelerate-prepared FP32 external output was not observed")
        for handle in handles:
            handle.remove()
        memory_result = memory.result(limits)
        require(memory_result["headroom"]["passed"], "Measured whole-process headroom is insufficient")
        context["evidence"].unchanged()
        ending = require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Allocation changed")
        outputs = {str(file.relative_to(output)): digest(file) for file in sorted(checkpoint.iterdir()) if file.is_file()}
        receipt = {"schema": "extension-native-model-admission-case-v1", "complete": True, "case": case, "mode": mode, "kind": kind,
            "binding_sha256": context["binding_sha256"], "plan_sha256": context["plan_sha256"],
            "prepared_sha256": context["prepared_sha256"], "code_sha256": context["code_sha256"],
            "job_id": allocation["job_id"], "allocation_start": allocation, "allocation_end": ending,
            "presentation_ids": identifiers, "optimizer_updates": expected_updates,
            "checks": {name: True for name in REQUIRED_CHECKS}, "memory": memory_result,
            "parity": observed["parity"], "independent_ce_abs": observed["independent_ce"],
            "reload_max_abs_logits": reload_difference, "gradient_norms": observed["gradient_norms"],
            "last_metrics": trainer.last_metrics, "logits_observations": observed["logits"],
            "actual_sdpa_operators": sdpa, "saved_student": saved, "finite_parameter_count": finite_count,
            "reference_tensor_inventory_sha256": value_digest(reference_before), "output_files": outputs,
            "versions": {**VERSIONS, "python": sys.version, "cuda": torch.version.cuda},
            "gpu_name": torch.cuda.get_device_name(0), "gpu_identity": gpu_identity,
            "raw_witnesses_exported": False, "resume_supported": False, "training_pilot": False,
            "cpu_preparation_and_longest_proof_recomputed_here": False, "elapsed_seconds": time.monotonic() - begun}
        write_new_json(output / "complete.json", receipt)
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {"schema": "extension-native-model-admission-case-failure-v1", "complete": False,
            "case": case, "binding_sha256": context["binding_sha256"], "exception": type(error).__name__,
            "message": str(error)[:4000], "retry": False, "automatic_downstream": False,
            "memory_stages_observed": memory.stages, "allocator_peak_allocated_seen": memory.whole_allocated,
            "allocator_peak_reserved_seen": memory.whole_reserved,
            "max_observed_nonallocator_bytes": memory.max_observed_nonallocator_bytes})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", required=True, help="Project-relative separately frozen diagnostic plan")
    parser.add_argument("--prepared", required=True, help="Project-relative private CPU-prepared cache")
    parser.add_argument("--output", required=True, help="New exclusive diagnostic output directory")
    parser.add_argument("--qa-receipt", required=True, help="New native CPU QA receipt; prior230 does not qualify")
    parser.add_argument("--admission-receipt", required=True, help="Explicit current allocated job binding")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case is None:
        receipt = run(args.root, args.plan, args.prepared, args.output, args.qa_receipt, args.admission_receipt)
        print(json.dumps({"complete": True, "cases": len(receipt["case_receipts"]), "training_pilot": False}))
    else:
        allocation = require_gpu_allocation()
        context = load_inputs(args.root, args.plan, args.prepared, args.qa_receipt, args.admission_receipt)
        parent = Path(args.output).resolve()
        require(parent.is_relative_to(context["root"] / "extension/runs"), "Unsafe diagnostic parent output")
        started = read_json(parent / "started.json")
        require(started.get("binding_sha256") == context["binding_sha256"] and
            started.get("allocation", {}).get("job_id") == allocation["job_id"] and started.get("cases") == list(CASES) and
            not (parent / "failure.json").exists() and not (parent / "complete.json").exists(),
            "A worker requires its active immutable parent attempt")
        for previous in CASES[:CASES.index(args.case)]:
            directory = parent / "cases" / previous
            # Parent already verified those bytes and tracks their immutable
            # stat/hash snapshots. Re-reading every preceding 12GB checkpoint
            # in every child would create quadratic shared-storage traffic.
            validate_case_receipt(read_json(directory / "complete.json"), context, previous, directory, verify_outputs=False)
        require(not any((parent / "cases" / later).exists() for later in CASES[CASES.index(args.case) + 1:]),
            "A diagnostic worker cannot run out of the sealed case order")
        run_case(context, args.case, parent / "cases" / args.case, allocation)
        print(json.dumps({"complete": True, "case": args.case}))


if __name__ == "__main__":
    main()
