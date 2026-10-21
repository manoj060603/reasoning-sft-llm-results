"""UNADMITTED original-only source calibration generator; no scoring or ranking.

The first separately authorized B200 run must establish the actual vLLM0.12
token-prompt interface. CPU fixture QA does not establish it. Only one batch of
the32 prepared requests is permitted; any partial/interface/identity failure is
preserved privately, without retry, refill or automatic downstream admission.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json, require, value_digest, write_new_json)
from extension.labeled_feasibility import inside
from extension import source_readiness_v2_retry as prepared_stage

native = prepared_stage.legacy.native
ROLE = "source_generation_original_calibration_20261021_v1"
PLAN = "extension/source_generation_plan.json"
PLAN_SCHEMA = "extension-source-calibration-generation-plan-v1"
QA_SCHEMA = "extension-source-calibration-generation-native-qa-v1"
ADMISSION_SCHEMA = "extension-source-calibration-generation-job-v1"
SCHEMA = "extension-source-calibration-generation-complete-v1"
PROFILE_SHA256 = "92a5854a906ccf89608fe941293c075446d98e3b2e463336f3c5f5272fc99ed0"
CACHE = {"path": "extension/runs/source-readiness-v2-retry-1226240/calibration_prompt_cache.json",
    "sha256": "6b9e9ff88a23aa73b9ea34cb4618d5858d0a23ef606e2892639fb953bbfede65"}
PARENT_DIRECTORY = "extension/report/source-readiness-v2-retry-20261021-1226240"
PARENT_PINS = {"complete.json": "3de54b64f58ba6b23bc1b22f65d8f85ee2cb34b127683052bf55522e1237ea7a",
    "unit_qa.json": "b45cc1fcc1dc1c19011274796142aa95381cd99b8c5021b4ba7e60cfd7ada1c1",
    "accounting.json": "8934ec8542d877f425b4abe8a44a9c0bf8c60345e72514d504a107f1a32cbcfa"}
SOURCE_PLAN = prepared_stage.LEGACY_PLAN
NEW_SOURCES = frozenset({PLAN, "extension/source_generation.py", "extension/source_generation.slurm", "tests/test_source_generation.py"})
METADATA_SOURCES = frozenset({"extension/joint_gpu_admission.py", "tests/test_joint_gpu_admission.py",
    "extension/joint_gpu_metadata_qa.py", "extension/joint_gpu_metadata_qa.slurm", "extension/source_generation_submission.py",
    "tests/test_source_generation_submission.py", "extension/source_generation_qa.py", "extension/source_generation_qa.slurm"})
RESOURCE = {"gpu_type": "B200", "gpus": 1, "host_cpus": 8, "host_ram_gib": 128,
    "wall_time_seconds": 7200, "reserve_seconds": 172800}
PROFILE = {"additional_stop_strings": [], "additional_stop_token_ids": [], "batch_size": 32, "dtype": "bfloat16",
    "frequency_penalty": 0.0, "generation_config": "vllm", "gpu_memory_utilization": .85, "hardware": "B200",
    "ignore_eos": False, "max_model_len": 16384, "max_new_tokens": 8192, "max_num_seqs": 128,
    "min_p": 0.0, "min_tokens": 0, "n": 8, "native_eos_token_id": 151645,
    "per_record_seed": "(20261021+int(record_id[:8],16)) modulo2**31", "presence_penalty": 0.0,
    "repetition_penalty": 1.0, "seed": 20261021, "status": "frozen_sampling_proposal_requires_separate_gpu_admission",
    "temperature": .6, "tensor_parallel_size": 1, "top_k": -1, "top_p": .95}
EOS_CONFIG_BYTES = b'{"eos_token_id":151645}\n'
EOS_CONFIG_SHA256 = "d5513acbf141afd5d7ddf7f8f0e382d48f7eba14e087fd866a93db14106d2d1e"
ENGINE_ADAPTER = {"schema": "extension-source-native-eos-directory-adapter-v1",
    "source_profile_generation_config": "vllm", "constructor_generation_config": "private_eos_only_directory",
    "relative_directory": "engine-config", "filename": "generation_config.json",
    "exact_utf8_json": EOS_CONFIG_BYTES.decode("utf-8"), "sha256": EOS_CONFIG_SHA256,
    "content": {"eos_token_id": 151645}, "sampling_parameters_unchanged": True,
    "expected_loader_metadata": {"transformers_version": "4.57.3"},
    "native_eos_only": True, "model_and_tokenizer_bytes_unchanged": True,
    "constructor_override_generation_config_used": False,
    "effective_cloned_sampling_parameters_verified_before_generate": True}


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Expected lowercase SHA256")
    return value


def _tokens(value, *, maximum=None):
    require(type(value) in (list, tuple) and all(type(token) is int and 0 <= token < 151936 for token in value),
        "Expected plain original-vocabulary token IDs")
    require(maximum is None or len(value) <= maximum, "Token sequence exceeds its frozen envelope")
    return list(value)


def validate_plan(plan):
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA and plan.get("role") == ROLE and
        plan.get("model_role") == "original" and plan.get("cache") == CACHE and plan.get("source_plan") == SOURCE_PLAN and
        plan.get("parent_evidence") == {name: {"path": PARENT_DIRECTORY + "/" + name, "sha256": sha} for name, sha in PARENT_PINS.items()} and
        plan.get("profile") == PROFILE and value_digest(PROFILE) == PROFILE_SHA256 and plan.get("engine_adapter") == ENGINE_ADAPTER and
        plan.get("resources") == RESOURCE and
        plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and plan.get("request_count") == 32 and
        plan.get("sample_count") == 256 and plan.get("native_eos_token_id") == 151645 and
        plan.get("one_generate_call") is True and plan.get("first_native_interface_check") is True and
        plan.get("native_qa") == {"schema": QA_SCHEMA, "collector_expected_tests": 32, "parent_tests_previously_passed": 478,
            "parent_tests_replayed": False,
            "test_pattern": "test_source_generation.py", "metadata_test_pattern": "test_source_generation_submission.py",
            "source_inventory_files": 95, "separate_metadata_source_files": 8, "cpu_only_fixtures": True,
            "actual_vllm_interface_established": False} and
        all(plan.get(key) is False for key in ("training", "grading", "ranking", "automatic_retry", "automatic_downstream")),
        "Only the original-only fixed32-request/256-sample calibration proposal is supported")
    return plan


def seal_engine_config(output):
    """Create an exclusive, exact EOS-only artifact after native run guards.

    The frozen profile's ``vllm`` selector means neutral sampling defaults. The
    public custom-directory selector is needed on v0.12 because its independent
    EOS loader still reads Qwen's secondary HF EOS under the literal selector.
    """
    directory = Path(output) / "engine-config"
    directory.mkdir(mode=0o700, exist_ok=False)
    path = directory / "generation_config.json"
    with path.open("xb") as stream:
        stream.write(EOS_CONFIG_BYTES)
    path.chmod(0o600)
    require(path.read_bytes() == EOS_CONFIG_BYTES and digest(path) == EOS_CONFIG_SHA256,
        "Fresh private EOS-only generation configuration differs")
    return {"directory": str(directory.resolve()), "filename": "generation_config.json",
        "sha256": EOS_CONFIG_SHA256, "content": {"eos_token_id": 151645}}


def validate_engine_config(config):
    """Verify the exact sealed artifact immediately before native initialization."""
    require(type(config) is dict and set(config) == {"directory", "filename", "sha256", "content"} and
        config["filename"] == "generation_config.json" and config["sha256"] == EOS_CONFIG_SHA256 and
        config["content"] == {"eos_token_id": 151645} and type(config["directory"]) is str,
        "Sealed generation adapter identity differs")
    directory = Path(config["directory"])
    path = directory / config["filename"]
    require(directory.is_absolute() and not directory.is_symlink() and directory.is_dir() and
        set(item.name for item in directory.iterdir()) == {"generation_config.json"} and not path.is_symlink() and
        path.is_file() and path.read_bytes() == EOS_CONFIG_BYTES and digest(path) == EOS_CONFIG_SHA256,
        "EOS-only private generation directory bytes differ")
    return str(directory)


def validate_cache(cache, plan=None):
    """Pure structural/provenance checks. Actual byte pins belong to load_inputs."""
    require(type(cache) is dict and cache.get("schema") == prepared_stage.CACHE_SCHEMA and cache.get("complete") is True and
        cache.get("plan_sha256") == prepared_stage.SCIENTIFIC_PLAN["sha256"] and
        cache.get("execution_plan_sha256") == "7526ff1b21dd2a8a53c39f47c91981ffe91496cd5f1270b9cddf50974cf3c39b" and
        cache.get("source_generation_profile") == PROFILE and cache.get("generation_performed") is False and
        cache.get("training_performed") is False and cache.get("raw_records_private") is True,
        "Complete unchanged CPU-prepared calibration cache is required")
    model, tokenizer = cache.get("model", {}), cache.get("tokenizer_identity", {})
    require(model.get("repository") == native.REPOSITORY and model.get("revision") == native.REVISION and
        model.get("files") == native.ORIGINAL_FILES and type(model.get("path")) is str and Path(model["path"]).is_absolute() and
        tokenizer.get("repository") == native.REPOSITORY and tokenizer.get("revision") == native.REVISION and
        tokenizer.get("files") == native.TOKENIZER_FILES and tokenizer.get("eos_token_id") == 151645,
        "Pinned original model/tokenizer identity differs")
    _sha(tokenizer.get("chat_template_sha256"))
    rows = cache.get("records")
    require(type(rows) is list and len(rows) == 32, "Exactly32 prepared records required; never drop/refill")
    ids, groups, domains = [], [], []
    for row in rows:
        require(type(row) is dict and set(row) == {"record", "prompt", "request_seed", "sample_indices"}, "Unexpected prepared row schema")
        record, prompt = row["record"], row["prompt"]
        identifier, group = _sha(record.get("id")), _sha(record.get("group_id"))
        ids.append(identifier)
        groups.append(group)
        domains.append(record.get("source_kind"))
        require(prompt.get("schema") == "source-native-prompt-v1" and prompt.get("record_id") == identifier and
            prompt.get("group_id") == group and prompt.get("source_kind") == record.get("source_kind") and
            row["request_seed"] == prepared_stage.source.sample_seed(identifier, 20261021) and row["sample_indices"] == list(range(8)),
            "Native prompt/seed/sample identity mismatch")
        tokens = _tokens(prompt.get("prompt_token_ids"), maximum=8192)
        proof = prompt.get("provenance", {})
        require(tokens and len(tokens) + 8192 <= 16384 and type(prompt.get("rendered_prompt")) is str and
            proof.get("prompt_token_ids_sha256") == value_digest(tokens) and
            proof.get("rendered_prompt_sha256") == prepared_stage.legacy.prepared_source.text_sha256(prompt["rendered_prompt"]) and
            proof.get("prompt_tokens") == len(tokens) and proof.get("tokenizer_identity") == tokenizer and
            proof.get("tokenizer_identity_sha256") == value_digest(tokenizer) and proof.get("gold_fields_used") is False and
            proof.get("truncation_performed") is False, "Prepared prompt token/native provenance differs")
    require(ids == sorted(ids) and len(set(ids)) == len(set(groups)) == 32 and
        dict(Counter(domains)) == {"math": 16, "arc_challenge": 16} and cache.get("calibration_ids_sha256") == value_digest(ids),
        "Calibration membership/order/domain/group seal differs")
    return {"record_count": 32, "sample_count": 256, "record_ids_sha256": value_digest(ids),
        "model_identity_sha256": value_digest(model), "tokenizer_identity_sha256": value_digest(tokenizer)}


def build_requests(cache):
    proof = validate_cache(cache)
    return [{"record_id": row["record"]["id"], "source_kind": row["record"]["source_kind"],
        "prompt_token_ids": list(row["prompt"]["prompt_token_ids"]), "request_seed": row["request_seed"],
        "prompt_sha256": row["prompt"]["provenance"]["rendered_prompt_sha256"],
        "prompt_token_ids_sha256": row["prompt"]["provenance"]["prompt_token_ids_sha256"],
        "model_identity_sha256": proof["model_identity_sha256"], "tokenizer_identity_sha256": proof["tokenizer_identity_sha256"]}
        for row in cache["records"]]


def snapshot_output(output):
    """Private raw-observation snapshot, collected before success validation."""
    def scalar(value):
        return value if value is None or type(value) in (str, int, bool) or type(value) is float and math.isfinite(value) else {"invalid_type": type(value).__qualname__}
    def tokens(value):
        return [scalar(item) for item in value] if type(value) in (list, tuple) else {"invalid_type": type(value).__qualname__}
    try:
        metrics = getattr(output, "metrics", None)
        timing = {key: scalar(getattr(metrics, key, None)) for key in ("arrival_time", "first_scheduled_time", "first_token_time",
            "last_token_time", "finished_time", "scheduler_time", "model_forward_time", "model_execute_time", "time_in_queue")}
        outputs = getattr(output, "outputs", None)
        observations = [{"engine_output_position": position,
            "engine_output_index": scalar(getattr(item, "index", None)), "text": scalar(getattr(item, "text", None)),
            "token_ids": tokens(getattr(item, "token_ids", None)), "finish_reason": scalar(getattr(item, "finish_reason", None)),
            "stop_reason": scalar(getattr(item, "stop_reason", None))} for position, item in enumerate(outputs)] if type(outputs) in (list, tuple) else {"invalid_type": type(outputs).__qualname__}
        return {"engine_request_id": scalar(getattr(output, "request_id", None)), "finished": scalar(getattr(output, "finished", None)),
            "prompt_token_ids": tokens(getattr(output, "prompt_token_ids", None)), "native_metrics_type": type(metrics).__qualname__,
            "observed_native_timing_fields": timing, "outputs": observations}
    except BaseException as error:
        return {"snapshot_error": type(error).__name__, "observed_type": type(output).__qualname__}


def collect_record(request, output):
    raw = snapshot_output(output)
    require("snapshot_error" not in raw and type(raw.get("outputs")) is list, "Malformed native request output interface")
    require(type(raw["engine_request_id"]) is str and raw["engine_request_id"] and len(raw["engine_request_id"]) <= 1024 and
        raw["finished"] is True and _tokens(raw["prompt_token_ids"], maximum=8192) == request["prompt_token_ids"] and
        value_digest(raw["prompt_token_ids"]) == request["prompt_token_ids_sha256"], "Engine request/prompt/finished proof differs")
    samples = []
    for item in raw["outputs"]:
        index, finish, stop = item["engine_output_index"], item["finish_reason"], item["stop_reason"]
        tokens = _tokens(item["token_ids"], maximum=8192)
        require(type(index) is int and 0 <= index < 8 and type(item["text"]) is str and type(finish) is str and finish in {"stop", "length"} and
            (stop is None or type(stop) is int and stop == 151645), "Missing/aborted/unknown engine completion or additional stop")
        require(finish != "length" or len(tokens) == 8192, "Unexpected shortened length finish")
        require(finish != "stop" or tokens and tokens[-1] == 151645, "Native EOS completion lacks its terminal token")
        samples.append({**item, "sample_index": index, "output_tokens": len(tokens), "token_ids": tokens,
            "text_sha256": prepared_stage.legacy.prepared_source.text_sha256(item["text"]), "token_ids_sha256": value_digest(tokens)})
    require(len(samples) == 8 and {row["sample_index"] for row in samples} == set(range(8)), "Missing/duplicate eight-sample indices")
    return {"schema": "extension-source-generation-record-v1", **deepcopy(request), "engine_request_id": raw["engine_request_id"],
        "finished": True, "samples": sorted(samples, key=lambda row: row["sample_index"]),
        "native_timing": raw["observed_native_timing_fields"], "native_metrics_type": raw["native_metrics_type"],
        "raw_engine_observation_sha256": value_digest(raw), "grading_performed": False}


def source_inventory(root):
    root = Path(root).resolve()
    prior = prepared_stage.source_inventory(root)
    complete_path = inside(root, PARENT_DIRECTORY + "/complete.json")
    require(digest(complete_path) == PARENT_PINS["complete.json"] and read_json(complete_path).get("source_inventory") == prior,
        "Preserve every one of the91 CPU-tested source bytes")
    result = dict(prior)
    for name in sorted(NEW_SOURCES):
        result[name] = prepared_stage.joint._record(root, name)
    require(len(result) == 95, "The separately bound95-source generation inventory is required")
    return result


def _current_admission(root, allocation, sources, plan_sha256, qa_sha256):
    ledger = read_json(Path(root) / "extension/job_ledger.json")
    require(ledger.get("schema") == prepared_stage.joint.previous.SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict,
        "Separate extension ledger identity differs")
    entry = ledger["jobs"].get(ROLE, {})
    binding = entry.get("binding", {})
    require(entry.get("status") == "submitted" and entry.get("job_id") == allocation["job_id"] and
        entry.get("source_sha256") == sources == binding.get("source_sha256") and binding.get("role") == ROLE and
        binding.get("plan_sha256") == plan_sha256 and binding.get("qa_sha256") == qa_sha256 and
        binding.get("cache_sha256") == CACHE["sha256"] and binding.get("profile_sha256") == PROFILE_SHA256 and
        binding.get("parent_complete_sha256") == PARENT_PINS["complete.json"] and binding.get("model_role") == "original" and
        binding.get("authorize_first_interface_checking_generation") is True and binding.get("requested_gpu_seconds") == 7200 and
        entry.get("requested_gpu_seconds") == 7200 and entry.get("requested_gpu_hours") == 2 and
        binding.get("wall_time_seconds") == 7200 and binding.get("closeout_reserve_seconds") == 172800 and
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False and
        entry.get("binding_sha256") == value_digest(binding), "New current-job original-only first-interface admission is required")
    return entry


def load_inputs(root, plan_path, qa_receipt, admission_receipt, allocation):
    """Called only after real single-B200/host/deadline guards, before heavy imports."""
    root, evidence = Path(root).resolve(), Evidence()
    plan_path, qa_path, admission_path = (inside(root, str(path)) for path in (plan_path, qa_receipt, admission_receipt))
    plan = validate_plan(evidence.read(plan_path))
    sources = source_inventory(root)
    qa, admission = evidence.read(qa_path), evidence.read(admission_path)
    counts = qa.get("test_counts", {})
    expected_counts = {"test_source_generation.py": 32}
    require(type(counts) is dict and set(counts) == set(expected_counts) | {"test_source_generation_submission.py"} and
        all(type(counts.get(name)) is int and counts[name] == expected for name, expected in expected_counts.items()) and
        type(counts["test_source_generation_submission.py"]) is int and counts["test_source_generation_submission.py"] > 0,
        "Native QA must cover collector32 and separately frozen metadata tests; prior478 is not replayed")
    require(qa.get("schema") == QA_SCHEMA and qa.get("success") is True and qa.get("gpu_disabled") is True and
        type(qa.get("tests_run")) is int and qa["tests_run"] == sum(counts.values()) and
        all(type(qa.get(key)) is int and qa[key] == 0 for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("plan_sha256") == evidence.check(plan_path) and qa.get("source_inventory") == sources and
        qa.get("code_sha256") == {name: item["sha256"] for name, item in sources.items()} and
        qa.get("cache_sha256") == CACHE["sha256"] and qa.get("parent_complete_sha256") == PARENT_PINS["complete.json"] and
        qa.get("wrapper_syntax_passed") is True and qa.get("actual_vllm_interface_established") is False and
        qa.get("parent_tests_previously_passed") == 478 and qa.get("parent_tests_replayed") is False and qa.get("collector_tests") == 32 and
        qa.get("metadata_tests") == counts["test_source_generation_submission.py"] and
        qa.get("compute_proof", {}).get("job_id") == qa.get("job_id") and qa.get("job_id") is not None and
        qa["compute_proof"].get("node") == qa.get("node") and qa["compute_proof"].get("allocated_cpus") == 2 and
        qa["compute_proof"].get("gpu_requested_or_allocated") is False,
        "Separate source-bound native CPU fixtures are required; they do not establish real vLLM inference")
    metadata_sources = qa.get("metadata_source_inventory", {})
    require(type(metadata_sources) is dict and set(metadata_sources) == METADATA_SOURCES,
        "Bind the separately tested eight metadata-admission sources without changing core95")
    for name, item in metadata_sources.items():
        path = inside(root, name)
        require(type(item) is dict and set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and
            0 < item["bytes"] <= 1 << 20 and path.stat().st_size == item["bytes"], "Invalid bounded metadata source identity")
        evidence.check(path, _sha(item["sha256"]))
    current = _current_admission(root, allocation, sources, evidence.check(plan_path), evidence.check(qa_path))
    require(admission.get("schema") == ADMISSION_SCHEMA and admission.get("status") == "admitted" and
        admission.get("job_id") == allocation["job_id"] and admission.get("plan_sha256") == evidence.check(plan_path) and
        admission.get("qa_sha256") == evidence.check(qa_path) and admission.get("code_sha256") == qa["code_sha256"] and
        admission.get("binding_sha256") == current["binding_sha256"] and admission.get("cache_sha256") == CACHE["sha256"] and
        admission.get("authorize_first_interface_checking_generation") is True and admission.get("automatic_downstream") is False,
        "A separate exact-source current GPU handoff is required")
    parents = {name: evidence.read(inside(root, PARENT_DIRECTORY + "/" + name), sha) for name, sha in PARENT_PINS.items()}
    parent = parents["complete.json"]
    require(parent.get("complete") is True and parent.get("job_id") == "1226240" and parent.get("tests_run") == 478 and
        parent.get("gpu_seconds") == 0 and parent.get("gold_self_comparisons_passed") == 32 and
        parent.get("native_gold_mutation_invariance_passed") is True and parent.get("input_snapshot_unchanged") is True and
        parent.get("native_qa_sha256") == PARENT_PINS["unit_qa.json"] and
        parent.get("output_sha256", {}).get("calibration_prompt_cache.json") == CACHE["sha256"], "The complete fixed-cohort CPU preparation is required")
    from extension.joint_gpu_admission import validate_accounting
    validate_accounting(parents["accounting.json"], "1226240", name="sft-extension-source-readiness-v2-calibration-retry")
    old = evidence.read(inside(root, SOURCE_PLAN["path"]), SOURCE_PLAN["sha256"])
    require(old.get("source_generation_profile") == PROFILE, "Frozen source sampling profile changed")
    for package, expected in old["runtime_versions"].items():
        require(importlib.metadata.version(package) == expected, "Unpinned native generation runtime: " + package)
    cache = evidence.read(inside(root, CACHE["path"]), CACHE["sha256"])
    proof = validate_cache(cache, plan)
    require(cache.get("code_sha256") == {name: item["sha256"] for name, item in parent["source_inventory"].items()} and
        cache.get("model") == {"repository": native.REPOSITORY, "revision": native.REVISION, **old["base"]}, "Prepared cache/model/CPU source identity mismatch")
    base = prepared_stage.legacy.prepared_source._validate_base(root, old["base"], evidence)
    for name, item in sources.items():
        evidence.check(inside(root, name), item["sha256"])
    evidence.unchanged()
    return {"root": root, "plan": plan, "plan_sha256": evidence.check(plan_path), "cache": cache, "cache_proof": proof,
        "requests": build_requests(cache), "base": base, "sources": sources, "qa_sha256": evidence.check(qa_path),
        "admission": current, "admission_sha256": evidence.check(admission_path), "evidence": evidence,
        "versions": {name: importlib.metadata.version(name) for name in old["runtime_versions"]}}


def replay_prompts(context):
    tokenizer = prepared_stage.legacy.prepared_source._load_tokenizer(context["base"])
    identity = context["cache"]["tokenizer_identity"]
    for row in context["cache"]["records"]:
        replay = prepared_stage.source.serialize_prompt(row["record"], tokenizer, identity, max_prompt_tokens=8192)
        require(replay == row["prompt"], "Actual original tokenizer/native prompt replay differs")
    return {"requests_replayed": 32, "native_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_identity_sha256": value_digest(identity), "labels_used_for_generation": False}


@contextmanager
def private_engine_logs(output):
    """Redirect inherited OS descriptors as well as Python streams for workers."""
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with (output / "engine.stdout.private.log").open("x", encoding="utf-8") as stdout, \
                (output / "engine.stderr.private.log").open("x", encoding="utf-8") as stderr:
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


ENGINE_STATE_FIELDS = ("llm_args", "hardware", "installed_sources", "sampling_proof",
    "effective_sampling_proof", "native_generation_fields", "actual_model_configuration")


class NativeInterfaceError(RuntimeError):
    """Carry safe initialization observations into the private failure journal."""
    def __init__(self, state):
        super().__init__("Pinned native engine initialization failed")
        self.native_engine_state = deepcopy(state)


class NativeEngine:
    """Pinned0.12 adapter. Construct only inside guarded/private native run."""
    def __init__(self, config):
        generation_directory = validate_engine_config(config.get("generation_config"))
        import torch
        import vllm.outputs
        import vllm.sampling_params
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        require(importlib.metadata.version("vllm") == "0.12.0" and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.is_bf16_supported() and "B200" in torch.cuda.get_device_name(0), "The actual original inference runtime/device differs")
        require({"prompts", "sampling_params", "use_tqdm"} <= set(inspect.signature(LLM.generate).parameters), "Unrecognized pinned vLLM generation interface")
        self.torch, self.config, self.SamplingParams, self.final_kind = torch, deepcopy(config), SamplingParams, RequestOutputKind.FINAL_ONLY
        self.hardware = native.assigned_gpu_identity(torch)
        self.before = self.memory()
        self.llm_args = {"model": config["base_path"], "tokenizer": config["base_path"], "revision": native.REVISION,
            "tokenizer_revision": native.REVISION, "dtype": "bfloat16", "max_model_len": 16384, "seed": 20261021,
            "tensor_parallel_size": 1, "gpu_memory_utilization": .85, "trust_remote_code": False,
            "generation_config": generation_directory, "max_num_seqs": 128}
        try:
            self.llm = LLM(**self.llm_args)
            tokenizer, model = self.llm.get_tokenizer(), self.llm.model_config
            self.native_generation_fields = deepcopy(model.try_get_generation_config())
            self.actual_model_configuration = {"model": str(model.model), "dtype": str(model.dtype),
                "max_model_len": model.max_model_len, "generation_config": str(model.generation_config),
                "diff_sampling_parameters": model.get_diff_sampling_param(), "quantization": model.quantization,
                "tokenizer_eos_token_id": tokenizer.eos_token_id,
                "chat_template_sha256": prepared_stage.legacy.prepared_source.text_sha256(tokenizer.chat_template),
                "input_processor_generation_config_fields": deepcopy(self.llm.input_processor.generation_config_fields)}
            self.installed_sources = {str(Path(inspect.getfile(item)).resolve()): digest(Path(inspect.getfile(item)))
                for item in (LLM, SamplingParams, vllm.outputs.RequestOutput, type(model), type(self.llm.input_processor))}
            require(tokenizer.eos_token_id == 151645 and self.actual_model_configuration["chat_template_sha256"] ==
                config["tokenizer_identity"]["chat_template_sha256"] and Path(model.model).resolve() == Path(config["base_path"]).resolve() and
                str(model.dtype) == "torch.bfloat16" and model.max_model_len == 16384 and
                Path(model.generation_config).resolve() == Path(generation_directory).resolve() and
                self.actual_model_configuration["diff_sampling_parameters"] == {} and model.quantization is None,
                "Actual vLLM model/tokenizer/default-generation configuration differs")
            require(self.native_generation_fields == {"eos_token_id": 151645, "transformers_version": "4.57.3"} and
                self.llm.input_processor.generation_config_fields == self.native_generation_fields and
                validate_engine_config(config["generation_config"]) == generation_directory,
                "Actual public EOS-only generation directory was not loaded unchanged")
        except BaseException as error:
            raise NativeInterfaceError({name: getattr(self, name, None) for name in ENGINE_STATE_FIELDS}) from error

    def memory(self):
        self.torch.cuda.synchronize()
        free, total = self.torch.cuda.mem_get_info(0)
        return {"sampled_driver_used_bytes": total - free, "total_bytes": total,
            "parent_allocated_bytes": self.torch.cuda.memory_allocated(0), "parent_reserved_bytes": self.torch.cuda.memory_reserved(0),
            "parent_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
            "qualification": "Driver usage is sampled; parent allocator peaks exclude allocations in vLLM worker processes."}

    def generate(self, requests):
        require(len(requests) == 32, "Never change the frozen one32-request batch")
        prompts = [{"prompt_token_ids": list(row["prompt_token_ids"])} for row in requests]
        parameters = [self.SamplingParams(n=8, temperature=.6, top_p=.95, top_k=-1, min_p=0.,
            presence_penalty=0., frequency_penalty=0., repetition_penalty=1., max_tokens=8192, min_tokens=0,
            seed=row["request_seed"], stop=[], stop_token_ids=[], ignore_eos=False, truncate_prompt_tokens=None,
            output_kind=self.final_kind, detokenize=True) for row in requests]
        self.sampling_proof = [{key: getattr(parameter, key) for key in ("n", "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "frequency_penalty", "repetition_penalty", "max_tokens", "min_tokens", "seed", "stop",
            "stop_token_ids", "ignore_eos", "truncate_prompt_tokens", "detokenize")} for parameter in parameters]
        # v0.12 InputProcessor clones parameters and separately merges native
        # generation-config EOS fields. A documented custom directory must be
        # checked through that exact effective operation before
        # allowing the sole real generate call. Never silently accept151643.
        fields = deepcopy(self.llm.input_processor.generation_config_fields)
        self.native_generation_fields = fields
        self.effective_sampling_proof = []
        for parameter in parameters:
            effective = parameter.clone()
            effective.update_from_generation_config(fields, 151645)
            self.effective_sampling_proof.append({"seed": effective.seed, "n": effective.n,
                "native_eos_token_id": 151645, "all_stop_token_ids": sorted(effective.all_stop_token_ids),
                "stop_token_ids": list(effective.stop_token_ids), "output_kind": effective.output_kind.name})
            require(effective.stop == [] and effective.stop_token_ids == [] and
                effective.all_stop_token_ids == {151645} and effective.ignore_eos is False and
                effective.truncate_prompt_tokens is None and effective.output_kind == self.final_kind,
                "Pinned vLLM effective EOS policy differs; no generation is permitted")
        self.before_generate = self.memory()
        started = time.monotonic()
        outputs = self.llm.generate(prompts, parameters, use_tqdm=False)
        self.elapsed = time.monotonic() - started
        self.after_generate = self.memory()
        return outputs

    def proof(self):
        return {"native_engine": True, "vllm_version": "0.12.0", "constructor": self.llm_args, "hardware": self.hardware,
            "engine_adapter": deepcopy(ENGINE_ADAPTER), "sealed_generation_config": deepcopy(self.config["generation_config"]),
            "actual_model_configuration": self.actual_model_configuration,
            "installed_sources_sha256": self.installed_sources, "sampling_parameters": self.sampling_proof,
            "effective_sampling_parameters": self.effective_sampling_proof, "native_generation_fields": self.native_generation_fields,
            "sampling_parameters_sha256": value_digest(self.sampling_proof), "one_generate_call": True, "request_count": 32,
            "native_eos_token_id": 151645, "labels_entered_engine": False, "model_loaded_from_verified_bytes": True,
            "memory": {"before_engine": self.before, "before_generate": self.before_generate, "after_generate": self.after_generate},
            "generate_elapsed_seconds": self.elapsed}


def run(root, plan_path, output, qa_receipt, admission_receipt, *, engine_factory=None):
    allocation = native.require_gpu_allocation()
    native._host_resources(allocation, RESOURCE)
    deadline_guard(180000)
    root = Path(root).resolve()
    proposed = Path(output)
    require(not proposed.is_symlink(), "Output aliases are forbidden")
    output = proposed.resolve()
    require(root == Path(__file__).resolve().parents[1] and output.parent == root / "extension/runs" and
        output.name == "source-generation-original-" + allocation["job_id"] and not output.exists(), "Exclusive current-job generation output required; no resume/retry")
    context = load_inputs(root, plan_path, qa_receipt, admission_receipt, allocation)
    old_umask = os.umask(0o077)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    generated, collected, engine = [], [], None
    try:
        write_new_json(output / "requests.private.json", {"schema": "extension-source-generation-requests-v1", "requests": context["requests"],
            "cache_sha256": CACHE["sha256"], "plan_sha256": context["plan_sha256"], "no_labels_in_requests": True})
        write_new_json(output / "started.json", {"schema": SCHEMA, "complete": False, "job_id": allocation["job_id"],
            "plan_sha256": context["plan_sha256"], "qa_sha256": context["qa_sha256"], "admission_sha256": context["admission_sha256"],
            "source_inventory": context["sources"], "request_count": 32, "sample_count": 256, "automatic_retry": False, "automatic_downstream": False})
        with private_engine_logs(output):
            replay = replay_prompts(context)
            context["evidence"].unchanged()
            require(_current_admission(root, allocation, context["sources"], context["plan_sha256"], context["qa_sha256"]) == context["admission"],
                "Current generation admission changed before model initialization")
            config = {"base_path": str(context["base"]), "repository": native.REPOSITORY, "revision": native.REVISION,
                "model_files": native.ORIGINAL_FILES, "tokenizer_identity": deepcopy(context["cache"]["tokenizer_identity"]),
                "profile": deepcopy(PROFILE), "generation_config": seal_engine_config(output)}
            engine = (NativeEngine if engine_factory is None else engine_factory)(config)
            generated = engine.generate(deepcopy(context["requests"]))
            require(type(generated) is list, "Pinned generation interface must return a list")
            write_new_json(output / "engine_outputs.private.json", [snapshot_output(row) for row in generated])
            require(len(generated) == 32, "Missing/extra request outputs; never refill or retry")
            seen = set()
            for index, (request, observed) in enumerate(zip(context["requests"], generated)):
                record = collect_record(request, observed)
                require(record["engine_request_id"] not in seen, "Duplicate opaque engine request identity")
                seen.add(record["engine_request_id"])
                write_new_json(output / ("record-" + str(index).zfill(2) + ".private.json"), record)
                collected.append(record)
            engine_proof = engine.proof()
            require(engine_proof.get("one_generate_call") is True and engine_proof.get("request_count") == 32 and
                engine_proof.get("native_eos_token_id") == 151645 and engine_proof.get("labels_entered_engine") is False,
                "Native adapter provenance is incomplete")
            write_new_json(output / "engine_proof.private.json", engine_proof)
        ending = native.require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Allocation changed")
        context["evidence"].unchanged()
        require(source_inventory(root) == context["sources"] and
            _current_admission(root, allocation, context["sources"], context["plan_sha256"], context["qa_sha256"]) == context["admission"],
            "Input/source/current-admission identity changed during generation")
        samples = [sample for row in collected for sample in row["samples"]]
        require(validate_engine_config(config["generation_config"]) == config["generation_config"]["directory"],
            "Runtime EOS configuration changed during generation")
        complete = {"schema": SCHEMA, "complete": True, "job_id": allocation["job_id"], "model_role": "original",
            "request_count": 32, "sample_count": len(samples), "length_finishes": sum(row["finish_reason"] == "length" for row in samples),
            "output_tokens": sum(row["output_tokens"] for row in samples), "plan_sha256": context["plan_sha256"],
            "cache_sha256": CACHE["sha256"], "parent_complete_sha256": PARENT_PINS["complete.json"], "profile_sha256": PROFILE_SHA256,
            "model_identity_sha256": context["cache_proof"]["model_identity_sha256"], "tokenizer_identity_sha256": context["cache_proof"]["tokenizer_identity_sha256"],
            "qa_sha256": context["qa_sha256"], "admission_sha256": context["admission_sha256"], "source_inventory": context["sources"],
            "native_interface_established": engine_proof.get("native_engine") is True and engine_factory is None,
            "fixture_factory_injected": engine_factory is not None, "native_prompt_replay": replay, "input_snapshot_unchanged": True,
            "engine_adapter": deepcopy(ENGINE_ADAPTER), "sealed_generation_config": deepcopy(config["generation_config"]),
            "compute_start": allocation, "compute_end": ending, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "private_output_sha256": {path.relative_to(output).as_posix(): digest(path)
                for path in sorted(output.rglob("*.json")) if path.is_file()},
            "unsealed_private_logs": {name: "Worker descriptors may still append; seal only after terminal external closeout."
                for name in ("engine.stdout.private.log", "engine.stderr.private.log")},
            "raw_requests_outputs_private": True, "grading": False, "ranking": False, "training": False,
            "automatic_retry": False, "automatic_downstream": False}
        write_new_json(output / "complete.json", complete)
        return complete
    except BaseException as error:
        write_new_json(output / "failure.private.json", {"exception": type(error).__name__, "traceback": traceback.format_exc(),
            "observed_request_outputs": [snapshot_output(row) for row in generated] if type(generated) is list else None,
            "native_engine_state": getattr(error, "native_engine_state", None) or
                {name: getattr(engine, name, None) for name in ENGINE_STATE_FIELDS},
            "validated_record_count": len(collected), "raw_evidence_private": True})
        write_new_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "job_id": allocation["job_id"],
            "exception": type(error).__name__, "validated_record_count": len(collected), "automatic_retry": False,
            "automatic_downstream": False, "no_generation_refill_or_replay": True})
        raise
    finally:
        os.umask(old_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    parser.add_argument("--admission-receipt", required=True)
    args = parser.parse_args(argv)
    try:
        complete = run(args.root, args.plan, args.output, args.qa_receipt, args.admission_receipt)
    except BaseException:
        raise RuntimeError("Source calibration generation failed. See private failure evidence.") from None
    print({key: complete[key] for key in ("complete", "request_count", "sample_count", "length_finishes", "output_tokens")}, flush=True)


if __name__ == "__main__":
    main()
