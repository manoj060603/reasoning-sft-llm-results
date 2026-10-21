"""Unadmitted, pure CPU calibration grading orchestration.

Inputs are bounded JSON BYTES already read by a future admitted outer loader.
This helper hashes those bytes against explicit prospective file pins before
parsing, validates every32x8 native completion before the first grading call,
then calls the unchanged source_evaluation_v2 worker sequentially. It performs
no file I/O, generation, model/tokenizer loading, ranking, selection or retry.
The outer loader still owns paths/permissions/current-job/runtime/source guards.

Raw rows, answers and grade receipts belong only in the returned PRIVATE payload.
Only the separately returned aggregate/hash summary is suitable for publication.
Calibration is engineering evidence, never method ranking or a reason to tune
sampling, parsing, worker deadlines or scientific thresholds after outcomes.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


SCHEMA = "extension-calibration-grading-v1"
INPUT_SCHEMA = "extension-calibration-grading-input-proof-v1"
CACHE_FILE_SHA256 = "6b9e9ff88a23aa73b9ea34cb4618d5858d0a23ef606e2892639fb953bbfede65"
GENERATION_COMPLETE_FILE_SHA256 = "2d2869b76b14af823c377d98ca524922b2ed1ed8d50e4282a4e826842de2c119"
GENERATION_JOB_ID = "1228872"
RECORD_NAMES = tuple("record-" + str(index).zfill(2) + ".private.json" for index in range(32))
MAX_CACHE_BYTES, MAX_COMPLETE_BYTES = 4 << 20, 2 << 20
MAX_RECORD_BYTES, MAX_ALL_RECORD_BYTES = 8 << 20, 128 << 20
COMPARATOR_SOURCES = {
    "utils.examples": "07aacef0144e11d7c9210beef2f4d1a9bf935d440bbb394e65a588eef861ce7b",
    "utils.grader": "43d858572b1dadda6fe3ec327db123bb5362fccd271ae58329c57f0c10521383",
    "utils.math_normalization": "9c0484a01b76bd4582bfe61c8ee55df375dbc09782f1b587fc21b21dd80e23a0",
    "utils.parser": "8d04ceb93e9ef50d61157fd1c8826a5cbf8d7f64cda85459f1166e19d8784208",
    "utils.utils": "4365ad38ea1c3c00926781684673e6dbc8f6e228d354a1f49f0d81cc0cfaff1e"}
COMPARATOR_VERSIONS = {"antlr4-python3-runtime": "4.7.2", "latex2sympy2": "1.9.1",
    "regex": "2025.11.3", "sympy": "1.14.0", "word2number": "1.1"}
CLAIMS = {"calibration_engineering_only": True, "method_ranking_permitted": False,
    "accuracy_driven_retuning_permitted": False, "intermediate_reasoning_verified": False,
    "generation_performed": False, "training_performed": False, "selection_performed": False,
    "automatic_retry": False, "automatic_downstream": False}


class CalibrationInputError(ValueError):
    pass


class CalibrationGradingAborted(RuntimeError):
    """The outer caller must persist .result privately, then stop without retry."""
    def __init__(self, result):
        self.result = deepcopy(result)
        self.completed_grades = deepcopy(result["private"]["observed_grades"])
        super().__init__("Calibration grading aborted; persist private result and stop without retry")


def _require(condition, message):
    if not condition:
        raise CalibrationInputError(message)


def file_sha256(payload):
    _require(type(payload) is bytes, "Supply immutable raw JSON bytes, never a path or parsed-file SHA claim")
    return hashlib.sha256(payload).hexdigest()


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Expected lowercase SHA256")
    return value


def _parse(payload, expected, maximum):
    _require(type(payload) is bytes and 0 < len(payload) <= maximum, "Raw JSON exceeds its explicit input bound")
    _require(file_sha256(payload) == _sha(expected), "Raw input FILE SHA differs from the prospective pin")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON input key")
            result[key] = value
        return result
    result = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(CalibrationInputError("Nonfinite JSON input")))
    _require(type(result) is dict, "Expected a JSON object")
    return result


def _apis():
    # Frozen callables only; imports neither load a tokenizer/model nor generate.
    from extension import source_evaluation_v2 as source
    from extension import source_generation_v3 as generation
    from extension.source_readiness_v2 import validate_comparator_result
    return source, generation, validate_comparator_result


def _comparator(expected, source):
    expected = source._expected_identity(expected)
    _require(expected["source_sha256"] == COMPARATOR_SOURCES and expected["runtime_versions"] == COMPARATOR_VERSIONS,
        "Keep the unchanged five comparator sources and pinned package versions")
    parents = set()
    for name, path in expected["loaded_modules"].items():
        parsed = PurePosixPath(path)
        _require(parsed.is_absolute() and ".." not in parsed.parts and parsed.name == name.split(".")[-1] + ".py",
            "Comparator paths must be explicit original module paths")
        parents.add(str(parsed.parent))
    _require(len(parents) == 1, "Comparator modules must share their declared source directory")
    return expected


def validate_inputs(cache_json_bytes, generation_complete_json_bytes, record_json_bytes, *,
                    expected_file_pins, expected_comparator, fixture=False):
    """Hash/parse/validate the ENTIRE inventory before any answer is graded.

    expected_file_pins has exactly cache_json_sha256/generation_complete_json_sha256.
    The pinned complete receipt supplies every authoritative record FILE SHA.
    Native pins are fixed above; synthetic differing pins require fixture=True,
    which also requires explicit fixture generation flags and later a fake grader.
    FILE and canonical VALUE SHA fields are deliberately distinct in the proof.
    """
    _require(type(fixture) is bool and type(expected_file_pins) is dict and set(expected_file_pins) ==
        {"cache_json_sha256", "generation_complete_json_sha256"}, "Explicit immutable prospective file pins required")
    pins = deepcopy(expected_file_pins)
    for sha in pins.values(): _sha(sha)
    if not fixture:
        _require(pins == {"cache_json_sha256": CACHE_FILE_SHA256,
            "generation_complete_json_sha256": GENERATION_COMPLETE_FILE_SHA256}, "Only the sealed native generation/cache files are supported")
    _require(type(record_json_bytes) is dict and set(record_json_bytes) == set(RECORD_NAMES) and
        all(type(payload) is bytes and 0 < len(payload) <= MAX_RECORD_BYTES for payload in record_json_bytes.values()) and
        sum(map(len, record_json_bytes.values())) <= MAX_ALL_RECORD_BYTES, "Exactly32 bounded immutable private record files required")
    raw_records = dict(record_json_bytes)  # Own immutable byte objects; no mutable caller dictionary alias.
    cache = _parse(cache_json_bytes, pins["cache_json_sha256"], MAX_CACHE_BYTES)
    complete = _parse(generation_complete_json_bytes, pins["generation_complete_json_sha256"], MAX_COMPLETE_BYTES)
    source, generation, _ = _apis()
    comparator = _comparator(expected_comparator, source)
    cache_proof = generation.validate_cache(cache)
    sources = complete.get("source_inventory", {})
    _require(type(sources) is dict and len(sources) == 128 and all(type(name) is str and name and
        not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and
        type(item) is dict and set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and 0 < item["bytes"] <= 1 << 20 and
        type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None
        for name, item in sources.items()), "Complete receipt must bind its exact source128 inventory")
    _require(complete.get("schema") == generation.SCHEMA and complete.get("complete") is True and
        complete.get("model_role") == "original" and complete.get("request_count") == 32 and complete.get("sample_count") == 256 and
        complete.get("cache_sha256") == pins["cache_json_sha256"] and complete.get("profile_sha256") == generation.PROFILE_SHA256 and
        complete.get("parent_complete_sha256") == generation.PARENT_PINS["complete.json"] and
        complete.get("model_identity_sha256") == cache_proof["model_identity_sha256"] and
        complete.get("tokenizer_identity_sha256") == cache_proof["tokenizer_identity_sha256"] and
        complete.get("plan_sha256") == sources.get(generation.PLAN, {}).get("sha256") and
        {generation.PLAN, "extension/source_generation_v3.py", "extension/gpu_identity_v2.py"} <= set(sources) and
        complete.get("native_interface_established") is (not fixture) and complete.get("fixture_factory_injected") is fixture and
        complete.get("input_snapshot_unchanged") is True and complete.get("raw_requests_outputs_private") is True and
        complete.get("engine_adapter") == generation.ENGINE_ADAPTER and
        all(complete.get(key) is False for key in ("grading", "ranking", "training", "automatic_retry", "automatic_downstream")),
        "Complete/native/source/model/tokenizer/profile/no-fixture generation proof differs")
    _require(fixture or complete.get("job_id") == GENERATION_JOB_ID, "Wrong sealed native generation allocation")
    inventory = complete.get("private_output_sha256")
    _require(type(inventory) is dict and set(name for name in inventory if re.fullmatch(r"record-[0-9]+\.private\.json", name)) == set(RECORD_NAMES),
        "Authoritative immutable output inventory must contain exactly these32 record files")
    for name, sha in inventory.items():
        _require(type(name) is str and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
            "Unsafe private manifest path")
        _sha(sha)
    requests = generation.build_requests(cache)
    records, engine_ids, record_files = [], set(), {}
    sample_fields = {"engine_output_position", "engine_output_index", "text", "token_ids", "finish_reason", "stop_reason",
        "sample_index", "output_tokens", "text_sha256", "token_ids_sha256"}
    record_fields = set(requests[0]) | {"schema", "engine_request_id", "finished", "samples", "native_timing", "native_metrics_type",
        "raw_engine_observation_sha256", "grading_performed"}
    for index, (name, request, prepared) in enumerate(zip(RECORD_NAMES, requests, cache["records"])):
        record = _parse(raw_records[name], inventory[name], MAX_RECORD_BYTES)
        source.validate_record(prepared["record"])
        _require(set(record) == record_fields and record.get("schema") == "extension-source-generation-record-v1" and
            all(record.get(key) == value for key, value in request.items()) and record.get("finished") is True and
            record.get("grading_performed") is False and type(record.get("engine_request_id")) is str and
            0 < len(record["engine_request_id"]) <= 1024 and record["engine_request_id"] not in engine_ids and
            type(record.get("native_timing")) is dict and type(record.get("native_metrics_type")) is str,
            "Record order/native prompt/seed/model/tokenizer/request identity differs")
        _sha(record.get("raw_engine_observation_sha256"))
        engine_ids.add(record["engine_request_id"])
        samples = record.get("samples")
        _require(type(samples) is list and len(samples) == 8 and [sample.get("sample_index") if type(sample) is dict else None
            for sample in samples] == list(range(8)), "All eight unique ordered sample indices are required")
        positions = []
        for sample_index, sample in enumerate(samples):
            _require(set(sample) == sample_fields and type(sample["sample_index"]) is int and
                type(sample["engine_output_index"]) is int and sample["engine_output_index"] == sample_index and
                type(sample["engine_output_position"]) is int and type(sample["text"]) is str and
                type(sample["token_ids"]) is list and len(sample["token_ids"]) <= 8192 and
                all(type(token) is int and 0 <= token < 151936 for token in sample["token_ids"]) and
                type(sample["output_tokens"]) is int and sample["output_tokens"] == len(sample["token_ids"]) and
                sample["text_sha256"] == hashlib.sha256(sample["text"].encode("utf-8")).hexdigest() and
                sample["token_ids_sha256"] == value_sha256(sample["token_ids"]), "Sample index/raw token/text hash metadata differs")
            finish, stop, tokens = sample["finish_reason"], sample["stop_reason"], sample["token_ids"]
            _require(finish in {"stop", "length"} and (stop is None or type(stop) is int and stop == 151645) and
                (finish != "length" or len(tokens) == 8192) and (finish != "stop" or tokens and tokens[-1] == 151645),
                "Aborted/unknown/shortened-length/non-native-stop generation cannot be graded as a complete batch")
            positions.append(sample["engine_output_position"])
        _require(sorted(positions) == list(range(8)), "Engine output positions must remain a unique permutation")
        record_files[name] = {"file_sha256": inventory[name], "value_sha256": value_sha256(record)}
        records.append(record)
    all_samples = [sample for record in records for sample in record["samples"]]
    _require(type(complete.get("length_finishes")) is int and complete["length_finishes"] == sum(sample["finish_reason"] == "length" for sample in all_samples) and
        type(complete.get("output_tokens")) is int and complete["output_tokens"] == sum(sample["output_tokens"] for sample in all_samples),
        "Complete receipt aggregate output/length counts differ")
    proof = {"schema": INPUT_SCHEMA, "cache_file_sha256": pins["cache_json_sha256"], "cache_value_sha256": value_sha256(cache),
        "generation_complete_file_sha256": pins["generation_complete_json_sha256"], "generation_complete_value_sha256": value_sha256(complete),
        "records": record_files, "record_count": 32, "sample_count": 256, "record_ids_sha256": cache_proof["record_ids_sha256"],
        "model_identity_sha256": cache_proof["model_identity_sha256"], "tokenizer_identity_sha256": cache_proof["tokenizer_identity_sha256"],
        "profile_sha256": generation.PROFILE_SHA256, "source_inventory_sha256": value_sha256(sources),
        "expected_comparator_sha256": value_sha256(comparator), "fixture_inputs": fixture,
        "physical_paths_permissions_current_job_runtime_verified_here": False}
    return {"cache": cache, "records": records, "complete": complete, "expected_comparator": comparator, "proof": proof, "fixture": fixture}


def _validate_grade(grade, row, sample, source, comparator, verify_comparator):
    _require(type(grade) is dict and grade.get("record_id") == row["id"] and grade.get("source_kind") == row["source_kind"] and
        type(grade.get("denominator")) is int and grade["denominator"] == 1 and type(grade.get("correct")) is bool and
        type(grade.get("rankable")) is bool and grade.get("response_sha256") == sample["text_sha256"] and
        grade.get("worker_protocol") == source.PROTOCOL and grade.get("caller_must_abort") is False,
        "Frozen grader receipt identity/denominator/protocol differs")
    if row["source_kind"] == "math" and grade.get("status") == "graded":
        verify_comparator(grade, comparator)
    # The frozen summarizer validates ordinary/error status semantics before a
    # grade can enter our aggregates. Do not duplicate/rewrite its status mapping.
    metrics = source.completion_metrics(sample["text"], sample["token_ids"], sample["finish_reason"], sample["stop_reason"], 151645)
    source.summarize_samples(row, [{"sample_index": index, "request_seed": source.sample_seed(row["id"]),
        "grade": grade, "metrics": metrics} for index in range(8)])
    return metrics


def _public_summary(private, *, complete):
    result = {"schema": SCHEMA, "complete": complete, "input_proof": deepcopy(private["input_proof"]),
        "fixture_grader_injected": private["fixture_grader_injected"], "frozen_grader_callable_used": not private["fixture_grader_injected"],
        "planned_record_count": 32, "planned_sample_count": 256, "observed_grade_count": len(private["observed_grades"]),
        "private_grades_value_sha256": value_sha256(private), "raw_records_answers_or_grades_included": False, **CLAIMS}
    if not complete:
        result.update(grading_aborted=True, message="Incomplete grading has no pass estimates")
        return result
    domains = {}
    for domain in ("math", "arc_challenge"):
        records = [row for row in private["record_summaries"] if row["source_kind"] == domain]
        samples = [sample for sample in private["observed_grades"] if sample["source_kind"] == domain]
        domains[domain] = {"record_count": len(records), "sample_count": len(samples), "denominator": len(samples),
            "correct_samples": sum(sample["grade"]["correct"] for sample in samples),
            "pass_at_1": math.fsum(row["pass_at_1"] for row in records) / len(records),
            "pass_at_8": math.fsum(row["pass_at_8"] for row in records) / len(records),
            "all_grades_rankable": all(sample["grade"]["rankable"] for sample in samples),
            "grading_errors": sum(not sample["grade"]["rankable"] for sample in samples),
            "status_counts": dict(sorted(Counter(sample["grade"]["status"] for sample in samples).items())),
            "length_finishes": sum(sample["metrics"]["length_finished"] for sample in samples),
            "stop_finishes": sum(sample["metrics"]["stop_finished"] for sample in samples),
            "native_eos_observed": sum(sample["metrics"]["native_eos_observed"] for sample in samples),
            "output_tokens": sum(sample["metrics"]["generated_tokens"] for sample in samples),
            "box_status_counts": dict(sorted(Counter(sample["metrics"]["box_status"] for sample in samples).items())),
            "stop_reason_counts": dict(sorted(Counter("none" if sample["metrics"]["stop_reason"] is None else "native_eos_id" for sample in samples).items())),
            "mean_repetition_8gram": math.fsum(sample["metrics"]["repetition_8gram"] for sample in samples) / len(samples)}
    result.update(domains=domains, repetition_definition="Frozen source_evaluation.completion_metrics: whitespace-word8grams; 1-unique/total, zero if no8grams",
        pass_estimate_definition="Per-domain mean over all16 records of correct_samples/8 and indicator(any correct); every errored sample retains denominator1")
    return result


def _safe_receipt(value):
    """A malformed injected receipt must not prevent private failure evidence."""
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if type(value) is dict and len(raw) <= 2 << 20:
            return json.loads(raw)
    except BaseException:
        pass
    return {"invalid_result_type": type(value).__name__, "private_observation_not_a_valid_grade": True}


def _abort(private, event, error):
    private["abort"] = deepcopy(event)
    result = {"private": deepcopy(private), "public": _public_summary(private, complete=False)}
    raise CalibrationGradingAborted(result) from error


def grade_calibration(cache_json_bytes, generation_complete_json_bytes, record_json_bytes, *,
                      expected_file_pins, expected_comparator, grader=None, fixture=False):
    """Return {private,public}; on abort, persist exception.result then stop.

    A supplied grader is permitted ONLY with explicit fixture=True. The default
    is the frozen source_evaluation_v2.grade_response, with fixed timeout5 and
    expected comparator identity passed on every call, in32-record/eight-index
    order. No thread/process pool, retry, replacement or response rewrite occurs.
    """
    _require((fixture and callable(grader)) or (not fixture and grader is None), "Fixture graders require explicit fixture inputs; native grading uses only the frozen callable")
    context = validate_inputs(cache_json_bytes, generation_complete_json_bytes, record_json_bytes,
        expected_file_pins=expected_file_pins, expected_comparator=expected_comparator, fixture=fixture)
    source, _, verify_comparator = _apis()
    grade = source.grade_response if grader is None else grader
    private = {"schema": SCHEMA, "input_proof": deepcopy(context["proof"]), "fixture_grader_injected": grader is not None,
        "observed_grades": [], "record_summaries": [], "raw_grades_answers_private": True, **CLAIMS}
    for prepared, record in zip(context["cache"]["records"], context["records"]):
        row, samples = prepared["record"], []
        for raw in record["samples"]:
            observation = {"record_id": row["id"], "source_kind": row["source_kind"], "sample_index": raw["sample_index"],
                "request_seed": record["request_seed"], "input_sample_value_sha256": value_sha256(raw), "receipt_validated": False}
            try:
                result = grade(deepcopy(row), raw["text"], timeout_seconds=5,
                    expected_comparator=deepcopy(context["expected_comparator"]))
            except source.SourceGraderCleanupError as error:
                observation["grade"] = _safe_receipt(error.result)
                observation["metrics"] = source.completion_metrics(raw["text"], raw["token_ids"], raw["finish_reason"], raw["stop_reason"], 151645)
                private["observed_grades"].append(observation)
                _abort(private, {"reason": "worker_cleanup_failed", "record_id": row["id"], "sample_index": raw["sample_index"], "result_persisted_in_exception": True}, error)
            except BaseException as error:
                _abort(private, {"reason": "unexpected_grader_exception", "record_id": row["id"], "sample_index": raw["sample_index"], "error_type": type(error).__name__}, error)
            observation["grade"] = _safe_receipt(result)
            private["observed_grades"].append(observation)
            try:
                _require(observation["grade"].get("private_observation_not_a_valid_grade") is not True,
                    "Grader receipt must be bounded finite plain JSON before aggregation")
                _require(not (type(result) is dict and (result.get("caller_must_abort") is True or result.get("status") == "cleanup_error" or
                    type(result.get("worker")) is dict and result["worker"].get("cleanup", {}).get("successful") is False)),
                    "A returned cleanup failure must also stop all subsequent grading")
                metrics = _validate_grade(result, row, raw, source, context["expected_comparator"], verify_comparator)
                observation["metrics"] = metrics
                observation["receipt_validated"] = True
                samples.append({"sample_index": raw["sample_index"], "request_seed": record["request_seed"], "grade": result, "metrics": metrics})
            except BaseException as error:
                _abort(private, {"reason": "invalid_or_unsafe_grader_receipt", "record_id": row["id"], "sample_index": raw["sample_index"], "error_type": type(error).__name__}, error)
        private["record_summaries"].append(source.summarize_samples(row, samples))
    _require(len(private["observed_grades"]) == 256 and len(private["record_summaries"]) == 32, "Complete grading inventory lost a denominator")
    return {"private": private, "public": _public_summary(private, complete=True)}
