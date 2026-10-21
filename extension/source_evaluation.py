"""Pure, source-only prompt and scoring contracts for the sealed MATH/ARC cohorts.

These helpers do not load data, a tokenizer, model weights or generation outputs.
The caller supplies already verified native objects and keeps returned questions,
gold values and responses private. Prompt construction deliberately validates only
question/option fields; it never consults solution or answer-key fields.

This is a new source-development proxy, not the historical benchmark grader.
MATH retains the pinned repository comparator's numeric tolerance and symbolic
heuristics (including its limitations for ordered tuples and units). Explicit
scalar percentages are normalized exactly BEFORE its percentage-stripping
normalizer, and its optional 100x percentage equivalence is disabled. Unsupported
gold percentage syntax blocks readiness; no question is dropped or replaced.

The comparator runs in a fresh process with an external hard work deadline. Its
internal timeout is disabled to avoid nested subprocess/Queue.get crash hangs.
Timeout, crash and implementation errors retain their denominator and prevent
ranking. Cleanup adds a bounded grace period to the requested work deadline.
No experiment, calibration iteration, cohort selection or GPU admission occurs.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import contextlib
import importlib
import importlib.machinery
import importlib.metadata
import math
import multiprocessing
from pathlib import Path
import re
import sys
import time
import types

from extension.labeled_feasibility import source_record
from extension.preserving_data import text_sha256, value_sha256
from reasoning.data import SYSTEM, prompt_id


ARC_INSTRUCTION = r"Select one original option label and put that label within \boxed{}."
SAMPLE_COUNT = 8
GLOBAL_SEED = 20261021
_SHA = re.compile(r"[0-9a-f]{64}")
_DECIMAL = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_SCALAR = r"[+-]?\s*" + _DECIMAL
_PACKAGES = ("sympy", "antlr4-python3-runtime", "latex2sympy2", "regex", "word2number")
_GRADER_FILES = {"utils.grader": "grader.py", "utils.parser": "parser.py",
    "utils.math_normalization": "math_normalization.py", "utils.utils": "utils.py",
    "utils.examples": "examples.py"}


class SourceEvaluationError(ValueError):
    """A supplied source record, native prompt or completion inventory is invalid."""


def _require(condition, message):
    if not condition:
        raise SourceEvaluationError(message)


def _sha(value):
    _require(type(value) is str and _SHA.fullmatch(value) is not None,
             "Expected a lowercase SHA256 identity")
    return value


def _question_fields(row):
    """Check question identity without accessing any gold field or gold hash."""
    _require(type(row) is dict, "Labeled record must be an object")
    identifier, group = _sha(row.get("id")), _sha(row.get("group_id"))
    kind = row.get("source_kind")
    _require(kind in ("math", "arc_challenge"), "Unsupported source kind")
    question, raw, origin = row.get("question"), row.get("original_record"), row.get("source")
    _require(type(question) is str and bool(question.strip()) and type(raw) is dict and
             type(origin) is dict and bool(origin), "Question and source identity are required")
    value_sha256(origin)  # finite plain JSON; this does not prove origin bytes
    _require(row.get("question_sha256") == prompt_id(question) and
             identifier == value_sha256({"source": origin, "question_sha256": prompt_id(question)}),
             "Question/source identity differs")
    if kind == "math":
        _require(raw.get("problem") == question, "Original MATH question bytes differ")
        user = question
    else:
        _require(raw.get("question") == question, "Original ARC question bytes differ")
        choices = raw.get("choices")
        _require(type(choices) is dict and set(choices) == {"label", "text"} and
            type(choices["label"]) is list and type(choices["text"]) is list and
            len(choices["label"]) == len(choices["text"]) >= 2 and
            all(type(item) is str and bool(item.strip()) for item in choices["label"] + choices["text"]) and
            len(set(choices["label"])) == len(choices["label"]), "Original ordered ARC choices are invalid")
        user = question + "\n\n" + "\n".join(label + ". " + text for label, text in
            zip(choices["label"], choices["text"])) + "\n\n" + ARC_INSTRUCTION
    return identifier, group, kind, user


def validate_record(row):
    """Validate the complete sealed row, including gold binding, without mutation.

    Group membership/disjointness and source-file hashes belong to the sealed
    cohort audit. A single row cannot prove the complete connected component.
    """
    _question_fields(row)
    try:
        rebuilt, quarantine = source_record(row["source_kind"], row["original_record"], row["source"])
    except (KeyError, TypeError, ValueError) as error:
        raise SourceEvaluationError("Original labeled row schema is invalid") from error
    _require(quarantine is None and row == dict(rebuilt, group_id=row["group_id"]),
             "Sealed source record/gold differs or is quarantined")
    return {"record_id": row["id"], "group_id": row["group_id"], "source_kind": row["source_kind"],
            "record_sha256": value_sha256(row), "complete_group_audit_performed_here": False}


def render_prompt(row):
    """Return native roles with exact question and original ARC option order."""
    _, _, _, user = _question_fields(row)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def serialize_prompt(row, tokenizer, tokenizer_identity, max_prompt_tokens=8192):
    """Bind supplied native template text/tokens; never truncate or read gold.

    Identity fields are caller verified. This pure function checks template/EOS
    consistency and text-token parity, not tokenizer files on disk.
    """
    identifier, group, kind, _ = _question_fields(row)
    _require(type(max_prompt_tokens) is int and max_prompt_tokens > 0,
             "An explicit positive prompt envelope is required")
    _require(type(tokenizer_identity) is dict and type(tokenizer_identity.get("files")) is dict and
             bool(tokenizer_identity["files"]), "Caller-verified tokenizer file identities are required")
    identity = deepcopy(tokenizer_identity)
    identity_sha = value_sha256(identity)
    for filename, checksum in identity["files"].items():
        _require(type(filename) is str and bool(filename), "Tokenizer file name is invalid")
        _sha(checksum)
    template = getattr(tokenizer, "chat_template", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    _require(type(template) is str and bool(template) and type(eos) is int and eos >= 0 and
             identity.get("chat_template_sha256") == text_sha256(template) and
             identity.get("eos_token_id") == eos, "Supplied native template/EOS identity differs")
    messages = render_prompt(row)
    token_ids = tokenizer.apply_chat_template(deepcopy(messages), tokenize=True, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(deepcopy(messages), tokenize=False, add_generation_prompt=True)
    _require(type(rendered) is str and type(token_ids) is list and bool(token_ids) and
             all(type(item) is int and item >= 0 for item in token_ids), "Native prompt must yield plain token IDs/text")
    parity = tokenizer.encode(rendered, add_special_tokens=False)
    _require(type(parity) is list and parity == token_ids, "Native rendered text/token sequence differs")
    _require(len(token_ids) <= max_prompt_tokens, "Complete source prompt exceeds its envelope; no truncation/refill")
    return {"schema": "source-native-prompt-v1", "record_id": identifier, "group_id": group,
        "source_kind": kind, "messages": messages, "rendered_prompt": rendered, "prompt_token_ids": token_ids,
        "provenance": {"record_id_namespace": "independent_labeled_source_record",
            "question_sha256": row["question_sha256"], "messages_sha256": value_sha256(messages),
            "system_text_sha256": text_sha256(messages[0]["content"]), "user_text_sha256": text_sha256(messages[1]["content"]),
            "rendered_prompt_sha256": text_sha256(rendered), "prompt_token_ids_sha256": value_sha256(token_ids),
            "prompt_tokens": len(token_ids), "max_prompt_tokens": max_prompt_tokens,
            "tokenizer_identity": identity, "tokenizer_identity_sha256": identity_sha, "gold_fields_used": False,
            "truncation_performed": False, "tokenizer_files_verified_here": False}}


def normalize_percentage(text):
    """Normalize one supported explicit scalar percentage with exact arithmetic.

    Supported scalar grammar: optional sign; ASCII integer/decimal; slash ratio
    of signed decimals; or \\frac/\\dfrac/\\tfrac{signed decimal}{signed decimal}.
    A single literal % or escaped \\% must be final. No scientific notation,
    prose, outer math delimiters, symbolic/mixed expressions or nested fractions
    are accepted when a percent mark is present. Non-percent math passes through.
    """
    _require(type(text) is str, "Math answer must be text")
    stripped = text.strip()
    if "%" not in stripped:
        return {"supported": True, "status": "no_percentage", "normalized": stripped, "reason": None}
    match = re.fullmatch(r"(.+?)(?:\\%)|(.+?)%", stripped, re.DOTALL)
    if match is None or stripped.count("%") != 1:
        return {"supported": False, "status": "unsupported_percentage", "normalized": None,
                "reason": "Expected one final scalar percentage mark"}
    body = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
    scalar = re.fullmatch(_SCALAR, body)
    slash = re.fullmatch(r"(" + _SCALAR + r")\s*/\s*(" + _SCALAR + r")", body)
    latex = re.fullmatch(r"([+-]?)\s*\\(?:frac|dfrac|tfrac)\s*\{\s*(" + _SCALAR +
                        r")\s*\}\s*\{\s*(" + _SCALAR + r")\s*\}", body)
    try:
        if scalar:
            value = Fraction(re.sub(r"\s+", "", body))
        elif slash:
            value = Fraction(re.sub(r"\s+", "", slash.group(1))) / Fraction(re.sub(r"\s+", "", slash.group(2)))
        elif latex:
            value = Fraction(re.sub(r"\s+", "", latex.group(2))) / Fraction(re.sub(r"\s+", "", latex.group(3)))
            if latex.group(1) == "-":
                value = -value
        else:
            raise ValueError("Unsupported scalar percentage syntax")
        value /= 100
    except (ValueError, ZeroDivisionError):
        return {"supported": False, "status": "unsupported_percentage", "normalized": None,
                "reason": "Unsupported scalar percentage syntax or zero denominator"}
    normalized = str(value.numerator) if value.denominator == 1 else str(value.numerator) + "/" + str(value.denominator)
    return {"supported": True, "status": "normalized_percentage", "normalized": normalized, "reason": None}


def validate_gold(row):
    """Private preflight; unsupported gold blocks the cohort before outcomes."""
    validate_record(row)
    kind = row["source_kind"]
    gold = row["label"]["answer"] if kind == "math" else row["label"]["answer_key"]
    percent = normalize_percentage(gold) if kind == "math" else {
        "supported": True, "status": "exact_original_option_label", "normalized": gold, "reason": None}
    nonempty = bool(gold.strip())
    return {"record_id": row["id"], "source_kind": kind, "supported": percent["supported"] and nonempty,
        "status": percent["status"] if nonempty else "empty_gold", "reason": percent["reason"] if nonempty else "Empty gold answer",
        "gold_text": gold, "percent_normalized_gold": percent["normalized"] if kind == "math" else None,
        "gold_sha256": text_sha256(gold)}


def extract_final_box(text):
    """Only the final literal boxed attempt counts; malformed later attempts fail."""
    _require(type(text) is str, "Response must be text")
    starts = [match.start() for match in re.finditer(r"\\boxed", text)]
    result = {"status": "missing_box", "answer": None, "attempts": len(starts), "start": None, "end": None}
    if not starts:
        return result
    start, opening = starts[-1], starts[-1] + len("\\boxed")
    result.update(start=start, status="malformed_final_box")
    if opening >= len(text) or text[opening] != "{":
        return result
    depth = 1
    for index in range(opening + 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                answer = text[opening + 1:index].strip()
                result.update(status="ok" if answer else "empty_final_box", answer=answer or None, end=index + 1)
                return result
    result["status"] = "unbalanced_final_box"
    return result


def sample_seed(record_id, global_seed=GLOBAL_SEED):
    _sha(record_id)
    _require(type(global_seed) is int and 0 <= global_seed < 2 ** 31, "Seed must be a nonnegative 31-bit integer")
    return (global_seed + int(record_id[:8], 16)) % (2 ** 31)


class _QuietWriter:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def _load_pinned_comparator():
    """Load the exact repo namespace in the disposable worker, not parent scope."""
    directory = Path(__file__).resolve().parents[1] / "eval" / "utils"
    for name in tuple(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    package = types.ModuleType("utils")
    package.__path__ = [str(directory)]
    package.__package__ = "utils"
    package.__spec__ = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = [str(directory)]
    sys.modules["utils"] = package
    importlib.import_module("utils.utils")
    grader = importlib.import_module("utils.grader")
    modules = {}
    for name, filename in _GRADER_FILES.items():
        module = sys.modules.get(name)
        actual = Path(getattr(module, "__file__", "")).resolve()
        _require(actual == (directory / filename).resolve(), "Unexpected loaded comparator module identity")
        modules[name] = str(actual)
    provenance = {"loaded_modules": modules,
        "runtime_versions": {name: importlib.metadata.version(name) for name in _PACKAGES},
        "include_percentage": False, "is_close": True, "internal_timeout": False,
        "timeout_owner": "external_spawn_process", "normalization_order": "explicit_percentage_then_pinned_strip_string"}
    return grader.math_equal, sys.modules["utils.parser"].strip_string, provenance


def _math_worker(sender, prediction, gold):
    try:
        with contextlib.redirect_stdout(_QuietWriter()), contextlib.redirect_stderr(_QuietWriter()):
            comparator, normalizer, provenance = _load_pinned_comparator()
            prediction, gold = normalizer(prediction), normalizer(gold)
            _require(type(prediction) is str and type(gold) is str, "Normalizer returned nontext")
            answer = comparator(prediction, gold, include_percentage=False, is_close=True, timeout=False)
            _require(type(answer) is bool, "Comparator returned a nonboolean")
        sender.send({"status": "graded", "correct": answer, "comparator_provenance": provenance})
    except BaseException as error:
        # No exception text: upstream errors may include private answers.
        sender.send({"status": "grader_error", "correct": False, "error_type": type(error).__name__})
    finally:
        sender.close()


def _compare_math(prediction, gold, timeout_seconds):
    """Bounded worker IPC; EOF/crash never enters an unbounded queue read."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_math_worker, args=(sender, prediction, gold))
    begun = time.monotonic()
    result = {"status": "worker_error", "correct": False, "error_type": "WorkerNotStarted"}
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0., timeout_seconds - (time.monotonic() - begun))):
            result = {"status": "timeout", "correct": False, "error_type": "ExternalGraderTimeout"}
        else:
            try:
                payload = receiver.recv()
            except (EOFError, OSError):
                result = {"status": "worker_crash", "correct": False, "error_type": "WorkerExitedWithoutResult"}
            else:
                if not (type(payload) is dict and payload.get("status") in ("graded", "grader_error") and
                        type(payload.get("correct")) is bool and (payload["status"] != "graded" or
                        type(payload.get("comparator_provenance")) is dict) and
                        (payload["status"] != "grader_error" or payload["correct"] is False and
                         type(payload.get("error_type")) is str)):
                    result = {"status": "worker_protocol_error", "correct": False, "error_type": "InvalidWorkerResult"}
                else:
                    result = payload
    except Exception as error:
        result = {"status": "worker_error", "correct": False, "error_type": type(error).__name__}
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            try:
                process.join(.05)
                if process.is_alive():
                    process.terminate()
                    process.join(.25)
                if process.is_alive():
                    process.kill()
                    process.join(.25)
                if process.is_alive():
                    result = {"status": "worker_cleanup_error", "correct": False,
                        "error_type": "WorkerAliveAfterBoundedKill", "prior_status": result["status"]}
                else:
                    process.close()
            except Exception as error:
                result = {"status": "worker_cleanup_error", "correct": False,
                    "error_type": type(error).__name__, "prior_status": result["status"]}
    return result


def grade_response(row, text, timeout_seconds=5):
    """Grade one response; every result has denominator one and explicit status."""
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 60,
             "A finite positive hard comparator deadline up to 60 seconds is required")
    _require(type(text) is str, "Response must be text")
    gold = validate_gold(row)
    extraction = extract_final_box(text)
    result = {"record_id": row["id"], "source_kind": row["source_kind"], "status": "incorrect",
        "correct": False, "extraction": extraction, "error": None, "denominator": 1,
        "rankable": True, "response_sha256": text_sha256(text)}
    if not gold["supported"]:
        result.update(status="unsupported_gold", rankable=False, error={"type": "UnsupportedGold", "reason": gold["reason"]})
    elif extraction["status"] != "ok":
        result["status"] = extraction["status"]
    elif row["source_kind"] == "arc_challenge":
        labels = row["original_record"]["choices"]["label"]
        prediction = extraction["answer"]
        result.update(status="graded" if prediction in labels else "invalid_option_label",
                      correct=prediction == gold["gold_text"] and prediction in labels)
    else:
        prediction = normalize_percentage(extraction["answer"])
        if not prediction["supported"]:
            result["status"] = "unsupported_prediction_percentage"
        else:
            comparison = _compare_math(prediction["normalized"], gold["percent_normalized_gold"], float(timeout_seconds))
            result.update(status=comparison["status"], correct=comparison["correct"])
            if comparison["status"] != "graded":
                result.update(rankable=False, error={"type": comparison.get("error_type", "UnknownGraderError")})
            else:
                result["comparator_provenance"] = comparison["comparator_provenance"]
    return result


def completion_metrics(text, token_ids, finish_reason, stop_reason, eos_token_id):
    """Deterministic diagnostics; generic 'stop' is not asserted to be native EOS."""
    _require(type(text) is str and type(token_ids) is list and
        all(type(value) is int and value >= 0 for value in token_ids) and
        finish_reason in ("stop", "length", "abort") and type(eos_token_id) is int and eos_token_id >= 0 and
        (stop_reason is None or type(stop_reason) in (str, int)), "Invalid completion metadata")
    words = text.split()
    grams = [tuple(words[index:index + 8]) for index in range(max(0, len(words) - 7))]
    return {"generated_tokens": len(token_ids), "finish_reason": finish_reason,
        "length_finished": finish_reason == "length", "stop_finished": finish_reason == "stop",
        "native_eos_observed": bool(token_ids and token_ids[-1] == eos_token_id) or stop_reason == eos_token_id,
        "stop_reason": stop_reason, "repetition_8gram": 0. if not grams else 1. - len(set(grams)) / len(grams),
        "box_status": extract_final_box(text)["status"], "response_sha256": text_sha256(text)}


def summarize_samples(row, samples, global_seed=GLOBAL_SEED):
    """One record's exact seeded eight-sample estimator, never best-of partials.

    Input rows contain sample_index/request_seed/grade/metrics. Raw text need not
    be duplicated here. Timeout/errors remain in the denominator but block use
    of the returned point estimates for ranking or selection.
    """
    _question_fields(row)
    expected_seed = sample_seed(row["id"], global_seed)
    _require(type(samples) is list and len(samples) == SAMPLE_COUNT and
        all(type(sample) is dict and type(sample.get("sample_index")) is int for sample in samples) and
        sorted(sample["sample_index"] for sample in samples) == list(range(SAMPLE_COUNT)),
        "Exactly the eight unique declared sample indices are required")
    for sample in samples:
        grade, metrics = sample.get("grade"), sample.get("metrics")
        _require(sample.get("request_seed") == expected_seed and type(grade) is dict and type(metrics) is dict and
            grade.get("record_id") == row["id"] and grade.get("source_kind") == row["source_kind"] and
            grade.get("denominator") == 1 and type(grade.get("correct")) is bool and type(grade.get("rankable")) is bool and
            type(metrics.get("generated_tokens")) is int and metrics["generated_tokens"] >= 0 and
            metrics.get("finish_reason") in ("stop", "length", "abort") and
            metrics.get("response_sha256") == grade.get("response_sha256"), "Sample identity or grading/metadata differs")
        _sha(grade.get("response_sha256"))
        ordinary = {"graded", "missing_box", "malformed_final_box", "unbalanced_final_box", "empty_final_box",
                    "invalid_option_label", "unsupported_prediction_percentage"}
        errors = {"unsupported_gold", "timeout", "worker_crash", "grader_error", "worker_protocol_error",
                  "worker_error", "worker_cleanup_error"}
        _require(grade.get("status") in ordinary | errors and
            (grade["rankable"] and grade["status"] in ordinary and grade.get("error") is None or
             not grade["rankable"] and grade["status"] in errors and type(grade.get("error")) is dict) and
            (not grade["correct"] or grade["status"] == "graded"), "Unknown or inconsistent grading state")
        repetition = metrics.get("repetition_8gram")
        _require(type(repetition) in (int, float) and math.isfinite(repetition) and 0 <= repetition <= 1,
                 "Completion repetition fraction is invalid")
        _require(grade["rankable"] or grade["correct"] is False, "Errored grading cannot count as correct")
    correct = sum(sample["grade"]["correct"] for sample in samples)
    return {"record_id": row["id"], "group_id": row["group_id"], "source_kind": row["source_kind"],
        "request_seed": expected_seed, "sample_count": SAMPLE_COUNT, "correct_samples": correct,
        "pass_at_1": correct / SAMPLE_COUNT, "pass_at_8": float(correct > 0),
        "denominator": SAMPLE_COUNT, "rankable": all(sample["grade"]["rankable"] and
            sample["metrics"]["finish_reason"] != "abort" for sample in samples),
        "grading_errors": sum(not sample["grade"]["rankable"] for sample in samples),
        "aborted_samples": sum(sample["metrics"]["finish_reason"] == "abort" for sample in samples),
        "length_finishes": sum(sample["metrics"]["finish_reason"] == "length" for sample in samples),
        "generated_token_sum": sum(sample["metrics"]["generated_tokens"] for sample in samples),
        "sample_inventory_sha256": value_sha256(sorted(samples, key=lambda sample: sample["sample_index"]))}
