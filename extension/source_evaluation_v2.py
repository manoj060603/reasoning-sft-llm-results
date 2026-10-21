"""UNADMITTED operational v2 of the source-only MATH worker.

All prompt, gold, percentage, final-box and ARC semantics delegate to the frozen
v1 helpers. Only the MATH worker lifecycle changes: one fresh process has a
10-second cold setup deadline, verified READY, one GO, then a separate five-
second normalization/comparison/result deadline. Cleanup joins total at most
.55 seconds. These are declared deadline/join budgets, not a guarantee that an
uninterruptible OS syscall returns within15.55 seconds. No retries or reuse.

This is not evidence that the historical five-second end-to-end contract passed.
The caller must supply prospective pinned comparator identities, preserve every
error denominator, and stop/persist the result if SourceGraderCleanupError is
raised. No data/model loading, generation, cohort selection or admission occurs.
"""
from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import secrets
import time

from extension import source_evaluation as v1
from extension.preserving_data import text_sha256, value_sha256


PROTOCOL = "source-grader-two-phase-v2"
FRAME_SCHEMA = "source-grader-worker-v2"
SETUP_SECONDS, WORK_SECONDS, CLEANUP_SECONDS = 10., 5., .55
MAX_FRAME_BYTES = 4092
MODULE_NAMES = frozenset(v1._GRADER_FILES)
PACKAGE_NAMES = frozenset(v1._PACKAGES)
ERROR_STATUSES = frozenset({"setup_timeout", "setup_error", "setup_identity_error", "comparison_timeout",
    "comparator_error", "worker_crash", "protocol_error", "worker_error", "cleanup_error"})

# Pure semantics are exactly the prior callable objects, not copied algorithms.
SourceEvaluationError = v1.SourceEvaluationError
validate_record, render_prompt, serialize_prompt = v1.validate_record, v1.render_prompt, v1.serialize_prompt
normalize_percentage, validate_gold, extract_final_box = v1.normalize_percentage, v1.validate_gold, v1.extract_final_box
sample_seed, completion_metrics = v1.sample_seed, v1.completion_metrics
SAMPLE_COUNT, GLOBAL_SEED, ARC_INSTRUCTION = v1.SAMPLE_COUNT, v1.GLOBAL_SEED, v1.ARC_INSTRUCTION


class SourceGraderCleanupError(RuntimeError):
    """Persist .result privately and abort; continuing would orphan worker state."""
    def __init__(self, result):
        self.result = deepcopy(result)
        super().__init__("Source grader worker cleanup failed; caller must persist result and stop")


class _ProtocolError(RuntimeError):
    pass


class _PhaseTimeout(TimeoutError):
    pass


class _WorkerCrash(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise SourceEvaluationError(message)


def _expected_identity(expected):
    _require(type(expected) is dict and set(expected) == {"loaded_modules", "source_sha256", "runtime_versions"},
        "Supply the prospective pinned comparator paths, five source hashes and runtime versions")
    _require(type(expected["loaded_modules"]) is dict and set(expected["loaded_modules"]) == MODULE_NAMES and
        all(type(path) is str and Path(path).is_absolute() for path in expected["loaded_modules"].values()) and
        type(expected["source_sha256"]) is dict and set(expected["source_sha256"]) == MODULE_NAMES and
        type(expected["runtime_versions"]) is dict and set(expected["runtime_versions"]) == PACKAGE_NAMES and
        all(type(version) is str and bool(version) for version in expected["runtime_versions"].values()),
        "Exactly the five prospective comparator module/runtime identities are required")
    for checksum in expected["source_sha256"].values():
        v1._sha(checksum)
    return deepcopy(expected)


def _actual_comparator():
    comparator, normalizer, old_proof = v1._load_pinned_comparator()
    proof = deepcopy(old_proof)
    proof["source_sha256"] = {}
    for name, filename in proof["loaded_modules"].items():
        checksum = hashlib.sha256()
        with Path(filename).open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                checksum.update(block)
        proof["source_sha256"][name] = checksum.hexdigest()
    proof.update(timeout_owner="external_two_phase_spawn_process", worker_protocol=PROTOCOL,
        setup_deadline_seconds=SETUP_SECONDS, comparison_deadline_seconds=WORK_SECONDS)
    return comparator, normalizer, proof


def _validate_proof(proof, expected):
    _require(type(proof) is dict and all(proof.get(key) == expected[key] for key in
        ("loaded_modules", "source_sha256", "runtime_versions")) and
        proof.get("include_percentage") is False and proof.get("is_close") is True and
        proof.get("internal_timeout") is False and proof.get("timeout_owner") == "external_two_phase_spawn_process" and
        proof.get("worker_protocol") == PROTOCOL and proof.get("setup_deadline_seconds") == SETUP_SECONDS and
        proof.get("comparison_deadline_seconds") == WORK_SECONDS and
        proof.get("normalization_order") == "explicit_percentage_then_pinned_strip_string",
        "READY did not verify the exact prospective comparator/source/runtime identity")


def _send(connection, request_id, phase, **values):
    frame = {"schema": FRAME_SCHEMA, "request_id": request_id, "phase": phase, **values}
    raw = json.dumps(frame, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_FRAME_BYTES:
        raise _ProtocolError("Worker frame exceeds the small atomic message budget")
    connection.send_bytes(raw)


def _decode(raw, request_id):
    def pairs(items):
        values = {}
        for key, value in items:
            if key in values:
                raise ValueError("Duplicate worker JSON key")
            values[key] = value
        return values
    try:
        frame = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite worker JSON")))
    except (ValueError, UnicodeError, TypeError) as error:
        raise _ProtocolError("Malformed worker JSON") from error
    if not (type(frame) is dict and frame.get("schema") == FRAME_SCHEMA and frame.get("request_id") == request_id and
            frame.get("phase") in {"ready", "go", "result", "setup_error", "comparator_error"}):
        raise _ProtocolError("Worker frame schema, request identity or phase differs")
    extras = {"ready": {"comparator_provenance", "worker_setup_elapsed_seconds", "worker_setup_cpu_seconds"},
        "go": set(), "result": {"correct", "worker_comparison_elapsed_seconds", "worker_comparison_cpu_seconds"},
        "setup_error": {"error_type"}, "comparator_error": {"error_type"}}
    if set(frame) != {"schema", "request_id", "phase"} | extras[frame["phase"]]:
        raise _ProtocolError("Worker frame has missing or unexpected fields")
    if frame["phase"] in ("setup_error", "comparator_error") and not (
            type(frame["error_type"]) is str and bool(frame["error_type"])):
        raise _ProtocolError("Worker error type must be explicit")
    return frame


def _read(connection, request_id, deadline, clock):
    remaining = deadline - clock()
    if remaining <= 0 or not connection.poll(remaining):
        raise _PhaseTimeout("Phase deadline expired")
    if clock() > deadline:
        raise _PhaseTimeout("Worker frame arrived after its phase deadline")
    try:
        raw = connection.recv_bytes(MAX_FRAME_BYTES)
    except EOFError as error:
        raise _WorkerCrash("Worker exited without its required frame") from error
    except OSError as error:
        raise _ProtocolError("Invalid or oversized worker frame") from error
    if clock() > deadline:
        raise _PhaseTimeout("Worker frame was not completely received within its phase deadline")
    return _decode(raw, request_id)


def _math_worker(connection, prediction, gold, request_id):
    """Imports/hash checks precede READY; answer normalization waits for GO."""
    phase, begun, cpu_begun = "setup", time.monotonic(), time.process_time()
    try:
        with contextlib.redirect_stdout(v1._QuietWriter()), contextlib.redirect_stderr(v1._QuietWriter()):
            comparator, normalizer, proof = _actual_comparator()
            _send(connection, request_id, "ready", comparator_provenance=proof,
                worker_setup_elapsed_seconds=time.monotonic() - begun,
                worker_setup_cpu_seconds=time.process_time() - cpu_begun)
            go = _read(connection, request_id, time.monotonic() + SETUP_SECONDS, time.monotonic)
            if set(go) != {"schema", "request_id", "phase"} or go["phase"] != "go":
                raise _ProtocolError("Exactly one payload-free GO is required")
            if connection.poll(0):
                try:
                    connection.recv_bytes(MAX_FRAME_BYTES)
                except EOFError as error:
                    raise _WorkerCrash("Parent exited before comparison") from error
                raise _ProtocolError("Duplicate control frame before comparison")
            phase, work_begun, work_cpu = "comparison", time.monotonic(), time.process_time()
            prediction, gold = normalizer(prediction), normalizer(gold)
            _require(type(prediction) is str and type(gold) is str, "Normalizer returned nontext")
            correct = comparator(prediction, gold, include_percentage=False, is_close=True, timeout=False)
            _require(type(correct) is bool, "Comparator returned a nonboolean")
            _send(connection, request_id, "result", correct=correct,
                worker_comparison_elapsed_seconds=time.monotonic() - work_begun,
                worker_comparison_cpu_seconds=time.process_time() - work_cpu)
    except BaseException as error:
        try:
            _send(connection, request_id, "setup_error" if phase == "setup" else "comparator_error",
                error_type=type(error).__name__)
        except BaseException:
            pass
    finally:
        connection.close()


def _cleanup(process, clock):
    begun, actions = clock(), []
    cleanup = {"successful": True, "actions": actions, "worker_alive": False, "exitcode": None,
        "declared_join_budget_seconds": CLEANUP_SECONDS}
    try:
        if process is not None and process.pid is not None:
            process.join(min(.05, max(0., CLEANUP_SECONDS - (clock() - begun))))
            if process.is_alive():
                process.terminate()
                actions.append("terminate")
                process.join(min(.25, max(0., CLEANUP_SECONDS - (clock() - begun))))
            if process.is_alive():
                process.kill()
                actions.append("kill")
                process.join(min(.25, max(0., CLEANUP_SECONDS - (clock() - begun))))
            cleanup.update(worker_alive=process.is_alive(), exitcode=process.exitcode)
            cleanup["successful"] = not cleanup["worker_alive"]
            if not cleanup["worker_alive"]:
                process.close()
    except BaseException as error:
        cleanup.update(successful=False, error_type=type(error).__name__)
    cleanup["elapsed_wall_seconds"] = clock() - begun
    return cleanup


def _compare_math(prediction, gold, expected_comparator, *, context=None, clock=None, process_clock=None, request_id=None):
    """One cold setup/GO/work lifecycle; injected primitives are fixture seams."""
    clock, process_clock = clock or time.monotonic, process_clock or time.process_time
    request_id = request_id or value_sha256({"prediction_sha256": text_sha256(prediction),
        "gold_sha256": text_sha256(gold), "nonce": secrets.token_hex(16)})
    v1._sha(request_id)
    timings = {"declared_setup_deadline_seconds": SETUP_SECONDS, "declared_comparison_deadline_seconds": WORK_SECONDS,
        "declared_cleanup_join_budget_seconds": CLEANUP_SECONDS, "declared_total_budget_seconds": 15.55,
        "budget_qualification": "deadline and join budgets; not an OS syscall timing guarantee",
        "setup_elapsed_seconds": None, "comparison_elapsed_seconds": None}
    result = {"status": "setup_identity_error", "correct": False, "error_type": "MissingOrInvalidExpectedComparator",
        "error_phase": "setup", "worker_protocol": PROTOCOL, "request_id": request_id,
        "timings": timings, "go_sent": False}
    try:
        expected = _expected_identity(expected_comparator)
    except (ValueError, TypeError):
        return result
    parent = child = process = None
    begun, cpu_begun, stage, work_begun = clock(), process_clock(), "setup", None
    try:
        context = context or multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_math_worker, args=(child, prediction, gold, request_id))
        process.start()
        child.close()
        ready = _read(parent, request_id, begun + SETUP_SECONDS, clock)
        if ready["phase"] == "setup_error":
            result.update(status="setup_error", error_type=ready.get("error_type", "WorkerSetupError"))
        else:
            if ready["phase"] != "ready":
                raise _ProtocolError("A verified READY is required before GO")
            try:
                _validate_proof(ready.get("comparator_provenance"), expected)
            except (ValueError, TypeError, KeyError):
                result.update(status="setup_identity_error", error_type="ComparatorIdentityMismatch")
            else:
                for key in ("worker_setup_elapsed_seconds", "worker_setup_cpu_seconds"):
                    if not (type(ready.get(key)) in (int, float) and math.isfinite(ready[key]) and ready[key] >= 0):
                        raise _ProtocolError("READY omitted finite worker setup timings")
                if clock() > begun + SETUP_SECONDS:
                    raise _PhaseTimeout("READY identity verification exceeded the setup deadline")
                if parent.poll(0):
                    # A legitimate worker is waiting for GO and sends no extra frame.
                    try:
                        parent.recv_bytes(MAX_FRAME_BYTES)
                    except EOFError as error:
                        raise _WorkerCrash("Worker exited before GO") from error
                    raise _ProtocolError("Extra frame before GO")
                go_at = clock()
                if go_at > begun + SETUP_SECONDS:
                    raise _PhaseTimeout("GO would begin after the setup deadline")
                timings["setup_elapsed_seconds"] = go_at - begun
                timings["worker_setup_elapsed_seconds"] = ready["worker_setup_elapsed_seconds"]
                timings["worker_setup_cpu_seconds"] = ready["worker_setup_cpu_seconds"]
                result["comparator_provenance"] = ready["comparator_provenance"]
                stage, work_begun = "comparison", go_at
                _send(parent, request_id, "go")
                result["go_sent"] = True
                frame = _read(parent, request_id, work_begun + WORK_SECONDS, clock)
                if frame["phase"] == "comparator_error":
                    result.update(status="comparator_error", error_type=frame.get("error_type", "ComparatorError"))
                else:
                    if frame["phase"] != "result" or type(frame.get("correct")) is not bool:
                        raise _ProtocolError("Exactly one boolean RESULT is required after GO")
                    for key in ("worker_comparison_elapsed_seconds", "worker_comparison_cpu_seconds"):
                        if not (type(frame.get(key)) in (int, float) and math.isfinite(frame[key]) and frame[key] >= 0):
                            raise _ProtocolError("RESULT omitted finite worker comparison timings")
                    if parent.poll(0):
                        try:
                            parent.recv_bytes(MAX_FRAME_BYTES)
                        except EOFError:
                            pass
                        else:
                            raise _ProtocolError("Duplicate or extra frame after RESULT")
                    result.update(status="graded", correct=frame["correct"], error_type=None)
                    timings["worker_comparison_elapsed_seconds"] = frame["worker_comparison_elapsed_seconds"]
                    timings["worker_comparison_cpu_seconds"] = frame["worker_comparison_cpu_seconds"]
    except _PhaseTimeout:
        result.update(status="setup_timeout" if stage == "setup" else "comparison_timeout", correct=False,
            error_type="SetupDeadlineExceeded" if stage == "setup" else "ComparisonDeadlineExceeded")
    except _WorkerCrash:
        result.update(status="worker_crash", correct=False, error_type="WorkerExitedWithoutRequiredFrame")
    except _ProtocolError:
        result.update(status="protocol_error", correct=False, error_type="WorkerProtocolViolation")
    except BaseException as error:
        result.update(status="worker_error", correct=False, error_type=type(error).__name__)
    finally:
        now = clock()
        if timings["setup_elapsed_seconds"] is None:
            timings["setup_elapsed_seconds"] = now - begun
        if work_begun is not None:
            timings["comparison_elapsed_seconds"] = now - work_begun
        timings["worker_lifecycle_before_cleanup_seconds"] = now - begun
        timings["parent_process_cpu_seconds"] = process_clock() - cpu_begun
        connection_error = None
        for connection in (parent, child):
            if connection is None:
                continue
            try:
                connection.close()
            except BaseException as error:
                connection_error = type(error).__name__
        cleanup = _cleanup(process, clock)
        if connection_error is not None:
            cleanup.update(successful=False, connection_close_error_type=connection_error)
        result["cleanup"] = cleanup
        timings["total_elapsed_including_cleanup_seconds"] = clock() - begun
        if not cleanup["successful"]:
            result.update(status="cleanup_error", correct=False, error_type="BoundedWorkerCleanupFailed",
                prior_status=result["status"], caller_must_abort=True)
        elif result["status"] == "graded" and cleanup["exitcode"] not in (None, 0) and not cleanup["actions"]:
            result.update(status="worker_crash", correct=False, error_type="WorkerExitedAbnormallyAfterResult")
        result["error_phase"] = None if result["status"] == "graded" else "cleanup" if result["status"] == "cleanup_error" else stage
    return result


def grade_response(row, text, timeout_seconds=5, *, expected_comparator=None):
    """Same scoring semantics, separately versioned two-phase MATH operation.

    Configuration/deadline changes are not accepted here. A cleanup exception
    carries the complete denominator-preserving receipt for private persistence.
    """
    _require(type(timeout_seconds) in (int, float) and timeout_seconds == WORK_SECONDS,
        "The prospective v2 comparison deadline is fixed at five seconds")
    _require(type(text) is str, "Response must be text")
    gold, extraction = validate_gold(row), extract_final_box(text)
    result = {"record_id": row["id"], "source_kind": row["source_kind"], "status": "incorrect", "correct": False,
        "extraction": extraction, "error": None, "denominator": 1, "rankable": True,
        "response_sha256": text_sha256(text), "worker_protocol": PROTOCOL, "caller_must_abort": False}
    if not gold["supported"]:
        result.update(status="unsupported_gold", rankable=False, error={"type": "UnsupportedGold", "reason": gold["reason"]})
    elif extraction["status"] != "ok":
        result["status"] = extraction["status"]
    elif row["source_kind"] == "arc_challenge":
        labels, prediction = row["original_record"]["choices"]["label"], extraction["answer"]
        result.update(status="graded" if prediction in labels else "invalid_option_label",
            correct=prediction == gold["gold_text"] and prediction in labels)
    else:
        prediction = normalize_percentage(extraction["answer"])
        if not prediction["supported"]:
            result["status"] = "unsupported_prediction_percentage"
        else:
            comparison = _compare_math(prediction["normalized"], gold["percent_normalized_gold"], expected_comparator)
            result.update(status=comparison["status"], correct=comparison["correct"], worker=comparison)
            if comparison["status"] != "graded":
                result.update(rankable=False, error={"type": comparison.get("error_type", "UnknownWorkerError"),
                    "phase": comparison.get("error_phase")})
            if "comparator_provenance" in comparison:
                result["comparator_provenance"] = comparison["comparator_provenance"]
            if comparison["status"] == "cleanup_error":
                result["caller_must_abort"] = True
                raise SourceGraderCleanupError(result)
    return result


def summarize_samples(row, samples, global_seed=GLOBAL_SEED):
    """Preserve original inventories; new phase errors block selection equally."""
    _require(type(samples) is list, "Supply the exact eight recorded samples")
    compatible = deepcopy(samples)
    for original, copied in zip(samples, compatible):
        grade = original.get("grade", {}) if type(original) is dict else {}
        if grade.get("status") == "cleanup_error" or grade.get("caller_must_abort") is True:
            raise SourceGraderCleanupError(grade)
        if grade.get("status") in ERROR_STATUSES:
            _require(grade.get("correct") is False and grade.get("rankable") is False and
                grade.get("denominator") == 1 and type(grade.get("error")) is dict,
                "New operational errors cannot count as correct or rankable")
            # Reuse all unchanged v1 identity/metric/denominator guards. This is
            # only an internal category adapter, never a rescoring or rewrite of
            # the persisted original sample inventory.
            copied["grade"]["status"] = "grader_error"
    summary = v1.summarize_samples(row, compatible, global_seed)
    summary["sample_inventory_sha256"] = value_sha256(sorted(samples, key=lambda item: item["sample_index"]))
    summary["worker_protocol"] = PROTOCOL
    return summary
