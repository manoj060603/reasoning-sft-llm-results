"""Evidence-only cold-start timing of the unchanged source math grader.

Only four declared public scalar fixtures, twice each, are used. Each fixture
gets the exact unchanged five-second grade_response and a separately labeled
instrumented cold-spawn trial with the same five-second total deadline. Phase
timings are diagnostic observations, never score/ranking/admission success.
No cohorts, cached prompts, model weights, generation or downstream jobs occur.
Run only through the separately authorized finite CPU allocation.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time


ROLE = "source_grader_diagnostic_20261021_v1"
SCHEMA = "extension-source-grader-diagnostic-v1"
OWN_SOURCES = frozenset({"extension/source_grader_diagnostic.py", "extension/source_grader_diagnostic.slurm"})
FIXTURES = (
    {"id": "percentage_positive", "prediction": "50%", "gold": ".5", "expected_correct": True},
    {"id": "percentage_negative", "prediction": "50%", "gold": "50", "expected_correct": False},
    {"id": "numeric_negative_large", "prediction": "1000", "gold": "10", "expected_correct": False},
    {"id": "numeric_negative_small", "prediction": ".1", "gold": "10", "expected_correct": False},
)
REPETITIONS, WORK_SECONDS, CLEANUP_SECONDS = 2, 5., .55
_EVENTS = ("child_entry", "comparator_loaded", "normalization_finished", "comparison_started", "comparison_finished")
_ENV_ROOT = "SFT_SOURCE_GRADER_DIAGNOSTIC_ROOT"
_ENV_OUTPUT = "SFT_SOURCE_GRADER_DIAGNOSTIC_OUTPUT"


def _private_input_guard(root, output):
    """Narrow guard inherited by spawn via its importable main-module bootstrap."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if root != Path(__file__).resolve().parents[1] or output.parent != root / "extension/runs" or not output.name.startswith("source-grader-diagnostic-"):
        raise RuntimeError("Invalid diagnostic-only input guard binding")
    forbidden_imports = {"torch", "transformers", "datasets", "vllm"}
    forbidden_suffixes = {".safetensors", ".parquet", ".jsonl", ".pt", ".pth"}

    def audit(event, arguments):
        if event == "import" and arguments and str(arguments[0]).split(".")[0] in forbidden_imports:
            raise RuntimeError("Model/data framework import forbidden in the public scalar diagnostic")
        if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        private_runs = path.is_relative_to(root / "extension/runs") and not path.is_relative_to(output)
        hf_cache = any(marker in str(path) for marker in ("/.cache/huggingface/", "/huggingface/hub/"))
        if private_runs or path.is_relative_to(root / "deadline/artifacts") or hf_cache or path.suffix in forbidden_suffixes:
            raise RuntimeError("Private dataset/model/cache input forbidden in the public scalar diagnostic")
    sys.addaudithook(audit)


# Fresh multiprocessing imports its main module before resolving a worker target.
# The marker is set only after the parent has verified CPU/current-job admission.
if os.environ.get(_ENV_ROOT) or os.environ.get(_ENV_OUTPUT):
    _private_input_guard(os.environ.get(_ENV_ROOT, ""), os.environ.get(_ENV_OUTPUT, ""))

from extension.cpu_feasibility import (deadline_guard, digest, read_json, require,
    require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import CRITERION, source_record
from extension import source_evaluation as scorer
from extension import source_readiness_retry_submission as prior


def source_inventory(root):
    inventory = prior.source_inventory(root)
    require(len(inventory) == 71, "Exactly the unchanged 71-source readiness retry is required")
    for name in sorted(OWN_SOURCES):
        raw = prior.metadata._small_bytes(Path(root) / name, Path(root))
        inventory[name] = {"sha256": prior._sha(raw), "bytes": len(raw)}
    require(len(inventory) == 73, "Bind the 71 unchanged sources plus only these two diagnostic files")
    return inventory


def _public_record(fixture):
    raw = {"problem": "Public scalar diagnostic: " + fixture["id"], "level": "Level 1", "type": "Algebra",
        "solution": "Fixed public gold. \\boxed{" + fixture["gold"] + "}"}
    record, quarantine = source_record("math", raw, {"public_diagnostic": fixture["id"]})
    require(quarantine is None, "The predeclared public fixture must not be quarantined")
    record["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [record["id"]]})
    return record


def _cleanup(process):
    """At most .55 seconds of joins; never close a still-live process."""
    begun, phases = time.monotonic(), []
    result = {"successful": True, "worker_alive": False, "actions": phases}
    try:
        if process.pid is not None:
            process.join(min(.05, max(0., CLEANUP_SECONDS - (time.monotonic() - begun))))
            if process.is_alive():
                process.terminate()
                phases.append("terminate")
                process.join(min(.25, max(0., CLEANUP_SECONDS - (time.monotonic() - begun))))
            if process.is_alive():
                process.kill()
                phases.append("kill")
                process.join(min(.25, max(0., CLEANUP_SECONDS - (time.monotonic() - begun))))
            result["worker_alive"] = process.is_alive()
            result["exitcode"] = process.exitcode
            result["successful"] = not result["worker_alive"]
            if not result["worker_alive"]:
                process.close()
    except BaseException as error:
        result.update(successful=False, error_type=type(error).__name__)
    result["elapsed_wall_seconds"] = time.monotonic() - begun
    result["join_budget_seconds"] = CLEANUP_SECONDS
    return result


def _phase_worker(sender, prediction, gold):
    sequence = 0

    def emit(event, **values):
        nonlocal sequence
        payload = {"event": event, "sequence": sequence, "monotonic_seconds": time.monotonic(),
            "process_cpu_seconds": time.process_time(), "pid": os.getpid(), **values}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 4092:
            raise RuntimeError("Instrumented public event exceeded its small atomic message bound")
        sender.send_bytes(raw)
        sequence += 1

    try:
        emit("child_entry")
        with contextlib.redirect_stdout(scorer._QuietWriter()), contextlib.redirect_stderr(scorer._QuietWriter()):
            comparator, normalizer, provenance = scorer._load_pinned_comparator()
            emit("comparator_loaded", comparator_provenance=provenance)
            prediction, gold = normalizer(prediction), normalizer(gold)
            require(type(prediction) is str and type(gold) is str, "Normalizer returned nontext")
            emit("normalization_finished", normalized_prediction=prediction, normalized_gold=gold)
            emit("comparison_started")
            correct = comparator(prediction, gold, include_percentage=False, is_close=True, timeout=False)
            require(type(correct) is bool, "Comparator returned nonboolean")
            emit("comparison_finished", correct=correct)
    except BaseException as error:
        emit("worker_error", error_type=type(error).__name__)
    finally:
        sender.close()


def instrumented_cold_trial(prediction, gold):
    """Observe phases under the unchanged total five-second cold work deadline."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_phase_worker, args=(sender, prediction, gold))
    begun, cpu_begun, events = time.monotonic(), time.process_time(), []
    deadline = begun + WORK_SECONDS
    status, error_type = "worker_error", "WorkerNotStarted"
    try:
        process.start()
        sender.close()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not receiver.poll(remaining):
                status, error_type = "timeout", "ExternalGraderTimeout"
                break
            try:
                event = json.loads(receiver.recv_bytes(4092))
            except EOFError:
                status, error_type = "worker_crash", "WorkerExitedWithoutResult"
                break
            require(type(event) is dict and event.get("sequence") == len(events) and
                event.get("event") in set(_EVENTS) | {"worker_error"} and type(event.get("pid")) is int and
                all(type(event.get(key)) in (int, float) and math.isfinite(event[key]) for key in
                    ("monotonic_seconds", "process_cpu_seconds")), "Invalid public timing worker message")
            require(event["event"] == "worker_error" or len(events) < len(_EVENTS) and
                event["event"] == _EVENTS[len(events)], "Public timing phases are out of order")
            event["parent_received_elapsed_wall_seconds"] = time.monotonic() - begun
            events.append(event)
            if event["event"] == "worker_error":
                status, error_type = "grader_error", event.get("error_type", "UnknownWorkerError")
                break
            if event["event"] == "comparison_finished":
                require(type(event.get("correct")) is bool, "Public comparison result is nonboolean")
                status, error_type = "graded", None
                break
    except BaseException as error:
        status, error_type = "worker_error", type(error).__name__
    finally:
        sender.close()
        receiver.close()
        cleanup = _cleanup(process)
    if not cleanup["successful"]:
        status, error_type = "worker_cleanup_error", "BoundedCleanupFailed"
    last_phase = events[-1]["event"] if events else "before_child_entry"
    active_phase = {"before_child_entry": "spawn_bootstrap", "child_entry": "comparator_import",
        "comparator_loaded": "normalization", "normalization_finished": "before_comparison",
        "comparison_started": "comparison", "comparison_finished": None, "worker_error": None}
    return {"status": status, "error_type": error_type, "events": events,
        "last_observed_phase": last_phase,
        "timed_out_phase": active_phase[last_phase] if status == "timeout" else None,
        "correct": events[-1].get("correct") if status == "graded" else None,
        "work_deadline_seconds": WORK_SECONDS, "elapsed_wall_seconds": time.monotonic() - begun,
        "parent_process_cpu_seconds": time.process_time() - cpu_begun, "cleanup": cleanup,
        "diagnostic_only": True, "rankable": False}


def _validate_admission(root, compute):
    sources = source_inventory(root)
    ledger = read_json(root / "extension/job_ledger.json")
    entry = ledger.get("jobs", {}).get(ROLE)
    require(type(entry) is dict and entry.get("status") == "submitted" and entry.get("job_id") == compute["job_id"] and
        entry.get("requested_gpu_seconds") == 0 and entry.get("requested_gpu_hours") == 0 and
        entry.get("source_sha256") == sources and type(entry.get("binding")) is dict and
        entry["binding"].get("source_sha256") == sources and entry["binding"].get("role") == ROLE and
        entry.get("binding_sha256") == value_digest(entry["binding"]), "Current exact-source one-use CPU admission is missing")
    return sources, entry


def run(root, output):
    compute = require_cpu_allocation()
    require(compute["allocated_cpus"] == 1, "Diagnostic requires exactly one allocated CPU and no GPU")
    deadline_guard(173400)
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root == Path(__file__).resolve().parents[1] and output.parent == root / "extension/runs" and
        output.name == "source-grader-diagnostic-" + compute["job_id"] and not output.exists() and not output.is_symlink(),
        "A fresh exact diagnostic output is required")
    sources, entry = _validate_admission(root, compute)
    os.environ[_ENV_ROOT], os.environ[_ENV_OUTPUT] = str(root), str(output)
    _private_input_guard(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    write_new_json(output / "started.json", {"schema": SCHEMA, "complete": False, "role": ROLE,
        "job_id": compute["job_id"], "source_inventory": sources, "admission_binding_sha256": entry["binding_sha256"],
        "public_fixtures": list(FIXTURES), "repetitions": REPETITIONS, "work_deadline_seconds": WORK_SECONDS})
    begun, observations = time.monotonic(), []
    try:
        for repetition in range(REPETITIONS):
            for fixture in FIXTURES:
                require(time.monotonic() - begun < 540, "Keep diagnostic closeout within its finite ten-minute allocation")
                record = _public_record(fixture)
                start, cpu_start = time.monotonic(), time.process_time()
                result = scorer.grade_response(record, "\\boxed{" + fixture["prediction"] + "}", timeout_seconds=5)
                unchanged = {"result": result, "elapsed_wall_seconds": time.monotonic() - start,
                    "parent_process_cpu_seconds": time.process_time() - cpu_start,
                    "work_deadline_seconds": WORK_SECONDS, "unchanged_grade_response": True}
                prediction, gold = scorer.normalize_percentage(fixture["prediction"]), scorer.normalize_percentage(fixture["gold"])
                require(prediction["supported"] and gold["supported"], "Fixed public scalar fixture became unsupported")
                instrumented = instrumented_cold_trial(prediction["normalized"], gold["normalized"])
                observation = {"repetition": repetition, "fixture": dict(fixture),
                    "exact_unchanged_grader": unchanged, "instrumented_cold_spawn": instrumented}
                observations.append(observation)
                write_new_json(output / ("fixture-" + str(repetition) + "-" + fixture["id"] + ".json"), observation)
                require(instrumented["cleanup"]["successful"], "A surviving worker blocks further diagnostic trials")
        require(source_inventory(root) == sources and _validate_admission(root, compute) == (sources, entry),
            "Frozen source/admission changed during the diagnostic")
        ending = require_cpu_allocation()
        require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
        outputs = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
        receipt = {"schema": SCHEMA, "complete": True, "diagnostic_completed": True, "grader_success_claimed": False,
            "cpu_only": True, "job_id": compute["job_id"], "role": ROLE, "gpu_seconds": 0,
            "source_inventory": sources, "code_sha256": {name: meta["sha256"] for name, meta in sources.items()},
            "admission_binding_sha256": entry["binding_sha256"], "compute_start": compute, "compute_end": ending,
            "public_fixtures": list(FIXTURES), "repetitions": REPETITIONS, "observations": observations,
            "source_snapshot_unchanged": True, "private_inputs_forbidden": True, "model_framework_imports_forbidden": True,
            "calibration_or_model_outcomes_used": False, "automatic_downstream": False, "rankable": False,
            "output_sha256": outputs, "elapsed_seconds": time.monotonic() - begun,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_new_json(output / "complete.json", receipt)
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "job_id": compute["job_id"],
            "exception": type(error).__name__, "completed_public_trials": len(observations),
            "source_inventory": sources, "retry": False, "automatic_downstream": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = run(arguments.root, arguments.output)
    print(json.dumps({"diagnostic_completed": result["diagnostic_completed"], "public_trials": len(result["observations"]),
        "grader_success_claimed": False, "gpu_seconds": 0}), flush=True)


if __name__ == "__main__":
    main()
