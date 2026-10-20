"""Bounded, idempotent ARC submissions for the fresh transfer experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from reasoning.data import write_json
from deadline.gpu_budget import BudgetError, require_capacity

ROOT = Path("deadline")
DEADLINE = datetime(2026, 10, 21, 3, 59, tzinfo=timezone.utc)
GPU_HOURS_CAP = 48


def guard_start(required_hours, now=None):
    """Recheck deadline capacity after queue delays, before model/data work."""
    if required_hours <= 0:
        raise ValueError("Deadline reserve must be positive.")
    now = now or datetime.now(timezone.utc)
    if (DEADLINE - now).total_seconds() < required_hours * 3600:
        raise RuntimeError("Queue delay leaves insufficient time to complete this stage and its successors.")


def scheduler_live_ids():
    """Read expanded allocation identities; an unavailable snapshot fails closed."""
    snapshot = subprocess.run(["squeue", "--clusters=owl", "--array", "--states=all", "--noheader",
                               "--user=" + getpass.getuser(), "--format=%i"],
                              capture_output=True, text=True, timeout=60, check=True)
    return [line.strip() for line in snapshot.stdout.splitlines() if line.strip()]


def submit(role, script, *, arguments=(), dependency=None, dependency_kind="afterok",
           gpu_hours=0, required_hours=6):
    """Dispatch lightweight scheduler/ledger work from login or compute nodes.

    Model, data and native checks remain compute-allocation work in the callers.
    This function touches only the small locked ledger and the scheduler CLI.
    """
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "job_ledger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / "job_ledger.json"
        ledger = json.loads(path.read_text()) if path.exists() else {"jobs": {}, "gpu_hours_cap": GPU_HOURS_CAP}
        if role in ledger["jobs"]:
            existing = ledger["jobs"][role]
            if existing.get("status") == "submitted":
                return existing["job_id"]
            raise RuntimeError(f"Submission {role} is uncertain/failed; reconcile with the scheduler before retrying.")
        now = datetime.now(timezone.utc)
        if (DEADLINE - now).total_seconds() < required_hours * 3600:
            raise RuntimeError(f"Insufficient deadline reserve for {role}; do not submit.")
        if dependency_kind not in ("afterok", "afterany"):
            raise ValueError("Invalid scheduler dependency kind")
        if dependency is not None and not re.fullmatch(r"[0-9]+(?::[0-9]+)*", str(dependency)):
            raise ValueError("Invalid scheduler dependency")
        # Closed allocations may reclaim unused reservations only after exact
        # terminal sacct evidence and a fresh expanded live-job check. Preserve
        # the cumulative original requests as a separate historical quantity.
        has_receipts = any(job.get("gpu_accounting_receipt") is not None for job in ledger["jobs"].values())
        current = scheduler_live_ids() if has_receipts else []
        try:
            budget = require_capacity(ledger, live_job_ids=current,
                                      additional_requested_gpu_hours=gpu_hours)
        except BudgetError as exc:
            raise RuntimeError("The fresh experiment's conservative 48-GPU-hour limit cannot be verified: " + str(exc)) from exc
        command = ["sbatch", "--clusters=owl", "--parsable"]
        if dependency is not None:
            command.append("--dependency=" + dependency_kind + ":" + str(dependency))
        command.extend([str(script), *map(str, arguments)])
        entry = {"status": "submitting", "command": command,
                 "requested_gpu_hours": gpu_hours, "requested_at_utc": now.isoformat()}
        ledger["jobs"][role] = entry
        write_json(path, ledger)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.SubprocessError as exc:
            entry.update(status="uncertain", error=str(exc))
            write_json(path, ledger)
            raise
        if completed.returncode:
            entry.update(status="submission_failed", error=completed.stderr[-2000:])
            write_json(path, ledger)
            raise RuntimeError(entry["error"])
        identifier = completed.stdout.strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+", identifier):
            entry.update(status="uncertain", response=completed.stdout[-1000:])
            write_json(path, ledger)
            raise RuntimeError("Ambiguous scheduler response; do not submit a duplicate.")
        entry.update(status="submitted", job_id=identifier, scheduler_response=completed.stdout.strip())
        write_json(path, ledger)
        print(json.dumps({"role": role, "job_id": identifier,
                          "cumulative_admitted_requested_gpu_hours":
                              budget["cumulative_admitted_requested_gpu_seconds"] / 3600 + gpu_hours,
                          "conservative_gpu_hour_bound": budget["projected_gpu_seconds"] / 3600}), flush=True)
        return identifier


def report():
    results = ROOT / "results"
    if not any((results / role / "manifest.json").is_file() for role in ("original", "random", "advanced")):
        print("No evaluation manifests exist yet; no result is inferred.", flush=True)
        return
    # Serial report writes avoid two concurrently finishing evaluations racing.
    with (ROOT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = [sys.executable, "-m", "deadline.audit_results", "--output-dir", str(ROOT / "report")]
        for role in ("original", "random", "advanced"):
            if (results / role / "manifest.json").is_file():
                command.extend(["--" + role, str(results / role)])
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit-initial", "submit-advanced", "submit-eval", "report", "guard-start"])
    parser.add_argument("--role", choices=["random", "advanced"])
    parser.add_argument("--reserve-hours", type=int)
    args = parser.parse_args()
    if args.action == "guard-start":
        if not os.environ.get("SLURM_JOB_ID") or args.reserve_hours is None:
            raise RuntimeError("A compute allocation and explicit deadline reserve are required.")
        guard_start(args.reserve_hours)
    elif args.action == "submit-initial":
        if not (ROOT / "artifacts/finalized.json").is_file():
            raise RuntimeError("Native tests and complete CPU preparation must pass first.")
        probes = submit("probes_and_scores", ROOT / "gpu_probe.slurm", gpu_hours=12, required_hours=20)
        submit("random_training", ROOT / "gpu_train.slurm", arguments=["random"], gpu_hours=6, required_hours=12)
        submit("original_evaluation", ROOT / "gpu_eval.slurm", arguments=["original"], gpu_hours=6)
        submit("transfer_selection", ROOT / "select_transfer.slurm", dependency=probes,
               required_hours=13)
    elif args.action == "submit-advanced":
        if not (ROOT / "artifacts/selection_complete.json").is_file():
            raise RuntimeError("The complete cross-fitted selection must be verified first.")
        submit("advanced_training", ROOT / "gpu_train.slurm", arguments=["advanced"],
               gpu_hours=6, required_hours=12)
    elif args.action == "submit-eval":
        if args.role is None or not (ROOT / "runs" / args.role / "complete.json").is_file():
            raise RuntimeError("A complete finite final model is required before terminal evaluation.")
        submit(args.role + "_evaluation", ROOT / "gpu_eval.slurm", arguments=[args.role], gpu_hours=6)
    else:
        report()


if __name__ == "__main__":
    main()
