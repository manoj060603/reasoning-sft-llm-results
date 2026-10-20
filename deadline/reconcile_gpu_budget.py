"""Record exact terminal GPU use without changing any original reservation.

Run on the Owl login node for lightweight scheduler/ledger metadata only.
This command submits no jobs. All unverified reservations remain fully charged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from deadline.gpu_budget import (conservative_budget, make_terminal_receipt,
                                 reconcile_ledger)
from deadline.pipeline import scheduler_live_ids
from reasoning.data import write_json


def reconcile(roles, root=Path("deadline")):
    with (root / "job_ledger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "job_ledger.json"
        ledger = json.loads(path.read_text())
        current = scheduler_live_ids()
        receipts = {}
        for role in roles:
            entry = ledger["jobs"][role]
            if entry.get("gpu_accounting_receipt") is not None:
                continue
            evidence = subprocess.run(["sacct", "--clusters=owl", "--allocations",
                "--duplicates", "--parsable2", "--noheader", "--jobs=" + str(entry["job_id"]),
                "--format=JobIDRaw,JobID,State%40,ElapsedRaw,AllocTRES%500"],
                check=True, capture_output=True, text=True, timeout=60)
            receipts[role] = make_terminal_receipt(role, entry, evidence.stdout,
                live_job_ids=current, observed_at_utc=datetime.now(timezone.utc).isoformat())
        # Recheck after reading accounting; restarted/active jobs fail closed.
        current = scheduler_live_ids()
        if receipts:
            ledger, audit = reconcile_ledger(ledger, receipts, live_job_ids=current)
            # The ledger includes the complete append-only audit in this write.
            write_json(path, ledger)
            audit_path = root / "operations" / ("gpu-budget-reconciliation-" +
                audit["audit_sha256"][:16] + ".json")
            write_json(audit_path, audit)
        result = conservative_budget(ledger, live_job_ids=current)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", nargs="+", choices=("original_evaluation", "probes_and_scores",
        "random_training", "advanced_training", "random_evaluation", "advanced_evaluation",
        "random_evaluation_continuation", "advanced_evaluation_continuation"))
    args = parser.parse_args()
    reconcile(args.roles)


if __name__ == "__main__":
    main()
