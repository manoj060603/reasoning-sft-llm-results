"""Check strict per-column domination of the supplied Fall 2025 leader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LEADER = {
    # (printed score, problem count, inferred solved count).  The inferred
    # count recovers the only integer numerator that rounds to the table value.
    "aime": (0.233, 30, 7),
    "aime25": (0.233, 30, 7),
    "math": (0.900, 500, 450),
    "cn_math_2024": (0.433, 30, 13),
    "kaoyan": (0.688, 199, 137),
    "amc": (0.800, 40, 32),
    "minerva": (0.346, 272, 94),
    "olympiadbench": (0.569, 675, 384),
    "gpqa": (0.606, 198, 120),
}


def required_correct(leader_solves: int) -> int:
    """Minimum integer solves that strictly exceeds the historical count."""
    return leader_solves + 1


def domination_report(comparison: dict) -> dict:
    rows, all_pass = {}, True
    for name, (target, n, leader_solves) in LEADER.items():
        observed = comparison[name]["candidate"]["pass_at_8"]
        solves = round(observed * comparison[name]["candidate"]["problems"])
        needed = required_correct(leader_solves)
        passed = solves >= needed
        all_pass &= passed
        rows[name] = {
            "historical_leader": target,
            "candidate": observed,
            "candidate_solves": solves,
            "strict_target_solves": needed,
            "solve_gap": max(0, needed - solves),
            "passed": passed,
        }
    return {"all_nine_strictly_exceeded": all_pass, "benchmarks": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("comparison", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = domination_report(json.loads(args.comparison.read_text()))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["all_nine_strictly_exceeded"] else 2)


if __name__ == "__main__":
    main()

