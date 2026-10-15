"""Apply the sealed development gate to matched concise-target SFT/DFT runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare import load_run
from .data import digest, write_json
from .report import summary_with_costs


def one(run: Path, benchmark: str) -> dict:
    _, data = load_run(run)
    if set(data) != {benchmark}:
        raise ValueError(f"{run}: expected only {benchmark}")
    result = summary_with_costs(data[benchmark])
    if result["problems"] != 200 or result["samples_per_problem"] != 8:
        raise ValueError(f"{run}: expected 200 questions x 8 samples")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", default="acereason_dev")
    parser.add_argument("--max-extra-truncations", type=int, default=40)
    args = parser.parse_args()
    runs = {"base": one(args.base, args.benchmark),
            "sft": one(args.sft, args.benchmark),
            "dft": one(args.dft, args.benchmark)}
    base = runs["base"]
    feasible = []
    for name in ("sft", "dft"):
        row = runs[name]
        row["delta_pass_at_1"] = row["pass_at_1"] - base["pass_at_1"]
        row["delta_pass_at_8"] = row["pass_at_8"] - base["pass_at_8"]
        row["extra_truncated_samples"] = row["truncated_samples"] - base["truncated_samples"]
        row["passes_gate"] = (
            row["pass_at_8"] >= base["pass_at_8"]
            and row["extra_truncated_samples"] <= args.max_extra_truncations
        )
        if row["passes_gate"]:
            feasible.append(name)
    selected = sorted(
        feasible,
        key=lambda name: (-runs[name]["pass_at_8"], -runs[name]["pass_at_1"],
                          runs[name]["truncated_samples"], name),
    )[0] if feasible else None
    result = {
        "schema": "reasoning-sft-concise-2k-selection-v1",
        "benchmark": args.benchmark,
        "rule": "candidate pass@8 >= base and extra truncations <= 40; choose pass@8, then pass@1, then fewer truncations",
        "selected": selected,
        "runs": runs,
        "manifest_sha256": {
            name: digest(path / "manifest.json")
            for name, path in {"base": args.base, "sft": args.sft, "dft": args.dft}.items()
        },
        "analysis_code_sha256": digest(Path(__file__)),
    }
    write_json(args.output, result)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest(args.output)}  {args.output.name}\n"
    )
    print(result)


if __name__ == "__main__":
    main()
