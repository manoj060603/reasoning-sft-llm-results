"""Select a weight-interpolation coefficient on the complete 32K development panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare import load_run
from .data import digest, write_json
from .report import summary_with_costs


def summarize_run(path: Path, benchmark: str) -> dict:
    _, data = load_run(path)
    if set(data) != {benchmark}:
        raise ValueError(f"{path}: expected only {benchmark}")
    row = summary_with_costs(data[benchmark])
    if row["problems"] != 200 or row["samples_per_problem"] != 8:
        raise ValueError(f"{path}: expected 200 questions x 8 samples")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=RESULT_DIRECTORY; repeat for each alpha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", default="acereason_dev")
    args = parser.parse_args()
    candidates = {}
    for spec in args.candidate:
        name, raw_path = spec.split("=", 1)
        candidates[name] = Path(raw_path)
    if not candidates:
        raise ValueError("at least one candidate is required")

    paths = {"base": args.base, "full": args.full, **candidates}
    runs = {name: summarize_run(path, args.benchmark) for name, path in paths.items()}
    base = runs["base"]
    for name, row in runs.items():
        row["delta_pass_at_1"] = row["pass_at_1"] - base["pass_at_1"]
        row["delta_pass_at_8"] = row["pass_at_8"] - base["pass_at_8"]
        row["extra_truncated_samples"] = row["truncated_samples"] - base["truncated_samples"]
        row["extra_repeated_8gram_fraction"] = (
            row["mean_repeated_8gram_fraction"] - base["mean_repeated_8gram_fraction"]
        )
    selected = sorted(candidates, key=lambda name: (
        -runs[name]["pass_at_8"], -runs[name]["pass_at_1"],
        runs[name]["truncated_samples"], runs[name]["mean_repeated_8gram_fraction"], name,
    ))[0]
    result = {
        "schema": "reasoning-sft-weight-interpolation-selection-v1",
        "benchmark": args.benchmark,
        "rule": "among interpolated models choose pass@8, then pass@1, fewer truncations, less repeated-8-gram fraction",
        "selected": selected,
        "runs": runs,
        "manifest_sha256": {name: digest(path / "manifest.json") for name, path in paths.items()},
        "analysis_code_sha256": digest(Path(__file__)),
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
