"""Choose an 8K-safe LLaMA-Factory checkpoint for matched 32K evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare import load_run
from .data import digest, write_json
from .report import summary_with_costs


def choose_safe(base: dict, rows: dict[str, dict], max_extra_truncations: int = 40,
                max_extra_repetition: float = 0.05) -> str | None:
    for row in rows.values():
        row["delta_pass_at_1"] = row["pass_at_1"] - base["pass_at_1"]
        row["delta_pass_at_8"] = row["pass_at_8"] - base["pass_at_8"]
        row["extra_truncated_samples"] = row["truncated_samples"] - base["truncated_samples"]
        row["extra_repeated_8gram_fraction"] = (
            row["mean_repeated_8gram_fraction"] - base["mean_repeated_8gram_fraction"]
        )
        row["passes_termination_screen"] = (
            row["extra_truncated_samples"] <= max_extra_truncations
            and row["extra_repeated_8gram_fraction"] <= max_extra_repetition
        )
    safe = [name for name, row in rows.items() if row["passes_termination_screen"]]
    return sorted(
        safe,
        key=lambda name: (
            -rows[name]["pass_at_8"],
            -rows[name]["pass_at_1"],
            rows[name]["truncated_samples"],
            name,
        ),
    )[0] if safe else None


def summarize_one(path: Path, benchmark: str) -> dict:
    _, data = load_run(path)
    if set(data) != {benchmark}:
        raise ValueError(f"{path}: expected only {benchmark}")
    result = summary_with_costs(data[benchmark])
    if result["problems"] != 200 or result["samples_per_problem"] != 8:
        raise ValueError(f"{path}: expected 200 questions x 8 samples")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True,
                        help="NAME=RESULT_DIR; repeat for each checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", default="acereason_dev")
    parser.add_argument("--max-extra-truncations", type=int, default=40)
    parser.add_argument("--max-extra-repetition", type=float, default=0.05)
    args = parser.parse_args()

    base = summarize_one(args.base, args.benchmark)
    rows = {}
    for item in args.candidate:
        name, sep, raw_path = str(item).partition("=")
        if not sep or not name or not raw_path:
            raise ValueError("candidate must be NAME=RESULT_DIR")
        path = Path(raw_path)
        row = summarize_one(path, args.benchmark)
        row["manifest_sha256"] = digest(path / "manifest.json")
        rows[name] = row

    selected = choose_safe(base, rows, args.max_extra_truncations, args.max_extra_repetition)
    result = {
        "schema": "reasoning-sft-lmf-control-selection-v1",
        "benchmark": args.benchmark,
        "rule": (
            "screen out checkpoints with >40 extra truncations or >0.05 extra mean repeated-8-gram "
            "fraction; among safe checkpoints choose pass@8, pass@1, fewer truncations"
        ),
        "base": base,
        "candidates": rows,
        "selected": selected,
        "base_manifest_sha256": digest(args.base / "manifest.json"),
        "analysis_code_sha256": digest(Path(__file__)),
    }
    write_json(args.output, result)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest(args.output)}  {args.output.name}\n"
    )
    print(result)


if __name__ == "__main__":
    main()
