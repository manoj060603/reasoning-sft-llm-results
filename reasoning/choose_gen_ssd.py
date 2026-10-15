"""Compare matched existing-target and student-guided full-SFT development runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare import load_run
from .data import digest, write_json
from .report import summary_with_costs


def one(path: Path, benchmark: str) -> dict:
    _, data = load_run(path)
    if set(data) != {benchmark}:
        raise ValueError(f"{path}: expected only {benchmark}")
    result = summary_with_costs(data[benchmark])
    if result["problems"] != 200 or result["samples_per_problem"] != 8:
        raise ValueError("expected 200 questions x 8 samples")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--existing", type=Path, required=True)
    p.add_argument("--guided", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", default="acereason_dev")
    p.add_argument("--max-extra-truncations", type=int, default=40)
    a = p.parse_args()
    runs = {name: one(path, a.benchmark) for name, path in {
        "base": a.base, "existing": a.existing, "guided": a.guided,
    }.items()}
    control, guided = runs["existing"], runs["guided"]
    selected = "guided" if (
        guided["pass_at_8"] > control["pass_at_8"]
        and guided["truncated_samples"] <= control["truncated_samples"] + a.max_extra_truncations
    ) else None
    result = {
        "schema": "reasoning-sft-gen-ssd-2k-selection-v1",
        "rule": "guided pass@8 must exceed matched existing-target full SFT and add no more than 40 truncations",
        "selected": selected, "runs": runs,
        "delta_guided_vs_existing": {
            "pass_at_1": guided["pass_at_1"] - control["pass_at_1"],
            "pass_at_8": guided["pass_at_8"] - control["pass_at_8"],
            "truncated_samples": guided["truncated_samples"] - control["truncated_samples"],
        },
        "manifest_sha256": {name: digest(path / "manifest.json") for name, path in {
            "base": a.base, "existing": a.existing, "guided": a.guided,
        }.items()},
        "analysis_code_sha256": digest(Path(__file__)),
    }
    write_json(a.output, result)
    print(result)


if __name__ == "__main__":
    main()
