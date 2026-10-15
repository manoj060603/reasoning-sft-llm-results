"""Render the measured capacity-matched rescue development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("results/arc-20261011-snapshot-2316"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    evaluations = args.snapshot / "evaluations"
    base = json.loads((evaluations / "rescue-base-dev/metrics.json").read_text())[
        "acereason_dev"
    ]
    candidate = json.loads(
        (evaluations / "rescue-candidate-dev/metrics.json").read_text()
    )["acereason_dev"]
    comparison = json.loads(
        (evaluations / "rescue-dev-comparison.json").read_text()
    )["acereason_dev"]

    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    labels = ["Pass@1", "Pass@8"]
    base_scores = [base["pass_at_1"] * 100, base["pass_at_8"] * 100]
    candidate_scores = [
        candidate["pass_at_1"] * 100,
        candidate["pass_at_8"] * 100,
    ]
    x = range(len(labels))
    width = 0.34
    axes[0].bar([i - width / 2 for i in x], base_scores, width, label="Base")
    axes[0].bar(
        [i + width / 2 for i in x], candidate_scores, width, label="Candidate"
    )
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 55)
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out accuracy")
    for i, values in enumerate(zip(base_scores, candidate_scores)):
        for offset, value in zip((-width / 2, width / 2), values):
            axes[0].text(i + offset, value + 1, f"{value:.1f}", ha="center")

    truncations = [base["truncated_samples"], candidate["truncated_samples"]]
    bars = axes[1].bar(["Base", "Candidate"], truncations, color=["#4c78a8", "#f58518"])
    axes[1].set_ylabel("Length-limited generations / 1,600")
    axes[1].set_title("Stopping behavior")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylim(0, 450)

    fig.suptitle("Capacity-matched route-DFT rescue: development gate rejected")
    fig.text(
        0.5,
        0.015,
        "Pass@8 delta -4.0 pp (95% CI -10.5 to +2.5); public suite skipped",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.3)
    fig.savefig(args.output / "rescue-development-gate.png", dpi=180)
    fig.savefig(args.output / "rescue-development-gate.pdf")
    (args.output / "rescue-development-gate-values.json").write_text(
        json.dumps(
            {
                "base": base,
                "candidate": candidate,
                "comparison": comparison,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
