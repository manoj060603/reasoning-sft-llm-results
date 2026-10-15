"""Select a shared 2K set of complete, concise, structurally clean teacher traces."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import alpaca, boxed, digest, read_rows, write_json
from .routes import route_select


def boxed_tail_chars(text: str) -> int | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    index = start + len(r"\boxed{")
    depth = 1
    while index < len(text):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return len(text) - index - 1
        index += 1
    return None


def structurally_clean(row: dict, feature: dict, max_response_tokens: int,
                       max_repetition: float, max_box_tail_chars: int) -> bool:
    output = row["output"]
    tail = boxed_tail_chars(output)
    return (
        feature["complete_box"]
        and boxed(output) is not None
        and feature["response_tokens"] <= max_response_tokens
        and feature["repetition"] <= max_repetition
        and output.count("<think>") == 1
        and output.count("</think>") == 1
        and output.count(r"\boxed{") == 1
        and tail is not None
        and tail <= max_box_tail_chars
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="artifacts/prepared_extended/pool.json")
    parser.add_argument("--features", default="artifacts/features_extended.json")
    parser.add_argument("--output", default="artifacts/datasets-concise-2k")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--cutoff", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=1536)
    parser.add_argument("--max-repetition", type=float, default=.10)
    parser.add_argument("--max-box-tail-chars", type=int, default=16)
    args = parser.parse_args()
    rows = read_rows(args.pool)
    features = read_rows(args.features)
    feature_map = {row["id"]: row for row in features}
    if len(feature_map) != len(features) or set(feature_map) != {row["id"] for row in rows}:
        raise ValueError("features must cover the exact pool once")
    eligible_rows = [row for row in rows if structurally_clean(
        row, feature_map[row["id"]], args.max_response_tokens,
        args.max_repetition, args.max_box_tail_chars
    )]
    if len(eligible_rows) < args.size:
        raise ValueError(f"only {len(eligible_rows)} structurally clean rows for size {args.size}")
    eligible_features = [feature_map[row["id"]] for row in eligible_rows]
    selected, route_eligible, quotas = route_select(
        eligible_rows, eligible_features, args.size, args.seed, args.cutoff
    )
    if len(selected) != len({row["id"] for row in selected}) or len(selected) != args.size:
        raise RuntimeError("selection is not exactly the requested unique set")
    random.Random(args.seed).shuffle(selected)
    output = Path(args.output)
    write_json(output / "concise.json", alpaca(selected))
    write_json(output / "dataset_info.json", {"concise": {"file_name": "concise.json"}})
    selected_features = [feature_map[row["id"]] for row in selected]
    manifest = {
        "schema": "reasoning-sft-concise-2k-v1",
        "arguments": vars(args),
        "pool_sha256": digest(args.pool),
        "features_sha256": digest(args.features),
        "selection_code_sha256": digest(Path(__file__)),
        "eligible_rows": len(eligible_rows),
        "route_eligible_rows": len(route_eligible),
        "selected_rows": len(selected),
        "selected_ids": [row["id"] for row in selected],
        "cluster_quotas": quotas,
        "response_tokens": {
            "total": sum(row["response_tokens"] for row in selected_features),
            "mean": sum(row["response_tokens"] for row in selected_features) / len(selected_features),
            "min": min(row["response_tokens"] for row in selected_features),
            "max": max(row["response_tokens"] for row in selected_features),
        },
        "complete_box_rows": sum(row["complete_box"] for row in selected_features),
        "max_observed_repetition": max(row["repetition"] for row in selected_features),
        "data_sha256": None,
        "source_sha256": {
            "protocol": digest(Path(__file__).resolve().parents[1] / "CONCISE_TARGET_EXPERIMENT.md"),
            "selector": digest(Path(__file__)),
            "trainer": digest(Path(__file__).resolve().parent / "train_adapter.py"),
            "evaluator": digest(Path(__file__).resolve().parent / "evaluate.py"),
            "finalizer": digest(Path(__file__).resolve().parent / "choose_concise.py"),
            "runner": digest(Path(__file__).resolve().parents[1] / "scripts/arc_concise_experiment.sh"),
            "launcher": digest(Path(__file__).resolve().parents[1] / "scripts/concise_experiment.slurm"),
        },
        "evidence_boundary": "Teacher answers are preserved unchanged from the approved pool and structurally checked; their mathematical correctness is not independently reverified.",
    }
    manifest["data_sha256"] = digest(output / "concise.json")
    write_json(output / "selection_manifest.json", manifest)
    (output / "selection_manifest.json.sha256").write_text(
        f"{digest(output / 'selection_manifest.json')}  selection_manifest.json\n"
    )
    print({"eligible": len(eligible_rows), "selected": len(selected),
           "response_tokens": manifest["response_tokens"],
           "manifest_sha256": digest(output / "selection_manifest.json")})


if __name__ == "__main__":
    main()
