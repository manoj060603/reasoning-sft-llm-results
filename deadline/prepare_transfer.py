"""Prepare a fresh source-only cross-fitted transfer selection experiment.

Run on a compute node. Original source targets are never rewritten. Public
benchmark questions enter exclusion checks only; their answers are not read.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

from reasoning.data import alpaca, boxed, digest, question, read_rows, write_json
from reasoning.features import repetition, token_ids
from reasoning.prepare import ExclusionIndex, prepare
from reasoning.select import allocate

SOURCE_REVISION = "ce815bd726c0c4ffdc0912789754646910188195"
SOURCE_SHA256 = "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def order_key(identifier, purpose, seed):
    return hashlib.sha256(f"{seed}:{purpose}:{identifier}".encode()).hexdigest()


def fold_of(identifier):
    return int(identifier[:8], 16) % 2


def make_cache_record(tokenizer, row, fold, cutoff=None):
    prompt, full = token_ids(tokenizer, row)
    if cutoff is not None:
        full = full[:cutoff]
    if len(prompt) >= len(full):
        raise ValueError("No supervised target remains: " + row["id"])
    return {"id": row["id"], "input_ids": full,
            "labels": [-100] * len(prompt) + full[len(prompt):],
            "fold": fold, "question_sha256": row["id"]}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def balanced_reservoir(features, size, seed):
    groups = defaultdict(list)
    for feature in features:
        groups[(feature["cluster"], min(15, feature["response_tokens"] // 512))].append(feature["id"])
    quotas = allocate({key: len(value) for key, value in groups.items()}, size)
    return sorted(identifier for key, values in groups.items()
                  for identifier in sorted(values, key=lambda x: order_key(x, "reservoir", seed))[:quotas[key]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="deadline/artifacts")
    parser.add_argument("--cutoff", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--reservoir-size", type=int, default=30000)
    parser.add_argument("--probe-size", type=int, default=512)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Preparation must run in an ARC compute allocation.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "preparation_complete.json").exists():
        print("Preparation already complete; no data is overwritten.", flush=True)
        return
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    started = time.monotonic()
    print("Downloading the pinned AceReason source.", flush=True)
    source = Path(hf_hub_download(repo_id="redsgnaoh/acereason11_100k", repo_type="dataset",
                  revision=SOURCE_REVISION, filename="data/acereason11_100k.json",
                  local_dir=str(out / "source")))
    source_sha = digest(source)
    if source_sha != SOURCE_SHA256:
        raise ValueError("Pinned source hash differs from the published source receipt.")
    benchmark_paths = sorted(Path("eval/data").glob("*/test.jsonl"))
    if len(benchmark_paths) != 8:
        raise ValueError("The eight assignment benchmark files are required.")
    exclusions = [question(row) for path in benchmark_paths for row in read_rows(path)]
    # Supplementary AIME25, when present, is used only as an additional question exclusion.
    supplementary = out / "benchmarks/aime25/test.jsonl"
    if supplementary.is_file():
        exclusions.extend(question(row) for row in read_rows(supplementary))
    pool, heldout, preparation = prepare(read_rows(source), ExclusionIndex(exclusions),
                                        val_size=1000, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_hash = hashlib.sha256(json.dumps({path.name: digest(path)
                         for path in sorted(Path(args.model).glob("*.json"))
                         if "model.safetensors" not in path.name}, sort_keys=True).encode()).hexdigest()
    features, retained = [], []
    for index, row in enumerate(pool):
        prompt, full = token_ids(tokenizer, row)
        # A public structural input bound applies equally to both arms. It avoids
        # truncating the question or silently discarding random-arm records later.
        if len(prompt) > args.cutoff - 128:
            preparation["prompt_capacity_excluded"] = preparation.get("prompt_capacity_excluded", 0) + 1
            continue
        feature = {"id": row["id"], "cluster": -1, "total_tokens": len(full),
                   "response_tokens": len(full) - len(prompt),
                   "complete_box": boxed(row["output"]) is not None,
                   "repetition": repetition(row["output"])}
        features.append(feature)
        retained.append(row)
        if (index + 1) % 2000 == 0:
            print(json.dumps({"tokenized": index + 1, "source_pool": len(pool),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    pool = retained
    # Preserve the expensive tokenization audit even if a later capacity or
    # clustering check fails. These are diagnostics, not completion receipts.
    write_json(out / "tokenization_pool.json", pool)
    write_json(out / "tokenization_features.json", features)
    lookup = {row["id"]: row for row in pool}
    eligible = [feature for feature in features if 128 <= feature["response_tokens"]
                and feature["total_tokens"] <= args.cutoff - 32
                and feature["complete_box"] and feature["repetition"] <= .15
                and lookup[feature["id"]]["output"].count("<think>")
                    == lookup[feature["id"]]["output"].count("</think>")]
    if len(eligible) < 15000 + 2 * args.probe_size:
        raise ValueError(f"Only {len(eligible)} eligible complete targets; cannot fill the fixed design.")
    print(json.dumps({"stage": "tokenization_complete", "prepared_pool": len(pool),
                      "eligible_complete_targets": len(eligible), "cutoff": args.cutoff}), flush=True)
    print("Clustering structurally eligible questions.", flush=True)
    texts = [lookup[feature["id"]]["instruction"] for feature in eligible]
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=20000,
                             sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    representation = normalize(TruncatedSVD(n_components=64, random_state=args.seed).fit_transform(matrix))
    clusters = MiniBatchKMeans(n_clusters=64, random_state=args.seed, n_init=3,
                               batch_size=1024).fit_predict(representation)
    for feature, cluster in zip(eligible, clusters):
        feature["cluster"] = int(cluster)
    probe_ids = {}
    for fold, name in [(0, "probeA"), (1, "probeB")]:
        ids = [feature["id"] for feature in eligible if fold_of(feature["id"]) == fold]
        if len(ids) < args.probe_size:
            raise ValueError("A probe fold has insufficient structural capacity.")
        probe_ids[name] = sorted(ids, key=lambda identifier: order_key(identifier, name, args.seed))[:args.probe_size]
        write_json(out / f"{name}_ids.json", probe_ids[name])
    reserved = set(probe_ids["probeA"]) | set(probe_ids["probeB"])
    acquisition = [feature for feature in eligible if feature["id"] not in reserved]
    reservoir = balanced_reservoir(acquisition, min(args.reservoir_size, len(acquisition)), args.seed)
    all_cache_ids = set(reservoir) | reserved
    write_json(out / "pool.json", pool)
    write_json(out / "heldout.json", heldout)
    pool_sha = digest(out / "pool.json")
    write_json(out / "features.json", features)
    write_json(out / "features.json.meta.json", {"input_sha256": pool_sha,
                "source_sha256": source_sha, "tokenizer_sha256": tokenizer_hash,
                "model_revision": MODEL_REVISION, "resolved_tokenizer_commit": MODEL_REVISION,
                "provenance": {"code_sha256": digest(__file__),
                               "local_tokenizer_files": {path.name: digest(path) for path in
                                 sorted(Path(args.model).glob("*.json"))
                                 if "model.safetensors" not in path.name}},
                "arguments": vars(args)})
    fold_assignments = {identifier: fold_of(identifier) for identifier in all_cache_ids}
    for fold in (0, 1):
        write_json(out / f"fold{fold}_ids.json", [identifier for identifier in reservoir
                                                 if fold_assignments[identifier] == fold])
    cache_path = out / "transfer_cache.jsonl"
    write_jsonl(cache_path, (make_cache_record(tokenizer, lookup[identifier], fold_assignments[identifier])
                           for identifier in sorted(all_cache_ids)))
    write_json(out / "cache_meta.json", {"cache_sha256": digest(cache_path),
        "model_path": args.model, "model_revision": MODEL_REVISION,
        "tokenizer_sha256": tokenizer_hash, "source_sha256": source_sha,
        "expected_rows": len(all_cache_ids), "cutoff": args.cutoff,
        "complete_original_targets": True, "prefix_truncated_count": 0})
    transfer_meta = {"input_sha256": pool_sha, "base_model_sha256": MODEL_REVISION,
        "tokenizer_sha256": tokenizer_hash, "source_sha256": source_sha,
        "probe_training_ids": probe_ids, "fold_assignments": fold_assignments,
        "fold_to_probe": {"0": "probeB", "1": "probeA"}, "reservoir_ids": reservoir,
        "provenance": {"seed": args.seed, "method": "cross_fitted_transfer_gain_with_coverage",
                       "cutoff": args.cutoff, "probe_rows_per_fold": args.probe_size,
                       "score": "(base_ce-probe_ce)/max(base_ce,1e-8)"}}
    write_json(out / "transfer_meta.json", transfer_meta)
    receipt = {"source_revision": SOURCE_REVISION, "source_sha256": source_sha,
        "model_revision": MODEL_REVISION, "pool_sha256": pool_sha,
        "counts": {"pool": len(pool), "heldout": len(heldout), "eligible": len(eligible),
                   "reservoir": len(reservoir), "probes": len(reserved)},
        "preparation": preparation, "benchmark_files": {str(path): digest(path) for path in benchmark_paths},
        "eligible_response_tokens": sum(feature["response_tokens"] for feature in eligible),
        "elapsed_seconds": time.monotonic() - started, "arguments": vars(args)}
    write_json(out / "preparation_complete.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
