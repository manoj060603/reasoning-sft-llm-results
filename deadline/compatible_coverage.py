"""Benchmark-blind, cross-fitted transfer-gain selection of original SFT pairs.

Primary policy: within square-root cluster quotas and proportional length cells,
take at most half from positive transfer-gain leaders and fill with seeded
coverage. A separately named coverage policy is available without probe scores.
No target is shortened, rewritten, or graded by this module.
"""
import argparse
import copy
import hashlib
import heapq
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from reasoning import features as feature_module
from reasoning.data import alpaca, boxed, digest, prompt_id, read_rows, write_json
from reasoning.prepare import ExclusionIndex


POLICY_VERSION = "crossfit-transfer-coverage-v1"
COLUMNS = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rank(seed, identifier, context=""):
    return hashlib.sha256(f"{seed}:{context}:{identifier}".encode()).hexdigest()


def allocate(capacities, size, square_root=False):
    """Capacity-capped highest-averages allocation; deterministic for all ties."""
    if type(size) is not int or size < 0:
        raise ValueError("selection size must be a nonnegative integer")
    if any(type(n) is not int or n < 0 for n in capacities.values()):
        raise ValueError("invalid cell capacities")
    if sum(capacities.values()) < size:
        raise ValueError(f"selection shortage: need {size}, have {sum(capacities.values())}")
    quotas = {key: 0 for key in capacities}
    weights = {key: math.sqrt(n) if square_root else n for key, n in capacities.items()}
    queue = [(-weights[key], str(key), key) for key, n in capacities.items() if n]
    heapq.heapify(queue)
    for _ in range(size):
        _, _, key = heapq.heappop(queue)
        quotas[key] += 1
        if quotas[key] < capacities[key]:
            heapq.heappush(queue, (-weights[key] / (quotas[key] + 1), str(key), key))
    return quotas


def _validate_pool(rows, features, feature_meta, pool_sha256):
    if not rows:
        raise ValueError("empty canonical pool")
    if feature_meta.get("input_sha256") != pool_sha256:
        raise ValueError("stale features: input SHA-256 does not match the pool")
    provenance = feature_meta.get("provenance", {})
    if provenance.get("code_sha256") != digest(feature_module.__file__):
        raise ValueError("stale features: reasoning.features code SHA-256 differs")
    tokenizer_identity = (feature_meta.get("resolved_tokenizer_commit") or
                          provenance.get("tokenizer_commit") or provenance.get("local_tokenizer_files"))
    if not tokenizer_identity:
        raise ValueError("feature metadata must identify the tokenizer commit or local tokenizer files")
    by_id = {}
    for row in rows:
        if any(not isinstance(row.get(k), str) for k in ("id", "instruction", "input", "output", "system")):
            raise ValueError("canonical rows require id/instruction/input/output/system strings")
        if row["input"] or not row["instruction"].strip() or not row["output"].strip() or not row["system"].strip():
            raise ValueError("canonical rows require folded input and nonempty prompt/target/system")
        if row["id"] != prompt_id(row["instruction"]):
            raise ValueError("canonical prompt ID does not match the question")
        if row["id"] in by_id:
            raise ValueError("duplicate canonical prompt ID")
        by_id[row["id"]] = row
    fmap = {}
    for feature in features:
        identifier = feature.get("id")
        if identifier in fmap:
            raise ValueError("duplicate feature ID")
        fmap[identifier] = feature
        for key in ("total_tokens", "response_tokens", "cluster"):
            if type(feature.get(key)) is not int or feature[key] < (0 if key == "cluster" else 1):
                raise ValueError(f"invalid feature {key}")
        if feature["response_tokens"] >= feature["total_tokens"]:
            raise ValueError("response tokens must leave a nonempty serialized prompt")
        if type(feature.get("complete_box")) is not bool:
            raise ValueError("complete_box must be boolean")
        rep = feature.get("repetition")
        if isinstance(rep, bool) or not isinstance(rep, (int, float)) or not math.isfinite(rep) or not 0 <= rep <= 1:
            raise ValueError("invalid repetition feature")
    if set(fmap) != set(by_id):
        raise ValueError("feature IDs must cover the exact canonical pool once")
    for identifier, row in by_id.items():
        if fmap[identifier]["complete_box"] != (boxed(row["output"]) is not None):
            raise ValueError("stale features: final boxed-answer flag disagrees with original target")
    return by_id, fmap


def _heldout_exclusions(rows, heldout_rows):
    if not heldout_rows:
        raise ValueError("all source held-out rows must be explicitly supplied")
    heldout_ids, questions = set(), []
    for row in heldout_rows:
        identifier = row.get("id")
        question = row.get("instruction", row.get("question", row.get("problem")))
        if isinstance(question, str) and question.strip():
            if row.get("input"):
                question += "\n" + row["input"]
            questions.append(question)
            heldout_ids.add(prompt_id(question))
        if isinstance(identifier, str) and identifier:
            heldout_ids.add(identifier)
        elif not isinstance(question, str) or not question.strip():
            raise ValueError("held-out row must identify a source question or prompt ID")
    index = ExclusionIndex(questions)
    excluded = {r["id"] for r in rows if r["id"] in heldout_ids or index.matches(r["instruction"])}
    return excluded, heldout_ids


def eligible(row, feature, cutoff):
    target = row["output"]
    tags_balanced = (target.count("<think>") == target.count("</think>") and
                     ("<think>" not in target or target.find("<think>") < target.find("</think>")))
    return (feature["total_tokens"] <= cutoff - 32 and feature["response_tokens"] >= 128
            and feature["complete_box"] and boxed(target) is not None
            and feature["repetition"] <= .15 and tags_balanced)


def select_random_control(rows, *, heldout_rows, probe_ids=(), size=15000, seed=42):
    """Pre-score uniform control with the same ordering and reservations as export.

    Returns canonical ``rows`` unchanged plus the reservation/sampling receipt.
    This needs no scores; callers can export ``alpaca(result['rows'])`` before
    probe training. Supply the complete source-heldout sets and BOTH probe sets.
    """
    if type(size) is not int or size <= 0:
        raise ValueError("random control size must be a positive integer")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError("random control requires unique canonical source IDs")
    probes = set(probe_ids)
    if not probes.issubset(ids):
        raise ValueError("every reserved probe ID must belong to the source pool")
    excluded, _ = _heldout_exclusions(rows, heldout_rows)
    reserved = excluded | probes
    available = sorted((row for row in rows if row["id"] not in reserved), key=lambda row: row["id"])
    if len(available) < size:
        raise ValueError(f"random pool shortage: need {size}, have {len(available)}")
    return dict(rows=copy.deepcopy(random.Random(seed).sample(available, size)),
                source_heldout_excluded_ids=sorted(excluded), reserved_probe_ids=sorted(probes),
                sampling_pool_rows=len(available))


def _validate_transfer(scores, meta, pool_sha256, by_id, excluded, fmap, cutoff):
    """Require two disjoint folds, opposite-fold probes, and exact CE arithmetic.

    Score rows: id, fold, probe_id, base_ce, probe_ce, gain.
    Metadata: input_sha256, base_model_sha256, tokenizer_sha256,
    reservoir_ids [id, ...], fold_assignments {id: fold},
    fold_to_probe {fold: probe_id}, and probe_training_ids {probe_id: [id, ...]}.
    Assignments cover exactly the scored reservoir plus the reserved probes.
    Probe training IDs must belong to the opposite candidate fold.
    """
    if not scores or not meta:
        raise ValueError("transfer_gain requires scores and cross-fit provenance")
    if meta.get("input_sha256") != pool_sha256:
        raise ValueError("stale transfer scores: pool SHA-256 mismatch")
    for key in ("base_model_sha256", "tokenizer_sha256"):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"transfer provenance requires {key}")
    raw_folds = meta.get("fold_assignments", {})
    if not isinstance(raw_folds, dict) or not raw_folds or any(
            type(value) not in (str, int) for value in raw_folds.values()):
        raise ValueError("fold_assignments must map source IDs to integer or string fold labels")
    folds = {identifier: str(value) for identifier, value in raw_folds.items()}
    routes, training = meta.get("fold_to_probe", {}), meta.get("probe_training_ids", {})
    reservoir = meta.get("reservoir_ids")
    if (not isinstance(reservoir, list) or not reservoir or
            len(set(reservoir)) != len(reservoir)):
        raise ValueError("cross-fit metadata requires unique reservoir_ids")
    reservoir_ids = set(reservoir)
    if len(set(folds.values())) != 2 or set(routes) != set(folds.values()) or len(set(routes.values())) != 2:
        raise ValueError("exactly two folds and two distinct opposite-fold probes are required")
    if set(training) != set(routes.values()):
        raise ValueError("probe training provenance must cover exactly the two routed probes")
    initializations = meta.get("probe_initialization_sha256")
    if not isinstance(initializations, dict) or (set(initializations) != set(training) or
            any(value != meta["base_model_sha256"] for value in initializations.values())):
        raise ValueError("probe initializations must match the pinned original base")
    checkpoints = meta.get("probe_checkpoint_sha256")
    if not isinstance(checkpoints, dict) or (set(checkpoints) != set(training) or any(
            not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value) for value in checkpoints.values())):
        raise ValueError("invalid probe checkpoint SHA-256 provenance")
    if not set(folds).issubset(by_id) or set(folds) & excluded:
        raise ValueError("fold IDs must be source-only and disjoint from all held-out rows")
    training_sets = {}
    for probe, identifiers in training.items():
        if not isinstance(identifiers, list) or not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("probe training IDs must be nonempty and unique")
        if not set(identifiers).issubset(folds):
            raise ValueError("every probe training ID must have a recorded source fold")
        training_sets[probe] = set(identifiers)
        for candidate_fold, routed_probe in routes.items():
            if probe == routed_probe and any(folds[identifier] == candidate_fold for identifier in identifiers):
                raise ValueError("cross-fit leakage: a probe trained on its candidate fold")
    if training_sets[next(iter(training_sets))] & training_sets[next(reversed(training_sets))]:
        raise ValueError("cross-fit probe training sets overlap")
    probe_ids = set().union(*training_sets.values())
    if reservoir_ids & probe_ids:
        raise ValueError("probe training IDs must be reserved from the final candidate reservoir")
    if set(folds) != reservoir_ids | probe_ids:
        raise ValueError("fold assignments must cover exactly reservoir and reserved probe IDs")
    if any(not eligible(by_id[identifier], fmap[identifier], cutoff) for identifier in probe_ids):
        raise ValueError("probe training contains an ineligible source trace")
    result = {}
    for score in scores:
        identifier = score.get("id")
        if identifier in result:
            raise ValueError("duplicate transfer score ID")
        if identifier not in reservoir_ids or identifier in excluded:
            raise ValueError("transfer score must belong to an unreserved source fold")
        fold, probe = folds[identifier], routes[folds[identifier]]
        if (type(score.get("fold")) not in (str, int) or str(score["fold"]) != fold or
                score.get("probe_id") != probe or identifier in training_sets[probe]):
            raise ValueError("cross-fit score/probe/fold mismatch")
        if not eligible(by_id[identifier], fmap[identifier], cutoff):
            raise ValueError("transfer reservoir contains an ineligible original trace")
        values = [score.get(key) for key in ("base_ce", "probe_ce", "gain")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("nonfinite or missing transfer CE/gain")
        base_ce, probe_ce, gain = values
        if base_ce <= 0 or probe_ce < 0:
            raise ValueError("transfer scoring requires positive base CE and nonnegative probe CE")
        expected = (base_ce - probe_ce) / max(base_ce, 1e-8)
        if not math.isclose(gain, expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("reported transfer gain disagrees with CE values")
        result[identifier] = gain
    if set(result) != reservoir_ids:
        raise ValueError("every reservoir candidate requires exactly one cross-fitted score")
    return result, probe_ids


def _length_stratum(tokens):
    # Fixed bins, never chosen using benchmark outcomes.
    for boundary in (1024, 2048, 4096, 8192, 16384):
        if tokens < boundary:
            return str(boundary)
    return "16384+"


def _choose(rows, fmap, gains, size, seed):
    clusters = defaultdict(lambda: defaultdict(list))
    for row in rows:
        f = fmap[row["id"]]
        clusters[str(f["cluster"])][_length_stratum(f["response_tokens"])].append(row)
    cluster_quotas = allocate({k: sum(map(len, cells.values())) for k, cells in clusters.items()},
                              size, square_root=True)
    selected, receipt = [], {}
    for cluster in sorted(clusters):
        cells = clusters[cluster]
        quotas = allocate({key: len(cell) for key, cell in cells.items()}, cluster_quotas[cluster])
        for stratum in sorted(cells):
            cell, quota = cells[stratum], quotas[stratum]
            positive = sorted((r for r in cell if gains is not None and gains[r["id"]] > 0),
                              key=lambda r: (-gains[r["id"]], _rank(seed, r["id"], "gain")))
            leaders = positive[:quota // 2]
            leader_ids = {row["id"] for row in leaders}
            coverage = sorted((r for r in cell if r["id"] not in leader_ids),
                              key=lambda r: _rank(seed, r["id"], "coverage"))[:quota - len(leaders)]
            selected.extend(leaders + coverage)
            receipt[f"{cluster}:{stratum}"] = dict(capacity=len(cell), quota=quota,
                                                     gain_leaders=len(leaders), coverage=len(coverage))
    selected.sort(key=lambda r: _rank(seed, r["id"], "training-order"))
    return selected, receipt


def select_datasets(rows, features, *, feature_meta, pool_sha256, heldout_rows,
                    size=15000, cutoff=4096, seed=42, strategy="transfer_gain",
                    transfer_scores=None, transfer_meta=None):
    """Return unchanged canonical datasets, explicit loader mapping and manifest.

    The random control is uniform over the complete prepared pool after source
    held-out exclusions, including overlength rows; exports are never truncated.
    Advanced selection is restricted to complete, context-safe original traces.
    """
    if type(size) is not int or size <= 0 or type(cutoff) is not int or cutoff <= 160:
        raise ValueError("positive size and usable cutoff are required")
    if strategy not in ("transfer_gain", "coverage"):
        raise ValueError("choose transfer_gain or explicitly named coverage policy")
    by_id, fmap = _validate_pool(rows, features, feature_meta, pool_sha256)
    excluded, heldout_ids = _heldout_exclusions(rows, heldout_rows)
    gains = None
    probe_ids = set()
    if strategy == "transfer_gain":
        gains, probe_ids = _validate_transfer(transfer_scores, transfer_meta, pool_sha256, by_id,
                                              excluded, fmap, cutoff)
    elif transfer_scores is not None or transfer_meta is not None:
        raise ValueError("coverage policy must not silently ignore supplied transfer scores")
    reserved_ids = excluded | probe_ids
    available = [by_id[key] for key in sorted(by_id) if key not in reserved_ids]
    if len(available) < size:
        raise ValueError(f"random pool shortage: need {size}, have {len(available)}")
    candidates = [r for r in available if eligible(r, fmap[r["id"]], cutoff)
                  and (gains is None or r["id"] in gains)]
    if len(candidates) < size:
        raise ValueError(f"advanced eligible reservoir shortage: need {size}, have {len(candidates)}")
    selected, cells = _choose(candidates, fmap, gains, size, seed)
    control = random.Random(seed).sample(available, size)
    datasets = {"advanced": selected, "random": control}
    summaries = {}
    for name, records in datasets.items():
        if len(records) != size or len({r["id"] for r in records}) != size:
            raise AssertionError("selection must preserve the exact unique example budget")
        summaries[name] = dict(
            rows=len(records), selected_ids=[r["id"] for r in records],
            original_targets_sha256=content_sha256([{k: r[k] for k in ("id", "output")} for r in records]),
            exported_rows_sha256=content_sha256(alpaca(records)),
            serialized_tokens=sum(fmap[r["id"]]["total_tokens"] for r in records),
            response_tokens=sum(fmap[r["id"]]["response_tokens"] for r in records),
            over_cutoff_ids=[r["id"] for r in records if fmap[r["id"]]["total_tokens"] > cutoff],
            beyond_safety_margin_ids=[r["id"] for r in records if fmap[r["id"]]["total_tokens"] > cutoff - 32])
    manifest = dict(
        policy_version=POLICY_VERSION, strategy=strategy,
        policy_label=("cross-fitted transfer gain with source coverage" if gains is not None else
                      "length/diversity coverage; no student-aware signal"),
        size=size, cutoff=cutoff, safety_margin=32, seed=seed,
        source_pool_sha256=pool_sha256, canonical_content_sha256=content_sha256(sorted(rows, key=lambda r: r["id"])),
        feature_content_sha256=content_sha256(sorted(features, key=lambda f: f["id"])),
        feature_meta_sha256=content_sha256(feature_meta), code_sha256=digest(__file__),
        heldout_source_ids_sha256=content_sha256(sorted(heldout_ids)),
        source_heldout_excluded_ids=sorted(excluded), reserved_probe_ids=sorted(probe_ids),
        prepared_pool_rows=len(rows),
        random_sampling_pool_rows=len(available), advanced_reservoir_rows=len(candidates),
        random_policy="uniform prepared pool after source-heldout and probe exclusions; original export untruncated",
        training_truncation_warning="Random control may truncate during training; verify and report actual processor counts and target/EOS survival.",
        target_modifications=0, system_mapping=COLUMNS, cells=cells, datasets=summaries,
        transfer_provenance=copy.deepcopy(transfer_meta) if gains is not None else None)
    if gains is not None:
        manifest["transfer_summary"] = dict(
            scored_candidates=len(gains), positive_gain_candidates=sum(gain > 0 for gain in gains.values()),
            gain_leaders=sum(cell["gain_leaders"] for cell in cells.values()),
            coverage_rows=sum(cell["coverage"] for cell in cells.values()),
            selected_positive_gain=sum(gains[row["id"]] > 0 for row in selected))
    manifest["selection_sha256"] = content_sha256(manifest)
    return dict(datasets=copy.deepcopy(datasets), manifest=manifest,
                dataset_info={name: dict(file_name=f"{name}.json", columns=dict(COLUMNS)) for name in datasets})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--heldout", action="append", required=True,
                        help="repeat for every source-only held-out set; benchmark data are not an input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=15000)
    parser.add_argument("--cutoff", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("transfer_gain", "coverage"), default="transfer_gain")
    parser.add_argument("--transfer-scores", action="append",
                        help="repeat to merge cross-fitted JSON/JSONL score shards")
    parser.add_argument("--transfer-meta")
    args = parser.parse_args()
    if bool(args.transfer_scores) != bool(args.transfer_meta):
        parser.error("--transfer-scores and --transfer-meta must be supplied together")
    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    scores = None
    if args.transfer_scores:
        scores = []
        for path in args.transfer_scores:
            text = Path(path).read_text(encoding="utf-8").lstrip()
            if text.startswith("{"):
                try:
                    shard = json.loads(text)
                except json.JSONDecodeError:
                    shard = read_rows(path)
                if isinstance(shard, dict):
                    shard = shard["scores"] if "scores" in shard else [shard]
            else:
                shard = read_rows(path)
            scores.extend(shard)
    result = select_datasets(
        read_rows(args.pool), read_rows(args.features),
        feature_meta=read_json(args.features + ".meta.json"), pool_sha256=digest(args.pool),
        heldout_rows=[row for path in args.heldout for row in read_rows(path)],
        size=args.size, cutoff=args.cutoff, seed=args.seed, strategy=args.strategy,
        transfer_scores=scores, transfer_meta=read_json(args.transfer_meta) if args.transfer_meta else None)
    out = Path(args.output)
    if any((out / filename).exists() for filename in ("advanced.json", "random.json", "selection_manifest.json", "dataset_info.json")):
        raise ValueError("refusing to overwrite existing selection artifacts")
    manifest = result["manifest"]
    manifest["source_file_sha256"] = {str(path): digest(path) for path in
                                      [args.pool, args.features, args.features + ".meta.json", *args.heldout,
                                       *([*args.transfer_scores, args.transfer_meta] if args.transfer_scores else [])]}
    manifest.pop("selection_sha256")
    manifest["selection_sha256"] = content_sha256(manifest)
    for name, records in result["datasets"].items():
        write_json(out / f"{name}.json", alpaca(records))
    write_json(out / "dataset_info.json", result["dataset_info"])
    write_json(out / "selection_manifest.json", manifest)
    print(json.dumps(dict(strategy=args.strategy, rows=args.size, selection_sha256=manifest["selection_sha256"])))


if __name__ == "__main__":
    main()
