"""Paired normalized-question cluster bootstrap for complete raw evaluations.

CPU-only standard-library analysis. Confidence intervals condition on the fixed
trained models and their eight recorded outputs per problem; they do not cover
training-seed variance or additional generation sampling. Benchmark macros use
independent within-benchmark cluster resampling and equal benchmark weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from deadline.audit_results import (EXPECTED, HISTORICAL_SEVEN, OFFICIAL, audit_run,
                                    compare_coverage, require)
from reasoning.benchmark import record_ids
from reasoning.data import digest, prompt_id


SEED = 20261019
RESAMPLES = 10000
RAW_EVIDENCE = "raw eight-sample predictions verified"
ROLES = ("original", "random", "advanced")
PAIRS = {"advanced_vs_random": "random", "advanced_vs_original": "original"}


def _percentile(values, quantile):
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _seed(benchmark, seed):
    return int(hashlib.sha256(f"{seed}:question-clusters:{benchmark}".encode()).hexdigest()[:16], 16)


def _diagnostics(records):
    samples = len(records) * 8
    finish_available = all(record["finish_reasons"] is not None for record in records)
    token_available = all(record["output_tokens"] is not None for record in records)
    truncations = sum(record["truncated"] for record in records) if finish_available else None
    total_tokens = sum(record["total_tokens"] for record in records) if token_available else None
    return dict(problems=len(records), samples=samples,
                truncated_samples=truncations,
                truncation_rate=truncations / samples if finish_available else None,
                output_tokens=total_tokens,
                mean_output_tokens=total_tokens / samples if token_available else None,
                mean_output_characters=sum(record["total_characters"] for record in records) / samples,
                finish_reasons_available_for_all_samples=finish_available,
                token_lengths_available_for_all_samples=token_available,
                missing_diagnostics=[name for name, present in (("finish_reasons", finish_available),
                                                                 ("output_tokens", token_available)) if not present])


def _load_records(path, benchmark, expected_ids):
    records, hashes = {}, {}
    for file in sorted((Path(path) / benchmark).glob("batch_*.json")):
        raw = file.read_bytes()
        hashes[str(file.resolve())] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw)
        for row in rows:
            identifier = row["problem_id"]
            question = row.get("question")
            require(isinstance(question, str) and bool(question.strip()),
                    f"{benchmark}/{identifier}: raw question text is required for strict cluster uncertainty")
            require(identifier not in records, f"{benchmark}: duplicate raw problem ID")
            finish, tokens = row.get("finish_reasons"), row.get("output_tokens")
            if finish is not None:
                require(isinstance(finish, list) and len(finish) == 8 and
                        all(isinstance(value, str) and bool(value) for value in finish),
                        f"{benchmark}/{identifier}: invalid recorded finish reasons")
            if tokens is not None:
                require(isinstance(tokens, list) and len(tokens) == 8 and
                        all(type(value) is int and value >= 0 for value in tokens),
                        f"{benchmark}/{identifier}: invalid recorded token lengths")
            flags = row["answers_correctness"]
            records[identifier] = dict(
                problem_id=identifier, question=question, question_cluster=prompt_id(question),
                pass_at_1=sum(flags) / 8, pass_at_8=float(any(flags)),
                finish_reasons=finish, truncated=sum(value == "length" for value in finish) if finish is not None else None,
                output_tokens=tokens, total_tokens=sum(tokens) if tokens is not None else None,
                total_characters=sum(map(len, row["generated_responses"])))
    require(set(records) == set(expected_ids), f"{benchmark}: incomplete raw question identities")
    ordered = [records[identifier] for identifier in expected_ids]
    require(record_ids(ordered) == expected_ids,
            f"{benchmark}: raw question identities differ from their manifest-bound record IDs")
    return ordered, hashes


def paired_cluster_bootstrap(candidate, comparator, *, resamples=RESAMPLES, seed=SEED):
    """Resample normalized-question clusters, retaining every duplicate record.

    Within a draw, sum all records in sampled clusters and divide by the sampled
    record count. This preserves the official record-weighted metric when cluster
    sizes differ; it does not average cluster means with equal weights.
    """
    require(type(resamples) is int and resamples >= 1000, "at least 1,000 bootstrap resamples are required")
    require(len(candidate) == len(comparator) and bool(candidate), "paired raw question coverage must be nonempty and complete")
    groups = {}
    for advanced, baseline in zip(candidate, comparator):
        require(advanced["problem_id"] == baseline["problem_id"] and
                advanced["question"] == baseline["question"],
                "paired problem IDs and raw questions must agree exactly")
        key = prompt_id(advanced["question"])
        require(key == advanced["question_cluster"] == baseline["question_cluster"],
                "paired normalized-question clusters differ")
        group = groups.setdefault(key, [0, 0.0, 0.0])
        group[0] += 1
        group[1] += advanced["pass_at_1"] - baseline["pass_at_1"]
        group[2] += advanced["pass_at_8"] - baseline["pass_at_8"]
    clusters = [groups[key] for key in sorted(groups)]
    sizes = [group[0] for group in clusters]
    draws = {"pass_at_1": [], "pass_at_8": []}
    generator = random.Random(seed)
    for _ in range(resamples):
        count, p1, p8 = 0, 0.0, 0.0
        for _ in clusters:
            group = clusters[generator.randrange(len(clusters))]
            count += group[0]
            p1 += group[1]
            p8 += group[2]
        draws["pass_at_1"].append(p1 / count)
        draws["pass_at_8"].append(p8 / count)
    observed = {}
    for metric in draws:
        point = mean(row[metric] for row in candidate) - mean(row[metric] for row in comparator)
        observed[metric] = _estimate(point, draws[metric],
                                     mean(row[metric] for row in candidate), mean(row[metric] for row in comparator))
    return dict(metrics=observed, draws=draws, problems=len(candidate), question_clusters=len(clusters),
                duplicate_question_records=len(candidate) - len(clusters),
                maximum_cluster_records=max(sizes), cluster_record_counts=sorted(sizes))


def _estimate(point, draws, advanced, comparator):
    interval = [_percentile(draws, .025), _percentile(draws, .975)]
    return dict(advanced=advanced, comparator=comparator, delta=point, delta_pp=100 * point,
                ci95=interval, ci95_pp=[100 * value for value in interval])


def _pooled_diagnostics(benchmark_diagnostics, names):
    by_role = {}
    for role in ROLES:
        values = [benchmark_diagnostics[name][role] for name in names]
        samples = sum(value["samples"] for value in values)
        finish = all(value["finish_reasons_available_for_all_samples"] for value in values)
        tokens = all(value["token_lengths_available_for_all_samples"] for value in values)
        truncated = sum(value["truncated_samples"] for value in values) if finish else None
        token_count = sum(value["output_tokens"] for value in values) if tokens else None
        by_role[role] = dict(samples=samples,
            problems=sum(value["problems"] for value in values), truncated_samples=truncated,
            truncation_rate=truncated / samples if finish else None,
            mean_output_tokens=token_count / samples if tokens else None,
            weighting="pooled recorded samples, distinct from equal-benchmark accuracy macro")
    return by_role


def compare_uncertainty(paths, *, resamples=RESAMPLES, seed=SEED, expected=None):
    require(set(paths) == set(ROLES), "supply exactly original, random, and advanced evaluation directories")
    require(type(resamples) is int and resamples >= 1000, "at least 1,000 bootstrap resamples are required")
    expected = EXPECTED if expected is None else expected
    runs = {role: audit_run(paths[role], expected=expected) for role in ROLES}
    protocol = compare_coverage(runs, names=())
    require(protocol["matched"], "Unmatched evaluator protocol: " + "; ".join(protocol["errors"]))
    result = dict(schema="deadline-paired-cluster-bootstrap-v1", seed=seed, resamples=resamples,
        confidence=.95, interval="percentile paired normalized-question cluster bootstrap",
        resampling_unit="normalized question within each benchmark; duplicate question records travel together",
        cluster_normalization="reasoning.data.prompt_id: NFKC, collapsed whitespace, case folding",
        metric_weighting="official problem-record mean; equal benchmark weights for complete macros",
        conditioning="Fixed trained models and their eight recorded outputs per problem; excludes training-seed and additional generation-sampling variance.",
        multiple_comparison_adjustment=False,
        historical_leaderboard_equivalence_verified=False,
        comparisons={pair: dict(benchmarks={}, macros={}) for pair in PAIRS},
        generation_diagnostics={}, unavailable_benchmarks={},
        sources={role: dict(path=str(Path(path).resolve()), manifest_sha256=digest(Path(path) / "manifest.json"))
                 for role, path in paths.items()})
    replicates = {pair: {} for pair in PAIRS}
    for benchmark in expected:
        entries = {role: runs[role]["benchmarks"].get(benchmark, {}) for role in ROLES}
        unavailable = [f"{role}: {entry.get('status', 'missing')}; {entry.get('evidence', 'no raw evidence')}"
                       for role, entry in entries.items()
                       if entry.get("status") != "complete" or entry.get("evidence") != RAW_EVIDENCE]
        if unavailable:
            result["unavailable_benchmarks"][benchmark] = unavailable
            continue
        matched = compare_coverage(runs, names=(benchmark,))
        require(matched["matched"], f"Unmatched {benchmark} evidence: " + "; ".join(matched["errors"]))
        records = {}
        for role in ROLES:
            ids = runs[role]["manifest"]["problem_ids"][benchmark]
            records[role], hashes = _load_records(paths[role], benchmark, ids)
            result["sources"][role].setdefault("raw_batch_sha256", {}).update(hashes)
        result["generation_diagnostics"][benchmark] = {role: _diagnostics(records[role]) for role in ROLES}
        for pair, role in PAIRS.items():
            estimate = paired_cluster_bootstrap(records["advanced"], records[role],
                                                resamples=resamples, seed=_seed(benchmark, seed))
            replicates[pair][benchmark] = estimate.pop("draws")
            result["comparisons"][pair]["benchmarks"][benchmark] = estimate
    for label, names in (("official_eight_macro", tuple(OFFICIAL)),
                         ("supplemented_historical_seven_macro", HISTORICAL_SEVEN)):
        for pair, role in PAIRS.items():
            available = result["comparisons"][pair]["benchmarks"]
            missing = [name for name in names if name not in available]
            if missing:
                result["comparisons"][pair]["macros"][label] = dict(
                    status="unavailable", metrics=None, missing_raw_matched_benchmarks=missing,
                    reason="No partial macro or aggregate-only substitute is permitted")
                continue
            metrics = {}
            for metric in ("pass_at_1", "pass_at_8"):
                advanced = mean(available[name]["metrics"][metric]["advanced"] for name in names)
                comparator = mean(available[name]["metrics"][metric]["comparator"] for name in names)
                draws = [mean(replicates[pair][name][metric][index] for name in names)
                         for index in range(resamples)]
                metrics[metric] = _estimate(advanced - comparator, draws, advanced, comparator)
            result["comparisons"][pair]["macros"][label] = dict(
                status="complete", metrics=metrics, benchmarks=list(names),
                stratification="independent question-cluster draws within each benchmark; equal benchmark mean",
                pooled_generation_diagnostics=_pooled_diagnostics(result["generation_diagnostics"], names))
    result["audit_errors"] = {role: run["errors"] for role, run in runs.items() if run["errors"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument("--" + role, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON result path; existing files are never replaced")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing uncertainty artifact")
    result = compare_uncertainty({role: getattr(args, role) for role in ROLES}, resamples=args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    completed = {pair: sum(value["status"] == "complete" for value in entry["macros"].values())
                 for pair, entry in result["comparisons"].items()}
    print(json.dumps(dict(output=str(args.output.resolve()),
        raw_matched_benchmarks=len(result["generation_diagnostics"]), complete_macros=completed,
        resamples=result["resamples"], seed=SEED), indent=2))


if __name__ == "__main__":
    main()
