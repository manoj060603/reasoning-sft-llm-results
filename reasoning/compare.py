"""Compare matched runs; never label the available-suite mean as historical AVG."""
import argparse
import json
from pathlib import Path
from statistics import mean
from .data import read_rows, write_json, prompt_id
from .metrics import paired_cluster_bootstrap, summarize

LEADERBOARD_BENCHMARKS = ('aime25', 'cn_math_2024', 'kaoyan', 'amc', 'minerva', 'olympiadbench', 'gpqa')
MODEL_IDENTITY_FIELDS = {'model', 'identity', 'lora_adapter', 'lora_identity'}


def check_protocol(baseline, candidate):
    # A comparison requires different model artifacts.  LoRA paths and hashes
    # identify the candidate just as the base model path and hashes do; they are
    # not sampling or grading protocol settings.
    for key in (set(baseline) | set(candidate)) - MODEL_IDENTITY_FIELDS:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError(f'incomparable protocol: {key}')


def load_run(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    results = {}
    for benchmark in manifest['data']:
        rows = [r for p in sorted((path/benchmark).glob('batch_*.json')) for r in read_rows(p)]
        if len({r['problem_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate result IDs')
        if len(rows) != manifest['problem_counts'][benchmark] or {r['problem_id'] for r in rows} != set(manifest['problem_ids'][benchmark]):
            raise ValueError(f'incomplete benchmark: {benchmark}')
        summarize(rows)
        results[benchmark] = rows
    return manifest, results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', default='artifacts/comparison.json')
    a = p.parse_args()
    bm, base = load_run(a.baseline)
    cm, cand = load_run(a.candidate)
    # Model identity is the only intended difference.
    check_protocol(bm, cm)
    results = {}
    for b in base:
        result = dict(baseline=summarize(base[b]), candidate=summarize(cand[b]))
        groups = {r['problem_id']: prompt_id(r['question']) if 'question' in r else r['problem_id']
                  for r in base[b]}
        result['unique_questions'] = len(set(groups.values()))
        for metric in ('pass_at_1','pass_at_8'):
            def scores(rows):
                return {r['problem_id']: (sum(r['answers_correctness'])/8 if metric=='pass_at_1'
                        else float(any(r['answers_correctness']))) for r in rows}
            result[metric] = paired_cluster_bootstrap(scores(base[b]), scores(cand[b]), groups)
        results[b] = result
    results['available_suite_macro'] = {metric: dict(
        baseline=mean(results[b]['baseline'][metric] for b in base),
        candidate=mean(results[b]['candidate'][metric] for b in base)) for metric in ['pass_at_1','pass_at_8']}
    results['historical_leaderboard_avg'] = None
    missing = sorted(set(LEADERBOARD_BENCHMARKS) - set(base))
    results['missing_leaderboard_benchmarks'] = missing
    results['matched_seven_benchmark_macro'] = None if missing else {
        metric: {run: mean(results[b][run][metric] for b in LEADERBOARD_BENCHMARKS)
                 for run in ('baseline', 'candidate')} for metric in ('pass_at_1', 'pass_at_8')}
    results['note'] = ('Only matched internal runs are compared. Historical data versions and runtime are not verified; '
                       'a separately sourced AIME25 does not establish exact historical comparability. '
                       'Confidence intervals resample normalized question clusters (keeping duplicate records together), '
                       'are per-benchmark, and are not multiple-comparison adjusted.')
    write_json(a.output, results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
