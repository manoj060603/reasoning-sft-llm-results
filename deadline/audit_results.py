#!/usr/bin/env python3
"""Audit team evaluation folders and write an honest three-model results table.

No GPU, network, publishing, or third-party packages are used. A complete metrics
receipt is accepted when raw predictions were intentionally kept off GitHub, but
is explicitly distinguished from verified per-question generations.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


OFFICIAL = {
    'aime': 30, 'amc': 40, 'math': 500, 'cn_math_2024': 30,
    'kaoyan': 199, 'minerva': 272, 'olympiadbench': 675, 'gpqa': 198,
}
HISTORICAL_SEVEN = ('aime25', 'cn_math_2024', 'kaoyan', 'amc',
                    'minerva', 'olympiadbench', 'gpqa')
EXPECTED = {**OFFICIAL, 'aime25': 30}
REQUIRED_SETTINGS = dict(n=8, temperature=.6, top_p=.95, max_tokens=32768,
                         system='Please reason step by step, and put your final answer within \\boxed{}.')
PROTOCOL = ('seed', 'n', 'temperature', 'top_p', 'max_tokens',
            'max_model_len', 'system', 'generation_config', 'top_k', 'min_p',
            'repetition_penalty', 'batch_size', 'tensor_parallel_size',
            'max_num_seqs', 'gpu_memory_utilization', 'versions', 'grader_sha256',
            'evaluator_sha256')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_metric(metric, count):
    require(type(metric.get('problems')) is int and metric['problems'] == count,
            'metric problem count differs from the frozen benchmark')
    require(metric.get('samples_per_problem') == 8, 'metric requires eight samples per problem')
    for name, denominator in (('pass_at_1', count * 8), ('pass_at_8', count)):
        value = metric.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                f'{name} must be a finite fraction in [0, 1]')
        require(abs(value * denominator - round(value * denominator)) < 1e-5,
                f'{name} is inconsistent with the stated sample/problem count')
    require(metric['pass_at_1'] <= metric['pass_at_8'] + 1e-12, 'pass@1 exceeds pass@8')
    if 'truncated_samples' in metric:
        t = metric['truncated_samples']
        require(type(t) is int and 0 <= t <= count * 8, 'invalid truncation count')
    return {k: metric[k] for k in ('problems', 'samples_per_problem', 'pass_at_1', 'pass_at_8')}


def raw_metric(paths, expected_ids):
    rows = []
    for path in paths:
        batch = read_json(path)
        require(isinstance(batch, list), 'prediction batch must be a JSON list')
        require(all(isinstance(row, dict) for row in batch), 'prediction rows must be objects')
        rows.extend(batch)
    require(len(rows) == len(expected_ids), 'raw prediction coverage is incomplete')
    ids = [r.get('problem_id') for r in rows]
    require(len(set(ids)) == len(ids) and set(ids) == set(expected_ids),
            'raw prediction identities are duplicated or differ from the manifest')
    correct = 0
    solved = 0
    truncated = 0
    has_finish = True
    for row in rows:
        flags = row.get('answers_correctness', [])
        texts = row.get('generated_responses', [])
        require(isinstance(flags, list) and len(flags) == 8 and all(type(x) is bool for x in flags),
                'every question must have eight boolean correctness values')
        require(isinstance(texts, list) and len(texts) == 8 and all(isinstance(x, str) for x in texts),
                'every question must have eight generated responses')
        for field in ('generated_answers', 'finish_reasons', 'output_tokens'):
            require(field not in row or (isinstance(row[field], list) and len(row[field]) == 8),
                    f'every question must have eight {field} entries when recorded')
        correct += sum(flags)
        solved += any(flags)
        has_finish = has_finish and 'finish_reasons' in row
        truncated += sum(x == 'length' for x in row.get('finish_reasons', []))
    result = dict(problems=len(rows), samples_per_problem=8,
                  pass_at_1=correct / (8 * len(rows)), pass_at_8=solved / len(rows))
    if has_finish:
        result['truncated_samples'] = truncated
    return result


def audit_run(path, expected=EXPECTED):
    path = Path(path)
    result = dict(path=str(path.resolve()), benchmarks={}, errors=[])
    try:
        manifest = read_json(path / 'manifest.json')
        require(isinstance(manifest, dict), 'manifest must be an object')
        for field, value in REQUIRED_SETTINGS.items():
            require(manifest.get(field) == value, f'manifest requires {field}={value!r}')
        for field in ('problem_counts', 'problem_ids', 'data'):
            require(isinstance(manifest.get(field), dict), f'missing manifest {field}')
        result['manifest'] = manifest
        stored = read_json(path / 'metrics.json') if (path / 'metrics.json').exists() else {}
        require(isinstance(stored, dict), 'metrics file must be a benchmark mapping')
    except (OSError, ValueError, TypeError) as exc:
        result['errors'].append(str(exc))
        return result
    for benchmark, count in expected.items():
        entry = dict(status='missing', evidence='none')
        result['benchmarks'][benchmark] = entry
        raw = sorted((path / benchmark).glob('batch_*.json'))
        if benchmark not in stored and not raw:
            continue
        try:
            require(manifest['problem_counts'].get(benchmark) == count,
                    'manifest count differs from the frozen benchmark')
            ids = manifest['problem_ids'].get(benchmark, [])
            require(len(ids) == count and len(set(ids)) == count,
                    'manifest must contain each benchmark record identity once')
            require(isinstance(manifest['data'].get(benchmark), str) and
                    bool(manifest['data'][benchmark]), 'missing benchmark data digest')
            if raw:
                metric = raw_metric(raw, ids)
                if benchmark in stored:
                    validate_metric(stored[benchmark], count)
                    for key in ('pass_at_1', 'pass_at_8'):
                        require(abs(stored[benchmark][key] - metric[key]) < 1e-10,
                                f'raw predictions disagree with stored {key}')
                    if 'truncated_samples' in stored[benchmark] and 'truncated_samples' in metric:
                        require(stored[benchmark]['truncated_samples'] == metric['truncated_samples'],
                                'raw predictions disagree with stored truncations')
                entry.update(status='complete', evidence='raw eight-sample predictions verified', metrics=metric)
            else:
                validate_metric(stored[benchmark], count)
                entry.update(status='complete', evidence='aggregate receipt only; raw predictions unavailable',
                             metrics=stored[benchmark])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            entry.update(status='invalid', error=str(exc))
            result['errors'].append(f'{benchmark}: {exc}')
    for label, names in (('official_eight_macro', tuple(OFFICIAL)),
                         ('supplemented_historical_seven_macro', HISTORICAL_SEVEN)):
        result[label] = None
        if all(result['benchmarks'].get(b, {}).get('status') == 'complete' for b in names):
            result[label] = {key: mean(result['benchmarks'][b]['metrics'][key] for b in names)
                             for key in ('pass_at_1', 'pass_at_8')}
    return result


def compare_coverage(runs, names=tuple(OFFICIAL)):
    """Require all three roles and identical evaluator/data for a matched claim."""
    errors = []
    for role in ('original', 'random', 'advanced'):
        if role not in runs:
            errors.append(f'{role}: no run supplied')
    active = [(role, r) for role, r in runs.items() if 'manifest' in r]
    if len(active) != len(runs):
        errors.append('one or more supplied runs has no valid manifest')
    if not active:
        return dict(matched=False, errors=errors or ['no valid runs'])
    base_role, base = active[0]
    for field in PROTOCOL:
        if field not in base['manifest']:
            errors.append(f'{base_role}: missing protocol field {field}')
    for role, run in active:
        for b in names:
            if run['benchmarks'].get(b, {}).get('status') != 'complete':
                errors.append(f'{role}: {b} incomplete or invalid')
        if role == base_role:
            continue
        bm, cm = base['manifest'], run['manifest']
        for field in PROTOCOL:
            if field not in cm or cm[field] != bm.get(field):
                errors.append(f'{role}: protocol differs at {field}')
        if ('hardware' in bm or 'hardware' in cm) and bm.get('hardware') != cm.get('hardware'):
            errors.append(f'{role}: recorded hardware differs or is missing')
        for field in ('problem_counts', 'problem_ids', 'data'):
            for b in names:
                if bm.get(field, {}).get(b) != cm.get(field, {}).get(b):
                    errors.append(f'{role}: {b} differs at {field}')
    return dict(matched=not errors, errors=errors)


def percent(metric):
    if metric is None:
        return '—'
    return f"{100 * metric['pass_at_1']:.2f}% / {100 * metric['pass_at_8']:.2f}%"


def markdown_report(runs, matching):
    lines = ['# Assignment result audit', '',
             'Values are pass@1 / pass@8; each macro gives equal weight to its stated benchmark sets.', '',
             '| Benchmark | Original | Random 15K | Advanced 15K |', '| --- | ---: | ---: | ---: |']
    for b in EXPECTED:
        cells = []
        for role in ('original', 'random', 'advanced'):
            entry = runs.get(role, {}).get('benchmarks', {}).get(b, {})
            cells.append(percent(entry.get('metrics') if entry.get('status') == 'complete' else None))
        lines.append(f"| {b} | {' | '.join(cells)} |")
    for label, field in (('Official eight macro', 'official_eight_macro'),
                         ('Supplemented historical seven macro', 'supplemented_historical_seven_macro')):
        lines.append(f"| {label} | " + ' | '.join(percent(runs.get(r, {}).get(field))
                     for r in ('original', 'random', 'advanced')) + ' |')
    lines.extend(['', 'The supplemented seven-task figure is an internal macro, not a verified historical leaderboard result.', '',
                  f"Matched original/random/advanced official-eight comparison: **{'verified' if matching['matched'] else 'not established'}**.", ''])
    for role in ('original', 'random', 'advanced'):
        if role not in runs:
            lines.append(f'- {role}: no evaluation folder supplied.')
            continue
        run = runs[role]
        raw = sum(e['evidence'] == 'raw eight-sample predictions verified' for e in run['benchmarks'].values())
        receipt = sum(e['status'] == 'complete' and e['evidence'].startswith('aggregate') for e in run['benchmarks'].values())
        lines.append(f'- {role}: {raw} benchmarks verified from raw predictions; {receipt} supported only by aggregate receipts.')
        lines.extend(f'  - {e}' for e in run['errors'])
    if matching['errors']:
        lines.extend(['', 'Comparison limitations:'])
        lines.extend(f'- {e}' for e in matching['errors'])
    lines.extend(['', 'This audit covers evaluation evidence only. It does not verify dataset size, training completion, model files, Hugging Face publication, or course submission.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('original', 'random', 'advanced'):
        parser.add_argument('--' + role, type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = {role: audit_run(getattr(args, role)) for role in ('original', 'random', 'advanced')
            if getattr(args, role) is not None}
    if not runs:
        parser.error('supply at least one evaluation folder')
    matching = compare_coverage(runs)
    public_runs = {role: {k: v for k, v in run.items() if k != 'manifest'} for role, run in runs.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'audit.json').write_text(json.dumps(dict(runs=public_runs, comparison=matching), indent=2) + '\n')
    (args.output_dir / 'results.md').write_text(markdown_report(runs, matching))
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), matched_comparison=matching['matched'],
                          errors=len(matching['errors'])), indent=2))


if __name__ == '__main__':
    main()
