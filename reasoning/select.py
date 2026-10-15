"""Export random, filtered-random, and difficulty/diversity SFT datasets."""
import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from .data import alpaca, digest, read_rows, write_json


def allocate(capacities, size):
    if size < 1 or sum(capacities.values()) < size:
        raise ValueError(f'cannot select {size} examples from {sum(capacities.values())}')
    quotas = {k: 0 for k in capacities}
    # Square-root allocation balances broad coverage with corpus prevalence.
    for _ in range(size):
        k = max((k for k in capacities if quotas[k] < capacities[k]),
                key=lambda k: (math.sqrt(capacities[k])/(quotas[k]+1), str(k)))
        quotas[k] += 1
    return quotas


def eligible(f, cutoff):
    return (0 < f['total_tokens'] <= cutoff - 32 and f['response_tokens'] >= 128
            and f['complete_box'] and f['repetition'] <= .35)


def advanced_select(rows, features, size, seed, cutoff, use_nll):
    ids = {r['id'] for r in rows}
    fmap = {f['id']: f for f in features}
    if len(fmap) != len(features) or set(fmap) != ids:
        raise ValueError('feature IDs must cover the exact pool once')
    pool = [r for r in rows if eligible(fmap[r['id']], cutoff)]
    if use_nll and any(fmap[r['id']].get('nll') is None for r in pool):
        raise ValueError('hybrid selection requires NLL for every eligible row; use --strategy diversity explicitly')
    ranks = {}
    buckets = defaultdict(list)
    if use_nll:
        for r in pool:
            f = fmap[r['id']]
            if not math.isfinite(f['nll']):
                raise ValueError('nonfinite NLL')
            buckets[int(math.log2(max(1, f['response_tokens'])))].append(f)
        for fs in buckets.values():
            fs.sort(key=lambda f: (f['nll'], f['id']))
            for i, f in enumerate(fs):
                ranks[f['id']] = (i+.5)/len(fs)
    rng = random.Random(seed)
    clusters = defaultdict(list)
    for row in sorted(pool, key=lambda r: r['id']):
        f = fmap[row['id']]
        # Moderate-high loss is a hypothesis, not a proven universal optimum.
        difficulty = 1 - abs(ranks.get(row['id'], .65)-.65)/.65
        score = (difficulty if use_nll else 0) - .3*f['repetition'] + rng.uniform(0, .15)
        clusters[f['cluster']].append((score, row))
    quotas = allocate({k: len(v) for k,v in clusters.items()}, size)
    chosen = [r for k, rs in sorted(clusters.items())
              for _, r in sorted(rs, key=lambda x: (-x[0], x[1]['id']))[:quotas[k]]]
    rng.shuffle(chosen)
    return chosen, pool, quotas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', default='artifacts/prepared/pool.json')
    p.add_argument('--validation', default='artifacts/prepared/validation.json')
    p.add_argument('--features', default='artifacts/features.json')
    p.add_argument('--validation-features', default='artifacts/validation_features.json')
    p.add_argument('--output', default='artifacts/datasets')
    p.add_argument('--size', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cutoff', type=int, default=16384)
    p.add_argument('--strategy', choices=['hybrid','diversity','route','capacity_mix'], default='hybrid')
    p.add_argument('--short-cutoff', type=int, default=3072,
                   help='complete-sequence cutoff for the short capacity_mix tranche')
    p.add_argument('--short-fraction', type=float, default=2/3,
                   help='fraction selected from the short tranche for capacity_mix')
    a = p.parse_args()
    rows, features = read_rows(a.pool), read_rows(a.features)
    # Fail on stale features, including changed responses under the same prompt IDs.
    for source, feature in [(a.pool, a.features), (a.validation, a.validation_features)]:
        import json
        meta = json.loads(Path(feature+'.meta.json').read_text())
        if meta['input_sha256'] != digest(source):
            raise ValueError(f'stale features: {feature}')
    if a.strategy == 'route':
        from .routes import route_select
        chosen, filtered, quotas = route_select(rows, features, a.size, a.seed, a.cutoff)
    elif a.strategy == 'capacity_mix':
        from .routes import capacity_mix_select
        chosen, filtered, quotas = capacity_mix_select(
            rows, features, a.size, a.seed, a.cutoff, a.short_cutoff, a.short_fraction)
    else:
        chosen, filtered, quotas = advanced_select(rows, features, a.size, a.seed, a.cutoff, a.strategy=='hybrid')
    val_rows = read_rows(a.validation)
    vf = {f['id']: f for f in read_rows(a.validation_features)}
    if set(vf) != {r['id'] for r in val_rows}:
        raise ValueError('validation feature mismatch')
    val = [r for r in val_rows if eligible(vf[r['id']], a.cutoff)]
    if not val:
        raise ValueError('no eligible validation rows')
    if {r['id'] for r in val_rows} & {r['id'] for r in rows}:
        raise ValueError('validation overlaps training pool')
    datasets = dict(random=random.Random(a.seed).sample(rows, a.size),
                    filtered_random=random.Random(a.seed).sample(filtered, a.size),
                    advanced=chosen, validation=val)
    out = Path(a.output)
    for name, records in datasets.items():
        write_json(out/f'{name}.json', alpaca(records))
    write_json(out/'dataset_info.json', {name: {'file_name': f'{name}.json'} for name in datasets})
    counts = {f['id']: f['response_tokens'] for f in features}
    counts.update({k: f['response_tokens'] for k,f in vf.items()})
    write_json(out/'selection_manifest.json', dict(arguments=vars(a), quotas=quotas,
        pool_sha256=digest(a.pool), features_sha256=digest(a.features),
        selected_ids={k: [r['id'] for r in v] for k,v in datasets.items()},
        response_tokens={k: sum(counts[r['id']] for r in v) for k,v in datasets.items()}))
    print({k: len(v) for k,v in datasets.items()})


if __name__ == '__main__':
    main()
