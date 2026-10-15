import math
import random
from statistics import mean


def pass_at_k(correct, n, k):
    if not (0 <= correct <= n and 1 <= k <= n):
        raise ValueError('require 0 <= correct <= n and 1 <= k <= n')
    return 1. if n-correct < k else 1-math.comb(n-correct,k)/math.comb(n,k)


def summarize(rows, n=8):
    if not rows:
        raise ValueError('empty results')
    values = []
    for row in rows:
        flags = row['answers_correctness']
        if len(flags) != n or any(type(x) is not bool for x in flags):
            raise ValueError('incomplete or invalid correctness flags')
        values.append((sum(flags)/n, pass_at_k(sum(flags),n,n)))
    return dict(problems=len(rows), samples_per_problem=n,
                pass_at_1=mean(v[0] for v in values), pass_at_8=mean(v[1] for v in values))


def paired_bootstrap(base, candidate, repeats=2000, seed=42):
    # Resample problems, not correlated generations of the same problem.
    if not base or set(base) != set(candidate):
        raise ValueError('paired comparisons require identical problem IDs')
    diffs = [candidate[k]-base[k] for k in sorted(base)]
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(repeats))
    return dict(delta=mean(diffs), ci95=[samples[int(.025*repeats)], samples[int(.975*repeats)]])


def paired_cluster_bootstrap(base, candidate, groups, repeats=2000, seed=42):
    """Keep repeated benchmark questions together while preserving record weights."""
    if not base or set(base) != set(candidate) or set(base) != set(groups):
        raise ValueError('paired comparisons and groups require identical record IDs')
    if len(set(groups.values())) == len(groups):
        return paired_bootstrap(base, candidate, repeats, seed)
    from collections import defaultdict
    clusters = defaultdict(list)
    for key in sorted(base):
        clusters[groups[key]].append(candidate[key] - base[key])
    totals = [(sum(clusters[key]), len(clusters[key])) for key in sorted(clusters)]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        draw = rng.choices(totals, k=len(totals))
        samples.append(sum(x[0] for x in draw) / sum(x[1] for x in draw))
    samples.sort()
    return dict(delta=mean(candidate[k]-base[k] for k in base),
                ci95=[samples[int(.025*repeats)], samples[int(.975*repeats)]])
