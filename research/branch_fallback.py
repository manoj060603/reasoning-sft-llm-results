"""CPU-only prototype for a frozen, source-only successor selection.

No trainer integration, dataset export, benchmark input, or GPU dependency.
The three lexical markers and paragraph step definition are an adaptation,
not a reproduction of the paper's complete reasoning-behavior classifier.
"""
import hashlib
import math
import re
from collections import defaultdict

CUTOFF = 8192

MARKERS = re.compile(r"\b(?:perhaps|another|alternatively)\b", re.I)


def branch_fraction(response):
    steps = [s.strip() for s in re.split(r"\n\s*\n", response) if s.strip()]
    if not steps:
        raise ValueError("empty response")
    return sum(bool(MARKERS.search(s)) for s in steps) / len(steps)


def propose(rows, features, size=15000):
    from reasoning.select import allocate, eligible
    rowmap = {r['id']: r for r in rows}
    if len(rowmap) != len(rows) or len({f['id'] for f in features}) != len(features):
        raise ValueError("duplicate IDs")
    if set(rowmap) != {f['id'] for f in features}:
        raise ValueError("source/feature coverage mismatch")
    groups = defaultdict(list)
    for f in features:
        if eligible(f, CUTOFF):
            groups[f['cluster']].append(f)
    quotas = allocate({k: len(v) for k, v in groups.items()}, size)
    selected, cells = [], []
    for cluster in sorted(groups):
        ordered = sorted(groups[cluster], key=lambda f: (f['response_tokens'], f['id']))
        strata = defaultdict(list)
        for i, f in enumerate(ordered):
            strata[min(3, 4*i//len(ordered))].append(f)
        sq = allocate({k: len(v) for k, v in strata.items()}, quotas[cluster]) if quotas[cluster] else {}
        for quartile, candidates in sorted(strata.items()):
            scored = [(branch_fraction(rowmap[f['id']]['output']), f['id'], f) for f in candidates]
            scored.sort(key=lambda x: (x[0], x[1]))
            # Freeze a 10% exclusion per cell. Never relax it to fill a quota.
            retained = scored[:len(scored)-math.ceil(len(scored)*.1)]
            need = sq.get(quartile, 0)
            if len(retained) < need:
                raise ValueError(f"insufficient retained capacity: {cluster}/{quartile}")
            ranked = sorted(retained, key=lambda x: hashlib.sha256(
                ('branch-fallback-42:'+x[1]).encode()).hexdigest())
            selected.extend({'id': rid, 'branch_fraction': score,
                'cluster': cluster, 'length_quartile': quartile,
                'original_response_sha256': hashlib.sha256(rowmap[rid]['output'].encode()).hexdigest()}
                for score, rid, _ in ranked[:need])
            cells.append({'cluster': cluster, 'quartile': quartile,
                'eligible': len(scored), 'retained': len(retained), 'quota': need})
    selected.sort(key=lambda x: x['id'])
    if len(selected) != size or len({s['id'] for s in selected}) != size:
        raise ValueError("wrong selection size")
    return selected, cells
