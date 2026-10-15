"""Lightweight lexical reasoning-route signatures; inspired by arXiv:2609.33780.

This is an adaptation, not the paper's full topology extractor. It measures
observable text patterns, not hidden reasoning or proof correctness.
"""
import re
import numpy as np

CUES = (
    r'\b(?:verify|check|confirm|substitut\w* back)\b',
    r'\b(?:wait|mistake|incorrect|instead|reconsider|backtrack)\b',
    r'\b(?:alternatively|another (?:way|approach|method)|different approach)\b',
    r'\b(?:case|cases|suppose|assume|if)\b',
    r'\b(?:therefore|thus|hence|consequently|so we)\b',
    r'\b(?:modulo|mod|remainder|divisib\w*|congru\w*)\b',
    r'\b(?:triangle|circle|angle|coordinate|geometr\w*)\b',
    r'\b(?:probability|expect\w*|combin\w*|count\w*|permutation)\b',
    r'\b(?:equation|factor\w*|polynomial|quadratic|algebra)\b',
    r'\b(?:derivative|integral|differentiat\w*|limit|continu\w*)\b',
    r'\b(?:symmetry|invariant|induction|contradiction|bound)\b',
    r'\b(?:enumerat\w*|brute.force|list all|systematic\w*)\b',
)
PATTERNS = [re.compile(x,re.I) for x in CUES]


def fingerprint(text):
    matches = sorted((m.start(),i) for i,p in enumerate(PATTERNS) for m in p.finditer(text))
    n=len(PATTERNS)
    counts=np.zeros(n); late=np.zeros(n); transitions=np.zeros((n,n))
    for pos,i in matches:
        counts[i]+=1
        late[i]+=pos >= len(text)/2
    for (_,i),(_,j) in zip(matches,matches[1:]): transitions[i,j]+=1
    late=np.divide(late,counts,out=np.zeros(n),where=counts>0)
    return np.concatenate([counts/max(1,counts.sum()),late,
        transitions.flatten()/max(1,transitions.sum()),(counts>0).astype(float)])


def farthest_select(vectors, quality, count, quality_weight=.2):
    """Coverage + bounded quality; O(pool * budget), no pairwise matrix."""
    x=np.asarray(vectors,dtype=np.float32)
    quality=np.asarray(quality,dtype=np.float32)
    if not 0 < count <= len(x) or not np.isfinite(x).all() or not np.isfinite(quality).all():
        raise ValueError('invalid selection inputs')
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    x=x/np.maximum(norms,1e-8)
    chosen=[]; available=np.ones(len(x),dtype=bool); distance=np.ones(len(x))
    next_idx=int(np.argmax(quality))
    for _ in range(count):
        chosen.append(next_idx);available[next_idx]=False
        d=((x-x[next_idx])**2).sum(axis=1)
        distance=np.minimum(distance,d)
        utility=(1-quality_weight)*np.minimum(distance/2,1)+quality_weight*quality
        utility[~available]=-np.inf
        next_idx=int(np.argmax(utility))
    return chosen


def route_select(rows, features, size, seed, cutoff):
    from .select import eligible, allocate
    from collections import defaultdict
    fmap={f['id']:f for f in features}
    if len(fmap)!=len(features) or set(fmap)!={r['id'] for r in rows}:
        raise ValueError('feature mismatch')
    pool=[r for r in rows if eligible(fmap[r['id']],cutoff)]
    groups=defaultdict(list)
    for r in sorted(pool,key=lambda r:r['id']):groups[fmap[r['id']]['cluster']].append(r)
    quotas=allocate({k:len(v) for k,v in groups.items()},size)
    # Corpus standardization before cluster-level coverage comparisons.
    ordered=[r for group in groups.values() for r in group]
    x=np.asarray([fingerprint(r['output']) for r in ordered])
    x=np.clip((x-x.mean(0))/np.maximum(x.std(0),1e-6),-5,5)
    offset=0; chosen=[]
    for key,group in groups.items():
        vectors=x[offset:offset+len(group)];offset+=len(group)
        if quotas[key] == 0:
            continue
        quality=np.array([1-fmap[r['id']]['repetition'] for r in group])
        chosen.extend(group[i] for i in farthest_select(vectors,quality,quotas[key]))
    import random
    random.Random(seed).shuffle(chosen)
    return chosen,pool,quotas


def capacity_mix_select(rows, features, size, seed, cutoff, short_cutoff, short_fraction):
    """Select a fixed short/medium trace mixture while preserving route coverage.

    Small models can fail to learn from uniformly long teacher traces.  The short
    tranche is selected first, then the remaining budget is filled from the
    complete cutoff-eligible pool without duplicates.  Both tranches use the
    same route-coverage selector.
    """
    from .select import eligible
    if not 0 < short_fraction < 1 or not 1 <= short_cutoff < cutoff:
        raise ValueError('invalid capacity-mixture settings')
    fmap = {f['id']: f for f in features}
    if len(fmap) != len(features) or set(fmap) != {r['id'] for r in rows}:
        raise ValueError('feature mismatch')
    short_size = round(size * short_fraction)
    short_rows = [r for r in rows if eligible(fmap[r['id']], short_cutoff)]
    short_features = [fmap[r['id']] for r in short_rows]
    short, _, short_quotas = route_select(
        short_rows, short_features, short_size, seed, short_cutoff)
    selected = {r['id'] for r in short}
    remaining_rows = [r for r in rows if r['id'] not in selected]
    remaining_features = [fmap[r['id']] for r in remaining_rows]
    medium, _, medium_quotas = route_select(
        remaining_rows, remaining_features, size - short_size, seed + 1, cutoff)
    chosen = short + medium
    import random
    random.Random(seed).shuffle(chosen)
    full_pool = [r for r in rows if eligible(fmap[r['id']], cutoff)]
    quotas = {'short': short_quotas, 'medium': medium_quotas,
              'short_size': short_size, 'medium_size': size - short_size}
    return chosen, full_pool, quotas
