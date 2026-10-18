"""Synthetic CPU contract checks; no benchmark questions or model scoring."""
import copy
import json
from research.branch_fallback import branch_fraction, propose

assert branch_fraction('Deduce x.\n\nTherefore y.') == 0
assert branch_fraction('Perhaps x.\n\nTherefore y.') == .5
assert branch_fraction('ANOTHER path.\n\nAlternatively z.') == 1
try:
    branch_fraction('  ')
except ValueError:
    pass
else:
    raise AssertionError('empty response accepted')

rows, features = [], []
for cluster in range(3):
    for i in range(80):
        rid = f'{cluster}:{i:03}'
        rows.append({'id':rid, 'output': ('Perhaps another path.\n\n' if i%5==0 else '')+'Deduce.\n\nConclude.'})
        features.append({'id':rid, 'cluster':cluster, 'response_tokens':128+i,
            'total_tokens':256+i, 'complete_box':True, 'repetition':0.})
before = copy.deepcopy((rows, features))
a, cells = propose(rows, features, size=60)
b, _ = propose(list(reversed(rows)), list(reversed(features)), size=60)
assert a == b and len(a) == 60
assert (rows, features) == before
assert all(c['quota'] <= c['retained'] for c in cells)
# Public-score-like metadata is ignored, even when poisoned with NaN.
for f in features:
    f.update(nll=float('nan'), benchmark_score=9999, failed_rollout=True)
assert propose(rows, features, size=60)[0] == a
try:
    propose(rows + rows[:1], features, size=60)
except ValueError:
    pass
else:
    raise AssertionError('duplicate IDs accepted')
try:
    propose(rows, features, size=len(rows))
except ValueError:
    pass
else:
    raise AssertionError('capacity failure was relaxed')
print(json.dumps({'synthetic_contract_checks':'passed','fixtures':240,
    'selected':60,'order_invariant':True,'source_unchanged':True,
    'public_score_metadata_ignored':True,'capacity_failure_closed':True,
    'actual_pool_prepared':False,'gpu_used':False}))
