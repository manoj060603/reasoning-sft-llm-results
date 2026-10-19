"""CPU-only inventory; text variants are not verified reasoning-path diversity."""
import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def inventory(rows):
    ids = set()
    exact, nfc = defaultdict(set), defaultdict(set)
    for row in rows:
        rid = row['id']
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError('missing or duplicate source ID')
        ids.add(rid)
        fields = tuple(row.get(k, '') for k in ('system', 'instruction', 'input'))
        out = row['output']
        if any(not isinstance(v, str) for v in fields) or not isinstance(out, str) or not out.strip():
            raise ValueError('invalid source text')
        if not fields[1].strip():
            raise ValueError('empty instruction')
        h = hashlib.sha256(out.encode()).hexdigest()
        exact[fields].add(h)
        nfc[tuple(unicodedata.normalize('NFC', v) for v in fields)].add(h)
    def summarize(groups):
        hist = Counter(len(v) for v in groups.values())
        return {'prompt_groups': len(groups), 'distinct_response_count_histogram': dict(sorted(hist.items())),
                'groups_with_at_least_two_text_variants': sum(n for k, n in hist.items() if k >= 2)}
    return {'rows': len(ids), 'exact_prompt': summarize(exact), 'nfc_prompt': summarize(nfc),
            'semantic_diversity_verified': False, 'answer_correctness_verified': False,
            'dataset_exported': False, 'gpu_used': False}


def self_test():
    a = {'id': 'a', 'instruction': 'Compute 2+3', 'output': 'Add to obtain 5.'}
    b = dict(a, id='b', output='Count forward three from two to obtain 5.')
    c = dict(a, id='c')
    assert inventory([a,b,c]) == inventory([c,b,a])
    assert inventory([a,b,c])['exact_prompt']['groups_with_at_least_two_text_variants'] == 1
    assert inventory([a,c])['exact_prompt']['groups_with_at_least_two_text_variants'] == 0
    assert inventory([dict(a, score=999, benchmark='poison')]) == inventory([a])
    u = dict(a, id='u', instruction='caf\u00e9')
    v = dict(b, id='v', instruction='cafe\u0301')
    x = inventory([u,v])
    assert x['exact_prompt']['prompt_groups'] == 2 and x['nfc_prompt']['prompt_groups'] == 1
    for rows in ([a,a], [dict(a, output='')], [dict(a, instruction='')], [dict(a, system=None)]):
        try: inventory(rows)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    print('PASS: order invariance, exact duplicates, variant counts, ignored outcome metadata, NFC distinction, malformed/duplicate rejection')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--pool'); p.add_argument('--expected-sha256'); p.add_argument('--expected-rows', type=int)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.pool or not a.expected_sha256 or a.expected_rows is None:
        p.error('sealed pool path, expected SHA-256 and row count are required')
    path = Path(a.pool)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    if h.hexdigest() != a.expected_sha256: raise ValueError('sealed source hash mismatch')
    with path.open() as f: rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != a.expected_rows: raise ValueError('sealed row count mismatch')
    result = inventory(rows)
    result.update(source_sha256=h.hexdigest(), auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__': main()
