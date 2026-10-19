"""CPU-only prompt overlap audit; benchmark answers and outcomes are never read.

The new 13-word rule is a conservative contamination exclusion, not a measure
of mathematical similarity. Chinese/no-space text remains covered only by exact
normalization and the pre-existing source preparation's character audit.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def grams(text):
    words = norm(text).split()
    return {' '.join(words[i:i+13]) for i in range(len(words)-12)}


def overlap(prompts, benchmarks):
    exact, words = set(), set()
    for text in benchmarks:
        exact.add(norm(text))
        words.update(grams(text))
    return [dict(id=key, exact=norm(text) in exact,
                 shared_13_word_gram=bool(grams(text) & words))
            for key, text in prompts if norm(text) in exact or grams(text) & words]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        sentence = ' '.join(str(i) for i in range(13))
        r = overlap([('a', sentence.upper()), ('b', 'prefix '+sentence), ('c', 'unrelated')], [sentence])
        assert [x['id'] for x in r] == ['a', 'b'] and r[0]['exact'] and not r[1]['exact']
        assert overlap([('x', '甲乙 丙丁')], ['甲乙   丙丁'])[0]['exact']
        assert not overlap([('x', '2 + 3')], ['2 - 3'])
        print('PASS exact normalization, 13-word overlap, short multilingual exact match, mathematical operators')
        return
    if a.root is None:
        p.error('--root is required')
    root = a.root
    selected_path = root/'artifacts/datasets-full-length-15k/selection_manifest.json'
    selected = json.loads(selected_path.read_text())
    pool_path = root/'artifacts/prepared_extended/pool.json'
    assert sha(pool_path) == selected['pool_sha256']
    pool = json.loads(pool_path.read_text())
    by_id = {r['id']: r for r in pool}
    ids = selected['selected_ids']
    assert len(ids) == len(set(ids)) == 15000
    baseline = json.loads((root/'results/full-base-suite-v2/manifest.json').read_text())
    expected = {'amc':40, 'aime25':30, 'aime':30, 'math':500,
                'cn_math_2024':30, 'kaoyan':199, 'minerva':272,
                'olympiadbench':675, 'gpqa':198}
    benchmarks, binding = [], {}
    for name, count in expected.items():
        path = root/'artifacts/benchmarks_extended'/name/'test.jsonl'
        digest = sha(path)
        assert digest == baseline['data'][name]
        raw = path.read_text()
        records = (json.loads(raw) if raw.lstrip().startswith('[') else
                   [json.loads(line) for line in raw.split('\n') if line.strip()])
        assert len(records) == count
        for r in records:
            text = next((r[k] for k in ('question','problem','Question','instruction','input')
                         if isinstance(r.get(k), str) and r[k].strip()), None)
            assert text is not None
            benchmarks.append(text)
        binding[name] = dict(sha256=digest, records=count)
    hits = overlap([(key, by_id[key]['instruction']) for key in ids], benchmarks)
    print(json.dumps(dict(schema='two-plan-prompt-contamination-v1',
                         selected_manifest_sha256=sha(selected_path),
                         source_data_sha256=sha(root/'artifacts/datasets-full-length-15k/advanced.json'),
                         pool_sha256=sha(pool_path), benchmark_binding=binding,
                         prompts=15000, benchmark_records=len(benchmarks), hits=hits,
                         excluded_source_ids=[hit['id'] for hit in hits],
                         rule='NFKC casefold whitespace exact or any shared 13 whitespace words',
                         benchmark_answers_or_outcomes_used=False, gpu_used=False,
                         chinese_word_segmentation=False), indent=2))


if __name__ == '__main__':
    main()
