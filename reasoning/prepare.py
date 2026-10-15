"""Normalize, decontaminate, deduplicate, and reserve validation before selection."""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from .data import canonical, digest, normalize, prompt_id, question, read_rows, write_json


def shingles(text, n=32):
    text = normalize(text)
    return {text[i:i+n] for i in range(max(0, len(text)-n+1))}


class ExclusionIndex:
    def __init__(self, questions, threshold=.8):
        self.exact = {prompt_id(q) for q in questions}
        self.threshold = threshold
        self.sets = [shingles(q) for q in questions]
        self.postings = defaultdict(list)
        for i, ss in enumerate(self.sets):
            for s in ss:
                self.postings[s].append(i)

    def matches(self, text):
        if prompt_id(text) in self.exact:
            return True
        ss = shingles(text)
        counts = Counter(i for s in ss for i in self.postings.get(s, ()))
        return any(count / min(len(ss), len(self.sets[i])) >= self.threshold
                   for i, count in counts.items())


def prepare(rows, exclusions, val_size=1000, seed=42):
    stats = Counter()
    groups = {}
    for raw in rows:
        stats['input'] += 1
        try:
            row = canonical(raw)
        except (ValueError, TypeError, KeyError):
            stats['invalid'] += 1
            continue
        if exclusions.matches(row['instruction']):
            stats['benchmark_overlap'] += 1
            continue
        if row['id'] in groups:
            stats['duplicate_prompt'] += 1
            continue
        groups[row['id']] = row
    unique = sorted(groups.values(), key=lambda r: r['id'])
    random.Random(seed).shuffle(unique)
    if val_size < 1 or val_size >= len(unique):
        raise ValueError('validation size must leave nonempty train and validation pools')
    val, candidates = unique[:val_size], unique[val_size:]
    # Also prevent near duplicates of validation questions from entering training.
    val_index = ExclusionIndex([r['instruction'] for r in val])
    train = [r for r in candidates if not val_index.matches(r['instruction'])]
    stats['validation_near_overlap'] = len(candidates) - len(train)
    stats.update(train_pool=len(train), validation=len(val))
    return train, val, dict(stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--eval-dir', default='eval/data')
    p.add_argument('--extra-exclusion', action='append', default=[])
    p.add_argument('--output', default='artifacts/prepared')
    p.add_argument('--val-size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    paths = sorted(Path(a.eval_dir).glob('*/test.jsonl')) + [Path(x) for x in a.extra_exclusion]
    if not paths:
        p.error('no exclusion benchmarks found')
    qs = [question(r) for path in paths for r in read_rows(path)]
    train, val, stats = prepare(read_rows(a.input), ExclusionIndex(qs), a.val_size, a.seed)
    out = Path(a.output)
    write_json(out/'pool.json', train)
    write_json(out/'validation.json', val)
    write_json(out/'manifest.json', dict(stats=stats, seed=a.seed, source_sha256=digest(a.input),
               exclusions={str(x): digest(x) for x in paths}))
    print(stats)


if __name__ == '__main__':
    main()
