"""Create a frozen development benchmark only from held-out training-source records."""
import argparse
import random
import json
from pathlib import Path
from .data import boxed, digest, read_rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='artifacts/prepared/validation.json')
    p.add_argument('--output', default='artifacts/dev/acereason_dev/test.jsonl')
    p.add_argument('--size', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude', action='append', default=[],
                   help='JSON/JSONL evaluation files whose id/problem_id values must be excluded')
    a = p.parse_args()
    rows = [r for r in read_rows(a.input) if boxed(r['output']) is not None]
    excluded = set()
    for path in a.exclude:
        for row in read_rows(path):
            value = row.get('id', row.get('problem_id'))
            if not value:
                raise ValueError(f'{path}: exclusion row has no id/problem_id')
            excluded.add(value)
    rows = [r for r in rows if r['id'] not in excluded]
    if not 1 <= a.size <= len(rows):
        p.error('not enough held-out boxed-answer examples for requested size')
    selected = random.Random(a.seed).sample(sorted(rows,key=lambda r:r['id']),a.size)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(dict(question=r['instruction'],answer=boxed(r['output']),id=r['id']),
                                  ensure_ascii=False)+'\n' for r in selected))
    write_json(str(dest)+'.meta.json',dict(source_sha256=digest(a.input),seed=a.seed,size=a.size,
        exclusions={path:digest(path) for path in a.exclude},excluded_ids=len(excluded),
        note='Reference answers extracted from held-out teacher traces; not independently verified.'))


if __name__=='__main__':main()
