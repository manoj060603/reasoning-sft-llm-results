"""Add benchmark exclusions without re-splitting or moving held-out rows."""
import argparse
from pathlib import Path
from .data import digest,question,read_rows,write_json
from .prepare import ExclusionIndex


def extend(pool,validation,questions):
    if {r['id'] for r in pool} & {r['id'] for r in validation}:
        raise ValueError('input training and validation pools overlap')
    index=ExclusionIndex(questions)
    return ([r for r in pool if not index.matches(r['instruction'])],
            [r for r in validation if not index.matches(r['instruction'])])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',default='artifacts/prepared/pool.json')
    p.add_argument('--validation',default='artifacts/prepared/validation.json')
    p.add_argument('--extra-exclusion',required=True,nargs='+')
    p.add_argument('--output',default='artifacts/prepared_extended')
    a=p.parse_args()
    pool,validation=read_rows(a.pool),read_rows(a.validation)
    questions=[question(r) for path in a.extra_exclusion for r in read_rows(path)]
    if not questions:raise ValueError('empty extra exclusion benchmark')
    train,heldout=extend(pool,validation,questions)
    root=Path(a.output)
    write_json(root/'pool.json',train);write_json(root/'validation.json',heldout)
    receipt=dict(original_pool_sha256=digest(a.pool),original_validation_sha256=digest(a.validation),
        extra_exclusions={path:digest(path) for path in a.extra_exclusion},
        train_before=len(pool),train_after=len(train),validation_before=len(validation),
        validation_after=len(heldout),heldout_rows_never_moved_to_train=True)
    write_json(root/'manifest.json',receipt);print(receipt)


if __name__=='__main__':main()
