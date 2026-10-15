"""Copy validated completed caches into a new run after fixing unevaluated IDs."""
import argparse
import copy
import json
import shutil
from pathlib import Path
from .benchmark import record_ids, duplicate_audit
from .data import digest, read_rows, write_json, question
from .evaluate import valid_result
from .metrics import summarize


def migrate(source, destination, data_dir):
    source, destination, data_dir = map(Path, (source, destination, data_dir))
    if destination.exists():
        raise ValueError('migration requires a new destination; nothing is overwritten')
    old = json.loads((source/'manifest.json').read_text())
    new = copy.deepcopy(old)
    audits, completed, changed = {}, {}, []
    for name, expected_hash in old['data'].items():
        data = data_dir/name/'test.jsonl'
        if digest(data) != expected_hash:
            raise ValueError(f'dataset changed: {name}')
        references = read_rows(data)
        if len(references) != old['problem_counts'][name]:
            raise ValueError(f'benchmark size changed: {name}')
        ids = record_ids(references)
        new['problem_ids'][name] = ids
        audits[name] = duplicate_audit(references)
        files = sorted((source/name).glob('batch_*.json'))
        if ids != old['problem_ids'][name]:
            changed.append(name)
            if files:
                raise ValueError(f'cannot relabel already-generated results: {name}')
            continue
        rows = [row for file in files for row in read_rows(file)]
        if len(rows) == len(references):
            if not all(valid_result(row, pid) and row['question'] == question(ref)
                       and row['gold_answer'] == str(ref['answer'])
                       for row, pid, ref in zip(rows, ids, references)):
                raise ValueError(f'invalid cached predictions: {name}')
            score = summarize(rows)
            score['truncated_samples'] = sum(reason == 'length' for row in rows
                                             for reason in row['finish_reasons'])
            completed[name] = (files, score)
    if not changed:
        raise ValueError('no record identity changes require migration')
    write_json(destination/'manifest.json', new)
    for name, (files, _) in completed.items():
        (destination/name).mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copyfile(file, destination/name/file.name)
    write_json(destination/'metrics.json', {name: score for name, (_, score) in completed.items()})
    receipt = dict(source=str(source), source_manifest_sha256=digest(source/'manifest.json'),
        changed_identity_benchmarks=changed, copied_complete_benchmarks=sorted(completed),
        datasets_unchanged=True, source_preserved=True, duplicate_audit=audits,
        note='Only unevaluated benchmark IDs changed. Completed predictions copied byte-for-byte.')
    write_json(destination/'cache_migration.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--data-dir', required=True)
    args = p.parse_args()
    print(json.dumps(migrate(args.source, args.destination, args.data_dir), indent=2))


if __name__ == '__main__':
    main()
