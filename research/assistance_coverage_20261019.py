"""Candidate B: saved assistance-associated utility plus complete-original coverage.

CPU-only; no inference, generation, scheduling or training. The old failed
categorical selector remains untouched. CLI requires sealed source, saved plan,
ledger and model receipts; every selected row passes the existing native 16K
processor before export. Missing/unsupported measurements remain unscored.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reasoning.data import alpaca, digest, read_rows, write_json, prompt_id
from reasoning.select import allocate, eligible
from reasoning.full_length import POOL_SHA, FEATURES_SHA, CUTOFF, processor, check_row, write_dataset_info

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SEED_TAG = 'assist-cover-42'
SCHEMA = 'assistance-coverage-original-16k-v1'


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def utility(records):
    """Exactly two supported normally-finished requests per condition; else None."""
    if len(records) != 4:
        return None
    indexed = {(r.get('condition'), r.get('sample')): r for r in records}
    if set(indexed) != {(c, s) for c in ('normal', 'assisted') for s in range(2)}:
        return None
    if any(r.get('finish_reason') != 'stop' or r.get('supported') is not True
           or type(r.get('correct')) is not bool for r in records):
        return None
    normal = (sum(indexed['normal', s]['correct'] for s in range(2)) + 1) / 4
    assisted = (sum(indexed['assisted', s]['correct'] for s in range(2)) + 1) / 4
    return (1 - normal) * max(assisted - normal, 0)


def joined_scores(rows, candidates, records, prefix_checker):
    """Reject mismatches as absent scoring evidence; never label them incorrect.

    Global source/model/tokenizer/generation/ledger bindings are checked by CLI
    before this function. Recompute original prefix with pinned tokenizer.
    """
    from reasoning.demonstration_response import stable_seed
    cmap = {c['id']: c for c in candidates}
    if len(cmap) != len(candidates):
        raise ValueError('duplicate candidate identity')
    indexed = defaultdict(list)
    seen = set()
    rejected = Counter()
    for record in records:
        key = record.get('key')
        if key in seen:
            raise ValueError('duplicate ledger request')
        seen.add(key)
        row_id = record.get('id')
        if row_id not in cmap:
            rejected['foreign_request'] += 1
            continue
        indexed[row_id].append(record)
    scores = {}
    for row_id, c in cmap.items():
        if row_id not in rows:
            rejected['missing_original'] += 1
            continue
        row = rows[row_id]
        prefix, reason = prefix_checker(row['output'])
        if (not prefix or reason or c.get('prefix_reason') is not None
                or c.get('prefix') != prefix or c.get('prefix_sha256') != text_hash(prefix)
                or c.get('response_sha256') != text_hash(row['output'])
                or c.get('prompt_sha256') != text_hash(row['instruction'])):
            rejected['original_or_prefix_join'] += 1
            continue
        group = indexed[row_id]
        valid = True
        for r in group:
            condition, sample = r.get('condition'), r.get('sample')
            if condition not in ('normal', 'assisted') or type(sample) is not int or sample not in (0, 1):
                valid = False
                break
            expected_prefix = prefix if condition == 'assisted' else ''
            if (r.get('key') != f'{row_id}:{condition}:{sample}'
                    or r.get('seed') != stable_seed(42, row_id, condition, sample)
                    or r.get('prefix_sha256') != text_hash(expected_prefix)
                    or not isinstance(r.get('text'), str)
                    or r.get('text_sha256') != text_hash(r['text'])):
                valid = False
                break
        score = utility(group) if valid else None
        if score is None:
            rejected['incomplete_unsupported_or_request_join'] += 1
        else:
            scores[row_id] = score
    return scores, dict(rejected)


def select(features, scores, size=15000):
    """Existing capped sqrt cluster/quartile allocations, at most half utility."""
    if len({f['id'] for f in features}) != len(features):
        raise ValueError('duplicate feature identity')
    groups = defaultdict(list)
    for f in features:
        if eligible(f, CUTOFF):
            groups[f['cluster']].append(f)
    # Feasibility before ranking; this inherits Candidate A's frozen allocation.
    quotas = allocate({k: len(v) for k, v in groups.items()}, size)
    result, receipts = [], []
    for cluster in sorted(groups):
        ordered = sorted(groups[cluster], key=lambda f: (f['response_tokens'], f['id']))
        cells = defaultdict(list)
        for i, f in enumerate(ordered):
            cells[min(3, 4 * i // len(ordered))].append(f)
        cq = allocate({k: len(v) for k, v in cells.items()}, quotas[cluster]) if quotas[cluster] else {k: 0 for k in cells}
        for quartile in sorted(cells):
            cell, q = cells[quartile], cq[quartile]
            ranked = sorted((f for f in cell if scores.get(f['id'], 0) > 0),
                            key=lambda f: (-scores[f['id']], f['id']))
            chosen = ranked[:q // 2]
            ids = {f['id'] for f in chosen}
            remaining = sorted((f for f in cell if f['id'] not in ids),
                               key=lambda f: (text_hash(SEED_TAG + ':' + f['id']), f['id']))
            selected = [(f, 'utility') for f in chosen] + [(f, 'coverage') for f in remaining[:q - len(chosen)]]
            result.extend({**f, 'length_quartile': quartile, 'selection_basis': basis,
                           'assistance_utility': scores.get(f['id'])} for f, basis in selected)
            receipts.append({'cluster': cluster, 'length_quartile': quartile, 'capacity': len(cell),
                             'quota': q, 'positive_scored_available': len(ranked),
                             'selected_by_utility': len(chosen), 'selected_by_coverage': q - len(chosen)})
    result.sort(key=lambda f: (text_hash(SEED_TAG + ':' + f['id']), f['id']))
    if len(result) != size or len({f['id'] for f in result}) != size:
        raise ValueError('selection size/uniqueness failure')
    return result, receipts


def prepare(args):
    from reasoning.demonstration_response import make_prefix, standard_requests, read_ledger, load_plan
    from transformers import AutoTokenizer
    pool, features = Path(args.pool), Path(args.features)
    if digest(pool) != POOL_SHA or digest(features) != FEATURES_SHA:
        raise ValueError('sealed original pool/features mismatch')
    negative = json.loads(Path(args.negative_receipt).read_text())
    if digest(args.saved_plan) != negative['plan_sha256'] or digest(args.ledger) != negative['ledger_sha256']:
        raise ValueError('saved plan/ledger differs from negative preparation receipt')
    # Existing load_plan checks schema, module, grader and model-manifest hashes.
    plan = load_plan(args.saved_plan)
    if (plan['pool_sha256'] != POOL_SHA or plan['features_sha256'] != FEATURES_SHA
            or plan['n_per_condition'] != 2 or plan['seed'] != 42
            or plan['cutoff'] != 8192 or plan['temperature'] != .6 or plan['top_p'] != .95 or plan['max_new_tokens'] != 2048
            or REVISION not in plan['model']):
        raise ValueError('scorer/model/tokenizer/generation provenance differs')
    records = list(read_ledger(args.ledger, standard_requests(plan)).values())
    rows, fs = read_rows(pool), read_rows(features)
    by_id = {r['id']: r for r in rows}
    if len(rows) != 83037 or len(by_id) != 83037 or len(fs) != 83037 or {f['id'] for f in fs} != set(by_id):
        raise ValueError('wrong exact source identity coverage')
    if any(prompt_id(r['instruction']) != r['id'] for r in rows):
        raise ValueError('source prompt identity mismatch')
    # Allocate before loading the tokenizer or inspecting measurements.
    select(fs, {}, args.size)
    tok = AutoTokenizer.from_pretrained(plan['model'], trust_remote_code=False, local_files_only=True)
    scores, rejected = joined_scores(by_id, plan['candidates'], records, lambda response: make_prefix(tok, response))
    picked, cells = select(fs, scores, args.size)
    if args.inventory_only:
        print(json.dumps({'schema': SCHEMA, 'status': 'actual_static_capacity_only',
            'eligible': sum(eligible(f, CUTOFF) for f in fs),
            'occupied_clusters': len({f['cluster'] for f in fs if eligible(f, CUTOFF)}),
            'selected_unique': len(picked), 'score_available': len(scores),
            'selected_by_utility': sum(f['selection_basis'] == 'utility' for f in picked),
            'positive_scores': sum(v > 0 for v in scores.values()),
            'score_rejections': rejected, 'native_processor_audit_complete': False,
            'dataset_exported': False, 'gpu_used': False, 'cells': cells}))
        return
    from reasoning.skill_graph import load_labels
    if digest(args.labels) != plan['labels_sha256']:
        raise ValueError('training-only static label ledger differs')
    labels = {r['id']: r for r in load_labels(Path(args.labels), [r['id'] for r in rows])}
    out = Path(args.output)
    if out.exists():
        raise ValueError('use a new output directory; preserving all existing experiments')
    write_dataset_info(out)
    tok, template, proc, converter = processor(plan['model'], out)
    # Recompute complete 16K eligibility through the actual pinned processor.
    # Any static/native discrepancy fails closed; it never silently changes cells.
    native = {}
    for f in fs:
        if eligible(f, CUTOFF):
            native[f['id']] = check_row(by_id[f['id']], tok, template, proc, converter)
            if len(native) % 500 == 0:
                print(f'Native complete-original 16K audit: {len(native)} eligible rows', flush=True)
    audit, data = [], []
    for f in picked:
        row = by_id[f['id']]
        total, supervised = native[f['id']]
        audit.append({**f, 'exact_tokens': total, 'supervised_tokens': supervised,
                      'original_response_sha256': text_hash(row['output'])})
        data.append(row)
    prior = json.loads(Path(args.candidate_a).read_text())
    prior_ids = prior['selected_ids']
    if isinstance(prior_ids, dict):
        prior_ids = prior_ids['advanced']
    distribution = {name: dict(Counter(labels[f['id']][name] for f in picked))
                    for name in ('language', 'domain', 'answer_form')}
    distribution['response_tokens'] = sorted(f['response_tokens'] for f in picked)
    distribution['serialized_tokens'] = sorted(f['exact_tokens'] for f in audit)
    distribution['supervised_tokens'] = sorted(f['supervised_tokens'] for f in audit)
    receipt = {'schema': SCHEMA, 'status': 'cpu_selected_and_native_audited_not_gpu_authorized',
               'source_pool_sha256': POOL_SHA, 'features_sha256': FEATURES_SHA,
               'saved_plan_sha256': digest(args.saved_plan), 'ledger_sha256': digest(args.ledger),
               'negative_receipt_sha256': digest(args.negative_receipt), 'module_sha256': digest(__file__),
               'allocation_module_sha256': digest(ROOT / 'reasoning/select.py'),
               'native_audit_module_sha256': digest(ROOT / 'reasoning/full_length.py'),
               'dataset_info_sha256': digest(out / 'dataset_info.json'),
               'selected_ids': [f['id'] for f in picked],
               'candidate_a_manifest_sha256': digest(args.candidate_a), 'hash_seed_tag': SEED_TAG,
               'model_revision': REVISION, 'utility_rule': '(1-(normal_correct+1)/4)*max((assisted_correct-normal_correct)/4,0)',
               'utility_cell_fraction_cap': .5, 'exact_selected_unique': len(picked),
               'eligible': sum(eligible(f, CUTOFF) for f in fs), 'score_available': len(scores),
               'distribution': distribution, 'labels_sha256': digest(args.labels),
               'score_rejections': rejected, 'selected_by_utility': sum(f['selection_basis'] == 'utility' for f in picked),
               'overlap_with_candidate_a': len(set(prior_ids) & {f['id'] for f in picked}),
               'supervised_tokens_per_epoch': sum(r['supervised_tokens'] for r in audit),
               'above_8k': sum(r['exact_tokens'] > 8192 for r in audit), 'cells': cells, 'selected': audit,
               'epochs': 3, 'optimizer_updates': 5625, 'training_launched': False,
               'native_eligible_audit_count': len(native),
               'causal_assistance_audit_complete': False, 'semantic_prefix_leakage_excluded': False}
    write_json(out / 'advanced.json', alpaca(data))
    receipt['data_sha256'] = digest(out / 'advanced.json')
    write_json(out / 'selection_manifest.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('selected', 'selected_ids', 'cells', 'distribution')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', default='artifacts/prepared_extended/pool.json')
    p.add_argument('--features', default='artifacts/features_extended.json')
    p.add_argument('--saved-plan', required=True)
    p.add_argument('--ledger', required=True)
    p.add_argument('--negative-receipt', default='research/demonstration_response_negative_preparation.json')
    p.add_argument('--candidate-a', required=True)
    p.add_argument('--labels', default='artifacts/skillgraph-asft-15k/skill_labels.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=15000, choices=[15000])
    p.add_argument('--inventory-only', action='store_true', help='report capacity without native audit/export')
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
