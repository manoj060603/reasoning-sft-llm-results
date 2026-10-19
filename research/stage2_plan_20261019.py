"""CPU-only, source-bound longest-original-response curriculum preparation.

Never launches training. Native serialization and new GPU budget remain separate gates.
Run --self-test, or --data advanced.json --selection selection_manifest.json.
Without --output this prints only an inventory; source and remote state stay unchanged.
"""
import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

DATA_SHA = '0ccdb3b3e7800f181cd99cb16f433d81d359d86e6d6fa0d089f61c36f56185cc'
SELECTION_SHA = 'f4a43fbc36ce72e74cb9a5f1afa92a31fd5071d753420657eb7440026f5ee7ed'
FIELDS = ('instruction', 'input', 'output', 'system')
COLUMNS = {'prompt': 'instruction', 'query': 'input', 'response': 'output', 'system': 'system'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prompt_id(text):
    norm = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
    return text_sha(norm)


def select_rows(rows, selection, size=3000, source_size=15000):
    """Rank only by original supervised token count, ties by source identity.

    The native all-row source audit's supervised count includes the same terminal
    EOS for every row. No outcomes, NLL, cluster scores or generated text are read.
    """
    entries = selection['selected']
    if len(rows) != source_size or len(entries) != source_size:
        raise ValueError('source row count mismatch')
    if size <= 0 or size > source_size:
        raise ValueError('invalid exact subset size')
    ids = selection['selected_ids']
    if len(ids) != source_size or len(set(ids)) != source_size:
        raise ValueError('source identities missing or duplicated')
    ranked = []
    for i, (row, entry) in enumerate(zip(rows, entries)):
        if any(not isinstance(row.get(k), str) for k in FIELDS):
            raise ValueError('all four original text fields are required')
        if entry['id'] != ids[i] or prompt_id(row['instruction']) != entry['id']:
            raise ValueError('original order or prompt identity mismatch')
        if text_sha(row['output']) != entry['original_response_sha256']:
            raise ValueError('original target bytes changed')
        length = entry['supervised_tokens']
        if type(length) is not int or not 0 < length <= entry['exact_tokens'] <= 16384:
            raise ValueError('invalid native length metadata')
        ranked.append((length, entry['id'], i))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    picked = ranked[:size]
    return [{k: rows[i][k] for k in FIELDS} for _, _, i in picked], picked


def exclude_without_backfill(subset, ranking, excluded):
    keep = [(row, rank) for row, rank in zip(subset, ranking) if rank[1] not in excluded]
    if not keep:
        raise ValueError('no stage2 rows remain after exclusions')
    return [x[0] for x in keep], [x[1] for x in keep]


PINNED_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
BENCHMARK_COUNTS = {'amc': 40, 'aime25': 30, 'aime': 30, 'math': 500,
                    'cn_math_2024': 30, 'kaoyan': 199, 'minerva': 272,
                    'olympiadbench': 675, 'gpqa': 198}


def validate_exclusion_receipt(receipt, source_ids, benchmark_manifest):
    if (receipt.get('schema') != 'two-plan-prompt-contamination-v1'
            or receipt.get('source_data_sha256') != DATA_SHA
            or receipt.get('selected_manifest_sha256') != SELECTION_SHA
            or receipt.get('pool_sha256') != 'c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9'
            or receipt.get('prompts') != 15000 or receipt.get('benchmark_records') != 1974
            or receipt.get('rule') != 'NFKC casefold whitespace exact or any shared 13 whitespace words'
            or receipt.get('benchmark_answers_or_outcomes_used') is not False
            or receipt.get('gpu_used') is not False
            or receipt.get('chinese_word_segmentation') is not False):
        raise ValueError('decontamination receipt source or exact/13gram rule mismatch')
    if benchmark_manifest is None:
        raise ValueError('decontamination receipt requires actual matched baseline manifest')
    baseline = json.loads(Path(benchmark_manifest).read_text())
    benchmarks = receipt.get('benchmark_binding', {})
    if set(benchmarks) != set(BENCHMARK_COUNTS):
        raise ValueError('decontamination receipt must bind all nine benchmarks')
    for key, count in BENCHMARK_COUNTS.items():
        item = benchmarks[key]
        if (item.get('records') != count or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))
                or baseline['data'][key] != item['sha256']):
            raise ValueError('decontamination benchmark count/hash does not match baseline')
    excluded_list = receipt.get('excluded_source_ids')
    if not isinstance(excluded_list, list) or any(not isinstance(v, str) for v in excluded_list):
        raise ValueError('decontamination excluded_source_ids missing')
    excluded = set(excluded_list)
    if len(excluded) != len(excluded_list) or not excluded <= set(source_ids):
        raise ValueError('decontamination exclusion IDs duplicated or outside frozen15K')
    hits = receipt.get('hits')
    if not isinstance(hits, list) or [h.get('id') for h in hits] != excluded_list:
        raise ValueError('decontamination exclusion IDs differ from actual hit records')
    if any(type(h.get('exact')) is not bool or type(h.get('shared_13_word_gram')) is not bool
           or not (h['exact'] or h['shared_13_word_gram']) for h in hits):
        raise ValueError('decontamination hit records lack exact or13gram matches')
    return excluded


def native_subset_audit(subset, ranking, model):
    """CPU tokenizer only. Verify actual native stage2 registration/conversion.

    This deliberately does not call full_length.check_row, whose original
    empty-input assumption is unsuitable for a reusable new preparer.
    """
    import tempfile
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.converter import get_dataset_converter
    model_path = Path(model)
    if model_path.name != PINNED_REVISION or not model_path.is_dir():
        raise ValueError('native audit requires local pinned original tokenizer snapshot')
    with tempfile.TemporaryDirectory(prefix='stage2-native-cpu-') as temp:
        registration = {'stage2_hard3k': {'file_name': 'stage2_hard3k.json', 'columns': COLUMNS}}
        Path(temp, 'dataset_info.json').write_text(json.dumps(registration))
        args = DataArguments(template='qwen', cutoff_len=16384, train_on_prompt=False, packing=False)
        attrs = get_dataset_list(['stage2_hard3k'], temp)
        if len(attrs) != 1 or any(getattr(attrs[0], k) != v for k, v in COLUMNS.items()):
            raise ValueError('actual native stage2 column registration mismatch')
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
        template = get_template_and_fix_tokenizer(tok, args)
        processor = SupervisedDatasetProcessor(template=template, tokenizer=tok, processor=None, data_args=args)
        converter = get_dataset_converter('alpaca', attrs[0], args)

        def check(row):
            converted = converter(row)
            user = '\n'.join(row[k] for k in ('instruction', 'input') if row[k])
            prompt = [{'role': 'user', 'content': user}]
            response = [{'role': 'assistant', 'content': row['output']}]
            if (converted['_system'] != row['system'] or converted['_prompt'] != prompt
                    or converted['_response'] != response):
                raise ValueError('native conversion changed original system/input/prompt/response')
            pairs = template.encode_multiturn(tok, prompt + response, row['system'], None)
            if len(pairs) != 1: raise ValueError('native row must have exactly one turn')
            source, target = pairs[0]
            expected = target + ([tok.eos_token_id] if template.efficient_eos else [])
            batch = processor.preprocess_dataset({k: [v] for k, v in converted.items()})
            if len(batch['input_ids']) != 1: raise ValueError('native processor dropped row')
            ids, labels = batch['input_ids'][0], batch['labels'][0]
            actual = [x for x in labels if x != -100]
            if (len(source) + len(expected) > 16384 or len(ids) > 16384
                    or actual != expected or ids != source + expected
                    or labels[:len(source)] != [-100] * len(source)
                    or tok.eos_token_id not in actual):
                raise ValueError('native truncation, mask, input or supervised EOS mismatch')
            decoded = tok.decode(actual, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not any(decoded.startswith(v) for v in (row['output'], unicodedata.normalize('NFC', row['output']))):
                raise ValueError('native response differs beyond tokenizer canonical composition')
            return len(ids), len(actual)

        fixture = {'instruction': 'Native prompt fixture.', 'input': 'Keep this input verbatim.',
                   'system': 'Explicit original system.', 'output': 'Reason y\u0304 carefully. \\boxed{5}'}
        check(fixture)
        fixture['output'] = 'Reason carefully. ' * 18000 + '\\boxed{5}'
        try: check(fixture)
        except ValueError: pass
        else: raise ValueError('native overlength rejection fixture failed')
        total_tokens = 0
        for row, (expected_count, _, _) in zip(subset, ranking):
            total, supervised = check(row)
            if supervised != expected_count:
                raise ValueError('native subset length differs from frozen original receipt')
            total_tokens += total
        return {'rows': len(subset), 'converter': 'native alpaca stage2_hard3k',
                'all_input_fields_preserved': True, 'prompt_tokens_masked': True,
                'original_targets_complete': True, 'eos_supervised': True,
                'native_supervised_count_matches_source': True,
                'serialized_tokens_per_epoch': total_tokens,
                'tokenizer_revision': PINNED_REVISION,
                'tokenizer_files_sha256': {p.name: digest(p) for p in sorted(model_path.glob('*.json'))}}



def prepare(data, selection, output=None, exclusions=None, exclusions_sha256=None, native_audit=False, model=None, benchmark_manifest=None, draft_config=None):
    import yaml
    if draft_config is None:
        raise ValueError('draft config path required for fail-closed do_train:false binding')
    config = yaml.safe_load(Path(draft_config).read_text())
    if config.get('do_train') is not False or 'deepspeed' in config:
        raise ValueError('draft config must disable training and avoid stale DeepSpeed')
    expected_config = {'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 4,
                       'num_train_epochs': 2, 'dataset': 'stage2_hard3k', 'cutoff_len': 16384,
                       'template': 'qwen', 'train_on_prompt': False, 'packing': False,
                       'learning_rate': 5e-6, 'warmup_ratio': 0.03, 'lr_scheduler_type': 'cosine',
                       'seed': 42, 'data_seed': 42, 'average_tokens_across_devices': True,
                       'finetuning_type': 'full', 'flash_attn': 'sdpa', 'overwrite_output_dir': False}
    if any(config.get(k) != v for k, v in expected_config.items()) or 'max_steps' in config:
        raise ValueError('draft config changes frozen stage2 curriculum')
    if digest(data) != DATA_SHA or digest(selection) != SELECTION_SHA:
        raise ValueError('frozen source hashes mismatch')
    rows = json.loads(Path(data).read_text(encoding='utf-8'))
    selected = json.loads(Path(selection).read_text(encoding='utf-8'))
    subset, ranking = select_rows(rows, selected)
    excluded_count = 0
    exclusion_hash = None
    if exclusions is not None:
        exclusion_hash = digest(exclusions)
        if not exclusions_sha256 or exclusion_hash != exclusions_sha256:
            raise ValueError('exclusion receipt requires its exact expected SHA256')
        receipt = json.loads(Path(exclusions).read_text(encoding='utf-8'))
        excluded = validate_exclusion_receipt(receipt, selected['selected_ids'], benchmark_manifest)
        # Freeze longest 3K FIRST, then remove audit hits; never backfill.
        subset, ranking = exclude_without_backfill(subset, ranking, excluded)
        excluded_count = 3000 - len(subset)
    native = native_subset_audit(subset, ranking, model) if native_audit else None
    summary = {
        'schema': 'stage2-longest-original-3k-cpu-draft-v1',
        'source_data_sha256': DATA_SHA, 'source_selection_sha256': SELECTION_SHA,
        'selection_rule': 'descending native supervised_tokens, tie ascending original id',
        'source_unique_prompts': 15000, 'subset_unique_prompts': len(subset),
        'ranked_before_exclusions': 3000, 'removed_without_backfill': excluded_count,
        'exclusion_receipt_sha256': exclusion_hash,
        'planned_epochs': 2, 'planned_example_presentations': 2 * len(subset),
        'planned_world_size': 2, 'planned_effective_batch': 8, 'planned_updates': 2 * math.ceil(len(subset) / 8),
        'native_update_schedule_requires_verification': True,
        'response_token_range_including_eos': [ranking[-1][0], ranking[0][0]],
        'supervised_tokens_per_epoch': sum(v[0] for v in ranking),
        'source_original_target_hashes_verified': True,
        'all_original_fields_preserved': True,
        'public_outcomes_used_for_ranking': False,
        'source_ids_sha256': text_sha('\n'.join(v[1] for v in ranking)),
        'native_subset_serialization_audit_complete': native is not None,
        'native_subset_audit': native,
        'decontamination_audit_complete': exclusions is not None,
        'decontamination_rule': 'exact plus13whitespacewordgram; removehitsafterlongest3Kwithoutbackfill' if exclusions else None,
        'decontamination_limitation': '13gram uses whitespace words; no Chinese segmentation or semantic overlap claim',
        'config_do_train': False,
        'draft_config_sha256': digest(draft_config),
        'benchmark_manifest_sha256': digest(benchmark_manifest) if exclusions else None,
        'token_accuracy_measured': False,
        'launch_ready': False,
        'blockers': ['40 GPU-hour cap fully committed; incremental budget unresolved',
                     'native all-row subset converter, mask and EOS audit not run',
                     'source-only contamination audit policy and execution unresolved',
                     'stage-2 native runtime, receipt and allocation guard not integrated',
                     'two-B200 longest-row throughput and deadline fit unverified'],
        'early_epoch_checkpoints': {'1875': 'unavailable; do not substitute 5600',
                                    '3750': 'unavailable; do not substitute 5600'},
        'no_95_percent_accuracy_gate_validated': True,
    }
    if native is not None:
        summary['blockers'].remove('native all-row subset converter, mask and EOS audit not run')
    if exclusions is not None:
        summary['blockers'].remove('source-only contamination audit policy and execution unresolved')
    if output is not None:
        out = Path(output)
        if out.exists():
            raise ValueError('refusing to overwrite existing stage2 preparation directory')
        out.mkdir(parents=True)
        payload = json.dumps(subset, ensure_ascii=False, indent=2) + '\n'
        (out / 'stage2_hard3k.json').write_text(payload, encoding='utf-8')
        (out / 'dataset_info.json').write_text(json.dumps({
            'stage2_hard3k': {'file_name': 'stage2_hard3k.json', 'columns': COLUMNS}}, indent=2) + '\n')
        summary['prepared_dataset_sha256'] = digest(out / 'stage2_hard3k.json')
        summary['dataset_info_sha256'] = digest(out / 'dataset_info.json')
        summary['selected'] = [{'id': ident, 'supervised_tokens': n, 'source_index': i,
                                'original_response_sha256': text_sha(rows[i]['output']),
                                'original_system_sha256': text_sha(rows[i]['system'])}
                               for n, ident, i in ranking]
        (out / 'stage2_preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {k: v for k, v in summary.items() if k != 'selected'}


def self_test():
    rows = [{'instruction': f'Problem {i}', 'input': 'keep input', 'system': 's\u0304',
             'output': f'original response {i}\n'} for i in range(6)]
    entries = [{'id': prompt_id(r['instruction']), 'supervised_tokens': 10 + i // 2,
                'exact_tokens': 20 + i // 2, 'original_response_sha256': text_sha(r['output']),
                'benchmark_score': -999} for i, r in enumerate(rows)]
    source = {'selected': entries, 'selected_ids': [r['id'] for r in entries]}
    subset, rank = select_rows(rows, source, 3, 6)
    assert [x[0] for x in rank] == [12, 12, 11]
    assert subset == [{k: rows[x[2]][k] for k in FIELDS} for x in rank]
    # Ties deterministically use IDs; arbitrary score metadata cannot change order.
    for entry in entries: entry['benchmark_score'] = 999
    assert select_rows(rows, source, 3, 6) == (subset, rank)
    assert rank[:2] == sorted(rank[:2], key=lambda x: x[1])
    reduced, reduced_rank = exclude_without_backfill(subset, rank, {rank[0][1]})
    assert reduced == subset[1:] and reduced_rank == rank[1:] and len(reduced) == 2
    try: exclude_without_backfill(subset, rank, {x[1] for x in rank})
    except ValueError: pass
    else: raise AssertionError('empty subset accepted')
    for changed in ('output', 'instruction'):
        corrupt = [dict(r) for r in rows]; corrupt[0][changed] += 'mutation'
        try: select_rows(corrupt, source, 3, 6)
        except ValueError: pass
        else: raise AssertionError('mutated source accepted')
    corrupt = [dict(r) for r in rows]; del corrupt[0]['input']
    try: select_rows(corrupt, source, 3, 6)
    except ValueError: pass
    else: raise AssertionError('missing input accepted')
    # Receipts must bind all nine baseline dataset hashes, not only assert a flag.
    import tempfile
    receipt = {'schema': 'two-plan-prompt-contamination-v1', 'source_data_sha256': DATA_SHA,
               'selected_manifest_sha256': SELECTION_SHA,
               'pool_sha256': 'c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9',
               'prompts': 15000, 'benchmark_records': 1974,
               'rule': 'NFKC casefold whitespace exact or any shared 13 whitespace words',
               'benchmark_answers_or_outcomes_used': False, 'gpu_used': False, 'chinese_word_segmentation': False,
               'benchmark_binding': {k: {'sha256': text_sha(k), 'records': v} for k, v in BENCHMARK_COUNTS.items()},
               'excluded_source_ids': [], 'hits': []}
    with tempfile.TemporaryDirectory(prefix='stage2-receipt-test-') as tmp:
        baseline = Path(tmp, 'manifest.json')
        baseline.write_text(json.dumps({'data': {k: text_sha(k) for k in BENCHMARK_COUNTS}}))
        assert validate_exclusion_receipt(receipt, source['selected_ids'], baseline) == set()
        receipt['benchmark_binding']['math']['sha256'] = '0' * 64
        try: validate_exclusion_receipt(receipt, source['selected_ids'], baseline)
        except ValueError: pass
        else: raise AssertionError('unmatched benchmark hash accepted')
    print('STAGE2_CPU_SELECTION_TESTS_PASSED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--data'); p.add_argument('--selection'); p.add_argument('--output')
    p.add_argument('--exclusions'); p.add_argument('--exclusions-sha256')
    p.add_argument('--native-audit', action='store_true'); p.add_argument('--model')
    p.add_argument('--benchmark-manifest'); p.add_argument('--draft-config', required=False)
    a = p.parse_args()
    if a.self_test: return self_test()
    if not a.data or not a.selection: p.error('--data and --selection are required')
    if a.native_audit and not a.model: p.error('--native-audit requires --model pinned snapshot')
    print(json.dumps(prepare(a.data, a.selection, a.output, a.exclusions, a.exclusions_sha256, a.native_audit, a.model, a.benchmark_manifest, a.draft_config), indent=2))


if __name__ == '__main__': main()
