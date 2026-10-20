"""Stage publication metadata from complete, hash-verified experiment artifacts.

Weights and original datasets are referenced in a manifest, never copied or
uploaded. The eight supplied benchmarks are mandatory; AIME25 is supplementary.
CLI hashing runs in a compute allocation. Tests call the local validation API.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from deadline.audit_results import OFFICIAL, HISTORICAL_SEVEN, audit_run, compare_coverage
from deadline.compatible_coverage import COLUMNS, content_sha256
from deadline.train_full import digest, fixed_order, json_digest, weight_files
from reasoning.data import alpaca, prompt_id


ROWS, CUTOFF, UPDATES = 15000, 8192, 1875
MODEL_SIDECARS = frozenset({
    'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'chat_template.jinja', 'model.safetensors.index.json',
})
SOURCE_SHA = 'f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8'
LIMITATIONS = (
    'The random control is uniform within the broad prepared source pool after deduplication, '
    'benchmark/source-heldout, prompt-capacity, and reserved-probe exclusions; it is not a '
    'literal uniform draw over the untouched original 100K rows. '
    'Original exported responses are unchanged. Random training may use declared 8K prefixes '
    'without an invented EOS. Advanced targets must fit completely. '
    'The advanced method combines eligibility filtering, reservoir construction, transfer gain, '
    'and coverage. Separate eligible-random and coverage-only ablations are needed to isolate '
    'their contributions. Relative teacher-forced loss reduction is a proxy, not mathematical '
    'verification or proof of causal data utility. Global novelty, state-of-the-art performance, '
    'and equivalence to a historical leaderboard are not established.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Files:
    """Recheck content hashes while avoiding repeated reads of the same GB shard."""
    def __init__(self):
        self.records = {}

    def hash(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f'missing required artifact: {path}')
        before = path.stat()
        stamp = (before.st_size, before.st_mtime_ns, before.st_ino)
        cached = self.records.get(str(path))
        if cached and cached['stamp'] == stamp:
            actual = cached['sha256']
        else:
            actual = digest(path)
            after = path.stat()
            require(stamp == (after.st_size, after.st_mtime_ns, after.st_ino), f'artifact changed during hashing: {path}')
            self.records[str(path)] = dict(sha256=actual, bytes=before.st_size, stamp=stamp)
        require(expected is None or actual == expected, f'artifact hash mismatch: {path}')
        return actual

    def unchanged(self):
        for path, record in self.records.items():
            now = Path(path).stat()
            require(record['stamp'] == (now.st_size, now.st_mtime_ns, now.st_ino),
                    f'artifact changed during packaging: {path}')

    def inventory(self):
        return {path: {key: row[key] for key in ('sha256', 'bytes')} for path, row in self.records.items()}


def seal(files, directory, name):
    path = directory / name
    files.hash(path)
    return read(path)


def relative_file(directory, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'invalid relative artifact path')
    return directory / path


def verify_selection(root, plan, files):
    artifacts = root / 'artifacts'
    prepared = seal(files, artifacts, 'preparation_complete.json')
    require(prepared.get('source_revision') == plan['source_revision'] and
            prepared.get('model_revision') == plan['base_revision'] and
            prepared.get('source_sha256') == SOURCE_SHA, 'source/base revision differs from the frozen plan')
    complete = seal(files, artifacts, 'selection_complete.json')
    require(complete.get('complete') is True and complete.get('schema') == 'deadline-crossfit-selection-complete-v1',
            'selection has no complete sealed receipt')
    require(complete.get('rows_per_arm') == ROWS and complete.get('cutoff') == CUTOFF and
            complete.get('random_control_ids_unchanged') is True and complete.get('advanced_truncated_rows') == 0,
            'selection size, context, random identity or advanced completeness differs')
    base = seal(files, artifacts, 'model_identity.json')
    base_payload = dict(base)
    base_checksum = base_payload.pop('identity_sha256', None)
    require(json_digest(base_payload) == base_checksum == complete.get('base_model_sha256'),
            'frozen original-base identity differs from selection')
    for name, info in base.get('files', {}).items():
        require(Path(name).name == name, 'invalid base identity filename')
        files.hash(Path(base['path']) / name, info['sha256'])
    hashes = complete.get('files_sha256', {})
    required = {'selected/random.json', 'selected/advanced.json', 'selected/dataset_info.json',
                'selected/selection_manifest.json', 'training/advanced/cache_meta.json',
                'training/advanced/dataset.json', 'training/advanced/selected_ids.json',
                'training/advanced/training.jsonl'}
    require(required.issubset(hashes), 'selection completion receipt does not bind every required output')
    for name, checksum in hashes.items():
        files.hash(relative_file(artifacts, name), checksum)
    selection = read(artifacts / 'selected/selection_manifest.json')
    identity = dict(selection)
    recorded = identity.pop('selection_sha256', None)
    require(recorded == content_sha256(identity) == complete.get('selection_sha256'), 'selection manifest checksum differs')
    require(selection.get('size') == ROWS and selection.get('cutoff') == CUTOFF and
            selection.get('seed') == plan['selection_seed'] and selection.get('strategy') == 'transfer_gain',
            'selection method differs from the frozen design')
    columns = read(artifacts / 'selected/dataset_info.json')
    require(all(columns.get(role, {}).get('columns') == COLUMNS for role in ('random', 'advanced')),
            'dataset prompt/input/response/system mapping is incomplete')
    for name in ('probeA', 'probeB'):
        info = complete.get('probe_artifacts', {}).get(name, {})
        folder = root / 'runs' / name
        files.hash(folder / 'probe_receipt.json', info.get('receipt_sha256', 'missing'))
        files.hash(folder / 'adapter/adapter_model.safetensors', info.get('checkpoint_sha256', 'missing'))
        require(read(folder / 'probe_receipt.json').get('complete') is True, 'probe training is incomplete')
    for fold in ('fold0', 'fold1'):
        info = complete.get('score_artifacts', {}).get(fold, {})
        folder = root / 'runs' / ('scores_' + fold)
        for name, field in (('score_receipt.json', 'receipt_sha256'), ('run_manifest.json', 'manifest_sha256'),
                            ('scores.jsonl', 'scores_sha256')):
            files.hash(folder / name, info.get(field, 'missing'))
        require(read(folder / 'score_receipt.json').get('complete') is True, 'opposite-fold scoring is incomplete')
    pool = seal(files, artifacts, 'pool.json')
    lookup = {row['id']: row for row in pool}
    require(len(lookup) == len(pool), 'prepared source pool has duplicate IDs')
    require(selection.get('source_pool_sha256') == files.hash(artifacts / 'pool.json'), 'prepared source pool differs from selection')
    return complete, selection, lookup


def verify_training(root, role, plan, selection_complete, selection, pool, files):
    folder, data = root / 'runs' / role, root / 'artifacts/training' / role
    manifest = seal(files, folder, 'manifest.json')
    complete = seal(files, folder, 'complete.json')
    binding = json_digest(manifest)
    require(complete.get('binding_sha256') == binding and complete.get('steps') == UPDATES and
            complete.get('full_parameter') is True and complete.get('unique_examples') == ROWS and
            complete.get('example_presentations') == ROWS and complete.get('padding_presentations') == 0,
            f'{role}: full training/unique example receipt is incomplete or mismatched')
    require(type(complete.get('training_loss')) in (float, int) and math.isfinite(complete['training_loss']),
            f'{role}: final training loss is not finite')
    recipe = manifest.get('recipe', {})
    expected = dict(max_steps=UPDATES, per_device_batch=1, accumulation=8, cutoff=CUTOFF,
                    learning_rate=2e-6, lr_scheduler_type='cosine', warmup_ratio=.03,
                    weight_decay=.01, max_grad_norm=1., parameter_dtype='float32',
                    optimizer_moment_dtype='float32', compute_dtype='bfloat16_autocast',
                    full_parameter=True, seed=42, data_seed=42)
    require(all(recipe.get(key) == value for key, value in expected.items()), f'{role}: training recipe differs')
    require(plan['final_training'].get('cutoff') == CUTOFF and plan['final_training'].get('updates') == UPDATES and
            plan['final_training'].get('learning_rate') == recipe['learning_rate'], 'plan/training recipe differs')
    require(isinstance(manifest.get('base_identity'), dict) and manifest['base_identity'],
            f'{role}: trained original-base identity is missing')
    for name, checksum in manifest['base_identity'].items():
        require(Path(name).name == name, 'invalid trained-base identity filename')
        files.hash(Path(manifest['base']) / name, checksum)
    meta = seal(files, data, 'cache_meta.json')
    require(meta == manifest.get('cache_metadata') and meta.get('dataset_role') == role and
            meta.get('rows') == ROWS and meta.get('cutoff') == CUTOFF and
            meta.get('base_model_sha256') == selection_complete.get('base_model_sha256') and
            meta.get('unique_rows_dropped') == 0 and meta.get('target_rewriting') is False,
            f'{role}: training cache role/source/metadata differs')
    files.hash(data / 'training.jsonl', meta.get('cache_sha256', 'missing'))
    files.hash(data / 'dataset.json', meta.get('dataset_sha256', 'missing'))
    files.hash(data / 'selected_ids.json', meta.get('selected_ids_sha256', 'missing'))
    require(manifest.get('cache', {}).get('cache_sha256') == meta['cache_sha256'] and
            manifest.get('cache', {}).get('cache_meta_sha256') == files.hash(data / 'cache_meta.json'),
            f'{role}: cached data differs from the trained manifest')
    ids = read(data / 'selected_ids.json')
    expected_ids = selection.get('datasets', {}).get(role, {}).get('selected_ids', [])
    require(ids == expected_ids and len(ids) == len(set(ids)) == ROWS and set(ids).issubset(pool),
            f'{role}: selected original identities differ')
    records = read(data / 'dataset.json')
    require(records == alpaca([pool[identifier] for identifier in ids]) and
            [prompt_id(row['instruction']) for row in records] == ids,
            f'{role}: original exported responses were modified or reordered')
    require(content_sha256(records) == selection['datasets'][role]['exported_rows_sha256'],
            f'{role}: exported dataset differs from sealed selection')
    audit = seal(files, folder, 'cache_audit.json')
    truncated, missing = meta.get('truncated_rows'), audit.get('supervised_eos_missing_rows')
    require(type(truncated) is int and type(missing) is int and 0 <= missing <= truncated <= ROWS and
            audit.get('cache_sha256') == meta['cache_sha256'] and audit.get('truncated_rows') == truncated,
            f'{role}: truncation/EOS audit is missing or inconsistent')
    require(role != 'advanced' or (truncated == missing == 0 and meta.get('truncation_policy') == 'none'),
            'advanced targets must retain complete original responses and EOS')
    consumption = seal(files, folder, 'consumption_final.json')
    order = fixed_order(ROWS, 42)
    require(consumption.get('binding_sha256') == binding and consumption.get('completed_updates') == UPDATES and
            consumption.get('consumed_indices') == order and consumption.get('order_sha256') == json_digest(order) and
            manifest.get('order_sha256') == json_digest(order) and consumption.get('padding_presentations') == 0,
            f'{role}: exact one-pass consumption proof differs')
    update_path = folder / 'finite_updates.jsonl'
    files.hash(update_path)
    steps = set()
    with update_path.open(encoding='utf-8') as stream:
        for line in stream:
            update = json.loads(line)
            step, norm = update.get('step'), update.get('gradient_norm')
            require(type(step) is int and 1 <= step <= UPDATES and type(norm) in (int, float) and
                    math.isfinite(norm) and norm >= 0 and update.get('actual_presentations') == step * 8,
                    f'{role}: invalid finite-update history')
            steps.add(step)
    # Explicit resume attempts can leave duplicate logs for discarded unsaved
    # updates. The bound final consumption receipt identifies the trained path.
    require(steps == set(range(1, UPDATES + 1)), f'{role}: finite-update history is incomplete')
    model = folder / 'model'
    require(Path(complete.get('model', '')).resolve() == model.resolve(), f'{role}: final model path differs')
    verification = complete.get('final_model_verification', {})
    require(verification.get('all_finite') is True and verification.get('parameter_dtype') == 'float32' and
            verification.get('parameter_names_verified', 0) > 0, f'{role}: final finite FP32 tensor receipt is absent')
    weights = weight_files(model)
    require(set(verification.get('files', {})) == {path.name for path in weights}, f'{role}: weight shard set differs')
    for path in weights:
        files.hash(path, verification['files'][path.name])
    if (model / 'model.safetensors.index.json').is_file():
        files.hash(model / 'model.safetensors.index.json', verification.get('model_index_sha256', 'missing'))
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        files.hash(model / name)
    for name in sorted(MODEL_SIDECARS):
        if (model / name).is_file():
            files.hash(model / name)
    return dict(role=role, folder=str(model.resolve()), dataset=str((data / 'dataset.json').resolve()),
                dataset_sha256=meta['dataset_sha256'], binding_sha256=binding,
                recipe=recipe, base=manifest.get('base'), base_identity=manifest.get('base_identity'),
                truncated_training_rows=truncated, missing_supervised_eos_rows=missing,
                finite_tensor_evidence='training completion receipt; serialized weight hashes rechecked',
                model_verification=verification)


def verify_evaluations(root, trained, files):
    for role in ('original', 'random', 'advanced'):
        for name in ('manifest.json', 'metrics.json'):
            files.hash(root / 'results' / role / name)
    runs = {role: audit_run(root / 'results' / role) for role in ('original', 'random', 'advanced')}
    matched = compare_coverage(runs)
    require(matched['matched'], 'mandatory eight-benchmark evaluation is missing or mismatched: ' + '; '.join(matched['errors']))
    for role, run in runs.items():
        folder = root / 'results' / role
        for name in ('manifest.json', 'metrics.json'):
            files.hash(folder / name)
        manifest = run['manifest']
        expected_model = Path(trained['random']['base']) if role == 'original' else Path(trained[role]['folder'])
        require(Path(manifest.get('model', '')).resolve() == expected_model.resolve(), f'{role}: evaluated model path differs')
        identity = manifest.get('identity', {})
        require(isinstance(identity, dict) and identity and 'repo' not in identity, f'{role}: local evaluated model hashes are required')
        for name, checksum in identity.items():
            require(Path(name).name == name, 'invalid model identity filename')
            files.hash(expected_model / name, checksum)
        required = {path.name for path in weight_files(expected_model)} | {'config.json', 'tokenizer_config.json', 'tokenizer.json'}
        require(required.issubset(identity), f'{role}: evaluation identity does not bind every model shard/tokenizer')
        for benchmark in OFFICIAL:
            for path in sorted((folder / benchmark).glob('batch_*.json')):
                files.hash(path)
    supplement = compare_coverage(runs, names=HISTORICAL_SEVEN)['matched']
    summary = dict(matched_official_eight=True, optional_aime25_matched=supplement,
                   historical_leaderboard_equivalence_verified=False, models={})
    for role, run in runs.items():
        names = list(OFFICIAL) + (['aime25'] if supplement else [])
        summary['models'][role] = dict(
            benchmarks={name: {'metrics': run['benchmarks'][name]['metrics'],
                               'evidence': run['benchmarks'][name]['evidence']} for name in names},
            official_eight_macro=run['official_eight_macro'],
            supplemented_historical_seven_internal_macro=run['supplemented_historical_seven_macro'] if supplement else None)
    return summary


def score_table(summary):
    lines = ['| Benchmark | Original p@1 / p@8 | Random p@1 / p@8 | Advanced p@1 / p@8 |',
             '| --- | ---: | ---: | ---: |']
    for name in OFFICIAL:
        cells = []
        for role in ('original', 'random', 'advanced'):
            metric = summary['models'][role]['benchmarks'][name]['metrics']
            cells.append(f"{100 * metric['pass_at_1']:.2f}% / {100 * metric['pass_at_8']:.2f}%")
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    cells = []
    for role in ('original', 'random', 'advanced'):
        metric = summary['models'][role]['official_eight_macro']
        cells.append(f"{100 * metric['pass_at_1']:.2f}% / {100 * metric['pass_at_8']:.2f}%")
    lines.append('| Official eight macro | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def model_card(role, trained, summary, plan):
    evidence = sorted({row['evidence'] for row in summary['models'][role]['benchmarks'].values()})
    title = 'Broad prepared-source random control' if role == 'random' else plan['method']
    return ('---\nlibrary_name: transformers\npipeline_tag: text-generation\n'
            'base_model: Qwen/Qwen2.5-3B-Instruct\nlicense: other\nlicense_name: qwen-research\n'
            f"license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/{plan['base_revision']}/LICENSE\n"
            'tags:\n- reasoning\n- supervised-fine-tuning\n---\n\n'
            f'# Qwen2.5-3B — {title}\n\n'
            '**Built with Qwen.** The published weight files were modified through supervised fine-tuning. '
            'The repository retains the upstream research-license agreement and attribution notice.\n\n'
            'Fresh full-parameter SFT from the original base; no previous team model initializes this checkpoint. '
            f"Base revision: `{plan['base_revision']}`. Training uses 15,000 original pairs, 8,192 tokens, "
            '1,875 updates, global batch eight, AdamW at 2e-6, cosine decay and 3% warmup. '
            'Parameters and optimizer moments use FP32; computation uses BF16 autocast. '
            'Inference evaluation loads BF16. The objective averages supervised-token CE within each sequence, '
            'then averages the examples in each update.\n\n'
            f"Actual training prefixes: {trained['truncated_training_rows']} truncated rows; "
            f"{trained['missing_supervised_eos_rows']} rows lack a supervised EOS.\n\n"
            'Evaluation: eight samples, temperature 0.6, top-p 0.95, maximum 32768 new tokens, '
            'native 32768 total context, unchanged supplied grader. The following table covers the eight supplied tasks.\n\n'
            + score_table(summary) + '\n\nEvidence: ' + '; '.join(evidence) + '.\n\n'
            + ('Supplementary AIME25 is complete and matched; its seven-task internal macro is separately recorded in evaluation_summary.json.\n\n'
               if summary['optional_aime25_matched'] else 'Supplementary AIME25 is not complete and matched across all three models; no seven-task aggregate is claimed.\n\n')
            + LIMITATIONS + f"\n\nThe upstream 3B base uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/{plan['base_revision']}/LICENSE). "
            'Its model-card license identifier is `other`, with license name `qwen-research`. '
            'Publication metadata is staged; repository URLs and upload completion are not inferred.\n')


def dataset_card(role, trained, plan):
    policy = ('Uniform sample from the broad prepared source pool, without advanced boxed-answer, repetition, '
              'complete-target length, topic or transfer-gain filters.' if role == 'random' else
              'Complete context-safe originals selected by cross-fitted transfer gain plus topic/length coverage. '
              'Opposite-fold probes rank relative response-loss reduction; at most half each quota cell uses positive-gain leaders.')
    return ('---\ntask_categories:\n- text-generation\nsize_categories:\n- 10K<n<100K\n---\n\n'
            f'# Reasoning SFT — {role} 15K original pairs\n\n'
            f"Source: [redsgnaoh/acereason11_100k](https://huggingface.co/datasets/redsgnaoh/acereason11_100k), revision `{plan['source_revision']}`. "
            'The dataset has 15,000 original instruction/input/output/system records. '
            'The original exported responses remain unchanged; this file is the dataset to upload, not a shortened token cache.\n\n'
            + policy + '\n\n'
            f"Dataset SHA-256: `{trained['dataset_sha256']}`. Selection seed: {plan['selection_seed']}. "
            'Held-out source questions, benchmark-question overlap and reserved probe training examples are excluded.\n\n'
            f"At the shared 8K training cutoff, {trained['truncated_training_rows']} selected rows use prefixes and "
            f"{trained['missing_supervised_eos_rows']} lack supervised EOS; these training effects do not rewrite this exported dataset.\n\n"
            + LIMITATIONS + '\n\nUpstream data attribution: [NVIDIA AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) '
            'lists CC BY 4.0. The actual class subset used here is credited separately to redsgnaoh/acereason11_100k. '
            'A separate subset license has not been verified or asserted.\n')


def build_package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'refusing to overwrite an existing delivery package')
    files = Files()
    plan = seal(files, root, 'experiment_plan.json')
    require(plan.get('selection', {}).get('cutoff') == CUTOFF and plan.get('final_training', {}).get('cutoff') == CUTOFF,
            'packager requires the frozen 8K experiment plan')
    complete, selection, pool = verify_selection(root, plan, files)
    trained = {role: verify_training(root, role, plan, complete, selection, pool, files) for role in ('random', 'advanced')}
    require(trained['random']['recipe'] == trained['advanced']['recipe'] and
            trained['random']['base_identity'] == trained['advanced']['base_identity'] and
            trained['random']['base'] == trained['advanced']['base'], 'trained arms do not share one base and recipe')
    summary = verify_evaluations(root, trained, files)
    files.unchanged()
    metadata = dict(schema='deadline-publication-metadata-v1', status='staged_metadata_not_uploaded',
                    rows_per_dataset=ROWS, cutoff=CUTOFF, selection_sha256=complete['selection_sha256'],
                    plan_sha256=files.hash(root / 'experiment_plan.json'), model_and_dataset_references=trained,
                    verified_input_files=files.inventory(), evaluation_scope='mandatory supplied eight; optional AIME25 separately gated',
                    source_copies_created=False, model_weight_copies_created=False, uploaded=False,
                    publication_destinations={'random_model': None, 'advanced_model': None,
                                              'random_dataset': None, 'advanced_dataset': None},
                    limitations=LIMITATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.publication-stage-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'package'
        staged.mkdir()
        for role in ('random', 'advanced'):
            for kind, text in (('models', model_card(role, trained[role], summary, plan)),
                               ('datasets', dataset_card(role, trained[role], plan))):
                folder = staged / kind / role
                folder.mkdir(parents=True)
                (folder / 'README.md').write_text(text, encoding='utf-8')
        for name, value in (('package_manifest.json', metadata), ('evaluation_summary.json', summary)):
            (staged / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        files.unchanged()
        require(not output.exists(), 'delivery output appeared during packaging')
        staged.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('deadline'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'hash model/data artifacts inside a compute allocation')
    metadata = build_package(args.root, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), status=metadata['status'], uploaded=False), indent=2))


if __name__ == '__main__':
    main()
