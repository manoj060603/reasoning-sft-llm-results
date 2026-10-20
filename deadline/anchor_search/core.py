"""CPU preparation for a separately sealed validation-guided anchor search.

No scheduling, training, publication or benchmark selection occurs here. Old
plans and artifacts are read-only. Each failed/new attempt needs a unique path.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import re
import shutil

from deadline.recovery.common import (MODEL_SIDECARS, canonical_architecture,
    digest, json_digest, read_json, require, require_compute, reserve_output,
    write_new_json)
from deadline.recovery.merge import (PINNED_BASE_REVISION, blend_fp32,
    checked_inputs, matching_specs, output_groups, weight_map)
from deadline.recovery.screen import (_safe_file_map, audit_sources,
    canonical_rows, model_inputs, render_prompt, strict_ids, text_digest)
from deadline.publish_huggingface import NOTICE, LICENSE_SHA256
from reasoning.data import boxed


HERE = Path(__file__).resolve().parent
PLAN = HERE / 'plan.json'
BASE = Path('/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots') / PINNED_BASE_REVISION
SCHEMA = 'deadline-anchor-search-prepared-v1'
CANDIDATE_SCHEMA = 'deadline-anchor-candidate-merge-v1'
ROLES = ('original', 'incumbent', 'alpha0125', 'alpha0375')
SPLITS = ('search', 'confirmation')
OLD_IDS_SHA256 = 'af899b05afe502400cebb6c2197a7fd454f66114198733838ffd7913f9975d8e'


def code_hashes():
    return {path.name: digest(path) for path in sorted(HERE.glob('*.py'))}


def validate_plan(plan):
    require(plan.get('schema') == 'deadline-validation-anchor-search-plan-v1' and
            plan.get('alpha_advanced_candidates') == {'alpha0125': .125, 'alpha0375': .375} and
            plan.get('controls') == {'original': 0., 'incumbent': .25}, 'Unexpected fixed candidate grid')
    selection = plan.get('source_selection', {})
    require(all(selection.get(key) == value for key, value in {
        'heldout_rows': 1000, 'exclude_previously_viewed_screen_rows': 64,
        'previous_screen_ids_sha256': OLD_IDS_SHA256, 'seed': 20261021,
        'search_rows': 192, 'confirmation_rows': 192,
        'selection_occurs_before_new_generation': True,
        'ordering': 'ascending sha256 of 20261021:anchor-v2:<canonical prompt id>',
    }.items()), 'Unexpected source selection plan')
    require(all(plan.get('source_protocol', {}).get(key) == value for key, value in {
        'n': 8, 'temperature': .6, 'top_p': .95, 'seed': 20261021,
        'max_tokens': 4096, 'max_model_len': 8192, 'max_num_seqs': 128,
        'gpu_memory_utilization': .85, 'dtype': 'bfloat16', 'tensor_parallel_size': 1,
        'generation_config': 'vllm', 'top_k': -1, 'min_p': 0.,
        'repetition_penalty': 1., 'native_prompt_tokens': True,
    }.items()), 'Unexpected source generation protocol')
    return plan


def choose_splits(rows, tokenizer, old_ids, plan=None):
    """Choose fresh source IDs before any generation, exporting no teacher text."""
    plan = validate_plan(read_json(PLAN) if plan is None else plan)
    by_id = canonical_rows(rows, 1000)
    old_ids = strict_ids(old_ids, 64)
    require(set(old_ids) <= set(by_id), 'Previous screen IDs are outside the source heldout')
    candidates, eligible_before = [], 0
    for identifier, row in by_id.items():
        answer = boxed(row['output'])
        if answer is None:
            continue
        prompt, tokens = render_prompt(tokenizer, row)
        if len(tokens) > 2048:
            continue
        eligible_before += 1
        if identifier in old_ids:
            continue
        candidates.append(dict(id=identifier, prompt_tokens=len(tokens),
            prompt_sha256=text_digest(prompt), prompt_ids_sha256=json_digest(tokens),
            teacher_answer_sha256=text_digest(answer)))
    candidates.sort(key=lambda item: text_digest('20261021:anchor-v2:' + item['id']))
    require(len(candidates) >= 384, 'Fewer than 384 eligible previously unused source questions')
    splits = {'search': candidates[:192], 'confirmation': candidates[192:384]}
    return splits, dict(heldout_rows=len(by_id), eligible_before_exclusion=eligible_before,
        excluded_previous_screen_rows=len(old_ids), eligible_unused_rows=len(candidates))


def _merge_candidate(directory, role, alpha, base, advanced, base_map, specifications,
                     advanced_map, inputs, compatibility, plan_sha, code, files):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized for CPU merging')
    directory = reserve_output(directory)
    model = directory / 'model'
    model.mkdir()
    groups = output_groups(specifications)
    require(len(groups) == 6 and len(specifications) == 434, 'Expected six shards and 434 Qwen tensors')
    binding = dict(schema=CANDIDATE_SCHEMA, role=role, alpha_advanced=alpha,
        alpha_base=1-alpha, inputs=inputs, compatibility=compatibility,
        plan_sha256=plan_sha, code_sha256=code, output_path=str(model),
        compute=require_compute(cpu_only=True), posthoc_validation_search=True,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'safetensors', 'transformers')})
    write_new_json(directory / 'merge_inputs.json', binding)
    result_map, weights = {}, {}
    for number, keys in enumerate(groups, 1):
        filename = f'model-{number:05d}-of-00006.safetensors'
        merged = {}
        for key in keys:
            with safe_open(str(base / base_map[key]), framework='pt', device='cpu') as first, \
                    safe_open(str(advanced / advanced_map[key]), framework='pt', device='cpu') as second:
                original, trained = first.get_tensor(key), second.get_tensor(key)
                merged[key] = blend_fp32(original, trained, alpha=alpha)
                del original, trained
            result_map[key] = filename
        save_file(merged, str(model / filename), metadata={'format': 'pt'})
        del merged
        gc.collect()
        weights[filename] = files.check(model / filename)
    total_bytes = sum(math.prod(value['shape']) * 4 for value in specifications.values())
    write_new_json(model / 'model.safetensors.index.json', {
        'metadata': {'total_size': total_bytes}, 'weight_map': result_map})
    for name in sorted(MODEL_SIDECARS - {'config.json'}):
        if (base / name).is_file():
            shutil.copyfile(base / name, model / name)
    config = read_json(base / 'config.json')
    canonical_architecture(config)
    config.pop('dtype', None)
    config['torch_dtype'] = 'float32'
    write_new_json(model / 'config.json', config)
    license_path = HERE.parent / 'licenses/Qwen2.5-3B-Instruct.LICENSE'
    files.check(license_path, LICENSE_SHA256)
    shutil.copyfile(license_path, model / 'LICENSE')
    (model / 'NOTICE').write_text(NOTICE + '\nBuilt with Qwen.\n'
        f'Modified by FP32 base anchoring: {(1-alpha)*100:g}% original Qwen2.5-3B-Instruct '
        f'and {alpha*100:g}% advanced full-SFT weights.\n'
        'Exploratory validation-guided follow-up after prior benchmark observations; '
        'no additional SFT, novelty or benchmark superiority is established.\n', encoding='utf-8')
    count = 0
    for name in sorted(weights):
        with safe_open(str(model / name), framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                value = tensors.get_tensor(key)
                require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                        'Nonfinite or non-FP32 output tensor: ' + key)
                count += 1
                del value
    require(count == 434, 'Final finite tensor count differs')
    output_files = {path.name: files.check(path) for path in sorted(model.iterdir()) if path.is_file()}
    files.unchanged()
    receipt = dict(binding, complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        merge_inputs_sha256=files.check(directory / 'merge_inputs.json'), output=dict(
            path=str(model), files=output_files, weight_files=weights, all_finite=True,
            finite_fp32_tensors=count, parameter_dtype='float32', total_tensor_bytes=total_bytes))
    write_new_json(directory / 'complete.json', receipt)
    return receipt


def prepare(output, root='deadline', base=BASE):
    compute = require_compute(cpu_only=True)
    output, root, base = (Path(value).resolve() for value in (output, root, base))
    require(not output.exists(), 'Output already exists; use a unique new search directory')
    plan = validate_plan(read_json(PLAN))
    code, plan_sha = code_hashes(), digest(PLAN)
    output = reserve_output(output)
    shutil.copyfile(PLAN, output / 'plan.json')
    files, primary_plan, advanced, finite, compatibility, parents = checked_inputs(root, base)
    files.check(PLAN, plan_sha)
    files.check(output / 'plan.json', plan_sha)
    require(finite['tensors'] == 434, 'Unexpected trained tensor count')
    incumbent = root / 'recovery-runs/base-anchor-75-25-20261020'
    incumbent_receipt = incumbent / 'complete.json'
    incumbent_sha = files.check(incumbent_receipt)
    old_merge, old_models = model_inputs(incumbent_receipt, verify_roles=())
    require(old_merge['inputs'] == parents, 'Incumbent has different exact original/advanced parents')
    require(Path(old_models['base']['path']) == base and
            Path(old_models['advanced']['path']) == advanced, 'Incumbent parent paths differ')
    for name, checksum in old_models['recovery']['files'].items():
        files.check(Path(old_models['recovery']['path']) / name, checksum)
    heldout, source_audit = audit_sources(root / 'artifacts', old_models)
    for name, checksum in source_audit['files'].items():
        files.check(Path(source_audit['artifacts']) / name, checksum)
    old_ids_path = root / 'operations/recovery-screen-ids-20261020.json'
    files.check(old_ids_path, OLD_IDS_SHA256)
    old_ids = strict_ids(read_json(old_ids_path), 64)
    old_prepared_path = incumbent / 'development/prepared.json'
    old_prepared_sha = files.check(old_prepared_path)
    old_prepared = read_json(old_prepared_path)
    require(old_prepared.get('selected_ids_sha256') == OLD_IDS_SHA256 and
            old_prepared.get('merge_receipt_sha256') == incumbent_sha and
            old_prepared.get('source_audit', {}).get('files', {}).get('heldout.json') ==
            source_audit['files']['heldout.json'], 'Previous development exclusion/source seal differs')
    files.check(incumbent / 'development/selected_ids.json', OLD_IDS_SHA256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    splits, counts = choose_splits(heldout, tokenizer, old_ids, plan)
    selected_ids_sha, split_sha = {}, {}
    for split, metadata in splits.items():
        write_new_json(output / (split + '_ids.json'), [item['id'] for item in metadata])
        write_new_json(output / (split + '.json'), metadata)
        selected_ids_sha[split] = files.check(output / (split + '_ids.json'))
        split_sha[split] = files.check(output / (split + '.json'))
    base_map, base_specs = weight_map(base)
    advanced_map, specs = weight_map(advanced)
    matching_specs(base_specs, specs)
    require(set(finite['files']) == set(advanced_map.values()) and len(specs) == 434,
            'Advanced shard/tensor inventory differs from actual training receipt')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA')
    torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))))
    models = {'original': old_models['base'], 'incumbent': old_models['recovery']}
    receipts = {}
    for role, alpha in plan['alpha_advanced_candidates'].items():
        folder = output / 'candidates' / role
        receipt = _merge_candidate(folder, role, alpha, base, advanced, base_map,
            specs, advanced_map, parents, compatibility, plan_sha, code, files)
        receipts[role] = dict(path=str(folder / 'complete.json'), sha256=files.check(folder / 'complete.json'))
        models[role] = {'path': receipt['output']['path'], 'files': receipt['output']['files']}
    grader_root = HERE.parents[1] / 'eval/utils'
    grader = {name: files.check(grader_root / name) for name in
              ('grader.py', 'parser.py', 'math_normalization.py')}
    require(code == code_hashes() and digest(PLAN) == plan_sha, 'Search source/plan changed during preparation')
    files.unchanged()
    prepared = dict(schema=SCHEMA, phase='prepare', complete=True, compute=compute,
        created_at_utc=datetime.now(timezone.utc).isoformat(), output=str(output), root=str(root),
        plan_sha256=plan_sha, code_sha256=code, models=models, parents=parents,
        splits=splits, selection_counts=counts, selected_ids_sha256=selected_ids_sha,
        split_metadata_sha256=split_sha,
        selected_ids_seal=json_digest({name: [item['id'] for item in rows] for name, rows in splits.items()}),
        source_audit=source_audit, artifacts_path=source_audit['artifacts'],
        old_screen_ids={'path': str(old_ids_path), 'sha256': OLD_IDS_SHA256},
        old_screen_ids_sha256=OLD_IDS_SHA256,
        old_screen_prepared={'path': str(old_prepared_path), 'sha256': old_prepared_sha},
        primary_plan_sha256=files.check(root / 'experiment_plan.json'),
        recovery_plan_sha256=files.check(HERE.parent / 'recovery/plan.json'),
        incumbent_merge_receipt={'path': str(incumbent_receipt), 'sha256': incumbent_sha},
        candidate_receipts=receipts, grader_sha256=grader)
    write_new_json(output / 'prepared.json', prepared)
    return prepared


def load_prepared(output, verify_models=False):
    """Validate immutable small seals/inventories; hash role weights on request."""
    output = Path(output).resolve()
    value = read_json(output / 'prepared.json')
    plan = validate_plan(read_json(PLAN))
    require(value.get('schema') == SCHEMA and value.get('phase') == 'prepare' and
            value.get('complete') is True and value.get('output') == str(output) and
            value.get('plan_sha256') == digest(PLAN) == digest(output / 'plan.json') and
            value.get('code_sha256') == code_hashes(), 'Prepared search plan/code/output identity changed')
    old = value['old_screen_ids']
    require(old.get('sha256') == OLD_IDS_SHA256 == value.get('old_screen_ids_sha256') and
            digest(old['path']) == OLD_IDS_SHA256, 'Previously viewed screen ID seal changed')
    old_ids = set(strict_ids(read_json(old['path']), 64))
    splits = value.get('splits', {})
    require(set(splits) == set(SPLITS), 'Both fixed source splits are required')
    ids = {}
    for split in SPLITS:
        require(digest(output / (split + '.json')) == value['split_metadata_sha256'][split] and
                read_json(output / (split + '.json')) == splits[split] and
                digest(output / (split + '_ids.json')) == value['selected_ids_sha256'][split],
                'Source split files changed')
        ids[split] = strict_ids(read_json(output / (split + '_ids.json')), 192)
        require([item.get('id') for item in splits[split]] == ids[split] and
                not set(ids[split]).intersection(old_ids), 'Source split IDs changed or overlap old screen')
        for item in splits[split]:
            require(set(item) == {'id', 'prompt_tokens', 'prompt_sha256', 'prompt_ids_sha256', 'teacher_answer_sha256'} and
                    type(item['prompt_tokens']) is int and 0 < item['prompt_tokens'] <= 2048 and
                    all(isinstance(item[key], str) and re.fullmatch('[0-9a-f]{64}', item[key]) for key in
                        ('prompt_sha256', 'prompt_ids_sha256', 'teacher_answer_sha256')),
                    'Malformed or answer-bearing source metadata')
    require(not set(ids['search']).intersection(ids['confirmation']) and
            json_digest(ids) == value.get('selected_ids_seal'), 'Source splits overlap or selection seal changed')
    root = Path(value['root'])
    require(digest(root / 'experiment_plan.json') == value['primary_plan_sha256'] and
            digest(HERE.parent / 'recovery/plan.json') == value['recovery_plan_sha256'], 'Delivered plans changed')
    for name in ('old_screen_prepared', 'incumbent_merge_receipt'):
        record = value[name]
        require(digest(record['path']) == record['sha256'], 'Prior source/model receipt changed')
    _, old_models = model_inputs(value['incumbent_merge_receipt']['path'], verify_roles=())
    models = value.get('models', {})
    require(set(models) == set(ROLES) and models['original'] == old_models['base'] and
            models['incumbent'] == old_models['recovery'], 'Original/incumbent models changed')
    require(set(value['candidate_receipts']) == set(plan['alpha_advanced_candidates']), 'Missing new candidate receipt')
    for role, alpha in plan['alpha_advanced_candidates'].items():
        record = value['candidate_receipts'][role]
        require(digest(record['path']) == record['sha256'], 'Candidate completion receipt changed')
        receipt = read_json(record['path'])
        result = receipt.get('output', {})
        require(receipt.get('schema') == CANDIDATE_SCHEMA and receipt.get('complete') is True and
                receipt.get('role') == role and receipt.get('alpha_advanced') == alpha and
                receipt.get('alpha_base') == 1-alpha and receipt.get('plan_sha256') == value['plan_sha256'] and
                receipt.get('code_sha256') == value['code_sha256'] and receipt.get('inputs') == value['parents'] and
                result.get('all_finite') is True and result.get('parameter_dtype') == 'float32' and
                result.get('finite_fp32_tensors') == 434 and
                models[role] == {'path': result.get('path'), 'files': result.get('files')} and
                digest(Path(record['path']).parent / 'merge_inputs.json') == receipt['merge_inputs_sha256'],
                'Candidate alpha/parent/finite/output binding differs')
        require(set(result.get('weight_files', {})) == {
            f'model-{number:05d}-of-00006.safetensors' for number in range(1, 7)} and
            all(result['files'].get(name) == checksum for name, checksum in result['weight_files'].items()),
            'Candidate six-shard identity differs')
    roles_to_hash = set(ROLES) if verify_models is True else (
        set() if verify_models is False else {verify_models} if isinstance(verify_models, str) else set(verify_models))
    require(roles_to_hash <= set(ROLES), 'Unknown model role requested for weight verification')
    for role, model in models.items():
        _safe_file_map(model['path'], model['files'], verify=role in roles_to_hash)
    grader_root = HERE.parents[1] / 'eval/utils'
    require(set(value['grader_sha256']) == {'grader.py', 'parser.py', 'math_normalization.py'} and
            all(digest(grader_root / name) == checksum for name, checksum in value['grader_sha256'].items()),
            'Source teacher-agreement grader changed')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('deadline'))
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    prepare(args.output, args.root, args.base)


if __name__ == '__main__':
    main()
