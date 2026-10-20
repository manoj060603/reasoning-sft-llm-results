"""One fixed post-hoc FP32 base anchor; CPU compute allocation only.

Run with --root deadline --base <pinned local snapshot> --output <new folder>.
No alpha argument, optimizer state, network access, scheduling or primary writes.
Only ordinary Qwen2 safetensors layouts with identical serialized keys/shapes
are supported. A failed attempt is retained; retry requires another unique path.
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

from .common import (MODEL_SIDECARS, VerifiedFiles, canonical_architecture, digest,
                     json_digest, read_json, require, require_compute,
                     reserve_output, verify_model_compatibility, write_new_json)


ALPHA_ADVANCED = 0.25
SHARD_BYTES = 2 * 1024 ** 3
PINNED_BASE_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
PINNED_BASE_IDENTITY = 'd9fe5b6890d4e60435ce5ae081c61d09bf6d3ad49ed4bbc06359cd81f360f7ee'
HERE = Path(__file__).resolve().parent


def blend_fp32(base, advanced, alpha=ALPHA_ADVANCED):
    """Pure tensor arithmetic helper; only the fixed coefficient is exposed by CLI."""
    import torch
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 <= alpha <= 1,
            'invalid interpolation coefficient')
    require(base.shape == advanced.shape, 'tensor shape mismatch')
    require(base.is_floating_point() and advanced.is_floating_point(), 'nonfloating tensor unsupported')
    require(base.device.type == advanced.device.type == 'cpu', 'merge tensors must be on CPU')
    require(bool(torch.isfinite(base).all()) and bool(torch.isfinite(advanced).all()),
            'input tensor contains nonfinite values')
    result = base.to(torch.float32).mul(1. - alpha).add(advanced.to(torch.float32), alpha=alpha)
    require(result.dtype == torch.float32 and bool(torch.isfinite(result).all()),
            'merged tensor contains nonfinite FP32 values')
    return result.contiguous()


def validate_training_receipt(manifest, complete):
    require(manifest.get('schema') == 'deadline-full-sft-v1', 'unsupported advanced training manifest')
    require(complete.get('binding_sha256') == json_digest(manifest), 'training completion binding differs')
    require(all(type(complete.get(key)) is int for key in
                ('steps', 'unique_examples', 'example_presentations', 'padding_presentations')),
            'training consumption counts must be integers')
    require(complete.get('steps') == 1875 and complete.get('unique_examples') == 15000 and
            complete.get('example_presentations') == 15000 and complete.get('padding_presentations') == 0 and
            complete.get('full_parameter') is True, 'advanced full-training consumption is incomplete')
    recipe = manifest.get('recipe', {})
    require(all(recipe.get(key) == value for key, value in {
        'max_steps': 1875, 'per_device_batch': 1, 'accumulation': 8, 'cutoff': 8192,
        'learning_rate': 2e-6, 'parameter_dtype': 'float32',
        'optimizer_moment_dtype': 'float32', 'compute_dtype': 'bfloat16_autocast',
        'full_parameter': True, 'seed': 42, 'data_seed': 42,
    }.items()), 'advanced recipe is not the fixed primary full-SFT recipe')
    meta = manifest.get('cache_metadata', {})
    require(meta.get('dataset_role') == 'advanced' and meta.get('rows') == 15000 and
            meta.get('cutoff') == 8192 and meta.get('truncated_rows') == 0,
            'training manifest is not the complete advanced-data arm')
    finite = complete.get('final_model_verification', {})
    require(finite.get('all_finite') is True and finite.get('parameter_dtype') == 'float32' and
            type(finite.get('tensors')) is int and finite['tensors'] > 0 and
            finite.get('parameter_names_verified') == finite['tensors'] and
            isinstance(finite.get('files'), dict) and finite['files'],
            'complete finite FP32 advanced-model receipt required')
    return finite


def weight_map(directory):
    """Return exact on-disk serialized keys; never infer tied/omitted weights."""
    from safetensors import safe_open
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        declared = read_json(index).get('weight_map')
        require(isinstance(declared, dict) and declared, 'empty safetensors index')
        names = set(declared.values())
    else:
        declared = None
        names = {'model.safetensors'}
    require(all(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', name)
                for name in names), 'unsafe or unsupported safetensors shard path')
    require({p.name for p in directory.glob('*.safetensors')} == names, 'extra or missing weight shards')
    found, specifications = {}, {}
    for name in sorted(names):
        with safe_open(str(directory / name), framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                require(key not in found, 'tensor appears in more than one shard')
                value = tensors.get_slice(key)
                found[key] = name
                specifications[key] = {'shape': list(value.get_shape()), 'dtype': value.get_dtype()}
    require(found and (declared is None or found == declared), 'index does not match actual tensor keys')
    return found, specifications


def matching_specs(base, advanced):
    require(set(base) == set(advanced), 'base/advanced serialized tensor keys differ')
    for key in sorted(base):
        require(base[key]['shape'] == advanced[key]['shape'], f'tensor shape mismatch: {key}')
        require(base[key]['dtype'] in ('BF16', 'F32') and advanced[key]['dtype'] == 'F32',
                f'unsupported input tensor precision: {key}')


def output_groups(specifications, maximum_bytes=SHARD_BYTES):
    require(type(maximum_bytes) is int and maximum_bytes > 0, 'invalid shard capacity')
    groups, group, size = [], [], 0
    for key in sorted(specifications):
        elements = math.prod(specifications[key]['shape'])
        require(elements > 0, 'empty tensors are unsupported')
        nbytes = elements * 4
        require(nbytes <= maximum_bytes, 'single tensor exceeds bounded output shard capacity')
        if group and size + nbytes > maximum_bytes:
            groups.append(group)
            group, size = [], 0
        group.append(key)
        size += nbytes
    if group:
        groups.append(group)
    require(bool(groups), 'no tensors to merge')
    return groups


def checked_inputs(root, base):
    files = VerifiedFiles()
    plan_path = root / 'experiment_plan.json'
    plan = read_json(plan_path)
    files.check(plan_path)
    require(plan.get('base_revision') == PINNED_BASE_REVISION, 'primary pinned base revision differs')
    identity_path = root / 'artifacts/model_identity.json'
    identity = read_json(identity_path)
    files.check(identity_path)
    require(set(identity) == {'path', 'files', 'model_type', 'identity_sha256'}, 'base identity schema differs')
    payload = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    require(identity['identity_sha256'] == json_digest(payload) == PINNED_BASE_IDENTITY and
            identity['model_type'] == 'qwen2' and Path(identity['path']).resolve() == base,
            'original base is not the immutable primary pinned model')
    prepared = read_json(root / 'artifacts/preparation_complete.json')
    files.check(root / 'artifacts/preparation_complete.json')
    require(prepared.get('model_revision') == PINNED_BASE_REVISION, 'preparation base revision differs')
    base_hashes = {}
    for name, info in identity['files'].items():
        require(Path(name).name == name and isinstance(info, dict), 'unsafe base identity entry')
        base_hashes[name] = files.check(base / name, info['sha256'])
        require((base / name).stat().st_size == info['bytes'], 'base identity size differs')
    advanced = (root / 'runs/advanced/model').resolve()
    manifest_path, receipt_path = root / 'runs/advanced/manifest.json', root / 'runs/advanced/complete.json'
    manifest, complete = read_json(manifest_path), read_json(receipt_path)
    manifest_sha, complete_sha = files.check(manifest_path), files.check(receipt_path)
    finite = validate_training_receipt(manifest, complete)
    require(Path(complete.get('model', '')).resolve() == advanced and
            Path(manifest.get('base', '')).resolve() == base and
            manifest['cache_metadata'].get('base_model_sha256') == PINNED_BASE_IDENTITY,
            'advanced receipt paths or original-base identity differ')
    for name, checksum in manifest.get('base_identity', {}).items():
        require(Path(name).name == name, 'unsafe training base identity path')
        files.check(base / name, checksum)
    require(manifest.get('base_identity'), 'training original-base file identity missing')
    advanced_hashes = {}
    for name, checksum in finite['files'].items():
        require(Path(name).name == name and name.endswith('.safetensors'), 'unsafe trained weight filename')
        advanced_hashes[name] = files.check(advanced / name, checksum)
    if (advanced / 'model.safetensors.index.json').is_file():
        advanced_hashes['model.safetensors.index.json'] = files.check(
            advanced / 'model.safetensors.index.json', finite.get('model_index_sha256', 'missing'))
    for directory, hashes in ((base, base_hashes), (advanced, advanced_hashes)):
        for name in sorted(MODEL_SIDECARS):
            if (directory / name).is_file():
                hashes[name] = files.check(directory / name, hashes.get(name))
        require({'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(hashes),
                'complete local model/tokenizer sidecars are required')
    compatibility = verify_model_compatibility(base, advanced)
    return files, plan, advanced, finite, compatibility, {
        'base': {'path': str(base), 'files': base_hashes, 'identity_sha256': identity['identity_sha256']},
        'advanced': {'path': str(advanced), 'files': advanced_hashes,
                     'training_manifest_sha256': manifest_sha, 'training_receipt_sha256': complete_sha},
    }


def merge(root, base, output):
    compute = require_compute(cpu_only=True)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized for CPU merge')
    torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))))
    root, base = Path(root).resolve(), Path(base).resolve()
    require(not Path(output).exists(), 'output already exists; use a unique new folder')
    files, primary_plan, advanced, finite, compatibility, inputs = checked_inputs(root, base)
    base_map, base_specs = weight_map(base)
    advanced_map, advanced_specs = weight_map(advanced)
    matching_specs(base_specs, advanced_specs)
    require(set(finite['files']) == set(advanced_map.values()) and
            finite['tensors'] == len(advanced_specs), 'advanced tensor count or shard set differs from receipt')
    groups = output_groups(advanced_specs)
    recovery_plan = read_json(HERE / 'plan.json')
    require(recovery_plan.get('alpha_advanced') == ALPHA_ADVANCED and
            recovery_plan.get('alpha_base') == 1 - ALPHA_ADVANCED and
            recovery_plan.get('alpha_sweep') is False, 'fixed recovery plan differs')
    output = reserve_output(output)
    model = output / 'model'
    model.mkdir()
    code = {str(path.relative_to(HERE)): files.check(path) for path in sorted(HERE.glob('*.py'))}
    binding = {'schema': 'deadline-base-anchor-merge-v1', 'alpha_advanced': ALPHA_ADVANCED,
               'alpha_base': 1 - ALPHA_ADVANCED, 'inputs': inputs, 'compatibility': compatibility,
               'primary_plan_sha256': files.check(root / 'experiment_plan.json'),
               'recovery_plan_sha256': files.check(HERE / 'plan.json'), 'code_sha256': code,
               'compute': compute, 'output_path': str(model),
               'posthoc_after_primary_benchmark_observations': True,
               'versions': {key: importlib.metadata.version(key) for key in ('torch', 'safetensors', 'transformers')}}
    write_new_json(output / 'merge_inputs.json', binding)
    write_new_json(output / 'recovery_plan.json', recovery_plan)
    result_map, weights = {}, {}
    for number, keys in enumerate(groups, 1):
        filename = f'model-{number:05d}-of-{len(groups):05d}.safetensors'
        merged = {}
        for key in keys:
            # Reopen per tensor so processed source mappings can be released.
            # Do not retain 6GB+12GB source mappings alongside output buffers.
            with safe_open(str(base / base_map[key]), framework='pt', device='cpu') as first, \
                    safe_open(str(advanced / advanced_map[key]), framework='pt', device='cpu') as second:
                original, trained = first.get_tensor(key), second.get_tensor(key)
                merged[key] = blend_fp32(original, trained)
                del original, trained
            result_map[key] = filename
        save_file(merged, str(model / filename), metadata={'format': 'pt'})
        del merged
        gc.collect()
        weights[filename] = digest(model / filename)
    total_bytes = sum(math.prod(value['shape']) * 4 for value in advanced_specs.values())
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
    license_path = root / 'licenses/Qwen2.5-3B-Instruct.LICENSE'
    require(license_path.is_file(), 'pinned Qwen Research License is required')
    files.check(license_path, 'ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9')
    shutil.copyfile(license_path, model / 'LICENSE')
    (model / 'NOTICE').write_text('Built with Qwen. Original Qwen2.5-3B-Instruct by the Qwen Team.\n'
        'Modified by fixed FP32 interpolation: 75% original base and 25% advanced full-SFT checkpoint.\n'
        'Exploratory post-hoc candidate; no benchmark superiority is established.\n', encoding='utf-8')
    count = 0
    for name in sorted(weights):
        with safe_open(str(model / name), framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                tensor = tensors.get_tensor(key)
                require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                        f'output verification failed: {key}')
                count += 1
                del tensor
    require(count == len(advanced_specs), 'output finite tensor count differs')
    files.unchanged()
    output_files = {path.name: digest(path) for path in sorted(model.iterdir()) if path.is_file()}
    receipt = {**binding, 'complete': True, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'output': {'path': str(model), 'files': output_files, 'weight_files': weights,
                          'finite_fp32_tensors': count, 'all_finite': True,
                          'parameter_dtype': 'float32', 'total_tensor_bytes': total_bytes},
               'merge_inputs_sha256': digest(output / 'merge_inputs.json')}
    write_new_json(output / 'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('deadline'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge(args.root, args.base, args.output)
    print('Recovery merge complete; this is not a training, evaluation or publication result.', flush=True)


if __name__ == '__main__':
    main()
