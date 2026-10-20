"""Tiny CPU-only recovery checks. Run in compute; no real weights are loaded."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from deadline.recovery.common import (VerifiedFiles, canonical_architecture, digest,
    json_digest, normalize_operational_config, require_compute, reserve_output, write_new_json)
from deadline.recovery.merge import (blend_fp32, matching_specs, merge,
    output_groups, validate_training_receipt, weight_map)


class RecoveryMergeTests(unittest.TestCase):
    def test_fixed_mix_and_endpoints_preserve_fp32_small_updates(self):
        base = torch.tensor([.1, 2., -4.], dtype=torch.float32)
        advanced = torch.tensor([.100008, 6., 4.], dtype=torch.float32)
        mixed = blend_fp32(base, advanced)
        self.assertEqual(mixed.dtype, torch.float32)
        torch.testing.assert_close(mixed, .75 * base + .25 * advanced, rtol=1e-6, atol=1e-8)
        self.assertGreater(float(mixed[0]), float(base[0]))
        torch.testing.assert_close(blend_fp32(base, advanced, 0), base, rtol=0, atol=0)
        torch.testing.assert_close(blend_fp32(base, advanced, 1), advanced, rtol=0, atol=0)

    def test_mixed_precision_inputs_are_computed_in_fp32(self):
        base = torch.tensor([1., 2.], dtype=torch.bfloat16)
        advanced = torch.tensor([3., 6.], dtype=torch.float32)
        torch.testing.assert_close(blend_fp32(base, advanced), torch.tensor([1.5, 3.]))

    def test_nonfinite_shape_integer_and_coefficient_rejected(self):
        good = torch.tensor([1.])
        for bad in (torch.tensor([float('nan')]), torch.tensor([float('inf')])):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                blend_fp32(bad, good)
        with self.assertRaisesRegex(ValueError, 'shape'):
            blend_fp32(good, torch.ones(2))
        with self.assertRaisesRegex(ValueError, 'nonfloating'):
            blend_fp32(torch.tensor([1]), torch.tensor([2]))
        for alpha in (-.1, 1.1, float('nan'), True):
            with self.assertRaisesRegex(ValueError, 'coefficient'):
                blend_fp32(good, good, alpha)

    def test_keys_shapes_and_advanced_precision_fail_closed(self):
        base = {'a': {'shape': [2, 2], 'dtype': 'BF16'}}
        advanced = {'a': {'shape': [2, 2], 'dtype': 'F32'}}
        matching_specs(base, advanced)
        for bad in ({'b': advanced['a']}, {'a': {'shape': [4], 'dtype': 'F32'}},
                    {'a': {'shape': [2, 2], 'dtype': 'BF16'}}):
            with self.assertRaises(ValueError):
                matching_specs(base, bad)

    def test_weight_index_must_match_actual_keys_and_shard_set(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_file({'a': torch.ones(2)}, str(folder / 'model.safetensors'))
            mapping, specs = weight_map(folder)
            self.assertEqual(mapping, {'a': 'model.safetensors'})
            self.assertEqual(specs['a'], {'shape': [2], 'dtype': 'F32'})
            write_new_json(folder / 'model.safetensors.index.json',
                           {'weight_map': {'wrong': 'model.safetensors'}})
            with self.assertRaisesRegex(ValueError, 'index'):
                weight_map(folder)
            (folder / 'model.safetensors.index.json').unlink()
            save_file({'b': torch.ones(1)}, str(folder / 'extra.safetensors'))
            with self.assertRaisesRegex(ValueError, 'extra or missing'):
                weight_map(folder)

    def test_bounded_shards_and_single_tensor_overflow(self):
        specs = {key: {'shape': [2], 'dtype': 'F32'} for key in ('b', 'a', 'c')}
        self.assertEqual(output_groups(specs, 16), [['a', 'b'], ['c']])
        with self.assertRaisesRegex(ValueError, 'single tensor'):
            output_groups(specs, 4)

    def test_output_and_receipts_refuse_any_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate'
            reserve_output(path)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                reserve_output(path)
            write_new_json(path / 'complete.json', {'complete': True})
            with self.assertRaises(FileExistsError):
                write_new_json(path / 'complete.json', {'complete': False})

    def test_hash_mismatch_and_input_change_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input'
            path.write_bytes(b'original')
            files = VerifiedFiles()
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                files.check(path, '0' * 64)
            files.check(path, digest(path))
            path.write_bytes(b'changed-longer')
            with self.assertRaisesRegex(ValueError, 'changed'):
                files.unchanged()

    @staticmethod
    def training_fixture():
        manifest = {'schema': 'deadline-full-sft-v1', 'recipe': {
            'max_steps': 1875, 'per_device_batch': 1, 'accumulation': 8, 'cutoff': 8192,
            'learning_rate': 2e-6, 'parameter_dtype': 'float32',
            'optimizer_moment_dtype': 'float32', 'compute_dtype': 'bfloat16_autocast',
            'full_parameter': True, 'seed': 42, 'data_seed': 42},
            'cache_metadata': {'dataset_role': 'advanced', 'rows': 15000,
                               'cutoff': 8192, 'truncated_rows': 0}}
        complete = {'binding_sha256': json_digest(manifest), 'steps': 1875,
            'unique_examples': 15000, 'example_presentations': 15000,
            'padding_presentations': 0, 'full_parameter': True,
            'final_model_verification': {'all_finite': True, 'parameter_dtype': 'float32',
                'tensors': 2, 'parameter_names_verified': 2,
                'files': {'model.safetensors': 'a' * 64}}}
        return manifest, complete

    def test_completion_must_bind_exact_advanced_recipe_and_consumption(self):
        manifest, complete = self.training_fixture()
        validate_training_receipt(manifest, complete)
        for key, value in (('steps', 1874), ('unique_examples', 14999),
                           ('example_presentations', 15001), ('padding_presentations', 1),
                           ('binding_sha256', 'x'), ('full_parameter', False)):
            bad = deepcopy(complete); bad[key] = value
            with self.assertRaises(ValueError):
                validate_training_receipt(manifest, bad)
        bad_manifest = deepcopy(manifest)
        bad_manifest['cache_metadata']['dataset_role'] = 'random'
        bad_complete = deepcopy(complete)
        bad_complete['binding_sha256'] = json_digest(bad_manifest)
        with self.assertRaisesRegex(ValueError, 'advanced-data'):
            validate_training_receipt(bad_manifest, bad_complete)

    def test_only_narrow_verified_operational_config_differences_are_normalized(self):
        base = {'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM'],
                'hidden_size': 2048, 'num_hidden_layers': 36, 'use_sliding_window': False,
                'bos_token_id': 151643, 'sliding_window': 32768, 'max_window_layers': 70}
        trained = {**base, 'bos_token_id': None, 'pad_token_id': 151643,
                   'sliding_window': None, 'layer_types': ['full_attention'] * 36}
        self.assertEqual(normalize_operational_config(base, [151643, 151645]),
                         normalize_operational_config(trained, [151643, 151645]))
        for changes in ({'pad_token_id': 999}, {'use_sliding_window': True},
                        {'layer_types': ['sliding_attention'] * 36}):
            with self.assertRaises(ValueError):
                normalize_operational_config({**trained, **changes}, [151643, 151645])
        self.assertNotEqual(canonical_architecture(base),
                            canonical_architecture({**base, 'hidden_size': 1024}))

    def test_compute_and_cpu_device_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'compute allocation'):
                require_compute(cpu_only=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                require_compute(cpu_only=True)

    def test_tiny_streaming_merge_writes_finite_checkpoint_without_training_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); base = root / 'base'; advanced = root / 'advanced'
            base.mkdir(); advanced.mkdir(); (root / 'licenses').mkdir()
            (root / 'experiment_plan.json').write_text('{}')
            source_license = Path(__file__).resolve().parents[1] / 'deadline/licenses/Qwen2.5-3B-Instruct.LICENSE'
            shutil.copyfile(source_license, root / 'licenses/Qwen2.5-3B-Instruct.LICENSE')
            config = {'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM']}
            (base / 'config.json').write_text(json.dumps(config))
            (base / 'tokenizer.json').write_text('{}')
            save_file({'a': torch.tensor([1., 2.], dtype=torch.bfloat16),
                       'b': torch.tensor([4.], dtype=torch.bfloat16)}, str(base / 'model.safetensors'))
            save_file({'a': torch.tensor([5., 6.]), 'b': torch.tensor([8.])},
                      str(advanced / 'model.safetensors'))
            finite = {'tensors': 2, 'files': {'model.safetensors': digest(advanced / 'model.safetensors')}}
            inputs = {'base': {'path': str(base), 'files': {}},
                      'advanced': {'path': str(advanced), 'files': {}}}
            output = root / 'new-candidate'
            with patch('deadline.recovery.merge.checked_inputs', return_value=(
                    VerifiedFiles(), {}, advanced, finite, {'fixture': True}, inputs)), \
                    patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}):
                receipt = merge(root, base, output)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    merge(root, base, output)
            self.assertTrue(receipt['complete'])
            self.assertEqual(receipt['output']['finite_fp32_tensors'], 2)
            shard = next(iter(receipt['output']['weight_files']))
            with safe_open(str(output / 'model' / shard), framework='pt', device='cpu') as tensors:
                torch.testing.assert_close(tensors.get_tensor('a'), torch.tensor([2., 3.]))
                torch.testing.assert_close(tensors.get_tensor('b'), torch.tensor([5.]))
            self.assertFalse((output / 'model/training_args.bin').exists())
            self.assertFalse((output / 'model/optimizer.pt').exists())
            self.assertEqual(read_json_local(output / 'model/config.json')['torch_dtype'], 'float32')


def read_json_local(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    unittest.main()
