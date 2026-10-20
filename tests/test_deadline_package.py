"""Synthetic complete receipts test packaging gates; no real model or scores."""
import json
from pathlib import Path
import tempfile
import unittest

from deadline.audit_results import OFFICIAL, PROTOCOL, REQUIRED_SETTINGS
from deadline.compatible_coverage import COLUMNS, content_sha256
from deadline.package_deliverables import ROWS, SOURCE_SHA, build_package
from deadline.train_full import digest, fixed_order, json_digest
from deadline.transfer_probe import model_identity
from reasoning.data import alpaca, prompt_id


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def fixture(root):
    """Attest small dummy weight files and 15K tiny source records in a temp dir."""
    base = root / 'base'
    base.mkdir(parents=True)
    (base / 'model.safetensors').write_bytes(b'synthetic original weights')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        write(base / name, {'model_type': 'qwen2'} if name == 'config.json' else {})
    identity = model_identity(base)
    base_sha = identity['identity_sha256']
    base_files = {name: info['sha256'] for name, info in identity['files'].items()}
    plan = dict(method='Cross-fitted Transfer Gain with Coverage', base_revision='base-pinned',
                source_revision='source-pinned', selection_seed=20261019,
                selection={'cutoff': 8192}, final_training={'cutoff': 8192, 'updates': 1875, 'learning_rate': 2e-6})
    write(root / 'experiment_plan.json', plan)
    artifacts = root / 'artifacts'
    write(artifacts / 'model_identity.json', identity)
    write(artifacts / 'preparation_complete.json', dict(source_revision='source-pinned',
                model_revision='base-pinned', source_sha256=SOURCE_SHA))
    pool = [dict(id=prompt_id(f'Question {i}'), instruction=f'Question {i}', input='',
                 output=f'Original \\boxed{{{i}}}', system='Please reason.') for i in range(ROWS)]
    ids = [row['id'] for row in pool]
    write(artifacts / 'pool.json', pool)
    records = alpaca(pool)
    selection = dict(size=ROWS, cutoff=8192, seed=20261019, strategy='transfer_gain',
        source_pool_sha256=digest(artifacts / 'pool.json'),
        datasets={role: {'selected_ids': ids, 'exported_rows_sha256': content_sha256(records)}
                  for role in ('random', 'advanced')})
    selection['selection_sha256'] = content_sha256(selection)
    write(artifacts / 'selected/selection_manifest.json', selection)
    write(artifacts / 'selected/dataset_info.json', {r: {'columns': COLUMNS} for r in ('random', 'advanced')})
    recipe = dict(max_steps=1875, per_device_batch=1, accumulation=8, cutoff=8192,
        learning_rate=2e-6, lr_scheduler_type='cosine', warmup_ratio=.03, weight_decay=.01,
        max_grad_norm=1., parameter_dtype='float32', optimizer_moment_dtype='float32',
        compute_dtype='bfloat16_autocast', full_parameter=True, seed=42, data_seed=42)
    order = fixed_order(ROWS, 42)
    model_folders = {'original': base}
    for role in ('random', 'advanced'):
        data, run = artifacts / 'training' / role, root / 'runs' / role
        write(artifacts / 'selected' / (role + '.json'), records)
        write(data / 'dataset.json', records)
        write(data / 'selected_ids.json', ids)
        # Actual token correctness was upstream's responsibility; packaging
        # binds those already-validated cache bytes rather than retokenizing.
        (data / 'training.jsonl').write_text('{"synthetic_cache_fixture":true}\n')
        meta = dict(dataset_role=role, rows=ROWS, cutoff=8192, base_model_sha256=base_sha,
            unique_rows_dropped=0, target_rewriting=False, truncated_rows=3 if role == 'random' else 0,
            truncation_policy='prefix_truncate' if role == 'random' else 'none',
            cache_sha256=digest(data / 'training.jsonl'), dataset_sha256=digest(data / 'dataset.json'),
            selected_ids_sha256=digest(data / 'selected_ids.json'))
        write(data / 'cache_meta.json', meta)
        manifest = dict(recipe=recipe, cache_metadata=meta, base=str(base.resolve()), base_identity=base_files,
            order_sha256=json_digest(order), cache={'cache_sha256': meta['cache_sha256'],
            'cache_meta_sha256': digest(data / 'cache_meta.json')})
        write(run / 'manifest.json', manifest)
        binding = json_digest(manifest)
        model = run / 'model'
        model.mkdir()
        (model / 'model.safetensors').write_bytes(('synthetic ' + role + ' weights').encode())
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            write(model / name, {})
        verification = dict(all_finite=True, parameter_dtype='float32', parameter_names_verified=1,
            files={'model.safetensors': digest(model / 'model.safetensors')}, model_index_sha256=None)
        write(run / 'complete.json', dict(binding_sha256=binding, steps=1875, full_parameter=True,
            unique_examples=ROWS, example_presentations=ROWS, padding_presentations=0,
            training_loss=.25, model=str(model.resolve()), final_model_verification=verification))
        write(run / 'cache_audit.json', dict(cache_sha256=meta['cache_sha256'],
            truncated_rows=meta['truncated_rows'], supervised_eos_missing_rows=2 if role == 'random' else 0))
        write(run / 'consumption_final.json', dict(binding_sha256=binding, completed_updates=1875,
            consumed_indices=order, order_sha256=json_digest(order), padding_presentations=0))
        (run / 'finite_updates.jsonl').write_text(''.join(json.dumps(dict(step=step,
            gradient_norm=1., actual_presentations=step * 8)) + '\n' for step in range(1, 1876)))
        model_folders[role] = model
    probe_artifacts, score_artifacts = {}, {}
    for name in ('probeA', 'probeB'):
        folder = root / 'runs' / name
        write(folder / 'probe_receipt.json', {'complete': True})
        (folder / 'adapter').mkdir()
        (folder / 'adapter/adapter_model.safetensors').write_bytes(b'synthetic adapter')
        probe_artifacts[name] = dict(receipt_sha256=digest(folder / 'probe_receipt.json'),
            checkpoint_sha256=digest(folder / 'adapter/adapter_model.safetensors'))
    for fold in ('fold0', 'fold1'):
        folder = root / 'runs' / ('scores_' + fold)
        write(folder / 'score_receipt.json', {'complete': True})
        write(folder / 'run_manifest.json', {})
        (folder / 'scores.jsonl').write_text('{"synthetic_score_fixture":true}\n')
        score_artifacts[fold] = dict(receipt_sha256=digest(folder / 'score_receipt.json'),
            manifest_sha256=digest(folder / 'run_manifest.json'), scores_sha256=digest(folder / 'scores.jsonl'))
    selected_files = [path for path in (artifacts / 'selected').iterdir() if path.is_file()]
    advanced_files = [path for path in (artifacts / 'training/advanced').iterdir() if path.is_file()]
    write(artifacts / 'selection_complete.json', dict(complete=True,
        schema='deadline-crossfit-selection-complete-v1', rows_per_arm=ROWS, cutoff=8192,
        random_control_ids_unchanged=True, advanced_truncated_rows=0, base_model_sha256=base_sha,
        selection_sha256=selection['selection_sha256'], probe_artifacts=probe_artifacts,
        score_artifacts=score_artifacts, files_sha256={str(path.relative_to(artifacts)): digest(path)
                                                   for path in selected_files + advanced_files}))
    for role, model in model_folders.items():
        protocol = {key: 0 for key in PROTOCOL}
        protocol.update(REQUIRED_SETTINGS)
        protocol.update(model=str(model.resolve()), max_model_len=32768,
            identity={p.name: digest(p) for p in model.iterdir() if p.is_file()},
            problem_counts=dict(OFFICIAL), problem_ids={b: [f'{b}-{i}' for i in range(n)] for b, n in OFFICIAL.items()},
            data={b: 'same frozen benchmark ' + b for b in OFFICIAL})
        write(root / 'results' / role / 'manifest.json', protocol)
        write(root / 'results' / role / 'metrics.json', {b: dict(problems=n, samples_per_problem=8,
            pass_at_1=.5, pass_at_8=1., truncated_samples=0) for b, n in OFFICIAL.items()})


class PublicationPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'deadline'
        fixture(self.root)
        self.output = Path(self.temp.name) / 'publication'

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_eight_stages_cards_only_without_optional_aime25(self):
        before = digest(self.root / 'artifacts/training/random/dataset.json')
        metadata = build_package(self.root, self.output)
        self.assertFalse(metadata['uploaded'])
        self.assertFalse(metadata['model_weight_copies_created'])
        self.assertFalse(list(self.output.rglob('*.safetensors')))
        self.assertFalse(list(self.output.rglob('dataset.json')))
        summary = json.loads((self.output / 'evaluation_summary.json').read_text())
        self.assertTrue(summary['matched_official_eight'])
        self.assertFalse(summary['optional_aime25_matched'])
        card = (self.output / 'models/random/README.md').read_text()
        self.assertIn('3 truncated rows; 2 rows lack', card)
        self.assertIn('aggregate receipt only', card)
        self.assertIn('Qwen Research License', card)
        self.assertIn('not a literal uniform draw', card)
        self.assertEqual(before, digest(self.root / 'artifacts/training/random/dataset.json'))

    def test_missing_final_model_fails_before_output_is_created(self):
        (self.root / 'runs/random/model/model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'weights/shards are incomplete'):
            build_package(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_mandatory_result_fails(self):
        path = self.root / 'results/advanced/metrics.json'
        values = json.loads(path.read_text())
        values.pop('gpqa')
        write(path, values)
        with self.assertRaisesRegex(ValueError, 'mandatory eight-benchmark'):
            build_package(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_mismatched_evaluation_protocol_fails(self):
        path = self.root / 'results/random/manifest.json'
        value = json.loads(path.read_text())
        value['seed'] = 99
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'protocol differs at seed'):
            build_package(self.root, self.output)

    def test_changed_original_dataset_hash_fails(self):
        path = self.root / 'artifacts/training/random/dataset.json'
        values = json.loads(path.read_text())
        values[0]['output'] = 'rewritten target'
        write(path, values)
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            build_package(self.root, self.output)

    def test_changed_saved_weights_fail(self):
        (self.root / 'runs/random/model/model.safetensors').write_bytes(b'changed saved weights')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            build_package(self.root, self.output)

    def test_wrong_training_recipe_is_rejected_even_with_updated_binding(self):
        path = self.root / 'runs/random/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['recipe']['parameter_dtype'] = 'bfloat16'
        write(path, manifest)
        complete_path = path.parent / 'complete.json'
        complete = json.loads(complete_path.read_text())
        complete['binding_sha256'] = json_digest(manifest)
        write(complete_path, complete)
        with self.assertRaisesRegex(ValueError, 'training recipe differs'):
            build_package(self.root, self.output)

    def test_existing_package_is_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            build_package(self.root, self.output)

    def test_complete_receipt_cannot_hide_missing_finite_updates(self):
        (self.root / 'runs/random/finite_updates.jsonl').write_text(
            json.dumps(dict(step=1875, gradient_norm=1., actual_presentations=15000)) + '\n')
        with self.assertRaisesRegex(ValueError, 'finite-update history is incomplete'):
            build_package(self.root, self.output)


if __name__ == '__main__':
    unittest.main()
