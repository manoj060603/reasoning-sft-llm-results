"""Offline guards and transactions; execute only in native CPU compute QA."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deadline import publish_anchor_followup as module
from deadline import publish_huggingface as hubtools
from deadline.audit_results import EXPECTED, PROTOCOL, REQUIRED_SETTINGS
from deadline.package_deliverables import Files


def winner_fixture(folder, role='alpha0125'):
    model = Path(folder).resolve() / 'candidates' / role / 'model'
    weights = {f'model-{index:05d}-of-00006.safetensors': f'{index:064x}' for index in range(1, 7)}
    inventory = dict(weights)
    inventory.update({name: 'a'*64 for name in ('config.json', 'generation_config.json',
        'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'NOTICE')})
    prepared = dict(schema=module.core.SCHEMA, plan_sha256=module.PLAN_SHA256,
        code_sha256={'core.py': 'c'*64}, parents={'base': 'fixed', 'advanced': 'fixed'},
        models={role: {'path': str(model), 'files': inventory}})
    decision = dict(schema=module.screen.SCHEMA, phase='decision', complete=True, accepted=True,
        winner=role, model_path=str(model), plan_sha256=module.PLAN_SHA256)
    merge = dict(schema=module.core.CANDIDATE_SCHEMA, complete=True, role=role,
        alpha_advanced=module.CANDIDATES[role], alpha_base=1-module.CANDIDATES[role],
        plan_sha256=module.PLAN_SHA256, inputs=prepared['parents'], code_sha256=prepared['code_sha256'],
        output=dict(path=str(model), files=inventory, weight_files=weights, all_finite=True,
                    finite_fp32_tensors=434, parameter_dtype='float32'))
    return prepared, decision, merge


def runs_fixture(model, model_files):
    manifest = {name: 0 for name in PROTOCOL}
    manifest.update(REQUIRED_SETTINGS)
    manifest.update(seed=0, max_model_len=32768, versions={'fixture': 'offline'},
        hardware=[{'gpu': 'NVIDIA B200'}], data={name: 'a'*64 for name in EXPECTED},
        problem_counts=dict(EXPECTED), problem_ids={name: [f'{index:064x}' for index in range(count)]
                                                   for name, count in EXPECTED.items()})
    result = {}
    for role in ('original', 'random', 'advanced', 'incumbent', 'followup'):
        copied = copy.deepcopy(manifest)
        copied.update(model=str(model), identity=module.incumbent_tools._model_loaded_files(model_files))
        result[role] = dict(manifest=copied, errors=[], benchmarks={name: dict(status='complete',
            evidence='raw eight-sample predictions verified', metrics=dict(problems=count,
                samples_per_problem=8, pass_at_1=0., pass_at_8=0., truncated_samples=0))
            for name, count in EXPECTED.items()})
    return result


class SourceAndEvaluationGuardTests(unittest.TestCase):
    def test_exact_both_candidate_alphas_are_supported(self):
        for role in module.CANDIDATES:
            values = winner_fixture('/tmp/anchor-unit-fixture', role)
            self.assertEqual(module.validate_winner(*values, '/tmp/anchor-unit-fixture')[0], role)

    def test_missing_or_rejected_source_confirmation_is_refused(self):
        for mutate in (lambda value: value.clear(), lambda value: value.update(accepted=False),
                       lambda value: value.update(phase='selection'), lambda value: value.update(complete=False)):
            prepared, decision, merge = winner_fixture('/tmp/anchor-unit-fixture')
            mutate(decision)
            with self.assertRaises(ValueError):
                module.validate_winner(prepared, decision, merge, '/tmp/anchor-unit-fixture')

    def test_wrong_winner_alpha_path_parent_or_finite_proof_is_refused(self):
        mutations = (
            lambda p,d,m: d.update(winner='incumbent'),
            lambda p,d,m: d.update(model_path='/tmp/other-model'),
            lambda p,d,m: m.update(alpha_advanced=.25),
            lambda p,d,m: m.update(alpha_base=.75),
            lambda p,d,m: m.update(inputs={'base': 'substituted'}),
            lambda p,d,m: m.update(plan_sha256='b'*64),
            lambda p,d,m: m['output'].update(finite_fp32_tensors=433),
            lambda p,d,m: m['output'].update(all_finite=False),
            lambda p,d,m: m['output'].update(parameter_dtype='bfloat16'),
            lambda p,d,m: m['output']['weight_files'].pop('model-00006-of-00006.safetensors'),
        )
        for mutate in mutations:
            values = winner_fixture('/tmp/anchor-unit-fixture')
            mutate(*values)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                module.validate_winner(*values, '/tmp/anchor-unit-fixture')

    def test_five_complete_raw_roles_and_exact_loaded_identity_are_accepted(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        output = merge['output']
        result = module.validate_evaluations(runs_fixture(output['path'], output['files']),
                                             output['path'], output['files'])
        self.assertTrue(result['matched'])

    def test_missing_ninth_task_aggregate_evidence_or_fifth_role_is_refused(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        output = merge['output']
        mutations = (
            lambda values: values.pop('incumbent'),
            lambda values: values['followup']['benchmarks'].pop('aime25'),
            lambda values: values['incumbent']['benchmarks']['gpqa'].update(status='missing'),
            lambda values: values['followup']['benchmarks']['math'].update(evidence='aggregate receipt only'),
        )
        for mutate in mutations:
            values = runs_fixture(output['path'], output['files'])
            mutate(values)
            with self.assertRaises(ValueError):
                module.validate_evaluations(values, output['path'], output['files'])

    def test_aime25_protocol_hardware_and_model_tampering_are_refused(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        output = merge['output']
        mutations = (
            lambda values: values['followup']['manifest']['data'].update(aime25='different'),
            lambda values: values['incumbent']['manifest'].update(seed=42),
            lambda values: values['followup']['manifest'].update(hardware={'gpu': 'different'}),
            lambda values: values['followup']['manifest']['identity'].update(tokenizer='unsealed'),
        )
        for mutate in mutations:
            values = runs_fixture(output['path'], output['files'])
            mutate(values)
            with self.assertRaises(ValueError):
                module.validate_evaluations(values, output['path'], output['files'])

    def test_prior_publication_is_required_before_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'deadline'
            root.mkdir()
            with self.assertRaises((OSError, ValueError)):
                module.validate_prior_publications(root, Files())
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_hardware_from_every_role_is_not_a_matched_b200_claim(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        output = merge['output']
        values = runs_fixture(output['path'], output['files'])
        for value in values.values():
            value['manifest'].pop('hardware')
        with self.assertRaisesRegex(ValueError, 'B200'):
            module.validate_evaluations(values, output['path'], output['files'])

    def test_altered_control_metrics_cannot_be_called_unchanged(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        runs = runs_fixture(merge['output']['path'], merge['output']['files'])
        for value in runs.values():
            value.update(official_eight_macro={'pass_at_1':0., 'pass_at_8':0.},
                         supplemented_historical_seven_macro={'pass_at_1':0., 'pass_at_8':0.})
        archived = dict(schema='deadline-optional-recovery-all-nine-evaluation-v1',
            all_nine_matched_raw=True, primary_comparison_unmodified=True,
            models={('recovery' if name == 'incumbent' else name): copy.deepcopy(value)
                    for name, value in runs.items() if name != 'followup'})
        module.validate_control_history(runs, archived)
        runs['incumbent']['benchmarks']['gpqa']['metrics']['pass_at_1'] = .125
        with self.assertRaisesRegex(ValueError, 'control benchmark evidence changed'):
            module.validate_control_history(runs, archived)

    def test_old_repository_collision_is_refused(self):
        with patch.dict(module.DESTINATIONS, {'alpha0125': module.incumbent_tools.REPO_ID}):
            with self.assertRaises(ValueError):
                module.repository_for('alpha0125')

    def test_protected_state_paths_are_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'deadline'
            root.mkdir()
            followup = root / 'anchor-search-runs/example'
            for state in (root/'hf/publication', root/'hf/optional-recovery-publication',
                          root/'runs/advanced/model', followup/'publication'):
                with self.assertRaises(ValueError):
                    module.validate_state_path(root, followup, state)
            self.assertEqual(list(root.iterdir()), [])


class FakeHub:
    def __init__(self):
        self.name = module.OWNER
        self.repos, self.writes, self.sequence = {}, [], 0
        self.hide_public, self.corrupt_public, self.fail_commit = False, False, None

    def whoami(self):
        return self.name

    def snapshot(self, repo_id, repo_type, *, public=False):
        if public and self.hide_public:
            return None
        value = copy.deepcopy(self.repos.get((repo_type, repo_id)))
        if value and public and self.corrupt_public:
            for name, row in value['files'].items():
                if name.endswith('.safetensors'):
                    row['lfs']['sha256'] = '0'*64
        return value

    def create(self, repo_id, repo_type):
        key = repo_type, repo_id
        if key in self.repos:
            raise RuntimeError('already exists')
        self.repos[key] = {'commit': '0'*40, 'files': {}}
        self.writes.append(('create', key))

    def commit(self, repo_id, repo_type, paths, parent):
        key = repo_type, repo_id
        value = self.repos[key]
        if value['commit'] != parent:
            raise RuntimeError('parent changed')
        for name, path in paths.items():
            data = Path(path).read_bytes()
            value['files'][name] = dict(size=len(data), blob_id=hashlib.sha1(
                b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(),
                lfs={'sha256': hubtools.sha(data), 'size': len(data)} if name.endswith('.safetensors') else None)
        self.sequence += 1
        value['commit'] = f'{self.sequence:040x}'
        self.writes.append(('commit', key, tuple(paths)))
        if self.fail_commit == self.sequence:
            raise TimeoutError('synthetic accepted commit with lost response')
        return value['commit']


class UncertaintyBindingTests(unittest.TestCase):
    def fixture(self):
        _, _, merge = winner_fixture('/tmp/anchor-unit-fixture')
        runs = runs_fixture(merge['output']['path'], merge['output']['files'])
        macro = {'pass_at_1':0., 'pass_at_8':0.}
        for value in runs.values():
            value.update(official_eight_macro=copy.deepcopy(macro),
                         supplemented_historical_seven_macro=copy.deepcopy(macro))
        def estimates():
            return {key: dict(advanced=0., comparator=0., delta=0., ci95=[-.01,.01])
                    for key in ('pass_at_1','pass_at_8')}
        pair = dict(benchmarks={name: {'metrics':estimates()} for name in EXPECTED},
            macros={name: dict(status='complete', metrics=estimates()) for name in
                    ('official_eight_macro','supplemented_historical_seven_macro')})
        native = dict(schema='deadline-paired-cluster-bootstrap-v1', seed=20261019,
            resamples=10000, confidence=.95, audit_errors={}, unavailable_benchmarks={},
            multiple_comparison_adjustment=False, interval='paired cluster percentile',
            conditioning='fixed outputs', resampling_unit='question cluster',
            metric_weighting='equal benchmark', sources={'original': {'path':'original'},
                'random':{'path':'incumbent'}, 'advanced':{'path':'followup'}},
            comparisons={name: copy.deepcopy(pair) for name in ('advanced_vs_original','advanced_vs_random')})
        return native, {'models':runs}

    def test_native_candidate_and_incumbent_aliases_are_explicit(self):
        native, evaluation = self.fixture()
        result = module.label_uncertainty(native, evaluation)
        self.assertEqual(result['alias_map'], {'original':'original','random':'incumbent','advanced':'followup'})
        self.assertEqual(set(result['comparisons']), {'followup_vs_original','followup_vs_incumbent'})
        metric = result['comparisons']['followup_vs_incumbent']['macros']['official_eight_macro']['metrics']['pass_at_1']
        self.assertIn('followup', metric)
        self.assertNotIn('advanced', metric)
        self.assertEqual(result['sources_under_native_aliases'], native['sources'])

    def test_empty_macro_missing_task_wrong_point_or_wrong_seed_is_refused(self):
        mutations = (
            lambda value: value['comparisons']['advanced_vs_original'].update(macros={}),
            lambda value: value['comparisons']['advanced_vs_random']['benchmarks'].pop('aime25'),
            lambda value: value['comparisons']['advanced_vs_original']['benchmarks']['gpqa']['metrics']['pass_at_1'].update(advanced=.125),
            lambda value: value.update(seed=0),
            lambda value: value.update(resamples=1000),
        )
        for mutate in mutations:
            native, evaluation = self.fixture()
            mutate(native)
            with self.assertRaises(ValueError):
                module.label_uncertainty(native, evaluation)


class TransactionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'deadline'
        self.root.mkdir()
        self.followup = self.root / 'anchor-search-runs/fixture'
        self.followup.mkdir(parents=True)
        self.state = self.root / 'hf/anchor-v2-publication'
        self.prior = self.root / 'prior-untouched.json'
        self.prior.write_bytes(b'preserve every delivered artifact')
        self.hub, self.files = FakeHub(), Files()
        paths = {}
        contents = {'README.md': b'Built with Qwen. Offline fixture.', module.MANIFEST: b'fixed-fixture-manifest'}
        contents.update({f'model-{index:05d}-of-00006.safetensors': f'fake-shard-{index}'.encode() for index in range(1, 7)})
        for name, data in contents.items():
            path = self.followup / name
            path.write_bytes(data)
            paths[name] = str(path)
        descriptors = {name: dict(hubtools.fingerprint(path, self.files), require_lfs=name.endswith('.safetensors'))
                       for name, path in paths.items()}
        self.identity = dict(schema=module.SCHEMA, role='alpha0125', publication_sha256='a'*64,
            prospective_goal_met=False, evaluation_summary_sha256='b'*64, uncertainty_sha256='c'*64)
        self.payload = dict(repo_id=module.repository_for('alpha0125'), repo_type='model', role='alpha0125',
            files=descriptors, local_paths=paths)
        for manager in (
            patch.dict(os.environ, {'SLURM_JOB_ID':'123', 'SLURM_JOB_PARTITION':'normal_q',
                'SLURM_JOB_GPUS':'', 'SLURM_GPUS':'', 'SLURM_GPUS_ON_NODE':''}),
            patch.object(module, 'validate_qa', return_value={'success':True}),
            patch.object(module, 'validate_prior_publications', return_value=({}, {}, {})),
            patch.object(module, 'prepare', side_effect=lambda *args: (
                copy.deepcopy(self.identity), copy.deepcopy(self.payload), self.files)),
        ):
            manager.start()
            self.addCleanup(manager.stop)

    def publish(self):
        return module.publish(self.root, self.followup, self.state, expected_owner=module.OWNER,
                              qa_receipt=self.root/'qa.json', hub=self.hub)

    def test_complete_negative_benchmark_goal_can_be_published_honestly(self):
        receipt = self.publish()
        self.assertTrue(receipt['complete'])
        self.assertFalse(receipt['prospective_goal_met'])
        self.assertEqual(self.prior.read_bytes(), b'preserve every delivered artifact')
        self.assertEqual(set(self.hub.repos), {('model', module.repository_for('alpha0125'))})

    def test_identical_retry_is_idempotent(self):
        first = self.publish()
        writes = copy.deepcopy(self.hub.writes)
        second = self.publish()
        self.assertEqual(first['publication_sha256'], second['publication_sha256'])
        self.assertEqual(self.hub.writes, writes)

    def test_accepted_lost_shard_commit_is_not_reuploaded(self):
        self.hub.fail_commit = 3
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertFalse((self.state/'publication_complete.json').exists())
        self.assertTrue(self.publish()['complete'])
        shard = 'model-00001-of-00006.safetensors'
        self.assertEqual(sum(item[0] == 'commit' and shard in item[2] for item in self.hub.writes), 1)

    def test_existing_conflicting_or_unknown_empty_repository_is_not_adopted(self):
        for tree in ({}, {'foreign.txt': {'size':0}}):
            self.hub.repos[('model', module.repository_for('alpha0125'))] = {'commit':'d'*40, 'files':tree}
            with self.assertRaises(ValueError):
                self.publish()
            self.assertEqual(self.hub.writes, [])

    def test_anonymous_visibility_or_hash_failure_prevents_completion(self):
        for name in ('hide_public', 'corrupt_public'):
            setattr(self.hub, name, True)
            with self.assertRaises(ValueError):
                self.publish()
            self.assertFalse((self.state/'publication_complete.json').exists())
            setattr(self.hub, name, False)

    def test_wrong_account_prevents_remote_writes(self):
        self.hub.name = 'other-owner'
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_scoped_native_qa_is_required_before_remote_operations(self):
        with patch.object(module, 'validate_qa', side_effect=ValueError('failed QA')):
            with self.assertRaises(ValueError):
                self.publish()
        self.assertEqual(self.hub.writes, [])
        self.assertFalse(self.state.exists())


if __name__ == '__main__':
    unittest.main()
