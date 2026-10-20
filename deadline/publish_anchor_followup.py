"""CPU-only closeout/publication of one confirmed validation-selected anchor.

Nothing is submitted here. Prior delivered artifacts remain immutable. A source
confirmation is required, but an honest complete benchmark result may be
published even when the prospective two-metric benchmark objective is unmet.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import unittest

from deadline import compare_uncertainty as analysis
from deadline import publish_huggingface as hubtools
from deadline import publish_optional_model as incumbent_tools
from deadline.anchor_search import core, screen
from deadline.audit_results import EXPECTED, OFFICIAL, audit_run, compare_coverage
from deadline.package_deliverables import Files
from deadline.recovery.common import digest, json_digest, require_compute, write_new_json
from reasoning.benchmark import record_ids
from reasoning.data import read_rows

HERE = Path(__file__).resolve().parent
OWNER = hubtools.OWNER
PLAN_SHA256 = 'f703aba665002f5a0eff60e4580255a961583d39ce2047c1b92ddb515affaaf4'
ANALYSIS_SHA256 = 'bbc9f515302d0cd2b19059e07140efaa778b1245c81515cd51021ef2a78b5b6f'
CANDIDATES = {'alpha0125': .125, 'alpha0375': .375}
DESTINATIONS = {role: OWNER + '/qwen25-3b-validation-anchor-' + role + '-20261020' for role in CANDIDATES}
MANIFEST = hubtools.MANIFEST
SCHEMA = 'deadline-validation-anchor-publication-v1'
require = hubtools.require


def source_inventory():
    names = ('deadline/publish_anchor_followup.py', 'tests/test_validation_anchor_publication.py',
        'deadline/operations/publish_anchor_followup.slurm', 'deadline/publish_huggingface.py',
        'deadline/publish_optional_model.py', 'deadline/package_deliverables.py',
        'deadline/audit_results.py', 'deadline/compare_uncertainty.py',
        'deadline/anchor_search/__init__.py', 'deadline/anchor_search/core.py',
        'deadline/anchor_search/screen.py', 'deadline/anchor_search/plan.json',
        'deadline/recovery/common.py', 'deadline/recovery/merge.py', 'deadline/recovery/screen.py',
        'deadline/recovery/plan.json', 'deadline/experiment_plan.json', 'reasoning/benchmark.py',
        'reasoning/data.py', 'reasoning/features.py', 'reasoning/evaluate.py',
        'eval/utils/grader.py', 'eval/utils/parser.py', 'eval/utils/math_normalization.py')
    return {name: digest(HERE.parent / name) for name in names}


def validate_qa(path):
    value = hubtools.read(path)
    require(value.get('schema') == 'deadline-validation-anchor-publication-native-qa-v1' and
            value.get('success') is True and value.get('tests_run', 0) > 0 and
            all(value.get(key) == 0 for key in ('failures', 'errors', 'skips')) and
            value.get('wrapper_syntax_passed') is True and
            value.get('source_sha256') == source_inventory() and
            value.get('allocation', {}).get('cuda_visible_devices') == '',
            'Successful unchanged scoped native CPU publication QA is required')
    return value


def run_native_qa(path):
    allocation = require_compute(cpu_only=True)
    require(not Path(path).exists(), 'Preserve existing native QA receipts')
    before = source_inventory()
    subprocess.run(['bash', '-n', str(HERE / 'operations/publish_anchor_followup.slurm')], check=True)
    suite = unittest.defaultTestLoader.discover(str(HERE.parent / 'tests'),
                                              pattern='test_validation_anchor_publication.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(before == source_inventory(), 'Scoped publication code changed during native QA')
    success = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    value = dict(schema='deadline-validation-anchor-publication-native-qa-v1', success=success,
        created_at_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
        allocation=allocation, source_sha256=before, tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped),
        wrapper_syntax_passed=True, network_or_upload_performed=False)
    write_new_json(path, value)
    require(success, 'Native publication tests failed; publication forbidden')
    return value


def validate_state_path(root, followup, state):
    root, followup, state = (Path(path).resolve() for path in (root, followup, state))
    require(state == root / 'hf/anchor-v2-publication' and
            not state.is_relative_to(followup) and not followup.is_relative_to(state),
            'Only the separate anchor-v2 publication state is permitted')
    return state


def repository_for(role):
    require(role in CANDIDATES, 'Only one of the two frozen source candidates may be published')
    value = DESTINATIONS[role]
    old = {OWNER + '/' + item[2] for item in hubtools.DESTINATIONS.values()} | {incumbent_tools.REPO_ID}
    require(value not in old and value == OWNER + '/qwen25-3b-validation-anchor-' + role + '-20261020',
            'Follow-up destination collides with a delivered artifact or differs from its fixed identity')
    return value


def validate_prior_publications(root, files):
    root = Path(root).resolve()
    primary_path = root / 'hf/publication/publication_complete.json'
    incumbent_path = root / 'hf/optional-recovery-publication/publication_complete.json'
    files.hash(primary_path)
    primary = incumbent_tools.validate_primary_receipt(hubtools.read(primary_path))
    files.hash(incumbent_path)
    incumbent = hubtools.read(incumbent_path)
    require(incumbent.get('schema') == 'deadline-optional-recovery-publication-complete-v1' and
            incumbent.get('complete') is True and incumbent.get('owner') == OWNER and
            incumbent.get('repo_id') == incumbent_tools.REPO_ID and
            incumbent.get('url') == 'https://huggingface.co/' + incumbent_tools.REPO_ID and
            incumbent.get('commit') == '51814ba26ab0dc052447aee1612bebcffdba43f2' and
            incumbent.get('public_access_verified') is True and
            incumbent.get('evaluated_and_published_weight_bytes_identical') is True and
            incumbent.get('primary_four_artifacts_unchanged') is True and
            isinstance(incumbent.get('files'), dict) and incumbent['files'],
            'Complete anonymously verified primary and incumbent publications are required first')
    return primary, incumbent, {'primary_publication_complete.json': primary_path,
                                'incumbent_publication_complete.json': incumbent_path}


def validate_winner(prepared, decision, merge, followup):
    followup = Path(followup).resolve()
    require(prepared.get('schema') == core.SCHEMA and prepared.get('plan_sha256') == PLAN_SHA256 and
            decision.get('schema') == screen.SCHEMA and decision.get('phase') == 'decision' and
            decision.get('complete') is True and decision.get('accepted') is True and
            decision.get('plan_sha256') == PLAN_SHA256,
            'An accepted sealed source-only search and confirmation are required')
    role = decision.get('winner')
    repository_for(role)
    model = followup / 'candidates' / role / 'model'
    require(Path(decision.get('model_path', '')).resolve() == model and
            prepared.get('models', {}).get(role, {}).get('path') == str(model),
            'Confirmed winner model path differs from the exact prepared candidate')
    output = merge.get('output', {})
    require(merge.get('schema') == core.CANDIDATE_SCHEMA and merge.get('complete') is True and
            merge.get('role') == role and merge.get('alpha_advanced') == CANDIDATES[role] and
            merge.get('alpha_base') == 1-CANDIDATES[role] and merge.get('plan_sha256') == PLAN_SHA256 and
            merge.get('inputs') == prepared.get('parents') and
            merge.get('code_sha256') == prepared.get('code_sha256') and
            output.get('all_finite') is True and output.get('finite_fp32_tensors') == 434 and
            output.get('parameter_dtype') == 'float32' and output.get('path') == str(model) and
            prepared['models'][role] == {'path': str(model), 'files': output.get('files')},
            'Winner alpha, parent, finite FP32 or model inventory binding differs')
    weights, inventory = output.get('weight_files', {}), output.get('files', {})
    require(set(weights) == {f'model-{index:05d}-of-00006.safetensors' for index in range(1, 7)} and
            all(inventory.get(name) == checksum and isinstance(checksum, str) and
                re.fullmatch('[0-9a-f]{64}', checksum) for name, checksum in weights.items()) and
            {'config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'NOTICE'} <= set(inventory),
            'Exact six finite FP32 shards and complete sidecars are required')
    return role, model


def validate_evaluations(runs, model, model_files):
    require(set(runs) == {'original', 'random', 'advanced', 'incumbent', 'followup'},
            'All five benchmark roles are required')
    for run in runs.values():
        incumbent_tools.require_all_nine_raw(run)
        hardware = run['manifest'].get('hardware')
        require(isinstance(hardware, list) and len(hardware) == 1 and
                hardware[0].get('gpu') == 'NVIDIA B200', 'A recorded single B200 is required for every role')
        require(all(type(run['benchmarks'][name]['metrics'].get('truncated_samples')) is int
                    for name in EXPECTED), 'Recorded length finishes are required for every benchmark')
    matching = compare_coverage(runs, names=tuple(EXPECTED))
    require(matching['matched'], 'Five-role raw evaluation protocol/data/hardware mismatch: ' + '; '.join(matching['errors']))
    manifest = runs['followup']['manifest']
    require(Path(manifest.get('model', '')).resolve() == Path(model).resolve() and
            manifest.get('identity') == incumbent_tools._model_loaded_files(model_files),
            'Evaluated follow-up weights/tokenizer differ from the confirmed merge output')
    return matching


def validate_control_history(runs, archived):
    require(archived.get('schema') == 'deadline-optional-recovery-all-nine-evaluation-v1' and
            archived.get('all_nine_matched_raw') is True and archived.get('primary_comparison_unmodified') is True,
            'The already-published complete four-role comparison is required')
    for role, old_role in (('original', 'original'), ('random', 'random'),
                           ('advanced', 'advanced'), ('incumbent', 'recovery')):
        old = archived.get('models', {}).get(old_role, {})
        for name in EXPECTED:
            require(runs[role]['benchmarks'][name]['metrics'] == old.get('benchmarks', {}).get(name, {}).get('metrics'),
                    'Delivered control benchmark evidence changed: ' + role + '/' + name)
        for label in ('official_eight_macro', 'supplemented_historical_seven_macro'):
            require(runs[role][label] == old.get(label), 'Delivered control macro changed: ' + role)


def _bind_control_archives(root, folders, files):
    archives = (
        (root / 'report/primary-final-20261020T1708Z/uncertainty.json',
         'e16b3b8c63cdbeddfcd1a41995e53278442df92f7bcf642edaf5d34463003823',
         {'original': 'original', 'random': 'random', 'advanced': 'advanced'}),
        (root / 'report/recovery-final-20261020T1730Z/uncertainty-native-candidate-alias.json',
         'fe7fc3930076391d4bb1618fe1900b4909ce2c7b8cc5d7cee92b7c3cb336d053',
         {'original': 'original', 'random': 'random', 'advanced': 'incumbent'}))
    for path, checksum, aliases in archives:
        files.hash(path, checksum)
        old = hubtools.read(path)
        for alias, role in aliases.items():
            source = old['sources'][alias]
            require(Path(source['path']).resolve() == folders[role].resolve(), 'Delivered raw control folder differs')
            files.hash(folders[role] / 'manifest.json', source['manifest_sha256'])
            require(bool(source.get('raw_batch_sha256')), 'Prior raw control inventory is missing')
            for batch, expected in source['raw_batch_sha256'].items():
                require(Path(batch).resolve().is_relative_to(folders[role].resolve()), 'Prior raw control path is outside its role')
                files.hash(batch, expected)


def _diagnostics(records):
    tokens, lengths = [], 0
    for row in records:
        counts, finishes = row.get('output_tokens'), row.get('finish_reasons')
        require(isinstance(counts, list) and len(counts) == 8 and
                all(type(count) is int and 0 <= count <= 32768 for count in counts) and
                isinstance(finishes, list) and len(finishes) == 8 and
                all(reason in ('length', 'stop') for reason in finishes), 'Incomplete recorded generation diagnostics')
        tokens.extend(counts)
        lengths += finishes.count('length')
    return dict(samples=len(tokens), length_finishes=lengths,
        total_output_tokens=sum(tokens), mean_output_tokens=sum(tokens)/len(tokens))


def label_uncertainty(native, evaluation):
    require(native.get('schema') == 'deadline-paired-cluster-bootstrap-v1' and
            native.get('seed') == 20261019 and native.get('resamples') == 10000 and
            native.get('confidence') == .95 and not native.get('audit_errors') and
            not native.get('unavailable_benchmarks') and
            native.get('multiple_comparison_adjustment') is False,
            'Complete unchanged 10K conditional bootstrap evidence is required')
    comparisons = {}
    for pair, comparator in (('advanced_vs_original', 'original'), ('advanced_vs_random', 'incumbent')):
        value = native['comparisons'][pair]
        require(set(value['benchmarks']) == set(EXPECTED) and set(value['macros']) ==
                {'official_eight_macro', 'supplemented_historical_seven_macro'}, 'Incomplete all-nine native uncertainty')
        for benchmark, estimates in value['benchmarks'].items():
            require(set(estimates.get('metrics', {})) == {'pass_at_1', 'pass_at_8'}, 'Incomplete native per-task intervals')
            for metric, estimate in estimates['metrics'].items():
                require(estimate['advanced'] == evaluation['models']['followup']['benchmarks'][benchmark]['metrics'][metric] and
                        estimate['comparator'] == evaluation['models'][comparator]['benchmarks'][benchmark]['metrics'][metric],
                        'Native per-task estimates differ from the matched evaluation')
        for label, macro in value['macros'].items():
            require(macro.get('status') == 'complete' and set(macro.get('metrics', {})) ==
                    {'pass_at_1', 'pass_at_8'}, 'Partial uncertainty macro is forbidden')
            for metric, estimate in macro['metrics'].items():
                require(estimate['advanced'] == evaluation['models']['followup'][label][metric] and
                        estimate['comparator'] == evaluation['models'][comparator][label][metric],
                        'Bootstrap point estimates differ from the five-role evaluation summary')
        comparisons['followup_vs_' + comparator] = value
    def relabel(value):
        if isinstance(value, dict):
            return {({'advanced': 'followup', 'random': 'incumbent'}.get(key, key)): relabel(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [relabel(item) for item in value]
        return value
    return dict(schema='deadline-validation-anchor-uncertainty-summary-v1', complete=True,
        alias_map={'original': 'original', 'random': 'incumbent', 'advanced': 'followup'},
        seed=20261019, resamples=10000, confidence=.95,
        interval=native['interval'], conditioning=native['conditioning'],
        resampling_unit=native['resampling_unit'], metric_weighting=native['metric_weighting'],
        multiple_comparison_adjustment=False, exploratory=True, posthoc=True,
        historical_leaderboard_equivalence_verified=False,
        comparisons=relabel(comparisons), sources_under_native_aliases=native['sources'],
        limitations=['Fixed models and eight recorded outputs only; no training-seed or additional generation variance',
                     'Validation-guided scalar search followed earlier benchmark observations',
                     'No multiple-comparison adjustment or general superiority/SOTA claim'])


def prepare(root, followup, state, qa_receipt):
    root, followup = Path(root).resolve(), Path(followup).resolve()
    state = validate_state_path(root, followup, state)
    validate_qa(qa_receipt)
    files = Files()
    files.hash(qa_receipt)
    primary, incumbent, prior_paths = validate_prior_publications(root, files)
    require(files.hash(HERE / 'anchor_search/plan.json') == PLAN_SHA256, 'Frozen validation search plan changed')
    prepared_path, decision_path = followup / 'prepared.json', followup / 'decision.json'
    files.hash(prepared_path)
    files.hash(decision_path)
    prepared = core.load_prepared(followup)
    decision = screen.validate_decision(followup)
    role = decision.get('winner')
    repository_for(role)
    record = prepared['candidate_receipts'][role]
    require(Path(record['path']).resolve() == followup / 'candidates' / role / 'complete.json', 'Candidate receipt path differs')
    files.hash(record['path'], record['sha256'])
    merge = hubtools.read(record['path'])
    role, model = validate_winner(prepared, decision, merge, followup)
    for name, checksum in merge['output']['files'].items():
        require(Path(name).name == name, 'Unsafe model inventory filename')
        files.hash(model / name, checksum)
    require(files.hash(model / 'LICENSE') == hubtools.LICENSE_SHA256 and
            hubtools.NOTICE in (model / 'NOTICE').read_text(encoding='utf-8') and
            'Built with Qwen' in (model / 'NOTICE').read_text(encoding='utf-8'), 'Pinned license or existing correct NOTICE differs')
    index = hubtools.read(model / 'model.safetensors.index.json')
    require(set(index.get('weight_map', {}).values()) == set(merge['output']['weight_files']) and
            index.get('metadata', {}).get('total_size') == merge['output']['total_tensor_bytes'], 'Candidate index/shard identity differs')
    for name, checksum in prepared['parents']['advanced']['files'].items():
        if name.endswith('.safetensors'):
            require(primary['advanced_model']['files'].get(name, {}).get('sha256') == checksum,
                    'Advanced parent differs from its published primary weights')
    for name, checksum in prepared['models']['incumbent']['files'].items():
        if name.endswith('.safetensors'):
            require(incumbent['files'].get(name, {}).get('sha256') == checksum,
                    'Incumbent control differs from its published weights')
    folders = {name: root / 'results' / name for name in ('original', 'random', 'advanced')}
    folders.update(incumbent=root / 'recovery-runs/base-anchor-75-25-20261020/benchmarks', followup=followup / 'benchmarks')
    batches = {}
    for name, folder in folders.items():
        files.hash(folder / 'manifest.json')
        files.hash(folder / 'metrics.json')
        batches[name] = {benchmark: sorted((folder / benchmark).glob('batch_*.json')) for benchmark in EXPECTED}
        for paths in batches[name].values():
            for path in paths:
                files.hash(path)
    runs = {name: audit_run(folder) for name, folder in folders.items()}
    validate_evaluations(runs, model, merge['output']['files'])
    archived_path = root / 'hf/optional-recovery-publication/package/evaluation_summary.json'
    files.hash(archived_path, incumbent['evaluation_summary_sha256'])
    validate_control_history(runs, hubtools.read(archived_path))
    _bind_control_archives(root, folders, files)
    for control, prepared_role in (('original', 'original'), ('incumbent', 'incumbent')):
        expected = prepared['models'][prepared_role]
        require(Path(runs[control]['manifest']['model']).resolve() == Path(expected['path']).resolve() and
                runs[control]['manifest']['identity'] == incumbent_tools._model_loaded_files(expected['files']),
                'Delivered control checkpoint identity differs: ' + control)
    advanced_identity = runs['advanced']['manifest']['identity']
    expected_advanced = incumbent_tools._model_loaded_files(prepared['parents']['advanced']['files'])
    require(all(advanced_identity.get(name) == checksum for name, checksum in expected_advanced.items()) and
            set(advanced_identity) - set(expected_advanced) <= {'training_args.bin'},
            'Delivered advanced model identity differs from the candidate parent')
    diagnostics = {}
    for name, run in runs.items():
        diagnostics[name] = {}
        for benchmark in EXPECTED:
            records = [row for path in batches[name][benchmark] for row in read_rows(path)]
            identifiers = record_ids(records)
            require(identifiers == run['manifest']['problem_ids'][benchmark] and
                    [row.get('problem_id') for row in records] == identifiers,
                    'Raw question/order identity changed: ' + name + '/' + benchmark)
            diagnostics[name][benchmark] = _diagnostics(records)
    original_macro, candidate_macro = (runs[name]['official_eight_macro'] for name in ('original', 'followup'))
    goal = {metric: candidate_macro[metric] > original_macro[metric] for metric in ('pass_at_1', 'pass_at_8')}
    evaluation = dict(schema='deadline-validation-anchor-five-role-evaluation-v1', complete=True,
        all_nine_matched_raw=True, winner=role, alpha_advanced=CANDIDATES[role],
        prospective_goal='Both supplied-eight macro metrics strictly exceed original; point estimates are not significance',
        prospective_goal_checks=goal, prospective_goal_met=all(goal.values()),
        primary_and_incumbent_unchanged=True, historical_leaderboard_equivalence_verified=False,
        models={name: {key: run[key] for key in ('benchmarks', 'official_eight_macro',
            'supplemented_historical_seven_macro')} | {'generation_diagnostics': diagnostics[name]}
            for name, run in runs.items()})
    require(files.hash(Path(analysis.__file__)) == ANALYSIS_SHA256, 'The unchanged conditional bootstrap is required')
    native = analysis.compare_uncertainty({'original': folders['original'], 'random': folders['incumbent'],
        'advanced': folders['followup']}, resamples=10000, seed=20261019)
    uncertainty = label_uncertainty(native, evaluation)
    sources = source_inventory()
    for name, checksum in sources.items():
        files.hash(HERE.parent / name, checksum)
    files.unchanged()
    package = state / 'package'
    package.mkdir(parents=True, exist_ok=True)
    def generated(name, value):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        hubtools.freeze(path, hubtools.canonical(value) if not isinstance(value, bytes) else value)
        return path
    evaluation_path = generated('evaluation_summary.json', evaluation)
    native_path = generated('uncertainty-native-candidate-alias.json', native)
    uncertainty['native_artifact_sha256'] = files.hash(native_path)
    uncertainty_path = generated('uncertainty.json', uncertainty)
    rows = ['| Benchmark | Original p@1 / p@8 | Random | Advanced | Incumbent 75/25 | Follow-up |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    order = ('original', 'random', 'advanced', 'incumbent', 'followup')
    for name in EXPECTED:
        rows.append('| ' + name + ' | ' + ' | '.join(
            f"{100*runs[control]['benchmarks'][name]['metrics']['pass_at_1']:.2f}% / {100*runs[control]['benchmarks'][name]['metrics']['pass_at_8']:.2f}%"
            for control in order) + ' |')
    for label, key in (('Supplied-eight macro', 'official_eight_macro'), ('Internal seven, supplementary', 'supplemented_historical_seven_macro')):
        rows.append('| ' + label + ' | ' + ' | '.join(f"{100*runs[name][key]['pass_at_1']:.4f}% / {100*runs[name][key]['pass_at_8']:.4f}%" for name in order) + ' |')
    ci_rows = []
    for pair, value in uncertainty['comparisons'].items():
        metrics = value['macros']['official_eight_macro']['metrics']
        ci_rows.append('| ' + pair + ' | ' + ' | '.join(
            f"{100*metrics[key]['delta']:+.4f} [{100*metrics[key]['ci95'][0]:.4f}, {100*metrics[key]['ci95'][1]:.4f}]" for key in ('pass_at_1', 'pass_at_8')) + ' |')
    alpha = CANDIDATES[role]
    card = ('---\nlibrary_name: transformers\npipeline_tag: text-generation\nlicense: other\nlicense_name: qwen-research\n'
        f'license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/{hubtools.BASE_REVISION}/LICENSE\n'
        'base_model: Qwen/Qwen2.5-3B-Instruct\ntags:\n- reasoning\n- model-merge\n- post-hoc\n---\n\n'
        f'# Validation-selected base anchor: {role}\n\n**Built with Qwen.** This is an exploratory post-hoc '
        f'FP32 merge: `{1-alpha:g} * original_base + {alpha:g} * primary_advanced`, not additional SFT. '
        'The original base is pinned at `' + hubtools.BASE_REVISION + '`. The advanced parent is the already published '
        f"[{primary['advanced_model']['repo_id']}]({primary['advanced_model']['tree_url']}). Parent/output shard hashes are sealed.\n\n"
        'The finite two-candidate grid (advanced fractions 0.125 and 0.375), 192 fresh source-search questions, '
        '192 disjoint confirmation questions, eight samples, score, guards and tie-break were fixed before new '
        'source generation. All 384 questions exclude the previously viewed 64 and actual training/probe data. '
        'Selection uses source-teacher agreement, not independently verified truth. One source winner passed the '
        'single fixed confirmation; benchmark outcomes did not choose or rerank the winner. Earlier primary and '
        '75/25 benchmark outcomes had already been viewed. All earlier artifacts remain unchanged.\n\n'
        'All nine tasks below are matched raw eight-sample, seed-0, 32K evaluations with identical data, ordered IDs, '
        'grader, B200 hardware and runtime. AIME25 and the internal-seven macro are supplementary, not verified '
        'historical leaderboard results.\n\n' + '\n'.join(rows) + '\n\n'
        f"Prospective objective (both supplied-eight point estimates above original): **{evaluation['prospective_goal_met']}**. "
        'Publication is not conditioned on that measured objective. Point estimates do not establish significance.\n\n'
        '| Supplied-eight difference, percentage points | pass@1 [95% CI] | pass@8 [95% CI] |\n'
        '| --- | ---: | ---: |\n' + '\n'.join(ci_rows) + '\n\n'
        'Intervals use the unchanged 10,000-draw paired normalized-question-cluster percentile bootstrap, seed '
        '20261019. They condition on fixed models/eight recorded outputs, exclude training-seed and additional '
        'generation variance, and have no multiple-comparison adjustment. The exact native analyzer aliases '
        '`original=original, random=incumbent, advanced=followup`; uncertainty.json labels these roles explicitly. '
        'No general superiority, novelty or SOTA is established.\n\n'
        'Six finite FP32 shards and every model/tokenizer sidecar retain their evaluated bytes; inference used BF16. '
        'The Qwen Research License and existing upstream NOTICE are preserved verbatim. No source solutions or '
        'raw development generations are published. Provenance includes source selection/confirmation, exact '
        'merge, five-role evaluation, conditional uncertainty and scoped native CPU QA.\n')
    card_path = generated('README.md', card.encode())
    modifications = generated('MODIFICATIONS.md', (f'Built with Qwen.\n\nAll six safetensors shards were modified by '
        f'FP32 interpolation: {(1-alpha)*100:g}% original and {alpha*100:g}% primary advanced full-SFT weights. '
        'No additional SFT occurred. Published weights, configuration, tokenizer, LICENSE and NOTICE preserve '
        'their evaluated input bytes. Selection is a post-hoc, source-only, two-candidate scalar search.\n').encode())
    paths = {name: model / name for name in merge['output']['files']}
    paths.update({'README.md': card_path, 'MODIFICATIONS.md': modifications,
        'evaluation_summary.json': evaluation_path, 'uncertainty.json': uncertainty_path,
        'uncertainty-native-candidate-alias.json': native_path})
    provenance = dict(prior_paths)
    provenance.update({'search_plan.json': HERE / 'anchor_search/plan.json', 'prepared.json': prepared_path,
        'selection.json': followup / 'selection.json', 'decision.json': decision_path,
        'native_publication_qa.json': Path(qa_receipt), 'merge_complete.json': Path(record['path']),
        'merge_inputs.json': Path(record['path']).parent / 'merge_inputs.json',
        'evaluation_manifest.json': folders['followup'] / 'manifest.json',
        'evaluation_metrics.json': folders['followup'] / 'metrics.json',
        'delivered_four_role_evaluation_summary.json': archived_path,
        'LICENSE.source.json': HERE / 'licenses/Qwen2.5-3B-Instruct.LICENSE.source.json'})
    for split in ('search', 'confirmation'):
        for name in (split + '.json', split + '_ids.json'):
            provenance[name] = followup / name
        for control in (core.ROLES if split == 'search' else ('original', 'incumbent', role)):
            for name in ('manifest.json', 'complete.json', 'observations.json'):
                provenance[f'screen/{split}/{control}/{name}'] = followup / 'screen' / split / control / name
    paths.update({'provenance/' + name: path for name, path in provenance.items()})
    descriptors = {name: dict(hubtools.fingerprint(path, files), require_lfs=name.endswith('.safetensors'))
                   for name, path in sorted(paths.items())}
    identity = dict(schema=SCHEMA, owner=OWNER, repo_id=repository_for(role), role=role,
        plan_sha256=PLAN_SHA256, source_sha256=sources, prepared_sha256=files.hash(prepared_path),
        decision_sha256=files.hash(decision_path), merge_receipt_sha256=files.hash(record['path']),
        evaluation_summary_sha256=files.hash(evaluation_path), uncertainty_sha256=files.hash(uncertainty_path),
        alpha_advanced=alpha, alpha_base=1-alpha, prospective_goal_met=evaluation['prospective_goal_met'],
        all_nine_matched_raw=True, posthoc=True, no_additional_sft=True, files=descriptors)
    identity['publication_sha256'] = hubtools.sha(hubtools.canonical(identity))
    require(not (state / 'publication_plan.json').exists() or hubtools.read(state / 'publication_plan.json') == identity,
            'Follow-up publication inputs changed; retain state and reconcile')
    manifest = generated(MANIFEST, identity)
    payload = dict(repo_id=identity['repo_id'], repo_type='model', role=role, files=dict(descriptors),
        local_paths={name: str(path.resolve()) for name, path in paths.items()})
    payload['files'][MANIFEST] = dict(hubtools.fingerprint(manifest, files), require_lfs=False)
    payload['local_paths'][MANIFEST] = str(manifest)
    files.unchanged()
    return identity, payload, files


def publish(root, followup, state, *, expected_owner, qa_receipt, hub=None):
    hubtools.compute_only()
    require(expected_owner == OWNER, 'Expected owner must explicitly equal adversarialbandit')
    state = validate_state_path(root, followup, state)
    validate_qa(qa_receipt)
    validate_prior_publications(root, Files())
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hub = hubtools.Hub() if hub is None else hub
        require(hub.whoami() == OWNER, 'Authenticated account differs from expected owner')
        identity, payload, files = prepare(root, followup, state, qa_receipt)
        require(payload['repo_id'] == repository_for(identity['role']) and payload['repo_type'] == 'model',
                'Prepared destination differs from the single confirmed winner')
        plan_path, journal_path = state / 'publication_plan.json', state / 'journal.json'
        if plan_path.exists():
            require(hubtools.read(plan_path) == identity, 'Immutable publication plan changed')
        else:
            hubtools.write(plan_path, identity)
        journal = hubtools.read(journal_path) if journal_path.exists() else dict(
            publication_sha256=identity['publication_sha256'], repo_id=payload['repo_id'], record={})
        require(journal.get('publication_sha256') == identity['publication_sha256'] and
                journal.get('repo_id') == payload['repo_id'], 'Follow-up journal identity changed')
        record = journal['record']
        snapshot = hubtools.inspect(hub, payload, record)
        if snapshot is None:
            hub.create(payload['repo_id'], 'model')
            record['created'] = True
            hubtools.write(journal_path, journal)
            snapshot = hubtools.inspect(hub, payload, record)
            require(snapshot is not None, 'Created destination is not observable; reconcile')
        if MANIFEST not in snapshot['files']:
            files.unchanged()
            committed = hub.commit(payload['repo_id'], 'model', {MANIFEST: payload['local_paths'][MANIFEST]}, snapshot['commit'])
            snapshot = hubtools.inspect(hub, payload, record)
            require(snapshot is not None and snapshot['commit'] == committed and MANIFEST in snapshot['files'],
                    'Publication manifest commit was not independently verified')
        missing = [name for name in payload['files'] if name not in snapshot['files']]
        small = [name for name in missing if not payload['files'][name]['require_lfs']]
        for names in ([small] if small else []) + [[name] for name in missing if name not in small]:
            files.unchanged()
            journal['active_files'] = names
            hubtools.write(journal_path, journal)
            committed = hub.commit(payload['repo_id'], 'model', {name: payload['local_paths'][name] for name in names}, snapshot['commit'])
            snapshot = hubtools.inspect(hub, payload, record)
            require(snapshot is not None and snapshot['commit'] == committed and all(name in snapshot['files'] for name in names),
                    'Follow-up content commit was not verified')
            record['last_verified_commit'] = committed
            hubtools.write(journal_path, journal)
        public = hubtools.inspect(hub, payload, record, public=True)
        require(public is not None and set(payload['files']) <= set(public['files']), 'Anonymous public tree is incomplete')
        proof = {name: hubtools.verify_file(public['files'][name], expected) for name, expected in payload['files'].items()}
        final = hubtools.inspect(hub, payload, record, public=True)
        require(final is not None and final['commit'] == public['commit'], 'Public head changed before completion')
        files.unchanged()
        receipt = dict(schema='deadline-validation-anchor-publication-complete-v1', complete=True,
            owner=OWNER, repo_id=payload['repo_id'], url='https://huggingface.co/' + payload['repo_id'],
            commit=public['commit'], public_access_verified=True, files=proof,
            publication_sha256=identity['publication_sha256'], winner=identity['role'],
            plan_sha256=PLAN_SHA256, evaluation_summary_sha256=identity.get('evaluation_summary_sha256'),
            uncertainty_sha256=identity.get('uncertainty_sha256'), prospective_goal_met=identity.get('prospective_goal_met'),
            evaluated_and_published_weight_bytes_identical=True, prior_five_artifacts_unchanged=True,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        hubtools.write(state / 'publication_complete.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('deadline'))
    parser.add_argument('--followup', type=Path, default=Path('deadline/anchor-search-runs/validation-v2-20261020-attempt2'))
    parser.add_argument('--state', type=Path, default=Path('deadline/hf/anchor-v2-publication'))
    parser.add_argument('--qa-receipt', type=Path, required=True)
    parser.add_argument('--expected-owner')
    parser.add_argument('--run-native-qa', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.run_native_qa:
        require(not args.prepare_only, 'Choose QA or publication preparation')
        run_native_qa(args.qa_receipt)
        return
    args.state = validate_state_path(args.root, args.followup, args.state)
    require(args.expected_owner == OWNER, 'Expected owner must explicitly equal adversarialbandit')
    def expired(_signal, _frame):
        raise TimeoutError('Follow-up CPU publication time budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(3300)
    try:
        if args.prepare_only:
            hubtools.compute_only()
            identity, _, _ = prepare(args.root, args.followup, args.state, args.qa_receipt)
            result = {'staged': True, 'uploaded': False, 'publication_sha256': identity['publication_sha256']}
        else:
            result = publish(args.root, args.followup, args.state, expected_owner=args.expected_owner, qa_receipt=args.qa_receipt)
        print(json.dumps({key: result[key] for key in ('complete', 'url', 'staged', 'uploaded') if key in result}), flush=True)
    except Exception as error:
        reason = str(error) if isinstance(error, hubtools.PublicationError) else None
        if args.state.exists():
            hubtools.write(args.state / 'failure.json', dict(complete=False, error_type=type(error).__name__,
                reason=reason, at_utc=datetime.now(timezone.utc).isoformat()))
        raise SystemExit('Follow-up publication incomplete (' + type(error).__name__ + '). ' + (reason or 'No success inferred.')) from None
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
