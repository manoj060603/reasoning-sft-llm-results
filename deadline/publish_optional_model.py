"""Separate optional post-hoc recovery publication, after primary delivery.

CPU allocation only. Existing primary publication code and four destinations
remain unchanged. Stage references to exact evaluated FP32 files; never convert
or copy model weights. A distinct manifest, state and remote repository own this
transaction. Completion requires anonymous verification of every remote file.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import signal

from deadline import publish_huggingface as primary
from deadline.audit_results import (EXPECTED, OFFICIAL, audit_run, compare_coverage,
                                    validate_metric)
from deadline.package_deliverables import Files
from deadline.recovery.common import digest, json_digest
from deadline.recovery.screen import model_inputs, validate_summary
from reasoning.benchmark import record_ids
from reasoning.data import read_rows


OWNER = primary.OWNER
REPO_ID = OWNER + '/qwen25-3b-base75-advanced25-posthoc-20261020'
MANIFEST = primary.MANIFEST
SCHEMA = 'deadline-optional-recovery-publication-v1'
HERE = Path(__file__).resolve().parent
require = primary.require
PublicationError = primary.PublicationError


def source_inventory():
    """Native QA may bind this inventory without altering recovery's old seal."""
    project = HERE.parent
    names = ('deadline/publish_optional_model.py', 'tests/test_optional_publication.py',
             'deadline/operations/publish_optional_model.slurm',
             'deadline/publish_huggingface.py', 'deadline/package_deliverables.py',
             'deadline/audit_results.py', 'deadline/recovery/common.py',
             'deadline/recovery/screen.py', 'deadline/recovery/plan.json',
             'deadline/experiment_plan.json', 'reasoning/benchmark.py', 'reasoning/data.py')
    return {name: digest(project / name) for name in names}


def validate_primary_receipt(value):
    require(value.get('schema') == 'deadline-hf-publication-complete-v1' and
            value.get('complete') is True and value.get('owner') == OWNER and
            value.get('evaluated_and_published_weight_bytes_identical') is True,
            'Successful primary four-artifact publication is required first')
    repositories = value.get('repositories', {})
    require(set(repositories) == set(primary.DESTINATIONS), 'Primary publication lacks its four destinations')
    for key, (kind, _, name) in primary.DESTINATIONS.items():
        row = repositories[key]
        prefix = 'datasets/' if kind == 'dataset' else ''
        require(row.get('repo_id') == OWNER + '/' + name and row.get('repo_type') == kind and
                row.get('url') == 'https://huggingface.co/' + prefix + OWNER + '/' + name and
                row.get('complete') is True and row.get('public_access_verified') is True and
                isinstance(row.get('commit'), str) and re.fullmatch('[0-9a-f]{40}', row['commit']) and
                isinstance(row.get('files'), dict) and bool(row['files']),
                'Primary destination receipt is incomplete or different: ' + key)
    require(REPO_ID not in {row['repo_id'] for row in repositories.values()},
            'Optional destination collides with primary publication')
    return repositories


def require_all_nine_raw(run):
    require(isinstance(run, dict) and not run.get('errors') and isinstance(run.get('manifest'), dict),
            'Evaluation has audit errors or lacks its native manifest')
    benchmarks = run.get('benchmarks', {})
    for name, count in EXPECTED.items():
        row = benchmarks.get(name, {})
        require(not row.get('errors') and row.get('status') == 'complete' and
                row.get('evidence') == 'raw eight-sample predictions verified',
                'All nine raw eight-sample evaluations are required: ' + name)
        validate_metric(row.get('metrics', {}), count)
    return run


def validate_recovery_receipt(receipt):
    require(receipt.get('schema') == 'deadline-base-anchor-merge-v1' and
            receipt.get('complete') is True and receipt.get('alpha_advanced') == .25 and
            receipt.get('alpha_base') == .75 and
            receipt.get('posthoc_after_primary_benchmark_observations') is True,
            'Complete fixed 75/25 post-hoc merge receipt is required')
    for field in ('primary_plan_sha256', 'recovery_plan_sha256'):
        require(isinstance(receipt.get(field), str) and re.fullmatch('[0-9a-f]{64}', receipt[field]),
                'Recovery plan identity is missing')
    output = receipt.get('output', {})
    weights, files = output.get('weight_files', {}), output.get('files', {})
    require(output.get('all_finite') is True and output.get('parameter_dtype') == 'float32' and
            type(output.get('finite_fp32_tensors')) is int and output['finite_fp32_tensors'] == 434,
            'Recovery model lacks the complete finite FP32 receipt')
    expected_shards = {f'model-{number:05d}-of-00006.safetensors' for number in range(1, 7)}
    require(isinstance(weights, dict) and set(weights) == expected_shards and isinstance(files, dict) and
            all(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum) and
                files.get(name) == checksum for name, checksum in weights.items()),
            'Exactly six sealed FP32 recovery shards are required')
    required = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
                'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'NOTICE'}
    require(required <= set(files) and all(Path(name).name == name and
            isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
            for name, checksum in files.items()), 'Recovery sidecar inventory is incomplete or unsafe')
    return receipt


def publication_notice(original_bytes):
    require(isinstance(original_bytes, bytes) and original_bytes,
            'Original recovery NOTICE is missing')
    return primary.NOTICE.encode('utf-8') + b'\n' + original_bytes


def _model_loaded_files(files):
    return {name: checksum for name, checksum in files.items()
            if Path(name).suffix in ('.json', '.safetensors', '.bin', '.model', '.jinja') or name == 'merges.txt'}


def validate_state_path(root, recovery, state):
    """Refuse writes into inputs or primary publisher state, even on failure."""
    root, recovery, state = (Path(path).resolve() for path in (root, recovery, state))
    protected = (root / 'hf/publication', root / 'publication-metadata',
                 root / 'runs', root / 'results', root / 'artifacts',
                 root / 'recovery', recovery)
    require(state != root and all(not state.is_relative_to(path) and
            not path.is_relative_to(state) for path in protected),
            'Optional publisher state must be separate from primary and evaluated model artifacts')
    return state


def prepare(root, recovery, state, primary_receipt):
    root, recovery, state = (Path(path).resolve() for path in (root, recovery, state))
    state = validate_state_path(root, recovery, state)
    files = Files()
    primary_receipt = Path(primary_receipt).resolve()
    primary_sha = files.hash(primary_receipt)
    primary_repositories = validate_primary_receipt(primary.read(primary_receipt))
    merge_path = recovery / 'complete.json'
    merge_sha = files.hash(merge_path)
    merge = validate_recovery_receipt(primary.read(merge_path))
    require(merge['primary_plan_sha256'] == files.hash(root / 'experiment_plan.json') and
            merge['recovery_plan_sha256'] == files.hash(root / 'recovery/plan.json'),
            'Primary or recovery plan changed after the evaluated merge')
    _, models = model_inputs(merge_path, verify_roles=())
    model = recovery / 'model'
    require(Path(merge['output']['path']).resolve() == model and
            Path(models['recovery']['path']).resolve() == model and
            models['recovery']['files'] == merge['output']['files'], 'Recovery model path/inventory differs')
    for name, checksum in merge['output']['files'].items():
        files.hash(model / name, checksum)
    require(files.hash(model / 'LICENSE') == primary.LICENSE_SHA256, 'Pinned Qwen Research License differs')
    index = primary.read(model / 'model.safetensors.index.json')
    require(set(index.get('weight_map', {}).values()) == set(merge['output']['weight_files']),
            'Recovery index does not bind the exact six shards')
    require(index.get('metadata', {}).get('total_size') == merge['output'].get('total_tensor_bytes'),
            'Recovery index tensor-byte total differs from the finite merge receipt')
    for name, checksum in merge['inputs']['advanced']['files'].items():
        if name.endswith('.safetensors'):
            require(primary_repositories['advanced_model']['files'].get(name, {}).get('sha256') == checksum,
                    'Recovery advanced lineage differs from published primary weights')
    validate_summary(recovery / 'development')
    files.hash(recovery / 'development/summary.json')
    folders = {role: root / 'results' / role for role in ('original', 'random', 'advanced')}
    folders['recovery'] = recovery / 'benchmarks'
    batches = {}
    # Seal files before parsing/auditing them; a concurrent change is caught by
    # Files.unchanged before any remote commit and before the final receipt.
    for role, folder in folders.items():
        files.hash(folder / 'manifest.json')
        files.hash(folder / 'metrics.json')
        batches[role] = {name: sorted((folder / name).glob('batch_*.json')) for name in EXPECTED}
        for paths in batches[role].values():
            for path in paths:
                files.hash(path)
    runs = {role: require_all_nine_raw(audit_run(folder)) for role, folder in folders.items()}
    matching = compare_coverage(runs, names=tuple(EXPECTED))
    require(matching['matched'], 'All-nine raw recovery comparison is mismatched: ' + '; '.join(matching['errors']))
    candidate_manifest = runs['recovery']['manifest']
    require(Path(candidate_manifest.get('model', '')).resolve() == model and
            candidate_manifest.get('identity') == _model_loaded_files(merge['output']['files']),
            'Evaluated recovery checkpoint differs from exact merge output bytes')
    for role, run in runs.items():
        for name in EXPECTED:
            records = []
            for path in batches[role][name]:
                records.extend(read_rows(path))
            ids = record_ids(records)
            require(ids == run['manifest']['problem_ids'][name] and
                    [row.get('problem_id') for row in records] == ids,
                    'Raw question/order identity differs from frozen manifest: ' + role + '/' + name)
    sources = source_inventory()
    for name, checksum in sources.items():
        files.hash(HERE.parent / name, checksum)
    evaluation = {'schema': 'deadline-optional-recovery-all-nine-evaluation-v1',
        'all_nine_matched_raw': True, 'primary_comparison_unmodified': True,
        'historical_leaderboard_equivalence_verified': False,
        'models': {role: {key: run[key] for key in
            ('benchmarks', 'official_eight_macro', 'supplemented_historical_seven_macro')} for role, run in runs.items()}}
    package = state / 'package'
    package.mkdir(parents=True, exist_ok=True)
    frozen = {}

    def generated(name, data):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        primary.freeze(path, data)
        frozen[name] = path
        return path

    evaluation_path = generated('evaluation_summary.json', primary.canonical(evaluation))
    original_notice = (model / 'NOTICE').read_bytes()
    staged_notice = generated('NOTICE', publication_notice(original_notice))
    notice_identity = {'original_recovery_notice_sha256': files.hash(model / 'NOTICE'),
                       'published_notice_sha256': files.hash(staged_notice),
                       'original_preserved_as': 'provenance/merge-NOTICE.original',
                       'published_notice_policy': 'Required upstream attribution line followed by verbatim original recovery NOTICE; original model directory unchanged'}
    provenance = {'merge_complete.json': merge_path, 'recovery_plan.json': root / 'recovery/plan.json',
        'primary_experiment_plan.json': root / 'experiment_plan.json',
        'primary_publication_complete.json': primary_receipt,
        'merge-NOTICE.original': model / 'NOTICE',
        'LICENSE.source.json': root / 'licenses/Qwen2.5-3B-Instruct.LICENSE.source.json',
        'evaluation_manifest.json': recovery / 'benchmarks/manifest.json',
        'evaluation_metrics.json': recovery / 'benchmarks/metrics.json'}
    for name in ('prepared.json', 'selected_ids.json', 'selected.json', 'summary.json', 'gpu_allocation.json'):
        provenance['development/' + name] = recovery / 'development' / name
    for role in ('base', 'advanced', 'recovery'):
        for name in ('manifest.json', 'complete.json', 'observations.json'):
            provenance['development/' + role + '/' + name] = recovery / 'development' / role / name
    rows = ['| Benchmark | Original p@1 / p@8 | Random p@1 / p@8 | Advanced p@1 / p@8 | Recovery p@1 / p@8 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name in EXPECTED:
        values = [runs[role]['benchmarks'][name]['metrics'] for role in runs]
        rows.append('| ' + name + ' | ' + ' | '.join(f"{100*v['pass_at_1']:.2f}% / {100*v['pass_at_8']:.2f}%" for v in values) + ' |')
    values = [runs[role]['official_eight_macro'] for role in runs]
    rows.append('| Official eight macro | ' + ' | '.join(f"{100*v['pass_at_1']:.2f}% / {100*v['pass_at_8']:.2f}%" for v in values) + ' |')
    advanced_url = primary_repositories['advanced_model']['url']
    advanced_commit = primary_repositories['advanced_model']['commit']
    card = ('---\nlibrary_name: transformers\npipeline_tag: text-generation\n'
        'base_model: Qwen/Qwen2.5-3B-Instruct\nlicense: other\nlicense_name: qwen-research\n'
        f'license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/{primary.BASE_REVISION}/LICENSE\n'
        'tags:\n- reasoning\n- model-merge\n- post-hoc\n---\n\n'
        '# Qwen2.5-3B: fixed base75 / advanced25 post-hoc recovery\n\n'
        '**Built with Qwen.** This is a separate exploratory weight interpolation, not a new SFT run. '
        'It preserves the original random/advanced assignment checkpoints and results.\n\n'
        '`theta_recovery = 0.75 * theta_original_base + 0.25 * theta_advanced_final`, computed in FP32. '
        f'Original parent: Qwen/Qwen2.5-3B-Instruct revision `{primary.BASE_REVISION}`. '
        f'Advanced parent: [{advanced_url}]({advanced_url}/tree/{advanced_commit}), commit `{advanced_commit}`. '
        'Exact parent shard hashes, merge code/plan identities and output hashes are sealed in the provenance. '
        'All six published FP32 weight shards and model/tokenizer sidecars retain their evaluated bytes. '
        'Inference evaluation used BF16; publication performs no weight conversion.\n\n'
        'The recovery was proposed after viewing primary benchmark regressions and long-response diagnostics. '
        'One coefficient was fixed before new generation; there was no alpha sweep. A deterministic source-only '
        '64-question screen passed the fixed teacher-answer-agreement/stopping gate. Teacher agreement is not '
        'independently verified truth. This development screen used one sample and a 4K output limit; it is '
        'separate from the benchmark protocol below.\n\n'
        'All nine benchmark tasks are verified from eight raw samples per question with matched data, ordered IDs, '
        'grader, seed 0, temperature 0.6, top-p 0.95, maximum 32768 new tokens and native 32768 total context. '
        'AIME25 is supplementary and excluded from the official-eight macro.\n\n' + '\n'.join(rows) + '\n\n'
        'These are measured results from one exploratory candidate. They establish neither significance, '
        'general superiority, novelty nor state-of-the-art performance. Historical leaderboard equivalence is '
        'unverified. The whole advanced parent policy combines eligibility, transfer gain and coverage; '
        'eligible-random/coverage-only ablations and repeated seeds were not run.\n\n'
        'The Qwen Research License applies. NOTICE retains the required upstream attribution and the original '
        'recovery notice verbatim; its exact original bytes are separately preserved under provenance.\n')
    card_path = generated('README.md', card.encode('utf-8'))
    modifications = generated('MODIFICATIONS.md', (
        'Built with Qwen.\n\nAll six model*.safetensors shards were modified by fixed FP32 interpolation '
        'of 75% original base and 25% primary advanced full-SFT weights. This interpolation adds no SFT updates. '
        'Serialized evaluated weights/config/tokenizer bytes remain unchanged during publication. The root NOTICE '
        'is staged as required upstream attribution plus the verbatim recovery notice; the original NOTICE is '
        'retained under provenance/merge-NOTICE.original. All source and published hashes are declared.\n').encode())
    paths = {name: model / name for name in merge['output']['files'] if name != 'NOTICE'}
    paths.update({'README.md': card_path, 'NOTICE': staged_notice, 'MODIFICATIONS.md': modifications,
                  'evaluation_summary.json': evaluation_path})
    paths.update({'provenance/' + name: path for name, path in provenance.items()})
    descriptors = {name: primary.fingerprint(path, files) for name, path in sorted(paths.items())}
    for name, descriptor in descriptors.items():
        descriptor['require_lfs'] = name.endswith('.safetensors')
    identity = {'schema': SCHEMA, 'owner': OWNER, 'repo_id': REPO_ID, 'role': 'optional_recovery',
        'merge_receipt_sha256': merge_sha, 'primary_publication_receipt_sha256': primary_sha,
        'evaluation_summary_sha256': files.hash(evaluation_path),
        'model_identity_sha256': json_digest(merge['output']['files']),
        'primary_plan_sha256': merge['primary_plan_sha256'], 'recovery_plan_sha256': merge['recovery_plan_sha256'],
        'alpha_base': .75, 'alpha_advanced': .25, 'posthoc': True, 'new_sft_updates': 0,
        'all_nine_matched_raw': True, 'notice': notice_identity, 'code_sha256': sources,
        'files': descriptors}
    identity['publication_sha256'] = primary.sha(primary.canonical(identity))
    existing = state / 'publication_plan.json'
    require(not existing.exists() or primary.read(existing) == identity,
            'Optional publication inputs changed; retain existing state and reconcile manually')
    manifest_path = generated(MANIFEST, primary.canonical(identity))
    payload = {'repo_id': REPO_ID, 'repo_type': 'model', 'role': 'optional_recovery',
               'files': dict(descriptors), 'local_paths': {name: str(path.resolve()) for name, path in paths.items()}}
    payload['files'][MANIFEST] = dict(primary.fingerprint(manifest_path, files), require_lfs=False)
    payload['local_paths'][MANIFEST] = str(manifest_path.resolve())
    files.unchanged()
    return identity, payload, files


def publish(root, recovery, state, *, expected_owner, primary_receipt, hub=None):
    primary.compute_only()
    require(expected_owner == OWNER, 'Expected owner must explicitly equal adversarialbandit')
    state = validate_state_path(root, recovery, state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hub = primary.Hub() if hub is None else hub
        require(hub.whoami() == OWNER, 'Authenticated account differs from expected owner')
        identity, payload, files = prepare(root, recovery, state, primary_receipt)
        require(payload['repo_id'] == REPO_ID and payload['repo_type'] == 'model' and
                payload.get('role') == 'optional_recovery', 'Optional destination identity differs')
        plan_path, journal_path = state / 'publication_plan.json', state / 'journal.json'
        if plan_path.exists():
            require(primary.read(plan_path) == identity, 'Optional publication identity changed')
        else:
            primary.write(plan_path, identity)
        journal = primary.read(journal_path) if journal_path.exists() else {
            'publication_sha256': identity['publication_sha256'], 'repo_id': REPO_ID, 'record': {}}
        require(journal.get('publication_sha256') == identity['publication_sha256'] and
                journal.get('repo_id') == REPO_ID, 'Optional journal identity differs')
        record = journal['record']
        snapshot = primary.inspect(hub, payload, record)
        if snapshot is None:
            hub.create(REPO_ID, 'model')
            record['created'] = True
            primary.write(journal_path, journal)
            snapshot = primary.inspect(hub, payload, record)
            require(snapshot is not None, 'Created optional repository is not observable; reconcile before retry')
        if MANIFEST not in snapshot['files']:
            files.unchanged()
            committed = hub.commit(REPO_ID, 'model', {MANIFEST: payload['local_paths'][MANIFEST]}, snapshot['commit'])
            snapshot = primary.inspect(hub, payload, record)
            require(snapshot is not None and snapshot['commit'] == committed and MANIFEST in snapshot['files'],
                    'Optional manifest commit was not independently verified')
        missing = [name for name in payload['files'] if name not in snapshot['files']]
        small = [name for name in missing if not payload['files'][name]['require_lfs']]
        for names in ([small] if small else []) + [[name] for name in missing if name not in small]:
            files.unchanged()
            journal['active_files'] = names
            primary.write(journal_path, journal)
            committed = hub.commit(REPO_ID, 'model', {name: payload['local_paths'][name] for name in names}, snapshot['commit'])
            snapshot = primary.inspect(hub, payload, record)
            require(snapshot is not None and snapshot['commit'] == committed and
                    all(name in snapshot['files'] for name in names), 'Optional content commit was not verified')
            record['last_verified_commit'] = committed
            primary.write(journal_path, journal)
        public = primary.inspect(hub, payload, record, public=True)
        require(public is not None and set(payload['files']) <= set(public['files']),
                'Optional public repository is incomplete')
        proof = {name: primary.verify_file(public['files'][name], expected) for name, expected in payload['files'].items()}
        final = primary.inspect(hub, payload, record, public=True)
        require(final is not None and final['commit'] == public['commit'], 'Optional public head changed before completion')
        files.unchanged()
        receipt = {'schema': 'deadline-optional-recovery-publication-complete-v1', 'complete': True,
            'owner': OWNER, 'repo_id': REPO_ID, 'url': 'https://huggingface.co/' + REPO_ID,
            'commit': public['commit'], 'public_access_verified': True, 'files': proof,
            'publication_sha256': identity['publication_sha256'],
            'merge_receipt_sha256': identity.get('merge_receipt_sha256'),
            'evaluation_summary_sha256': identity.get('evaluation_summary_sha256'),
            'evaluated_and_published_weight_bytes_identical': True,
            'primary_four_artifacts_unchanged': True,
            'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        primary.write(state / 'publication_complete.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('deadline'))
    parser.add_argument('--recovery', type=Path, default=Path('deadline/recovery-runs/base-anchor-75-25-20261020'))
    parser.add_argument('--state', type=Path, default=Path('deadline/hf/optional-recovery-publication'))
    parser.add_argument('--primary-receipt', type=Path, default=Path('deadline/hf/publication/publication_complete.json'))
    parser.add_argument('--expected-owner', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=5400)
    args = parser.parse_args()
    require(60 <= args.timeout_seconds <= 7200, 'Optional publication timeout must be 60..7200 seconds')
    args.state = validate_state_path(args.root, args.recovery, args.state)
    def expired(_signum, _frame):
        raise TimeoutError('Optional publication time budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_seconds)
    try:
        if args.prepare_only:
            primary.compute_only()
            require(args.expected_owner == OWNER, 'Unexpected owner')
            identity, _, _ = prepare(args.root, args.recovery, args.state, args.primary_receipt)
            result = {'staged': True, 'uploaded': False, 'publication_sha256': identity['publication_sha256']}
        else:
            result = publish(args.root, args.recovery, args.state, expected_owner=args.expected_owner,
                             primary_receipt=args.primary_receipt)
        print(json.dumps({key: result[key] for key in ('complete', 'url', 'staged', 'uploaded') if key in result}), flush=True)
    except Exception as error:
        reason = str(error) if isinstance(error, PublicationError) else None
        primary.write(args.state / 'failure.json', {'complete': False, 'error_type': type(error).__name__,
            'reason': reason, 'at_utc': datetime.now(timezone.utc).isoformat(),
            'instruction': 'Retain optional state and retry identical inputs; ambiguous/conflicting destinations require reconciliation. Primary publication is unaffected.'})
        raise SystemExit('Optional publication incomplete (' + type(error).__name__ + '). ' +
                         ((reason + '. ') if reason else '') + 'No success inferred.') from None
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
