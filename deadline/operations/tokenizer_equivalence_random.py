"""Read-only tokenizer audit in the already running random evaluation allocation."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def main():
    if os.environ.get('SLURM_JOB_ID') != '1208059' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This audit requires existing job1208059 with CUDA hidden')
    root = Path('/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019')
    output = root / 'deadline/operations/tokenizer-equivalence-random.json'
    if output.exists():
        raise FileExistsError('Refusing to replace an existing tokenizer audit receipt')
    import transformers
    import transformers.tokenization_utils_base as tokenizer_source
    import tokenizers
    from transformers import AutoTokenizer
    from reasoning.data import SYSTEM, question, read_rows
    from reasoning.benchmark import record_ids

    base = Path('/home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1')
    trained = root / 'deadline/runs/random/model'
    paths = dict(original=base, random=trained)
    # Omit fix_mistral_regex entirely, as in the evaluator. local_files_only
    # restricts access to existing local assets and changes no tokenization flag.
    loaded = {role: AutoTokenizer.from_pretrained(str(path), local_files_only=True)
              for role, path in paths.items()}
    backend = {role: json.loads(tok.backend_tokenizer.to_str()) for role, tok in loaded.items()}
    components = {}
    for name in sorted(set(backend['original']) | set(backend['random'])):
        components[name] = dict(equal=backend['original'].get(name) == backend['random'].get(name),
            original_sha256=value_hash(backend['original'].get(name)),
            random_sha256=value_hash(backend['random'].get(name)))
    properties = {
        'backend_json': backend['original'] == backend['random'],
        'pre_tokenizer': backend['original'].get('pre_tokenizer') == backend['random'].get('pre_tokenizer'),
        'added_vocab': loaded['original'].get_added_vocab() == loaded['random'].get_added_vocab(),
        'special_tokens': loaded['original'].special_tokens_map == loaded['random'].special_tokens_map,
        'special_token_ids': loaded['original'].all_special_ids == loaded['random'].all_special_ids,
        'chat_template': loaded['original'].chat_template == loaded['random'].chat_template,
    }
    expected = {'amc': 40, 'aime25': 30, 'aime': 30, 'math': 500, 'cn_math_2024': 30,
                'kaoyan': 199, 'minerva': 272, 'olympiadbench': 675, 'gpqa': 198}
    benchmark_receipts, mismatches, total = {}, [], 0
    for benchmark, count in expected.items():
        source = root / 'deadline/artifacts/benchmarks' / benchmark / 'test.jsonl'
        rows = read_rows(source)
        if len(rows) != count:
            raise ValueError(f'Unexpected {benchmark} record count')
        hashes = {role: hashlib.sha256() for role in paths}
        for row, identifier in zip(rows, record_ids(rows), strict=True):
            encoded = {}
            for role, tokenizer in loaded.items():
                rendered = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': question(row)}],
                    tokenize=False, add_generation_prompt=True)
                encoded[role] = (rendered, tokenizer.encode(rendered, add_special_tokens=False))
                hashes[role].update(canonical([identifier, *encoded[role]]) + b'\n')
            if encoded['original'] != encoded['random']:
                mismatches.append(dict(benchmark=benchmark, problem_id=identifier,
                    rendered_equal=encoded['original'][0] == encoded['random'][0],
                    original_tokens=len(encoded['original'][1]), random_tokens=len(encoded['random'][1])))
            total += 1
        benchmark_receipts[benchmark] = dict(records=count, data_sha256=digest(source),
            original_rendered_and_token_ids_sha256=hashes['original'].hexdigest(),
            random_rendered_and_token_ids_sha256=hashes['random'].hexdigest(),
            equal=hashes['original'].digest() == hashes['random'].digest())
    cases = ['1234567890 + 3.1415926535', r'\frac{123}{456} = \boxed{\sqrt{2}}',
             '数学：计算 1＋2。', 'Café cafe\u0301 naïve Ελληνικά عربى', '😀🧮𝛼 ℝ ≤ ≥ ≠',
             ' leading  multiple   spaces ', '\tindent\tvalue\n\nnext\r\nend\n',
             'UPPERlower CamelCase ABCdef 007', '<|im_start|>user\nQuestion<|im_end|>\n', '']
    diagnostics = []
    for text in cases:
        values = {}
        for role, tokenizer in loaded.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            values[role] = dict(ids=ids, decoded=tokenizer.decode(ids, skip_special_tokens=False),
                decoded_skip_special=tokenizer.decode(ids, skip_special_tokens=True))
        diagnostics.append(dict(text=text, equivalent=values['original'] == values['random'],
                                original=value_hash(values['original']), random=value_hash(values['random'])))
    source_path = Path(tokenizer_source.__file__)
    warning_source = inspect.getsource(tokenizer_source.PreTrainedTokenizerBase._patch_mistral_regex)
    tokenizer_metadata = {}
    for role, path in paths.items():
        config = json.loads((path / 'config.json').read_text())
        tokenizer_metadata[role] = dict(path=str(path), tokenizer_class=type(loaded[role]).__name__,
            vocab_size=loaded[role].vocab_size, model_type=config.get('model_type'),
            config_transformers_version=config.get('transformers_version'),
            fix_mistral_regex_attribute=getattr(loaded[role], 'fix_mistral_regex', None),
            files_sha256={name: digest(path / name) for name in (
                'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                'special_tokens_map.json', 'vocab.json', 'merges.txt') if (path / name).is_file()})
    receipt = dict(schema='deadline-tokenizer-equivalence-v1', complete=True,
        equivalent=all(properties.values()) and not mismatches and all(d['equivalent'] for d in diagnostics),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        operation='Read-only tokenization comparison; no model weights loaded or outputs inspected',
        benchmark_outcomes_inspected=False, reference_answers_used=False,
        tokenization_flags_changed=False, scientific_files_modified=False,
        allocation={key: os.environ.get(key) for key in (
            'SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK',
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM')},
        hostname=socket.gethostname(), python=sys.executable, argv=sys.argv,
        script_sha256=digest(__file__), transformers_version=transformers.__version__,
        tokenizers_version=tokenizers.__version__, transformers_source=str(source_path),
        transformers_source_sha256=digest(source_path), mistral_patch_source_sha256=value_hash(warning_source),
        warning_source_url='https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/tokenization_utils_base.py',
        warning_explanation='Local fast vocabulary>100K plus config saved with4.57.3 bypasses the older non-Mistral-model skip; default logs and sets a false flag, while the regex replacement requires explicit/saved true.',
        tokenizer_metadata=tokenizer_metadata, semantic_properties_equal=properties,
        backend_components=components, benchmark_records_compared=total,
        benchmarks=benchmark_receipts, prompt_mismatches=mismatches, diagnostic_cases=diagnostics)
    if total != 1974:
        raise ValueError('Expected all1974 benchmark records')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(receipt=str(output), equivalent=receipt['equivalent'],
        benchmark_records_compared=total, prompt_mismatches=len(mismatches),
        semantic_properties_equal=properties, transformers_version=transformers.__version__,
        allocation=receipt['allocation']), indent=2))
    if not receipt['equivalent']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
