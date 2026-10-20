"""Small receipt and compatibility helpers for the isolated recovery candidate."""
import hashlib
import json
import os
from pathlib import Path


TOKENIZER_SIDECARS = frozenset({
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
    'chat_template.jinja',
})
MODEL_SIDECARS = TOKENIZER_SIDECARS | {'config.json', 'generation_config.json'}
CONFIG_RUNTIME_FIELDS = frozenset({
    '_name_or_path', '_commit_hash', 'transformers_version', 'torch_dtype',
    'dtype', '_attn_implementation_autoset', '_attn_implementation_internal',
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    """Never replace any receipt, including a partial file left by interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def reserve_output(path):
    path = Path(path).resolve()
    require(not path.exists(), 'output already exists; use a new unique directory')
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_compute(cpu_only=False):
    require(bool(os.environ.get('SLURM_JOB_ID')), 'use an existing compute allocation')
    if cpu_only:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible in ('', '-1', 'NoDevFiles'), 'CPU merge requires CUDA_VISIBLE_DEVICES empty')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return {key.lower(): os.environ.get(key) for key in
            ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')}


class VerifiedFiles:
    def __init__(self):
        self.records = {}

    @staticmethod
    def stamp(path):
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def check(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f'missing required input: {path}')
        before = self.stamp(path)
        checksum = digest(path)
        require(before == self.stamp(path), f'input changed during hashing: {path}')
        require(expected is None or checksum == expected, f'input hash mismatch: {path}')
        self.records[str(path)] = {'stamp': before, 'sha256': checksum, 'bytes': before[0]}
        return checksum

    def unchanged(self):
        for path, record in self.records.items():
            require(record['stamp'] == self.stamp(path), f'input changed during recovery: {path}')


def canonical_architecture(config):
    require(config.get('model_type') == 'qwen2', 'only Qwen2 architecture is supported')
    require(config.get('architectures') == ['Qwen2ForCausalLM'], 'unsupported model architecture')
    require(not config.get('quantization_config'), 'quantized input weights are unsupported')
    return {key: value for key, value in config.items() if key not in CONFIG_RUNTIME_FIELDS}


def normalize_operational_config(config, special_ids):
    """Only reconcile verified inactive-window and explicit-prompt BOS/pad metadata."""
    config = dict(config)
    require(config.get('use_sliding_window') is False, 'sliding-window attention is unsupported')
    layers = config.get('num_hidden_layers')
    require(type(layers) is int and layers > 0, 'invalid layer count')
    layer_types = config.get('layer_types', ['full_attention'] * layers)
    require(layer_types == ['full_attention'] * layers, 'non-full-attention layer layout differs')
    require(151643 in special_ids, 'expected Qwen end-of-text special ID is missing')
    for key in ('bos_token_id', 'pad_token_id'):
        require(config.get(key) in (None, 151643), f'unexpected operational token ID: {key}')
        config.pop(key, None)
    require(config.get('sliding_window') in (None, 32768), 'unexpected inactive window metadata')
    config.pop('sliding_window', None)
    # Both fields affect window attention only, explicitly disabled above.
    config.pop('max_window_layers', None)
    config['layer_types'] = layer_types
    return canonical_architecture(config)


def verify_model_compatibility(base, advanced):
    """Expand defaults with one installed version; compare semantic tokenizer state."""
    require_compute()
    from transformers import AutoConfig, AutoTokenizer
    configs = [AutoConfig.from_pretrained(str(path), local_files_only=True,
               trust_remote_code=False).to_dict() for path in (base, advanced)]
    tokenizers = [AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                  trust_remote_code=False) for path in (base, advanced)]
    states = []
    for tokenizer in tokenizers:
        require(tokenizer.is_fast, 'only complete fast-tokenizer backends are supported')
        states.append({
            'backend': json.loads(tokenizer.backend_tokenizer.to_str()),
            'added_vocab': tokenizer.get_added_vocab(),
            'special_tokens': {key: str(value) for key, value in tokenizer.special_tokens_map.items()},
            'special_ids': tokenizer.all_special_ids,
            'chat_template': tokenizer.chat_template,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
        })
    require(states[0] == states[1], 'tokenizer backend, IDs or template differ')
    normalized = [normalize_operational_config(config, states[0]['special_ids']) for config in configs]
    require(normalized[0] == normalized[1], 'architecture/config semantics differ')
    from reasoning.data import SYSTEM
    prompt_cases = ['What is 2 + 2?', 'Simplify α + α.\nUse \\boxed{}.', 'Whitespace:\t a  b\n\n c']
    for question in prompt_cases:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}]
        ids, rendered = [], []
        for tokenizer in tokenizers:
            native = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            require(tokenizer.encode(text, add_special_tokens=False) == native,
                    'native rendered prompt unexpectedly adds special tokens')
            ids.append(native)
            rendered.append(text)
        require(ids[0] == ids[1] and rendered[0] == rendered[1], 'native prompt rendering or token IDs differ')
    generation = []
    for path in (base, advanced):
        value = read_json(Path(path) / 'generation_config.json')
        for key in ('bos_token_id', 'pad_token_id'):
            require(value.get(key) in (None, 151643), f'unexpected generation operational token ID: {key}')
        generation.append({key: item for key, item in value.items()
                           if key not in ('transformers_version', '_from_model_config',
                                          'bos_token_id', 'pad_token_id')})
    require(generation[0] == generation[1], 'generation defaults differ; do not silently choose one')
    operational = ('bos_token_id', 'pad_token_id', 'sliding_window', 'max_window_layers', 'layer_types')
    allowed_differences = {key: {'base': configs[0].get(key), 'advanced': configs[1].get(key)}
                           for key in operational if configs[0].get(key) != configs[1].get(key)}
    return {'architecture_sha256': json_digest(normalized[0]),
            'tokenizer_semantics_sha256': json_digest(states[0]),
            'generation_semantics_sha256': json_digest(generation[0]),
            'tokenizer_backends_identical': True, 'config_semantics_identical': True,
            'generation_defaults_identical_after_validated_operational_ids': True,
            'allowed_operational_config_differences': allowed_differences,
            'native_prompt_parity_cases': len(prompt_cases),
            'operational_rule': 'Only Qwen151643 BOS/pad metadata and disabled full-attention window metadata may differ; prompts are native explicit token IDs. Output copies original-base defaults.'}
