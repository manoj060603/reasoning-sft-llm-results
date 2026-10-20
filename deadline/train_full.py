"""Single-GPU full-parameter SFT from an immutable, already-tokenized cache.

This runner never selects data, truncates targets, starts jobs, or uploads models.
Its loss is the mean supervised-token CE for each example, averaged over the
examples in an optimizer update. Metadata must explicitly declare any upstream
random-arm prefix truncation. Advanced targets must remain complete.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def update_schedule(rows, accumulation, steps):
    require(type(rows) is int and rows > 0, 'row count must be positive')
    require(type(accumulation) is int and accumulation > 0, 'accumulation must be positive')
    updates = math.ceil(rows / accumulation)
    require(steps == updates, f'exactly one pass requires {updates} updates, not {steps}')
    return dict(unique_examples=rows, optimizer_updates=updates,
                example_presentations=rows, padding_presentations=0,
                full_update_examples=accumulation,
                final_update_examples=rows % accumulation or accumulation)


def fixed_order(rows, seed=42):
    order = list(range(rows))
    random.Random(seed).shuffle(order)
    return order


def validate_cache(cache, meta_path, expected_rows=15000, cutoff=8192):
    """Validate all cached tokens/masks before loading the GPU model."""
    meta = read_json(meta_path)
    require(isinstance(meta, dict), 'cache metadata must be an object')
    expected_hash = meta.get('cache_sha256', meta.get('training_cache_sha256'))
    actual_hash = digest(cache)
    require(expected_hash == actual_hash, 'cache hash is missing or differs from cache metadata')
    declared_rows = meta.get('rows', meta.get('expected_rows', meta.get('count')))
    require(declared_rows == expected_rows, 'cache metadata must declare the expected row count')
    require(meta.get('cutoff') == cutoff, 'cache metadata cutoff differs from the training cutoff')
    role = meta.get('dataset_role')
    require(role in ('random', 'advanced'), 'dataset_role must be random or advanced')
    policy = meta.get('truncation_policy')
    require(policy in ('none', 'prefix_truncate'), 'declare none or prefix_truncate truncation_policy')
    truncated = meta.get('truncated_rows')
    require(type(truncated) is int and 0 <= truncated <= expected_rows,
            'truncated_rows must be an explicit valid count')
    require(policy != 'none' or truncated == 0, 'none truncation policy requires zero truncated rows')
    require(role != 'advanced' or (policy == 'none' and truncated == 0),
            'advanced cache must preserve complete original targets')
    rows, ids, questions = [], set(), set()
    with Path(cache).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'cache line {line_number}: blank lines are not permitted')
            row = json.loads(line)
            require(isinstance(row, dict), f'cache line {line_number}: expected an object')
            identifier, question = row.get('id'), row.get('question_sha256')
            require(isinstance(identifier, str) and identifier and identifier not in ids,
                    f'cache line {line_number}: missing or duplicate id')
            require(isinstance(question, str) and len(question) == 64 and
                    all(x in '0123456789abcdef' for x in question) and question not in questions,
                    f'cache line {line_number}: invalid or duplicate question_sha256')
            require(isinstance(row.get('fold'), (str, int)) and type(row.get('fold')) is not bool,
                    f'cache line {line_number}: fold must be declared')
            tokens, labels = row.get('input_ids'), row.get('labels')
            require(isinstance(tokens, list) and isinstance(labels, list) and
                    2 <= len(tokens) == len(labels) <= cutoff,
                    f'cache line {line_number}: invalid token/label lengths; no implicit truncation')
            require(all(type(x) is int and x >= 0 for x in tokens),
                    f'cache line {line_number}: token IDs must be nonnegative integers')
            require(all(type(x) is int for x in labels), f'cache line {line_number}: invalid labels')
            require(labels[0] == -100, f'cache line {line_number}: prompt must start masked')
            active = [i for i, x in enumerate(labels) if x != -100]
            require(active and active[0] > 0 and active == list(range(active[0], len(labels))),
                    f'cache line {line_number}: require one masked prompt followed by a supervised response')
            require(all(labels[i] == tokens[i] for i in active),
                    f'cache line {line_number}: labels must preserve the original cached token IDs')
            ids.add(identifier)
            questions.add(question)
            rows.append(row)
    require(len(rows) == expected_rows, f'expected {expected_rows} rows, found {len(rows)}')
    return rows, meta, dict(cache_sha256=actual_hash, cache_meta_sha256=digest(meta_path),
                           rows=len(rows), max_length=max(len(r['input_ids']) for r in rows),
                           supervised_tokens=sum(sum(x != -100 for x in r['labels']) for r in rows),
                           id_order_sha256=json_digest([r['id'] for r in rows]),
                           truncation_policy=policy, truncated_rows=truncated, dataset_role=role)


class Consumption:
    """Record actual loss forwards, independently of data-loader prefetching."""
    def __init__(self, order, consumed=None):
        self.order = list(order)
        require(sorted(self.order) == list(range(len(order))), 'sampler order must be an exact permutation')
        self.indices = list(consumed or [])
        require(self.indices == self.order[:len(self.indices)] and len(self.indices) <= len(self.order),
                'resume consumption does not match the frozen data order')

    def consume(self, index):
        require(len(self.indices) < len(self.order), 'extra presentation after the complete one-pass dataset')
        require(type(index) is int and index == self.order[len(self.indices)],
                'actual example order differs from the frozen one-pass sampler')
        self.indices.append(index)

    def receipt(self, step, accumulation):
        require(len(self.indices) == min(step * accumulation, len(self.order)),
                'consumed row count differs from the completed optimizer updates')
        return dict(completed_updates=step, consumed_indices=self.indices,
                    order_sha256=json_digest(self.order), example_presentations=len(self.indices),
                    unique_indices=len(set(self.indices)), padding_presentations=0)

    def verify_complete(self):
        require(self.indices == self.order and len(set(self.indices)) == len(self.order),
                'every unique training row must be used exactly once')


def weight_files(path):
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if index.exists():
        mapping = read_json(index).get('weight_map', {})
        require(isinstance(mapping, dict) and mapping, 'empty model shard index')
        names = sorted(set(mapping.values()))
        require(all(isinstance(x, str) and Path(x).name == x for x in names), 'invalid model shard filename')
        files = [path / name for name in names]
    else:
        files = [path / 'model.safetensors']
    require(all(p.is_file() and p.stat().st_size > 0 for p in files), 'model weights/shards are incomplete')
    return files


def base_identity(base):
    base = Path(base)
    require(base.is_dir(), '--base must be an existing local cached model directory')
    files = weight_files(base)
    require((base / 'config.json').is_file(), 'base model configuration is missing')
    files += [p for p in base.iterdir() if p.is_file() and p not in files and
              (p.suffix in ('.json', '.model', '.jinja') or p.name in ('merges.txt', 'vocab.txt'))]
    return {p.name: digest(p) for p in sorted(files)}


def resume_receipt(checkpoint, binding, total, accumulation, order):
    checkpoint = Path(checkpoint)
    require(checkpoint.is_dir(), 'explicit resume checkpoint does not exist')
    weight_files(checkpoint)
    for filename in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'consumption.json'):
        require((checkpoint / filename).is_file(), f'incomplete resume checkpoint: missing {filename}')
    state, receipt = read_json(checkpoint / 'trainer_state.json'), read_json(checkpoint / 'consumption.json')
    require(receipt.get('binding_sha256') == binding, 'resume checkpoint belongs to another data/model/recipe')
    step = state.get('global_step')
    require(type(step) is int and step > 0 and receipt.get('completed_updates') == step,
            'resume optimizer step differs from the consumption receipt')
    require(receipt.get('order_sha256') == json_digest(order), 'resume sampler permutation changed')
    consumption = Consumption(order, receipt.get('consumed_indices'))
    require(len(consumption.indices) == min(step * accumulation, total), 'resume presentation count is invalid')
    return consumption, step


def verify_saved_model(path, expected_names):
    """Inspect all serialized tensors on CPU inside the compute allocation."""
    import torch
    from safetensors import safe_open
    path = Path(path)
    files, keys, tensors = weight_files(path), set(), 0
    mapping = read_json(path / 'model.safetensors.index.json')['weight_map'] if (path / 'model.safetensors.index.json').exists() else None
    for file in files:
        with safe_open(str(file), framework='pt', device='cpu') as stored:
            for key in stored.keys():
                require(key not in keys, f'duplicate saved tensor {key}')
                require(mapping is None or mapping.get(key) == file.name, f'shard index mismatch at {key}')
                tensor = stored.get_tensor(key)
                if key in expected_names:
                    require(tensor.dtype == torch.float32, f'final trained parameter {key} is not FP32')
                if tensor.is_floating_point():
                    require(bool(torch.isfinite(tensor).all()), f'nonfinite final tensor {key}')
                keys.add(key)
                tensors += 1
                del tensor
    require(mapping is None or set(mapping) == keys, 'model shard index has missing or extra tensor keys')
    require(set(expected_names).issubset(keys), 'final model is missing trained parameter tensors')
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        require((path / filename).is_file(), f'final model missing {filename}')
    return dict(tensors=tensors, all_finite=True, parameter_dtype='float32',
                parameter_names_verified=len(expected_names),
                files={p.name: digest(p) for p in files},
                model_index_sha256=digest(path / 'model.safetensors.index.json') if mapping else None)


def trainer_class(trainer_base, torch, consumption, order):
    """Expose the small Trainer adaptation for a native CPU accumulation check."""
    class FixedSampler(torch.utils.data.Sampler):
        def __iter__(self):
            return iter(order)

        def __len__(self):
            return len(order)

    class FullTrainer(trainer_base):
        def _get_train_sampler(self, train_dataset=None):
            return FixedSampler()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Ignore HF's accumulated token denominator. Each sequence has its
            # own supervised-token mean; Trainer then averages the examples.
            index = int(inputs.pop('row_index').item())
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            require(loss.numel() == 1 and bool(torch.isfinite(loss.detach())), 'nonfinite scalar training loss')
            consumption.consume(index)
            return (loss, outputs) if return_outputs else loss

    return FullTrainer


def check_fp32_state(model, optimizer=None, require_moments=False):
    """Require FP32 master parameters and Adam moments under BF16 autocast."""
    import torch
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    require(parameters and all(parameter.dtype == torch.float32 for parameter in parameters),
            'every trainable parameter must use FP32 storage; BF16 autocast is compute-only')
    require(not require_moments or optimizer is not None, 'optimizer is required to verify FP32 Adam moments')
    moments = 0
    if optimizer is not None:
        underlying = getattr(optimizer, 'optimizer', optimizer)
        for state in underlying.state.values():
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if name in state:
                    require(state[name].dtype == torch.float32, f'Adam {name} must use FP32 storage')
                    moments += 1
        require(not require_moments or moments > 0, 'Adam moments were not initialized after the optimizer update')
    return dict(trainable_parameter_tensors=len(parameters), fp32_moment_tensors=moments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'cache', 'cache-meta', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1875)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--resume', type=Path, help='explicit complete HF checkpoint; no automatic retry')
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_JOB_PARTITION', '').startswith('login'),
            'training, full hashing and tensor scans must run inside a compute allocation')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'this declared runner supports exactly one GPU process')
    require(args.accumulation == 8 and args.steps == 1875, 'frozen recipe requires accumulation8 and1875 updates')
    rows, meta, cache_audit = validate_cache(args.cache, args.cache_meta)
    schedule = update_schedule(len(rows), args.accumulation, args.steps)
    identity = base_identity(args.base)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from reasoning import memory_ce
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one assigned CUDA GPU is required')
    require(importlib.metadata.version('transformers') == '4.57.3', 'validated Trainer semantics require Transformers4.57.3')
    require(torch.cuda.is_bf16_supported(), 'BF16 training is required')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate', 'safetensors')}
    recipe = dict(max_steps=args.steps, per_device_batch=1, accumulation=args.accumulation,
                  learning_rate=2e-6, lr_scheduler_type='cosine', warmup_ratio=.03,
                  weight_decay=.01, max_grad_norm=1., bf16=True, gradient_checkpointing=True,
                  attention='sdpa', seed=42, data_seed=42, cutoff=8192,
                  loss='per-example mean of shifted supervised-token CE; mean across update examples',
                  full_parameter=True, optimizer='adamw_torch', save_steps=256, save_total_limit=2,
                  loss_kernel='reasoning.memory_ce; BF16 logits and 128-token FP32 CE chunks',
                  parameter_dtype='float32', compute_dtype='bfloat16_autocast',
                  optimizer_moment_dtype='float32',
                  clipping='finite-checked callback; native Trainer clipping disabled to fail before optimizer update')
    order = fixed_order(len(rows))
    manifest = dict(schema='deadline-full-sft-v1', base=str(args.base.resolve()), base_identity=identity,
                    cache=cache_audit, cache_metadata=meta, recipe=recipe, schedule=schedule, versions=versions,
                    source_hashes={'train_full.py': digest(__file__), 'memory_ce.py': digest(memory_ce.__file__)},
                    order_sha256=json_digest(order))
    binding = json_digest(manifest)
    out = args.output.resolve()
    if args.resume:
        require(out.is_dir() and (out / 'manifest.json').is_file(), 'resume output has no original manifest')
        require(read_json(out / 'manifest.json') == manifest, 'resume manifest differs from original model/cache/code/recipe')
        require(not (out / 'complete.json').exists(), 'completed output cannot be retrained or overwritten')
        require(args.resume.resolve().parent == out, 'resume checkpoint must belong to this output directory')
        consumption, starting_step = resume_receipt(args.resume, binding, len(rows), args.accumulation, order)
    else:
        require(not out.exists() or not any(out.iterdir()), 'nonempty output requires an explicit valid --resume checkpoint')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'manifest.json', manifest)
        consumption, starting_step = Consumption(order), 0
    set_seed(42)
    memory_ce.install()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    eos = tokenizer.eos_token_id
    require(type(eos) is int, 'base tokenizer must define an EOS token')
    missing_eos = sum(eos not in [x for x in row['labels'] if x != -100] for row in rows)
    require(cache_audit['dataset_role'] != 'advanced' or missing_eos == 0,
            'advanced cache must supervise EOS in every complete original response')
    require(missing_eos <= cache_audit['truncated_rows'], 'missing supervised EOS exceeds declared truncated rows')
    write_json(out / 'cache_audit.json', {**cache_audit, 'supervised_eos_missing_rows': missing_eos,
                                         'eos_token_id': eos, 'schedule': schedule})
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
    require(not hasattr(model, 'peft_config'), 'full-parameter training cannot start from an adapter wrapper')
    for parameter in model.parameters():
        require(parameter.requires_grad, 'all original model parameters must be trainable')
    check_fp32_state(model)
    require(all(max(r['input_ids']) < model.config.vocab_size for r in rows), 'cache token exceeds model vocabulary')
    model.config.use_cache = False
    expected_names = [name for name, _ in model.named_parameters()]

    class CachedRows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            return dict(input_ids=row['input_ids'], labels=row['labels'], row_index=index)

    def collate(batch):
        require(len(batch) == 1, 'per-example normalization requires microbatch size one')
        row = batch[0]
        ids = torch.tensor([row['input_ids']], dtype=torch.long)
        return dict(input_ids=ids, labels=torch.tensor([row['labels']], dtype=torch.long),
                    attention_mask=torch.ones_like(ids), row_index=torch.tensor([row['row_index']]))

    class Guard(TrainerCallback):
        def on_pre_optimizer_step(self, training_args, state, control, model=None, optimizer=None, **kwargs):
            check_fp32_state(model, optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            value = float(norm.detach())
            require(math.isfinite(value), 'nonfinite gradient norm before optimizer update')
            with (out / 'finite_updates.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=state.global_step + 1, gradient_norm=value,
                                             actual_presentations=len(consumption.indices))) + '\n')

        def on_optimizer_step(self, training_args, state, control, model=None, optimizer=None, **kwargs):
            check_fp32_state(model, optimizer, require_moments=True)

        def on_save(self, training_args, state, control, **kwargs):
            receipt = consumption.receipt(state.global_step, args.accumulation)
            write_json(out / f'checkpoint-{state.global_step}' / 'consumption.json',
                       {**receipt, 'binding_sha256': binding})

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            for name in ('loss', 'grad_norm', 'learning_rate', 'train_loss'):
                if name in (logs or {}):
                    require(math.isfinite(float(logs[name])), f'nonfinite training log {name}')

    training_args = TrainingArguments(output_dir=str(out), overwrite_output_dir=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.accumulation,
        max_steps=args.steps, num_train_epochs=1., learning_rate=2e-6,
        lr_scheduler_type='cosine', warmup_ratio=.03, weight_decay=.01,
        max_grad_norm=0., optim='adamw_torch', bf16=True, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='no', save_strategy='steps', save_steps=256, save_total_limit=2,
        save_safetensors=True, save_only_model=False, logging_steps=1,
        logging_nan_inf_filter=False, report_to=[], push_to_hub=False,
        seed=42, data_seed=42, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_drop_last=False, disable_tqdm=True)
    FullTrainer = trainer_class(Trainer, torch, consumption, order)
    trainer = FullTrainer(model=model, args=training_args, train_dataset=CachedRows(),
                          data_collator=collate, processing_class=tokenizer, callbacks=[Guard()])
    trainer.model_accepts_loss_kwargs = False
    require(trainer.accelerator.num_processes == 1 and trainer.args.n_gpu == 1,
            'unexpected distributed or data-parallel execution')
    write_json(out / 'runtime.json', dict(gpu=torch.cuda.get_device_name(),
        device_count=torch.cuda.device_count(), starting_step=starting_step,
        binding_sha256=binding, versions=versions))
    began = time.monotonic()
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    require(trainer.state.global_step == args.steps, 'training did not reach the fixed final update')
    consumption.verify_complete()
    require(math.isfinite(float(result.training_loss)), 'nonfinite final reported training loss')
    final = out / 'model'
    require(not final.exists(), 'final model directory already exists; refusing overwrite')
    model.config.use_cache = True
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    verified = verify_saved_model(final, expected_names)
    receipt = consumption.receipt(args.steps, args.accumulation)
    write_json(out / 'consumption_final.json', {**receipt, 'binding_sha256': binding})
    trainer.save_state()
    write_json(out / 'complete.json', dict(binding_sha256=binding, steps=args.steps,
        unique_examples=len(rows), example_presentations=len(consumption.indices),
        padding_presentations=0, full_parameter=True, model=str(final),
        final_model_verification=verified, training_loss=float(result.training_loss),
        current_attempt_seconds=time.monotonic() - began,
        peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(complete=True, output=str(out), steps=args.steps), indent=2), flush=True)


if __name__ == '__main__':
    main()
