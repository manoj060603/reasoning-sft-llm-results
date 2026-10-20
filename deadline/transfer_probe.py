"""Train source-only LoRA probes and score disjoint folds by transferable CE gain.

Cache JSONL: {id, input_ids, labels, fold, question_sha256}. Labels must be the
unaltered input token at every supervised position, with a masked prompt prefix
and a complete supervised EOS. The preparation's native completeness audit is
bound through cache_meta.json; this module cannot recover raw target provenance
from token IDs alone. The deadline recipe uses complete 8192-token examples;
4096-token fixtures/runs require an explicit matching cutoff. All model loading
is local-only.

The probe is a selection instrument, never a terminal benchmark candidate. A
fold is scored only by the opposite fold's probe. Gain is recorded UNCLIPPED as
(base_ce - probe_ce) / max(base_ce, 1e-8); the downstream selector owns any
positive-only acquisition rule. No benchmark text or outcomes enter this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable


SCHEMA = "deadline-crossfit-transfer-probe-v1"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_ids(path: Path) -> list[str]:
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or not values:
        raise ValueError("ID file must contain a nonempty JSON list")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("IDs must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("duplicate IDs in ID file")
    return values


def checked_fold(value: Any) -> int:
    if type(value) not in (int, str) or value not in (0, 1, "0", "1"):
        raise ValueError("fold must be 0 or 1")
    return int(value)


def transfer_gain(base_ce: float, probe_ce: float) -> float:
    if not all(math.isfinite(value) and value >= 0 for value in (base_ce, probe_ce)):
        raise ValueError("nonfinite or negative cross entropy")
    gain = (base_ce - probe_ce) / max(base_ce, 1e-8)
    if not math.isfinite(gain):
        raise ValueError("nonfinite transfer gain")
    return gain


class TokenCache:
    """Seekable JSONL cache; retain offsets/identities rather than all token arrays."""

    def __init__(self, path: Path, meta_path: Path, base: Path, cutoff: int = 8192):
        self.path, self.meta_path = Path(path).resolve(), Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text())
        if type(cutoff) is not int or cutoff not in (4096, 8192):
            raise ValueError("probe cutoff must be an explicit 4096 or 8192 tokens")
        self.cutoff = cutoff
        if self.meta.get("cutoff") != self.cutoff:
            raise ValueError(f"this probe requires the audited complete {self.cutoff}-token cache")
        if (self.meta.get("complete_original_targets") is not True or
                self.meta.get("prefix_truncated_count") != 0):
            raise ValueError("cache metadata must attest complete original targets without truncation")
        if Path(self.meta.get("model_path", "")).resolve() != Path(base).resolve():
            raise ValueError("cache and requested base model paths differ")
        self.meta_sha256 = digest(self.meta_path)
        self.rows: dict[str, dict[str, Any]] = {}
        questions: dict[str, str] = {}
        h = hashlib.sha256()
        self.stat = self.path.stat()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                h.update(raw)
                if not raw.strip():
                    raise ValueError("blank line in token cache")
                row = json.loads(raw)
                key, question = row.get("id"), row.get("question_sha256")
                if not isinstance(key, str) or not key or key in self.rows:
                    raise ValueError("missing or duplicate cache ID")
                if not isinstance(question, str) or not re.fullmatch(r"[0-9a-f]{64}", question):
                    raise ValueError("missing or invalid question SHA256")
                if question in questions:
                    raise ValueError("duplicate question identity in cache")
                questions[question] = key
                self.rows[key] = {"offset": offset, "fold": checked_fold(row.get("fold")),
                                  "question_sha256": question,
                                  "length": len(row.get("input_ids", []))}
        self.sha256 = h.hexdigest()
        if self.sha256 != self.meta.get("cache_sha256"):
            raise ValueError("cache SHA256 mismatch")
        if not self.rows:
            raise ValueError("empty token cache")
        if self.meta.get("expected_rows", len(self.rows)) != len(self.rows):
            raise ValueError("cache row count differs from preparation metadata")
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        now = self.path.stat()
        if (now.st_size, now.st_mtime_ns, now.st_ino) != (
                self.stat.st_size, self.stat.st_mtime_ns, self.stat.st_ino):
            raise ValueError("token cache changed during execution")
        if digest(self.meta_path) != self.meta_sha256:
            raise ValueError("cache metadata changed during execution")

    def get(self, key: str, eos_ids: set[int] | None = None,
            vocab_size: int | None = None,
            post_eos_trailer: tuple[int, ...] = ()) -> dict[str, Any]:
        if key not in self.rows:
            raise ValueError(f"unknown cache ID: {key}")
        with self.path.open("rb") as stream:
            stream.seek(self.rows[key]["offset"])
            row = json.loads(stream.readline())
        if row.get("id") != key or row.get("question_sha256") != self.rows[key]["question_sha256"]:
            raise ValueError("cache row identity changed")
        if checked_fold(row.get("fold")) != self.rows[key]["fold"]:
            raise ValueError("cache row fold changed")
        inputs, labels = row.get("input_ids"), row.get("labels")
        if not isinstance(inputs, list) or not isinstance(labels, list):
            raise ValueError("input_ids and labels must be lists")
        if not 2 <= len(inputs) <= self.cutoff or len(inputs) != len(labels):
            raise ValueError("overlength, empty or mismatched token sequence")
        if any(type(token) is not int or token < 0 for token in inputs):
            raise ValueError("invalid input token")
        if vocab_size is not None and any(token >= vocab_size for token in inputs):
            raise ValueError("input token exceeds model vocabulary")
        target_started = False
        for position, (token, label) in enumerate(zip(inputs, labels)):
            if type(label) is not int:
                raise ValueError("invalid label")
            if label == -100:
                if target_started:
                    raise ValueError("masked hole after response begins")
            else:
                if position == 0 or label != token:
                    raise ValueError("labels must preserve tokens after a masked prompt")
                target_started = True
        if labels[0] != -100 or not target_started or labels[-1] == -100:
            raise ValueError("missing masked prompt or complete supervised response")
        if eos_ids is not None and labels[-1] not in eos_ids:
            # Qwen's native chat template emits <|im_end|> followed by a
            # supervised newline. Preserve those bytes/tokens exactly; never
            # strip the newline or append another EOS to satisfy this check.
            trailer_length = len(post_eos_trailer)
            if (not trailer_length or len(labels) <= trailer_length or
                    tuple(labels[-trailer_length:]) != post_eos_trailer or
                    labels[-trailer_length - 1] not in eos_ids):
                raise ValueError("complete supervised EOS/template trailer is missing")
        return row

    def require_fold(self, ids: Iterable[str], fold: int) -> None:
        fold = checked_fold(fold)
        for key in ids:
            if key not in self.rows or self.rows[key]["fold"] != fold:
                raise ValueError("ID absent from declared source fold")

    def provenance(self) -> dict[str, Any]:
        return {"path": str(self.path), "cache_sha256": self.sha256,
                "cache_meta_sha256": self.meta_sha256, "rows": len(self.rows),
                "cutoff": self.cutoff, "preparation_metadata": self.meta}


def model_identity(base: Path) -> dict[str, Any]:
    """Hash local safetensors model bytes, config and tokenizer identity files."""
    base = Path(base).resolve()
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("bounded-head probe currently supports the Qwen2 architecture only")
    index_path = base / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weights = sorted(set(index["weight_map"].values()))
        if any(Path(name).name != name for name in weights):
            raise ValueError("invalid model shard path")
    elif (base / "model.safetensors").is_file():
        weights = ["model.safetensors"]
    else:
        raise ValueError("local safetensors model is required")
    names = weights + [name for name in (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja")
        if (base / name).is_file()]
    files = {}
    for name in sorted(names):
        path = base / name
        before = path.stat()
        files[name] = {"sha256": digest(path), "bytes": before.st_size}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("model file changed while hashing")
    identity = {"path": str(base), "files": files, "model_type": "qwen2"}
    identity["identity_sha256"] = value_digest(identity)
    return identity


def check_disjoint(cache: TokenCache, candidates: list[str], fold: int,
                   probe_receipt: dict[str, Any]) -> None:
    cache.require_fold(candidates, fold)
    if not probe_receipt.get("complete") or probe_receipt.get("schema") != SCHEMA:
        raise ValueError("probe is not a completed training artifact")
    if checked_fold(probe_receipt.get("training_fold")) == checked_fold(fold):
        raise ValueError("same-fold probe scoring is forbidden")
    training_ids = probe_receipt.get("training_ids", [])
    training_questions = set(probe_receipt.get("training_question_sha256", []))
    if not training_ids or len(set(training_ids)) != len(training_ids):
        raise ValueError("invalid probe training identities")
    cache.require_fold(training_ids, probe_receipt["training_fold"])
    if training_questions != {cache.rows[key]["question_sha256"] for key in training_ids}:
        raise ValueError("probe training question identities differ from its cache")
    if set(candidates).intersection(training_ids):
        raise ValueError("candidate overlaps probe training ID")
    if any(cache.rows[key]["question_sha256"] in training_questions for key in candidates):
        raise ValueError("candidate overlaps probe training question")
    if len(training_questions) != len(training_ids):
        raise ValueError("incomplete or duplicate training question identities")


def _packages() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "peft", "safetensors", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _load_model(base: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.config.use_cache = False
    return model.to(device)


def _eos_ids(config) -> set[int]:
    values = config.eos_token_id
    if isinstance(values, int):
        values = [values]
    if not values or any(type(value) is not int for value in values):
        raise ValueError("base configuration has no explicit EOS identity")
    return set(values)


def _post_eos_trailer(base: Path) -> tuple[int, ...]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True,
                                              trust_remote_code=False)
    return tuple(tokenizer.encode("\n", add_special_tokens=False))


def _batch(rows: list[dict[str, Any]], device: str, pad_id: int):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    attention = torch.zeros_like(inputs)
    for index, row in enumerate(rows):
        size = len(row["input_ids"])
        inputs[index, :size] = torch.tensor(row["input_ids"], device=device)
        labels[index, :size] = torch.tensor(row["labels"], device=device)
        attention[index, :size] = 1
    return inputs, labels, attention


def _qwen_backbone(model):
    raw = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(raw.config, "model_type", None) != "qwen2":
        raise ValueError("unsupported architecture")
    return raw.model, raw.lm_head


def batch_sequence_ce(model, inputs, labels, attention, chunk_tokens: int = 128):
    """Per-sequence shifted/masked FP32 CE, projecting only small token chunks.

    Inference-only. The backbone sees padding masks; the vocabulary head receives
    supervised positions alone. No [batch, length, vocabulary] tensor is created.
    """
    import torch
    import torch.nn.functional as F

    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    backbone, head = _qwen_backbone(model)
    hidden = backbone(input_ids=inputs, attention_mask=attention, use_cache=False,
                      return_dict=True).last_hidden_state
    results = []
    for index in range(inputs.shape[0]):
        shifted = labels[index, 1:]
        valid = shifted != -100
        targets = shifted[valid]
        states = hidden[index, :-1][valid]
        count = targets.numel()
        if count == 0:
            raise ValueError("no supervised shifted response tokens")
        total = torch.zeros((), dtype=torch.float64, device=hidden.device)
        for offset in range(0, count, chunk_tokens):
            logits = head(states[offset:offset + chunk_tokens]).float()
            loss = F.cross_entropy(logits, targets[offset:offset + chunk_tokens], reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite scored cross entropy")
            total += loss.double()
            del logits, loss
        value = float(total / count)
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("nonfinite per-sequence cross entropy")
        results.append((value, int(count)))
    return results


def _finite_adapter(adapter_dir: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    weights = adapter_dir / "adapter_model.safetensors"
    count, total_bytes = 0, 0
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite saved adapter tensor: {key}")
            count += 1
            total_bytes += value.numel() * value.element_size()
    if not count:
        raise ValueError("empty saved adapter")
    return {"tensor_count": count, "tensor_bytes": total_bytes,
            "files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                      for path in sorted(adapter_dir.iterdir()) if path.is_file()}}


def _probe_parameter_precision(model) -> dict[str, Any]:
    """PEFT must promote trainable LoRA parameters without promoting the base."""
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ValueError("every trainable LoRA adapter parameter must have FP32 storage")
    if not frozen or any(parameter.dtype != torch.bfloat16 for parameter in frozen):
        raise ValueError("the frozen probe base must retain BF16 parameter storage")
    return {"base_parameter_storage": "torch.bfloat16",
            "adapter_parameter_storage": "torch.float32",
            "trainable_parameter_tensors": len(trainable),
            "frozen_parameter_tensors": len(frozen),
            "optimizer_moments_verified": False}


def _adam_moment_precision(optimizer, trainable) -> dict[str, Any]:
    import torch

    checked = 0
    for parameter in trainable:
        state = optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if value is None or value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"AdamW {name} must exist in FP32 with the adapter parameter shape")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite AdamW {name}")
            checked += 1
    if not checked:
        raise ValueError("no AdamW moments were checked after the first update")
    return {"optimizer_moments_verified": True, "optimizer_moment_storage": "torch.float32",
            "optimizer_moment_tensors": checked, "optimizer_moment_verification_update": 1}


def train_probe(*, base: Path, cache_path: Path, cache_meta: Path, train_ids_path: Path,
                fold: int, probe_id: str, output: Path, steps: int = 64,
                accumulation: int = 8, batch_size: int = 1, lr: float = 5e-5,
                lora_rank: int = 8, seed: int = 42, device: str = "cuda:0",
                cutoff: int = 8192) -> dict[str, Any]:
    if min(steps, accumulation, batch_size, lora_rank) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid probe training configuration")
    base, output = Path(base).resolve(), Path(output).resolve()
    if not probe_id or output == base or base in output.parents:
        raise ValueError("invalid probe name or output inside base")
    cache = TokenCache(cache_path, cache_meta, base, cutoff=cutoff)
    ids = read_ids(train_ids_path)
    cache.require_fold(ids, fold)
    identity = model_identity(base)
    manifest = {"schema": SCHEMA, "purpose": "source-only-selection-probe",
                "implementation_sha256": digest(Path(__file__)),
                "probe_id": probe_id, "training_fold": checked_fold(fold),
                "base": identity, "cache": cache.provenance(), "training_ids": ids,
                "training_ids_file_sha256": digest(train_ids_path),
                "training_question_sha256": [cache.rows[key]["question_sha256"] for key in ids],
                "steps": steps, "accumulation": accumulation, "microbatch_size": batch_size,
                "effective_batch_size": accumulation * batch_size, "lr": lr,
                "lora_rank": lora_rank, "lora_alpha": 2 * lora_rank,
                "lora_dropout": 0.0, "target_modules": list(TARGET_MODULES),
                "optimizer": "AdamW", "weight_decay": 0.0, "schedule": "constant",
                "precision_policy": {"frozen_base_storage": "torch.bfloat16",
                                     "trainable_adapter_storage": "torch.float32",
                                     "adamw_moment_storage": "torch.float32"},
                "objective": "mean microbatch response-token CE; equal microbatch accumulation",
                "seed": seed, "device": device, "packages": _packages()}
    manifest_sha = value_digest(manifest)
    receipt_path = output / "probe_receipt.json"
    if output.exists():
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("complete") and receipt.get("manifest_sha256") == manifest_sha:
                for name, info in receipt["adapter"]["files"].items():
                    if Path(name).name != name or digest(output / "adapter" / name) != info["sha256"]:
                        raise ValueError("completed adapter was modified")
                return receipt
        raise FileExistsError("probe output exists without an identical completed artifact")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "run_manifest.json", manifest)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _load_model(base, device)
    eos = _eos_ids(model.config)
    trailer = _post_eos_trailer(base)
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = min(eos)
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=list(TARGET_MODULES)),
        autocast_adapter_dtype=True)
    runtime_precision = _probe_parameter_precision(model)
    atomic_json(output / "runtime_precision.json", runtime_precision)
    print(json.dumps({"stage": "probe_precision", "probe_id": probe_id,
                      **runtime_precision}), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable probe adapter parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    generator = random.Random(seed)
    order = list(ids)
    generator.shuffle(order)
    position = 0
    started = time.monotonic()
    presented, supervised, updates, seen_training_ids = 0, 0, [], set()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with (output / "training.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss, step_tokens = 0.0, 0
            for _ in range(accumulation):
                keys = []
                for _ in range(batch_size):
                    if position == len(order):
                        generator.shuffle(order)
                        position = 0
                    keys.append(order[position])
                    position += 1
                rows = [cache.get(key, eos, model.config.vocab_size, trailer) for key in keys]
                seen_training_ids.update(keys)
                inputs, labels, attention = _batch(rows, device, pad_id)
                # Probe training uses native CE. At the declared 8K cap this is a
                # bounded native allocation; inference scoring below chunks the
                # head and never materializes full-sequence vocabulary logits.
                loss = model(input_ids=inputs, labels=labels, attention_mask=attention,
                             use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite probe training loss")
                (loss / accumulation).backward()
                step_loss += float(loss.detach()) / accumulation
                step_tokens += int((labels[:, 1:] != -100).sum())
                presented += len(rows)
                del loss, inputs, labels, attention
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1:
                runtime_precision.update(_adam_moment_precision(optimizer, trainable))
                atomic_json(output / "runtime_precision.json", runtime_precision)
                print(json.dumps({"stage": "probe_precision_after_first_update", "probe_id": probe_id,
                                  **runtime_precision}), flush=True)
            if any(not torch.isfinite(parameter).all() for parameter in trainable):
                raise FloatingPointError("nonfinite updated probe adapter")
            supervised += step_tokens
            record = {"update": step, "loss": step_loss, "gradient_norm": float(norm),
                      "supervised_tokens": step_tokens, "presentations": presented, "lr": lr}
            log.write(canonical(record).decode() + "\n")
            log.flush()
            updates.append(record)
            if step == 1 or step % 8 == 0 or step == steps:
                print(json.dumps({"stage": "probe_training", "probe_id": probe_id,
                                  "completed_updates": step, "total_updates": steps,
                                  "presentations": presented,
                                  "elapsed_seconds": round(time.monotonic() - started, 3),
                                  "loss": step_loss, "gradient_norm": float(norm)},
                                 allow_nan=False), flush=True)
    cache.assert_unchanged()
    model.save_pretrained(output / "adapter", safe_serialization=True)
    adapter = _finite_adapter(output / "adapter")
    receipt = {**manifest, "manifest_sha256": manifest_sha, "complete": True,
               "completed_updates": len(updates), "declared_training_rows": len(ids),
               "unique_training_rows": len(seen_training_ids),
               "example_presentations": presented, "supervised_token_presentations": supervised,
               "training_seconds": time.monotonic() - started,
               "runtime_precision": runtime_precision,
               "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                              if device.startswith("cuda") else None),
               "trainable_parameters": sum(parameter.numel() for parameter in trainable),
               "training_log_sha256": digest(output / "training.jsonl"), "adapter": adapter}
    atomic_json(receipt_path, receipt)
    return receipt


class ScoreJournal:
    """Atomic per-batch records, immutable provenance, and a final completion seal."""

    def __init__(self, output: Path, manifest: dict[str, Any], batches: list[list[str]]):
        self.output, self.manifest, self.batches = Path(output), manifest, batches
        flattened = [key for batch in batches for key in batch]
        if not flattened or len(set(flattened)) != len(flattened):
            raise ValueError("empty or duplicate candidate batches")
        self.sha256 = value_digest(manifest)
        if self.output.exists():
            recorded = self.output / "run_manifest.json"
            if not recorded.is_file() or json.loads(recorded.read_text()) != manifest:
                raise ValueError("scoring output has incompatible provenance")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            atomic_json(self.output / "run_manifest.json", manifest)
        (self.output / "parts").mkdir(exist_ok=True)

    def part_path(self, index: int) -> Path:
        return self.output / "parts" / f"{index:08d}.json"

    def _validate(self, index: int, records: list[dict[str, Any]]) -> None:
        if [record.get("id") for record in records] != self.batches[index]:
            raise ValueError("scored batch IDs or order differ")
        for record in records:
            if (record.get("fold") != self.manifest["scoring_fold"] or
                    record.get("probe_id") != self.manifest["probe_id"]):
                raise ValueError("scored batch fold/probe differs")
            if (type(record.get("supervised_tokens")) is not int or
                    record["supervised_tokens"] < 1):
                raise ValueError("invalid supervised token count")
            expected = transfer_gain(record["base_ce"], record["probe_ce"])
            if record.get("gain") != expected:
                raise ValueError("scored batch transfer gain is inconsistent")

    def read(self, index: int) -> list[dict[str, Any]] | None:
        path = self.part_path(index)
        if not path.exists():
            return None
        part = json.loads(path.read_text())
        if part.get("manifest_sha256") != self.sha256:
            raise ValueError("score part provenance differs")
        records = part["records"]
        if part.get("records_sha256") != value_digest(records):
            raise ValueError("score part checksum differs")
        self._validate(index, records)
        return records

    def commit(self, index: int, records: list[dict[str, Any]]) -> None:
        self._validate(index, records)
        prior = self.read(index)
        if prior is not None:
            if prior != records:
                raise ValueError("refusing to overwrite an existing scored batch")
            return
        atomic_json(self.part_path(index), {"manifest_sha256": self.sha256,
                    "records_sha256": value_digest(records), "records": records})

    def finish(self, extra: dict[str, Any]) -> dict[str, Any]:
        prior = self.completed()
        if prior is not None:
            return prior
        destination = self.output / "scores.jsonl"
        temporary = self.output / f"scores.jsonl.tmp-{os.getpid()}"
        count, seen, part_hashes = 0, set(), {}
        try:
            with temporary.open("xb") as stream:
                for index in range(len(self.batches)):
                    records = self.read(index)
                    if records is None:
                        raise ValueError("cannot seal incomplete scoring output")
                    part_hashes[self.part_path(index).name] = digest(self.part_path(index))
                    for record in records:
                        if record["id"] in seen:
                            raise ValueError("duplicate scored candidate")
                        seen.add(record["id"])
                        stream.write(canonical(record) + b"\n")
                        count += 1
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = {"schema": SCHEMA, "complete": True, "manifest_sha256": self.sha256,
                   "records": count, "scores_sha256": digest(destination),
                   "parts_sha256": part_hashes, **extra}
        atomic_json(self.output / "score_receipt.json", receipt)
        return receipt

    def completed(self) -> dict[str, Any] | None:
        path = self.output / "score_receipt.json"
        if not path.exists():
            return None
        receipt = json.loads(path.read_text())
        if not receipt.get("complete") or receipt.get("manifest_sha256") != self.sha256:
            raise ValueError("invalid scoring completion seal")
        expected_parts = {self.part_path(index).name for index in range(len(self.batches))}
        if set(receipt.get("parts_sha256", {})) != expected_parts:
            raise ValueError("incomplete scoring completion seal")
        count = 0
        for index in range(len(self.batches)):
            records = self.read(index)
            if records is None or digest(self.part_path(index)) != receipt["parts_sha256"][self.part_path(index).name]:
                raise ValueError("completed score part changed")
            count += len(records)
        if count != receipt.get("records") or digest(self.output / "scores.jsonl") != receipt.get("scores_sha256"):
            raise ValueError("completed score output changed")
        return receipt


def score_fold(*, base: Path, cache_path: Path, cache_meta: Path, candidate_ids_path: Path,
               fold: int, probe_id: str, probe: Path, output: Path, batch_size: int = 4,
               chunk_tokens: int = 128, device: str = "cuda:0",
               cutoff: int = 8192) -> dict[str, Any]:
    if batch_size < 1 or chunk_tokens < 1:
        raise ValueError("batch and head chunk sizes must be positive")
    base, probe, output = Path(base).resolve(), Path(probe).resolve(), Path(output).resolve()
    if output == probe or probe in output.parents or output == base or base in output.parents:
        raise ValueError("scoring output cannot be inside an input model")
    cache = TokenCache(cache_path, cache_meta, base, cutoff=cutoff)
    ids = read_ids(candidate_ids_path)
    receipt_path = probe / "probe_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    check_disjoint(cache, ids, fold, receipt)
    if receipt.get("probe_id") != probe_id:
        raise ValueError("requested probe ID differs from its receipt")
    identity = model_identity(base)
    if receipt.get("base") != identity:
        raise ValueError("scoring base differs from the probe's exact base")
    if receipt.get("cache", {}).get("cache_sha256") != cache.sha256:
        raise ValueError("scoring cache differs from probe training cache")
    for name, info in receipt["adapter"]["files"].items():
        if Path(name).name != name or digest(probe / "adapter" / name) != info["sha256"]:
            raise ValueError("saved probe adapter changed")
    ids.sort(key=lambda key: (cache.rows[key]["length"], key))
    batches = [ids[offset:offset + batch_size] for offset in range(0, len(ids), batch_size)]
    manifest = {"schema": SCHEMA, "probe_id": probe_id, "scoring_fold": checked_fold(fold),
                "implementation_sha256": digest(Path(__file__)),
                "training_fold": receipt["training_fold"], "probe_receipt_sha256": digest(receipt_path),
                "base": identity, "cache": cache.provenance(), "candidate_ids": ids,
                "candidate_ids_file_sha256": digest(candidate_ids_path),
                "batch_size": batch_size, "head_chunk_tokens": chunk_tokens,
                "loss": "shifted masked CE in FP32; per-sequence response-token mean",
                "gain": "(base_ce-probe_ce)/max(base_ce,1e-8), unclipped",
                "device": device, "packages": _packages()}
    journal = ScoreJournal(output, manifest, batches)
    prior = journal.completed()
    if prior is not None:
        return prior
    missing = [index for index in range(len(batches)) if journal.read(index) is None]
    started = time.monotonic()
    memory = None
    if missing:
        import torch
        from peft import PeftModel

        raw = _load_model(base, device)
        eos = _eos_ids(raw.config)
        trailer = _post_eos_trailer(base)
        pad_id = raw.config.pad_token_id if raw.config.pad_token_id is not None else min(eos)
        model = PeftModel.from_pretrained(raw, str(probe / "adapter"), is_trainable=False,
                                         local_files_only=True)
        model.eval()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        scoring_loop_started = time.monotonic()
        new_rows = 0
        with torch.inference_mode():
            for completed_new, index in enumerate(missing, 1):
                rows = [cache.get(key, eos, model.config.vocab_size, trailer) for key in batches[index]]
                inputs, labels, attention = _batch(rows, device, pad_id)
                with model.disable_adapter():
                    baseline = batch_sequence_ce(model, inputs, labels, attention, chunk_tokens)
                adapted = batch_sequence_ce(model, inputs, labels, attention, chunk_tokens)
                records = []
                for row, (base_ce, n_base), (probe_ce, n_probe) in zip(rows, baseline, adapted):
                    if n_base != n_probe:
                        raise ValueError("base/probe supervised token counts differ")
                    records.append({"id": row["id"], "fold": checked_fold(fold),
                                    "probe_id": probe_id, "base_ce": base_ce,
                                    "probe_ce": probe_ce, "gain": transfer_gain(base_ce, probe_ce),
                                    "supervised_tokens": n_base})
                journal.commit(index, records)
                new_rows += len(records)
                if completed_new == 1 or completed_new % 25 == 0 or completed_new == len(missing):
                    print(json.dumps({"stage": "probe_scoring", "probe_id": probe_id,
                                      "scoring_fold": checked_fold(fold),
                                      "new_batches": completed_new,
                                      "total_new_batches": len(missing),
                                      "completed_batches": len(batches) - len(missing) + completed_new,
                                      "total_batches": len(batches), "new_rows": new_rows,
                                      "scoring_loop_elapsed_seconds": round(
                                          time.monotonic() - scoring_loop_started, 3),
                                      "invocation_elapsed_seconds": round(time.monotonic() - started, 3)},
                                     allow_nan=False), flush=True)
                del inputs, labels, attention
        if device.startswith("cuda"):
            memory = torch.cuda.max_memory_allocated(device)
    cache.assert_unchanged()
    return journal.finish({"probe_id": probe_id, "scoring_fold": checked_fold(fold),
                           "new_batches_this_invocation": len(missing),
                           "scoring_seconds_this_invocation": time.monotonic() - started,
                           "peak_cuda_allocated_bytes_this_invocation": memory})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    score = commands.add_parser("score")
    for command in (train, score):
        command.add_argument("--base", required=True, type=Path)
        command.add_argument("--cache", required=True, type=Path)
        command.add_argument("--cache-meta", required=True, type=Path)
        command.add_argument("--fold", required=True, type=int, choices=(0, 1))
        command.add_argument("--probe-id", required=True)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--cutoff", type=int, choices=(4096, 8192), default=8192,
                             help="must exactly match audited cache metadata; deadline recipe is 8192")
    train.add_argument("--train-ids", required=True, type=Path)
    train.add_argument("--steps", type=int, default=64)
    train.add_argument("--accumulation", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    score.add_argument("--candidate-ids", required=True, type=Path)
    score.add_argument("--probe", required=True, type=Path)
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument("--chunk-tokens", type=int, default=128)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["cache_path"] = args.pop("cache")
    if command == "train":
        args["train_ids_path"] = args.pop("train_ids")
        result = train_probe(**args)
    else:
        args["candidate_ids_path"] = args.pop("candidate_ids")
        result = score_fold(**args)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
