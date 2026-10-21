"""UNADMITTED fresh, bounded training driver; it never selects training data.

This new module is outside the earlier 51-source admission. A caller must first
freeze a plan, an unchanged complete private cache, CPU QA and a current GPU
admission. The main pilot is the first 256 updates of a 1,875-update schedule;
the separately named integration diagnostic exercises accumulation-eight tails.
There is no resume, retry, checkpoint ranking, promotion, evaluation or upload.
Existing preserving hooks and the global causal-loss mapping remain unchanged.

Cache preparation is a separate CPU-stage obligation. Native replay here checks
every supplied original pair, not population selection/decontamination anew.
Model memory success, throughput and scientific efficacy are not established by
this source. Native GA8 integration and exact model/shape admission are required
before a pilot. The short GA8 tail diagnostic does NOT establish a long forward
with Adam moments AND already accumulated gradients resident. A separate native
long-after-Adam diagnostic is unimplemented and mandatory before32K promotion.
Post-step fault detection provides no transactional rollback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, value_digest, write_new_json)
from extension.native_model_admission import (CASES, MODES, ORIGINAL_FILES, REPOSITORY,
    REVISION, TOKENIZER_FILES, VERSIONS, REQUIRED_CHECKS as NATIVE_REQUIRED_CHECKS,
    REQUIRED_CODE as NATIVE_REQUIRED_CODE,
    MemoryTrace, _checkpoint_inventory,
    _host_resources, _project_path, assigned_gpu_identity, assert_finite_model,
    check_parity, require_gpu_allocation, reserve_output, tensor_inventory)
from extension.labeled_feasibility import CRITERION
from reasoning.data import prompt_id

SCHEMA = "extension-preserving-pilot-driver-plan-v1"
CACHE_SCHEMA = "extension-pilot-cache-v1"
REQUIRED_CODE = frozenset({"extension/pilot_driver.py", "tests/test_pilot_driver.py",
    "extension/preserving_trainer.py", "extension/preserving_objectives.py",
    "extension/objectives.py", "extension/preserving_data.py",
    "extension/native_model_admission.py", "extension/cpu_feasibility.py",
    "reasoning/data.py", "reasoning/features.py"}) | NATIVE_REQUIRED_CODE
SOURCE_PINS = frozenset({"preparation", "reserved_source_seal", "benchmark_question_audit",
    "base_model_identity", "native_model_admission", "pool", "features", "reserved_pool_ids",
    "benchmark_pool_ids", "exclusion_receipt"})
REQUIRED_CHECKS = frozenset({"fresh_original", "native_replay", "equal_example_ga8_tail",
    "exact_once_order", "full_horizon_lr_vector", "exact_save_boundaries", "finite_fp32_weights",
    "finite_fp32_moments", "reference_unchanged", "optimizer_isolation", "checkpoint_isolation",
    "checkpoint_reload_parity", "headroom", "global_loss_unchanged", "input_snapshot_unchanged"})


def _integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, name + " must be an integer within its declared bound")
    return value


def _finite(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        name + " must be finite and within its declared bound")
    return float(value)


def _sha(value):
    require(type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        "A lowercase SHA256 identity is required")
    return value


def cosine_factor(step, *, warmup_steps=57, total_steps=1875):
    """Pinned HF cosine LambdaLR convention; update k uses factor(k - 1).

    LambdaLR initializes index0 before the first update, whose LR is therefore
    zero. The scheduler then advances after each optimizer update. This function
    is independent of the number of pilot updates and must never decay at256.
    """
    _integer(step, "scheduler step", 0)
    _integer(warmup_steps, "warmup steps", 0)
    _integer(total_steps, "full schedule steps")
    require(warmup_steps < total_steps and step <= total_steps, "Scheduler index/horizon differs")
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0., .5 * (1. + math.cos(math.pi * progress)))


def update_groups(count, accumulation):
    _integer(count, "presentations")
    _integer(accumulation, "accumulation")
    return [list(range(start, min(start + accumulation, count)))
        for start in range(0, count, accumulation)]


def checkpoint_steps(updates, save_steps=64):
    """One diagnostic checkpoint64 and the final checkpoint, never128/192."""
    _integer(updates, "updates")
    require(save_steps == 64, "Only the prospective diagnostic checkpoint64 is supported")
    return ([64] if updates > 64 else []) + [updates]


def rng_fingerprint():
    """Hash, but do not publish or offer to restore, the current RNG states."""
    import pickle
    import random
    import numpy
    import torch
    return {"python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(numpy.random.get_state(), protocol=4)).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": hashlib.sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()}


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == SCHEMA, "Unknown new driver plan")
    require(plan.get("stage") in ("pilot", "integration_diagnostic"), "Name the bounded stage explicitly")
    require(plan.get("objective") in MODES, "Unknown preserving objective")
    _finite(plan.get("beta"), "beta")
    require(plan["objective"] != "sft" or plan["beta"] == 0, "SFT control declares beta0")
    _integer(plan.get("seed"), "seed", 0)
    require(_finite(plan.get("learning_rate"), "peak LR") > 0, "Peak LR must be positive")
    _finite(plan.get("weight_decay"), "weight decay")
    require(_finite(plan.get("clip_norm"), "clip norm") > 0, "Explicit positive clipping norm required")
    _integer(plan.get("chunk_tokens"), "chunk tokens")
    for name, expected in {"checkpoint_chunks": True, "accumulation": 8, "microbatch": 1,
            "parameter_dtype": "float32", "reference_dtype": "float32", "compute_dtype": "bfloat16",
            "attention": "sdpa", "gradient_checkpointing": True, "use_reentrant": False,
            "epochs": 1, "resume": False, "automatic_downstream": False}.items():
        require(type(plan.get(name)) is type(expected) and plan[name] == expected, "Unsupported driver setting: " + name)
    require(plan.get("schedule") == {"kind": "cosine", "total_steps": 1875,
        "warmup_steps": 57, "num_cycles": .5}, "Use the exact full-horizon cosine prefix")
    exposure = plan.get("exposure", {})
    count = _integer(exposure.get("examples"), "bound example count")
    expected = len(update_groups(count, 8))
    require(exposure.get("updates") == expected, "The exact one-pass tail update count differs")
    if plan["stage"] == "pilot":
        require(count == 2048 and expected == 256 and plan.get("cutoff") == 32768 and
            plan.get("checkpoints") == [64, 256], "Pilot envelope is2048 complete examples/256 updates/[64,256]")
    else:
        require(count in (9, 17) and plan.get("cutoff") == 512 and plan.get("checkpoints") == [expected],
            "Integration diagnostic is a fresh9- or17-example short GA8 tail check with final save only")
    resources = plan.get("resources", {})
    for key, lower in (("wall_time_seconds", 1), ("reserve_seconds", 172800),
                       ("host_cpus", 1), ("host_ram_gib", 128), ("scratch_free_gib", 160)):
        _integer(resources.get(key), key, lower)
    require(resources.get("gpu_type") == "B200" and resources.get("gpus") == 1, "One B200 is required")
    tolerances = plan.get("tolerances", {})
    require(set(tolerances) == {"logit_max_abs", "ce_abs", "kl_abs", "reload_logit_max_abs",
        "headroom_min_bytes", "headroom_fraction"}, "Explicit numerical/headroom limits required")
    for key in tolerances:
        _finite(tolerances[key], key)
    require(type(tolerances["headroom_min_bytes"]) is int and tolerances["headroom_min_bytes"] > 0 and
        all(tolerances[key] > 0 for key in tolerances) and tolerances["headroom_fraction"] < 1,
        "Positive native numerical limits and integral memory headroom are required")
    require(tolerances["headroom_min_bytes"] >= 16 << 30 and tolerances["headroom_fraction"] >= .1 and
        all(tolerances[key] <= upper for key, upper in (("logit_max_abs", .001), ("ce_abs", 1e-5),
            ("kl_abs", 1e-5), ("reload_logit_max_abs", .001))),
        "Do not weaken the admitted numerical bounds or16GiB/10percent memory headroom")
    require(plan.get("optimizer", "adamw_torch") == "adamw_torch", "Only FP32-state native AdamW is supported")
    return plan


def validate_cache(cache, plan, plan_sha256=None):
    """Pure structural/hash checks; native tokenizer replay is a later check."""
    validate_plan(plan)
    require(isinstance(cache, dict) and cache.get("schema") == CACHE_SCHEMA, "Unknown complete private cache")
    _sha(cache.get("plan_sha256"))
    if plan_sha256 is not None:
        require(cache["plan_sha256"] == plan_sha256, "Private cache is bound to a different plan")
    base = cache.get("base", {})
    require(base.get("repo_id") == REPOSITORY and base.get("revision") == REVISION and
        base.get("files") == ORIGINAL_FILES and type(base.get("path")) is str and Path(base["path"]).is_absolute(),
        "Fresh pinned original base file inventory is required")
    serial = cache.get("serialization", {})
    role = "pilot" if plan["stage"] == "pilot" else "diagnostic"
    require(serial.get("context_role") == role and serial.get("cutoff") == plan["cutoff"] and
        serial.get("eos_token_id") == 151645 and serial.get("vocab_size") == 151936 and
        type(serial.get("terminal_text")) is str and serial["terminal_text"] and
        type(serial.get("terminal_suffix")) is list and serial["terminal_suffix"] and
        serial["terminal_suffix"][0] == 151645 and
        all(type(value) is int and 0 <= value < 151936 for value in serial["terminal_suffix"]),
        "Declare unchanged native serialization including EOS and following suffix")
    rows = cache.get("examples")
    require(type(rows) is list and len(rows) == plan["exposure"]["examples"], "Cache example count differs")
    ids, active_lengths = [], []
    for row in rows:
        require(type(row) is dict and set(row) == {"id", "input_ids", "labels", "attention_mask", "original_row", "provenance"},
            "Use complete preserving_data records")
        original, proof = row["original_row"], row["provenance"]
        require(type(original) is dict and set(original) == {"id", "instruction", "input", "output", "system"} and
            all(type(value) is str for value in original.values()) and original["input"] == "" and
            all(original[name].strip() for name in ("instruction", "output", "system")) and
            row["id"] == original["id"] == prompt_id(original["instruction"]), "Original pair identity differs")
        tokens, labels = row["input_ids"], row["labels"]
        require(type(tokens) is list and 2 <= len(tokens) <= serial["cutoff"] and
            all(type(value) is int and 0 <= value < serial["vocab_size"] for value in tokens) and
            type(labels) is list and len(labels) == len(tokens) and
            all(type(value) is int and (value == -100 or 0 <= value < serial["vocab_size"]) for value in labels) and
            type(row["attention_mask"]) is list and all(type(value) is int for value in row["attention_mask"]) and
            row["attention_mask"] == [1] * len(tokens),
            "No truncation, padding, unsupported tokens or incomplete cache is allowed")
        start = next((i for i, value in enumerate(labels) if value != -100), len(tokens))
        suffix = serial["terminal_suffix"]
        require(0 < start < len(tokens) - len(suffix) and labels == [-100] * start + tokens[start:] and
            tokens[-len(suffix):] == suffix, "Native response mask/tail differs")
        require(type(proof) is dict and proof.get("row_id") == row["id"] and
            proof.get("row_sha256") == value_digest(original) and proof.get("prompt_tokens") == start and
            proof.get("total_tokens") == len(tokens) and proof.get("active_tokens") == len(tokens) - start and
            proof.get("prompt_token_ids_sha256") == value_digest(tokens[:start]) and
            proof.get("full_token_ids_sha256") == value_digest(tokens) and proof.get("labels_sha256") == value_digest(labels) and
            proof.get("eos_token_id") == serial["eos_token_id"] and proof.get("native_terminal_suffix") == suffix and
            proof.get("context_role") == role and proof.get("cutoff") == serial["cutoff"] and
            proof.get("complete_original_target") is True and proof.get("truncated") is False and
            proof.get("target_rewritten") is False and proof.get("causal_shift_applied") is False,
            "Unchanged complete response provenance differs")
        ids.append(row["id"])
        active_lengths.append(len(tokens) - start)
    require(len(set(ids)) == len(ids), "A one-pass cache cannot duplicate original question IDs")
    require(cache.get("ordered_ids_sha256") == value_digest(ids) and cache.get("examples_sha256") == value_digest(rows),
        "Ordered presentations or private cache value hash differs")
    require(cache.get("parity_row_id") in ids and
        len(rows[ids.index(cache["parity_row_id"])]["input_ids"]) <= 512,
        "CPU preparation must prebind a short selected row for parity; GPU code chooses no questions")
    for name in ("source_identity", "tokenizer_identity", "exclusion_receipt"):
        require(type(cache.get(name)) is dict and cache[name], "Missing exact cache identity: " + name)
    require(cache["tokenizer_identity"].get("files") == TOKENIZER_FILES, "Tokenizer bytes differ from original")
    pins = cache.get("source_audit", {})
    require(type(pins) is dict and SOURCE_PINS <= set(pins) and
        (plan["stage"] != "pilot" or {"accumulation8_admission", "long_after_adam_admission"} <= set(pins)),
        "Required native preparation/admission evidence is missing")
    for pin in pins.values():
        require(type(pin) is dict and set(pin) == {"path", "sha256"} and type(pin["path"]) is str and pin["path"],
            "Explicit source file path/SHA required")
        _sha(pin["sha256"])
    if plan["stage"] == "integration_diagnostic":
        require(len(set(active_lengths)) > 1, "Tail diagnostic must include unequal target lengths")
    return cache


def validate_joint_shape_admission(receipt, *, plan, code_sha256, maximum_tokens):
    """Require the missing joint long/Adam/accumulated-gradient native envelope.

    This validates a future separately versioned diagnostic's pinned evidence;
    it does not run that diagnostic or infer it from GA2-long + GA8-short runs.
    The currently implemented short diagnostic can never satisfy this schema.
    """
    _integer(maximum_tokens, "actual maximum selected tokens")
    require(maximum_tokens <= 32768 and type(receipt) is dict and
        receipt.get("schema") == "extension-pilot-long-after-adam-admission-v1" and receipt.get("complete") is True and
        receipt.get("accumulation") == 8 and type(receipt.get("completed_optimizer_updates")) is int and
        receipt["completed_optimizer_updates"] >= 2,
        "A separate completed long-after-Adam accumulation8 diagnostic is required")
    require(all(receipt.get("settings", {}).get(name) == plan[name] for name in
        ("objective", "beta", "chunk_tokens", "checkpoint_chunks", "parameter_dtype", "reference_dtype", "compute_dtype", "attention")) and
        all(receipt.get("code_sha256", {}).get(name) == code_sha256[name] for name in REQUIRED_CODE) and
        receipt.get("base", {}).get("repo_id") == REPOSITORY and receipt["base"].get("revision") == REVISION and
        receipt["base"].get("files") == ORIGINAL_FILES and receipt.get("versions") == VERSIONS and
        "B200" in receipt.get("gpu_identity", {}).get("name", ""),
        "Long joint-envelope evidence differs from the actual original/code/objective/runtime path")
    require(all(receipt.get(name) is True for name in ("source_native_serialization_verified", "global_loss_unchanged",
        "finite_weights", "finite_moments", "reference_unchanged")) and
        receipt.get("memory", {}).get("headroom", {}).get("passed") is True,
        "Long joint-envelope receipt lacks successful finite/reference/headroom evidence")
    shape = receipt.get("shape_evidence", {})
    observations = shape.get("observations")
    require(shape.get("scope") == "long_after_adam_accumulation" and
        shape.get("long_context_joint_peak_proven") is True and type(observations) is list and observations,
        "Short-tail or first-Adam evidence cannot establish the joint long-context peak")
    for observation in observations:
        require(type(observation) is dict and all(type(observation.get(name)) is int and observation[name] >= 0 for name in
            ("tokens", "optimizer_updates_before_forward", "student_parameters_with_adam_moments",
             "resident_gradient_parameters_before_forward", "successful_backwards_in_current_update")),
            "Joint-envelope observations require actual integral pre-forward state counts")
    require(type(shape.get("maximum_complete_tokens")) is int and
        shape["maximum_complete_tokens"] == max(value["tokens"] for value in observations) and
        any(value["tokens"] >= maximum_tokens and value["tokens"] <= 32768 and
            value["optimizer_updates_before_forward"] >= 1 and
            value["student_parameters_with_adam_moments"] == 434 and
            value["resident_gradient_parameters_before_forward"] == 434 and
            1 <= value["successful_backwards_in_current_update"] < 8 and
            value.get("forward_complete") is True and value.get("backward_complete") is True and
            value.get("optimizer_update_after_backward_complete") is True for value in observations),
        "No completed sufficiently long forward has both initialized Adam and resident accumulated gradients")
    return receipt


class ConsumptionLedger:
    """Actual loss-forward starts, completed backwards and optimizer boundaries.

    Dataset access/prefetch never calls this ledger. Failed forwards/backwards
    retain a pending presentation and cannot yield a complete receipt. LR values
    here are observations; the independent schedule oracle belongs to the runner.
    """
    def __init__(self, rows, accumulation, *, event_sink=None):
        require(type(rows) is list and rows, "Supply the exact nonempty ordered cache")
        self.rows, self.accumulation = rows, _integer(accumulation, "accumulation")
        self.groups = update_groups(len(rows), accumulation)
        self.presentations, self.updates, self.pending, self.open_update = [], [], None, None
        require(event_sink is None or callable(event_sink), "A journal sink must be callable")
        self.event_sink = event_sink

    def _event(self, name, value):
        if self.event_sink is not None:
            self.event_sink({"event": name, **value})

    def begin(self, row_index):
        _integer(row_index, "row index", 0)
        require(self.pending is None and self.open_update is None and row_index == len(self.presentations) and
            row_index < len(self.rows), "Forward is repeated, skipped, reordered or overlaps an update")
        row = self.rows[row_index]
        tokens, labels, proof = row["input_ids"], row["labels"], row["provenance"]
        require(proof["full_token_ids_sha256"] == value_digest(tokens) and proof["labels_sha256"] == value_digest(labels),
            "Presentation tokens/targets changed")
        self.pending = {"row_index": row_index, "id": row["id"], "total_tokens": len(tokens),
            "active_tokens": sum(value != -100 for value in labels),
            "eos_tokens": sum(value == proof["eos_token_id"] for value in labels),
            "full_token_ids_sha256": proof["full_token_ids_sha256"], "labels_sha256": proof["labels_sha256"],
            "backward_complete": False}
        self._event("forward_started", self.pending)

    def backward_complete(self, row_index):
        require(self.pending is not None and self.pending["row_index"] == row_index,
            "Backward has no matching actual loss forward")
        self.pending["backward_complete"] = True
        self._event("backward_complete", self.pending)
        self.presentations.append(self.pending)
        self.pending = None

    @staticmethod
    def _lrs(values):
        require(type(values) in (list, tuple) and values, "Observed optimizer group learning rates are required")
        return [_finite(value, "observed LR") for value in values]

    def before_update(self, global_step, learning_rates):
        require(self.pending is None and self.open_update is None and
            global_step == len(self.updates) + 1 and global_step <= len(self.groups), "Optimizer boundary is missing/reordered")
        indices = self.groups[global_step - 1]
        require(len(self.presentations) == indices[-1] + 1 and
            [item["row_index"] for item in self.presentations[indices[0]:]] == indices,
            "Update must use exactly the declared successful backward group, including the partial tail")
        self.open_update = {"update": global_step, "indices": indices,
            "ids": [self.rows[index]["id"] for index in indices], "microbatches": len(indices),
            "learning_rates_before": self._lrs(learning_rates)}
        self._event("before_optimizer_update", self.open_update)

    def after_update(self, global_step, learning_rates):
        require(self.open_update is not None and self.open_update["update"] == global_step,
            "Completed optimizer update has no matching boundary")
        observed = self._lrs(learning_rates)
        require(observed == self.open_update["learning_rates_before"],
            "HF optimizer callback must occur before the scheduler advances")
        self.open_update["learning_rates_after"] = observed
        self._event("optimizer_update_complete", self.open_update)
        self.updates.append(self.open_update)
        self.open_update = None

    def finalize(self):
        require(self.pending is None and self.open_update is None and len(self.presentations) == len(self.rows) and
            len(self.updates) == len(self.groups), "An incomplete run cannot yield a complete consumption receipt")
        ids = [value["id"] for value in self.presentations]
        require(ids == [row["id"] for row in self.rows] and len(set(ids)) == len(ids), "Exact unique once-only order differs")
        return {"complete": True, "presentation_ids": ids, "presentation_indices": list(range(len(ids))),
            "presentations": self.presentations, "updates": self.updates, "examples": len(ids),
            "optimizer_updates": len(self.updates), "ordered_ids_sha256": value_digest(ids),
            **{key: sum(value[key] for value in self.presentations) for key in ("total_tokens", "active_tokens", "eos_tokens")}}


def make_pilot_trainer_class(trainer_base=None):
    """Add ordering/consumption only; never change the frozen objective hooks.

    An injected base must already be a preserving Trainer. This seam allows
    tiny native CPU fixtures without loading Qwen or admitting a real run.
    """
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback
    if trainer_base is None:
        from extension.preserving_trainer import make_trainer_class
        trainer_base = make_trainer_class()

    class PilotTrainer(trainer_base):
        def __init__(self, *args, consumption, **kwargs):
            require(isinstance(consumption, ConsumptionLedger), "Supply the actual ordered-consumption ledger")
            self.consumption = consumption
            super().__init__(*args, **kwargs)
            require(self.args.gradient_accumulation_steps == consumption.accumulation and
                self.args.per_device_train_batch_size == 1 and not self.args.dataloader_drop_last and
                self.args.max_steps in (-1, len(consumption.groups)), "Trainer exposure differs from one-pass ledger")
            owner = self

            class Boundaries(TrainerCallback):
                def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
                    owner.consumption.before_update(state.global_step + 1,
                        [float(group["lr"]) for group in optimizer.param_groups])

                def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
                    owner.consumption.after_update(state.global_step + 1,
                        [float(group["lr"]) for group in optimizer.param_groups])
            self.add_callback(Boundaries())

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, **kwargs):
            if model.training and torch.is_grad_enabled():
                index = inputs.get("row_index")
                require(isinstance(index, torch.Tensor) and index.numel() == 1, "Retain the actual source row_index")
                index = int(index.item())
                require(0 <= index < len(self.consumption.rows), "Actual row index is outside the bound cache")
                row = self.consumption.rows[index]
                for name in ("input_ids", "labels", "attention_mask"):
                    actual = inputs.get(name)
                    require(isinstance(actual, torch.Tensor) and actual.dtype == torch.int64 and
                        actual.shape == (1, len(row[name])) and actual.detach().cpu().tolist() == [row[name]],
                        "Actual forward tokens/mask differ from the bound cache: " + name)
                self.consumption.begin(index)
            return super().compute_loss(model, inputs, **kwargs)

        def training_step(self, model, inputs, num_items_in_batch=None):
            index = inputs.get("row_index")
            require(isinstance(index, torch.Tensor) and index.numel() == 1, "Actual batch lacks its row_index")
            result = super().training_step(model, inputs, num_items_in_batch)
            self.consumption.backward_complete(int(index.item()))
            return result

    return PilotTrainer


def _read_pin(root, evidence, pin):
    return evidence.read(_project_path(root, pin["path"]), pin["sha256"])


def _bound_native_input(root, evidence, receipt, key):
    """Resolve an immutable predecessor's actual plan/cache, never a guess."""
    expected = _sha(receipt.get(key))
    matches = [name for name, record in receipt.get("input_files", {}).items()
        if isinstance(record, dict) and record.get("sha256") == expected]
    require(len(matches) == 1, "Native predecessor lacks one exact plan/prepared input binding")
    return evidence.read(_project_path(root, matches[0]), expected)


def load_inputs(root, plan_path, cache_path, qa_receipt, admission_receipt):
    """Bind actual files; this is intentionally incompatible with old smoke QA."""
    root = Path(root).resolve()
    evidence = Evidence()
    paths = {key: _project_path(root, value) for key, value in
        (("plan", plan_path), ("cache", cache_path), ("qa", qa_receipt), ("admission", admission_receipt))}
    values = {key: evidence.read(path) for key, path in paths.items()}
    plan = validate_plan(values["plan"])
    plan_hash, cache_hash = digest(paths["plan"]), digest(paths["cache"])
    cache = validate_cache(values["cache"], plan, plan_hash)
    qa, admission = values["qa"], values["admission"]
    require(qa.get("schema") == "extension-pilot-driver-native-qa-v1" and qa.get("success") is True and
        qa.get("plan_sha256") == plan_hash and qa.get("cache_sha256") == cache_hash and
        type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and
        all(type(qa.get(key)) is int and qa[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and qa.get("gpu_disabled") is True,
        "New plan/cache-bound successful CPU QA is required; older341 checks do not admit this driver")
    codes = qa.get("code_sha256", {})
    require(type(codes) is dict and REQUIRED_CODE <= set(codes), "QA must bind driver, tests and unchanged helpers")
    for name, expected in codes.items():
        code_path = _project_path(root, name)
        require(code_path.suffix in (".py", ".json", ".slurm") and code_path.stat().st_size <= 2 << 20,
            "Only explicit bounded source/configuration files enter code QA")
        evidence.check(code_path, _sha(expected))
    require(admission.get("schema") == "extension-pilot-driver-admission-v1" and admission.get("admitted") is True and
        admission.get("plan_sha256") == plan_hash and admission.get("cache_sha256") == cache_hash and
        admission.get("qa_sha256") == digest(paths["qa"]) and admission.get("code_sha256") == codes,
        "A separate current-job exact-source GPU admission is required")
    base = Path(cache["base"]["path"])
    require(base.is_dir() and not base.is_symlink(), "Pinned original local model folder is unavailable")
    for name, expected in ORIGINAL_FILES.items():
        evidence.check(base / name, expected)
    proofs = {}
    for name, pin in cache["source_audit"].items():
        if name in ("pool", "features"):
            evidence.check(_project_path(root, pin["path"]), pin["sha256"])
        else:
            proofs[name] = _read_pin(root, evidence, pin)
    preparation = proofs["preparation"]
    require(preparation.get("schema") == "extension-pilot-cache-preparation-v1" and preparation.get("complete") is True and
        preparation.get("plan_sha256") == plan_hash and preparation.get("examples_sha256") == cache["examples_sha256"] and
        preparation.get("ordered_ids_sha256") == cache["ordered_ids_sha256"] and
        preparation.get("source_identity") == cache["source_identity"] and
        preparation.get("tokenizer_identity") == cache["tokenizer_identity"] and
        preparation.get("exclusion_receipt_sha256") == value_digest(cache["exclusion_receipt"]),
        "A complete independently prepared cache proof is required")
    input_files = preparation.get("input_files")
    require(type(input_files) is dict and input_files, "Preparation must preserve every actual source/exclusion input SHA")
    for name, record in input_files.items():
        require(type(record) is dict and "sha256" in record, "Malformed preparation input inventory")
        original_inputs = {(base / filename).resolve() for filename in ORIGINAL_FILES}
        candidate = Path(name)
        input_path = candidate.resolve() if candidate.is_absolute() and candidate.resolve() in original_inputs else _project_path(root, name)
        require(type(record.get("bytes")) is int and record["bytes"] == input_path.stat().st_size,
            "Preparation input size differs from the actual source/base file")
        evidence.check(input_path, _sha(record["sha256"]))
    exclusions = cache["exclusion_receipt"]
    require(exclusions.get("schema") == "preserving-pool-exclusions-v1" and
        exclusions.get("id_namespace") == "canonical_pool_prompt_sha256_v1" and
        set(exclusions.get("components", {})) == {"reserved_source_evaluation", "final_benchmark_questions"},
        "Both actual source and all-nine benchmark exclusion components are required")
    audit = cache["source_audit"]
    require(cache["source_identity"].get("pool_sha256") == audit["pool"]["sha256"] and
        cache["source_identity"].get("features_sha256") == audit["features"]["sha256"],
        "Actual full pool/features are not bound to cache source identity")
    pool = evidence.read(_project_path(root, audit["pool"]["path"]), audit["pool"]["sha256"])
    require(type(pool) is list and pool and all(type(row) is dict and type(row.get("id")) is str for row in pool),
        "The pinned complete canonical pool schema is unavailable")
    by_id = {row["id"]: row for row in pool}
    require(len(by_id) == len(pool) and exclusions.get("population_ids_sha256") == value_digest(sorted(by_id)),
        "Canonical source population/unique IDs differ from exclusion proof")
    require(all(row["id"] in by_id and value_digest(row["original_row"]) == value_digest(by_id[row["id"]])
        for row in cache["examples"]), "A cache pair is missing or rewritten relative to its actual canonical source")
    del pool, by_id
    reserved, blocked = proofs["reserved_pool_ids"], proofs["benchmark_pool_ids"]
    for identifiers in (reserved, blocked):
        require(type(identifiers) is list and identifiers == sorted(set(identifiers)) and
            all(type(value) is str and _sha(value) for value in identifiers), "Exclusion IDs must be unique canonical sorted SHAs")
    seal, benchmark = proofs["reserved_source_seal"], proofs["benchmark_question_audit"]
    require(seal.get("schema") == "extension-development-seal-v2" and seal.get("complete") is True and
        seal.get("input_stats_unchanged_at_closeout") is True and
        seal.get("output_sha256", {}).get("reserved_training_pool_ids.json") == audit["reserved_pool_ids"]["sha256"] and
        seal.get("reservation", {}).get("reserved_training_pool_ids_sha256") == value_digest(reserved),
        "Actual repaired development reservation does not bind reserved pool IDs")
    require(benchmark.get("schema") == "extension-final-benchmark-pool-exclusions-v1" and benchmark.get("complete") is True and
        benchmark.get("pool_sha256") == audit["pool"]["sha256"] and benchmark.get("criterion") == CRITERION and
        benchmark.get("scope") == "all_nine_final_benchmarks" and
        set(benchmark.get("benchmark_question_sha256", {})) ==
            {"aime", "aime25", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"} and
        benchmark.get("excluded_pool_ids_sha256") == value_digest(blocked) and
        benchmark.get("excluded_pool_ids_file_sha256") == audit["benchmark_pool_ids"]["sha256"] and
        benchmark.get("answers_or_outcomes_used") is False, "Actual all-nine question-only audit is missing")
    for name, expected in benchmark["benchmark_question_sha256"].items():
        evidence.check(root / "deadline/artifacts/benchmarks" / name / "test.jsonl", _sha(expected))
    require(proofs["exclusion_receipt"] == exclusions, "Actual exclusion receipt bytes differ from the embedded value")
    excluded = set()
    for name, source, ids in (("reserved_source_evaluation", "reserved_source_seal", reserved),
                             ("final_benchmark_questions", "benchmark_question_audit", blocked)):
        component = exclusions["components"][name]
        scope = "all_reserved_source_splits" if name == "reserved_source_evaluation" else "all_nine_final_benchmarks"
        require(component.get("audit_receipt_sha256") == audit[source]["sha256"] and
            component.get("pool_ids") == ids and component.get("scope") == scope and
            component.get("criterion_sha256") == value_digest(CRITERION) and
            _sha(component.get("question_inventory_sha256")),
            "Exclusions are not bound to the actual source audit")
        excluded.update(component["pool_ids"])
    require(not excluded.intersection(row["id"] for row in cache["examples"]), "Reserved/evaluation-overlapping row entered training")
    identity = proofs["base_model_identity"]
    require(identity.get("path") == str(base) and identity.get("model_type") == "qwen2" and
        {name: record.get("sha256") for name, record in identity.get("files", {}).items()} == ORIGINAL_FILES and
        identity.get("identity_sha256") == value_digest({key: value for key, value in identity.items() if key != "identity_sha256"}),
        "Actual pinned base identity receipt differs")
    smoke = proofs["native_model_admission"]
    require(smoke.get("schema") == "extension-native-model-admission-v1" and smoke.get("complete") is True and
        set(smoke.get("case_receipts", {})) == set(CASES) and smoke.get("input_snapshot_unchanged") is True,
        "All-eight actual-model objective/length admission is mandatory")
    require(all(smoke.get("code_sha256", {}).get(name) == codes[name] for name in NATIVE_REQUIRED_CODE),
        "The earlier actual-model admission did not test these unchanged native hooks")
    smoke_plan = _bound_native_input(root, evidence, smoke, "plan_sha256")
    smoke_cache = _bound_native_input(root, evidence, smoke, "prepared_sha256")
    settings = smoke_plan.get("settings", {})
    require(all(settings.get(old) == plan[new] for old, new in
        (("student_dtype", "parameter_dtype"), ("reference_dtype", "reference_dtype"),
         ("attention", "attention"), ("gradient_checkpointing", "gradient_checkpointing"),
         ("checkpoint_use_reentrant", "use_reentrant"), ("chunk_tokens", "chunk_tokens"),
         ("checkpoint_chunks", "checkpoint_chunks"))) and settings.get("bf16") is True and
        smoke_cache.get("base", {}).get("files") == ORIGINAL_FILES and
        smoke_cache.get("serialization", {}).get("terminal_suffix") == cache["serialization"]["terminal_suffix"],
        "Earlier native admission used another objective/precision/context path")
    admitted_long = smoke_cache.get("rows", {}).get("long", {}).get("input_ids", [])
    require(type(admitted_long) is list and admitted_long and
        max(len(row["input_ids"]) for row in cache["examples"]) <= len(admitted_long),
        "A selected complete row exceeds the actually admitted longest-sequence envelope")
    for case, pin in smoke["case_receipts"].items():
        case_receipt = _read_pin(root, evidence, pin)
        require(case_receipt.get("complete") is True and case_receipt.get("case") == case and
            set(case_receipt.get("checks", {})) == NATIVE_REQUIRED_CHECKS and
            all(value is True for value in case_receipt["checks"].values()) and
            case_receipt.get("memory", {}).get("headroom", {}).get("passed") is True,
            "An actual-model case was incomplete or failed")
    if plan["stage"] == "pilot":
        integration = proofs["accumulation8_admission"]
        require(integration.get("schema") == "extension-pilot-driver-complete-v1" and integration.get("complete") is True and
            integration.get("stage") == "integration_diagnostic" and integration.get("accumulation") == 8 and
            all(integration.get("code_sha256", {}).get(name) == codes[name] for name in REQUIRED_CODE) and
            set(integration.get("checks", {})) == REQUIRED_CHECKS and
            all(value is True for value in integration["checks"].values()) and
            all(integration.get("settings", {}).get(name) == plan[name] for name in
                ("objective", "beta", "chunk_tokens", "checkpoint_chunks", "parameter_dtype", "reference_dtype", "compute_dtype", "attention")),
            "The new GA8 driver still needs its own matching actual-model tail admission")
        validate_joint_shape_admission(proofs["long_after_adam_admission"], plan=plan, code_sha256=codes,
            maximum_tokens=max(len(row["input_ids"]) for row in cache["examples"]))
    evidence.unchanged()
    binding = {"plan_sha256": plan_hash, "cache_sha256": cache_hash,
        "qa_sha256": digest(paths["qa"]), "admission_sha256": digest(paths["admission"]), "code_sha256": codes}
    return {"root": root, "paths": paths, "plan": plan, "cache": cache, "qa": qa, "admission": admission,
        "code_sha256": codes, **binding, "binding_sha256": value_digest(binding), "evidence": evidence}


def _validate_complete(receipt, context, output):
    plan = context["plan"]
    require(receipt.get("schema") == "extension-pilot-driver-complete-v1" and receipt.get("complete") is True and
        receipt.get("binding_sha256") == context["binding_sha256"] and receipt.get("stage") == plan["stage"] and
        receipt.get("plan_sha256") == context["plan_sha256"] and receipt.get("cache_sha256") == context["cache_sha256"] and
        receipt.get("accumulation") == 8 and receipt.get("code_sha256") == context["code_sha256"] and
        set(receipt.get("checks", {})) == REQUIRED_CHECKS and all(value is True for value in receipt["checks"].values()) and
        receipt.get("consumption", {}).get("examples") == plan["exposure"]["examples"] and
        receipt["consumption"].get("optimizer_updates") == plan["exposure"]["updates"] and
        receipt["consumption"].get("ordered_ids_sha256") == context["cache"]["ordered_ids_sha256"] and
        set(receipt.get("checkpoints", {})) == {str(step) for step in plan["checkpoints"]},
        "Driver completion is incomplete/misbound")
    consumption = receipt["consumption"]
    require(consumption.get("complete") is True and type(consumption.get("updates")) is list and
        len(consumption["updates"]) == plan["exposure"]["updates"], "Actual update boundaries are incomplete")
    replay = ConsumptionLedger(context["cache"]["examples"], 8)
    for update, indices in enumerate(replay.groups, 1):
        for index in indices:
            replay.begin(index)
            replay.backward_complete(index)
        recorded = consumption["updates"][update - 1]
        replay.before_update(update, recorded.get("learning_rates_before"))
        replay.after_update(update, recorded.get("learning_rates_after"))
    require(value_digest(consumption) == value_digest(replay.finalize()), "Returned consumption differs from exact tokens/EOS/order/update replay")
    rng = receipt.get("rng", {})
    require(rng.get("seed") == plan["seed"] and rng.get("resume_supported") is False,
        "Initialization RNG/seed binding is missing")
    for phase in ("before_model_initialization", "before_train", "after_train"):
        require(set(rng.get(phase, {})) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "Incomplete RNG fingerprint")
        for checksum in rng[phase].values():
            _sha(checksum)
    used, following = receipt.get("lr_used"), receipt.get("lr_after_scheduler")
    require(type(used) is list and type(following) is list and
        len(used) == len(following) == plan["exposure"]["updates"], "Complete actual LR vectors are required")
    for index, (before, after) in enumerate(zip(used, following)):
        require(type(before) is list and type(after) is list and before and len(before) == len(after) and
            all(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, plan["learning_rate"] * cosine_factor(index), rel_tol=1e-12, abs_tol=1e-18) for value in before) and
            all(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, plan["learning_rate"] * cosine_factor(index + 1), rel_tol=1e-12, abs_tol=1e-18) for value in after),
            "Returned actual LR prefix differs from the independent full-horizon oracle")
    files = receipt.get("output_files", {})
    require(type(files) is dict and files, "No actual saved-model output files")
    for name, expected in files.items():
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe saved output path")
        require(digest(output / name) == expected, "Saved output changed")
    expected_files = {str(path.relative_to(output)) for path in (output / "checkpoints").rglob("*") if path.is_file()}
    expected_files.update(("presentations.jsonl", "consumption.json"))
    require(set(files) == expected_files, "Exact checkpoint/journal file inventory differs")
    for index, update in enumerate(consumption["updates"]):
        require(update["learning_rates_before"] == used[index] and update["learning_rates_after"] == used[index],
            "Consumed update and independent LR observations disagree")
    return receipt


def run(root, plan_path, cache_path, output, qa_receipt, admission_receipt, *, execute=None):
    """Single fresh attempt; injectable execution is a fixture seam, not bypass."""
    allocation = require_gpu_allocation()
    context = load_inputs(root, plan_path, cache_path, qa_receipt, admission_receipt)
    require(str(context["admission"].get("job_id")) == allocation["job_id"], "Admission belongs to a different allocation")
    resources = context["plan"]["resources"]
    deadline_guard(resources["wall_time_seconds"] + resources["reserve_seconds"])
    proposed = Path(output)
    require(not proposed.is_symlink(), "Output cannot alias another attempt")
    output = proposed.resolve()
    require(output.is_relative_to(context["root"] / "extension/runs"), "Driver outputs must remain under extension/runs")
    output = reserve_output(output)
    write_new_json(output / "started.json", {"schema": "extension-pilot-driver-started-v1", "complete": False,
        "binding_sha256": context["binding_sha256"], "stage": context["plan"]["stage"], "allocation": allocation,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "resume": False, "automatic_downstream": False})
    try:
        require(shutil.disk_usage(output).free >= resources["scratch_free_gib"] * (1 << 30), "Explicit free scratch is unavailable")
        result = (execute or execute_training)(context, output, allocation)
        context["evidence"].unchanged()
        _validate_complete(result, context, output)
        ending = require_gpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "Allocation changed")
        result["allocation_end"] = ending
        write_new_json(output / "complete.json", result)
        return result
    except BaseException as error:
        trace, ledger = context.get("_memory_trace"), context.get("_consumption")
        journal = context.get("_journal")
        if journal is not None:
            try:
                journal.flush()
                os.fsync(journal.fileno())
                journal.close()
            except BaseException:
                pass
        if trace is not None:
            try:
                trace.record("failure_snapshot")
            except BaseException:
                pass
        write_new_json(output / "failure.json", {"schema": "extension-pilot-driver-failure-v1", "complete": False,
            "binding_sha256": context["binding_sha256"], "exception": type(error).__name__, "message": str(error)[:4000],
            "retry": False, "resume": False, "automatic_downstream": False,
            "completed_presentations": [] if ledger is None else ledger.presentations,
            "pending_presentation": None if ledger is None else ledger.pending,
            "completed_update_boundaries": [] if ledger is None else ledger.updates,
            "memory_stages_observed": [] if trace is None else trace.stages,
            "allocator_peak_allocated_seen": None if trace is None else trace.whole_allocated,
            "allocator_peak_reserved_seen": None if trace is None else trace.whole_reserved,
            "max_observed_nonallocator_bytes": None if trace is None else trace.max_observed_nonallocator_bytes})
        raise


def execute_training(context, output, allocation):
    """Actual-model path; admitted compute only, no data choice or generation."""
    actual = require_gpu_allocation()
    require(actual["job_id"] == allocation["job_id"] == context["admission"]["job_id"] and
        actual["node"] == allocation["node"], "Actual training is outside its separately admitted GPU allocation")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed
    from transformers.loss.loss_utils import LOSS_MAPPING
    from extension.preserving_data import verify_serialized_examples
    from extension.preserving_trainer import check_optimizer, disjoint_models

    plan, cache = context["plan"], context["cache"]
    limits, serial, resources = plan["tolerances"], cache["serialization"], plan["resources"]
    for package, expected in VERSIONS.items():
        require(importlib.metadata.version(package) == expected, "Unadmitted runtime version: " + package)
    _host_resources(allocation, resources)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
        "B200" in torch.cuda.get_device_name(0), "Exactly one assigned B200 is required")
    gpu_identity = assigned_gpu_identity(torch)
    begun, base = time.monotonic(), Path(cache["base"]["path"])
    memory, initial_loss = MemoryTrace(torch), LOSS_MAPPING["ForCausalLM"]
    require(initial_loss.__module__ == "transformers.loss.loss_utils", "Global causal loss mapping was already patched")
    context["_memory_trace"] = memory
    set_seed(plan["seed"])
    rng = {"seed": plan["seed"], "resume_supported": False, "before_model_initialization": rng_fingerprint()}
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    replay = []
    for start in range(0, len(cache["examples"]), 4):
        replay.append(verify_serialized_examples(cache["examples"][start:start + 4], tokenizer,
            source_identity=cache["source_identity"], tokenizer_identity=cache["tokenizer_identity"],
            exclusion_receipt_sha256=value_digest(cache["exclusion_receipt"]), cutoff=serial["cutoff"],
            context_role=serial["context_role"], eos_token_id=serial["eos_token_id"],
            terminal_suffix=tuple(serial["terminal_suffix"]), terminal_text=serial["terminal_text"], vocab_size=serial["vocab_size"]))

    def load_model(path):
        model, info = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32, attn_implementation="sdpa",
            local_files_only=True, trust_remote_code=False, output_loading_info=True)
        require(all(not info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Partial/initialized model load is unsupported")
        return model.to("cuda")

    with memory.phase("fresh_original_fp32_student_reference"):
        student, reference = load_model(base), load_model(base).eval().requires_grad_(False)
        student.config.use_cache = reference.config.use_cache = False
        require(student.config.model_type == "qwen2" and student.config.vocab_size == 151936 and
            student.config.max_position_embeddings == 32768 and student.config.attention_dropout == 0 and
            student.config.use_sliding_window is False, "Architecture/context/dropout differs from admitted original")
        disjoint_models(student, reference)
        original = tensor_inventory(reference)
        require(len(original) == 434 and tensor_inventory(student) == original, "Fresh student/reference are not identical pinned originals")
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    journal_path = output / "presentations.jsonl"
    journal = journal_path.open("x", encoding="utf-8")
    context["_journal"] = journal

    def append_event(event):
        journal.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()
        if event["event"] in ("before_optimizer_update", "optimizer_update_complete"):
            os.fsync(journal.fileno())

    ledger = ConsumptionLedger(cache["examples"], 8, event_sink=append_event)
    context["_consumption"] = ledger
    observed = {"lr_used": [], "lr_after_scheduler": [], "metrics": [], "gradient_norms": [],
        "checkpoints": {}, "external_student_logits_dtypes": set(), "reference_logits_dtypes": set(), "parity": None,
        "actual_sdpa_operators": []}
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    rows = cache["examples"]
    short = next(row for row in rows if row["id"] == cache["parity_row_id"])
    require(len(short["input_ids"]) <= 512, "A prebound short row is required for independent CE/reload checks")

    def batch(row):
        return {name: torch.tensor([row[name]], dtype=torch.int64, device="cuda")
            for name in ("input_ids", "labels", "attention_mask")}

    def forward(row):
        return {key: value for key, value in batch(row).items() if key != "labels"}

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return {**{key: rows[index][key] for key in ("input_ids", "labels", "attention_mask")}, "row_index": index}

    def collate(items):
        require(len(items) == 1, "Only microbatch1 is admitted")
        return {key: torch.tensor([value], dtype=torch.int64) for key, value in items[0].items()}

    Base = make_pilot_trainer_class()

    class MeasuredTrainer(Base):
        def compute_loss(self, model, inputs, **kwargs):
            with memory.phase("loss_forward_and_objective"):
                value = super().compute_loss(model, inputs, **kwargs)
            return value

        def training_step(self, model, inputs, num_items_in_batch=None):
            if not observed["metrics"]:
                with memory.phase("actual_forward_and_backward"), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False,
                        profile_memory=False, with_stack=False) as profile:
                    result = super().training_step(model, inputs, num_items_in_batch)
                observed["actual_sdpa_operators"] = sorted({item.key for item in profile.key_averages()
                    if "_scaled_dot_product_" in item.key and any(name in item.key for name in ("flash", "efficient", "cudnn", "math"))})
                require(observed["actual_sdpa_operators"], "Actual SDPA dispatch was not observed in the first training presentation")
            else:
                with memory.phase("actual_forward_and_backward"):
                    result = super().training_step(model, inputs, num_items_in_batch)
            observed["metrics"].append({"row_index": int(inputs["row_index"].item()), **self.last_metrics})
            return result

        def compute_loss_context_manager(self):
            # Hook the returned prepared output, after Accelerate's conversion;
            # the factory still owns exactly one causal shift and all objectives.
            return super().compute_loss_context_manager()

        def create_scheduler(self, num_training_steps, optimizer=None):
            require(num_training_steps == plan["exposure"]["updates"], "Trainer update count differs")
            require(self.lr_scheduler is None, "A fresh scheduler is required")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, cosine_factor)
            return self.lr_scheduler

    args = TrainingArguments(output_dir=str(output / "trainer"), bf16=True, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1,
        max_steps=plan["exposure"]["updates"], learning_rate=plan["learning_rate"],
        weight_decay=plan["weight_decay"], lr_scheduler_type="cosine", warmup_steps=57,
        max_grad_norm=0., optim="adamw_torch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        label_smoothing_factor=0., eval_strategy="no", push_to_hub=False, save_strategy="no",
        logging_strategy="no", report_to=[], disable_tqdm=True, dataloader_num_workers=0,
        dataloader_drop_last=False, dataloader_pin_memory=False, seed=plan["seed"], data_seed=plan["seed"])
    trainer = MeasuredTrainer(model=student, reference_model=reference, args=args, train_dataset=Dataset(),
        data_collator=collate, consumption=ledger, objective=plan["objective"], beta=plan["beta"],
        chunk_tokens=plan["chunk_tokens"], checkpoint_chunks=True, clip_norm=plan["clip_norm"],
        eos_token_id=serial["eos_token_id"], terminal_suffix=tuple(serial["terminal_suffix"]))

    class Observe(TrainerCallback):
        def __init__(self):
            self.optimizer_frame = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            require(state.global_step == 0, "A fresh model/optimizer/scheduler at update0 is required")
            # Wrap ONLY this student's already-prepared forward; no global loss
            # patch. No logits or graph is retained in receipt state.
            previous = model.forward

            def measured_forward(*items, **named):
                result = previous(*items, **named)
                logits = result["logits"] if isinstance(result, dict) else result.logits
                observed["external_student_logits_dtypes"].add(str(logits.dtype))
                memory.record("prepared_student_output")
                return result
            model.forward = measured_forward
            previous_reference = reference.forward

            def measured_reference(*items, **named):
                result = previous_reference(*items, **named)
                logits = result["logits"] if isinstance(result, dict) else result.logits
                observed["reference_logits_dtypes"].add(str(logits.dtype))
                memory.record("reference_output")
                return result
            reference.forward = measured_reference
            self.original_student_forward, self.original_reference_forward = previous, previous_reference
            with memory.phase("prepared_zero_update_short_parity"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.train()
                p, q = model(**forward(short), use_cache=False).logits.float(), reference(**forward(short), use_cache=False).logits.float()
                labels = batch(short)["labels"][:, 1:]
                logp, logq = p[:, :-1].log_softmax(-1), q[:, :-1].log_softmax(-1)
                ce = torch.nn.functional.cross_entropy(p[:, :-1].reshape(-1, p.shape[-1]), labels.reshape(-1), ignore_index=-100)
                qce = torch.nn.functional.cross_entropy(q[:, :-1].reshape(-1, q.shape[-1]), labels.reshape(-1), ignore_index=-100)
                kl = (logq.exp() * (logq - logp)).sum(-1)[labels != -100].mean()
                observed["parity"] = check_parity({"max_abs_logits": float((p - q).abs().max()),
                    "ce_abs": abs(float(ce - qce)), "kl_abs": abs(float(kl))}, limits)
                model.eval()
                trainer.compute_loss(model, batch(short))
                observed["independent_ce_abs"] = abs(trainer.last_metrics["ce"] - float(ce))
                require(observed["independent_ce_abs"] <= limits["ce_abs"], "Independent shifted CE differs")
                model.train()

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            assert_finite_model(model, gradients=True)
            expected = plan["learning_rate"] * cosine_factor(state.global_step)
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            require(all(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-18) for value in lrs), "Actual used LR differs from full-horizon oracle")
            observed["lr_used"].append(lrs)
            observed["gradient_norms"].append(trainer.last_gradient_norm)
            self.optimizer_frame = memory.begin("adam_update_" + str(state.global_step + 1))

        def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            check_optimizer(model, optimizer, reference)
            memory.end(self.optimizer_frame)
            self.optimizer_frame = None

        def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            expected = plan["learning_rate"] * cosine_factor(state.global_step)
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            require(all(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-18) for value in lrs), "Scheduler advanced at the wrong boundary")
            observed["lr_after_scheduler"].append(lrs)
            require(time.monotonic() - begun < resources["wall_time_seconds"], "Bounded run time exhausted")
            deadline_guard(resources["reserve_seconds"])
            if state.global_step in plan["checkpoints"]:
                with memory.phase("checkpoint_" + str(state.global_step)):
                    assert_finite_model(student)
                    inventory = tensor_inventory(student)
                    directory = checkpoints / ("step-" + str(state.global_step))
                    require(not directory.exists(), "Checkpoint cannot be overwritten")
                    trainer.save_model(str(directory))
                    for name in TOKENIZER_FILES:
                        shutil.copyfile(base / name, directory / name)
                    saved = _checkpoint_inventory(directory, inventory)
                    observed["checkpoints"][str(state.global_step)] = {"directory": str(directory),
                        "tensor_inventory_sha256": value_digest(inventory), "saved_student": saved}

        def on_train_end(self, args, state, control, model=None, **kwargs):
            model.forward = self.original_student_forward
            reference.forward = self.original_reference_forward

    trainer.add_callback(Observe())
    rng["before_train"] = rng_fingerprint()
    with memory.phase("actual_one_pass_training"):
        trainer.train()
    rng["after_train"] = rng_fingerprint()
    journal.close()
    consumption = ledger.finalize()
    require(trainer.state.global_step == plan["exposure"]["updates"] and
        trainer.training_forward_examples == len(rows), "Actual one-pass exposure differs")
    with memory.phase("final_fp32_finite_reference_and_optimizer_scan"):
        finite_count = assert_finite_model(student)
        optimizer = check_optimizer(student, trainer.optimizer, reference)
        require(optimizer.state and all(parameter in optimizer.state for parameter in student.parameters()), "Missing actual Adam moment allocation")
        require(tensor_inventory(reference) == original and not reference.training and
            all(parameter.grad is None for parameter in reference.parameters()), "Frozen original reference changed")
        final_inventory = tensor_inventory(student)
        require(any(final_inventory[name]["sha256"] != original[name]["sha256"] for name in original), "No FP32 parameter changed")
    final = checkpoints / ("step-" + str(plan["exposure"]["updates"]))
    with memory.phase("final_saved_student_reload_parity"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        student.eval()
        expected_logits = student(**forward(short), use_cache=False).logits.float().cpu()
        # Free the independent reference before a third FP32 model is loaded;
        # its complete byte invariance has already been checked. This is not an
        # empty-cache operation and does not erase preceding memory peaks.
        reference.to("cpu")
        reloaded = load_model(final).eval()
        actual_logits = reloaded(**forward(short), use_cache=False).logits.float().cpu()
        reload_difference = float((expected_logits - actual_logits).abs().max())
        require(reload_difference <= limits["reload_logit_max_abs"], "Saved student reload parity differs")
        del reloaded, expected_logits, actual_logits
    require(LOSS_MAPPING["ForCausalLM"] is initial_loss, "Global CE mapping changed")
    require("torch.float32" in observed["external_student_logits_dtypes"], "Prepared Accelerate FP32 logits were not observed")
    memory_result = memory.result(limits)
    require(memory_result["headroom"]["passed"], "Measured whole-run headroom is insufficient")
    context["evidence"].unchanged()
    write_new_json(output / "consumption.json", consumption)
    files = {str(path.relative_to(output)): digest(path) for path in sorted(checkpoints.rglob("*")) if path.is_file()}
    files["presentations.jsonl"] = digest(journal_path)
    files["consumption.json"] = digest(output / "consumption.json")
    checks = {name: True for name in sorted(REQUIRED_CHECKS)}
    receipt = {"schema": "extension-pilot-driver-complete-v1", "complete": True, "stage": plan["stage"],
        "binding_sha256": context["binding_sha256"], "plan_sha256": context["plan_sha256"],
        "cache_sha256": context["cache_sha256"], "code_sha256": context["code_sha256"],
        "allocation": allocation, "gpu_identity": gpu_identity, "accumulation": 8,
        "shape_evidence": {"scope": "short_accumulation_tail_only" if plan["stage"] == "integration_diagnostic" else "training_run_only",
            "maximum_complete_tokens": max(len(row["input_ids"]) for row in rows),
            "long_context_joint_peak_proven": False,
            "resident_adam_and_gradient_state_before_forward_measured": False,
            "observations": [], "qualification": "This run is not a long-after-Adam joint-envelope admission. The short GA8 tail check cannot promote32K pilots; a separately pinned native diagnostic must prove moments plus accumulated gradients before a longest forward."},
        "settings": {name: plan[name] for name in ("objective", "beta", "chunk_tokens", "checkpoint_chunks",
            "parameter_dtype", "reference_dtype", "compute_dtype", "attention")}, "rng": rng,
        "optimizer": {"name": "adamw_torch", "moment_dtype": "float32", "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2, "adam_epsilon": args.adam_epsilon, "weight_decay": args.weight_decay},
        "consumption": consumption, "lr_used": observed["lr_used"], "lr_after_scheduler": observed["lr_after_scheduler"],
        "checkpoints": observed["checkpoints"], "checks": checks, "memory": memory_result,
        "reference_tensor_inventory_sha256": value_digest(original), "finite_parameter_count": finite_count,
        "final_student_tensor_inventory_sha256": value_digest(final_inventory), "native_replay": replay,
        "reload_max_abs_logits": reload_difference, "parity": observed["parity"],
        "independent_ce_abs": observed["independent_ce_abs"], "presentation_metrics": observed["metrics"],
        "gradient_norms": observed["gradient_norms"], "external_student_logits_dtypes": sorted(observed["external_student_logits_dtypes"]),
        "clipping": [{"update": index + 1, "pre_clip_gradient_norm": norm,
            "max_gradient_norm": plan["clip_norm"], "clipped": norm > plan["clip_norm"]}
            for index, norm in enumerate(observed["gradient_norms"])],
        "reference_logits_dtypes": sorted(observed["reference_logits_dtypes"]), "output_files": files,
        "actual_sdpa_operators": observed["actual_sdpa_operators"],
        "input_files": context["evidence"].files, "versions": {**VERSIONS, "python": sys.version, "cuda": torch.version.cuda},
        "elapsed_seconds": time.monotonic() - begun, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_supported": False, "automatic_downstream": False, "raw_pairs_exported": False,
        "scientific_promotion_performed": False, "transactional_optimizer_rollback": False}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    parser.add_argument("--admission-receipt", required=True)
    args = parser.parse_args()
    result = run(args.root, args.plan, args.cache, args.output, args.qa_receipt, args.admission_receipt)
    print({"complete": True, "stage": result["stage"], "automatic_downstream": False})


if __name__ == "__main__":
    main()
