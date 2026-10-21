"""Unadmitted paired numerical diagnostic using actual HF Trainer.training_step.

No model/data loading, file I/O, CLI, scheduler submission or native provenance
admission is provided. Already-constructed Trainers, nine sealed short batches,
original reference, observation contexts and EVERY comparison/shadow hook are
required. The caller owns their source/current-job/resource/file identities and
the actual native numerical gates. This helper does not run Trainer.train and
cannot establish full-loop, long-context, unique-row or pilot admission.

Only the inherited pinned training_step owns loss division/backward. The
controller explicitly owns diagnostic preparation and group/callback/optimizer/
scheduler boundaries. Two live state sets are compared before mutation; no
gradient, moment or Parameter is cloned to a full host snapshot. The only state
copies here are bounded short-token/RNG metadata. Hook implementations must use
the separately reviewed bounded numerical and private-shadow mechanisms.

All failures stop this version. Optimizer failure is not transactional rollback:
the first branch may already have changed when the second fails. A typed error
carries bounded partial events, and the caller must preserve that private result.
Authoring this module executes nothing; CPU review/QA and native admission remain
separate future stages.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import random

import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from transformers import Trainer
from transformers.trainer_callback import DefaultFlowCallback, PrinterCallback

from extension.preserving_trainer import _FiniteUpdates, _optimizer, check_optimizer


SCHEMA = "extension-paired-training-step-controller-v1"
BRANCHES = ("dense", "candidate")
ROLES = ("short0", "short1", "short2", "short0", "short1", "short2", "short0", "short1", "short2")
GROUPS = ((0, 1, 2, 3, 4, 5, 6, 7), (8,))
PEAK_LR, WEIGHT_DECAY, BETA = 2e-6, .01, .05
WARMUP_STEPS, TOTAL_STEPS, SEED = 57, 1875, 20261021
MAX_EVENTS, MAX_JOURNAL_BYTES, MAX_RECEIPT_BYTES = 256, 2 << 20, 128 << 10
MAX_FINITE_CHUNK = 1048576


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _json_copy(value, maximum=MAX_RECEIPT_BYTES):
    _require(_plain(value), "Receipts/events must contain finite plain JSON metadata only")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _require(len(payload) <= maximum, "Bounded diagnostic metadata limit exceeded")
    return json.loads(payload)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PairedStepHooks:
    """Required native-worker seams; no optional or silently skipped gate.

    Each callable receives keyword stage, dense_trainer, candidate_trainer,
    reference_model and context. compare_scalars also receives dense/candidate
    plain scalar metadata; compare_deltas also receives initial_shadow.
    create_shadow returns ShadowArtifact; others return a complete/passed JSON
    receipt. Numerical/identity/delta receipts must declare compared_tensor_count
    equal to the explicit expected count. Hook source/provenance and actual math
    are caller obligations; accepting a receipt is not native evidence by itself.
    """
    verify_invariants: object
    verify_identity: object
    compare_scalars: object
    compare_tensors: object
    create_shadow: object
    compare_deltas: object

    def __post_init__(self):
        _require(all(callable(getattr(self, name)) for name in self.__dataclass_fields__),
            "Every paired numerical/invariant/identity/shadow hook is mandatory")


@dataclass(frozen=True)
class ShadowArtifact:
    """Opaque bounded reader owned by the caller; never serialized by this helper."""
    handle: object
    receipt: dict


class PairedControllerFailure(RuntimeError):
    def __init__(self, result):
        self.result = _json_copy(result, MAX_JOURNAL_BYTES)
        super().__init__("Paired training_step diagnostic failed; preserve its private partial result")


class _Journal:
    def __init__(self):
        self.events = []

    def add(self, name, **values):
        event = _json_copy({"event": name, **values})
        _require(len(self.events) < MAX_EVENTS, "Too many diagnostic events")
        proposed = self.events + [event]
        _json_copy(proposed, MAX_JOURNAL_BYTES - (64 << 10))
        self.events.append(event)


def _finite(tensor):
    _require(isinstance(tensor, torch.Tensor) and tensor.is_contiguous(), "Expected a contiguous tensor")
    flat = tensor.detach().view(-1)
    for start in range(0, flat.numel(), MAX_FINITE_CHUNK):
        _require(bool(torch.isfinite(flat.narrow(0, start, min(MAX_FINITE_CHUNK, flat.numel() - start))).all()),
            "Nonfinite diagnostic tensor")


def _parameters(model, expected, *, trainable):
    _require(isinstance(model, torch.nn.Module), "An already-constructed original model is required")
    values = dict(model.named_parameters())
    _require(type(expected) is int and 1 <= expected <= 434 and len(values) == expected and
        len({id(value) for value in values.values()}) == expected, "Explicit unique Parameter count differs")
    devices = {value.device for value in values.values()}
    _require(len(devices) == 1 and next(iter(devices)).type in ("cpu", "cuda"), "One original CPU/CUDA device required")
    for value in values.values():
        _require(value.dtype == torch.float32 and value.is_leaf and value.is_contiguous() and
            value.requires_grad is trainable and not torch.is_inference(value), "Original normal FP32 Parameter policy differs")
    return values


def _signature(values):
    return {name: (id(value), value.untyped_storage().data_ptr(), tuple(value.shape), tuple(value.stride()),
        str(value.device), str(value.dtype), value.requires_grad) for name, value in values.items()}


def _reference_versions(reference):
    return {name: (id(value), value.untyped_storage().data_ptr(), tuple(value.shape), str(value.dtype),
        str(value.device), value._version) for name, value in (*reference.named_parameters(), *reference.named_buffers())}


def _batch_record(batch):
    allowed = {"input_ids", "labels", "attention_mask", "position_ids", "row_index"}
    _require(type(batch) is dict and {"input_ids", "labels", "attention_mask"} <= set(batch) <= allowed,
        "An unchanged explicit short native batch is required")
    ids = batch["input_ids"]
    _require(isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1 and
        2 <= ids.shape[1] <= 512, "Only the sealed microbatch1 short envelope is supported")
    records = {}
    for name, value in batch.items():
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and value.is_contiguous() and
            not value.requires_grad and not torch.is_inference(value), "Native batch fields require normal contiguous int64 tensors")
        _require((value.numel() == 1 if name == "row_index" else value.shape == ids.shape), "Short batch field shape differs")
        # These are <=512-token input fields, never a model/state transfer.
        payload = value.detach().to(device="cpu", copy=True).numpy().tobytes()
        records[name] = {"shape": list(value.shape), "sha256": hashlib.sha256(payload).hexdigest()}
    return records


def _rng_capture(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _rng_hash(state):
    numpy = state["numpy"]
    metadata = {"python": state["python"], "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
        "cuda": None if state["cuda"] is None else hashlib.sha256(state["cuda"].numpy().tobytes()).hexdigest()}
    # RNG tuple trees are bounded generator metadata, not tensors from a model.
    return _digest(metadata)


def _rng_restore(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _lrs(trainer):
    values = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    _require(values and all(math.isfinite(value) for value in values), "Nonfinite or missing optimizer LR")
    return values


def _expect_lr(trainer, numerator):
    values, expected = _lrs(trainer), PEAK_LR * numerator / WARMUP_STEPS
    _require(all(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-18) for value in values),
        "Diagnostic cosine-prefix learning rate differs")
    return values


def _gradients(trainer, parameters):
    result = {}
    for name, value in parameters.items():
        gradient = value.grad
        _require(isinstance(gradient, torch.Tensor) and gradient.shape == value.shape and
            gradient.dtype == torch.float32 and gradient.device == value.device and gradient.is_contiguous() and
            not gradient.requires_grad, "Every student Parameter needs its actual normal FP32 gradient")
        _finite(gradient)
        result[name] = gradient
    return result


def _state(trainer, parameters, step):
    raw = check_optimizer(trainer.model, trainer.optimizer, trainer.reference_model)
    _require(type(raw) is torch.optim.AdamW and set(raw.state) == set(parameters.values()), "Exact original AdamW state membership required")
    for parameter in parameters.values():
        value = raw.state[parameter]
        _require(type(value) is dict and set(value) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected or missing Adam state")
        counter = value["step"]
        _require(isinstance(counter, torch.Tensor) and counter.numel() == 1 and counter.dtype == torch.float32 and
            counter.is_contiguous() and not counter.requires_grad and not torch.is_inference(counter),
            "Actual Adam step must be a normal finite FP32 scalar")
        _finite(counter)
        _require(float(counter.detach().item()) == step, "Actual Adam per-Parameter step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = value[name]
            _require(isinstance(moment, torch.Tensor) and moment.dtype == torch.float32 and
                moment.shape == parameter.shape and moment.device == parameter.device and
                moment.is_contiguous() and not moment.requires_grad, "Adam moment shape/storage differs")
            _finite(moment)
            if name == "exp_avg_sq":
                flat = moment.detach().view(-1)
                for start in range(0, flat.numel(), MAX_FINITE_CHUNK):
                    _require(bool((flat.narrow(0, start, min(MAX_FINITE_CHUNK, flat.numel() - start)) >= 0).all()),
                        "Negative actual Adam second moment")


def _callback(trainer, name):
    trainer.control = getattr(trainer.callback_handler, name)(trainer.args, trainer.state, trainer.control)


def _optimizer_groups(trainer, parameters):
    lookup = {id(value): name for name, value in parameters.items()}
    raw = _optimizer(trainer.optimizer)
    result = []
    for group in raw.param_groups:
        result.append({"names": [lookup[id(value)] for value in group["params"]],
            "defaults": {name: list(value) if type(value) is tuple else value
                for name, value in group.items() if name != "params"}})
    return _json_copy(result)


def _prepare(trainer, reference, expected, branch):
    _require(isinstance(trainer, Trainer) and trainer.training_step.__func__ is Trainer.training_step,
        "Use actual unchanged inherited Trainer.training_step")
    _require(trainer.reference_model is reference and trainer.optimizer is None and trainer.lr_scheduler is None and
        not getattr(trainer, "_paired_controller_started", False) and not getattr(trainer, "_prototype_train_started", False),
        "Fresh already-constructed Trainer/reference required; no resume or reuse")
    args, accelerator = trainer.args, trainer.accelerator
    _require(args.gradient_accumulation_steps == 8 and args.per_device_train_batch_size == 1 and
        args.max_grad_norm == 0 and not args.fp16 and not args.torch_compile and
        args.learning_rate == PEAK_LR and args.weight_decay == WEIGHT_DECAY and args.warmup_steps == WARMUP_STEPS and
        args.seed == args.data_seed == SEED and args.label_smoothing_factor == 0 and
        not args.remove_unused_columns and not args.dataloader_drop_last and args.torch_empty_cache_steps is None and
        getattr(args.optim, "value", args.optim) == "adamw_torch" and getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type) == "cosine" and
        args.adam_beta1 == .9 and args.adam_beta2 == .999 and args.adam_epsilon == 1e-8 and
        not args.push_to_hub and trainer.compute_loss_func is None and trainer.model_accepts_loss_kwargs is False,
        "Exact diagnostic accumulation/AdamW/cosine/scalar-loss settings required")
    _require(all(getattr(getattr(args, name), "value", getattr(args, name)) == "no"
        for name in ("eval_strategy", "logging_strategy", "save_strategy")) and list(args.report_to) == [] and args.disable_tqdm,
        "No evaluation/logging/checkpoint automation or integration callbacks")
    _require(accelerator.num_processes == 1 and getattr(accelerator.state.distributed_type, "value", accelerator.state.distributed_type) == "NO" and
        accelerator.gradient_accumulation_steps == 1 and accelerator.scaler is None and
        not trainer.is_deepspeed_enabled and not trainer.is_fsdp_enabled and not accelerator._models and
        not accelerator._optimizers and not accelerator._schedulers and not accelerator._dataloaders,
        "Fresh single-process Accelerator with divisor1 and no prepared objects required")
    parameters = _parameters(trainer.model, expected, trainable=True)
    device = next(iter(parameters.values())).device
    _require(args.bf16 is (device.type == "cuda") and accelerator.mixed_precision == ("bf16" if args.bf16 else "no"),
        "CUDA requires BF16; CPU is a separate FP32 oracle seam")
    _require(trainer.objective in ("sft", "ce_kl", "asft", "prior_anchored") and trainer.beta == BETA and
        trainer.chunk_tokens == 128 and trainer.checkpoint_chunks is True and trainer.clip_norm == 1.,
        "Frozen objective/chunk/owned-clipping configuration required")
    _require(args.gradient_checkpointing and args.gradient_checkpointing_kwargs == {"use_reentrant": False} and
        hasattr(trainer.model, "gradient_checkpointing_enable") and not trainer.model.config.use_cache,
        "Original non-reentrant decoder checkpoint/cache policy required")
    callbacks = trainer.callback_handler.callbacks
    _require([type(value) for value in callbacks] == [DefaultFlowCallback, PrinterCallback, _FiniteUpdates] and
        callbacks[-1].owner is trainer,
        "Only the inherited finite/clip callback and passive standard callbacks are allowed")
    _require(trainer.state.global_step == 0 and trainer.training_forward_examples == 0 and
        all(value.grad is None for value in parameters.values()), "Fresh zero-step/no-training-forward/no-gradient students required")
    trainer._paired_controller_started = True
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.create_optimizer_and_scheduler(num_training_steps=TOTAL_STEPS)
    raw, scheduler = _optimizer(trainer.optimizer), trainer.lr_scheduler
    _require(type(raw) is torch.optim.AdamW and not raw.state and type(scheduler) is torch.optim.lr_scheduler.LambdaLR and
        scheduler.optimizer is raw, "Fresh ordinary AdamW/LambdaLR required")
    _require(raw.defaults["betas"] == (.9, .999) and raw.defaults["eps"] == 1e-8 and
        raw.defaults["amsgrad"] is False and raw.defaults["maximize"] is False and
        raw.defaults["capturable"] is False and raw.defaults["differentiable"] is False and
        raw.defaults["foreach"] is None and raw.defaults["fused"] is None,
        "Preserve the ordinary unfused AdamW defaults without new backend flags")
    decay = set(trainer.get_decay_parameter_names(trainer.model))
    expected_names = [[name for name in parameters if name in decay], [name for name in parameters if name not in decay]]
    group_names = [[name for value in group["params"] for name, parameter in parameters.items() if value is parameter]
        for group in raw.param_groups]
    _require(group_names == expected_names and len(raw.param_groups) == 2 and
        raw.param_groups[0]["weight_decay"] == WEIGHT_DECAY and raw.param_groups[1]["weight_decay"] == 0.,
        "Keep the actual original-name HF decay and excluded-norm/bias groups")
    prepared, optimizer = accelerator.prepare(trainer.model, trainer.optimizer)
    _require(prepared is trainer.model and accelerator.unwrap_model(prepared, keep_fp32_wrapper=True) is prepared and
        type(optimizer) is AcceleratedOptimizer and optimizer.optimizer is raw and optimizer.scaler is None,
        "Exactly-once original-model AMP/AcceleratedOptimizer preparation required")
    trainer.optimizer = optimizer
    _require(trainer.lr_scheduler is scheduler and scheduler.optimizer is raw and not accelerator._schedulers and
        list(accelerator._models) == [trainer.model] and list(accelerator._optimizers) == [optimizer] and
        not accelerator._dataloaders, "Do not prepare/replace the raw scheduler, reference or dataloader")
    check_optimizer(trainer.model, optimizer, reference)
    trainer.state.max_steps, trainer.state.num_train_epochs, trainer.state.train_batch_size = 2, 1, 1
    trainer.state.epoch = 0.
    for name in ("model", "optimizer", "lr_scheduler"):
        setattr(trainer.callback_handler, name, getattr(trainer, name))
    _expect_lr(trainer, 0)
    return parameters, device


def run_paired_training_steps(dense_trainer, candidate_trainer, batches, *, reference_model, hooks,
                             observers, expected_parameters=434):
    """Run only the explicit nine-short paired numerical diagnostic.

    Both actual Trainer subclasses must expose paired_observer (the supplied
    object). Their source-reviewed compute_loss override must use nested
    observer.phase('forward', presentation_index=self.paired_presentation_index)
    around super().compute_loss without changing its loss. The controller owns
    the enclosing backward phase and installs/removes both observer contexts.
    No duck-typed fake Trainer is a production substitution. Small explicit
    counts and fake observation hooks belong only to future CPU fixtures.
    """
    journal, completed, updates, shadow = _Journal(), {name: 0 for name in BRANCHES}, {name: 0 for name in BRANCHES}, None
    attempts = {name: 0 for name in BRANCHES}
    result = {"schema": SCHEMA, "complete": False, "execution_kind": "paired_native_trainer_training_step_controller",
        "trainer_train_called": False, "full_trainer_train_loop_established": False,
        "unique_row_driver_established": False, "long_context_memory_established": False,
        "native_provenance_established": False, "training_or_pilot_admitted": False,
        "seed_contract": {"configured_python_numpy_torch_seed": SEED,
            "initialization_and_actual_rng_seeding_owned_by_caller": True,
            "seed_configuration_alone_proves_rng_state": False, "controller_reseeds_generators": False}}
    trainers = {"dense": dense_trainer, "candidate": candidate_trainer}
    try:
        _require(importlib.metadata.version("transformers") == "4.57.3" and importlib.metadata.version("accelerate") == "1.12.0" and
            str(torch.__version__).split("+")[0] == "2.9.0" and np.__version__ == "1.26.4", "Pinned diagnostic runtime required")
        _require(type(hooks) is PairedStepHooks and type(observers) is dict and set(observers) == set(BRANCHES),
            "All paired numerical hooks and two explicit observers are mandatory")
        _require(type(batches) in (tuple, list) and len(batches) == 9 and dense_trainer is not candidate_trainer,
            "Exactly nine sealed short batches and two separate Trainers required")
        batch_records = [_batch_record(value) for value in batches]
        # Bind repetitions semantically by original token/mask fields, not a
        # diagnostic row_index that can deliberately differ per presentation.
        semantic = [{name: value for name, value in row.items() if name != "row_index"} for row in batch_records]
        _require(len({_digest(row) for row in semantic[:3]}) == 3 and
            all(semantic[index] == semantic[index % 3] for index in range(9)), "Exact three-witness repetition schedule required")
        reference = _parameters(reference_model, expected_parameters, trainable=False)
        _require(not reference_model.training and all(value.grad is None for value in reference.values()), "Separate frozen FP32 reference required in EVERY case")
        reference_start = _reference_versions(reference_model)
        parameter_sets, devices = {}, {}
        for branch in BRANCHES:
            observer = observers[branch]
            _require(getattr(trainers[branch], "paired_observer", None) is observer and callable(getattr(observer, "phase", None)) and
                callable(getattr(observer, "snapshot", None)) and callable(getattr(observer, "validate", None)) and
                hasattr(observer, "__enter__") and hasattr(observer, "__exit__"),
                "Source-reviewed nested-forward Trainer observer binding required")
            parameter_sets[branch], devices[branch] = _prepare(trainers[branch], reference_model, expected_parameters, branch)
            journal.add("branch_prepared", branch=branch, parameters=expected_parameters, original_model_retained=True,
                accelerator_accumulation_steps=1, scheduler_prepared=False)
        device = devices["dense"]
        _require(devices["candidate"] == device and next(iter(reference.values())).device == device and
            list(parameter_sets["dense"]) == list(parameter_sets["candidate"]) == list(reference) and
            all(parameter_sets["dense"][name].shape == parameter_sets["candidate"][name].shape == reference[name].shape for name in reference),
            "All original Parameter names/shapes/devices must match")
        storages = [{(str(value.device), value.untyped_storage().data_ptr()) for value in values.values()}
            for values in (parameter_sets["dense"], parameter_sets["candidate"], reference)]
        _require(not any(storages[i] & storages[j] for i in range(3) for j in range(i)), "Student/reference Parameter storage cannot overlap")
        initial_signatures = {branch: _signature(values) for branch, values in parameter_sets.items()}
        _require(trainers["dense"].objective == trainers["candidate"].objective and
            trainers["dense"].terminal_suffix == trainers["candidate"].terminal_suffix and
            trainers["dense"].eos_token_id == trainers["candidate"].eos_token_id,
            "Paired objective/native termination contracts differ")
        _require(_optimizer_groups(trainers["dense"], parameter_sets["dense"]) ==
            _optimizer_groups(trainers["candidate"], parameter_sets["candidate"]), "Actual AdamW groups/defaults/order differ")

        def context(**values):
            return {"expected_parameters": expected_parameters, "objective": dense_trainer.objective,
                "completed_backwards": dict(completed), "optimizer_updates": dict(updates), **values}

        def invoke(name, stage, *, tensor_count=False, **extra):
            value = getattr(hooks, name)(stage=stage, dense_trainer=dense_trainer, candidate_trainer=candidate_trainer,
                reference_model=reference_model, context=context(), **extra)
            artifact = value if name == "create_shadow" else None
            if artifact is not None:
                _require(type(artifact) is ShadowArtifact and artifact.handle is not None, "A real opaque initial-shadow artifact is required")
                value = artifact.receipt
            receipt = _json_copy(value)
            _require(type(receipt) is dict and receipt.get("complete") is True and receipt.get("passed") is True and
                receipt.get("stage") == stage, "Required hook did not complete/pass its exact stage")
            if tensor_count:
                _require(type(receipt.get("compared_tensor_count")) is int and receipt["compared_tensor_count"] == expected_parameters,
                    "The hook must cover every exact compared Parameter")
            journal.add("hook_complete", hook=name, stage=stage, receipt=receipt)
            return artifact if artifact is not None else receipt

        invoke("verify_invariants", "prepared")
        invoke("verify_identity", "initial_original", tensor_count=True)
        journal.add("logical_rng_entry", configured_seed=SEED, actual_state_sha256=_rng_hash(_rng_capture(device)),
            initial_seeding_provenance_owned_by_caller=True)
        with ExitStack() as stack:
            for branch in BRANCHES:
                stack.enter_context(observers[branch])
                _callback(trainers[branch], "on_train_begin")
                _callback(trainers[branch], "on_epoch_begin")
            for update_index, indices in enumerate(GROUPS):
                size = len(indices)
                for branch in BRANCHES:
                    trainer = trainers[branch]
                    _require(trainer.state.global_step == update_index and
                        all(value.grad is None for value in parameter_sets[branch].values()), "Previous diagnostic group was not finalized")
                    trainer.current_gradient_accumulation_steps = size
                    _callback(trainer, "on_step_begin")
                for offset, index in enumerate(indices):
                    _require([_batch_record(value) for value in batches] == batch_records, "Sealed short input batches changed")
                    boundary = offset == size - 1
                    for trainer in trainers.values():
                        _require(trainer.accelerator.gradient_accumulation_steps == 1, "Accelerator loss divisor changed")
                        trainer.accelerator.sync_gradients = boundary
                    before = _rng_capture(device)
                    returned, metrics, after_dense = {}, {}, None
                    for branch in BRANCHES:
                        if branch == "candidate":
                            _rng_restore(before, device)
                        trainer, observer = trainers[branch], observers[branch]
                        trainer.paired_presentation_index = index
                        journal.add("training_step_begin", branch=branch, presentation_index=index, group_size=size,
                            source_role=ROLES[index], boundary=boundary, rng_before_sha256=_rng_hash(before))
                        with observer.phase("backward", presentation_index=index):
                            loss = trainer.training_step(trainer.model, batches[index], num_items_in_batch=None)
                        _synchronize(device)
                        _require(isinstance(loss, torch.Tensor) and loss.numel() == 1 and not loss.requires_grad and
                            bool(torch.isfinite(loss)), "Actual training_step must return finite detached scaled loss")
                        _require(trainer.training_forward_examples == completed[branch] + 1,
                            "Each actual training_step must consume exactly one unchanged training example")
                        returned[branch] = float(loss.item())
                        metrics[branch] = _json_copy(trainer.last_metrics)
                        completed[branch] += 1
                        after = _rng_capture(device)
                        if branch == "dense":
                            after_dense = after
                        else:
                            _require(_rng_hash(after) == _rng_hash(after_dense), "Paired logical RNG progression differs")
                        journal.add("training_step_complete", branch=branch, presentation_index=index,
                            scaled_loss=returned[branch], rng_after_sha256=_rng_hash(after))
                    _rng_restore(after_dense, device)
                    invoke("compare_scalars", "presentation_" + str(index), dense={"scaled_loss": returned["dense"],
                        "group_size": size, "metrics": metrics["dense"]}, candidate={"scaled_loss": returned["candidate"],
                        "group_size": size, "metrics": metrics["candidate"]})
                    if not boundary:
                        for trainer in trainers.values():
                            _callback(trainer, "on_substep_end")
                _synchronize(device)
                for branch in BRANCHES:
                    _require(trainers[branch].accelerator.sync_gradients is True, "Prepared optimizer step would be skipped")
                    _gradients(trainers[branch], parameter_sets[branch])
                invoke("verify_invariants", "before_update_" + str(update_index + 1))
                invoke("compare_tensors", "raw_gradients_" + str(update_index + 1), tensor_count=True)
                for branch in BRANCHES:
                    _callback(trainers[branch], "on_pre_optimizer_step")
                    _require(type(trainers[branch].last_gradient_norm) is float and math.isfinite(trainers[branch].last_gradient_norm),
                        "Inherited finite/clip callback did not expose a finite norm")
                invoke("compare_tensors", "clipped_gradients_" + str(update_index + 1), tensor_count=True)
                invoke("compare_scalars", "clip_norm_" + str(update_index + 1),
                    dense={"clip_norm": dense_trainer.last_gradient_norm}, candidate={"clip_norm": candidate_trainer.last_gradient_norm})
                for branch in BRANCHES:
                    trainer = trainers[branch]
                    used = _expect_lr(trainer, update_index)
                    journal.add("optimizer_step_begin", branch=branch, update=update_index + 1, used_lrs=used)
                    attempts[branch] += 1
                    trainer.optimizer.step()
                    updates[branch] += 1
                    _callback(trainer, "on_optimizer_step")
                    _require(trainer.accelerator.optimizer_step_was_skipped is False and trainer.optimizer.step_was_skipped is False,
                        "An actual prepared optimizer step was skipped")
                    _state(trainer, parameter_sets[branch], update_index + 1)
                    journal.add("optimizer_step_complete", branch=branch, update=update_index + 1)
                for kind in ("adam_first_moments", "adam_second_moments", "updated_weights"):
                    invoke("compare_tensors", kind + "_" + str(update_index + 1), tensor_count=True)
                if update_index == 0:
                    invoke("verify_identity", "post_lr0_original", tensor_count=True)
                    shadow = invoke("create_shadow", "post_lr0_pre_update2", tensor_count=True)
                else:
                    _require(shadow is not None, "The exact post-LR0 initial shadow is missing")
                    receipt = invoke("compare_deltas", "actual_update2_deltas", tensor_count=True, initial_shadow=shadow.handle)
                    _require(receipt.get("positive_update_groups") == {branch: {"tied": True, "backbone": True} for branch in BRANCHES},
                        "Both actual stored update groups must be positive in both branches")
                for branch in BRANCHES:
                    trainer = trainers[branch]
                    trainer.lr_scheduler.step()
                    after = _expect_lr(trainer, update_index + 1)
                    trainer.model.zero_grad()
                    _require(all(value.grad is None for value in parameter_sets[branch].values()), "Original model.zero_grad did not clear every gradient")
                    trainer.state.global_step += 1
                    trainer.state.epoch = (indices[-1] + 1) / 9
                    _callback(trainer, "on_step_end")
                    journal.add("group_finalized", branch=branch, update=update_index + 1, after_scheduler_lrs=after,
                        group_size=size, global_step=trainer.state.global_step)
            for branch in BRANCHES:
                _callback(trainers[branch], "on_epoch_end")
                _callback(trainers[branch], "on_train_end")
                observed = observers[branch].validate(require_backward_recompute=branch == "candidate",
                    require_all_layers=True, require_reference=dense_trainer.objective != "sft")
                journal.add("observer_validated", branch=branch, metadata=_json_copy(observed))
        _require(completed == {branch: 9 for branch in BRANCHES} and updates == {branch: 2 for branch in BRANCHES},
            "Incomplete paired backward/update counts")
        _require(_reference_versions(reference_model) == reference_start and not reference_model.training and
            all(value.grad is None for value in reference.values()), "Original frozen reference changed")
        _require([_batch_record(value) for value in batches] == batch_records and
            all(_signature(dict(trainers[branch].model.named_parameters())) == initial_signatures[branch] for branch in BRANCHES),
            "Original Parameter identity/layout or native inputs changed")
        invoke("verify_invariants", "final")
        return _json_copy({**result, "complete": True, "expected_parameters": expected_parameters,
            "fixture_count_is_not_native_proof": expected_parameters != 434, "completed_backwards": completed,
            "optimizer_updates": updates, "optimizer_step_attempts": attempts,
            "actual_group_sizes": [8, 1], "events": journal.events}, MAX_JOURNAL_BYTES)
    except BaseException as error:
        failure = {**result, "completed_backwards": completed, "optimizer_updates": updates,
            "optimizer_step_attempts": attempts, "partial_optimizer_state_may_have_changed": bool(any(attempts.values())),
            "error_type": type(error).__name__, "caller_must_preserve_private_partial_result_and_abort": True,
            "events": journal.events}
        raise PairedControllerFailure(failure) from error
