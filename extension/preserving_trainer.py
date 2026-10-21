"""Training-only HF hooks for the separate preservation-objective prototype.

No model loading, cache building, data ordering, CLI, scheduler or recipe is
provided. Callers must run in admitted compute and supply an already verified
fresh original student/reference and an explicit native terminal token suffix.
This does not establish initialization provenance from an in-memory model alone.

Supports one process and microbatch1, with pinned HF/Accelerate accumulation
semantics. HF evaluation/prediction and checkpoint RESUME are unsupported. Presentation receipts,
final weight scans, reference/file identities, and actual longest-sequence B200
memory admission remain requirements of a future runner. Full logits remain
resident (including Accelerate's possible FP32 output conversion).
Pre-update checks are not rollback: optimizer faults detected after its step
can already have mutated state. No transactional recovery is claimed here.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.metadata
import math
from numbers import Real

import torch
from transformers import Trainer, TrainerCallback

from extension.preserving_objectives import MODES, objective_loss


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_student(model):
    parameters = list(model.parameters())
    require(parameters and all(parameter.requires_grad and parameter.dtype == torch.float32
                               for parameter in parameters),
            "Full student parameters must all be trainable FP32 storage")
    require(not hasattr(model, "peft_config"), "A full fresh student cannot be an adapter wrapper")
    return parameters


def _storage(parameter):
    return str(parameter.device), parameter.untyped_storage().data_ptr()


def disjoint_models(student, reference):
    require(student is not reference, "Student and reference must be separate models")
    first, second = list(student.parameters()), list(reference.parameters())
    require(first and second and not {id(value) for value in first}.intersection(id(value) for value in second),
            "Student/reference parameter objects overlap")
    require(not {_storage(value) for value in first if value.numel()}.intersection(
            _storage(value) for value in second if value.numel()), "Student/reference parameter storage overlaps")


def check_reference(student, reference):
    if reference is None:
        return
    disjoint_models(student, reference)
    require(not reference.training and all(not parameter.requires_grad and parameter.grad is None and
            parameter.dtype in (torch.float32, torch.bfloat16) for parameter in reference.parameters()),
            "Reference must remain frozen, in eval mode, with FP32/BF16 storage and no gradients")


def _optimizer(optimizer):
    seen = set()
    while hasattr(optimizer, "optimizer"):
        require(id(optimizer) not in seen, "Cyclic optimizer wrapper")
        seen.add(id(optimizer))
        optimizer = optimizer.optimizer
    return optimizer


def check_optimizer(model, optimizer, reference=None):
    underlying = _optimizer(optimizer)
    require(underlying is not None and hasattr(underlying, "param_groups"), "A real optimizer is required")
    parameters = check_student(model)
    actual = [parameter for group in underlying.param_groups for parameter in group["params"]]
    require(len(actual) == len({id(value) for value in actual}) and
            {id(value) for value in actual} == {id(value) for value in parameters},
            "Optimizer must contain every student parameter exactly once and no other parameters")
    if reference is not None:
        check_reference(model, reference)
        require(not {id(value) for value in actual}.intersection(id(value) for value in reference.parameters()),
                "Reference parameters entered the optimizer")
    for state in underlying.state.values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                value = state[name]
                require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                        bool(torch.isfinite(value).all()), "Adam moments must remain finite FP32")
    return underlying


def finite_update_guard(model, optimizer, reference=None, clip_norm=None):
    """Reject known bad gradients/moments before step; no post-step rollback."""
    require(clip_norm is None or (isinstance(clip_norm, Real) and not isinstance(clip_norm, bool) and
            math.isfinite(float(clip_norm)) and clip_norm > 0), "clip_norm must be positive finite or None")
    check_optimizer(model, optimizer, reference)
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    require(parameters, "No student gradients are available before optimizer update")
    norm = torch.nn.utils.clip_grad_norm_(parameters,
        float("inf") if clip_norm is None else float(clip_norm), error_if_nonfinite=True)
    require(bool(torch.isfinite(norm)), "Nonfinite student gradient norm before optimizer update")
    return float(norm.detach().item())


def response_batch(inputs, eos_token_id, terminal_suffix):
    """Validate unchanged native targets and return an unshifted forward batch.

    Right padding may follow the declared native suffix and must be masked. The
    caller supplies the suffix, which can include a newline after EOS; no token
    is appended, rewritten, truncated, or guessed here.
    """
    require(isinstance(inputs, dict) and {"input_ids", "labels"} <= set(inputs) and
            set(inputs) <= {"input_ids", "labels", "attention_mask", "position_ids", "row_index"},
            "Unsupported model batch or missing input_ids/labels")
    require(type(eos_token_id) is int and eos_token_id >= 0 and isinstance(terminal_suffix, tuple) and
            bool(terminal_suffix) and all(type(value) is int and value >= 0 for value in terminal_suffix) and
            eos_token_id in terminal_suffix, "Supply the actual native terminal suffix including EOS")
    ids, labels = inputs["input_ids"], inputs["labels"]
    require(isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor) and
            ids.dtype == labels.dtype == torch.int64 and ids.ndim == labels.ndim == 2 and
            ids.shape == labels.shape and ids.shape[0] == 1 and ids.shape[1] >= 2 and ids.device == labels.device,
            "Expected same-device int64 [1,T] native tokens and labels")
    require(bool((ids >= 0).all()) and bool(((labels == -100) | (labels >= 0)).all()), "Invalid native token/label IDs")
    mask = inputs.get("attention_mask", torch.ones_like(ids))
    require(isinstance(mask, torch.Tensor) and mask.shape == ids.shape and mask.device == ids.device and
            mask.dtype in (torch.int64, torch.int32, torch.bool) and bool(((mask == 0) | (mask == 1)).all()),
            "Invalid native attention mask")
    real = int(mask[0].sum().item())
    require(real >= 2 and bool((mask[0, :real] == 1).all()) and bool((mask[0, real:] == 0).all()),
            "Only contiguous native tokens followed by right padding are supported")
    active = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    require(active.numel() >= len(terminal_suffix) and int(active[0]) > 0 and
            torch.equal(active, torch.arange(int(active[0]), real, device=ids.device)) and
            torch.equal(labels[0, active], ids[0, active]),
            "Require a masked prompt followed by one unchanged complete response and masked padding")
    tail = torch.tensor(terminal_suffix, dtype=torch.int64, device=ids.device)
    require(real >= len(terminal_suffix) and torch.equal(labels[0, real - len(terminal_suffix):real], tail),
            "Complete response is missing the declared native EOS/terminal suffix")
    forward = {key: inputs[key] for key in ("input_ids", "position_ids") if key in inputs}
    if "position_ids" in forward:
        value = forward["position_ids"]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and
                value.shape == ids.shape and value.device == ids.device and bool((value >= 0).all()),
                "Invalid position_ids")
    forward["attention_mask"] = mask
    return forward, labels, {"supervised_tokens": int(active.numel()),
        "eos_tokens": int((labels[0, active] == eos_token_id).sum().item()), "examples": 1}


def scalar_metrics(metrics):
    require(isinstance(metrics, dict), "Objective metrics must be a dictionary")
    result = {}
    for name, value in metrics.items():
        require(isinstance(value, torch.Tensor) and not value.requires_grad and value.grad_fn is None,
                "Objective metrics must be detached tensors")
        if name == "active_counts":
            require(value.ndim == 1 and value.numel() == 1, "Expected microbatch1 active_counts")
            continue
        require(value.numel() == 1 and bool(torch.isfinite(value)), "Objective metrics must be finite scalars")
        result[name] = float(value.item())
    return result


def eos_metrics(student, targets, reference, objective, metrics, eos_token_id, chunk_tokens):
    """Detached EOS-only vocabulary chunks; derive complementary token means.

    Metrics cover supervised EOS-ID positions, including the verified native
    suffix, not an inferred text stop. No full vocabulary tensor is copied.
    """
    with torch.no_grad():
        positions = torch.nonzero(targets[0] == eos_token_id, as_tuple=False).flatten()
        eos_count, total = int(positions.numel()), int(metrics["active_tokens"].item())
        require(eos_count > 0, "Shift removed every supervised EOS token")
        sums = {name: 0. for name in ("ce", "weighted_ce", "kl", "effective_weight")}
        for start in range(0, eos_count, chunk_tokens):
            selected = positions[start:start + chunk_tokens]
            logp = student[0].detach().index_select(0, selected).float().log_softmax(-1)
            ce = -logp[:, eos_token_id]
            weights, kl = torch.ones_like(ce), torch.zeros_like(ce)
            if objective != "sft":
                logq = reference[0].detach().index_select(0, selected).float().log_softmax(-1)
                q = logq.exp()
                if objective == "asft":
                    weights = logp[:, eos_token_id].exp()
                elif objective == "prior_anchored":
                    weights = logq[:, eos_token_id].exp()
                kl = (q * (logq - logp)).sum(-1)
            for name, value in (("ce", ce), ("weighted_ce", weights * ce),
                                ("kl", kl), ("effective_weight", weights)):
                sums[name] += float(value.sum().item())
        result = {"eos_tokens": eos_count, "non_eos_tokens": total - eos_count}
        for name, value in sums.items():
            result["eos_" + name] = value / eos_count
            result["non_eos_" + name] = ((float(metrics[name].item()) * total - value) /
                (total - eos_count)) if total > eos_count else 0.
        require(all(math.isfinite(value) for value in result.values()), "Nonfinite detached EOS diagnostics")
        return result


class _FiniteUpdates(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.owner.last_gradient_norm = finite_update_guard(model, optimizer,
            self.owner.reference_model, self.owner.clip_norm)

    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        check_optimizer(model, optimizer, self.owner.reference_model)


def make_trainer_class(trainer_base=Trainer):
    """Return the bounded integration class; no experiment recipe is selected."""
    class PreservingTrainer(trainer_base):
        def __init__(self, *args, objective, eos_token_id, terminal_suffix,
                     reference_model=None, beta=0.05, chunk_tokens=128,
                     checkpoint_chunks=True, clip_norm=None, **kwargs):
            require(importlib.metadata.version("transformers") == "4.57.3" and
                    importlib.metadata.version("accelerate") == "1.12.0", "Native accumulation hooks require pinned HF4.57.3/Accelerate1.12.0")
            require(isinstance(objective, str) and objective in MODES and (objective == "sft" or reference_model is not None),
                    "Known objective and frozen original reference are required")
            require(isinstance(beta, Real) and not isinstance(beta, bool) and math.isfinite(float(beta)) and beta >= 0 and
                    type(chunk_tokens) is int and chunk_tokens > 0 and type(checkpoint_chunks) is bool,
                    "Invalid explicit objective/chunk configuration")
            student = kwargs.get("model", args[0] if args else None)
            training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
            require(isinstance(student, torch.nn.Module) and kwargs.get("model_init") is None,
                    "Supply an already verified fresh original student; model_init is unsupported")
            check_student(student)
            require(training_args is not None and training_args.per_device_train_batch_size == 1 and
                    training_args.max_grad_norm == 0 and not training_args.fp16 and
                    training_args.label_smoothing_factor == 0 and not training_args.remove_unused_columns and
                    getattr(training_args.eval_strategy, "value", training_args.eval_strategy) == "no" and
                    not training_args.push_to_hub and
                    kwargs.get("compute_loss_func") is None,
                    "Require microbatch1, native clipping disabled, no FP16/smoothing/custom denominator/Hub upload, and retained batch columns")
            require(clip_norm is None or (isinstance(clip_norm, Real) and not isinstance(clip_norm, bool) and
                    math.isfinite(float(clip_norm)) and clip_norm > 0), "Invalid explicit clip_norm")
            # Validate the token suffix before any forward or reference mutation.
            require(type(eos_token_id) is int and eos_token_id >= 0 and isinstance(terminal_suffix, tuple) and
                    terminal_suffix and all(type(value) is int and value >= 0 for value in terminal_suffix) and
                    eos_token_id in terminal_suffix, "Explicit native EOS/terminal suffix required")
            if reference_model is not None:
                require(isinstance(reference_model, torch.nn.Module), "Reference must be a model")
                disjoint_models(student, reference_model)
            self.reference_model = reference_model
            self.objective, self.beta = objective, beta
            self.chunk_tokens, self.checkpoint_chunks = chunk_tokens, checkpoint_chunks
            self.eos_token_id, self.terminal_suffix, self.clip_norm = eos_token_id, terminal_suffix, clip_norm
            self.last_metrics, self.last_gradient_norm, self.training_forward_examples = {}, None, 0
            self._prototype_train_started = False
            super().__init__(*args, **kwargs)
            require(self.accelerator.num_processes == 1 and self.args.n_gpu <= 1 and
                    not self.is_deepspeed_enabled and not self.is_fsdp_enabled,
                    "Only one unsharded training process is supported")
            self.model_accepts_loss_kwargs = False
            if self.reference_model is not None:
                self.reference_model.to(self.args.device).eval().requires_grad_(False)
                if hasattr(self.reference_model, "config"):
                    self.reference_model.config.use_cache = False
            check_reference(self.model, self.reference_model)
            if self.optimizer is not None:
                check_optimizer(self.model, self.optimizer, self.reference_model)
            self.add_callback(_FiniteUpdates(self))

        def create_optimizer(self):
            result = super().create_optimizer()
            check_optimizer(self.model, self.optimizer, self.reference_model)
            return result

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.last_metrics = {}
            check_student(model)
            check_reference(model, self.reference_model)
            forward, labels, response = response_batch(inputs, self.eos_token_id, self.terminal_suffix)
            reference_logits = None
            if self.objective != "sft":
                context = torch.autocast(device_type=labels.device.type, dtype=torch.bfloat16) if self.args.bf16 else nullcontext()
                # no_grad creates normal detached tensors usable by checkpointed
                # autograd; inference_mode is intentionally not used.
                with torch.no_grad(), context:
                    reference_outputs = self.reference_model(**forward, use_cache=False)
                    reference_logits = (reference_outputs["logits"] if isinstance(reference_outputs, dict)
                                        else reference_outputs.logits).detach()
            outputs = model(**forward, use_cache=False)
            student_logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            require(isinstance(student_logits, torch.Tensor) and student_logits.ndim == 3 and
                    student_logits.shape[:2] == labels.shape, "Student must return all native positions without an internal shift")
            if reference_logits is not None:
                require(reference_logits.shape == student_logits.shape and not reference_logits.requires_grad and
                        not torch.is_inference(reference_logits), "Reference logits must be normal detached aligned tensors")
            loss, metrics = objective_loss(student_logits[:, :-1, :], labels[:, 1:],
                objective=self.objective, reference_logits=None if reference_logits is None else reference_logits[:, :-1, :],
                beta=self.beta, chunk_tokens=self.chunk_tokens, checkpoint_chunks=self.checkpoint_chunks)
            require(loss.numel() == 1 and bool(torch.isfinite(loss.detach())), "Nonfinite scalar training loss")
            diagnostics = eos_metrics(student_logits[:, :-1, :], labels[:, 1:],
                None if reference_logits is None else reference_logits[:, :-1, :],
                self.objective, metrics, self.eos_token_id, self.chunk_tokens)
            self.last_metrics = {**scalar_metrics(metrics), **response, **diagnostics}
            if model.training:
                self.training_forward_examples += response["examples"]
            if return_outputs:
                return loss, {"loss": loss.detach(), "metrics": dict(self.last_metrics)}
            return loss

        def evaluate(self, *args, **kwargs):
            raise NotImplementedError("These are training-only hooks; HF evaluation is not implemented")

        def predict(self, *args, **kwargs):
            raise NotImplementedError("These are training-only hooks; HF prediction is not implemented")

        def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
            require(resume_from_checkpoint is None, "Bound resume/presentation receipts are not implemented by these prototype hooks")
            require(not self._prototype_train_started, "Use a fresh trainer/model for a new attempt; repeated train calls are unsupported")
            self._prototype_train_started = True
            return super().train(resume_from_checkpoint=None, trial=trial,
                                 ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

    return PreservingTrainer
