"""Unadmitted, training-only Qwen integration for the checkpointed head helper.

The original Qwen2ForCausalLM remains Trainer.model, its optimizer target and its
ordinary save/checkpoint target. Only compute_loss is replaced: the original
backbones are called directly under an explicit Accelerator.autocast context,
and the original tied heads are called inside checkpointed_head_loss. No root
forward is rebound, no model/module wrapper is installed, and no full-sequence
vocabulary output is requested or returned. The frozen preservation Trainer
owns accumulation, partial tails, callbacks, clipping, scheduling and backward.

This is new implementation preparation, outside the already admitted CPU36
helper scope. No actual Qwen/BF16/CUDA differential or memory gate has passed
for this integration. In-memory class, tensor-count and tie checks do NOT prove
original checkpoint bytes or initialization provenance; a separately admitted
native runner must prove those, observation hooks through backward/recompute,
all434 gradients/updates, standard save/reload, and whole-process headroom.

The supplied reference is retained only by the Trainer, outside the student's
registered modules, optimizer and checkpoint. It has separate frozen FP32
storage. SFT does not need a reference. The sole causal shift is below, after
the unchanged complete native response/EOS/tail has passed response_batch.
No model/data loading, selection, generation, CLI, job or automatic admission
is provided. HF evaluate/predict, resume and repeated train calls inherit the
frozen prototype's explicit refusals. No manual loss/accumulation scaling is
introduced, and num_items_in_batch is intentionally unused as in that contract.
"""
from __future__ import annotations

import math

import torch
from transformers import Trainer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

from extension.checkpointed_head_objectives import TERM_NAMES, checkpointed_head_loss
from extension.preserving_trainer import (
    check_reference,
    check_student,
    make_trainer_class as make_preserving_trainer_class,
    require,
    response_batch,
)


EXPECTED_PARAMETER_TENSORS = 434
OBSERVATION_SCOPE = "compute_loss_forward_only_not_backward_or_recompute_proof"


def _qwen_parts(model, *, name, trainable):
    """Check the narrow original-model boundary, never its checkpoint origin."""
    require(type(model) is Qwen2ForCausalLM and type(model.model) is Qwen2Model,
            name + " must be the original unwrapped Qwen2 causal model/backbone")
    require(not hasattr(model, "peft_config") and getattr(model, "hf_device_map", None) is None and
            not any(hasattr(module, "_orig_mod") or hasattr(module, "_hf_hook") for module in model.modules()),
            name + " cannot be an adapter, compiled, dispatched or offloaded model")
    parameters = list(model.parameters())
    require(len(parameters) == EXPECTED_PARAMETER_TENSORS and
            all(parameter.is_leaf and parameter.dtype == torch.float32 and
                parameter.requires_grad is trainable for parameter in parameters),
            name + " requires all434 original leaf Parameter tensors in FP32 storage")
    require(len({parameter.device for parameter in parameters}) == 1 and
            parameters[0].device.type in ("cpu", "cuda"),
            name + " requires one ordinary CPU/CUDA device without parameter sharding")
    require(model.config.model_type == "qwen2" and model.config.use_cache is False and
            model.config._attn_implementation == "sdpa",
            name + " requires the caller's unchanged Qwen/SDPA/cache-disabled configuration")
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    require(type(head) is torch.nn.Linear and head is model.lm_head and head.bias is None and
            isinstance(embedding, torch.nn.Embedding) and head.weight is embedding.weight and
            head.in_features == model.config.hidden_size and head.out_features == model.config.vocab_size,
            name + " requires the original bias-free head and exact tied embedding Parameter")
    require(not torch.is_inference(head.weight), name + " cannot hold inference Parameters")
    if not trainable:
        require(not model.training and all(parameter.grad is None for parameter in parameters),
                name + " must remain frozen eval-mode FP32 with no gradients")
    return model.model, head, embedding.weight


def _hidden(outputs, labels, *, name, detached):
    hidden = outputs.get("last_hidden_state") if isinstance(outputs, dict) else getattr(outputs, "last_hidden_state", None)
    require(isinstance(hidden, torch.Tensor) and hidden.ndim == 3 and
            hidden.shape[:2] == labels.shape and hidden.device == labels.device and
            hidden.dtype in (torch.float32, torch.bfloat16) and not torch.is_inference(hidden),
            name + " must return aligned normal FP32/BF16 final-normalized hidden states")
    if detached:
        require(not hidden.requires_grad and hidden.grad_fn is None,
                name + " hidden states must be detached by no_grad")
    return hidden


def _autocast_observation(device):
    enabled = bool(torch.is_autocast_enabled(device.type))
    dtype = torch.get_autocast_dtype(device.type) if enabled else None
    require(not enabled or dtype == torch.bfloat16, "Only the caller's BF16 autocast context is supported")
    return {"device_type": device.type, "enabled": enabled,
            "dtype": None if dtype is None else str(dtype)}


def _hidden_observation(hidden):
    return {"shape": [int(size) for size in hidden.shape], "dtype": str(hidden.dtype),
            "device_type": hidden.device.type, "requires_grad": bool(hidden.requires_grad),
            "is_inference": bool(torch.is_inference(hidden))}


class _HeadObservations:
    """Aggregate temporary public-hook observations without retaining tensors.

    Hooks are removed before compute_loss returns; these observations therefore
    do not prove autocast or projection shapes during checkpoint recomputation.
    A future native worker must install its own bounded hooks through backward.
    """
    def __init__(self, chunk_tokens, vocabulary):
        self.chunk_tokens, self.vocabulary = chunk_tokens, vocabulary
        self.calls, self.rows, self.max_rows = 0, 0, 0
        self.input_dtypes, self.output_dtypes, self.autocast_states = set(), set(), set()

    def hook(self, module, arguments, output):
        require(len(arguments) == 1 and isinstance(arguments[0], torch.Tensor) and
                arguments[0].ndim == 2 and isinstance(output, torch.Tensor) and output.ndim == 2 and
                output.shape[0] == arguments[0].shape[0] and output.shape[1] == self.vocabulary and
                0 < output.shape[0] <= self.chunk_tokens,
                "Original head must project only one bounded active-position chunk")
        observation = _autocast_observation(arguments[0].device)
        self.calls += 1
        self.rows += int(output.shape[0])
        self.max_rows = max(self.max_rows, int(output.shape[0]))
        self.input_dtypes.add(str(arguments[0].dtype))
        self.output_dtypes.add(str(output.dtype))
        self.autocast_states.add((observation["device_type"], observation["enabled"], observation["dtype"]))

    def snapshot(self):
        return {"scope": OBSERVATION_SCOPE, "calls": self.calls, "active_rows_projected": self.rows,
                "maximum_projection_rows": self.max_rows, "vocabulary_columns": self.vocabulary,
                "input_dtypes": sorted(self.input_dtypes), "output_dtypes": sorted(self.output_dtypes),
                "autocast": [{"device_type": device, "enabled": enabled, "dtype": dtype}
                             for device, enabled, dtype in sorted(self.autocast_states,
                                 key=lambda value: (value[0], value[1], str(value[2])))],
                "tensors_or_per_vocabulary_values_retained": False}


def _scalar_metrics(metrics, response):
    names = ("loss", "ce", "weighted_ce", "kl", "effective_weight", "active_tokens",
             "eos_tokens", "non_eos_tokens") + tuple(
                 prefix + name for prefix in ("eos_", "non_eos_") for name in TERM_NAMES)
    require(type(metrics) is dict and all(name in metrics for name in names),
            "Checkpointed objective is missing bounded native diagnostics")
    result = {}
    for name in names:
        value = metrics[name]
        require(isinstance(value, torch.Tensor) and value.numel() == 1 and
                not value.requires_grad and value.grad_fn is None and bool(torch.isfinite(value).all()),
                "Checkpointed metrics must remain finite detached microbatch1 scalars")
        result[name] = float(value.item())
    require(result["active_tokens"] == response["supervised_tokens"] and
            result["eos_tokens"] == response["eos_tokens"] and result["eos_tokens"] > 0 and
            result["active_tokens"] == result["eos_tokens"] + result["non_eos_tokens"],
            "The sole causal shift must preserve every supervised native EOS/tail target")
    require(all(math.isfinite(value) for value in result.values()), "Nonfinite scalar diagnostic")
    return {**result, **response}


def make_trainer_class(trainer_base=Trainer):
    """Keep the frozen preservation Trainer contract with a hidden-only loss.

    Constructor keywords are unchanged: objective, eos_token_id, terminal_suffix,
    reference_model=None, beta=.05, chunk_tokens=128, checkpoint_chunks=True and
    clip_norm=None. The supplied original must already have SDPA and use_cache
    disabled. A provided reference must have FP32 storage; the inherited
    constructor moves/freezes it without registering it under the student.
    FP32 CPU operation is an oracle seam, not a BF16/CUDA admission.
    """
    preserving_base = make_preserving_trainer_class(trainer_base)

    class CheckpointedHeadTrainer(preserving_base):
        def __init__(self, *args, **kwargs):
            original = kwargs.get("model", args[0] if args else None)
            reference = kwargs.get("reference_model")
            training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
            require(training_args is not None and not getattr(training_args, "torch_compile", False),
                    "Compiled training is outside this original-model integration")
            _qwen_parts(original, name="Student", trainable=True)
            if reference is not None:
                # The inherited constructor owns eval/requires_grad policy, but
                # must not silently accept lower-precision reference storage.
                require(isinstance(reference, torch.nn.Module) and
                        all(parameter.dtype == torch.float32 for parameter in reference.parameters()),
                        "Reference parameters require original FP32 storage")
            super().__init__(*args, **kwargs)
            self._checkpointed_head_original_student = original
            self.last_execution_metadata = {}
            self._original_parts(self.model)

        def _original_parts(self, model):
            require(model is self.model is self._checkpointed_head_original_student and
                    self.accelerator.num_processes == 1 and
                    getattr(self.accelerator.state.distributed_type, "value",
                            self.accelerator.state.distributed_type) == "NO" and
                    not self.is_deepspeed_enabled and not self.is_fsdp_enabled,
                    "Only the identical original student in one unsharded process is supported")
            require(self.accelerator.unwrap_model(model, keep_fp32_wrapper=True) is model,
                    "Unwrap must be an identity operation retaining the original AMP wrapper")
            require(self.accelerator.mixed_precision == ("bf16" if self.args.bf16 else "no"),
                    "Accelerator precision must match the explicit BF16 or FP32 oracle configuration")
            check_student(model)
            check_reference(model, self.reference_model)
            student = _qwen_parts(model, name="Student", trainable=True)
            reference = None
            if self.reference_model is not None:
                require(not any(module is self.reference_model for module in model.modules()),
                        "Reference cannot be registered inside the original student checkpoint")
                reference = _qwen_parts(self.reference_model, name="Reference", trainable=False)
                require(reference[1].weight.shape == student[1].weight.shape and
                        reference[1].weight.device == student[1].weight.device,
                        "Original student/reference tied heads must have identical shapes and device")
            return student, reference

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.last_metrics = {}
            self.last_execution_metadata = {"complete": False, "scope": OBSERVATION_SCOPE,
                "original_student_is_trainer_model": model is self.model,
                "full_sequence_logits_requested": False, "causal_shifts": 1,
                "backward_recompute_or_native_dtype_admission_established": False}
            student, reference = self._original_parts(model)
            forward, labels, response = response_batch(inputs, self.eos_token_id, self.terminal_suffix)
            student_backbone, student_head, student_embedding = student
            reference_hidden = None
            reference_head = reference_embedding = None
            student_observations = _HeadObservations(self.chunk_tokens, student_head.out_features)
            reference_observations = None
            handles = []
            try:
                # Calling .model directly bypasses Accelerate's wrapper on the
                # root causal model. Own this public autocast context explicitly.
                with self.accelerator.autocast():
                    context = _autocast_observation(labels.device)
                    require(context["enabled"] is bool(self.args.bf16),
                            "The direct-backbone path did not enter the configured autocast context")
                    self.last_execution_metadata["explicit_accelerator_autocast"] = context
                    if self.objective != "sft":
                        require(reference is not None, "Anchored objective requires a separate original reference")
                        reference_backbone, reference_head, reference_embedding = reference
                        with torch.no_grad():
                            reference_outputs = reference_backbone(**forward, use_cache=False,
                                output_attentions=False, output_hidden_states=False, return_dict=True)
                            reference_hidden = _hidden(reference_outputs, labels, name="Reference", detached=True)
                        self.last_execution_metadata["reference_hidden"] = _hidden_observation(reference_hidden)
                        del reference_outputs
                    outputs = student_backbone(**forward, use_cache=False, output_attentions=False,
                        output_hidden_states=False, return_dict=True)
                    student_hidden = _hidden(outputs, labels, name="Student", detached=False)
                    self.last_execution_metadata["student_hidden"] = _hidden_observation(student_hidden)
                    del outputs
                    handles.append(student_head.register_forward_hook(student_observations.hook))
                    if reference_head is not None:
                        reference_observations = _HeadObservations(self.chunk_tokens, reference_head.out_features)
                        handles.append(reference_head.register_forward_hook(reference_observations.hook))
                    # Sole causal shift. The pure helper performs no other shift.
                    loss, metrics = checkpointed_head_loss(student_hidden[:, :-1, :], labels[:, 1:],
                        student_head=student_head, student_embedding_weight=student_embedding,
                        eos_token_id=self.eos_token_id,
                        reference_hidden=None if reference_hidden is None else reference_hidden[:, :-1, :],
                        reference_head=reference_head, reference_embedding_weight=reference_embedding,
                        objective=self.objective, beta=self.beta, chunk_tokens=self.chunk_tokens,
                        checkpoint_chunks=self.checkpoint_chunks)
                require(isinstance(loss, torch.Tensor) and loss.numel() == 1 and
                        bool(torch.isfinite(loss.detach()).all()), "Nonfinite scalar checkpointed training loss")
                self.last_metrics = _scalar_metrics(metrics, response)
                if model.training:
                    self.training_forward_examples += response["examples"]
                self.last_execution_metadata["complete"] = True
                if return_outputs:
                    return loss, {"loss": loss.detach(), "metrics": dict(self.last_metrics)}
                return loss
            finally:
                for handle in handles:
                    handle.remove()
                self.last_execution_metadata["student_head"] = student_observations.snapshot()
                self.last_execution_metadata["reference_head"] = (
                    None if reference_observations is None else reference_observations.snapshot())

    return CheckpointedHeadTrainer
