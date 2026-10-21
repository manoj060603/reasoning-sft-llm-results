"""Untested pure checkpointed-head reference for the preservation experiment.

This is implementation preparation, not a passed numerical/memory admission or
a training runner. Inputs are ALREADY causally aligned: hidden[b, t] predicts
targets[b, t]. The caller owns the sole causal shift, unchanged complete native
EOS/tail labels, tokenizer/source identity, full-model provenance, original
parameter precision, and the active BF16-autocast context. No model/data loading,
selection, generation, backward/optimizer call, CLI, or framework patch occurs.

The ORIGINAL tied bias-free ``nn.Linear`` head projects at most128 positions in
the same non-reentrant checkpoint region as the FP32 full-vocabulary objective.
No full-sequence vocabulary tensor is returned or collected. Reference projection
is performed inside that region under no_grad, using detached reference hidden
states and a separate frozen FP32 head. Only loss retains a graph; all metrics
are detached scalars or one value per example. No extra head call computes EOS
metrics. FP32 head storage and FP32/BF16 hidden states are the deliberately narrow
scope; Linear subclasses/parametrizations/wrappers are refused. Forward hooks on
an original Linear remain the caller's responsibility.

ASFT uses detached online target probability plus forward KL (base||student),
https://arxiv.org/abs/2509.23753. Prior weighting is motivated by PriFT,
https://arxiv.org/abs/2606.09396; its combination here with KL is experimental,
with no novelty or measured-superiority claim. As in the frozen local prototype,
each example divides by ALL supervised-token counts, then examples are averaged;
the weight sum is never a denominator. This is an ASFT normalization adaptation.

The mathematical objective is unchanged; BF16 GEMM shape/summation/recomputation
can change rounding. Independent dense-loss, gradient, tied-weight, update and
actual native memory gates must pass before use. Parameter identity checks below
prove only the supplied head/embedding alias and separate head storage, not the
origin of a full model or hidden states. Do not mutate inputs/heads between this
call and backward. Pinned PyTorch2.9 non-reentrant checkpointing preserves the
caller's autocast settings; there is no mutable per-loop closure or manual
gradient-accumulation scaling here.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


MODES = frozenset({"sft", "ce_kl", "asft", "prior_anchored"})
HIDDEN_DTYPES = frozenset({torch.float32, torch.bfloat16})
MAX_CHUNK_TOKENS = 128
TERM_NAMES = ("ce", "weighted_ce", "kl", "effective_weight")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(bool(torch.isfinite(value).all().item()), name + " must be finite")


def _storage(parameter):
    return str(parameter.device), parameter.untyped_storage().data_ptr()


def _head(head, embedding_weight, *, name, trainable, device, hidden_size):
    """Validate the supplied original-Linear/tied-Parameter boundary only."""
    _require(type(head) is nn.Linear and head.bias is None,
             name + " must be an original bias-free nn.Linear, without a subclass wrapper")
    weight = head.weight
    _require(isinstance(weight, nn.Parameter) and isinstance(embedding_weight, nn.Parameter)
             and weight is embedding_weight and weight.is_leaf,
             name + " must use the exact supplied tied embedding Parameter")
    _require(weight.layout == torch.strided and weight.ndim == 2 and
             weight.shape[0] > 0 and weight.shape[1] == hidden_size and
             type(head.in_features) is int and head.in_features == hidden_size and
             type(head.out_features) is int and head.out_features == weight.shape[0],
             name + " has an invalid dense head shape")
    _require(weight.device == device and weight.dtype == torch.float32 and
             weight.requires_grad is trainable,
             name + " requires same-device FP32 storage and the declared gradient policy")
    _require(not torch.is_inference(weight), name + " cannot be an inference tensor")
    if not trainable:
        _require(not head.training and weight.grad is None,
                 name + " must remain eval-mode frozen with no gradients")
    _finite(weight, name + " weight")
    return weight


def _hidden(value, *, name, shape=None, device=None):
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided and
             value.dtype in HIDDEN_DTYPES and value.ndim == 3 and
             all(size > 0 for size in value.shape),
             name + " must be a nonempty dense FP32/BF16 [B,T,H] tensor")
    _require(value.device.type in ("cpu", "cuda") and not torch.is_inference(value),
             name + " must be a normal CPU/CUDA tensor")
    if shape is not None:
        _require(value.shape == shape and value.device == device,
                 name + " must match the student hidden shape and device")
    # Masked positions are checked too, before any active-row selection.
    _finite(value, name + ", including masked positions")


def _autocast(hidden):
    """Observe, never alter, the caller's compute context."""
    enabled = torch.is_autocast_enabled(hidden.device.type)
    if enabled:
        _require(torch.get_autocast_dtype(hidden.device.type) == torch.bfloat16,
                 "Only caller-owned BF16 autocast is supported")
    _require(hidden.dtype == torch.float32 or enabled,
             "BF16 hidden with FP32 head requires caller-owned BF16 autocast")


def _chunk_terms(student_hidden, labels, reference_hidden, student_weight,
                 reference_weight, student_head, reference_head, mode, beta,
                 eos_token_id):
    """One active chunk; only the first scalar return value retains a graph.

    Explicit weight arguments keep the tied leaf visible to checkpoint even if
    supplied hidden states do not require gradients. Projection still calls the
    original module. Module/weight objects are neither cloned nor detached.
    """
    _require(student_head.weight is student_weight,
             "Student tied head changed before checkpoint computation")
    logits = student_head(student_hidden)
    _require(logits.ndim == 2 and logits.shape == (labels.numel(), student_weight.shape[0]),
             "Student head returned an invalid bounded projection")
    _finite(logits, "student active logits")
    log_p = torch.log_softmax(logits.float(), dim=-1)
    _finite(log_p, "FP32 student log probabilities")
    gold_log_p = log_p.gather(-1, labels[:, None]).squeeze(-1)
    ce = -gold_log_p
    if mode == "sft":
        weights, kl = torch.ones_like(ce), torch.zeros_like(ce)
    else:
        _require(reference_head.weight is reference_weight,
                 "Reference tied head changed before checkpoint computation")
        with torch.no_grad():
            reference_logits = reference_head(reference_hidden.detach())
            _require(reference_logits.shape == logits.shape,
                     "Reference head returned an invalid bounded projection")
            _finite(reference_logits, "reference active logits")
            log_q = torch.log_softmax(reference_logits.float(), dim=-1)
            _finite(log_q, "FP32 reference log probabilities")
            q = log_q.exp()
            if mode == "asft":
                weights = gold_log_p.detach().exp()
            elif mode == "prior_anchored":
                weights = log_q.gather(-1, labels[:, None]).squeeze(-1).exp()
            else:  # ce_kl: isolate anchoring without probability weighting.
                weights = torch.ones_like(ce)
        # Exact forward KL over the entire vocabulary; no chosen-token proxy.
        kl = (q * (log_q - log_p)).sum(dim=-1)
    weighted_ce = weights * ce
    loss = (weighted_ce + beta * kl).sum()
    _finite(loss, "chunk loss")
    eos = labels == eos_token_id
    result = [loss]
    with torch.no_grad():
        for name, values in zip(TERM_NAMES, (ce, weighted_ce, kl, weights)):
            values = values.detach()
            _finite(values, "detached " + name)
            result.extend((values.sum(), values[eos].sum()))
    return tuple(result)


def _mean_with_zero(sums, counts):
    """Equal-example diagnostic mean; absent EOS/nonEOS has explicit zero."""
    denominator = counts.clamp_min(1).to(dtype=torch.float32)
    return torch.where(counts > 0, sums / denominator, torch.zeros_like(sums)).mean().detach()


def checkpointed_head_loss(student_hidden, targets, *, student_head,
                           student_embedding_weight, eos_token_id,
                           reference_hidden=None, reference_head=None,
                           reference_embedding_weight=None, objective="sft",
                           beta=0.05, chunk_tokens=128, checkpoint_chunks=True):
    """Return differentiable loss and bounded detached token/EOS metrics.

    ``student_hidden[B,T,H]`` and ``targets[B,T]`` are already aligned, without
    an internal causal shift. Only int64 ``-100`` labels are ignored. Every
    example needs active tokens; zero EOS counts are allowed by this pure math
    API and reported explicitly. The native caller must separately require its
    unchanged complete EOS/tail, as in the existing response_batch contract.

    ``student_head.weight is student_embedding_weight`` is required, with an
    analogous complete frozen reference tuple for anchored modes. SFT needs no
    reference; a supplied complete reference tuple is validated but not projected
    for SFT. Reference hidden inputs are detached internally. All head weights
    are FP32; compatible caller-owned BF16 autocast is required for BF16 hidden.

    Existing metric keys are loss, ce, weighted_ce, kl, effective_weight,
    active_counts[B] and active_tokens. Also return eos_counts[B], eos_tokens,
    non_eos_counts[B], non_eos_tokens, and for each TERM_NAMES entry: *_sums[B],
    eos_*_sums[B], non_eos_*_sums[B], eos_* and non_eos_*. Diagnostic means first
    divide each row by its relevant count (zero if absent), then average rows.
    Total loss/ordinary scalar metrics always use ALL active tokens per row.
    No metric has more than B elements, requires gradients, or retains a graph.

    chunk_tokens is a plain integer in[1,128]. Disabling checkpointing is only a
    prospective differential-oracle seam: it can retain intermediates from all
    chunks and is not a bounded native-memory admission. No backward or manual
    accumulation division is performed; the caller/Trainer owns those once.
    """
    _require(isinstance(objective, str) and objective in MODES, "unknown objective")
    _require(isinstance(beta, Real) and not isinstance(beta, bool) and
             math.isfinite(float(beta)) and float(beta) >= 0,
             "beta must be finite and nonnegative")
    beta = float(beta)
    _require(type(chunk_tokens) is int and 1 <= chunk_tokens <= MAX_CHUNK_TOKENS,
             "chunk_tokens must be a plain integer in[1,128]")
    _require(type(checkpoint_chunks) is bool, "checkpoint_chunks must be bool")
    _require(type(eos_token_id) is int and eos_token_id >= 0,
             "eos_token_id must be a nonnegative plain integer")
    _hidden(student_hidden, name="student_hidden")
    _autocast(student_hidden)
    batch, positions, hidden_size = student_hidden.shape
    _require(isinstance(targets, torch.Tensor) and targets.layout == torch.strided and
             targets.dtype == torch.int64 and targets.shape == (batch, positions) and
             targets.device == student_hidden.device and not torch.is_inference(targets),
             "targets must be normal same-device int64 [B,T]")
    student_weight = _head(student_head, student_embedding_weight, name="student_head",
                           trainable=True, device=student_hidden.device, hidden_size=hidden_size)
    vocabulary = student_weight.shape[0]
    _require(eos_token_id < vocabulary, "eos_token_id is outside the vocabulary")
    active = targets != -100
    counts = active.sum(dim=1)
    _require(bool((counts > 0).all().item()), "every example must contain active supervised tokens")
    _require(bool(((targets == -100) | ((targets >= 0) & (targets < vocabulary))).all().item()),
             "targets contain an invalid ID; only -100 may be ignored")

    reference_values = (reference_hidden, reference_head, reference_embedding_weight)
    has_reference = any(value is not None for value in reference_values)
    _require(not has_reference or all(value is not None for value in reference_values),
             "Supply the complete reference hidden/head/tied-weight tuple")
    _require(objective == "sft" or has_reference, "reference required for anchored objectives")
    reference_weight = None
    if has_reference:
        _hidden(reference_hidden, name="reference_hidden", shape=student_hidden.shape,
                device=student_hidden.device)
        _autocast(reference_hidden)
        reference_weight = _head(reference_head, reference_embedding_weight,
            name="reference_head", trainable=False, device=student_hidden.device, hidden_size=hidden_size)
        _require(reference_weight.shape == student_weight.shape and
                 reference_head is not student_head and reference_weight is not student_weight and
                 _storage(reference_weight) != _storage(student_weight),
                 "Student/reference head parameters and storage must be disjoint and same-shaped")
        reference_hidden = reference_hidden.detach()

    eos_counts = ((targets == eos_token_id) & active).sum(dim=1)
    non_eos_counts = counts - eos_counts
    loss_rows = []
    total_rows = {name: [] for name in TERM_NAMES}
    eos_rows = {name: [] for name in TERM_NAMES}
    for example in range(batch):
        loss_sum = student_hidden.new_zeros((), dtype=torch.float32)
        total_sums = {name: loss_sum.detach().clone() for name in TERM_NAMES}
        eos_sums = {name: loss_sum.detach().clone() for name in TERM_NAMES}
        for start in range(0, positions, chunk_tokens):
            end = min(positions, start + chunk_tokens)
            mask = active[example, start:end]
            if not bool(mask.any().item()):
                continue
            student = student_hidden[example, start:end][mask]
            labels = targets[example, start:end][mask]
            reference = None
            if objective != "sft":
                reference = reference_hidden[example, start:end][mask]
            arguments = (student, labels, reference, student_weight, reference_weight,
                         student_head, reference_head, objective, beta, eos_token_id)
            if checkpoint_chunks and torch.is_grad_enabled() and (
                    student.requires_grad or student_weight.requires_grad):
                parts = checkpoint(_chunk_terms, *arguments, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                parts = _chunk_terms(*arguments)
            loss_sum = loss_sum + parts[0]
            for index, name in enumerate(TERM_NAMES):
                total_sums[name] = total_sums[name] + parts[1 + 2 * index]
                eos_sums[name] = eos_sums[name] + parts[2 + 2 * index]
        loss_rows.append(loss_sum / counts[example].to(dtype=torch.float32))
        for name in TERM_NAMES:
            total_rows[name].append(total_sums[name])
            eos_rows[name].append(eos_sums[name])

    loss = torch.stack(loss_rows).mean()
    _finite(loss, "objective loss")
    metrics = {"loss": loss.detach(), "active_counts": counts.detach(),
               "active_tokens": counts.sum().detach(), "eos_counts": eos_counts.detach(),
               "eos_tokens": eos_counts.sum().detach(),
               "non_eos_counts": non_eos_counts.detach(),
               "non_eos_tokens": non_eos_counts.sum().detach()}
    for name in TERM_NAMES:
        sums = torch.stack(total_rows[name]).detach()
        eos_sums = torch.stack(eos_rows[name]).detach()
        non_eos_sums = sums - eos_sums
        metrics[name] = (sums / counts.to(dtype=torch.float32)).mean().detach()
        metrics[name + "_sums"] = sums
        metrics["eos_" + name + "_sums"] = eos_sums
        metrics["non_eos_" + name + "_sums"] = non_eos_sums
        metrics["eos_" + name] = _mean_with_zero(eos_sums, eos_counts)
        metrics["non_eos_" + name] = _mean_with_zero(non_eos_sums, non_eos_counts)
    for name, value in metrics.items():
        _require(not value.requires_grad and value.grad_fn is None and value.numel() <= batch,
                 "Metrics must remain bounded detached scalars or per-example vectors")
        _finite(value, "metric " + name)
    return loss, metrics
