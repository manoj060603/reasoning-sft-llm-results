"""Small reference objectives for the separate extension experiment.

Inputs are already causally ALIGNED: ``student_logits[b, t]`` predicts
``targets[b, t]``. The caller performs the causal shift and supplies ``-100``
for every position outside the supervised response. This module never shifts,
loads models, reads a dataset, changes targets, or starts work.

``asft`` follows detached online-probability weighting plus forward KL from a
frozen starting model (https://arxiv.org/abs/2509.23753). ``prior_anchored``
combines frozen target-probability weighting motivated by PriFT
(https://arxiv.org/abs/2606.09396) with the same forward KL. The latter is an
experimental combination, with no claim of novelty or measured superiority.

All modes first divide each example's sum by ALL its supervised-token count,
then average examples, matching the earlier local training runner. They NEVER
divide by the sum of token weights. This equal-example reduction is an ASFT
normalization adaptation, not a claim of exact reproduction of its recipe.

FP32 token chunks bound temporary vocabulary tensors. Optional non-reentrant
checkpointing recomputes chunk intermediates during backward, avoiding retention
of every chunk's full-vocabulary log-probabilities/reference probabilities.
Caller-supplied complete logits remain resident; this is not a streaming model
forward. All reference inputs and all probability weights are detached.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch.utils.checkpoint import checkpoint


MODES = frozenset({"sft", "asft", "prior_anchored"})
FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(bool(torch.isfinite(value).all().item()), name + " must be finite")


def _terms(student, labels, reference, mode, beta):
    """One active-token chunk; only the first return value carries gradients."""
    log_p = torch.log_softmax(student.float(), dim=-1)
    _finite(log_p, "FP32 student log probabilities")
    gold_log_p = log_p.gather(-1, labels[:, None]).squeeze(-1)
    ce = -gold_log_p
    if mode == "sft":
        weights = torch.ones_like(ce)
        kl = torch.zeros_like(ce)
    else:
        with torch.no_grad():
            log_q = torch.log_softmax(reference.detach().float(), dim=-1)
            _finite(log_q, "FP32 reference log probabilities")
            q = log_q.exp()
            if mode == "asft":
                weights = gold_log_p.detach().exp()
            else:
                weights = log_q.gather(-1, labels[:, None]).squeeze(-1).exp()
        # Forward KL(base || student), over the ENTIRE vocabulary at each
        # supervised position. A chosen-token log-ratio is not this quantity.
        kl = (q * (log_q - log_p)).sum(dim=-1)
    weighted_ce = weights * ce
    loss = (weighted_ce + beta * kl).sum()
    _finite(loss, "chunk loss")
    return (loss, ce.detach().sum(), weighted_ce.detach().sum(),
            kl.detach().sum(), weights.detach().sum())


def objective_loss(student_logits, targets, *, objective="sft",
                   reference_logits=None, beta=0.05, chunk_tokens=128,
                   checkpoint_chunks=True):
    """Return differentiable scalar loss and detached tensor metrics.

    Args:
        student_logits: Floating ``[examples, positions, vocabulary]`` tensor.
        targets: Same-device int64 ``[examples, positions]`` aligned target IDs;
            only ``-100`` is ignored. Every example needs a supervised token.
        objective: ``sft``, ``asft``, or experimental ``prior_anchored``.
        reference_logits: Same-shaped/device floating frozen-base logits,
            required for both anchored modes; gradients never reach this input.
        beta: Finite nonnegative forward-KL coefficient; ignored by plain SFT.
        chunk_tokens: Positive integer upper bound on positions per chunk.
        checkpoint_chunks: Recompute active vocabulary chunks in backward when
            autograd is enabled; does not change the mathematical objective.

    Metrics ``ce``, ``weighted_ce``, ``kl`` and ``effective_weight`` use the same
    equal-example reduction as loss. ``active_counts`` retains per-example int64
    counts and ``active_tokens`` their total. ``loss`` is a detached scalar.
    No metrics retain a graph. EOS-specific metrics belong to the caller.
    """
    _require(isinstance(objective, str) and objective in MODES,
             "unknown objective")
    _require(isinstance(beta, Real) and not isinstance(beta, bool) and
             math.isfinite(float(beta)) and float(beta) >= 0,
             "beta must be finite and nonnegative")
    beta = float(beta)
    _require(type(chunk_tokens) is int and chunk_tokens > 0,
             "chunk_tokens must be a positive integer")
    _require(type(checkpoint_chunks) is bool, "checkpoint_chunks must be bool")
    _require(isinstance(student_logits, torch.Tensor) and
             student_logits.dtype in FLOAT_DTYPES and student_logits.ndim == 3,
             "student_logits must be a floating [B,T,V] tensor")
    batch, positions, vocabulary = student_logits.shape
    _require(batch > 0 and positions > 0 and vocabulary > 0,
             "student_logits dimensions must be nonempty")
    _require(isinstance(targets, torch.Tensor) and targets.dtype == torch.int64 and
             targets.shape == (batch, positions),
             "targets must be int64 with shape [B,T]")
    _require(targets.device == student_logits.device,
             "targets and student_logits must share a device")
    if reference_logits is not None:
        _require(isinstance(reference_logits, torch.Tensor) and
                 reference_logits.dtype in FLOAT_DTYPES and
                 reference_logits.shape == student_logits.shape,
                 "reference_logits must be a same-shaped floating tensor")
        _require(reference_logits.device == student_logits.device,
                 "reference_logits and student_logits must share a device")
    _require(objective == "sft" or reference_logits is not None,
             "reference_logits required for anchored objectives")
    active = targets != -100
    counts = active.sum(dim=1)
    _require(bool((counts > 0).all().item()),
             "every example must contain active supervised tokens")
    _require(bool(((targets == -100) | ((targets >= 0) &
                                      (targets < vocabulary))).all().item()),
             "targets contain an invalid ID; only -100 may be ignored")

    losses, ce_rows, weighted_rows, kl_rows, weight_rows = [], [], [], [], []
    for example in range(batch):
        loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        ce_sum = loss_sum.detach().clone()
        weighted_sum = loss_sum.detach().clone()
        kl_sum = loss_sum.detach().clone()
        weight_sum = loss_sum.detach().clone()
        for start in range(0, positions, chunk_tokens):
            end = min(positions, start + chunk_tokens)
            student = student_logits[example, start:end]
            _finite(student, "student_logits, including masked positions")
            reference = None
            if reference_logits is not None:
                reference = reference_logits[example, start:end].detach()
                _finite(reference, "reference_logits, including masked positions")
            mask = active[example, start:end]
            count = int(mask.sum().item())
            if count == 0:
                continue
            labels = targets[example, start:end]
            if count != end - start:
                student, labels = student[mask], labels[mask]
                if reference is not None:
                    reference = reference[mask]
            if checkpoint_chunks and torch.is_grad_enabled() and student.requires_grad:
                parts = checkpoint(_terms, student, labels, reference, objective, beta,
                                   use_reentrant=False, preserve_rng_state=False)
            else:
                parts = _terms(student, labels, reference, objective, beta)
            loss_sum = loss_sum + parts[0]
            ce_sum = ce_sum + parts[1]
            weighted_sum = weighted_sum + parts[2]
            kl_sum = kl_sum + parts[3]
            weight_sum = weight_sum + parts[4]
        denominator = counts[example].to(dtype=torch.float32)
        losses.append(loss_sum / denominator)
        ce_rows.append(ce_sum / denominator)
        weighted_rows.append(weighted_sum / denominator)
        kl_rows.append(kl_sum / denominator)
        weight_rows.append(weight_sum / denominator)
    loss = torch.stack(losses).mean()
    _finite(loss, "objective loss")
    metrics = {
        "loss": loss.detach(),
        "ce": torch.stack(ce_rows).mean().detach(),
        "weighted_ce": torch.stack(weighted_rows).mean().detach(),
        "kl": torch.stack(kl_rows).mean().detach(),
        "effective_weight": torch.stack(weight_rows).mean().detach(),
        "active_counts": counts.detach(),
        "active_tokens": counts.sum().detach(),
    }
    return loss, metrics
