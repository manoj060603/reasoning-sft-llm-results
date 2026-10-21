"""Four objective modes for unadmitted preservation-training preparation.

This module performs no shift, model loading, dataset access or job admission.
Inputs are already causally aligned: each [B,T,V] logit predicts the corresponding
int64 [B,T] target; only -100 is ignored. Every example needs an active token.

The three existing modes delegate to the frozen, tested prototype unchanged.
The new ``ce_kl`` control uses ordinary CE plus beta * KL(reference || student),
over the full vocabulary and supervised positions only. This isolates anchoring
from probability weighting. All modes divide by ALL active tokens per example,
then average examples, rather than dividing by weight sums.

FP32 token chunks and optional non-reentrant checkpointing bound temporary
vocabulary tensors, but caller-provided full logits remain resident. Metrics
and every reference/weight input are detached. No training recipe is chosen.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.utils.checkpoint import checkpoint

from extension import objectives as prototype


MODES = prototype.MODES | frozenset({"ce_kl"})
_require, _finite = prototype._require, prototype._finite


def _ce_kl_terms(student, labels, reference, beta):
    """One active-token chunk; only the loss return value retains a graph."""
    log_p = torch.log_softmax(student.float(), dim=-1)
    _finite(log_p, "FP32 student log probabilities")
    ce = -log_p.gather(-1, labels[:, None]).squeeze(-1)
    with torch.no_grad():
        log_q = torch.log_softmax(reference.detach().float(), dim=-1)
        _finite(log_q, "FP32 reference log probabilities")
        q = log_q.exp()
    kl = (q * (log_q - log_p)).sum(dim=-1)
    loss = (ce + beta * kl).sum()
    _finite(loss, "chunk loss")
    return loss, ce.detach().sum(), kl.detach().sum()


def objective_loss(student_logits, targets, *, objective="sft",
                   reference_logits=None, beta=0.05, chunk_tokens=128,
                   checkpoint_chunks=True):
    """Return scalar loss plus the prototype's detached metric dictionary.

    Modes: ``sft``, ``ce_kl``, ``asft`` and ``prior_anchored``. The existing
    three modes retain their precise prototype behavior and validation.
    ``ce_kl`` requires same-shaped/device floating frozen-reference logits,
    even when beta=0, and performs one combined CE/KL FP32 log-softmax pass
    per active chunk. Beta must be finite and nonnegative.

    Metrics are ``loss``, ``ce``, ``weighted_ce``, ``kl``, ``effective_weight``,
    ``active_counts`` and ``active_tokens``. In ce_kl, weighted_ce equals ce
    and effective_weight is one. No returned metric retains a graph.
    """
    if not isinstance(objective, str) or objective != "ce_kl":
        return prototype.objective_loss(student_logits, targets, objective=objective,
            reference_logits=reference_logits, beta=beta, chunk_tokens=chunk_tokens,
            checkpoint_chunks=checkpoint_chunks)

    _require(isinstance(beta, Real) and not isinstance(beta, bool) and
             math.isfinite(float(beta)) and float(beta) >= 0,
             "beta must be finite and nonnegative")
    beta = float(beta)
    _require(type(chunk_tokens) is int and chunk_tokens > 0,
             "chunk_tokens must be a positive integer")
    _require(type(checkpoint_chunks) is bool, "checkpoint_chunks must be bool")
    _require(isinstance(student_logits, torch.Tensor) and
             student_logits.dtype in prototype.FLOAT_DTYPES and student_logits.ndim == 3,
             "student_logits must be a floating [B,T,V] tensor")
    batch, positions, vocabulary = student_logits.shape
    _require(batch > 0 and positions > 0 and vocabulary > 0,
             "student_logits dimensions must be nonempty")
    _require(isinstance(targets, torch.Tensor) and targets.dtype == torch.int64 and
             targets.shape == (batch, positions), "targets must be int64 with shape [B,T]")
    _require(targets.device == student_logits.device,
             "targets and student_logits must share a device")
    _require(reference_logits is not None, "reference_logits required for anchored objectives")
    _require(isinstance(reference_logits, torch.Tensor) and
             reference_logits.dtype in prototype.FLOAT_DTYPES and
             reference_logits.shape == student_logits.shape,
             "reference_logits must be a same-shaped floating tensor")
    _require(reference_logits.device == student_logits.device,
             "reference_logits and student_logits must share a device")
    active = targets != -100
    counts = active.sum(dim=1)
    _require(bool((counts > 0).all().item()), "every example must contain active supervised tokens")
    _require(bool(((targets == -100) | ((targets >= 0) & (targets < vocabulary))).all().item()),
             "targets contain an invalid ID; only -100 may be ignored")

    losses, ce_rows, kl_rows = [], [], []
    for example in range(batch):
        loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        ce_sum, kl_sum = loss_sum.detach().clone(), loss_sum.detach().clone()
        for start in range(0, positions, chunk_tokens):
            end = min(positions, start + chunk_tokens)
            student = student_logits[example, start:end]
            reference = reference_logits[example, start:end].detach()
            _finite(student, "student_logits, including masked positions")
            _finite(reference, "reference_logits, including masked positions")
            mask = active[example, start:end]
            count = int(mask.sum().item())
            if count == 0:
                continue
            labels = targets[example, start:end]
            if count != end - start:
                student, labels, reference = student[mask], labels[mask], reference[mask]
            if checkpoint_chunks and torch.is_grad_enabled() and student.requires_grad:
                parts = checkpoint(_ce_kl_terms, student, labels, reference, beta,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                parts = _ce_kl_terms(student, labels, reference, beta)
            loss_sum = loss_sum + parts[0]
            ce_sum = ce_sum + parts[1]
            kl_sum = kl_sum + parts[2]
        denominator = counts[example].to(dtype=torch.float32)
        losses.append(loss_sum / denominator)
        ce_rows.append(ce_sum / denominator)
        kl_rows.append(kl_sum / denominator)
    loss = torch.stack(losses).mean()
    _finite(loss, "objective loss")
    ce = torch.stack(ce_rows).mean().detach()
    metrics = {
        "loss": loss.detach(), "ce": ce, "weighted_ce": ce,
        "kl": torch.stack(kl_rows).mean().detach(),
        "effective_weight": loss.new_ones(()).detach(),
        "active_counts": counts.detach(), "active_tokens": counts.sum().detach(),
    }
    return loss, metrics
