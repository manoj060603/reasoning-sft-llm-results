"""Frozen-reference token weighting from PriFT (arXiv:2606.09396)."""

from __future__ import annotations

import torch


def reference_target_probs(reference_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(reference_logits.float(), dim=-1)
    return probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)


def reference_mass(
    reference_logits: torch.Tensor, labels: torch.Tensor, vocab_chunk: int = 4096
) -> torch.Tensor:
    """Return sum_v p_ref(v) 1[p_ref(v) <= p_ref(y)] using bounded memory."""
    if vocab_chunk <= 0:
        raise ValueError("PriFT vocabulary chunk must be positive.")
    target_logit = reference_logits.gather(1, labels.unsqueeze(-1)).float()
    max_logit = reference_logits.max(dim=-1, keepdim=True).values.float()
    target_exp = torch.exp(target_logit - max_logit)
    denominator = torch.zeros_like(target_exp)
    supported_mass = torch.zeros_like(target_exp)
    for start in range(0, reference_logits.shape[-1], vocab_chunk):
        values = torch.exp(reference_logits[:, start : start + vocab_chunk].float() - max_logit)
        denominator += values.sum(dim=-1, keepdim=True)
        supported_mass += torch.where(values <= target_exp, values, torch.zeros_like(values)).sum(
            dim=-1, keepdim=True
        )
    return (supported_mass / denominator).squeeze(-1)


def prift_cross_entropy(
    online_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    method: str = "mass",
    mass_threshold: float = 0.5,
    vocab_chunk: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute PriFT-weighted CE, normalized by all supervised tokens as published."""
    if method not in ("prob", "mass"):
        raise ValueError("PriFT method must be `prob` or `mass`.")
    if not 0.0 <= mass_threshold <= 1.0:
        raise ValueError("PriFT mass threshold must lie in [0, 1].")
    vocab = online_logits.shape[-1]
    flat_online = online_logits.reshape(-1, vocab)
    flat_reference = reference_logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1).to(flat_online.device)
    valid = flat_labels != ignore_index
    if not valid.any():
        return flat_online.sum() * 0.0

    logits = flat_online[valid].float()
    ref_logits = flat_reference[valid]
    targets = flat_labels[valid]
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    with torch.no_grad():
        if method == "prob":
            weights = reference_target_probs(ref_logits, targets)
        else:
            weights = (reference_mass(ref_logits, targets, vocab_chunk) >= mass_threshold).float()
    return (losses * weights).sum() / valid.sum()
