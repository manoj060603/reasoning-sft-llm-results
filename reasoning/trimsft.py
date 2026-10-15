"""Trimmed logit-gap SFT loss from arXiv:2609.09707."""
from typing import Optional

import torch


def trimsft_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    margin: float = 1.5,
    bandwidth: float = 0.8,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Apply detached Gaussian weights to CE using the gold-vs-runner-up logit gap."""
    if bandwidth <= 0:
        raise ValueError("TrimSFT bandwidth must be positive.")
    valid_mask = target != ignore_index
    if not valid_mask.any():
        return source.sum() * 0.0

    valid_source = source[valid_mask]
    valid_target = target[valid_mask]
    per_token_loss = torch.nn.functional.cross_entropy(valid_source, valid_target, reduction="none")

    with torch.no_grad():
        gold_logits = valid_source.gather(-1, valid_target.unsqueeze(-1)).squeeze(-1)
        top_values, top_indices = valid_source.topk(k=2, dim=-1)
        competitor_logits = torch.where(top_indices[:, 0] == valid_target, top_values[:, 1], top_values[:, 0])
        gaps = gold_logits - competitor_logits
        weights = torch.exp(-((gaps - margin) ** 2) / (2.0 * bandwidth**2))

    weighted_losses = per_token_loss * weights
    if num_items_in_batch is not None:
        total_loss = weighted_losses.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        return total_loss / num_items_in_batch
    return weighted_losses.mean()
