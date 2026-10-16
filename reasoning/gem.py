"""Memory-bounded Generalized Entropy Minimization loss.

This independently implements the linear GEM objective from arXiv:2408.16673.
The sharpened distribution is detached, so the gradient for a supervised token is
``softmax(logits / beta) - one_hot(target)``.
"""

from __future__ import annotations

import torch


class _LinearGEM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, beta, vocab_chunk, ignore_index):
        if not 0.0 < beta <= 1.0:
            raise ValueError("GEM beta must lie in (0, 1].")
        if vocab_chunk <= 0:
            raise ValueError("GEM vocabulary chunk must be positive.")

        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_labels = labels.reshape(-1).to(flat_logits.device)
        valid = flat_labels != ignore_index
        count = int(valid.sum().item())
        if count == 0:
            ctx.empty = True
            ctx.logits_shape = logits.shape
            ctx.logits_dtype = logits.dtype
            ctx.logits_device = logits.device
            return flat_logits.sum() * 0.0

        selected = flat_logits[valid]
        accumulator_dtype = torch.float64 if selected.dtype == torch.float64 else torch.float32
        scaled_max = selected.to(accumulator_dtype).amax(dim=-1, keepdim=True) / beta
        denominator = torch.zeros_like(scaled_max)
        weighted_logits = torch.zeros_like(scaled_max)
        for start in range(0, vocab, vocab_chunk):
            values = selected[:, start : start + vocab_chunk].to(accumulator_dtype)
            weights = torch.exp(values / beta - scaled_max)
            denominator.add_(weights.sum(dim=-1, keepdim=True))
            weighted_logits.add_((weights * values).sum(dim=-1, keepdim=True))

        targets = flat_labels[valid]
        target_logits = selected.gather(1, targets.unsqueeze(1)).to(accumulator_dtype)
        loss = (weighted_logits / denominator - target_logits).mean()

        ctx.empty = False
        ctx.beta = beta
        ctx.vocab_chunk = vocab_chunk
        ctx.ignore_index = ignore_index
        ctx.count = count
        ctx.accumulator_dtype = accumulator_dtype
        ctx.save_for_backward(logits, labels, scaled_max, denominator)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty:
            return (
                torch.zeros(ctx.logits_shape, dtype=ctx.logits_dtype, device=ctx.logits_device),
                None,
                None,
                None,
                None,
            )

        logits, labels, scaled_max, denominator = ctx.saved_tensors
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_labels = labels.reshape(-1).to(flat_logits.device)
        valid = flat_labels != ctx.ignore_index
        selected = flat_logits[valid]
        selected_grad = torch.empty_like(selected)
        scale = grad_output.to(ctx.accumulator_dtype) / ctx.count

        for start in range(0, vocab, ctx.vocab_chunk):
            values = selected[:, start : start + ctx.vocab_chunk].to(ctx.accumulator_dtype)
            q = torch.exp(values / ctx.beta - scaled_max) / denominator
            selected_grad[:, start : start + ctx.vocab_chunk] = (q * scale).to(selected_grad.dtype)

        targets = flat_labels[valid]
        rows = torch.arange(ctx.count, device=selected_grad.device)
        selected_grad[rows, targets] -= scale.to(selected_grad.dtype)
        flat_grad = torch.zeros_like(flat_logits)
        flat_grad[valid] = selected_grad
        return flat_grad.reshape_as(logits), None, None, None, None


def gem_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    beta: float = 0.7,
    vocab_chunk: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return mean linear GEM loss over non-ignored labels."""
    return _LinearGEM.apply(logits, labels, beta, vocab_chunk, ignore_index)
