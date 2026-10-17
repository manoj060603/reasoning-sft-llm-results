"""Bounded-memory implementation of the pinned Transformers plain causal CE.

Same float32 token CE, masking, shift and denominator; no reweighting.
"""
import torch
import torch.nn.functional as F


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, denominator, ignore_index, chunk):
        ctx.save_for_backward(logits, labels, denominator)
        ctx.ignore_index, ctx.chunk = ignore_index, chunk
        loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        for start in range(0, labels.numel(), chunk):
            loss += F.cross_entropy(logits[start:start+chunk].float(), labels[start:start+chunk],
                                    ignore_index=ignore_index, reduction='sum')
        return loss / denominator

    @staticmethod
    def backward(ctx, scale):
        logits, labels, denominator = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for start in range(0, labels.numel(), ctx.chunk):
            target = labels[start:start+ctx.chunk]
            valid = target != ctx.ignore_index
            probs = logits[start:start+ctx.chunk].float().softmax(-1)
            rows = torch.arange(len(target), device=target.device)[valid]
            probs[rows, target[valid]] -= 1
            probs[~valid] = 0
            grad[start:start+ctx.chunk] = (probs * (scale / denominator)).to(logits.dtype)
        return grad, None, None, None, None


def causal_ce(logits, labels, vocab_size, num_items_in_batch=None,
              ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    flat = logits.view(-1, vocab_size)
    target = shift_labels.reshape(-1).to(logits.device)
    denominator = (target != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, device=logits.device, dtype=torch.float32)
    return _ChunkedCE.apply(flat, target, denominator, ignore_index, 128)


def install():
    from transformers.loss.loss_utils import LOSS_MAPPING
    LOSS_MAPPING['ForCausalLM'] = causal_ce
