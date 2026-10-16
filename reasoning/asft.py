"""Response-only anchored SFT objective from arXiv:2509.23753.

The reference is frozen.  DFT weights are detached online gold-token
probabilities; the KL term is KL(reference || online), restricted to supervised
response tokens.  Chunking bounds temporary float32 vocabulary allocations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def anchored_loss(logits: torch.Tensor, reference_logits: torch.Tensor,
                  labels: torch.Tensor, *, kl_weight: float = 0.03,
                  token_chunk: int = 128, ignore_index: int = -100) -> torch.Tensor:
    if logits.shape != reference_logits.shape or logits.shape[:-1] != labels.shape:
        raise ValueError("online, reference, and label shapes do not match")
    if kl_weight < 0 or token_chunk < 1:
        raise ValueError("invalid KL weight or token chunk")
    vocab = logits.shape[-1]
    online = logits.reshape(-1, vocab)
    prior = reference_logits.reshape(-1, vocab)
    targets = labels.reshape(-1)
    valid = torch.nonzero(targets != ignore_index, as_tuple=True)[0]
    if len(valid) == 0:
        return online.sum() * 0
    total = online.new_zeros((), dtype=torch.float32)
    for start in range(0, len(valid), token_chunk):
        indices = valid[start:start + token_chunk]
        current = online.index_select(0, indices).float()
        frozen = prior.index_select(0, indices).float().detach()
        gold = targets.index_select(0, indices)
        logp = F.log_softmax(current, dim=-1)
        gold_logp = logp.gather(1, gold[:, None]).squeeze(1)
        dft = -gold_logp * gold_logp.detach().exp()
        ref_p = F.softmax(frozen, dim=-1)
        kl = (ref_p * (ref_p.clamp_min(1e-30).log() - logp)).sum(dim=-1)
        total = total + (dft + kl_weight * kl).sum()
    return total / len(valid)
