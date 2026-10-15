# PriFT-mass full 15K experiment

## Reason for the change

The first completed TrimSFT benchmark columns were already below both the matched base and historical targets. That rejects moderate-logit-gap weighting for the strict all-column objective. The next run uses a distinct signal and also corrects a training-context mismatch in the ordinary full-SFT run.

**PriFT: Prior-Support Guided Supervised Fine-Tuning** ([arXiv:2606.09396](https://arxiv.org/abs/2606.09396), [official code](https://github.com/wang-kee/PriFT)) reports large Avg@16 and Pass@16 improvements on mathematical reasoning by computing token weights from a frozen pretrained reference. Unlike DFT and TrimSFT, the weights do not move with the online optimization trajectory. This matters here because previous training repeatedly damaged strong base-model columns.

## Frozen prior-support objective

For each supervised assistant token, the frozen original Qwen model supplies distribution `p_ref`. PriFT-mass computes

\[
u_t=\sum_{v:p_{ref}(v)\le p_{ref}(y_t)}p_{ref}(v)
\]

and retains the token loss when `u_t >= 0.5`, exactly matching the official repository's default rule. The weighted loss is divided by the number of supervised tokens, rather than the number retained, matching the published implementation. Prompt tokens remain masked. Only the online model receives gradients.

## Prespecified training

- Original pinned Qwen2.5-3B-Instruct is both initialization and frozen reference.
- Exact approved 15,000-example `advanced` dataset, SHA-256 `dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a`.
- Full-parameter response-only training, one epoch equivalent, 1,875 updates, effective batch eight.
- `8,192`-token cutoff. The selection manifest was built with this bound. Earlier LLaMA-Factory full SFT used 4,096 and could truncate otherwise approved complete traces before their answer or end token.
- Learning rate `1e-5`, cosine decay, 3% warmup, BF16, gradient clipping 1.0.
- Final checkpoint only. No public benchmark tuning, checkpoint selection, development gate, or early stopping.

## Evaluation and decision rule

Run all nine assignment benchmarks with temperature 0.6, top-p 0.95, eight samples, 32,768 maximum new tokens, and the fixed assignment prompt. Report pass@1, pass@8, termination diagnostics, paired uncertainty, the seven-benchmark macro, and strict integer solve thresholds for every Fall 2025 column. Historical equivalence remains unverified.

The model is said to beat every historical column only if the domination receipt passes all nine integer thresholds. Paper results motivate the experiment but do not guarantee this outcome on Qwen2.5-3B-Instruct or this dataset.
