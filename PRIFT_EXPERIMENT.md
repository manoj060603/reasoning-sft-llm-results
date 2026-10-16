# PriFT-mass full 15K experiment

## Interim result and failure diagnosis

Training completed all 1,875 updates and the fixed nine-benchmark evaluation is still running. The first five completed columns already reject the strict all-column hypothesis:

| Benchmark | Matched base pass@8 | PriFT pass@8 | Delta | Fall 2025 target |
|---|---:|---:|---:|---:|
| AMC | 77.50% | 65.00% | -12.50 pp | 82.50% |
| AIME25 | 13.33% | 10.00% | -3.33 pp | 26.67% |
| AIME24 | 23.33% | 13.33% | -10.00 pp | 26.67% |
| MATH | 85.20% | 82.00% | -3.20 pp | 90.20% |
| CN Math 2024 | 23.33% | 26.67% | +3.33 pp | 46.67% |

PriFT improves CN Math over the matched base, but it misses every historical target above and materially regresses AMC, AIME24, and MATH. Truncations on these five columns total 209. The accuracy losses therefore cannot be explained solely by the severe nontermination seen in ordinary full SFT.

The leading mechanisms, pending the complete suite and token-retention audit, are:

1. The prior-support mask can discard low-support target tokens. Those tokens may be precisely the novel steps needed to teach harder AIME and AMC solutions.
2. Retained tokens still push toward one fixed demonstration. PriFT does not directly preserve the alternative solution paths that pass@8 rewards.
3. The 1e-5 learning rate is five times the 2e-6 rate used by the strongest completed ordinary full-SFT model, so the retained gradients may still move the model too far from useful base capabilities.
4. An 8K cutoff preserves complete targets but also exposes the model to long single-trajectory traces. Prior filtering does not directly supervise early stopping or penalize repetition.
5. The approved 15K pool may not cover the missing competition skills densely enough. The earlier compact-balanced run showed that coarse domain quotas alone do not solve this coverage problem.

These are causal hypotheses, not established facts. The final report will separate measured evidence from the token-retention and error analyses needed to distinguish them.

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
