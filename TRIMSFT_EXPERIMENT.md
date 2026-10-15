# TrimSFT full 15K experiment

## Objective

Test whether token-level supervision focused on moderately uncertain reasoning tokens can improve the assignment's pass@8 score without changing the sealed 15,000-example dataset, base model, or official nine-benchmark evaluation protocol.

This is one prespecified full run. Public benchmark results are not used to select a checkpoint, tune the weighting parameters, or stop training.

## Method and source

The implementation follows **Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning** (EMNLP 2026 Findings; [arXiv:2609.09707](https://arxiv.org/abs/2609.09707)). For each supervised assistant token, it computes the logit gap

\[
\Delta_t = z_{t,y_t} - \max_{v \ne y_t} z_{t,v}
\]

and applies the detached weight

\[
w_t = \exp\left(-\frac{(\Delta_t - 1.5)^2}{2(0.8)^2}\right)
\]

to token cross entropy. The margin 1.5 and bandwidth 0.8 are the paper's fixed main setting. Prompt labels remain masked. The weighting calculation is detached from autograd, so it changes which supervised tokens contribute to the loss without creating gradients through the weights.

## Assignment compliance

- Base: pinned Qwen2.5-3B-Instruct commit `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Data: exactly 15,000 unique examples from the existing approved `advanced` dataset.
- Dataset SHA-256: `dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a`.
- Trainer: repository-pinned LLaMA-Factory, full-parameter response-only SFT.
- Context: 4,096 training tokens; no packing; no prompt loss.
- Schedule: one epoch equivalent, 1,875 optimizer updates, effective batch 8, cosine decay, 3% warmup, BF16, gradient clipping 1.0.
- Learning rate: `1e-5`. This is a prespecified transfer choice between the prior conservative instruction-tuning rate (`2e-6`) and the paper's base-model rate (`5e-5`); it is not tuned on public benchmarks.
- Evaluation: all nine benchmarks, temperature 0.6, top-p 0.95, eight samples, 32,768 maximum new tokens, fixed system prompt.
- Selection: final checkpoint only. No development gate, public-suite checkpoint selection, or early stopping.

## Required evidence

The run is accepted as complete only if it records 15,000 rows, 15,000 unique selected IDs, the expected dataset hash, exactly 1,875 updates, finite logged losses and gradient norms, complete predictions for every benchmark, and matching evaluation manifests. Results are compared with the matched base and with the Fall 2025 per-column solve thresholds. Historical leaderboard equivalence remains unverified.

## Interpretation boundary

TrimSFT is supported by recent published evidence, but it is not guaranteed to beat every historical column. The strict claim will be made only if the final domination receipt shows at least one more solved problem than the Fall 2025 leader on each of the nine benchmarks.
