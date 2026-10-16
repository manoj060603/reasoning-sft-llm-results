# Full 15K GEM experiment

## Status

Staged as the sequential successor to the full PriFT-mass run. It must not be submitted while another project GPU job is active.

## Motivation

The strongest completed ordinary SFT model increased seven-benchmark pass@8 from 49.55% to 51.52%, but reduced pass@1 and produced 1,964 truncated samples. PriFT preserves prior-supported target tokens, but its first five benchmark results show that preserving token support does not preserve enough sampled solution diversity or acquire the missing AIME/AMC skills.

GEM replaces one-hot cross-entropy gradients with a detached sharpened model distribution. With the published linear objective and beta 0.7, its token gradient is `softmax(logits / beta) - one_hot(target)`. This still teaches the verified target while distributing the positive update across plausible alternatives. The prespecified hypothesis is higher pass@8 diversity with less collapse than ordinary SFT.

## Fixed protocol

- Base: pinned Qwen2.5-3B-Instruct.
- Data: exactly 15,000 unique approved examples, SHA-256 `dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a`.
- Training: full parameters, response-only labels, 8,192-token cutoff, effective batch 8, 1,875 updates, learning rate 2e-6.
- GEM: linear objective, beta 0.7, detached sharpened distribution, memory-bounded vocabulary chunks.
- Evaluation: all nine benchmarks, temperature 0.6, top-p 0.95, eight samples, 32,768 maximum new tokens.
- No pilot, development gate, checkpoint selection, public-score tuning, or overlap with another GPU job.

## Decision rule

Report every benchmark and the seven-benchmark macro. The strict goal passes only if every Fall 2025 leader solve-count threshold is met. A weak completed run is a method result and will not be repeated unchanged.

## References

- GEM paper: <https://arxiv.org/abs/2408.16673>
- Official implementation reviewed at commit `e5b1430979fa12bf8ab7398b2ccc71dff795bbee`: <https://github.com/liziniu/GEM>

## Terminal result (job 1166712)

Training completed exactly 1,875 updates with finite logs, and the final checkpoint completed all nine benchmarks. The matched seven-benchmark macro was **21.81% pass@1 / 50.53% pass@8**, compared with **23.55% / 49.55%** for base. Thus GEM gained **+0.98 pass@8 points** while losing **−1.74 pass@1 points**. The pass@8 gain was distributed across MATH (+0.20 pp), CN Math (+6.67 pp), Kaoyan (+4.02 pp), OlympiadBench (+1.04 pp), and GPQA (+3.54 pp), while AMC, AIME25, AIME24, and Minerva declined.

The run produced 417 truncations across 15,792 samples, substantially fewer than ordinary full SFT's 1,964 but more than base's 141. It failed the strict Fall 2025 column gate, clearing only Minerva and GPQA. The result supports GEM as a partial diversity/termination improvement, but it does not supply the missing AIME/AMC skills and does not surpass the 51.52% ordinary full-SFT macro.

Receipts: `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-suite-comparison.json`, `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-domination.json`, and the complete raw generations under `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-suite/`.
