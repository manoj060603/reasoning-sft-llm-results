# Model-aware 15K GEM experiment

## Motivation

The prior full runs changed targets or losses while retaining the same route-selected 15K training identities. That dataset was selected for TF-IDF route diversity and static quality checks, but it contains no score for what the pinned Qwen2.5-3B-Instruct base finds easy, learnable, or unsupported. This is now the most concrete unresolved variable: GEM improved MATH, CN Math, and Kaoyan while AIME and AMC remained below base, so preserving output diversity alone did not supply the missing problem coverage.

[Select2Reason](https://arxiv.org/abs/2505.17266) reports that difficulty-aware selection can match or improve full-data long-CoT tuning with a small selected fraction. [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284) emphasizes unique prompt coverage and careful filtering. [GEM](https://arxiv.org/abs/2408.16673) directly targets the output diversity relevant to pass@8. These results motivate the experiment; they do not guarantee transfer to this 3B model or evaluator.

## Frozen design

- Source: the complete sealed and decontaminated 83,037-row approved pool.
- Difficulty proxy: teacher-forced mean NLL under the pinned original base over the first 512 response tokens. This is a model-relative learning signal, not a correctness or solve-probability estimate.
- Eligibility: complete boxed answer, at least 128 response tokens, repeated-8-gram fraction at most 0.35, and complete Qwen serialization within the 8,192-token cutoff.
- Selection: exact 15,000 unique prompts. Allocate across the existing 128 question clusters with square-root quotas, then favor moderate-high NLL within response-length buckets, centered at the prespecified 65th percentile. The selector and seed are frozen before evaluation.
- Training: full-parameter response-only linear GEM from the original base, beta 0.7, learning rate 2e-6, effective batch eight, exactly 1,875 updates, and one final checkpoint.
- Evaluation: all nine benchmarks, temperature 0.6, top-p 0.95, eight samples, and 32,768 maximum new tokens.

No benchmark prompt, answer, score, or error category enters scoring or selection. There is no pilot, development gate, public-score tuning, or checkpoint selection.

## Decision rule

The run succeeds on the strict project objective only if the final model clears every prespecified Fall 2025 solve-count column under this evaluator. Complete nine-benchmark results, generation diagnostics, manifests, and uncertainty intervals are retained even if the strict gate fails. Historical runtime equivalence remains unverified, so a numeric clearance is reported as an internal reproduction result rather than an official leaderboard win.
