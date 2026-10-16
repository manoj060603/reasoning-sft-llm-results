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

## Terminal result (job 1169839)

The complete pool scorer produced finite NLL values for exactly **83,037 unique approved IDs**. The selector chose exactly **15,000 unique 8K-safe IDs**. Their dataset SHA-256 is `c083d54960969a56ce6944f17782d8676b0bf4830c434a8cf3b68dd2723858d3`. Training completed exactly **1,875 finite updates**, and the final checkpoint completed all nine benchmarks.

| Benchmark | Matched base pass@8 | Model-aware GEM pass@8 | Delta |
|---|---:|---:|---:|
| AIME24 | 23.33% | 16.67% | −6.67 pp |
| AIME25 | 13.33% | 13.33% | 0.00 pp |
| MATH | 85.20% | 85.40% | +0.20 pp |
| CN Math 2024 | 23.33% | 23.33% | 0.00 pp |
| Kaoyan | 56.78% | 61.81% | +5.03 pp |
| AMC | 77.50% | 65.00% | −12.50 pp |
| Minerva | 49.63% | 45.59% | −4.04 pp |
| OlympiadBench | 49.48% | 48.89% | −0.59 pp |
| GPQA | 76.77% | 80.30% | +3.54 pp |
| **Seven-benchmark macro** | **49.55%** | **48.32%** | **−1.23 pp** |

Pass@1 also fell from **23.55% to 21.00%**. The run generated 284 truncations across 15,792 samples, fewer than ordinary GEM's 417 and ordinary full SFT's 1,964. It passed the contextual Fall 2025 thresholds only on Minerva and GPQA. Difficulty-aware reselection therefore improved termination and Kaoyan but reduced broad accuracy. A plausible interpretation is that teacher-forced NLL identifies surprising teacher tokens, which can include noisy or unsupported trajectories; it is not a correctness or learnability oracle.

Receipts: `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-suite-comparison.json`, `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-domination.json`, `results/arc-20261011-snapshot-2316/training/lmf-gem-modelaware-15000-8k/`, and complete raw generations under `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-suite/`.
