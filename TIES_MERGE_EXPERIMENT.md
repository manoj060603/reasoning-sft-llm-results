# Fixed TIES merge experiment

## Motivation

The strongest full SFT checkpoint has the best complete seven-benchmark macro (51.52%) but produces 1,964 truncated samples. The independently trained GEM checkpoint has a lower macro (50.53%) but only 417 truncations and preserves more sampling diversity. Their errors and generation behavior are complementary, while both descend from the same pinned Qwen2.5-3B-Instruct base and use the same sealed 15,000 unique training prompts.

[TIES-Merging](https://arxiv.org/abs/2306.01708) removes small task-vector changes, resolves conflicting update signs, and merges only sign-aligned updates. [Model Soups](https://arxiv.org/abs/2203.05482) provides broader evidence that weight averaging compatible fine-tuned models can improve robustness without additional inference cost. These papers motivate the experiment; they do not imply that this merge will beat either constituent here.

## Frozen design

- Base: pinned original Qwen2.5-3B-Instruct.
- Experts: the completed ordinary full-SFT and ordinary GEM checkpoints.
- Merge: layerwise TIES, top 20% task-vector magnitude retained per tensor, magnitude-based sign election, disjoint mean, scale 1.0.
- Large-tensor cutoff: deterministic uniform sample of at most 1,000,000 entries per task vector.
- Selection: no development or public benchmark score is used to set density, scale, experts, or checkpoints.
- Evaluation: one merged checkpoint on all nine benchmarks at temperature 0.6, top-p 0.95, eight samples, and 32,768 maximum new tokens.

The model remains derived from the assignment's sealed 15,000 unique prompts. The merge performs no additional training and adds no benchmark data.

## Decision rule

The strict goal is satisfied only if this single fixed checkpoint clears every prespecified Fall 2025 solve-count column under the internal evaluator. Historical runtime and dataset equivalence remain unverified, so even a numerical clearance would be an internal reproduction result rather than an official leaderboard claim.


## Fixed TIES merge terminal result (job 1174986, October 16, 2026)

The merge completed with 434 finite BF16 tensors from the pinned base, ordinary full SFT, and ordinary GEM checkpoints. It used the frozen top-20% layerwise TIES recipe at scale 1.0. The nine-benchmark run completed with 8 predictions for all 1,974 problems (15,792 samples); its manifest and question IDs match the base, and the exact protocol was temperature 0.6, top-p 0.95, and 32,768 maximum new tokens.

| Benchmark | Matched base pass@8 | TIES pass@8 | Paired delta and 95% CI | Solves | Strict target | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| AIME24 | 23.33% | 16.67% | -6.67 pp [-23.33, +10.00] | 5/30 | 8/30 | no |
| AIME25 | 13.33% | 16.67% | +3.33 pp [-6.67, +16.67] | 5/30 | 8/30 | no |
| MATH | 85.20% | 84.00% | -1.20 pp [-3.60, +1.20] | 420/500 | 451/500 | no |
| CN Math 2024 | 23.33% | 33.33% | +10.00 pp [-10.00, +30.00] | 10/30 | 14/30 | no |
| Kaoyan | 56.78% | 58.29% | +1.51 pp [-5.03, +8.04] | 116/199 | 138/199 | no |
| AMC | 77.50% | 75.00% | -2.50 pp [-12.50, +7.50] | 30/40 | 33/40 | no |
| Minerva | 49.63% | 49.26% | -0.37 pp [-4.78, +4.04] | 134/272 | 95/272 | yes |
| OlympiadBench | 49.48% | 50.81% | +1.33 pp [-1.33, +4.15] | 343/675 | 385/675 | no |
| GPQA | 76.77% | 77.78% | +1.01 pp [-5.56, +8.08] | 154/198 | 121/198 | yes |

The seven-benchmark pass@8 macro is **51.59%**, versus **49.55%** for the matched base and **51.52%** for ordinary full SFT. The pass@1 macro is **23.79%**, versus **23.55%** for base. The small macro gain is not the objective: TIES clears only **2/9** strict columns. It has 602/15,792 truncated samples, mean output length 2,313 tokens, and mean repeated-8-gram fraction 0.0653. The paired intervals are per benchmark and unadjusted for multiple comparisons.

The result indicates that merging the two existing experts does not supply the missing skills: MATH remains 31 solves short, OlympiadBench 42 short, Kaoyan 22 short, and the small AIME/CN/AMC sets also miss. Fewer truncations than full SFT are insufficient to close these gaps. This candidate is complete and will not be retuned on public benchmarks. Raw predictions, matched manifest, comparison, strict-gate receipt, and merge provenance are in `results/arc-20261011-snapshot-2316/evaluations/lmf-ties-sft-gem-15k-*` and `results/arc-20261011-snapshot-2316/ties-sft-gem-15k/ties_merge.json`. Historical runtime equivalence remains unverified.
