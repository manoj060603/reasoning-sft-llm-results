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
