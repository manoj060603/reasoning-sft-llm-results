---
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-3B-Instruct
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE
tags:
- reasoning
- model-merge
- post-hoc
---

# Qwen2.5-3B: fixed base75 / advanced25 post-hoc recovery

**Built with Qwen.** This is a separate exploratory weight interpolation, not a new SFT run. It preserves the original random/advanced assignment checkpoints and results.

`theta_recovery = 0.75 * theta_original_base + 0.25 * theta_advanced_final`, computed in FP32. Original parent: Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Advanced parent: [https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019/tree/550bdcc89a9352335d6030d6834fcff0f94d697f), commit `550bdcc89a9352335d6030d6834fcff0f94d697f`. Exact parent shard hashes, merge code/plan identities and output hashes are sealed in the provenance. All six published FP32 weight shards and model/tokenizer sidecars retain their evaluated bytes. Inference evaluation used BF16; publication performs no weight conversion.

The recovery was proposed after viewing primary benchmark regressions and long-response diagnostics. One coefficient was fixed before new generation; there was no alpha sweep. A deterministic source-only 64-question screen passed the fixed teacher-answer-agreement/stopping gate. Teacher agreement is not independently verified truth. This development screen used one sample and a 4K output limit; it is separate from the benchmark protocol below.

All nine benchmark tasks are verified from eight raw samples per question with matched data, ordered IDs, grader, seed 0, temperature 0.6, top-p 0.95, maximum 32768 new tokens and native 32768 total context. AIME25 is supplementary and excluded from the official-eight macro.

| Benchmark | Original p@1 / p@8 | Random p@1 / p@8 | Advanced p@1 / p@8 | Recovery p@1 / p@8 |
| --- | ---: | ---: | ---: | ---: |
| aime | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% | 6.25% / 23.33% |
| amc | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% | 39.06% / 75.00% |
| math | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% | 65.33% / 84.60% |
| cn_math_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% | 12.50% / 33.33% |
| kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% | 21.11% / 54.27% |
| minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% | 29.46% / 48.16% |
| olympiadbench | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% | 27.41% / 49.48% |
| gpqa | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% | 30.43% / 76.26% |
| aime25 | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% | 2.50% / 10.00% |
| Official eight macro | 29.30% / 52.86% | 16.23% / 36.70% | 18.79% / 42.95% | 28.94% / 55.56% |

These are measured results from one exploratory candidate. They establish neither significance, general superiority, novelty nor state-of-the-art performance. Historical leaderboard equivalence is unverified. The whole advanced parent policy combines eligibility, transfer gain and coverage; eligible-random/coverage-only ablations and repeated seeds were not run.

The Qwen Research License applies. NOTICE retains the required upstream attribution and the original recovery notice verbatim; its exact original bytes are separately preserved under provenance.
