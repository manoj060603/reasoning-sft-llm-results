---
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-3B-Instruct
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE
tags:
- reasoning
- supervised-fine-tuning
---

# Qwen2.5-3B — Cross-fitted Transfer Gain with Coverage

**Built with Qwen.** The published weight files were modified through supervised fine-tuning. The repository retains the upstream research-license agreement and attribution notice.

Fresh full-parameter SFT from the original base; no previous team model initializes this checkpoint. Base revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Training uses 15,000 original pairs, 8,192 tokens, 1,875 updates, global batch eight, AdamW at 2e-6, cosine decay and 3% warmup. Parameters and optimizer moments use FP32; computation uses BF16 autocast. Inference evaluation loads BF16. The objective averages supervised-token CE within each sequence, then averages the examples in each update.

Actual training prefixes: 0 truncated rows; 0 rows lack a supervised EOS.

Evaluation: eight samples, temperature 0.6, top-p 0.95, maximum 32768 new tokens, native 32768 total context, unchanged supplied grader. The following table covers the eight supplied tasks.

| Benchmark | Original p@1 / p@8 | Random p@1 / p@8 | Advanced p@1 / p@8 |
| --- | ---: | ---: | ---: |
| aime | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% |
| amc | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% |
| math | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% |
| cn_math_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% |
| kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% |
| minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% |
| olympiadbench | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% |
| gpqa | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% |
| Official eight macro | 29.30% / 52.86% | 16.23% / 36.70% | 18.79% / 42.95% |

Evidence: raw eight-sample predictions verified.

Supplementary AIME25 is complete and matched; its seven-task internal macro is separately recorded in evaluation_summary.json.

The random control is uniform within the broad prepared source pool after deduplication, benchmark/source-heldout, prompt-capacity, and reserved-probe exclusions; it is not a literal uniform draw over the untouched original 100K rows. Original exported responses are unchanged. Random training may use declared 8K prefixes without an invented EOS. Advanced targets must fit completely. The advanced method combines eligibility filtering, reservoir construction, transfer gain, and coverage. Separate eligible-random and coverage-only ablations are needed to isolate their contributions. Relative teacher-forced loss reduction is a proxy, not mathematical verification or proof of causal data utility. Global novelty, state-of-the-art performance, and equivalence to a historical leaderboard are not established.

The upstream 3B base uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). Its model-card license identifier is `other`, with license name `qwen-research`. Publication metadata is staged; repository URLs and upload completion are not inferred.
