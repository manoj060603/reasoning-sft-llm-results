# Strict all-benchmark objective

The next objective is stronger than improving the seven-benchmark macro: one fixed model must strictly exceed the Fall 2025 leader in every displayed pass@8 column under the assignment sampler.

| Benchmark | Historical leader | Minimum strict target | Current full SFT | Additional solves needed |
|---|---:|---:|---:|---:|
| AIME24 | 7/30 (0.233 shown) | 8/30 = 0.267 | 5/30 | 3 |
| AIME25 | 7/30 (0.233 shown) | 8/30 = 0.267 | 6/30 | 2 |
| MATH | 450/500 = 0.900 | 451/500 = 0.902 | 425/500 | 26 |
| CN Math 2024 | 13/30 (0.433 shown) | 14/30 = 0.467 | 10/30 | 4 |
| Kaoyan | about 137/199 = 0.688 | 138/199 = 0.693 | 119/199 | 19 |
| AMC | 32/40 = 0.800 | 33/40 = 0.825 | 28/40 | 5 |
| Minerva | about 94/272 = 0.346 | 95/272 = 0.349 | 133/272 | already above |
| OlympiadBench | about 384/675 = 0.569 | 385/675 = 0.570 | 333/675 | 52 |
| GPQA | about 120/198 = 0.606 | 121/198 = 0.611 | 157/198 | already above |

The printed historical decimals do not expose their exact underlying counts. The strict targets above conservatively convert each displayed score to the next integer solve count. Historical dataset and runtime equivalence remain unverified.

## Full-run recipe

The current full SFT learned useful mathematical behavior but also learned the teacher's long hidden deliberation style, producing 1,964 truncations. The new treatment changes the supervision rather than merely adding more optimization:

1. Start from the original Qwen2.5-3B-Instruct model.
2. Use 15,000 unique, decontaminated math questions from the sealed AceReason pool.
3. Remove the complete `<think>...</think>` section and supervise the polished derivation after `</think>`; do not truncate targets.
4. Require a complete boxed answer, 96–2,048 response tokens, no malformed thinking tags, repeated 8-gram fraction at most 0.08, and complete Qwen serialization within 4,096 tokens.
5. Allocate the dataset across geometry, number theory, algebra, combinatorics, calculus/linear algebra, probability, and other mathematics with fixed quotas. Classification uses general mathematical vocabulary and never public benchmark text.
6. Train all 15,000 examples for 1,875 updates with full-parameter response-only SFT at `1e-6`, half the previous learning rate, to reduce base-capability drift.
7. Run the complete nine-benchmark suite directly. No pilot, public-benchmark checkpoint selection, or partial result may be presented as success.
8. Apply `reasoning.domination` to require every column to pass independently. Also report pass@1, truncation, repetition, and confidence intervals.

This recipe is motivated by evidence that long-chain distillation can burden small models with redundant steps, while pruning and student-compatible trajectories can retain reasoning with shorter outputs ([Efficient Long CoT Reasoning in Small Language Models](https://arxiv.org/abs/2505.18440)). Rejection-sampling distillation from larger models is established, but the completed Gen-SSD experiment showed that stronger generated targets alone did not transfer reliably here. The present run therefore isolates a different intervention: complete visible solutions, broader competition-math coverage, and a smaller parameter update.

There is no defensible guarantee that a 3B model will clear all nine thresholds. The result is successful only if the final nine-column gate says so.

## Execution

ARC Owl job `1156820` was submitted on October 15, 2026 with an eight-hour B200 limit. The CPU preflight loaded exactly 15,000 unique rows, verified the dataset SHA-256 `fb1a2dfe71f239be755d0bceb03e443888ecfd895ce9cbb3b1465b36bc091e48`, and confirmed the fixed domain counts. The full job has no pilot, development gate, checkpoint selection, or public-suite tuning; it trains all 1,875 updates and evaluates all nine benchmarks.

## Compact balanced 15K terminal result (job 1156820)

The strict all-benchmark experiment completed on October 15, 2026. It trained exactly 15,000 unique rows for all 1,875 planned updates. The dataset SHA-256 was `fb1a2dfe71f239be755d0bceb03e443888ecfd895ce9cbb3b1465b36bc091e48`. All 1,875 logged losses and gradient norms were finite (loss range 0.1881–0.5890). The complete nine-benchmark evaluation contains eight samples for every problem and uses question IDs matched exactly to the base manifest.

| Benchmark | Base pass@1 | Compact pass@1 | Base pass@8 | Compact pass@8 | Pass@8 delta and paired 95% CI | Compact trunc. | Strict Fall 2025 column gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| AIME24 | 7.08% | 3.33% | 23.33% | 13.33% | -10.00 pp [-20.00, +0.00] | 3 | no |
| AIME25 | 2.50% | 0.83% | 13.33% | 6.67% | -6.67 pp [-23.33, +6.67] | 0 | no |
| MATH | 65.88% | 59.90% | 85.20% | 82.60% | -2.60 pp [-5.40, +0.20] | 5 | no |
| CN Math 2024 | 7.50% | 2.50% | 23.33% | 16.67% | -6.67 pp [-23.33, +10.00] | 2 | no |
| Kaoyan | 22.42% | 22.55% | 56.78% | 57.29% | +0.50 pp [-6.03, +6.60] | 0 | no |
| AMC | 44.38% | 32.81% | 77.50% | 62.50% | -15.00 pp [-27.50, -5.00] | 2 | no |
| Minerva | 31.07% | 28.58% | 49.63% | 48.16% | -1.47 pp [-5.51, +2.57] | 4 | yes |
| OlympiadBench | 27.50% | 22.04% | 49.48% | 42.22% | -7.26 pp [-10.22, -4.30] | 24 | no |
| GPQA | 29.48% | 27.59% | 76.77% | 80.30% | +3.54 pp [-2.02, +9.09] | 4 | yes |

The seven-benchmark macro fell from **49.55% to 44.83% pass@8** (-4.72 points) and from **23.55% to 19.56% pass@1** (-3.99 points). The unweighted nine-benchmark pass@8 macro fell from 50.60% to 45.53%. Per-benchmark confidence intervals are paired, cluster-aware where duplicate prompts exist, and unadjusted for multiple comparisons.

The compact targets repaired generation behavior: total truncations fell from **141 to 44** out of 15,792 samples, mean output length fell from 1064.5 to 695.7 tokens, and mean repeated-8-gram fraction fell from 0.0384 to 0.0288. Accuracy did not transfer: only Minerva and GPQA cleared their strict historical score thresholds, so the all-nine gate is **failed (2/9 columns)**.

This result rejects the hypothesis that post-`</think>` compact targets plus balanced quotas and a smaller learning rate would dominate every column. It also shows that stopping quality can improve while reasoning accuracy regresses. The earlier ordinary full-SFT model remains the strongest complete candidate by matched seven-benchmark pass@8 at 51.52%; the compact model scores 44.83%. No official historical leaderboard claim is made because dataset and runtime equivalence remain unverified.

Auditable receipts: `results/arc-20261011-snapshot-2316/compact-balanced/final-report.json`, `results/arc-20261011-snapshot-2316/evaluations/lmf-compact-balanced-15000-suite-comparison.json`, `results/arc-20261011-snapshot-2316/evaluations/lmf-compact-balanced-15000-domination.json`, and the complete raw predictions under `results/arc-20261011-snapshot-2316/evaluations/lmf-compact-balanced-15000-suite/`.


## GEM and model-aware GEM terminal results (October 16, 2026)

Both full 15K GEM experiments completed exactly 1,875 finite updates and all nine benchmarks. Ordinary GEM scored **50.53%** seven-benchmark pass@8 versus **49.55%** for matched base, with **417/15,792 truncations**. Model-aware GEM scored all 83,037 approved pool rows, selected exactly 15,000 unique 8K-safe examples, and scored **48.32%** pass@8 with **284/15,792 truncations**. Its matched pass@1/pass@8 values were **21.00% / 48.32%**, versus **23.55% / 49.55%** for base.

Ordinary GEM improved the target macro by 0.98 points but failed the strict per-column objective. Model-aware reselection improved Kaoyan by 5.03 points and GPQA by 3.54 points over base, tied base on AIME25 and CN Math, and regressed most other columns. Both cleared only the contextual Minerva and GPQA Fall 2025 thresholds. The ordinary full-SFT model remains the highest measured seven-benchmark candidate at **51.52%**, with its previously documented pass@1 and truncation costs.

This evidence rejects teacher-forced base NLL as a sufficient standalone selection signal for this pool. NLL measures surprising teacher tokens and can favor noisy or unsupported trajectories; it does not establish correctness or learnability. Complete receipts and predictions are under `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-*` and `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-*`. Historical runtime equivalence remains unverified, so these are internal matched results rather than official leaderboard claims.


## TIES merge terminal result (job 1174986)

The frozen no-training full-SFT/GEM TIES merge completed all nine benchmarks. Its seven-benchmark pass@8 macro was **51.59%** versus **49.55%** for matched base, with 602/15,792 truncations. It cleared only **Minerva and GPQA**, failing the strict all-nine goal (2/9). The largest remaining strict solve gaps are OlympiadBench 42, MATH 31, and Kaoyan 22. Full per-benchmark pass@1/pass@8, paired confidence intervals, generation diagnostics, and provenance are in [TIES_MERGE_EXPERIMENT.md](TIES_MERGE_EXPERIMENT.md). This is an internal matched evaluation; historical equivalence is unverified.
