# Final measured results

## Direct answer

**The project’s best model beats the matched local baseline on the target seven-benchmark pass@8 macro, but the latest compact-balanced model does not.**

- **Matched base:** 49.55% pass@8
- **Best model, full-parameter 15K SFT:** **51.52% pass@8** — **+1.97 percentage points**
- **Latest compact-balanced 15K SFT:** 44.83% pass@8 — **−4.72 points**
- **Illustrative Fall 2025 leader:** 52.50% AVG; the best model is 0.98 points lower, and exact historical comparability is unverified.

The best model’s +1.97-point pass@8 gain has a paired 95% interval of **[−1.89, +5.74] points**, so the observed improvement is not statistically decisive. Its pass@1 fell from 23.55% to 21.85%, with a paired interval of **[−3.16, −0.22] points**.

## Complete nine-benchmark pass@8 results

| Benchmark | Matched base | Best full SFT | Delta | Compact-balanced SFT | Compact delta |
|---|---:|---:|---:|---:|---:|
| AIME24 | 23.33% | 16.67% | −6.67 pp | 13.33% | −10.00 pp |
| AIME25 | 13.33% | 20.00% | +6.67 pp | 6.67% | −6.67 pp |
| MATH | 85.20% | 85.00% | −0.20 pp | 82.60% | −2.60 pp |
| CN Math 2024 | 23.33% | 33.33% | +10.00 pp | 16.67% | −6.67 pp |
| Kaoyan | 56.78% | 59.80% | +3.02 pp | 57.29% | +0.50 pp |
| AMC | 77.50% | 70.00% | −7.50 pp | 62.50% | −15.00 pp |
| Minerva | 49.63% | 48.90% | −0.74 pp | 48.16% | −1.47 pp |
| OlympiadBench | 49.48% | 49.33% | −0.15 pp | 42.22% | −7.26 pp |
| GPQA | 76.77% | 79.29% | +2.53 pp | 80.30% | +3.54 pp |
| **Seven-benchmark macro** | **49.55%** | **51.52%** | **+1.97 pp** | **44.83%** | **−4.72 pp** |

The seven-benchmark macro follows the assignment’s AVG set: AIME25, CN Math 2024, Kaoyan, AMC, Minerva, OlympiadBench, and GPQA.

**This is not a broad win.** Best full SFT improves pass@8 on only **4 of 9 benchmarks** and regresses on **5 of 9**. Within the seven benchmarks used for AVG it improves 4 and regresses on 3. The +1.97-point macro is driven mainly by CN Math 2024 (+10.00 points) and AIME25 (+6.67 points), both 30-question sets with large sampling uncertainty. Compact-balanced SFT is worse: it improves only Kaoyan and GPQA and regresses on the other 7 benchmarks.

## Generation behavior

| Metric across all nine benchmarks | Base | Best full SFT | Compact-balanced SFT |
|---|---:|---:|---:|
| Truncated samples | **141 / 15,792** | 1,964 / 15,792 | **44 / 15,792** |
| Seven-benchmark pass@1 | **23.55%** | 21.85% | 19.56% |
| Seven-benchmark pass@8 | 49.55% | **51.52%** | 44.83% |

The full SFT model produced the highest target score but severely worsened termination. Compact targets repaired termination and shortened outputs, but accuracy regressed substantially. This shows that better stopping behavior did not translate into better benchmark accuracy.

## Fall 2025 comparison

Neither trained model beats the Fall 2025 leader in every displayed column. Both clear only the Minerva and GPQA historical thresholds. The full SFT model’s 51.52% seven-benchmark macro is numerically below the historical 52.50% leader.

These historical comparisons are contextual only. The Fall 2025 dataset revisions, AIME25 source, software stack, seeds, batching behavior, and raw predictions have not been established as identical. The defensible claim is limited to the matched internal comparison: **full 15K SFT improved observed pass@8 by 1.97 points over the local base, while reducing pass@1 and greatly increasing truncation.**

## Verification

- Compact-balanced training used exactly 15,000 unique rows with dataset SHA-256 `fb1a2dfe71f239be755d0bceb03e443888ecfd895ce9cbb3b1465b36bc091e48`.
- It completed all 1,875 updates with finite losses and gradient norms.
- Both complete evaluations contain eight predictions for every problem across all nine benchmarks.
- Candidate and base question manifests match exactly.
- Raw comparison: `results/arc-20261011-snapshot-2316/evaluations/lmf-full-15000-sft-suite-comparison.json`.
- Compact terminal report: `results/arc-20261011-snapshot-2316/compact-balanced/final-report.json`.


## GEM and model-aware GEM terminal results (October 16, 2026)

Both full 15K GEM experiments completed exactly 1,875 finite updates and all nine benchmarks. Ordinary GEM scored **50.53%** seven-benchmark pass@8 versus **49.55%** for matched base, with **417/15,792 truncations**. Model-aware GEM scored all 83,037 approved pool rows, selected exactly 15,000 unique 8K-safe examples, and scored **48.32%** pass@8 with **284/15,792 truncations**. Its matched pass@1/pass@8 values were **21.00% / 48.32%**, versus **23.55% / 49.55%** for base.

Ordinary GEM improved the target macro by 0.98 points but failed the strict per-column objective. Model-aware reselection improved Kaoyan by 5.03 points and GPQA by 3.54 points over base, tied base on AIME25 and CN Math, and regressed most other columns. Both cleared only the contextual Minerva and GPQA Fall 2025 thresholds. The ordinary full-SFT model remains the highest measured seven-benchmark candidate at **51.52%**, with its previously documented pass@1 and truncation costs.

This evidence rejects teacher-forced base NLL as a sufficient standalone selection signal for this pool. NLL measures surprising teacher tokens and can favor noisy or unsupported trajectories; it does not establish correctness or learnability. Complete receipts and predictions are under `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-*` and `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-*`. Historical runtime equivalence remains unverified, so these are internal matched results rather than official leaderboard claims.


## TIES merge terminal result (job 1174986)

The frozen no-training full-SFT/GEM TIES merge completed all nine benchmarks. Its seven-benchmark pass@8 macro was **51.59%** versus **49.55%** for matched base, with 602/15,792 truncations. It cleared only **Minerva and GPQA**, failing the strict all-nine goal (2/9). The largest remaining strict solve gaps are OlympiadBench 42, MATH 31, and Kaoyan 22. Full per-benchmark pass@1/pass@8, paired confidence intervals, generation diagnostics, and provenance are in [TIES_MERGE_EXPERIMENT.md](TIES_MERGE_EXPERIMENT.md). This is an internal matched evaluation; historical equivalence is unverified.


## SkillGraph ASFT result at successor handoff (October 17, 2026)

Job 1180491 completed exactly 1,875 finite full-parameter training updates. The current candidate has failed the strict all-nine objective. The user authorized moving on without waiting for the remaining evaluation; available complete predictions and their manifests are preserved non-destructively. Cancellation of an unfinished evaluation tail, if needed for the handoff, is a user-requested transition rather than a training failure. Unfinished benchmarks are unmeasured, not zero scores.

| Benchmark | Solves | Strict target | Pass@1 | Pass@8 | Truncated samples |
|---|---:|---:|---:|---:|---:|
| amc | 25/40 | 33 | 35.94% | 62.50% | 45 |
| aime25 | 4/30 | 8 | 4.58% | 13.33% | 49 |
| aime | 4/30 | 8 | 5.83% | 13.33% | 73 |
| math | 415/500 | 451 | 64.83% | 83.00% | 267 |
| cn_math_2024 | 11/30 | 14 | 10.42% | 36.67% | 72 |
| kaoyan | 114/199 | 138 | 26.13% | 57.29% | 258 |
| minerva | 121/272 | 95 | 27.76% | 44.49% | 146 |

OlympiadBench and GPQA are not complete in this snapshot. The matched seven-benchmark macro is therefore unavailable. Paired pass@1/pass@8 confidence intervals, output lengths and repetition are recorded in `results/arc-20261011-snapshot-2316/lmf-skillgraph-asft-15000-8k-suite/available_results_audit.json`. Candidate/base protocol and completed-problem identities match. All seven completed columns except Minerva miss the frozen strict target; this candidate is retired rather than rerun unchanged.

The successor is original-demonstration response selection plus a fixed plain-SFT curriculum, documented in [DEMONSTRATION_RESPONSE_SFT_DESIGN.md](DEMONSTRATION_RESPONSE_SFT_DESIGN.md). Exactly 15,000 unique original prompts/targets and the unchanged final all-nine evaluator remain mandatory. Preparation/category capacity, the fresh control audit, and transfer remain unmeasured. Historical leaderboard equivalence is not established.
