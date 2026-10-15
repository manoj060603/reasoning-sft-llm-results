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
