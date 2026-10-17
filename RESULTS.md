# Final measured results

## Direct answer

**The highest completed seven-benchmark pass@8 macro is TIES at 51.59%, versus 49.55% for the matched base. No completed candidate clears all nine frozen Fall 2025 gates.**

| Completed candidate | Seven-benchmark pass@8 | Gain over matched base | Strict Fall 2025 gates cleared |
|---|---:|---:|---:|
| TIES full-SFT/GEM merge | **51.59%** | **+2.04 pp** | 2/9 |
| Ordinary full-parameter 15K SFT | 51.52% | +1.97 pp | 2/9 |
| Ordinary GEM | 50.53% | +0.98 pp | 2/9 |
| Matched base | 49.55% | — | 2/9 |

**The earlier “4/9” result means ordinary full SFT improved over the matched base on four benchmarks: AIME25, CN Math 2024, Kaoyan, and GPQA. It does not mean four Fall 2025 gates were cleared.** Both ordinary full SFT and TIES clear only Minerva and GPQA. The [complete-run gate audit](research/completed_nine_benchmark_gate_audit_20261017.json) records exact solve counts and source metric hashes.

The ordinary full-SFT +1.97-point macro gain has a paired 95% interval of **[−1.89, +5.74] points**, so that observed improvement is not statistically decisive. Its pass@1 fell from 23.55% to 21.85%, with a paired interval of **[−3.16, −0.22] points**. Those intervals belong to ordinary full SFT, not TIES. Historical runtime and dataset equivalence remain unverified.

The new full-length, system-aligned three-epoch candidate is unmeasured; B200 job **1187777** was still pending with reason Priority at the latest October 17 scheduler check. See [FULL_LENGTH_SFT_EXPERIMENT.md](FULL_LENGTH_SFT_EXPERIMENT.md).

## Complete nine-benchmark pass@8 results

| Benchmark | Matched base | Ordinary full SFT | Delta | Compact-balanced SFT | Compact delta |
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

**This is not a broad win.** Ordinary full SFT improves pass@8 on only **4 of 9 benchmarks** and regresses on **5 of 9**. Within the seven benchmarks used for AVG it improves 4 and regresses on 3. The +1.97-point macro is driven mainly by CN Math 2024 (+10.00 points) and AIME25 (+6.67 points), both 30-question sets with large sampling uncertainty. Compact-balanced SFT is worse: it improves only Kaoyan and GPQA and regresses on the other 7 benchmarks.

## Generation behavior

| Metric across all nine benchmarks | Base | Ordinary full SFT | Compact-balanced SFT |
|---|---:|---:|---:|
| Truncated samples | **141 / 15,792** | 1,964 / 15,792 | **44 / 15,792** |
| Seven-benchmark pass@1 | **23.55%** | 21.85% | 19.56% |
| Seven-benchmark pass@8 | 49.55% | **51.52%** | 44.83% |

Ordinary full SFT produced a strong measured average but severely worsened termination; the later TIES merge slightly exceeded its average. Compact targets repaired termination and shortened outputs, but accuracy regressed substantially. This shows that better stopping behavior did not translate into better benchmark accuracy.

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

Ordinary GEM improved the target macro by 0.98 points but failed the strict per-column objective. Model-aware reselection improved Kaoyan by 5.03 points and GPQA by 3.54 points over base, tied base on AIME25 and CN Math, and regressed most other columns. Both cleared only the contextual Minerva and GPQA Fall 2025 thresholds. At that stage ordinary full SFT led at **51.52%**; the later TIES merge reached **51.59%**, still clearing only 2/9 strict gates.

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

At the actual handoff (2026-10-17 04:20 UTC), job **1180491** was cancelled after seven complete benchmarks; OlympiadBench and GPQA remain unmeasured. Transition receipt: `results/arc-20261011-snapshot-2316/lmf-skillgraph-asft-15000-8k-suite/transition.json`. Full successor preparation job **1183868** was submitted only after the queue was empty, initially pending B200 resources. See the frozen demonstration-response design and prelaunch/submission receipts; no success is inferred from CPU checks or submission.


## Demonstration-response preparation recovery (October 17, 2026)

Preparation job **1183868** failed after 2:03:06 at a broad empty/invalid-completion guard. All **49,416** saved requests validated, with **60,900** remaining. The failed output was not retained, so its precise finish reason is unknown. The concrete guard fix saves empty/whitespace-only `stop`/`length` outputs as unsupported/censored, never using them as evidence of an incorrect solve. Unexpected finish reasons remain fatal with diagnostics. Twenty-nine local and ten ARC CPU tests passed; all method AST nodes outside generation handling and all plan method fields except the code digest are unchanged. No GPU pilot or benchmark-driven change occurred.

Exactly one resume, **B200 job 1186667**, was submitted after an empty queue was confirmed; its initial status is **pending resources**. Original logs/code/plan and saved generations are preserved, and the failed allocation remains within the five-allocation cap. Recovery receipts: `research/demonstration_response_recovery_1183868.json` and `research/demonstration_response_resume_1186667.json`. The current automation follows the resumed job. This is an operational preparation transition, not a trained candidate result or a 9/9 win.


## Demonstration-response preparation: negative capacity result (October 17)

All **110,316** generation requests completed and validated over **30,000** unique sealed candidates. The frozen predicate yielded **5,955 retention, 1,239 manageable and 2,978 currently missed** examples against required counts 3,000/6,000/6,000; 19,828 candidates were outside those categories. The selector therefore stopped before export, the independent audit or training. No benchmark result exists for this candidate, no quota was relaxed, and no unchanged rerun is scheduled. The negative receipt and complete final log are `research/demonstration_response_negative_preparation.json` and `research/demonstration_response_capacity_failure_1186667.log`; original ledgers remain on ARC.

Normal completion counts were 58,410 stop / 1,590 length; assisted counts were 49,150 stop / 1,166 length. There were 28,567 prompts with two complete/supported normal draws under the frozen extractor, so truncation alone does not explain the capacity failure. Successful prefix-conditioned answering was an inadequate acquisition-selection predicate at the specified scale. This does not prove that all other examples are unlearnable.

The explicitly new candidate is [complete original 16K solutions with three-epoch SFT](FULL_LENGTH_SFT_EXPERIMENT.md), using the assignment reference recipe and a standalone length/cluster selector that reads no failed student generations. Its CPU audit also identified missing system-column mappings in previous exports; this is a concrete loader mismatch, not an established cause of benchmark performance. All selected rows must pass the actual converter and processor with explicit system preservation before submission. No unmeasured improvement or historical leaderboard equivalence is claimed.

## Full-length candidate submitted after complete CPU checks

Full B200 job **1187777** was submitted only after an empty queue and complete real-row converter/processor verification of **15,000 unique unchanged original examples**. There are **5,524** complete selected solutions above 8K, max exact length **16,350**. Explicit dataset column mappings now preserve the intended system instruction through the actual loader. Eight source responses undergo only the pinned tokenizer's canonical Unicode composition; original exported bytes/hashes are unchanged.

The fixed candidate trains plain full-parameter response-only CE at 16K, LR5e-6, batch eight, three epochs/**5,625 updates** (45,000 presentations), then evaluates the final checkpoint on all nine with the unchanged sampler. Initial state: **PENDING (Priority)**. This differs from the previous 1,875-update recipes; repeated epochs do not increase the unique dataset size. Full preparation/submission receipts are under `results/arc-20261011-snapshot-2316/full-length/preparation/`. No new benchmark score or 9/9 success is claimed.
