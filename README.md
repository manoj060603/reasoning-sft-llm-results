# Fresh reasoning SFT experiment

**Cross-fitted Transfer Gain with Coverage** selects 15,000 original AceReason
examples and trains a new model from the original `Qwen/Qwen2.5-3B-Instruct`.
It compares that model with the original and a matched 15,000-example random
control. Earlier team models are not used for initialization or probe scoring.

This project addresses the [Virginia Tech reasoning SFT assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM),
due **October 20, 2026, at 11:59 p.m. Eastern**.

## Verified progress

Verified snapshot: **October 20, 2026, at 1:12 p.m. Eastern / 17:12 UTC**.
Both full-SFT runs and all three primary evaluations are complete: **9/9 tasks
per model, including all eight supplied tasks and supplementary AIME25**. The
[native final audit](deadline/report/primary-final-20261020T1708Z/audit.json)
verifies every result from eight raw responses per problem, with matched
protocol, benchmark identities and hardware and no audit errors.

| Stage | Verified state |
|---|---|
| Source preparation and both probes | Complete; 30,000 opposite-fold candidate scores |
| New-method selection | Complete; 15,000 unchanged examples, zero training truncation |
| Both final full-SFT models | Complete; 1,875 updates each, 15,000 unique examples used once, 434 finite FP32 tensors per model |
| Original, random and advanced evaluations | All nine complete for each model; full matched comparison verified |
| Primary native CPU closeout | `1208135` completed successfully; final audit, uncertainty and publication package created |
| Hugging Face publication | Publisher `1209745` running; verified upload-completion receipt and four public links remain pending |

**The advanced policy beats the prepared-source random control, but both SFT
models regress from the original.** On the supplied-eight macro, advanced gains
**2.5548 percentage points pass@1** over random (95% conditional bootstrap CI
**[1.6733, 3.4813]**) and **6.2555 points pass@8** (**[3.1852, 9.5150]**).
The [preserved native uncertainty](deadline/report/primary-final-20261020T1708Z/uncertainty.json)
uses 10,000 paired normalized-question-cluster draws. These intervals condition
on the fixed models and eight recorded outputs; they exclude training-seed and
additional generation-sampling variance, with no multiple-comparison adjustment.

| Benchmark | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
|---|---:|---:|---:|
| AMC 2023 | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% |
| Kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% |
| Minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% |
| OlympiadBench | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% |
| GPQA | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% |
| **Supplied-eight macro** | **29.30% / 52.86%** | **16.23% / 36.70%** | **18.79% / 42.95%** |
| Supplemented seven, internal macro | 23.06% / 45.58% | 11.80% / 32.67% | 14.03% / 37.92% |

The [complete results table](deadline/report/primary-final-20261020T1708Z/results.md)
uses equal benchmark weights for each stated macro. The supplementary seven-task
macro is not a verified historical leaderboard result. Advanced has higher
pass@1 than random on all eight supplied tasks; pass@8 is higher on six, tied
on MATH and lower on AMC. It trails random on supplementary AIME25. Both SFT
models are below original on all eight supplied tasks in both metrics; both
improve on supplementary AIME25. Advanced's supplied-eight deficit versus
original is **10.5157 points pass@1 and 9.9072 points pass@8**.

Generation behavior also regressed. Across the **15,552 supplied-task samples
per model**, original has **152 length finishes (0.98%)**, random **9,010
(57.93%)**, and advanced **7,526 (48.39%)**. Mean generated tokens per response
are **1,084.50 / 19,998.02 / 17,324.52**, respectively. These pooled diagnostics
have different weighting from the equal-benchmark accuracy macros. Advanced
reduces excessive generation relative to random but remains far from original;
finish reasons alone do not establish causation. The [report](deadline/FINAL_REPORT.md)
includes per-task intervals, the accuracy-blind AMC repetition diagnostic, and
limitations from unequal eligibility/token exposure and missing ablations.
The primary selection method, models and evaluation protocol were not changed
in response to the measured scores.

A separate [post-hoc recovery experiment](deadline/recovery/plan.json) tests a
fixed blend of 75% original weights and 25% advanced weights. Its CPU build
verified 434 finite FP32 tensors, and 31 native checks passed. The fixed
[64-question source-heldout screen](deadline/report/recovery-development-20261020/summary.json)
passed: teacher-answer matches were original **18/64**, advanced **10/64** and
recovery **19/64**; length-limit finishes were **1/64**, **49/64** and **1/64**.
This one-sample, 4K development result supports restored stopping versus advanced;
it does not establish benchmark improvement or superiority to the original.

Initial recovery benchmark job `1214775` started at **14:54:26 UTC**, using the
same all-nine, eight-sample, 32K protocol in a bounded 45-minute allocation.
The [native manifest audit](deadline/operations/recovery-evaluation-manifest-audit-20261020.json)
confirms matching protocol, data, IDs, grader, runtime and B200 hardware. Its
full-suite results remain pending. That first stage reached its time limit,
retaining seven complete tasks. Primary SFT checkpoints, results and publication
identities remain separate. The first recovery task is
[raw-audited AMC](deadline/report/recovery-benchmark-partial-20261020T1502Z/audit.json):
**39.0625% pass@1 / 75.00% pass@8**, versus advanced **25.00% / 45.00%** and
original **42.50% / 70.00%**. Recovery improves over advanced in both metrics on
AMC, while original retains higher pass@1. Full-suite results and significance
remain unestablished. Recovery now has seven audited tasks:
MATH **65.325% / 84.60%**, CN Math **12.50% / 33.33%**, and Kaoyan
**21.1055% / 54.2714%** are also verified. The [five-task audit](deadline/report/recovery-benchmark-partial-20261020T1521Z/audit.json)
and [additional Kaoyan audit](deadline/report/recovery-benchmark-partial-20261020T1524Z/additional-audit.json)
retain exact comparisons. MATH pass@1 remains below original; CN exceeds original
in both recorded metrics, and Kaoyan ties original pass@1 with higher pass@8.
Minerva is also verified at **29.4577% / 48.1618%**, versus original **30.3309% / 47.7941%** ([native audit](deadline/report/recovery-benchmark-partial-20261020T1535Z/minerva-audit.json)).
Only OlympiadBench and GPQA remain outside the audited recovery snapshot.
An [unchanged recovery continuation](deadline/operations/recovery-benchmark-continuation-1-admission-20261020.json),
job `1216562`, is **running**, starting at **12:51:19 p.m. Eastern / 16:51:19 UTC**
with a 45-minute allocation ending at **1:36:19 p.m. Eastern / 17:36:19 UTC**.
It retains the initial seven saved tasks and uses the same model, protocol and
output directory. No recovery full macro or significance is established.

The [runtime continuation record](deadline/operations/evaluation-resume-20261020.json)
documents retaining six completed benchmarks and fully regenerating the unsaved
Minerva batch under the frozen settings. No interrupted partial response enters
the scores. Interrupted work is included in GPU accounting; its exact unsaved
generated-token total is unknown. Both regenerated Minerva results were
verified; both primary continuations have now completed, and their results passed native closeout.

Native budget/pipeline checks previously passed **30 tests with no failures,
errors or skips** in a compute allocation. Both initial evaluation parents
ended at their time limit after 21,616 and 21,608 seconds; each is conservatively
charged 21,660 GPU-seconds. The [random accounting proof](deadline/operations/gpu-budget-reconciliation-d155f13d26ffca69.json)
and [advanced proof](deadline/operations/gpu-budget-reconciliation-637c6b2ed64206b8.json)
therefore establish the primary-only terminal-plus-open-reservation bound of **47.05 of 48
GPU-hours (169,380 seconds)**. This is **not actual elapsed GPU usage**.
Before recovery, admitted requests totaled 70 hours; the 88-hour all-attempt history
includes 18 hours verified never admitted. Final resource reconciliation and publication remain pending; primary comparisons are complete.

The recovery screen completed in 316 seconds and is charged 360 GPU-seconds
under its [terminal accounting proof](deadline/operations/gpu-budget-reconciliation-e85de9ae41202753.json).
The initial benchmark stage timed out after 2,703 seconds; its
[terminal proof](deadline/operations/gpu-budget-reconciliation-c0f4181a2d92af85.json)
charges 2,760 seconds and retains seven completed tasks.

**Historical resource bound, 16:53 UTC:** after advanced's completed continuation
was charged 31,200 GPU-seconds, closed charges were 102,900 seconds. Adding the
then-full random and recovery-continuation reservations gave
**163,200 seconds = 45.333333 / 48 GPU-hours**. This preserved reservation snapshot
is historical, not actual elapsed usage.

**Current resource bound, 17:11:50 UTC:** random continuation `1210309` completed
after **35,613 seconds** and is conservatively charged **35,640 GPU-seconds** by
its [terminal proof](deadline/operations/gpu-budget-reconciliation-029e44effa49a4c6.json).
Verified closed charges now total **138,540 seconds (38.483333 GPU-hours)**.
The only remaining GPU reservation is the 2,700-second recovery continuation,
so the bound is **141,240 seconds = 39.233333 / 48 GPU-hours**, with **31,560
seconds** uncommitted. Final consumed use awaits recovery terminal accounting.
Historical admitted requests remain **72.25 hours**, and all-attempt requests
**90.25 hours**; no request history is erased.

The [primary closeout receipt](deadline/operations/primary-closeout-complete-20261020.json)
confirms CPU job `1208135` completed successfully in 159 seconds. It produced
matched results, paired uncertainty and a verified [staged publication package](deadline/report/primary-final-20261020T1708Z/publication-package/package_manifest.json).
Publisher `1209745` is running. Hugging Face publication remains pending its
verified completion receipt and four public artifact links.

## Read the experiment

- [Method, controls and execution workflow](deadline/README.md)
- [Report draft and assignment answers](deadline/FINAL_REPORT.md)
- [Complete primary results and audit](deadline/report/primary-final-20261020T1708Z/results.md)
- [Paired uncertainty and generation diagnostics](deadline/report/primary-final-20261020T1708Z/uncertainty.json)
- [Requirement-by-requirement checklist](deadline/assignment_checklist.json)
- [Frozen scientific plan](deadline/experiment_plan.json)

The method ranks response loss reductions measured by a probe trained on the
opposite question fold, then combines positive-gain leaders with seeded topic
and length coverage. Both final models use the same fresh base, full-parameter
training recipe, example count, update count and evaluation protocol. The report
discloses differences in eligibility, truncation and supervised token counts;
the comparison tests the complete selection method rather than an isolated
transfer-gain ablation.

## Reproducibility and delivery

The [fresh source snapshot](https://github.com/manoj060603/reasoning-sft-llm-results/commit/45e4d38784f1fdad4c033031f3b2d4c32c6c94b2)
is public and verified against its corresponding local source tree; see the
[publication receipt](deadline/report/source_publication_receipt.json).
A [verified source snapshot](https://github.com/manoj060603/reasoning-sft-llm-results/commit/3003d5860e829135ebda53f7e1826a92927eb1f1)
matches local commit `62e98f15e3e6775e0502ee00079e7701a74562ea` and entire tree
`386e161cc13632530f171c055158e0f56f27506c`. It includes the primary 10:10 UTC
partial audit and separately labeled recovery preparation. Source publication
does not establish completed recovery generation, benchmark gains, or publication
of model and dataset artifacts.

Heavy preparation, training, evaluation and native checks run in ARC compute
allocations. Exact source revisions, selected IDs, actual example consumption,
model hashes and evaluation manifests are retained. The original reservation
plan was 42 GPU-hours; the fresh experiment's consumption cap remains 48 hours.
Any unused-reservation reconciliation preserves historical requests and charges
the full reservation for every open or unverified allocation.

Matched primary results, uncertainty estimates and their report interpretation
are complete. Remaining delivery work includes verified links for two public
model repositories and two public dataset repositories, final resource/source
reconciliation and team submission. The publisher is in progress; source
availability and upload calls do not establish verified publication. Course
submission remains a human action.

The code license is in [LICENSE](LICENSE). Published derivative model weights
must carry the [pinned upstream Qwen Research License](deadline/licenses/Qwen2.5-3B-Instruct.LICENSE),
attribution and modification notices. Dataset provenance and upstream license
details are recorded in the report and generated dataset cards.

## Preserved earlier work

[UPSTREAM_PROGRESS_README.md](UPSTREAM_PROGRESS_README.md) preserves the previous
README, earlier team progress and the assignment tutorial. Its model scores and
experiment status refer to earlier work. Other existing root-level reports are
also historical unless explicitly linked as fresh experiment evidence above.
