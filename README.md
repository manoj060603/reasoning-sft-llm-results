# Fresh reasoning SFT experiment

**Cross-fitted Transfer Gain with Coverage** selects 15,000 original AceReason
examples and trains a new model from the original `Qwen/Qwen2.5-3B-Instruct`.
It compares that model with the original and a matched 15,000-example random
control. Earlier team models are not used for initialization or probe scoring.

This project addresses the [Virginia Tech reasoning SFT assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM),
with an original deadline of **October 20, 2026, at 11:59 p.m. Eastern**.

## One week extension and new research phase

The user reported a one-week extension on October 20, making the working
deadline **October 27, 2026, at 11:59 p.m. Eastern**. The user subsequently
removed the earlier 48-hour GPU cap for the new extension. Its jobs will still
have bounded allocations and staged admission. The earlier **39.65-hour**
charge, original cap, completed experiments and all five published artifacts
below remain preserved as historical evidence.

The [new research review](extension/research.json) covers 16 relevant primary
sources on data selection, long-reasoning degradation and preservation during
fine-tuning. The [extension campaign](extension/campaign.json) prioritizes a
CPU feasibility audit, bounded diagnostic pilots, matched data-by-objective
controls and seed replication before one final recipe receives unchanged
nine-task evaluation. The leading hypothesis combines complete,
structure-aware, base-compatible examples with an anchored training objective.
This is a proposed synthesis of published ideas; superiority and novelty are
not established. Exact training and acceptance rules must be sealed before
GPU work. No new candidate benchmark result is claimed at this stage.

The first [native CPU feasibility audit](extension/report/feasibility-20261020-1222225/complete.json)
completed successfully in **221 seconds, with zero GPU use**. All **64 native
checks** passed with no failures, errors or skips; the [QA receipt](extension/report/feasibility-20261020-1222225/native_qa.json)
and [terminal accounting](extension/report/feasibility-20261020-1222225/accounting.json)
bind the executed sources and real compute allocation.

The audit verified the earlier exports against their original source pairs.
All 15,000 advanced examples have balanced reasoning tags and fit the complete
8K serialization, while 5,806 random examples lost their tails to prefix
truncation. The advanced examples also already have lower source repetition
and lower English branch-keyword density than random. These observations do
not establish the cause of generated repetition or prove a better selector;
they make another completeness or keyword filter alone an insufficient
explanation of the earlier failure. There are **51,111 complete prepared-pool
examples at 8K**, including a 32-token reserve, for an eligibility-matched
control. This count covers the finalized pool, rather than all untouched
100,000 source records.

The next stage is to seal labeled development splits and small causal pilots
for data compatibility, preservation objectives and a separately identified
verified concise-response arm. No new GPU training or inference has started.

The next [labeled-data capacity audit](extension/report/labeled-feasibility-20261020-1222926/complete.json)
also completed successfully: **54 seconds, zero GPUs, all 108 native checks
passed**. Pinned official training sources contain 7,500 MATH records and 1,119
ARC-Challenge records. Screening against prior training, probes, the scored
reservoir, viewed validation and all nine benchmark question sets leaves
**6,013 usable MATH records and 1,119 ARC-Challenge records**, in 7,077 retained
question groups. Four MATH records have unresolved boxed-answer extraction;
1,483 more are excluded directly or through a related question group.
The [prospective CPU plan](extension/labeled_plan.json), [native QA](extension/report/labeled-feasibility-20261020-1222926/native_qa.json)
and [accounting](extension/report/labeled-feasibility-20261020-1222926/accounting.json)
preserve the exact execution.

The screening is complete for its declared lexical criterion: normalized token
trigram-set Jaccard similarity of at least 0.8, with connected-group exclusion.
It does not guarantee removal of every semantic paraphrase. Correct answer
keys and solution text do not enter the matching. These surviving counts are
sufficient for the proposed development and confirmation targets, subject to
whole-group allocation. **That capacity stage chose no splits and trained or
benchmarked no new model.** ARC-Challenge supplies a general science
retention check; it does not establish GPQA performance.

The first split-sealing attempt passed **all 192 native CPU checks**, including
the new four-objective loss and actual tiny-model Trainer checks. It then
[stopped before assigning splits](extension/report/development-seal-failed-20261021-1223566/delivery_summary.json)
because an official MATH difficulty field did not satisfy the assumed
`Level 1`–`Level 5` schema. The 45-second, zero-GPU failure and its passing
[QA receipt](extension/report/development-seal-failed-20261021-1223566/native_qa.json)
are preserved. The separate [CPU metadata diagnostic](extension/report/development-level-diagnostic-20261021-1223713/complete.json)
completed in two seconds and found exactly one official `Level ?` field. A
versioned repair preserves that record in an explicit metadata stratum. The
initial diagnostic request failed before reaching the scheduler because its
environment omitted the Slurm configuration; its rejection is preserved and
the successful fresh request uses the existing tested environment helper.
These checks do not establish real Qwen/B200
memory feasibility, and no new GPU training or inference has started.

The [versioned split seal](extension/report/development-seal-v2-20261021-1223943/complete.json)
subsequently **completed in 99 seconds with zero GPUs**. All **230 native
checks** passed without failures, errors or skips. The frozen whole-group
allocations are:

| Purpose | MATH records | ARC-Challenge records | Question groups |
|---|---:|---:|---:|
| Calibration | 16 | 16 | 32 |
| Development | 512 | 256 | 766 |
| Single-use confirmation | 1,024 | 512 | 1,527 |

The same lexical audit checked all 83,029 prepared training-pool questions and
reserved **67 matching IDs** for exclusion from future training, probes and
selection scoring. None overlaps prior training, probes or the earlier scored
reservoir. Full-pool screening against the nine benchmark question sets remains
a separate preparation requirement. Raw labeled records and split IDs remain
private on ARC; the [QA](extension/report/development-seal-v2-20261021-1223943/native_qa.json)
and [accounting](extension/report/development-seal-v2-20261021-1223943/accounting.json)
are public. The next step is actual Qwen/B200 memory and training integration
admission. New model training and new benchmark results remain pending.

The subsequent [native preparation audit](extension/report/native-smoke-preparation-20261021-1224497/complete.json)
completed in **215 seconds with zero GPUs and all 341 checks passing**. It
screened the full prepared pool against all nine benchmark question sets,
including both AMC representations. One additional benchmark-overlapping row
and the 67 reserved-source rows are excluded, leaving **82,961 eligible
original pairs**. This establishes the declared lexical screen, with the
semantic limitations described above. Four private diagnostic examples were
replayed with the original tokenizer, including the longest complete eligible
example at **31,776 tokens**. No training draw has been selected.

The B200 numerical and memory diagnostic is admitted as job **1224577** and
remains queued as of October 21 at 01:45 UTC. Its eight fresh cases test four
objectives on short and long examples. A separate, prospective diagnostic is
also required to verify a long forward while Adam state and accumulated
gradients are already resident; short accumulation checks alone cannot prove
that memory envelope. The pilot driver fails closed without that evidence.

The source-grading preparation has encountered two preserved CPU failures.
The [first attempt](extension/report/source-readiness-failed-20261021-1225252/accounting.json)
ran 445 checks and exposed one fixture that tried to construct an official
empty answer already quarantined by the source loader. A pinned fixture-only
repair left the scorer and protocol unchanged. The
[fresh retry](extension/report/source-readiness-v2-failed-20261021-1225360/accounting.json)
then passed **464 of 465 checks**; a negative percentage comparison timed out
under the fixed five-second, fresh-process grading limit. Its comparator
imports and symbolic work need separate timing evidence before an operational
repair is chosen. Both attempts stopped before calibration preparation and
used zero GPUs. No new training, generated source validation, benchmark result
or demonstrated improvement is claimed by these checks.

An [evidence-only CPU timing diagnostic](extension/report/source-grader-diagnostic-20261021-1225426/delivery_summary.json)
subsequently completed in **29 seconds, using zero GPUs**. Four fixed public
scalar fixtures, repeated twice, produced the expected results in all eight
unchanged grading calls and all eight separately instrumented calls. The first
unchanged call took **4.694 seconds**; later calls took approximately 1.25–1.66
seconds. These observations show little margin in the five-second cold-worker
limit, but do not reproduce the earlier full-QA environment or establish its
cause. The scorer and timeout remain unchanged, the failed QA remains failed,
and source-grading reliability is still awaiting an explicit operational repair
and fresh native validation.

The separate [joint-memory CPU QA](extension/report/joint-memory-qa-20261021-1225557/delivery_summary.json)
completed in **48 seconds with zero GPUs**. All **410 scoped checks** passed:
341 legacy checks, 44 pilot-driver checks and 25 joint-diagnostic checks.
This scope uses synthetic CPU fixtures and binds 79 source files. It explicitly
excludes the source-grader tests; it establishes neither grader repair nor
actual Qwen/B200 memory feasibility.

The separate [GPU admission metadata checks](extension/report/joint-gpu-metadata-qa-20261021-1225720/delivery_summary.json)
then passed **all 46 tests in a four-second CPU allocation**, using zero GPUs.
They cover scheduler GPU formats, the two-allocation limit, duplicate blocking,
terminal CPU receipts and exact source bindings. They do not replay the 410
scientific checks or establish actual model feasibility. The second bounded
B200 diagnostic is now admitted as **1225727**, with its [exact admission
receipt](extension/operations/joint-memory-admission-job-1225727.json). It tests
a long forward after Adam state and accumulated gradients coexist, across all
four objectives, using 17 disposable presentations and a final partial batch.
Both B200 diagnostics remain queued as of **02:37 UTC on October 21**. Their
combined open reservation is four GPU-hours; extension GPU use is still zero
at that snapshot. No pilot, generated development score or new benchmark has
started. A separate unique-example, actual pilot-driver accumulation check is
also required before long pilots.

The separately versioned source worker uses a prospectively frozen ten-second
setup budget, followed by five seconds for comparison and a bounded cleanup.
Its [new CPU attempt](extension/report/source-readiness-v2-failed-20261021-1225820/delivery_summary.json)
ran **474 checks in a 71-second, zero-GPU allocation**: 472 passed and two
readiness fixtures errored. Both errors came from a fixture passing an absolute
plan path to the required relative-path guard; the actual job wrapper uses a
relative path. The 410 legacy checks and all 40 new worker checks, including
the pinned public scalar comparisons, passed. The overall stage remains failed
and stopped before preparing the private calibration cache. Its exact sources,
logs and accounting are preserved. A separate fixture repair must receive fresh
native validation; these observations establish neither full grading readiness
nor the earlier five-second end-to-end contract.

## Additional recovery follow-up: closed without promotion

A separate follow-up, authorized October 20 after delivery, tested **12.5%
and 37.5% advanced-weight blends** against the original and the published 25%
recovery. Its [prospective plan](deadline/anchor_search/plan.json) uses 192 fresh
source-heldout search questions and 192 disjoint confirmation questions, eight
responses each. Only one candidate passing both fixed source stages may receive
a full unchanged nine-task benchmark evaluation. This is exploratory work after
prior benchmark observations; it does not replace the delivered experiment.

CPU preparation retry `1218982` **completed successfully in 252 seconds**.
Both new checkpoints have six finite FP32 shards and 434 checked tensors. The
[preparation receipt](deadline/operations/anchor-v2-preparation-complete-20261020.json)
and [native source/model/split seal](deadline/report/anchor-v2-preparation-20261020/prepared.json)
preserve exact identities. The [32 native checks](deadline/operations/anchor-v2-native-qa-20261020-attempt1.json)
passed with zero failures, errors or skips. The [operational retry](deadline/operations/anchor-v2-prepare-retry-20261020.json)
retains the initial missing-metadata failure and unchanged exclusion-ID repair;
neither CPU attempt used a GPU.

**Source validation started at 19:53:23 UTC / 3:53 p.m. Eastern** on a B200,
inside the frozen deadline reserve. Job `1219074` obtained an earlier backfill
allocation despite the previous 23:34 UTC queue estimate. Its [native allocation
seal](deadline/report/anchor-v2-preparation-20261020/gpu_allocation.json) records
the full 90-minute limit and one GPU; the [earlier backfill adjustment](deadline/operations/anchor-v2-backfill-adjustment-20261020.json)
changed no scientific setting. The original, incumbent and both new blends
completed the fresh search split. The 12.5% candidate was selected, then failed
independent confirmation. **The published 25% incumbent is retained; no new
candidate benchmark was admitted.** The final outcome is detailed
[below](#separate-validation-guided-follow-up-no-promotion).
The separate closeout/publisher passed [20 native CPU checks](deadline/operations/anchor-followup-publication-native-qa-20261020-attempt1.json)
and was prepared to audit all five roles, compute conditional uncertainty and
publish only a confirmed checkpoint after all nine raw benchmark tasks. It did
not run because confirmation failed; no new checkpoint was published.
These checks performed no network access or upload.

At source admission, the conservative reservation bound was **40.533333 / 48
GPU-hours**. A later three-hour candidate benchmark would have produced a
**43.533333 / 48-hour** bound before terminal reconciliation; that benchmark was
not admitted. These are historical reservation bounds. Source selection uses teacher agreement, which is
not verified ground truth. All previously completed assignment artifacts remain
available regardless of this exploratory follow-up.

## Verified progress

Verified snapshot: **October 20, 2026, at 1:41 p.m. Eastern / 17:41 UTC**.
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
| Hugging Face publication | Complete; two models and two datasets anonymously verified at their pinned commits |
| Separate post-hoc recovery | All nine tasks raw-verified; distinct recovery checkpoint publicly verified |

Publisher `1209745` completed at **17:15:11 UTC**, using 336 seconds of CPU
allocation and no GPU. Its [completion receipt](deadline/report/primary-hf-publication-20261020/publication_complete.json)
verifies public access, every published file, and unchanged evaluated FP32 weight
bytes. The [snapshot](deadline/report/primary-hf-publication-20261020/snapshot.json)
preserves the native publication evidence.

| Primary artifact | Public repository | Verified commit |
|---|---|---|
| Random 15K model | [qwen25-3b-reasoning-random15k-20261019](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-random15k-20261019) | [860dd82d](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-random15k-20261019/tree/860dd82d4aa28b650555d6b4787b44711e4d0326) |
| Advanced 15K model | [qwen25-3b-reasoning-transfer-coverage15k-20261019](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019) | [550bdcc8](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019/tree/550bdcc89a9352335d6030d6834fcff0f94d697f) |
| Random original dataset | [reasoning-random15k-20261019](https://huggingface.co/datasets/adversarialbandit/reasoning-random15k-20261019) | [26077e02](https://huggingface.co/datasets/adversarialbandit/reasoning-random15k-20261019/tree/26077e0219ac79c8518e370dd7736091200d0da3) |
| Advanced original dataset | [reasoning-transfer-coverage15k-20261019](https://huggingface.co/datasets/adversarialbandit/reasoning-transfer-coverage15k-20261019) | [aab32e0b](https://huggingface.co/datasets/adversarialbandit/reasoning-transfer-coverage15k-20261019/tree/aab32e0bf630ba32db492af0b4af0b2632c32a06) |

**The advanced policy beats the prepared-source random control, but both SFT
models regress from the original.** On the supplied-eight macro, advanced gains
**2.5548 percentage points pass@1** over random (95% conditional bootstrap CI
**[1.6733, 3.4813]**) and **6.2555 points pass@8** (**[3.1852, 9.5150]**).
The [preserved native uncertainty](deadline/report/primary-final-20261020T1708Z/uncertainty.json)
uses 10,000 paired normalized-question-cluster draws. These intervals condition
on the fixed models and eight recorded outputs; they exclude training-seed and
additional generation-sampling variance, with no multiple-comparison adjustment.

| Benchmark | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 | Recovery, post-hoc pass@1 / pass@8 |
|---|---:|---:|---:|---:|
| AMC 2023 | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% | 39.06% / 75.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% | 6.25% / 23.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% | 65.33% / 84.60% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% | 12.50% / 33.33% |
| Kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% | 21.11% / 54.27% |
| Minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% | 29.46% / 48.16% |
| OlympiadBench | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% | 27.41% / 49.48% |
| GPQA | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% | 30.43% / 76.26% |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% | 2.50% / 10.00% |
| **Supplied-eight macro** | **29.30% / 52.86%** | **16.23% / 36.70%** | **18.79% / 42.95%** | **28.94% / 55.56%** |
| Supplemented seven, internal macro | 23.06% / 45.58% | 11.80% / 32.67% | 14.03% / 37.92% | 23.21% / 49.50% |

The recovery column is a separate fixed post-hoc weight interpolation, not a new
SFT model or a replacement for the primary comparison. Its [final native summary](deadline/report/recovery-final-20261020T1730Z/evaluation_summary.json)
verifies all nine tasks for all four roles from matched raw eight-sample outputs.

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

A separate [post-hoc recovery experiment](deadline/recovery/plan.json) tested a
fixed blend of 75% original weights and 25% advanced weights. Its CPU build
verified 434 finite FP32 tensors, and 31 native checks passed. The fixed
[64-question source-heldout screen](deadline/report/recovery-development-20261020/summary.json)
passed: teacher-answer matches were original **18/64**, advanced **10/64** and
recovery **19/64**; length-limit finishes were **1/64**, **49/64** and **1/64**.
This one-sample, 4K development result supports restored stopping versus advanced;
it does not establish benchmark improvement or superiority to the original.

Initial recovery job `1214775` retained seven complete tasks at its time limit.
The [unchanged continuation](deadline/operations/recovery-benchmark-continuation-1-admission-20261020.json),
`1216562`, completed successfully after **1,940 seconds**, finishing OlympiadBench
and GPQA with the same weights, eight-sample 32K protocol, data, IDs, grader,
runtime and B200 hardware. Recovery GPQA is **30.4293% / 76.2626%**, with **9
length finishes out of 1,584 responses**. Its supplied-eight macro is
**28.94218107% / 55.55548699%**, versus original **29.30267835% / 52.86146036%**:
pass@1 is **0.3605 points lower** and pass@8 **2.6940 points higher**. This is a
tradeoff, not dominance or a SOTA result. Recovery has **197 / 15,552 supplied-task
length finishes**, versus original 152, random 9,010 and advanced 7,526. The report
retains the per-task length counts. The completed [native recovery uncertainty](deadline/report/recovery-final-20261020T1730Z/uncertainty.json)
gives supplied-eight recovery-minus-original 95% intervals of **[−1.2973, 0.6356]
points pass@1** and **[−0.1363, 5.6131] points pass@8**. Both include zero: this
conditional analysis finds **no clear supplied-eight difference from original**.
Recovery exceeds random by **12.7100 points pass@1 [11.1310, 14.3782]** and
**18.8567 points pass@8 [15.1679, 22.6705]**. These 10,000 paired question-cluster
draws use seed 20261019, exclude training-seed and additional generation variance,
and make no multiple-comparison adjustment. The fixed post-hoc candidate and
point estimates establish neither general superiority nor SOTA. CPU analysis
`1217465` completed in **27 seconds**, with no GPU or primary-output changes.

The distinct [recovery checkpoint](https://huggingface.co/adversarialbandit/qwen25-3b-base75-advanced25-posthoc-20261020)
is anonymously verified at [commit 51814ba26ab0dc052447aee1612bebcffdba43f2](https://huggingface.co/adversarialbandit/qwen25-3b-base75-advanced25-posthoc-20261020/tree/51814ba26ab0dc052447aee1612bebcffdba43f2).
Its [publication receipt](deadline/report/recovery-final-20261020T1730Z/publication_complete.json)
and [native snapshot](deadline/report/recovery-final-20261020T1730Z/snapshot.json)
confirm completion at **17:28:51 UTC**; CPU publisher `1217227` completed in 120
seconds after **23 native CPU checks** and wrapper syntax checks passed.
All six evaluated FP32 weight shards and model/tokenizer sidecars retain their
exact bytes. Publication adds the required upstream attribution sentence to
NOTICE while preserving the original recovery notice verbatim and separately
under provenance; both hashes are declared. The coefficient was fixed before
recovery generation, after primary outcomes had been viewed; no additional SFT
or coefficient sweep occurred. All four primary publications remain unchanged.

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
includes 18 hours verified never admitted. These are historical reservation
snapshots; the terminal resource reconciliation is reported below.

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

**Historical resource bound, 17:11:50 UTC:** random continuation `1210309` completed
after **35,613 seconds** and is conservatively charged **35,640 GPU-seconds** by
its [terminal proof](deadline/operations/gpu-budget-reconciliation-029e44effa49a4c6.json).
Verified closed charges then totaled **138,540 seconds (38.483333 GPU-hours)**.
Adding the then-open 2,700-second recovery continuation gave
**141,240 seconds = 39.233333 / 48 GPU-hours**, before its terminal reconciliation.
Historical admitted requests remain **72.25 hours**, and all-attempt requests
**90.25 hours**; no request history is erased.

**Final GPU accounting:** all GPU allocations are terminal. The [final exact
reconciliation](deadline/operations/gpu-budget-reconciliation-43fae58f06119f1c.json)
records **140,025 actual elapsed GPU-seconds = 38.895833 GPU-hours** and
**140,520 conservatively minute-rounded GPU-seconds = 39.033333 / 48 GPU-hours**.
The recovery continuation's 1,940 seconds are charged as 1,980 seconds. Interrupted
and timed-out work is included. Historical admitted/all-attempt reservations
remain **72.25 / 90.25 hours**. CPU packaging and publication add zero GPU usage.

The [primary closeout receipt](deadline/operations/primary-closeout-complete-20261020.json)
confirms CPU job `1208135` completed successfully in 159 seconds. It produced
matched results, paired uncertainty and a verified [staged publication package](deadline/report/primary-final-20261020T1708Z/publication-package/package_manifest.json).
Publisher `1209745` completed successfully. Its four public artifacts and exact
verified commits are linked above. Manual course submission remains pending.

## Read the experiment

- [Method, controls and execution workflow](deadline/README.md)
- [Report and assignment answers](deadline/FINAL_REPORT.md)
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

The public source is [manoj060603/reasoning-sft-llm-results, main](https://github.com/manoj060603/reasoning-sft-llm-results/tree/main).
The [report](deadline/FINAL_REPORT.md) links the complete native primary and
recovery evaluations, all five verified model/dataset publications, and terminal
GPU accounting. This source snapshot contains the final documentation and
native evidence. The [earlier source-publication receipt](deadline/report/source_publication_receipt.json)
is retained as historical snapshot verification.

Heavy preparation, training, evaluation and native checks run in ARC compute
allocations. Exact source revisions, selected IDs, actual example consumption,
model hashes and evaluation manifests are retained. The original reservation
plan was 42 GPU-hours; the fresh experiment's consumption cap remains 48 hours.
Any unused-reservation reconciliation preserves historical requests and charges
the full reservation for every open or unverified allocation.

The mandatory technical delivery is complete: both trained models, both original
15K dataset exports, matched primary results, uncertainty estimates and report
interpretation are available, with anonymous verification of all four Hugging
Face repositories. The optional recovery checkpoint is separately published and
verified, terminal GPU accounting is complete, and the separate recovery
uncertainty analysis is complete. The final source snapshot is ready; manual team
submission remains pending. Recovery uncertainty is labeled separately from the
unchanged primary CI claims. Course submission remains a human action.

The code license is in [LICENSE](LICENSE). Published derivative model weights
must carry the [pinned upstream Qwen Research License](deadline/licenses/Qwen2.5-3B-Instruct.LICENSE),
attribution and modification notices. Dataset provenance and upstream license
details are recorded in the report and generated dataset cards.

## Separate validation-guided follow-up: no promotion

A later [frozen source-only search](deadline/anchor_search/plan.json) tested
advanced-weight fractions 0.125 and 0.375 on 192 fresh held-out questions, with
192 disjoint questions reserved for independent confirmation. Earlier published
models and benchmark results were preserved. The [search selection](deadline/report/anchor-v2-final-20261020/selection.json)
chose 0.125, with teacher-agreement pass@1 **30.078125%**, pass@8 **53.125%** and
harmonic score **0.3840962441**. The 0.375 candidate failed the search length and
repetition limits.

Independent confirmation did **not** support promotion:

| Source confirmation, 192 questions | Teacher-agreement pass@1 | Teacher-agreement pass@8 |
|---|---:|---:|
| Original | 25.78125% | 52.08333% |
| Incumbent 75% base / 25% advanced | 28.38542% | 52.60417% |
| Candidate 87.5% base / 12.5% advanced | 25.97656% | 48.43750% |

The [complete decision](deadline/report/anchor-v2-final-20261020/decision.json)
is **accepted=false**: the candidate failed the fixed agreement floors and had
lower harmonic score than incumbent, despite passing the confirmation quality
limits. The published 75/25 incumbent is retained. No new candidate benchmarks,
Hugging Face checkpoint publication or alternate-candidate retry followed.
These are source-teacher agreement proxies, not verified truth or benchmark
accuracy. The [follow-up closeout](deadline/report/anchor-v2-final-20261020/closeout.json)
records this negative result and its separate resource accounting. The 17:31 GPU
totals above cover the earlier primary/fixed recovery; later launch reservation
bounds are explicitly historical.

Source job `1219074` completed successfully at **20:29:26 UTC / 4:29 p.m.
Eastern**, using 2,163 seconds on one B200. Including this follow-up, actual
elapsed GPU use is **39.496667 hours** and the per-allocation rounded charge is
**39.65 / 48 hours**, supported by the [terminal reconciliation](deadline/operations/gpu-budget-reconciliation-be6bf2aca0eee450.json).
The [native evidence archive](deadline/report/anchor-v2-final-20261020/native-source-evidence.tar.gz)
preserves all seven source-role observations, manifests and completion receipts,
with [SHA-256 inventory](deadline/report/anchor-v2-final-20261020/source-evidence-inventory.json).
There is no pending experiment or additional demonstrated benchmark gain.

## Preserved earlier work

[UPSTREAM_PROGRESS_README.md](UPSTREAM_PROGRESS_README.md) preserves the previous
README, earlier team progress and the assignment tutorial. Its model scores and
experiment status refer to earlier work. Other existing root-level reports are
also historical unless explicitly linked as fresh experiment evidence above.
