# Reasoning-SFT project: detailed implementation and results report

**Report date:** 2026-10-14  
**Local project:** `/Users/rohitsalla/Documents/Playground/Project-Reasoning-SFT-LLM-main`  
**ARC project:** `/projects/tealab/rohit/reasoning-sft-20261011`  
**Base model:** `Qwen/Qwen2.5-3B-Instruct`

## Scope and status

This document covers only the reasoning-SFT leaderboard project. It does not
include evidence, experiments, or conclusions from the separate continual-RL
project.

The principal training and evaluation work is complete. ARC jobs `1145532` and
`1145533` both completed successfully. The strongest complete
public-suite result is the full-parameter 15,000-example SFT model. It raises the
matched seven-benchmark pass@8 macro from **49.55% to 51.52%**, while reducing
pass@1 from **23.55% to 21.85%** and increasing length-limit terminations from
**141 to 1,964** across the complete nine-benchmark suite.

The selected `alpha=0.50` interpolation was subsequently evaluated on the full
nine-benchmark suite. Its seven-benchmark pass@8 macro is **50.63%**, between
base at **49.55%** and full SFT at **51.52%**. It reduces full-SFT truncations
from 1,964 to 343, but does not preserve the full model's score.

The repaired student-guided experiment generated 2,000 verified, 4K-safe matched
examples and trained both 250-update arms. Guided targets beat the existing-target
control by one pass@8 point on development, then lost by one point on fresh
confirmation and scored 2.5 points below the original base. It supplies no
evidence of a general improvement.

## What was implemented

The repository now contains an auditable end-to-end system for:

- preparing exactly 15,000 decontaminated reasoning examples;
- quality filtering, topic balancing, route signatures, and diversity-aware
  selection;
- response-only SFT, DFT, and anchored DFT objectives;
- LoRA and full-parameter LLaMA-Factory training;
- resumable checkpoints, finite-loss checks, and hash-bound provenance;
- evaluation at temperature `0.6`, top-p `0.95`, eight samples per question,
  and a `32,768` maximum-new-token budget;
- stable record identifiers and clustered resampling for duplicate Kaoyan
  prompts;
- pass@1, pass@8, paired cluster-bootstrap confidence intervals, truncation,
  output length, boxed-answer coverage, and repeated-8-gram diagnostics;
- manifest-bound resumable generation so cached batches cannot be silently
  reused with a different model, dataset, grader, or decoding protocol;
- full-model weight interpolation with tensor-key, dtype, finiteness, and model
  index verification;
- Gen-SSD-style student-guided teacher generation with symbolic final-answer
  verification and matched question identities.

The matched baseline covers 1,974 benchmark records and 15,792 sampled
responses. The suite consists of AMC, AIME25, AIME24, MATH, CN Math 2024,
Kaoyan, Minerva, OlympiadBench, and GPQA. The project macro follows the seven
underlined leaderboard tasks and excludes AIME24 and MATH.

## Final full-parameter 15K experiment

### Training configuration

The final run starts from the original Qwen2.5-3B-Instruct checkpoint and
updates every model parameter.

| Setting | Value |
|---|---:|
| Selected training rows | 15,000 |
| Rows loaded and used | 15,000 |
| Silent/overlength drops | 0 |
| Objective | response-only SFT |
| Training cutoff | 4,096 tokens |
| Learning rate | `2e-6` |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| Optimizer updates | 1,875 / 1,875 |
| Training time | 1:06:29 |
| Final reported loss | 0.5780 |
| Numerical checks | all recorded losses and gradient norms finite |

The final weights remain preserved on ARC at
`runs/lmf-full-15000-sft`. Job `1137895` finished training and eight benchmark
evaluations. Job `1142866` resumed the exact manifest-bound output directory,
completed all 198 GPQA questions, and produced the final nine-benchmark
comparison. No partial suite is treated as complete.

### Complete nine-benchmark result

All values below use the same evaluator, prompts, decoding parameters, sample
count, grading code, and benchmark records for base and candidate.

| Benchmark | Q | Base p@1 | Full-SFT p@1 | Delta | Base p@8 | Full-SFT p@8 | Delta | Base trunc. | Full-SFT trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMC | 40 | 44.38% | 37.50% | -6.88 pp | 77.50% | 70.00% | -7.50 pp | 2 | 49 |
| AIME25 | 30 | 2.50% | 3.33% | +0.83 pp | 13.33% | 20.00% | +6.67 pp | 4 | 54 |
| AIME24 | 30 | 7.08% | 4.58% | -2.50 pp | 23.33% | 16.67% | -6.67 pp | 10 | 71 |
| MATH | 500 | 65.88% | 64.00% | -1.88 pp | 85.20% | 85.00% | -0.20 pp | 14 | 257 |
| CN Math 2024 | 30 | 7.50% | 7.92% | +0.42 pp | 23.33% | 33.33% | +10.00 pp | 10 | 75 |
| Kaoyan | 199 | 22.42% | 21.23% | -1.19 pp | 56.78% | 59.80% | +3.02 pp | 1 | 289 |
| Minerva | 272 | 31.07% | 27.30% | -3.77 pp | 49.63% | 48.90% | -0.74 pp | 4 | 128 |
| OlympiadBench | 675 | 27.50% | 26.33% | -1.17 pp | 49.48% | 49.33% | -0.15 pp | 92 | 871 |
| GPQA | 198 | 29.48% | 29.36% | -0.13 pp | 76.77% | 79.29% | +2.53 pp | 4 | 170 |

Per-benchmark paired pass@8 intervals are:

| Benchmark | Pass@8 delta | Paired 95% CI |
|---|---:|---:|
| AMC | -7.50 pp | [-17.50, 0.00] pp |
| AIME25 | +6.67 pp | [-6.67, +20.00] pp |
| AIME24 | -6.67 pp | [-20.00, +6.67] pp |
| MATH | -0.20 pp | [-2.60, +2.20] pp |
| CN Math 2024 | +10.00 pp | [-10.00, +26.67] pp |
| Kaoyan | +3.02 pp | [-3.47, +9.69] pp |
| Minerva | -0.74 pp | [-4.41, +2.57] pp |
| OlympiadBench | -0.15 pp | [-3.11, +2.81] pp |
| GPQA | +2.53 pp | [-4.04, +8.59] pp |

### Macro result and uncertainty

| Aggregate | Metric | Base | Full SFT | Delta | Paired 95% CI |
|---|---|---:|---:|---:|---:|
| Seven scored benchmarks | pass@1 | 23.5497% | 21.8526% | -1.6971 pp | [-3.1587, -0.2195] pp |
| Seven scored benchmarks | pass@8 | 49.5474% | 51.5222% | +1.9748 pp | [-1.8852, +5.7400] pp |
| All nine benchmarks | pass@1 | 26.4229% | 24.6169% | -1.8061 pp | not aggregated in receipt |
| All nine benchmarks | pass@8 | 50.5962% | 51.3691% | +0.7730 pp | not aggregated in receipt |

The observed seven-benchmark pass@8 gain is positive, but the interval crosses
zero. The pass@1 regression is supported by its interval. The candidate is
therefore the highest measured model by the target pass@8 macro, but the result
does not establish a reliable broad improvement.

The gain is also concentrated. AIME25 gains two solved questions and CN Math
gains three; because both are 30-question tasks with equal macro weight, those
five solves contribute about **+2.38 macro points**. The other five scored
benchmarks collectively contribute about **-0.41 points**. This is evidence of
a real measured change, but not of uniform reasoning improvement.

### Generation behavior

The strongest negative result is termination behavior:

| Diagnostic across nine benchmarks | Base | Full SFT | Change |
|---|---:|---:|---:|
| Truncated samples | 141 / 15,792 | 1,964 / 15,792 | +1,823 |
| Truncation rate | 0.89% | 12.44% | +11.55 pp |

Mean output length increases on every benchmark, often by several multiples.
Examples include AMC from 1,057 to 6,163 tokens, AIME25 from 1,434 to 8,800,
Kaoyan from 869 to 7,373, and OlympiadBench from 1,442 to 6,566. The observed
failure is repetition and overgeneration. The current experiments do not fully
isolate whether its cause is target structure, full-parameter optimization, or
their interaction.

## Weight interpolation

### Method

No new training was performed. For each parameter tensor, the experiment forms

`theta_alpha = theta_base + alpha * (theta_full_sft - theta_base)`

for `alpha` values 0.25, 0.50, and 0.75. Interpolation is computed in float32
and stored in BF16. All three outputs contain 434 matching finite tensors and
complete model indices.

The prespecified selector compares only the three interpolations on a
training-disjoint 200-question development panel. It ranks by pass@8, then
pass@1, fewer truncations, and lower repeated-8-gram fraction. The selected
model is then compared with base on a fresh disjoint 200-question confirmation
panel. Both panels use the official 32K/eight-sample decoding protocol.

### Development result

| Model | Pass@1 | Pass@8 | Trunc. / 1,600 | Mean tokens | Boxed | Repeated 8-gram |
|---|---:|---:|---:|---:|---:|---:|
| Base | 23.31% | 47.50% | 29 | 1,415 | 1,571 | 0.058 |
| Full SFT | 23.00% | **53.00%** | 275 | 6,979 | 1,352 | 0.188 |
| Alpha 0.25 | 23.19% | 45.00% | **25** | 1,417 | 1,574 | 0.059 |
| Alpha 0.50 | **23.25%** | 51.50% | 55 | 2,141 | 1,542 | 0.075 |
| Alpha 0.75 | 22.88% | 50.50% | 119 | 3,678 | 1,480 | 0.115 |

The interpolation-only rule selected **alpha 0.50**. The full endpoint's 53.0%
was higher, but it was not an eligible interpolation and had 275 truncations.

### Fresh confirmation result

| Metric | Base | Alpha 0.50 | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Pass@1 | 26.7500% | 26.1875% | -0.5625 pp | [-2.9375, +1.7500] pp |
| Pass@8 | 53.5000% | 55.5000% | +2.0000 pp | [-2.5000, +7.0000] pp |
| Truncations | 21 | 56 | +35 | — |
| Mean output tokens | 1,290 | 2,265 | +975 | — |
| Boxed-answer samples | 1,580 | 1,540 | -40 | — |
| Repeated-8-gram fraction | 0.060 | 0.079 | +0.019 | — |

Alpha 0.50 retains a numerical pass@8 advantage with much less damage than the
full endpoint, but it does not return to base-like generation. Truncations are
2.67 times the base count, mean length is 1.76 times base, and repetition is
1.32 times base. Both accuracy intervals cross zero.

The selected checkpoint receipt records:

- model-index SHA-256:
  `45e07d77bbeb02675d70133f336cb9e5d86f54929ff1357453a787115fe257b8`;
- 434 tensors;
- total tensor bytes: `6,171,877,376`;
- output dtype: BF16.

This experiment is exploratory because it was motivated by the public-suite
result. The later complete public-suite evaluation is reported in the final
follow-up section below.

## Student-guided verified teacher generation

### Intended comparison

The implemented Gen-SSD-style experiment uses
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` as teacher and the original
Qwen2.5-3B-Instruct as student scorer. It was designed to collect exactly 2,000
symbolically verified guided solutions, then train two matched full-parameter
arms on the same question identities:

1. existing approved target solutions;
2. student-guided teacher target solutions.

The compute-scaled configuration uses 1,024-token chunks, four candidates for
the first chunk and two thereafter, at most 4,096 generated tokens, and up to
three attempts. Both SFT arms would use 250 updates, effective batch eight,
response-only loss, a 4K cutoff, learning rate `2e-6`, and seed 42.

### Actual terminal result

ARC job `1144570` failed after 17:31 during teacher generation. The pinned
teacher revision was
`916b56a44061fd5cd7d6a8fb632557ed4f724f60`. The run atomically preserved 240
question records, including 96 symbolically verified correct trajectories.

The exact failure was a termination-handling bug. vLLM returned candidates with
zero generated tokens and finish reason `stop` for a prefix that was already
complete. The code discarded empty strings and then raised:

`teacher returned no continuation for an active problem`

This was not an out-of-memory error, numerical divergence, or failed SFT run.
No 2,000-row dataset was exported, neither arm trained, and no development or
confirmation evaluation started. There is no score from this experiment.

The implementation now treats an all-empty, all-`stop` candidate set as normal
trajectory termination and includes a regression test. This first job was not
restarted under its original monitor. A later separately authorized run tested
the repaired implementation and is reported in the final follow-up section.

## Earlier experiments and decisions

| Experiment | Scale | Result | Decision |
|---|---:|---|---|
| Filtered-random SFT pilot | 512 rows | 21.88% dev p@8; 187/256 trunc. | rejected |
| Route-diverse SFT pilot | 512 rows | 37.50% dev p@8; 146/256 trunc. | rejected |
| Route-diverse DFT pilot | 512 rows | 46.88% dev p@8; 29/256 trunc. | runner-up |
| Route-diverse anchored DFT pilot | 512 rows | 50.00% dev p@8; 8/256 trunc. | selected for scale-up |
| Route-ASFT full run | 15,000 rows | all 1,875 updates completed; only 3 public tasks evaluated; regressions and overgeneration | no complete macro |
| Capacity-matched route-DFT rescue | 15,000 rows | 43.50% dev p@8 vs 47.50% base; 408 vs 31 trunc. | sealed gate rejected; public suite skipped |
| Concise SFT | 2,000 rows | 40.00% dev p@8; 596 trunc. | gate rejected |
| Concise DFT | 2,000 rows | 46.00% dev p@8; 93 trunc. | gate rejected |
| Full-parameter SFT | 15,000 rows | 51.52% seven-task public p@8 vs 49.55% base | strongest complete public result |
| Weight interpolation | no training | alpha 0.50: 50.63% public macro vs 49.55% base; 343 truncations | below 51.52% full SFT |
| Student-guided teacher targets | 2,000 matched rows × two 250-step arms | guided 51.0% confirm p@8 vs 52.0% control and 53.5% base | hypothesis not confirmed |

The small pilot correctly identified that DFT-style weighting and anchoring
could control output length, but its 32-question selection panel did not predict
full-scale behavior. The 15K route-ASFT and route-DFT results show why small
pilot wins cannot be promoted to leaderboard claims. The concise experiment
shows that selecting shorter existing traces alone is insufficient. The final
full-parameter run demonstrates a measurable pass@8 change, while interpolation
shows that part of that change can survive a smaller parameter displacement.

## Comparison with the illustrative Fall 2025 table

The table supplied with the assignment reports:

- historical Qwen2.5-3B-Instruct baseline AVG: 44.6%;
- historical leading AVG: 52.5%;
- current matched full-SFT macro: 51.52%;
- current matched base macro: 49.55%.

Numerically, the full-SFT candidate is **0.98 percentage points below** the
illustrative 52.5% leader and **1.97 points above** its matched local base.
These numbers do not prove an official position. The historical dataset
revisions, AIME25 source, runtime stack, batching, seeds, and raw prediction
artifacts have not been verified as identical. The local matched base itself is
4.95 points above the historical baseline, which demonstrates that direct rank
comparison would confound model change with evaluation differences.

The defensible statement is:

> Under the repository's matched evaluator, the full-parameter 15K SFT model
> has the highest complete measured pass@8 macro at 51.52%, versus 49.55% for
> the local base. The improvement is statistically uncertain, concentrated in
> two small math sets, and accompanied by a supported pass@1 regression and
> substantially worse termination.

## Evidence boundaries

What the project demonstrates:

- all 15,000 selected examples can be trained for all 1,875 planned updates
  with finite full-parameter optimization;
- the final model has a complete, matched nine-benchmark evaluation;
- its target seven-benchmark pass@8 macro is 1.97 points above matched base;
- the same model has worse pass@1 and much worse stopping behavior;
- alpha-0.50 interpolation reproduces a numerical pass@8 gain on a fresh
  confirmation panel while reducing, but not eliminating, overgeneration;
- the complete pipeline preserves enough manifests, hashes, raw predictions,
  and logs to audit these claims.

What the project does not demonstrate:

- a statistically clear seven-benchmark pass@8 improvement;
- a large-margin win over the Fall 2025 leader;
- exact comparability to the historical leaderboard;
- that the full-SFT gain generalizes uniformly across reasoning domains;
- a confirmed student-guided-target accuracy improvement;
- an alpha-0.50 score above the full-SFT public result.

## Recommended next decision

The best current public candidate remains the completed full-SFT model at
51.52% pass@8. Alpha 0.50 scored 50.63% and offers a better termination tradeoff,
but not a higher target score. The repaired guided-target arm failed fresh
confirmation and should not advance to the public suite. A next experiment
should directly control target length and stopping rather than expanding these
two completed branches.

## Final follow-ups completed on 2026-10-14

Job `1145532` evaluated fixed alpha 0.50 on all nine benchmarks. Its matched
seven-benchmark macro was **23.18% pass@1 / 50.63% pass@8**, versus
**23.55% / 49.55%** for base. The pass@8 delta was +1.08 points with paired
95% CI [-2.34, +4.52]. It produced 343 truncations, compared with 141 for base
and 1,964 for full SFT. This is a useful tradeoff, but it is 0.89 points below
full SFT and 1.87 points below the illustrative historical leader.

Job `1145533` passed its preflight, exported exactly 2,000 verified matched
examples below the 4,096-token serialized limit, and trained both arms for all
250 finite updates. Development pass@8 was 47.5% base, 47.5% existing targets,
and 48.5% guided targets. On fresh confirmation it was 53.5%, 52.0%, and 51.0%,
respectively. Guided versus existing changed by -1.0 point, CI [-7.0, +5.0];
guided versus base changed by -2.5 points, CI [-8.0, +2.5]. Confirmation
truncations were 21, 144, and 191. The guided hypothesis was not confirmed.

## Preserved artifacts

Primary full-SFT evidence:

- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-report.json`
- `results/arc-20261011-snapshot-2316/evaluations/lmf-full-15000-sft-suite-comparison.json`
- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-comparison.svg`

Interpolation evidence:

- `results/arc-20261011-snapshot-2316/interpolation/final-report.json`
- `results/arc-20261011-snapshot-2316/interpolation/interpolation-selection.json`
- `results/arc-20261011-snapshot-2316/interpolation/interpolation-confirm-comparison.json`
- `results/arc-20261011-snapshot-2316/interpolation/interpolation-confirmation.svg`
- `results/arc-20261011-snapshot-2316/interpolation/provenance/alpha-050/`

Student-guided failure evidence:

- `results/arc-20261011-snapshot-2316/gen-ssd-partial/terminal-summary.json`
- `results/arc-20261011-snapshot-2316/gen-ssd-partial/`

Consolidated documentation:

- `FULL_RESULTS_AND_IMPLEMENTATION.md`
- `PROJECT_PROGRESS_20261012.md`
- `INTERPOLATION_EXPERIMENT.md`
- `GEN_SSD_EXPERIMENT.md`
- `IMPLEMENTATION.md`

The latest packaged deliverable is
`/Users/rohitsalla/Documents/Playground/Project-Reasoning-SFT-LLM-interpolation-complete-20261014.zip`,
with SHA-256
`e6a271df5631e858e844db8d1d336edfb9ec229d838a514256dbf9c5d97dee6d`.
It contains code, configuration, tests, documentation, summarized measured
results, and lightweight provenance. Raw generation files and multi-gigabyte
weight shards are intentionally excluded; the remote full and interpolated
weights remain preserved.

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
