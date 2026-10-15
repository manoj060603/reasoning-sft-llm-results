# Reasoning-SFT leaderboard project: progress and measured results

Date: 2026-10-12  
Project root: `/Users/rohitsalla/Documents/Playground/Project-Reasoning-SFT-LLM-main`  
ARC root: `/projects/tealab/rohit/reasoning-sft-20261011`  
Base model: `Qwen/Qwen2.5-3B-Instruct`

## Scope boundary

This report covers only the 15,000-example reasoning-SFT leaderboard project:
data selection, SFT objectives, Qwen2.5-3B training, and the nine mathematical
reasoning benchmarks. It does not use or refer to the separate continual-learning
or RLVR capability-frontier project. No result from that project is evidence for
the scores reported here.

## Current status

The implementation, benchmark harness, matched base evaluation, four-way pilot,
three 15K training runs, and the final full-parameter nine-benchmark evaluation
are complete. Full SFT has the highest measured pass@8 macro (51.52% versus
49.55% for base); base retains higher pass@1 and far better termination.

Two final follow-ups are also complete. Alpha-0.50 interpolation scored 50.63%
on the complete seven-benchmark macro with 343 total suite truncations, below
full SFT but much better behaved. The repaired 2K student-guided comparison
trained both matched arms; guided passed development by one pass@8 point against
the existing-target control, then lost on fresh confirmation (51.0% guided,
52.0% existing, 53.5% base) and increased truncations to 191. Neither follow-up
supersedes full SFT as the strongest public candidate.

The rescue job `1123493` is terminal. SLURM reports `FAILED` with exit code `1:0`
after 2:05:05 because the script deliberately raised an error when the sealed
development gate rejected the checkpoint. This is an intentional gate outcome,
not an interrupted training run.

The follow-up concise-target job `1133003` is also terminal, but it did not run
the planned training comparison. It stopped after 10:47 when the comparison
utility incorrectly treated `lora_identity` as a protocol field. Before that
failure, rescue milestone step 300 completed the 200-question diagnostic:
21.5625% pass@1, 48.000% pass@8, and 245/1,600 length-limited samples, versus
23.375%, 47.500%, and 31/1,600 for the matched base. The +0.5-point pass@8 change
does not offset the additional 214 truncations. Steps 900 and 1,500 were not
evaluated, and the 2,000-example concise SFT/DFT arms never started. There is no
concise-target model score to report.

The repaired fast-track job `1133616` subsequently completed both 2,000-example
arms in 38:13. Both used all 2,000 rows with zero drops and completed 250 updates
with finite logs. On the 200-question development panel, concise SFT scored
15.3125% pass@1 and 40.000% pass@8 with 596/1,600 truncations. Concise DFT scored
23.4375% pass@1 and 46.000% pass@8 with 93/1,600 truncations. The matched base was
23.375%, 47.500%, and 31/1,600. DFT's paired changes were +0.0625 points pass@1
(95% CI [-2.500, +2.4375]) and -1.500 points pass@8 (95% CI [-7.000, +4.000]).
Neither arm passed the sealed gate, so the fresh confirmation set and public
suite remained unopened. The final selection is `null`; the matched base remains
the strongest defensible model.

There is no measured large-margin or statistically clear leaderboard
improvement. The full-SFT candidate has a complete nine-benchmark score, a
tentative +1.97-point matched pass@8 macro signal, and a supported pass@1
regression with severe overgeneration.

## What has been implemented

- Exact benchmark evaluation at temperature 0.6, top-p 0.95, eight samples per
  question, and a 32,768-token generation upper limit.
- Stable record identifiers and duplicate-question clustering for the supplied
  199-row Kaoyan set.
- Nine matched benchmark evaluations: the eight supplied datasets plus a
  separately sourced AIME25 supplement.
- Quality filtering, topic balancing, reasoning-route signatures, and
  quality-weighted farthest-point selection.
- SFT, Dynamic Fine-Tuning (DFT), and anchored DFT objectives.
- LoRA training with resumable checkpoints, finite-loss and gradient checks,
  merged-model export, provenance manifests, and hash-bound result receipts.
- A four-way 512-example pilot and two full 15,000-example training attempts.
- A held-out 200-question gate that prevents a weak full checkpoint from
  consuming public-benchmark evaluation budget.

Local pipeline and objective tests passed before the full runs. The fetched
snapshot preserves training logs, manifests, predictions, comparison files, the
final rescue adapter, and figures under
`results/arc-20261011-snapshot-2316/`.

## Matched full B200 baseline

The baseline contains 1,974 benchmark records and 15,792 sampled responses.

| Benchmark | Questions | Pass@1 | Pass@8 | Length-limited samples |
|---|---:|---:|---:|---:|
| AMC | 40 | 44.38% | 77.50% | 2 |
| AIME25 | 30 | 2.50% | 13.33% | 4 |
| AIME24 | 30 | 7.08% | 23.33% | 10 |
| MATH | 500 | 65.88% | 85.20% | 14 |
| CN_MATH_24 | 30 | 7.50% | 23.33% | 10 |
| KAOYAN | 199 | 22.42% | 56.78% | 1 |
| MINERVA | 272 | 31.07% | 49.63% | 4 |
| OLYMPIADBENCH | 675 | 27.50% | 49.48% | 92 |
| GPQA | 198 | 29.48% | 76.77% | 4 |

The seven-benchmark macro used by this project excludes AIME24 and MATH:

- **Pass@1: 23.5497%**
- **Pass@8: 49.5474%**

The all-nine macro is 26.4229% pass@1 and 50.5962% pass@8.

The historical Fall 2025 table's leading AVG is 52.5%. Our matched baseline's
49.55% seven-benchmark pass@8 macro is about 2.95 percentage points lower, but
the historical dataset revisions, runtime, batching, and AIME25 source have not
been verified as identical. This is context, not an official rank comparison.

## Four-way 512-example pilot

All four pilots used the same 32 held-out development questions, eight samples
per question, 512 training examples, 64 optimizer updates, and an 8K screening
limit.

| Pilot | Development pass@1 | Development pass@8 | Mean output tokens | Length-limited samples |
|---|---:|---:|---:|---:|
| Filtered-random SFT | 7.42% | 21.88% | 6,445.8 | 187/256 |
| Route-diverse SFT | 15.23% | 37.50% | 5,380.2 | 146/256 |
| Route-diverse DFT | 21.88% | 46.88% | 1,874.9 | 29/256 |
| Route-diverse anchored DFT | **25.00%** | **50.00%** | **1,245.5** | **8/256** |

The sealed pilot rule selected route-ASFT. Its +3.125-point pass@8 advantage
over route-DFT was uncertain: paired 95% interval [-9.375, +15.625] points. The
pilot therefore suggested that DFT-style weighting and anchoring improved
stopping behavior, while it did not establish which objective had better true
accuracy.

On matched L40S AMC, the selected pilot scored 38.125% pass@1 and 70.00% pass@8,
versus 41.5625% and 67.50% for the base. The changes were -3.4375 and +2.50
points, with intervals crossing zero. This was a mixed pilot result rather than
a reliable win.

## First full 15K attempt: route-ASFT

Training itself completed correctly:

- 15,000 input rows and 15,000 used rows;
- zero overlength drops;
- 1,875 optimizer updates;
- 64,275,521 complete-sequence tokens;
- 14,286.8 seconds of optimization;
- finite logged losses and gradient norms.

The full candidate reproduced the long-response/stopping failure that the pilot
had partly controlled. Evaluation was stopped after three benchmarks because the
checkpoint was already materially worse and length-limited generations were
high.

| Benchmark | Base pass@1 | Candidate pass@1 | Base pass@8 | Candidate pass@8 | Base trunc. | Candidate trunc. |
|---|---:|---:|---:|---:|---:|---:|
| AMC | 44.38% | 40.63% | 77.50% | 67.50% | 2 | 52 |
| AIME25 | 2.50% | 4.17% | 13.33% | 13.33% | 4 | 66 |
| AIME24 | 7.08% | 6.25% | 23.33% | 20.00% | 10 | 83 |

This run has no complete seven-benchmark macro. Reporting one from three
benchmarks would be misleading.

## Capacity-matched 15K rescue: route-DFT

The rescue addressed the pilot-to-full extrapolation:

- a 4,096-token complete-sequence cutoff instead of 8,192;
- 15,000 unique route-diverse examples;
- 36,706,060 response tokens, mean 2,447.1 per selected example;
- route-DFT instead of anchored DFT;
- learning rate 1e-5 instead of 5e-5;
- adapter checkpoints every 300 updates;
- a sealed 200-question base-versus-candidate gate before public evaluation.

Training completed correctly:

- 15,000 input rows and 15,000 used rows;
- zero dropped rows;
- all 1,875 updates;
- 38,012,362 serialized complete-sequence tokens;
- 6,500.6 seconds of optimization;
- all logged losses and gradient norms finite;
- final adapter and merged checkpoint written.

The development gate rejected the checkpoint:

| 200 questions × 8 samples | Base | Rescue candidate | Delta |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 19.938% | -3.438 pp, 95% CI [-6.125, -0.563] |
| Pass@8 | 47.500% | 43.500% | -4.000 pp, 95% CI [-10.500, +2.500] |
| Length-limited generations | 31/1,600 | 408/1,600 | +377 |

The sealed rules required candidate pass@8 to be at least the base and allowed
at most 40 additional truncations. The candidate failed both conditions. Public
evaluation was intentionally skipped, so it has no leaderboard AVG.

## What the evidence says

1. Route selection helped the small pilot relative to filtered-random SFT, but
   the selection policy also changed the length distribution. The pilot does not
   isolate route diversity as the cause.
2. DFT and anchoring substantially reduced pilot response length and
   length-limit failures. Their accuracy advantage over each other was uncertain.
3. Scaling the recipe from 512 to 15,000 examples changed the training-token
   distribution enough to invalidate the pilot extrapolation.
4. Reducing sequence capacity and learning rate made training faster and
   cheaper, but it did not fix generalization or stopping at full scale.
5. Both full candidates exhibit excessive nontermination. Better data selection
   alone is unlikely to produce a large leaderboard margin unless target traces
   are shortened, repaired, or regenerated.
6. The base model is already strong on AMC, MATH, and GPQA. The largest matched
   weaknesses are AIME25, AIME24, and CN_MATH_24. Improving only AMC cannot move
   the seven-benchmark macro enough to establish a large win.

## Current best model and defensible claim

The full-parameter 15K LLaMA-Factory model has the highest measured matched
pass@8 macro: 51.52% versus 49.55% for base. The +1.97-point interval crosses
zero, pass@1 regresses by 1.70 points with an interval below zero, and
truncations rise from 141 to 1,964. The defensible claim is therefore a tentative
pass@8 signal accompanied by a clear efficiency and pass@1 regression.

The project has produced a validated training/evaluation system and three
full-scale results. It has not produced a statistically clear improvement or a
model that exceeds the illustrative historical 52.5% leader.

## Most important constraints for the next idea

Any next recipe should satisfy these requirements before another full 15K run:

1. Demonstrate improved accuracy and controlled truncation on the 200-question
   gate, not only the 32-question pilot.
2. Use short, verifier-checked targets or teacher-rewritten traces with an
   explicit token budget. The raw long teacher traces have failed twice.
3. Include a matched base and a simple SFT control under the same prompt,
   sampler, hardware, and context limits.
4. Tune only on held-out development data. Keep the nine benchmark test sets
   sealed until the gate passes.
5. Optimize the seven-benchmark macro rather than AMC alone, with explicit
   attention to AIME25 and CN_MATH_24.
6. Stop early when output length or repeated-token diagnostics exceed the base;
   both were leading indicators of the full-run failures.
7. Require a clear pilot margin before scaling. The previous ASFT-versus-DFT
   difference was much smaller than its uncertainty.

## Artifact index

- Full baseline metrics:
  `results/arc-20261011-snapshot-2316/evaluations/full-base-suite-v2/metrics.json`
- Pilot objective comparison:
  `results/arc-20261011-snapshot-2316/evaluations/pilot-objective-comparison.json`
- Pilot AMC metrics:
  `results/arc-20261011-snapshot-2316/evaluations/pilot-route-asft-amc/metrics.json`
- Partial first-full-run metrics:
  `results/arc-20261011-snapshot-2316/evaluations/final-15000-route-asft-suite/metrics.json`
- Rescue gate comparison:
  `results/arc-20261011-snapshot-2316/evaluations/rescue-dev-comparison.json`
- Rescue training receipt:
  `results/arc-20261011-snapshot-2316/training/rescue-15000-capacity-route-dft/complete.json`
- Rescue adapter:
  `results/arc-20261011-snapshot-2316/adapters/rescue-15000-capacity-route-dft/adapter_model.safetensors`
- Experiment narrative: `ADVANCED_EXPERIMENTS.md`
- Rescue design and terminal outcome: `RESCUE_EXPERIMENT.md`
- Implementation guide: `IMPLEMENTATION.md`

## Reproducibility limitation

The supplied benchmark files and matched local comparisons are auditable. Exact
historical leaderboard comparability remains unverified because its dataset
revisions, runtime stack, batching behavior, and AIME25 source are unavailable.
Accordingly, this report makes no official leaderboard-rank claim.

## 2026-10-14: full-parameter LLaMA-Factory run

The 2K LLaMA-Factory pilot was cancelled at the user's request and replaced by
a full 15K, full-parameter run. Job `1137895` used all 15,000 rows and completed
1,875/1,875 updates with finite logs. Training took 1:06:29 and final model
weights were saved successfully. Job `1142866` resumed the exact evaluation
manifest and completed the ninth benchmark, GPQA.

The final matched seven-benchmark pass@8 macro is **51.52%**, versus **49.55%**
for base, a +1.97-point change with paired 95% CI **[-1.89, +5.74]**. Pass@1
falls from **23.55% to 21.85%**, a -1.70-point change with paired 95% CI
**[-3.16, -0.22]**. Total truncations increase from 141 to 1,964. The candidate
therefore has the best observed matched pass@8 macro, but the gain is uncertain
and comes with a supported pass@1 regression and severe overgeneration.

The candidate's 51.52% is numerically below the illustrative Fall 2025 leader's
52.5%, and exact historical dataset/runtime equivalence is not verified. No
official leaderboard-win claim is supported.

Exact results and the final figure are:

- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-report.json`
- `results/arc-20261011-snapshot-2316/evaluations/lmf-full-15000-sft-suite-comparison.json`
- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-comparison.svg`
# Student-guided 2K experiment

ARC job `1144570` terminated during data generation after 240 saved questions
and 96 verified-correct traces. The cause was a local zero-token EOS handling
bug, not a training divergence or GPU failure. No SFT arm ran and no score can
be reported. Partial receipts were fetched non-destructively, and the source
now handles this vLLM termination case with a regression test. The job was not
restarted.

## Completed interpolation follow-up

ARC job `1144222` completed all five development evaluations and matched
base/selected confirmation. Alpha 0.50 was selected. On fresh confirmation it
scored 26.19% pass@1 and 55.50% pass@8, compared with base at 26.75% and
53.50%. The pass@8 delta was +2.00 points with paired 95% CI [−2.50, +7.00];
the pass@1 delta was −0.56 points with CI [−2.94, +1.75]. Truncations increased
21→56 and mean output tokens 1,290→2,265. This is a promising numerical pass@8
result with unresolved uncertainty and generation regressions, not a public
suite or historical leaderboard confirmation.
