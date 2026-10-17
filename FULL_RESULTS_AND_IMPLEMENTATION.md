> **Execution update (October 17, 2026):** delayed four-A30 job 655846 was cancelled before GPU use and replaced by full training job **656067 on TWO Falcon A30s**. Two ranks use batch one and accumulation four, preserving effective batch eight, the same 15K complete targets, 16K context, three epochs and 5,625 updates. CPU loss/gradient, two-rank checkpoint and data/config checks passed before submission (27fb563). At most eight two-hour A30-pair chunks consume 32 physical GPU-hours; eight hours remain reserved for final single-B200 evaluation under the unchanged evaluator. Monitoring checks every15 minutes. Queue estimates and completion remain unverified; see [FULL_LENGTH_SFT_EXPERIMENT.md](FULL_LENGTH_SFT_EXPERIMENT.md).

# Reasoning-SFT project: implemented ideas and full measured results

**Report date:** 2026-10-13  
**Base model:** `Qwen/Qwen2.5-3B-Instruct`  
**Local project:** `/Users/rohitsalla/Documents/Playground/Project-Reasoning-SFT-LLM-main`  
**ARC project:** `/projects/tealab/rohit/reasoning-sft-20261011`

## Executive result

The project implemented and tested the proposed advanced reasoning-SFT strategy:
quality and contamination filtering, reasoning-route diversity selection,
student-aware difficulty control, ordinary SFT, Dynamic Fine-Tuning (DFT),
anchored DFT, capacity-matched 15K training, concise-target selection, matched
pass@1/pass@8 evaluation, termination diagnostics, uncertainty estimates, and
sealed development gates.

The implementation is complete and reproducible. The final full-parameter SFT
model has the highest measured matched pass@8 macro, while the unmodified base
retains higher pass@1, substantially shorter outputs, and fewer truncations.

- Matched base seven-benchmark macro: **23.5497% pass@1 / 49.5474% pass@8**.
- Full-SFT seven-benchmark macro: **21.8526% pass@1 / 51.5222% pass@8**.
- Historical Fall 2025 baseline shown by the assignment: **44.6% pass@8 AVG**.
- Historical Fall 2025 leader shown by the assignment: **52.5% pass@8 AVG**.
- The local matched base is numerically 4.95 points above the historical
  baseline and 2.95 points below the historical leader, but exact historical
  comparability is unverified.
- Full SFT completed all nine benchmarks. Its +1.97-point pass@8 change has a
  paired 95% interval that crosses zero; its -1.70-point pass@1 change has an
  interval below zero. No official historical leaderboard AVG is claimed.
- Fixed alpha-0.50 interpolation completed all nine benchmarks at **50.63%**
  seven-benchmark pass@8, below full SFT but with 343 rather than 1,964
  truncations.
- The repaired student-guided 2K experiment completed both matched arms. Guided
  targets scored **51.0%** pass@8 on fresh confirmation, versus **52.0%** for
  existing targets and **53.5%** for base; the strategy was not confirmed.

The supplied Fall 2025 table also contains a numerical inconsistency: its
second-place row prints `52.4%`, but the seven scored columns sum to 3.121 and
average **44.586%**. The first-place row averages to the printed 52.5%. This
strengthens the need to verify the current official evaluator and source table
before treating every historical AVG as a reliable target.

The final concise-target comparison was especially informative. DFT nearly
preserved base accuracy, while ordinary SFT collapsed, but DFT still missed the
sealed gate:

| 200-question development panel | Base | Concise SFT | Concise DFT |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 15.3125% | 23.4375% |
| Pass@8 | 47.500% | 40.000% | 46.000% |
| Length-limited samples | 31/1,600 | 596/1,600 | 93/1,600 |

DFT changed pass@1 by **+0.0625 points**, 95% CI
**[-2.500, +2.4375]**, and pass@8 by **-1.500 points**, 95% CI
**[-7.000, +4.000]**. It also produced 62 more truncations than base, while the
preregistered limit was 40. The fresh confirmation set and public suite were
therefore left unopened.

## Project boundary

This report covers only the reasoning-SFT leaderboard project. It does not use
results, hypotheses, data, or runs from the separate continual-learning or RLVR
capability-frontier project.

## Evaluation protocol

Public benchmark evaluation follows the assignment protocol:

- temperature `0.6`;
- top-p `0.95`;
- eight samples per problem;
- maximum generation budget `32,768` tokens;
- system prompt: `Please reason step by step, and put your final answer within \boxed{}.`;
- the supplied answer extractor and symbolic grader;
- pass@1 as average sample correctness and pass@8 as any-correct accuracy.

Development screens use the same temperature, top-p, sample count, system
prompt, and grader, with an 8,192-token generation budget for practical early
stopping. Comparisons use identical problem IDs and paired bootstrap intervals.
Duplicate Kaoyan prompts are retained as distinct records and resampled in
normalized-question clusters.

The seven-benchmark macro follows the assignment AVG columns represented by
`aime25`, `cn_math_2024`, `kaoyan`, `amc`, `minerva`, `olympiadbench`, and
`gpqa`. AIME24 and MATH are reported but excluded from that macro.

## Ideas implemented

### 1. Quality-filtered, contamination-aware data preparation

The pipeline canonicalizes single-turn records, hashes normalized questions,
deduplicates prompts, reserves held-out questions before selection, and removes
exact or high character-shingle overlap with supplied benchmarks. It measures
tokens with the actual Qwen chat template and requires complete boxed answers,
balanced reasoning tags where applicable, bounded repetition, and complete
serialized examples below the selected context limit.

These checks establish structural quality and separation. They do not prove the
mathematical correctness of every teacher trace.

Primary implementation:

- `reasoning/prepare.py`
- `reasoning/audit.py`
- `reasoning/devset.py`
- `reasoning/data.py`

### 2. Reasoning-route diversity selection

The selection policy favors broad prompt and reasoning-route coverage rather
than repeated responses to the same prompt. It combines topic clustering,
quality gates, route signatures, seeded allocation, and farthest-point style
coverage. Student teacher-forced loss is ranked within response-length bins to
reduce raw-length confounding.

This design was inspired by the data-quality and diversity results in
[AceReason 1.1](https://arxiv.org/abs/2506.13284),
[s1](https://arxiv.org/abs/2501.19393),
[LIMO](https://arxiv.org/abs/2502.03387), and
[OpenThoughts](https://arxiv.org/abs/2506.04178). It is an experimental
synthesis rather than an exact reproduction of one paper.

Primary implementation:

- `reasoning/select.py`
- `reasoning/routes.py`
- `reasoning/features.py`
- `reasoning/capacity.py`

### 3. SFT, DFT, and anchored DFT objectives

Three response-token objectives were implemented in the native single-GPU LoRA
trainer:

1. **SFT:** ordinary response-token cross entropy.
2. **DFT:** detached model-probability weighting of token cross entropy.
3. **Anchored DFT:** DFT plus `0.03 ×` forward KL from the frozen,
   adapter-disabled base distribution.

The anchor tests whether base-distribution retention prevents the drift warned
about by Anchored SFT. Objective tests verify the DFT gradient and the forward-KL
direction.

Primary implementation:

- `reasoning/train_adapter.py`
- `reasoning/objectives.py`
- `tests/test_advanced.py`

### 4. Capacity-matched rescue

After the first 15K model developed severe nontermination, the rescue restored
the pilot's 4,096-token complete-sequence regime, reduced the learning rate to
`1e-5`, switched to route-DFT, retained intermediate adapters every 300 updates,
and placed a sealed 200-question gate before public evaluation.

The 15K mixture contained at least 10,000 shorter examples and 5,000 additional
medium examples. It reduced mean response length from about 4,191 tokens in the
first full run to 2,447 tokens.

Primary implementation:

- `reasoning/capacity.py`
- `scripts/arc_rescue_prepare.sh`
- `scripts/arc_rescue.sh`
- `RESCUE_EXPERIMENT.md`

### 5. Concise-target SFT versus DFT

The final hypothesis tested whether complete, low-repetition, shorter targets
could preserve reasoning while repairing termination. Exactly 2,000 unchanged
approved teacher traces were selected with:

- one complete reasoning block;
- exactly one balanced final box;
- no more than 16 characters after the box;
- no more than 1,536 response tokens;
- repeated word 8-gram fraction at most 0.10;
- complete serialized length below 4,096 tokens.

The selected set contains 2,165,147 response tokens, mean 1,082.57 per example,
and 2,321,662 serialized training tokens. SFT and DFT used the exact same ordered
records, original base model, LoRA rank 64, accumulation eight, `1e-5` learning
rate, and 250 updates.

Primary implementation:

- `reasoning/concise.py`
- `reasoning/choose_concise.py`
- `scripts/arc_concise_fasttrack.sh`
- `CONCISE_TARGET_EXPERIMENT.md`

### 6. Reproducible evaluation and guarded execution

The evaluator records model and adapter hashes, dataset IDs and hashes, grader
hashes, package versions, sampling settings, hardware, predictions, correctness,
and truncation. Atomic cached batches permit safe resume. Comparisons reject
incomplete data or mismatched evaluation settings while correctly allowing model
and LoRA identity to differ.

Development gates prevent weak checkpoints from consuming public evaluation
budget. The concise gate required candidate pass@8 at least equal to base and no
more than 40 additional truncations.

Primary implementation:

- `reasoning/evaluate.py`
- `reasoning/compare.py`
- `reasoning/report.py`
- `reasoning/choose_concise.py`

## Full matched base results

The complete B200 baseline contains 1,974 benchmark problems and 15,792 sampled
responses.

| Benchmark | Problems | Pass@1 | Pass@8 | Length-limited samples |
|---|---:|---:|---:|---:|
| AMC | 40 | 44.375% | 77.500% | 2 |
| AIME25 | 30 | 2.500% | 13.333% | 4 |
| AIME24 | 30 | 7.083% | 23.333% | 10 |
| MATH | 500 | 65.875% | 85.200% | 14 |
| CN_MATH_24 | 30 | 7.500% | 23.333% | 10 |
| KAOYAN | 199 | 22.425% | 56.784% | 1 |
| MINERVA | 272 | 31.066% | 49.632% | 4 |
| OLYMPIADBENCH | 675 | 27.500% | 49.481% | 92 |
| GPQA | 198 | 29.482% | 76.768% | 4 |

| Macro | Pass@1 | Pass@8 |
|---|---:|---:|
| Seven assignment-scored benchmarks | **23.5497%** | **49.5474%** |
| All nine measured benchmarks | **26.4229%** | **50.5962%** |

## Experiment 1: four-way 512-example pilot

All arms used 512 examples, 64 optimizer updates, the same 32 held-out
questions, eight samples per question, and an 8,192-token screening limit.

| Pilot arm | Pass@1 | Pass@8 | Mean output tokens | Truncations |
|---|---:|---:|---:|---:|
| Filtered-random SFT | 7.4219% | 21.875% | 6,445.8 | 187/256 |
| Route-diverse SFT | 15.2344% | 37.500% | 5,380.2 | 146/256 |
| Route-diverse DFT | 21.875% | 46.875% | 1,874.9 | 29/256 |
| Route-diverse anchored DFT | **25.000%** | **50.000%** | **1,245.5** | **8/256** |

The pilot showed that route selection improved over filtered-random SFT and that
DFT-style objectives sharply reduced output length and truncation. The anchored
arm was selected, but its +3.125-point pass@8 advantage over route-DFT was
uncertain: paired 95% CI **[-9.375, +15.625]**.

On matched L40S AMC, the selected pilot scored 38.125% pass@1 and 70.000%
pass@8, versus 41.5625% and 67.500% for base. The differences were -3.4375
points, 95% CI [-9.0625, +2.1875], and +2.500 points, 95% CI
[-10.000, +15.000]. This was a mixed pilot result, not a reliable win.

## Experiment 2: first full 15K route-ASFT

Training completed correctly:

- 15,000 input and used rows;
- zero overlength drops;
- 1,875 optimizer updates;
- 64,275,521 serialized complete-sequence tokens;
- 14,286.8 seconds of optimization;
- finite losses and gradient norms.

Evaluation was stopped after three completed public benchmarks because
nontermination and regression were already material.

| Benchmark | Base P@1 | ASFT P@1 | Base P@8 | ASFT P@8 | Base trunc. | ASFT trunc. |
|---|---:|---:|---:|---:|---:|---:|
| AMC | 44.375% | 40.625% | 77.500% | 67.500% | 2 | 52 |
| AIME25 | 2.500% | 4.167% | 13.333% | 13.333% | 4 | 66 |
| AIME24 | 7.083% | 6.250% | 23.333% | 20.000% | 10 | 83 |

The run has no complete seven-benchmark macro. A three-benchmark partial mean is
not reported as a leaderboard score.

## Experiment 3: capacity-matched 15K route-DFT rescue

The rescue used 15,000 unique 4K-capacity examples, route-DFT, learning rate
`1e-5`, 1,875 updates, and a sealed 200-question development gate.

Training verification:

- 15,000 input and used rows;
- zero drops;
- 38,012,362 serialized tokens;
- 6,500.6 seconds of optimization;
- all 1,875 losses and gradient norms finite;
- final merged model and 479,005,064-byte LoRA adapter written.

| Rescue gate | Base | Candidate | Paired difference |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 19.9375% | -3.4375 pp, 95% CI [-6.125, -0.5625] |
| Pass@8 | 47.500% | 43.500% | -4.000 pp, 95% CI [-10.500, +2.500] |
| Truncations | 31/1,600 | 408/1,600 | +377 |

The checkpoint failed both gate conditions. Public evaluation was skipped.

### Rescue nontermination diagnosis

The 408 length-limited candidate responses were classified as:

- 321 repetitive loops without an answer;
- 75 nonrepetitive unfinished solutions;
- 12 responses that produced an extractable answer and then continued to the
  limit; only one of those 12 was correct.

The base had 31 length-limited responses: 21 repetitive and 10 nonrepetitive
unfinished. Stopping immediately after a box could recover only a small part of
the candidate failure. The dominant problem was unstable or excessively long
reasoning before a usable answer.

The training serialization audit found complete boxed answers and supervised
assistant-end tokens. Base and candidate generation settings matched. This
ruled out missing final-answer labels, missing end-token supervision, and a
sampler mismatch as the main cause.

## Experiment 4: intermediate step-300 rescue diagnostic

The step-300 adapter completed one 200-question diagnostic before the original
concise job stopped on a comparator bug.

| Model | Pass@1 | Pass@8 | Truncations |
|---|---:|---:|---:|
| Base | 23.375% | 47.500% | 31/1,600 |
| Rescue step 300 | 21.5625% | 48.000% | 245/1,600 |

The essentially unchanged pass@8 did not offset 214 additional truncations.
Steps 900 and 1,500 were not evaluated. The comparator defect was fixed and a
regression test added before the fast-track concise run.

## Experiment 5: concise-target SFT versus DFT

Both arms completed all 250 updates on the same 2,000 records. Each used all
2,000 rows, dropped zero, processed 2,321,662 serialized tokens, and logged only
finite losses and gradient norms. Training took 645.1 seconds for SFT and 641.0
seconds for DFT. Both 479,005,064-byte adapters are preserved locally.

| Metric | Base | Concise SFT | Concise DFT |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 15.3125% | 23.4375% |
| Pass@8 | 47.500% | 40.000% | 46.000% |
| Generated tokens | 1,541,195 | 6,206,924 | 2,299,342 |
| Mean output tokens | 963.25 | 3,879.33 | 1,437.09 |
| Truncations | 31 | 596 | 93 |
| Boxed-answer samples | 1,569 | 998 | 1,494 |
| Mean repeated-8-gram fraction | 0.0567 | 0.3027 | 0.0903 |

Paired uncertainty:

| Arm versus base | Pass@1 delta, 95% CI | Pass@8 delta, 95% CI |
|---|---:|---:|
| Concise SFT | -8.0625 pp [-11.000, -5.125] | -7.500 pp [-14.500, -1.000] |
| Concise DFT | +0.0625 pp [-2.500, +2.4375] | -1.500 pp [-7.000, +4.000] |

SFT failed decisively. DFT prevented most of the SFT collapse and preserved
pass@1, but it did not beat base pass@8 and exceeded the truncation allowance by
22 samples. The selection receipt records `selected: null`. Fresh confirmation
and public-suite evaluation were correctly skipped.

## Aggregate interpretation

1. **The advanced ideas were implemented and tested.** The project includes
   route-diverse selection, student-aware length control, DFT, anchored DFT,
   capacity matching, concise-target controls, guarded evaluation, and paired
   uncertainty estimates.
2. **Small-pilot gains did not survive scale.** The 512-example screen favored
   anchored DFT, but the first 15K ASFT checkpoint regressed and generated many
   unfinished outputs.
3. **DFT is consistently safer than ordinary SFT for this data.** In both the
   pilot and concise experiment, DFT greatly reduced the length and repetition
   failure relative to SFT. It did not establish an accuracy win over base.
4. **Selecting shorter existing traces was insufficient.** The concise dataset
   cut target length substantially, yet both trained models generated longer
   responses than base. Target length alone does not control learned stopping.
5. **The observed failure is repetition and overgeneration; its cause remains
   unresolved.** All rows were used, logs were finite, answers and end tokens
   were supervised, evaluation protocols matched, and predictions were
   complete. The cancelled matched-trainer pilot means these results do not
   isolate data, update magnitude, adaptation method, or trainer behavior.
6. **No large-margin result exists.** The final full-parameter model raises the
   matched seven-benchmark pass@8 macro by 1.97 points, but its interval crosses
   zero, pass@1 falls significantly, and truncations rise sharply. It does not
   establish a reliable or historical-leaderboard win.

## What the results support next

The strongest remaining direction is to change the supervision itself rather
than select shorter examples from the same teacher distribution. A credible
next experiment would use newly rewritten or verifier-generated traces that are
short, mathematically complete, and explicitly terminate after the boxed answer.
It should include:

- a fresh SFT control and DFT arm from the original base;
- answer and trace verification, not only structural filtering;
- explicit repetition and post-answer continuation penalties or filtering;
- development selection on accuracy and termination together;
- a fresh confirmation panel before public evaluation;
- multiple seeds before scaling to 15,000 examples.

The existing development set has informed several choices and must not be
treated as fresh confirmation data.

## Verification status

- All current local tests: **27/27 passed** with
  `python3 -m unittest discover -s tests -v`.
- Concise SFT: 250/250 updates, 2,000/2,000 rows, zero drops, finite logs.
- Concise DFT: 250/250 updates, 2,000/2,000 rows, zero drops, finite logs.
- Both concise development evaluations contain all 200 problem records and
  match their manifests.
- Both concise adapters were fetched and verified at 479,005,064 bytes.
- Remote merged weights remain preserved on ARC.

## Artifact index

### Reports

- `PROJECT_PROGRESS_20261012.md` — chronological project status.
- `ADVANCED_EXPERIMENTS.md` — strategy, pilot, and full-run narrative.
- `RESCUE_EXPERIMENT.md` — capacity-matched rescue design and outcome.
- `CONCISE_TARGET_EXPERIMENT.md` — concise-target design and outcome.
- `IMPLEMENTATION.md` — installation and reproducibility guide.

### Exact result receipts

- `results/arc-20261011-snapshot-2316/full-baseline-summary.json`
- `results/arc-20261011-snapshot-2316/evaluations/full-base-suite-v2/metrics.json`
- `results/arc-20261011-snapshot-2316/evaluations/pilot-objective-comparison.json`
- `results/arc-20261011-snapshot-2316/evaluations/pilot-versus-base-amc.json`
- `results/arc-20261011-snapshot-2316/evaluations/final-15000-route-asft-suite/metrics.json`
- `results/arc-20261011-snapshot-2316/evaluations/rescue-dev-comparison.json`
- `results/arc-20261011-snapshot-2316/rescue-nontermination-audit.json`
- `results/arc-20261011-snapshot-2316/evaluations/concise-2000-sft-comparison.json`
- `results/arc-20261011-snapshot-2316/evaluations/concise-2000-dft-comparison.json`
- `results/arc-20261011-snapshot-2316/evaluations/concise-2000-selection.json`

### Preserved adapters

- `results/arc-20261011-snapshot-2316/adapters/rescue-15000-capacity-route-dft/`
- `results/arc-20261011-snapshot-2316/adapters/concise-2000-sft/`
- `results/arc-20261011-snapshot-2316/adapters/concise-2000-dft/`

## Evidence boundary

The internal comparisons are matched and auditable. Exact equivalence to the
Fall 2025 leaderboard is not established because the historical dataset
revisions, AIME25 source, runtime stack, batching behavior, seeds, and complete
prediction artifacts are unavailable. Numerical comparison to that table is
context only. This report makes no official leaderboard-rank or winning-model
claim.

## Full-parameter LLaMA-Factory 15K run

Job `1137895` replaced the cancelled 2K trainer pilot. It trained every model
parameter on all 15,000 selected rows for all 1,875 planned updates. Training
completed in 1:06:29 with finite losses and gradient norms; the final training
loss was 0.5780. Job `1142866` then completed the missing GPQA evaluation and
produced a verified nine-benchmark comparison.

| Benchmark | Base pass@1 | Full-SFT pass@1 | Base pass@8 | Full-SFT pass@8 | Pass@8 delta | Base trunc. | Full-SFT trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMC | 44.38% | 37.50% | 77.50% | 70.00% | -7.50 pp | 2 | 49 |
| AIME25 | 2.50% | 3.33% | 13.33% | 20.00% | +6.67 pp | 4 | 54 |
| AIME24 | 7.08% | 4.58% | 23.33% | 16.67% | -6.67 pp | 10 | 71 |
| MATH | 65.88% | 64.00% | 85.20% | 85.00% | -0.20 pp | 14 | 257 |
| CN Math 2024 | 7.50% | 7.92% | 23.33% | 33.33% | +10.00 pp | 10 | 75 |
| Kaoyan | 22.42% | 21.23% | 56.78% | 59.80% | +3.02 pp | 1 | 289 |
| Minerva | 31.07% | 27.30% | 49.63% | 48.90% | -0.74 pp | 4 | 128 |
| OlympiadBench | 27.50% | 26.33% | 49.48% | 49.33% | -0.15 pp | 92 | 871 |
| GPQA | 29.48% | 29.36% | 76.77% | 79.29% | +2.53 pp | 4 | 170 |

The matched seven-benchmark macro used by this project (AIME25, CN Math 2024,
Kaoyan, AMC, Minerva, OlympiadBench, and GPQA) is:

| Metric | Base | Full-SFT | Delta | Paired cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| pass@1 | 23.55% | 21.85% | -1.70 pp | [-3.16, -0.22] pp |
| pass@8 | 49.55% | 51.52% | +1.97 pp | [-1.89, +5.74] pp |

The nine-benchmark unweighted macro changes from 50.60% to 51.37% pass@8 and
from 26.42% to 24.62% pass@1. The pass@8 macro gain is not statistically clear,
and it is accompanied by a supported pass@1 loss and a truncation increase from
141 to 1,964 samples. Mean candidate output length rose sharply on every major
benchmark. This model is the strongest measured candidate by matched pass@8
macro, but it does not establish a reliable improvement or a historical
leaderboard win. Its 51.52% matched macro is numerically 0.98 points below the
illustrative Fall 2025 leader's 52.5%, with historical comparability unverified.

Complete artifacts:

- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-report.json`
- `results/arc-20261011-snapshot-2316/evaluations/lmf-full-15000-sft-suite-comparison.json`
- `results/arc-20261011-snapshot-2316/lmf-full-15000-final-comparison.svg`
# Student-guided generation follow-up (job 1144570)

The separate Gen-SSD-style 2K experiment failed during teacher-data generation
after preserving 240 question receipts, including 96 verified-correct traces.
vLLM returned zero-token `stop` completions for a prefix that had already
terminated; the first implementation discarded those empty completions and
raised an exception. No training or evaluation occurred, so this run supplies
no accuracy result. The termination case is fixed and tested in the source,
but the job was not restarted.

## Weight-interpolation confirmation

The no-training interpolation run selected alpha 0.50 on the training-disjoint
development panel. On a fresh 200-question confirmation panel, alpha 0.50
reached 55.50% pass@8 versus 53.50% for base, a paired +2.00-point delta with
95% CI [−2.50, +7.00]. Pass@1 was 26.19% versus 26.75%, a −0.56-point delta
with CI [−2.94, +1.75]. The accuracy differences remain uncertain.
Truncations rose from 21 to 56, mean output length from 1,290 to 2,265, and
repeated-8-gram fraction from 0.060 to 0.079. This supports the tradeoff
hypothesis but does not establish a public-suite improvement.

## Final interpolation and student-guided follow-ups

ARC job `1145532` fixed alpha at 0.50 before evaluating the complete nine-task
suite. The seven-task macro was 23.18% pass@1 and 50.63% pass@8, compared with
23.55% and 49.55% for base. The +1.08-point pass@8 interval was [-2.34, +4.52].
Total truncations were 343, between base's 141 and full SFT's 1,964. The model
does not beat full SFT's 51.52% target score.

ARC job `1145533` exported 2,000 symbolically answer-verified guided traces and
matched existing targets with identical ordered IDs. Exact serialized maxima
were 4,095 and 4,064 tokens, respectively. Both arms completed 250 finite
full-parameter updates. Guided passed the development gate by +1.0 pass@8 point
against existing targets, but on fresh confirmation scored 51.0% versus 52.0%
for existing targets and 53.5% for base. It also increased truncations to 191,
versus 144 and 21. No public evaluation was performed for the 2K arms.

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
