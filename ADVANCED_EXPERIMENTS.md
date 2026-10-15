# Advanced strategy and live ARC experiments

The second iteration runs actual Qwen2.5-3B-Instruct experiments on the user's existing ARC `tml_2026` allocation. It does not substitute synthetic test outcomes for benchmark scores.

## Changes grounded in papers

1. **Reasoning-route coverage.** [Selecting Diverse SFT Traces Improves Post-RL Generalization (September 2026)](https://arxiv.org/abs/2609.33780) motivates coverage of visible reasoning routes, beyond question topics. `reasoning/routes.py` builds a 180-dimensional signature from 12 cue families: normalized frequencies, late occurrence fractions, ordered transitions, and presence indicators. It standardizes signatures over the eligible pool and performs quality-weighted farthest-point selection inside question-topic clusters. This is a lightweight adaptation, not the authors' full topology extractor. Their post-RL findings do not establish an improvement for this SFT-only 3B experiment.
2. **Dynamic Fine-Tuning.** [DFT paper](https://arxiv.org/abs/2508.05629) and [authors' code](https://github.com/yongliang-wu/DFT) motivate probability-weighted supervised updates. The objective is `mean(stop_gradient(p_target) * CE_target)` over response tokens. The new ARC LoRA runner implements it with chunked cross entropy; tests compare the gradient with the explicit reference expression. The LLaMA-Factory config generator also exposes the bundled `use_dft_loss` option via `--loss dft`.
3. **Controlled ablations and retention.** The pilot compares filtered-random SFT, route-diverse SFT, route-diverse DFT, and route-diverse anchored DFT. This separates selection from objective changes. [Anchored SFT](https://arxiv.org/abs/2509.23753) warns that probability weighting alone can drift; a fourth pilot implements DFT plus 0.03 times forward KL from the frozen, adapter-disabled base distribution. The coefficient follows the [authors' current mixed-precision recommendation](https://github.com/zhuchichi56/ASFT), rather than the paper's 0.05 math setting. The reference forward pass uses evaluation mode and no gradients. Development accuracy determines whether this anchoring helps.

## Execution plan

- Base benchmark: AMC 2023, 40 questions, 8 samples each, temperature 0.6, top-p 0.95, 32,768 generation upper limit and native 32K total context.
- Pilot curation: 512 examples per variant, 4,096 complete-sequence cutoff. Same clean eligible pool and held-out development questions for all variants.
- Pilot training: BF16 LoRA rank 64/alpha 128, dropout .05, gradient accumulation 8, 64 optimizer steps, learning rate 5e-5, cosine schedule, 3% warmup, seed 42. This is approximately one pass over each 512-example set, not the final 15K experiment.
- Measured pilot response-token budgets: filtered-random **1,289,808**, route-diverse **913,033** (29.2% fewer). Example counts and optimizer steps match; token budgets do not. The selection ablation therefore tests the entire selection policy, including its length distribution, rather than isolating route coverage alone. Objective ablations use the identical route dataset.
- Development: 32 frozen held-out source questions, eight samples per question, temperature .6, top-p .95, and an **8,192-new-token screening limit**, submitted as one 32-question batch for every variant. The initial 32K development attempt was discontinued for runtime cost before a complete development score was available; its directory is retained as incomplete. The shorter limit applies equally to all four variants and can change their ranking relative to a full-budget screen. AMC retains its **32,768-new-token upper limit**, native 32K context, and batches of eight questions for every model. The development winner is evaluated on AMC first. Development answers come from held-out teacher traces and are not independently verified.
- Initial GPU allocation budget: approximately four L40S-hours total. Prior jobs used 2m20s, 17m55s, 10m34s, 11m51s, and 29m16s. The remaining screening/evaluation job is capped at 2h48m. The CPU preparation job had an independent three-hour limit and completed successfully. They leave other existing jobs and environments untouched.
- Full 15K exports are prepared as well. A final training recipe should be chosen from development results; public test scores must not be used to choose it.

## Where to inspect

Local receipts and fetched scores live in `results/arc-20261011/`. The remote project is `/projects/tealab/rohit/reasoning-sft-20261011`.

- Preparation: Tinkercliffs SLURM job `7945713`.
- Base evaluation: Falcon SLURM job `631840` (retry after an initialization-only C++ library-path failure in job `631837`). The baseline completed; its driver then exited 3 because preparation was not ready, so SLURM labels the combined job FAILED. This does not invalidate the complete baseline predictions.
- Pilot training/evaluation: Falcon SLURM job `632295`, bounded to 2h48m. Job `631874` encountered a zero-quota topic-cluster edge case before training; the fix has a regression test. All 19 tests subsequently passed on ARC. Job `632069` completed all 64 control-training steps but failed during merged-model export because optional DeepSpeed import needed `CUDA_HOME`. Supplying CUDA 12.8 only to training/export fixed this; a tiny serialization round-trip passed, and job `632126` successfully exported the step-64 control without retraining. Its 32K development generation was deliberately stopped for runtime cost before a complete development score. Job `632295` reuses the merged control and runs all four development screens at 8K. Official AMC settings are unchanged. Full attempt receipts are in `results/arc-20261011/submission.json`.
- Training script: `reasoning/train_adapter.py` (new single-GPU runner using the existing ARC runtime through a private virtual environment).
- CPU tests: `python3 -m unittest discover -s tests -v`.

Checkpoints carry optimizer, scheduler, Torch RNG and data-order state. Completed adapters are merged for vLLM evaluation. Manifests bind data and code hashes to runs. The initial pilot uses the installed ARC stack (Torch 2.10.0, Transformers 4.57.6, vLLM 0.19.0), so comparisons must use this same stack. It is distinct from the earlier proposed, uninstalled dependency pins.

For export-only resumes, the runner's `complete.json` duration and peak-memory fields describe the resumed process. Use `train.jsonl` for optimization timing; the control's 64 updates took 567.5 seconds before its initial export failure.

No result should be described as beating the historical leaderboard unless it uses that leaderboard's complete benchmark set. The archive lacks AIME25, and its illustrative AVG excludes AIME24 and MATH. A separate 30-question AIME25 supplement is now sourced from `math-ai/aime25` at commit `563bb8404243c5f09de6ec262f2db674fe5bce9b`; this does not verify the historical dataset version. The supplied eight benchmarks remain byte-identical. AMC or development improvements alone cannot establish a leaderboard win.

The supplemental exclusion pass preserves the original held-out membership and found zero additional lexical overlaps in the training pool, held-out pool, or 32-question pilot. Feature reuse checks input bytes, code, model/tokenizer files, feature settings, and package versions. CPU job `7946299` completed a separate 15,000-example 8K export in `artifacts/datasets-final-8k`. The 8K filtered-random and route exports have 65,694,426 and 62,866,732 response tokens (4.3% fewer for route), respectively; validation has 420 eligible examples. This differs from the pilot token-budget ratio. Preparation is not completed model training. Local pipeline and objective checks now pass 24 tests, including duplicate-record handling and safe cache migration.

## Measured baseline

The actual Qwen2.5-3B-Instruct AMC run completed on October 11, 2026: **41.5625% pass@1, 67.5% pass@8**, 40 questions and 320 generations. Three generations reached the length limit. Predictions, grader decisions, hashes and the runtime manifest are in `results/arc-20261011/evaluations/base-amc/`. These values are the local experimental baseline; the archive's historical table reports a different run.


## Full-size continuation (submitted; no completed final score yet)

`scripts/arc_final.sh train` recomputes the choice from all four completed development screens, checks the 15,000-row 8K selection, and runs 1,875 updates with accumulation eight (one pass). It does not silently promote a 512-example pilot to a full-size result. `scripts/arc_final.sh evaluate` evaluates matched base/candidate runs on all nine benchmark names, using the separately sourced AIME25. Optional benchmark arguments allow smaller batches of work or the additional AIME24 (`aime`) and MATH (`math`) sets. The pilot does not launch this continuation itself. The full B200 experiment has a six-hour total allocation cap. Initial job `1119134` failed after 136 seconds without generations because FlashInfer could not find `nvcc`. Retry job `1119463` verified the CUDA 12.8/isolated FlashInfer cache repair, then was stopped after 223 seconds to avoid small-batch long-tail idling. Job `1119701` completed five baseline benchmarks in 936 seconds, then failed because the supplied Kaoyan has four pairs of identical prompts. The evaluator now assigns distinct stable record IDs while retaining all 199 questions, unchanged text and answers. Comparisons bootstrap identical-question clusters together. A validated migration copies the five complete, unchanged benchmark caches byte-for-byte into `full-base-suite-v2`; it refuses to relabel any benchmark with existing generations. Retry `1120144` is capped at 5h38m, keeping all B200 attempts below six allocated GPU-hours. It finishes the baseline, validates all development runs, trains the selected recipe, and evaluates the full candidate. The full suite includes all eight supplied benchmarks plus the separately sourced AIME25; its seven-benchmark macro excludes AIME24 and MATH.

The continuation uses the actual ARC native LoRA runner, rather than requiring a second LLaMA-Factory installation. Full-size 8K memory use and runtime have not been validated; resumable checkpoints are saved every 100 updates. The full B200 job uses fresh baseline/candidate directories, records GPU type/capability/memory in both manifests, and increases decoding concurrency to 128 while queuing each benchmark in one batch (batch-size setting 1024). Both arms retain eight samples, temperature .6, top-p .95, 32K generation/context limits, and the same grader. It does not reuse the L40S pilot baseline as an exactly matched B200 run.

## Development failure diagnosis

The filtered-random SFT control completed 32 questions × eight samples at the common 8K screening limit: 7.421875% pass@1 and 21.875% pass@8. It produced 1,650,115 tokens (mean 6,445.8), with 187/256 length-limited responses. Only 73 responses contained a complete boxed answer. Its mean repeated-word-8-gram fraction was 0.544. Inspection found repeated derivation loops, including responses that reached the correct value but failed to terminate. These are development observations, not an AMC comparison or proof that another objective solves the problem.

This behavior is consistent with the stopping-control failure investigated in [Making Small Language Models Efficient Reasoners](https://arxiv.org/abs/2505.07961). [Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143) also motivates testing reasoning complexity appropriate to small models. No TLDR reinforcement learning, teacher rewriting, or Mix Distillation result is claimed here: the implemented trials remain the declared selection/SFT/DFT/anchoring comparisons using original training responses.


The completed route-SFT screen improved pass@1 to 15.234375% and pass@8 to 37.5%, solving five additional development questions without losing any control-solved question. Paired bootstrap intervals were +2.34375 to +14.453125 percentage points for pass@1 and +3.125 to +28.125 points for pass@8. These unadjusted, single-seed intervals on 32 questions are screening evidence, not a robust leaderboard claim. Mean output length fell to 5,380.2 tokens and repeated-8-gram fraction to 0.435; 146/256 responses still hit the length limit. The selection policy also changes training token length, so this does not isolate route coverage as the cause.


The completed route-DFT screen reached 21.875% pass@1 and 46.875% pass@8, with 479,982 generated tokens (mean 1,874.9), 29/256 length-limited responses, and repeated-8-gram fraction 0.124. Against route-SFT, the observed gains were +6.640625 and +9.375 percentage points, but paired bootstrap intervals crossed zero: [-0.390625, +14.453125] for pass@1 and [-3.125, +21.875] for pass@8. Generation token usage fell by 65.2%. This is a measured efficiency improvement and a preliminary accuracy gain, not proof of a universally superior objective.


## Completed development screen

All rows below use 32 frozen held-out questions × eight samples, 8K generation limit, and 512 training examples / 64 updates. Every pilot used all 512 examples, had finite logged losses/gradient norms, and exported its merged checkpoint.

| Pilot | Pass@1 | Pass@8 | Mean output tokens | Length-limited samples |
|---|---:|---:|---:|---:|
| Filtered-random SFT | 7.42% | 21.88% | 6,445.8 | 187/256 |
| Route-diverse SFT | 15.23% | 37.50% | 5,380.2 | 146/256 |
| Route-diverse DFT | 21.88% | 46.88% | 1,874.9 | 29/256 |
| Route-diverse anchored DFT | 25.00% | 50.00% | 1,245.5 | 8/256 |

The prespecified development criterion selected `route-asft`. Its +3.125-point pass@8 edge over DFT has a paired interval of [-9.375, +15.625] points, so the screen does not establish a reliable accuracy advantage between those two objectives. The selected candidate's output token total is 318,860, 80.7% below the filtered-random control. Training optimization durations were 567.5s, 387.8s, 385.3s, and 501.0s in table order. These are pilot development results; public AMC and full-size benchmark results are recorded separately.

## Completed pilot AMC comparison

The development-selected route-ASFT pilot achieved **38.125% pass@1 and 70.00% pass@8**, versus the matched L40S base at **41.5625% and 67.50%** (40 questions, eight samples each). The paired differences are −3.4375 percentage points for pass@1 (95% interval −9.0625 to +2.1875) and +2.50 points for pass@8 (−10.00 to +15.00). Eight candidate generations hit the length limit versus three for the base. This is a mixed, statistically uncertain result, not a clear win. It is a 512-example pilot, not the full 15K model. Raw predictions and the comparison are preserved in `results/arc-20261011-snapshot-2316/evaluations/`.

The filtered-random SFT AMC control was stopped on October 12 UTC after spending over 30 minutes in its first eight-question batch. It could not complete all 40 questions within the remaining pilot allocation. Its public AMC score is **unavailable**, not zero. All completed development results and the selected pilot AMC run remain valid; full B200 execution is independent.

## Full B200 baseline

All nine B200 baseline evaluations are complete: 1,974 records and 15,792 generated responses. Their seven-benchmark macro is **23.5497% pass@1 / 49.5474% pass@8**; the all-nine macro is **26.4229% / 50.5962%**. These are the matched base results used by the full-size experiments.

| Full B200 baseline | Pass@1 | Pass@8 |
|---|---:|---:|
| amc | 44.38% | 77.50% |
| aime25 | 2.50% | 13.33% |
| aime | 7.08% | 23.33% |
| math | 65.88% | 85.20% |
| cn_math_2024 | 7.50% | 23.33% |
| kaoyan | 22.42% | 56.78% |
| minerva | 31.07% | 49.63% |
| olympiadbench | 27.50% | 49.48% |
| gpqa | 29.48% | 76.77% |

These B200 baselines are distinct from the L40S pilot AMC baseline because hardware and batching differ.

## Capacity-matched rescue

The first full route-ASFT run completed all 1,875 updates but failed its stopping
objective: on AMC, AIME25, and AIME24 it produced 52, 66, and 83 length-limited
samples, versus 2, 4, and 10 for the matched base. Its pass@8 values were 67.5%,
13.33%, and 20.0%, versus 77.5%, 13.33%, and 23.33%. The run was stopped while
MATH was incomplete; no full-suite AVG is claimed.

The revised implementation is documented in [RESCUE_EXPERIMENT.md](RESCUE_EXPERIMENT.md).
It restored the pilot's 4K capacity regime, used a 10K-short/5K-medium route
mixture, route-DFT at 1e-5, retained milestones, and a 200-question accuracy and
termination gate before public evaluation. ARC job `7950879` prepared 15,000
unique examples with 36,706,060 response tokens (mean 2,447.1); all 25 tests
passed.

After explicit approval, Owl job `1123493` completed all 1,875 updates in
6,500.6 seconds. It used all 15,000 rows, dropped zero, and logged finite losses
and gradient norms. The development gate rejected the checkpoint: pass@1 was
19.938% versus 23.375% for base (delta -3.438 points, 95% CI [-6.125, -0.563]);
pass@8 was 43.500% versus 47.500% (delta -4.000 points, 95% CI [-10.500,
+2.500]). Candidate truncations rose to 408/1,600 from 31/1,600. The script
therefore skipped all public benchmarks as designed. There is no rescue
seven-benchmark macro and no evidence of a leaderboard improvement. The final
adapter, complete logs, manifests, and matched 200-question predictions are
preserved under `results/arc-20261011-snapshot-2316/`.
