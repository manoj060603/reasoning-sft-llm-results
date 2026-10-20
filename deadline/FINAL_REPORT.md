# Report: Cross-fitted Transfer Gain with Coverage for Reasoning SFT

**Snapshot: 20 October 2026, 1:41 p.m. Eastern / 17:41 UTC.** Both full-SFT runs and all three primary evaluations are complete: each model has nine raw-verified tasks, including all eight supplied benchmarks and supplementary AIME25. Native CPU closeout produced a matched audit, complete macros, paired uncertainty and publication metadata. The advanced policy improves over the prepared-source random control, while both SFT models perform worse than the original on every supplied task. Both trained models and both original 15K datasets are published and anonymously verified at pinned Hugging Face commits. The separately labeled post-hoc recovery also has all nine matched raw evaluations, its own verified public checkpoint and complete conditional uncertainty analysis. Its supplied-eight point estimate has slightly lower pass@1 and higher pass@8 than original, with both difference intervals including zero. Terminal GPU accounting is complete and the final source snapshot is ready; manual team submission remains pending. The recovery results do not replace the primary experiment.

We investigate whether a source-only selection policy can improve reasoning SFT over a random control. Two small LoRA training runs measure how examples in the opposite source fold respond to adaptation. The final selector combines relative response-loss reduction with structural eligibility and topic/length coverage. Both final models start from the original Qwen2.5-3B-Instruct checkpoint. Previous team checkpoints and scores are not results of this experiment.

The original model achieves **29.30267835% pass@1 / 52.86146036% pass@8** on the supplied-eight equal-benchmark macro, compared with random **16.23217035% / 36.69876417%** and advanced **18.78695793% / 42.95429078%**. Advanced exceeds random by **2.5548 percentage points pass@1** and **6.2555 points pass@8**; their conditional 95% paired bootstrap intervals are **[1.6733, 3.4813]** and **[3.1852, 9.5150]** points. This answers the assignment's random-control comparison positively for this fixed experiment, while the comparison against the original is negative. The [assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM#8-submission) asks for original and fine-tuned performance, comparison and analysis, final hyperparameters, and the selection method; all primary findings are reported below.

## 1. Data and comparison scope

The source is the assignment's [redsgnaoh/acereason11_100k](https://huggingface.co/datasets/redsgnaoh/acereason11_100k) subset. We add no external training examples and generate no replacement training targets. Canonicalization folds any separate input into the instruction, trims surrounding whitespace, and applies a fixed system prompt. Selection and dataset export then preserve those canonical prompt/response pairs without shortening or rewriting responses.

Preparation deduplicates normalized question identities, reserves source-only held-out questions, excludes lexical overlap with public benchmark questions, and applies a prompt-capacity bound that leaves room for supervision. Benchmark questions are used only for exclusion; their reference answers and measured outcomes do not rank training examples. Supplementary AIME25 exclusions are applied before probe training. All source-held-out questions and both probe training sets are excluded from both final arms. Lexical exclusion reduces direct overlap but cannot rule out semantic contamination or overlap already present in the base model's pretraining.

| Completed data preparation and selection | Observed value |
| --- | ---: |
| Broad prepared source pool | 83,029 rows |
| Reserved probe examples | 1,024 total; 512 per fold |
| Random sampling population after probe reservations | 82,005 rows |
| Frozen candidate scoring reservoir | 30,000 rows |
| Random control export | 15,000 unique canonical pairs |
| Advanced export | 15,000 unique canonical pairs |
| Advanced training rows truncated or dropped | 0 |

The random control is sampled uniformly, without replacement, from the **82,005-row broad prepared source population**, using selection seed `20261019`. It does not apply the advanced boxed-answer, repetition, complete-target length, topic, or gain criteria. The assignment describes random selection from its 100K source; our control is random after the shared preparation and reservation exclusions, **not a literal uniform draw over the untouched 100K**. This distinction must accompany the final submission; no course-specific acceptance of that interpretation is assumed.

Original exported datasets retain complete canonical responses in both arms. The training token cache is a separate artifact: random sequences longer than 8,192 tokens use their original prefix, without appending a fabricated EOS. Exactly **5,806 of 15,000 random rows (38.71%)** are prefix-truncated; the training cache audit confirms those same **5,806 rows lack a supervised EOS**. Advanced selection requires complete original targets to fit and records zero truncations. Neither arm silently drops selected examples.

## 2. Data-selection method and rationale

[DavIR](https://arxiv.org/abs/2310.13008v2) motivates using relative loss reduction during fine-tuning as a model-dependent selection signal. [SmallToLarge (S2L)](https://arxiv.org/abs/2403.07384v2) motivates extracting selection information from training trajectories. Our design is an experimental synthesis of those ideas with opposite-fold probes and coverage constraints. It does not reproduce either paper's complete algorithm: the probes use LoRA on the same base architecture, rather than a smaller model family, and scoring uses one completed probe checkpoint per fold. Global novelty and superior benchmark performance are not established.

The procedure is:

1. Split source question identities deterministically into two folds. Reserve 512 eligible examples per fold for two disjoint probes.
2. Train each rank-eight LoRA probe from the original base for 64 updates, with effective batch eight and learning rate `5e-5`. Probe examples are selection instruments and never enter either final training set.
3. Score each frozen reservoir example with the original model and the probe trained on its opposite fold. For a candidate `i`, compute `gain_i = (CE_base(i) - CE_opposite_probe(i)) / max(CE_base(i), 1e-8)`. Each CE averages shifted, response-only token losses for that original target.
4. Allocate capacity-capped quotas across question clusters using square-root cluster weights, then allocate response-length cells proportionally within each cluster. At most `floor(cell_quota / 2)` places use the highest positive-gain candidates; deterministic seeded coverage fills the remainder.
5. Export exactly 15,000 selected canonical pairs and initialize the final full-parameter model from the original base, not from a probe.

Advanced eligibility requires at least 128 response tokens, a complete boxed answer, total serialized length at most `8192 - 32` tokens, repeated eight-gram fraction at most `0.15`, and simple think-marker sanity checks. These are structural checks, not mathematical verification. Question coverage uses character TF-IDF features, a 64-dimensional SVD representation, and 64 MiniBatchKMeans clusters; fixed response-length strata constrain the final allocation. The initial reservoir is coverage-balanced and frozen before gain scoring.

Selection is now complete: **7,438 examples entered through gain-leader slots and 7,562 through seeded coverage**. All **30,000 scored reservoir candidates had positive gain**, including all 15,000 selected examples. Therefore, positivity alone did not distinguish selected from rejected reservoir examples; relative ordering and coverage quotas determined the selected set. This observation demonstrates response-loss reduction under the probes, not a downstream accuracy benefit.

The initial 4K CPU capacity audit found only 15,938 eligible complete targets, below the 16,024 needed for 15,000 final examples plus 1,024 probe reservations. Both arms and probes moved to 8K before GPU training or score computation. The broad random control was also fixed before those outcomes were available. These design corrections are recorded in [prelaunch_corrections.json](prelaunch_corrections.json).

## 3. Final model hyperparameters

The following recipe was used for **both** completed final SFT arms. Mirrored training-completion and runtime receipts establish the actual update counts, exact example consumption, model integrity and runtime for each arm. They establish training completion rather than benchmark superiority.

| Setting | Random and advanced final models |
| --- | --- |
| Initialization | Original `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Framework | Native Hugging Face Transformers Trainer `4.57.3` |
| Trainable parameters | Full model; no final-model LoRA |
| Dataset size and pass count | 15,000 unique examples; one pass |
| Updates | 1,875 |
| Microbatch / accumulation / global batch | 1 / 8 / 8 on one GPU |
| Training context | 8,192 tokens |
| Optimizer | PyTorch AdamW |
| Learning rate / schedule | `2e-6` / cosine |
| Warmup ratio / weight decay | `0.03` / `0.01` |
| Gradient norm clipping | `1.0`, with a finite check before the optimizer update |
| Parameter and optimizer-moment storage | FP32 |
| Forward computation | BF16 autocast |
| Attention / memory controls | SDPA, gradient checkpointing, chunked FP32 cross entropy |
| Training and data seed | `42` |
| Checkpoint policy | Every 256 updates; retain two; evaluate the fixed final checkpoint |

The [random training receipt](report/random_training_receipt.json) and [advanced training receipt](report/advanced_training_receipt.json) each confirm **1,875 updates, 15,000 unique examples, exactly 15,000 presentations, and zero padding presentations**. Each serialized model contains 434 verified finite FP32 tensors, with hashes for all three weight shards.

| Completed-run measurement | Random | Advanced |
| --- | ---: | ---: |
| Training through final verification | 6,646.64 seconds / 110.78 minutes | 5,226.94 seconds / 87.12 minutes |
| Peak PyTorch allocated GPU memory | 61,797,884,416 bytes | 61,806,027,264 bytes |
| Verified finite FP32 tensors | 434 | 434 |

The [random runtime](report/random_training_runtime.json) and [advanced runtime](report/advanced_training_runtime.json) receipts both record one NVIDIA B200, Torch 2.9.0, Transformers 4.57.3 and Accelerate 1.12.0. These timings describe training through final model verification; scheduler allocation time also includes job setup and is accounted for separately. The mirrored [native bootstrap preflight](report/evaluation_bootstrap_preflight-8067759.json) records 123 tests and six timed/untimed grader cases, with zero failures, errors or skips. The later [continuation guard preflight](operations/runtime-contingency-native-qa.json) records 30 passing budget/pipeline tests in an existing compute allocation, identical evaluation argument tails and valid wrapper syntax.

The loss masks the system and user prompt. It averages shifted supervised-token cross entropy **within each example**, then averages the eight examples contributing to an update. This avoids implicitly assigning a larger update weight to a longer response. Native Qwen end markers and template trailers remain supervised when present; prefix-truncated random examples can lose them. Parameter storage and Adam moments stay FP32 so small learning-rate updates can accumulate rather than being rounded directly into BF16 parameter storage.

Both arms use the same system message: `Please reason step by step, and put your final answer within \boxed{}.` The implementation uses native Trainer rather than the tutorial's LLaMA-Factory execution path. Dataset exports retain the requested instruction/input/output/system mapping. Deterministic sampling and consumption receipts are intended to prove one presentation of every selected row, with no padding presentations or silent row removal.

**Equal examples and updates do not imply equal token exposure or compute.** The random cache contains **85,547,725 supervised training tokens**; the advanced cache contains **63,636,440**. The random arm therefore has approximately **1.34 times** the supervised token exposure. The advanced originals contain 65,018,386 serialized tokens including prompt/template tokens. These differences are part of the observed selection policy and must be considered alongside any eventual accuracy difference.

## 4. Evaluation protocol

The original and both final models use the supplied rule-based grader and one declared generation protocol: eight samples per question, temperature `0.6`, top-p `0.95`, seed `0`, top-k disabled (`-1`), minimum probability `0`, and repetition penalty `1.0`. The maximum requested output is 32,768 new tokens, within a native **32,768-token total context**; the prompt consumes some of that window, so the output limit is an upper bound. Evaluation loads BF16 weights. Each run must bind its actual checkpoint, and benchmark-data, grader, evaluator, package, hardware, and generation settings must agree before reporting a matched comparison.

For each benchmark, **pass@1** is the average fraction of the eight recorded samples graded correct, averaged over problem records. **Pass@8** is the fraction of records with at least one correct sample. It is not a majority-vote score. The supplied-eight macro gives each benchmark equal weight, rather than pooling all questions. The supplementary seven-task internal macro uses AIME25, CN_MATH_2024, Kaoyan, AMC, Minerva, OlympiadBench, and GPQA. It is kept separate from the supplied-eight macro and is not a verified historical leaderboard rank.

MATH has **500 actual JSON records**. The official file and local copy are byte-identical, with SHA-256 `a239272930e5202bf1d5275f7fa475a56687c775a61f1354a6d3371bf9dd7a94`. Its 499 newline characters reflect a missing final trailing newline and explain the assignment README's stale 499 count; evaluation retains all 500 records. Repeated benchmark questions also retain their distinct record identities.

The saved random checkpoint emitted a Mistral-regex warning from Transformers 4.57.3's local-model detection heuristic. A [read-only audit inside the evaluation allocation](operations/tokenizer-equivalence-random.json) confirmed identical complete tokenizer backends, vocabulary, special tokens, chat templates, and rendered/tokenized inputs for all 1,974 benchmark records, plus ten diagnostic encode/decode cases. The original configuration records Transformers 4.43.1 and the newly saved configuration records 4.57.3, which triggers the heuristic differently. The [pinned implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/tokenization_utils_base.py) only replaces the regex when its fix flag is true; no flag was changed here. Evaluation continued with the frozen configuration.

The completed [native uncertainty analysis](report/primary-final-20261020T1708Z/uncertainty.json) uses 10,000 paired percentile bootstrap draws, seed `20261019`, of normalized-question clusters. Duplicate questions travel together; each draw retains record weights within a benchmark, and macros average independent within-benchmark draws equally. Kaoyan retains 199 records in 195 normalized-question clusters. The 95% intervals condition on the fixed trained models and eight recorded responses. They exclude training-seed variation and uncertainty from additional generation samples, and no multiple-comparison adjustment is applied. These limits apply to every interval reported below.

## 5. Complete primary performance results

The [native final audit](report/primary-final-20261020T1708Z/audit.json), produced by CPU closeout `1208135`, verifies all nine benchmarks for original, random and advanced from raw eight-sample predictions. All per-run and comparison errors are empty; generation settings, grader/evaluator, software, benchmark data, ordered record identities and hardware match. Each model has 1,974 questions and 15,792 recorded responses across all nine tasks. The supplied eight contain 1,944 questions and 15,552 responses per model. The [snapshot receipt](report/primary-final-20261020T1708Z/snapshot.json) binds the exact native audit, uncertainty and result-table files. Earlier immutable partial audits remain historical evidence.

Values below are percentages. All macro values use unrounded benchmark scores.

| Benchmark | Problems | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
| --- | ---: | ---: | ---: | ---: |
| AIME 2024 | 30 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% |
| AMC 2023 | 40 | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% |
| MATH | 500 | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% |
| CN_MATH_2024 | 30 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% |
| Kaoyan | 199 | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% |
| Minerva | 272 | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% |
| OlympiadBench | 675 | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% |
| GPQA | 198 | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% |
| AIME25, supplementary | 30 | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% |

| Complete benchmark set | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
| --- | ---: | ---: | ---: |
| Supplied eight, equal-benchmark macro | 29.30267835% / 52.86146036% | 16.23217035% / 36.69876417% | 18.78695793% / 42.95429078% |
| Supplemented seven, internal macro | 23.06258478% / 45.58452613% | 11.80462326% / 32.67477809% | 14.03390430% / 37.91918947% |

AIME25 is supplementary and excluded from the supplied-eight macro. The seven-task figure is an internal aggregate, not a verified historical leaderboard rank.

The preserved uncertainty file also records generation diagnostics. Length finishes count responses ending at the output/context limit, not necessarily incorrect responses.

| Task | Responses per role | Original length finishes | Random length finishes | Advanced length finishes |
| --- | ---: | ---: | ---: | ---: |
| AMC | 320 | 3 | 190 | 169 |
| AIME24 | 240 | 10 | 204 | 192 |
| MATH | 4,000 | 19 | 1,508 | 1,235 |
| CN_MATH_2024 | 240 | 8 | 215 | 159 |
| Kaoyan | 1,592 | 3 | 1,009 | 667 |
| Minerva | 2,176 | 5 | 1,269 | 1,088 |
| OlympiadBench | 5,400 | 99 | 3,526 | 3,062 |
| GPQA | 1,584 | 5 | 1,089 | 954 |
| AIME25, supplementary | 240 | 0 | 203 | 185 |

| Pooled supplied-eight generation diagnostic | Original | Random | Advanced |
| --- | ---: | ---: | ---: |
| Recorded responses | 15,552 | 15,552 | 15,552 |
| Length finishes | 152 | 9,010 | 7,526 |
| Length-finish rate | 0.98% | 57.93% | 48.39% |
| Mean generated tokens per response | 1,084.50 | 19,998.02 | 17,324.52 |

These diagnostics pool recorded responses and therefore have different weighting from the equal-benchmark accuracy macros. Advanced generates shorter outputs and reaches the limit less often than random, but both SFT arms remain far more verbose than original. These associations do not prove that truncation alone caused their accuracy regressions.

## 6. Comparison and interpretation

The complete primary result has two parts. **Advanced improves over the prepared-source random control**, with higher pass@1 on every supplied task and a positive supplied-eight macro difference in both metrics. Pass@8 is higher on six supplied tasks, tied on MATH and lower on AMC. **Both SFT models regress from the original on all eight supplied tasks in both metrics.** Both improve over original on supplementary AIME25, where advanced trails random. No leaderboard position or state-of-the-art result is established.

The following differences and 95% paired cluster-bootstrap intervals are in **percentage points**, copied from the native uncertainty artifact:

| Macro comparison | Pass@1 delta [95% CI] | Pass@8 delta [95% CI] |
| --- | ---: | ---: |
| Advanced − random, supplied eight | +2.5548 [1.6733, 3.4813] | +6.2555 [3.1852, 9.5150] |
| Advanced − original, supplied eight | −10.5157 [−12.1662, −8.9820] | −9.9072 [−13.4104, −6.5129] |
| Advanced − random, internal seven | +2.2293 [1.3228, 3.1623] | +5.2444 [2.2177, 8.3882] |
| Advanced − original, internal seven | −9.0287 [−10.6391, −7.5160] | −7.6653 [−11.2364, −4.0049] |

The supplied-eight advanced-minus-random intervals are entirely above zero, while advanced-minus-original intervals are entirely below zero. This supports improvement over this random control and a regression against original **conditional on these fixed runs and outputs**. It does not establish stability across training seeds or isolate a causal transfer-gain effect. The eight-sample generation realization is fixed, and multiple comparisons are unadjusted.

| Advanced − random by task | Pass@1 delta [95% CI], points | Pass@8 delta [95% CI], points |
| --- | ---: | ---: |
| AIME24 | +2.08 [−0.42, 5.42] | +10.00 [−3.33, 23.33] |
| AMC | +2.50 [−1.56, 6.56] | −2.50 [−12.50, 7.50] |
| MATH | +1.500 [−0.125, 3.075] | 0.00 [−2.80, 2.80] |
| CN_MATH_2024 | +4.17 [1.25, 7.50] | +16.67 [3.33, 30.00] |
| Kaoyan | +1.38 [−0.82, 3.57] | +6.03 [0.00, 12.25] |
| Minerva | +2.16 [0.37, 4.00] | +5.15 [0.74, 9.56] |
| OlympiadBench | +1.72 [0.69, 2.80] | +2.07 [−0.74, 5.04] |
| GPQA | +4.92 [2.84, 7.01] | +12.63 [5.05, 20.20] |
| AIME25, supplementary | −1.25 [−2.92, 0.00] | −3.33 [−10.00, 0.00] |

Many task intervals include or touch zero, despite positive point differences. For example, advanced MATH is 51.225% / 78.20%, versus random 49.725% / 78.20% and original 65.90% / 83.80%; its +1.50-point pass@1 difference has an interval crossing zero. GPQA is 14.141414% / 54.040404% for advanced versus random 9.217172% / 41.414141% and original 28.409091% / 69.696970%. Complete per-task advanced-versus-original intervals, exact values, cluster details and diagnostic availability are preserved in [uncertainty.json](report/primary-final-20261020T1708Z/uncertainty.json).

The **primary experiment's** selection method, trained checkpoints and scientific evaluation settings were unchanged in response to the measured scores. Its operational continuations addressed runtime only. Both arms received the same example count and optimizer recipe, but not equal supervised tokens or eligibility: the complete-target advanced policy avoids the 5,806 truncated random training prefixes and generates fewer unfinished outputs. The comparison tests that whole policy; it cannot attribute its gain solely to transfer scoring.

A later [accuracy-blind AMC diagnostic](operations/stopping-diagnostic-20261020-amc.json) quantifies generated repetition without inspecting answers or correctness. Across 320 responses per role, median output length is 765 tokens for the original and 32,526 for advanced. Repeated word eight-gram fraction exceeds 25% in 5 original responses and 162 advanced responses. Advanced has 169 length-limit finishes versus 3 for the original; only 6 of those 169 advanced length-limit responses contain any boxed-answer marker, which does not imply correctness. Actual EOS settings match across roles, and advanced has zero missing supervised EOS markers. These observations establish repetitive, unfinished generation on AMC; they do not prove that repetition alone caused the accuracy regression. Prior research also reports degradation from limited long-CoT supervision in models of at most 3B parameters, including Qwen2.5; that is context rather than a causal diagnosis of this run ([Through the Valley](https://arxiv.org/abs/2506.07712v2)).

Random's missing supervised EOS markers may aggravate stopping problems, but they cannot be the sole explanation because advanced has none missing. Both models learn from teacher-forced targets limited to 8K, while free generation can continue to the native 32K context limit. The combination of repetitive unfinished generation and a teacher-loss selection proxy is consistent with learning a long-thinking style without reliable free-generation control. This is a hypothesis, not a demonstrated causal effect of full-parameter SFT, model capacity, or a particular training component.

The current pair tests the **whole advanced policy**: complete-target structural filtering, reservoir construction, transfer-gain ranking, and coverage allocation. It does not isolate the causal contribution of transfer gain. In particular, complete-target selection changes response lengths and EOS survival relative to the broad random control, while the supervised-token budgets differ substantially. Separate eligible-pool random and coverage-only arms on the same reservoir, quotas, and training recipe would be needed to separate these contributions. Neither ablation has been run.

Only one final training seed is used. Small benchmarks, duplicate questions, rule-based grading, lexical-only contamination controls, and the limited source domain constrain interpretation. All-positive probe gains also show why the score should be treated as a learning proxy: reduced teacher-forced loss does not establish solution correctness, improved free generation, or causal training utility. The measured result answers the assignment comparison: advanced beats this random control under the stated conditional uncertainty, while failing to improve the original. Global novelty and state-of-the-art performance are not established.

### Separate exploratory recovery

After viewing the primary outcomes and stopping diagnostic, we fixed a separate [recovery plan](recovery/plan.json): `theta_recovery = 0.75 * theta_original + 0.25 * theta_advanced_final`. This weight interpolation is neither additional data selection nor another SFT run. It does not replace either primary checkpoint or its results. The fixed coefficient is not swept against benchmark scores, and the experiment is explicitly post-hoc.

The [CPU merge receipt](operations/recovery-merge-20261020-receipt.json) verifies 434 finite FP32 tensors and actual input/output hashes. Its [native compute checks](operations/recovery-native-qa-20261020-attempt1.json) passed 31 tests with no failures, errors or skips. A [source audit](operations/recovery-screen-preparation-20261020-receipt.json) seals 64 eligible source-heldout questions before generation and verifies their disjointness from the 83,029-row prepared pool, both 512-example probes and both 15K final training sets. The development test uses one sample, temperature 0.6, top-p 0.95, seed 20261020, at most 4,096 generated tokens and an 8,192-token total context for original, advanced and recovery. It measures agreement with extracted teacher answers, which are not independently verified truth; it is not the primary eight-sample, 32K benchmark protocol.

The completed [development screen](report/recovery-development-20261020/summary.json) passed its fixed gate. All three models used the same 64 sealed questions and single-sample 4K protocol in one B200 allocation (`1214124`, 316 seconds, completed successfully):

| Source-heldout development measure | Original | Advanced | Recovery |
| --- | ---: | ---: | ---: |
| Teacher-answer matches | 18/64 | 10/64 | 19/64 |
| Length-limit finishes | 1/64 | 49/64 | 1/64 |
| Boxed-answer completions | 63/64 | 20/64 | 63/64 |
| Total generated tokens | 52,307 | 237,076 | 52,852 |

Recovery satisfies the predeclared agreement and stopping gate. This supports restored stopping versus advanced in this development setting. The one additional teacher match versus the original is not credible evidence of superiority; teacher agreement is not verified ground-truth accuracy or benchmark improvement. The coefficient, IDs, generation settings and gate were not changed after these outputs.

After exact terminal accounting reclaimed the unused development reservation, initial benchmark job `1214775` started at **14:54:26 UTC**, with a 45-minute allocation. Its [native entry preflight](operations/recovery-initial-eval-preflight-20261020.json) and [manifest audit](operations/recovery-evaluation-manifest-audit-20261020.json) bind the same fixed model and all-nine, eight-sample, 32K primary protocol, data, ordered IDs, grader, runtime and B200 hardware. The [admission receipt](operations/recovery-benchmark-initial-admission-20261020.json) preserves that first stage, which reached its time limit retaining seven complete atomic task batches. An identical-protocol continuation, `1216562`, then completed successfully after **1,940 seconds**, finishing OlympiadBench and GPQA in the same output directory without changing weights, protocol or saved results. The [final native snapshot](report/recovery-final-20261020T1730Z/snapshot.json) records successful GPU evaluation and separate CPU publication; all four primary artifact identities remain unchanged.

The [final recovery evaluation summary](report/recovery-final-20261020T1730Z/evaluation_summary.json) verifies **all nine tasks for all four roles from matched raw eight-sample records**, including ordered question identities, data, evaluator and hardware. It confirms the previously reported primary values unchanged. Values below are pass@1 / pass@8 percentages; AIME25 remains supplementary, and each macro gives equal weight to its stated benchmark set.

| Complete benchmark comparison | Original | Random | Primary advanced | Recovery, post-hoc |
| --- | ---: | ---: | ---: | ---: |
| AMC | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% | 39.06% / 75.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% | 6.25% / 23.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% | 65.33% / 84.60% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% | 12.50% / 33.33% |
| Kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% | 21.11% / 54.27% |
| Minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% | 29.46% / 48.16% |
| OlympiadBench | 28.26% / 51.85% | 17.15% / 38.37% | 18.87% / 40.44% | 27.41% / 49.48% |
| GPQA | 28.41% / 69.70% | 9.22% / 41.41% | 14.14% / 54.04% | 30.43% / 76.26% |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% | 2.50% / 10.00% |
| **Supplied-eight macro** | **29.30267835% / 52.86146036%** | **16.23217035% / 36.69876417%** | **18.78695793% / 42.95429078%** | **28.94218107% / 55.55548699%** |
| Supplemented seven, internal macro | 23.06258478% / 45.58452613% | 11.80462326% / 32.67477809% | 14.03390430% / 37.91918947% | 23.20892122% / 49.50150894% |

The [continuation admission receipt](operations/recovery-benchmark-continuation-1-admission-20261020.json) binds the same fixed weights, unchanged protocol, three original wrapper arguments and existing output directory, with `afterany:1214775`. It started at **12:51:19 p.m. Eastern / 16:51:19 UTC** and retained all seven initially validated tasks. No new coefficient, method or decoding setting was introduced. The internal seven-task macro is not a verified historical leaderboard result.

Recovery's supplied-eight macro is **0.3605 percentage points lower in pass@1** and **2.6940 points higher in pass@8** than original. It therefore does not dominate original. On MATH and Minerva, it has lower pass@1 and higher pass@8; CN and GPQA exceed original in both recorded metrics, and Kaoyan ties original pass@1 with higher pass@8. OlympiadBench remains below original in both metrics. Supplementary AIME25 remains mixed against the trained controls. Recovery exceeds both SFT arms on all eight supplied tasks in both metrics, but that is a post-hoc candidate result, not a new causal test of data selection or evidence of general superiority or SOTA.

Recovery GPQA is **30.42929293% / 76.26262626%**, with 9 length-limit finishes. The complete native summary provides these per-task counts:

| Length-limit finishes | Recorded responses | Original | Random | Primary advanced | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMC | 320 | 3 | 190 | 169 | 4 |
| AIME24 | 240 | 10 | 204 | 192 | 19 |
| MATH | 4,000 | 19 | 1,508 | 1,235 | 27 |
| CN_MATH_2024 | 240 | 8 | 215 | 159 | 6 |
| Kaoyan | 1,592 | 3 | 1,009 | 667 | 1 |
| Minerva | 2,176 | 5 | 1,269 | 1,088 | 9 |
| OlympiadBench | 5,400 | 99 | 3,526 | 3,062 | 122 |
| GPQA | 1,584 | 5 | 1,089 | 954 | 9 |
| **Supplied eight, pooled** | **15,552** | **152** | **9,010** | **7,526** | **197** |
| AIME25, supplementary | 240 | 0 | 203 | 185 | 3 |
| All nine, pooled | 15,792 | 152 | 9,213 | 7,711 | 200 |

Recovery greatly reduces length-limit finishes relative to both SFT arms while remaining slightly above original on the pooled supplied tasks. Mean recorded output length is **1,242.21 tokens** for recovery versus **1,084.50** for original on the supplied tasks. These pooled generation diagnostics have different weighting from the equal-benchmark accuracy macros and do not establish a causal explanation.

The completed [recovery uncertainty summary](report/recovery-final-20261020T1730Z/uncertainty.json) uses the unchanged percentile paired normalized-question-cluster bootstrap: 10,000 resamples, seed `20261019`, duplicate-question records moving together, independent draws within benchmarks and equal benchmark macro weights. CPU-only job `1217465` completed successfully in **27 seconds**, changing no primary outputs, weights or generation settings. The [exact native output](report/recovery-final-20261020T1730Z/uncertainty-native-candidate-alias.json) reuses the analyzer's field `advanced` as an **alias for recovery only**; the labeled summary maps it back to recovery. The primary advanced checkpoint and its previously reported uncertainty remain unchanged.

| Recovery comparison | Pass@1 difference, points [95% CI] | Pass@8 difference, points [95% CI] |
| --- | ---: | ---: |
| Recovery − random, supplied eight | +12.7100 [11.1310, 14.3782] | +18.8567 [15.1679, 22.6705] |
| Recovery − original, supplied eight | −0.3605 [−1.2973, 0.6356] | +2.6940 [−0.1363, 5.6131] |
| Recovery − original, supplementary internal seven | +0.1463 [−0.8880, 1.2617] | +3.9170 [0.7216, 7.2807] |

**Both supplied-eight recovery-minus-original intervals include zero**, so this conditional analysis establishes no clear difference from original on either macro metric. The supplementary internal-seven pass@8 interval is above zero, while its pass@1 interval includes zero; that benchmark set is separate from the assignment's supplied eight and has no verified historical-leaderboard equivalence. These intervals condition on the fixed models and eight recorded outputs, exclude training-seed and additional generation-sampling variance, and have no multiple-comparison adjustment. The recovery was designed after primary observations, with one coefficient fixed before new generation and no coefficient sweep. Neither the intervals nor the point estimates establish general superiority, novelty or SOTA.

The separate recovery publisher passed **23 native CPU tests**, with no failures, errors or skips, plus wrapper syntax checks. Its [native QA receipt](operations/optional-publication-native-qa-20261020-attempt1.json) binds the unchanged source inventory. After all nine raw recovery tasks and primary publication were verified, CPU job `1217227` completed in **120 seconds**, producing the [successful publication receipt](report/recovery-final-20261020T1730Z/publication_complete.json) at **17:28:51 UTC**. The [separate recovery repository](https://huggingface.co/adversarialbandit/qwen25-3b-base75-advanced25-posthoc-20261020) is anonymously verified at [commit `51814ba26ab0dc052447aee1612bebcffdba43f2`](https://huggingface.co/adversarialbandit/qwen25-3b-base75-advanced25-posthoc-20261020/tree/51814ba26ab0dc052447aee1612bebcffdba43f2). All six FP32 weight shards and model/tokenizer sidecars preserve their evaluated bytes, with no conversion or additional SFT. The card declares the fixed 75% original / 25% advanced lineage, post-hoc timing, and absence of a coefficient sweep or superiority/SOTA claim. The four primary repositories and their dependencies were not changed.

Publication preserves the original recovery NOTICE verbatim as `provenance/merge-NOTICE.original`, while the published root NOTICE prepends the required Qwen Research License attribution sentence to that same text. Its [publication plan](report/recovery-final-20261020T1730Z/publication_plan.json) declares original NOTICE SHA-256 `13ba3e070a8a64020760ed8cf39b30fe20ec298cf7a0ee2dbba2a26d47082449` and published NOTICE SHA-256 `fd21b38cf5c1021fbfb347a0f482383761822f6ed6f6f3fb6e05e4799d00c77e`. This attribution augmentation changes neither weights nor the evaluated model directory. The [published model card](report/recovery-final-20261020T1730Z/README.model.md), [NOTICE](report/recovery-final-20261020T1730Z/NOTICE.published) and [modification statement](report/recovery-final-20261020T1730Z/MODIFICATIONS.md) are preserved locally with the native publication snapshot.

## 7. Provenance and delivery status

| Artifact identity | Revision or SHA-256 |
| --- | --- |
| Base model revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Class subset revision | `ce815bd726c0c4ffdc0912789754646910188195` |
| Source `data/acereason11_100k.json` | `f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8` |
| Frozen experiment plan at this snapshot | `080488bb494c52ca71d385b2f041821e8c71c2c994aa6439c1c6f24d8227635d` |
| Completed selection manifest content hash | `6576564357f9d708c3b8263414490df1b340035e66e892193a13a7f762a24d56` |
| Advanced original dataset export | `511fbcc1f947c93d7c62f02497265332c4bab57bb228279bc5f9c41467f04199` |
| Random original dataset export | `3709a2ece0468d5346130dc4e6075610fb11234cd5f407c2bf26a64d791abec7` |
| Original metrics JSON at this snapshot | `d9886e3c2df1892965eebd2869ada22520cb07bf181e94c1822f02d53fc12a5d` |
| Final primary native audit | `4021099030f13a46de912752a040a020fb66acdba2c1ef9f5f05303fbb7800b5` |
| Final primary native uncertainty | `e16b3b8c63cdbeddfcd1a41995e53278442df92f7bcf642edaf5d34463003823` |
| Final primary native result table | `4244c5b4dc92d2a1d6452c06dc40219fd44751d0dacc24d6d90aec1ca07f4928` |
| Primary Hugging Face completion receipt | `259623a64c1715516607c783303d10676dd84c2bd6e31851906b935c9f206d73` |
| Recovery OlympiadBench four-role raw audit | `e38bf08b0d85c4b24859c4098f1403e6687dda6d9322632835f7a03ea9255818` |
| Optional recovery publisher native QA receipt | `e9da20d04b102ecec3f9986c66a60e289adb4a9941aa8512df2091bc3d96a67e` |
| Complete four-role recovery evaluation summary | `027180fcc55446a57b6186f70e6ba6fb4fe984ad8512c0003e0aa21addaa4c7d` |
| Separate recovery publication completion receipt | `7b905af5c4408a00890fb6434a0aac9b8145ead3477fe1bfc18891d36a7de8d9` |
| Labeled recovery uncertainty summary | `04af178f22243faab992bec74e98e89c41cb31b81079b0cfb6a92f4f1301a633` |
| Exact native recovery uncertainty, candidate alias | `fe7fc3930076391d4bb1618fe1900b4909ce2c7b8cc5d7cee92b7c3cb336d053` |
| Final terminal GPU reconciliation | `323dc7cc3bad52b763e1f4bfb47b09340b501c884fc844e6cb222cea0180b8d4` |

The [primary publication receipt](report/primary-hf-publication-20261020/publication_complete.json) records completion at **17:15:11 UTC**. All four repositories below were independently verified through anonymous public access at their exact commits, including remote LFS SHA-256/size or Git blob/size checks for every file. The serialized evaluated FP32 model weights and original dataset exports were published without conversion or substitution. The [publication snapshot](report/primary-hf-publication-20261020/snapshot.json) preserves the receipt and native execution evidence.

| Primary artifact | Public repository | Verified commit |
| --- | --- | --- |
| Random 15K model | [qwen25-3b-reasoning-random15k-20261019](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-random15k-20261019) | [860dd82d4aa28b650555d6b4787b44711e4d0326](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-random15k-20261019/tree/860dd82d4aa28b650555d6b4787b44711e4d0326) |
| Advanced 15K model | [qwen25-3b-reasoning-transfer-coverage15k-20261019](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019) | [550bdcc89a9352335d6030d6834fcff0f94d697f](https://huggingface.co/adversarialbandit/qwen25-3b-reasoning-transfer-coverage15k-20261019/tree/550bdcc89a9352335d6030d6834fcff0f94d697f) |
| Random original dataset | [reasoning-random15k-20261019](https://huggingface.co/datasets/adversarialbandit/reasoning-random15k-20261019) | [26077e0219ac79c8518e370dd7736091200d0da3](https://huggingface.co/datasets/adversarialbandit/reasoning-random15k-20261019/tree/26077e0219ac79c8518e370dd7736091200d0da3) |
| Advanced original dataset | [reasoning-transfer-coverage15k-20261019](https://huggingface.co/datasets/adversarialbandit/reasoning-transfer-coverage15k-20261019) | [aab32e0bf630ba32db492af0b4af0b2632c32a06](https://huggingface.co/datasets/adversarialbandit/reasoning-transfer-coverage15k-20261019/tree/aab32e0bf630ba32db492af0b4af0b2632c32a06) |

The reproducibility record includes selected question IDs, source reservations, actual probe and score receipts, source/model hashes, cache masks, finite-update logs, exact consumption receipts, final tensor checks and evaluation manifests. The [frozen plan](experiment_plan.json), [implementation overview](README.md), and [assignment checklist](assignment_checklist.json) describe the execution contract. Heavy training and verification run in ARC compute allocations. The original plan reserved 42 GPU-hours with a fixed 48 GPU-hour consumption cap.

After VPN access was restored and native checks passed, two unchanged [evaluation continuations](operations/evaluation-runtime-contingency.md) were admitted: random `1210309`, one GPU for 16 hours, and advanced `1210310`, one GPU for 12 hours, each after its parent evaluation terminated. Both primary continuations are now **COMPLETED**, and the native final audit verifies all nine tasks for each role. Start guards required 19 and 15 hours respectively before the assignment deadline, including three hours for closeout/publication. The [admission receipt](operations/evaluation-continuation-admission.json) records CPU closeout `1208135` depending on successful completion of both continuations and publisher `1209745` depending on closeout. The [primary closeout receipt](operations/primary-closeout-complete-20261020.json) confirms native CPU job `1208135` subsequently completed successfully in 159 seconds, producing the final audit, paired uncertainty and [staged publication package](report/primary-final-20261020T1708Z/publication-package/package_manifest.json). Publisher `1209745` then completed successfully in **336 seconds of CPU allocation**, with the four public artifact commits and anonymous verification recorded above.

The [resume record](operations/evaluation-resume-20261020.json) preserves the operational history: six complete benchmark files per role were validated and retained; the interrupted, unsaved Minerva batch was regenerated in full under the frozen per-problem seeds. No partial response from the interrupted batch enters the scored results. The new 10:10 UTC raw audit establishes completion of both regenerated Minerva batches. Exact unsaved generated-token totals from the interrupted work are unknown; they are not estimated from progress counters. All interrupted allocation time remains included in GPU accounting.

| Initial evaluation parent | Terminal state | Elapsed seconds | Conservative charged GPU-seconds |
| --- | --- | ---: | ---: |
| Random `1208059` | TIMEOUT | 21,616 | 21,660 |
| Advanced `1208060` | TIMEOUT | 21,608 | 21,660 |

The [random reconciliation](operations/gpu-budget-reconciliation-d155f13d26ffca69.json) and [advanced reconciliation](operations/gpu-budget-reconciliation-637c6b2ed64206b8.json) charge each terminal allocation's GPU count times its elapsed time rounded up to a whole minute, including the small elapsed overruns beyond the requested six hours. Primary terminal charges totaled **19.05 GPU-hours** before recovery work. Adding the full **28-hour** reservation for the then-running continuations gave the historical primary-only bound of **47.05 / 48 GPU-hours (169,380 seconds)**. That was a conservative terminal-plus-open-reservation bound, **not actual elapsed GPU usage**. At that stage, admitted requests totaled **70 hours**, distinct from the **88-hour** all-attempt request history, which includes 18 hours verified never admitted. No historical request is erased; open or unverified allocations retained their full reservation until terminal accounting.

The recovery development allocation completed in 316 seconds and is conservatively charged 360 GPU-seconds (0.10 hour), replacing its original 0.75-hour reservation only after the [terminal accounting proof](operations/gpu-budget-reconciliation-e85de9ae41202753.json). This reduced the bound to 47.15 hours. The initial benchmark allocation then reached its 45-minute limit after 2,703 seconds, retaining seven complete tasks; its [terminal accounting](operations/gpu-budget-reconciliation-c0f4181a2d92af85.json) charges 2,760 seconds. At that historical stage, terminal charges plus full live primary reservations gave 47.9167 / 48 hours, and further recovery admission awaited capacity reclamation.

**Historical resource snapshot, 16:53 UTC:** advanced continuation `1210310` completed after **31,149 elapsed seconds**, conservatively charged **31,200 GPU-seconds** by its [exact terminal reconciliation](operations/gpu-budget-reconciliation-b2705ce9203eb224.json). Closed charges then totaled **102,900 seconds (28.583333 hours)**. Adding the full **16-hour random** and **0.75-hour recovery-continuation** reservations gave **163,200 seconds = 45.333333 / 48 GPU-hours**. The [continuation admission proof](operations/recovery-benchmark-continuation-1-admission-20261020.json) preserves that historical bound; it is not actual elapsed GPU usage.

**Historical resource bound, 17:11:50 UTC:** random continuation `1210309` completed after **35,613 elapsed seconds**, charged **35,640 GPU-seconds** by its [terminal reconciliation](operations/gpu-budget-reconciliation-029e44effa49a4c6.json). Closed charges then totaled **138,540 seconds (38.483333 hours)**. Adding recovery continuation `1216562`'s then-open **2,700-second** reservation gave **141,240 seconds = 39.233333 / 48 GPU-hours**. This preserved snapshot was a conservative closed-plus-open bound, not final elapsed usage.

**Final GPU accounting, 17:31 UTC:** all GPU allocations are terminal. Recovery continuation `1216562` completed after **1,940 elapsed seconds**, conservatively charged **1,980 GPU-seconds** by the [final terminal reconciliation](operations/gpu-budget-reconciliation-43fae58f06119f1c.json). The complete primary-plus-recovery experiment consumed **140,025 actual elapsed GPU-seconds = 38.895833 GPU-hours**. Per-allocation rounding up to whole minutes gives **140,520 charged GPU-seconds = 39.033333 / 48 GPU-hours**, below the unchanged cap. Interrupted work, timeouts and elapsed overruns are included. Historical admitted requests remain **72.25 hours**, and all-attempt requests remain **90.25 hours**, including 18 hours verified never admitted; no request history is erased. CPU preparation, closeout, uncertainty analysis and both publications allocate no GPU. Both primary and recovery publications are complete, with separate checkpoint identities.

**Built with Qwen.** The [3B base model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/README.md) declares `license: other`, `license_name: qwen-research`, and links to the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). The [NVIDIA AceReason-1.1-SFT card](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT#license) declares CC BY 4.0. Dataset attribution must credit NVIDIA and the actual redsgnaoh class subset separately; a separate license for that subset has not been verified or asserted here.

**Source repository:** the implementation is public on [manoj060603/reasoning-sft-llm-results, main](https://github.com/manoj060603/reasoning-sft-llm-results/tree/main). This report links the complete [primary audit](report/primary-final-20261020T1708Z/audit.json), [four-role recovery comparison](report/recovery-final-20261020T1730Z/evaluation_summary.json), separately labeled uncertainty analyses, both publication completion receipts and final terminal resource accounting. The final documentation and evidence snapshot is ready for source publication verification. The [earlier source-publication receipt](report/source_publication_receipt.json) remains historical snapshot evidence.

The mandatory technical delivery is complete: both trained-model repositories and both original-dataset repositories are public and verified, and the primary comparisons, uncertainty and report interpretation are complete. The separate post-hoc recovery checkpoint and all nine matched benchmark results are also published and verified, with its own completed uncertainty analysis and terminal GPU accounting. The staged package retains its historical `staged_metadata_not_uploaded` status because it is an immutable pre-upload artifact; the later successful publication receipts establish delivery. The final 17:41 UTC documentation and evidence snapshot is ready; manual team submission remains pending. Publication of technical artifacts does not establish course submission. The team must submit the repository, report and model/dataset links by **20 October 2026 at 11:59 p.m. Eastern**.

## 8. Separate validation-guided follow-up: negative confirmation

After the delivered primary and fixed 75/25 recovery outcomes were viewed, a separately [frozen follow-up plan](anchor_search/plan.json) tested two FP32 base anchors with advanced-weight fractions **0.125** and **0.375**. It reserved **192 fresh source-search questions and 192 disjoint confirmation questions**, excluding the earlier 64-question screen and actual training/probe data. This was an exploratory source-validation search, with fixed scoring, quality limits and tie-breaks before new generation. It changes none of the preceding models, benchmark outputs, confidence intervals or public artifacts.

The [search selection receipt](report/anchor-v2-final-20261020/selection.json) selected **alpha0125**: source teacher-agreement pass@1 **0.30078125**, pass@8 **0.53125**, and harmonic score **0.3840962441**. The other candidate, **alpha0375**, failed the search quality limits: **56 length finishes exceeded 53**, and mean repeated eight-gram fraction **0.0649861761 exceeded 0.0648796302**. Search selection was not confirmation of an improvement.

| Independent source confirmation, 192 questions | Teacher-agreement pass@1 | Teacher-agreement pass@8 |
| --- | ---: | ---: |
| Original | 25.78125% | 52.08333% |
| Incumbent, 75% original / 25% advanced | 28.38542% | 52.60417% |
| Selected candidate, 87.5% original / 12.5% advanced | 25.97656% | 48.43750% |

The selected candidate failed independent confirmation. Its harmonic score **0.3381725722** was below incumbent's **0.3687366024**. Its pass@1 **0.259765625** missed the incumbent-based floor **0.2738541667**; pass@8 **0.484375** missed both the original-based floor **0.5008333333** and incumbent-based floor **0.5060416667**. Confirmation length/repetition quality gates passed, but the fixed agreement criteria did not. The [final decision](report/anchor-v2-final-20261020/decision.json) is **complete=true, accepted=false**.

The delivered 75/25 incumbent is retained. **No new candidate benchmark evaluation or Hugging Face checkpoint publication occurred, and no alternate candidate was retried after failed confirmation.** Teacher agreement is a source-development proxy, not verified ground truth or assignment benchmark accuracy. The discrepancy between search and confirmation is a negative result under the fixed procedure; it establishes no further improvement. The [follow-up closeout](report/anchor-v2-final-20261020/closeout.json) preserves the separate terminal decision and resource accounting. Earlier GPU totals in Section 7 are the dated completed primary/fixed-recovery snapshot.

Source job `1219074` ended **COMPLETED, exit 0:0 at 20:29:26 UTC**, after **2,163 seconds on one B200**, conservatively charged **2,220 GPU-seconds**. Adding it to the earlier experiment gives **142,188 actual elapsed GPU-seconds = 39.496667 hours**, and **142,740 minute-rounded charged GPU-seconds = 39.65 / 48 hours**. The [terminal accounting proof](operations/gpu-budget-reconciliation-be6bf2aca0eee450.json) preserves all historical requests: **73.75 admitted requested hours** and **91.75 all-attempt requested hours**, including the earlier 18 hours verified never admitted. These requests are not actual usage. No new benchmark allocation was requested. The [native source evidence archive](report/anchor-v2-final-20261020/native-source-evidence.tar.gz) and its [SHA-256 inventory](report/anchor-v2-final-20261020/source-evidence-inventory.json) retain the prepared receipt, bounded allocation seal, selection, decision and all seven source-role manifests, observations and completion receipts. Raw observations contain agreement and diagnostic flags and output hashes, not generated text. All five earlier public artifacts and all preceding benchmark results remain unchanged. The prospective goal of exceeding original on both benchmark metrics was **not demonstrated**; no additional experiment is pending.

## Final reconciliation before submission

- Retain both completed training receipts, checkpoint hashes, exact consumption, timing, memory and runtime evidence; check that the final published artifacts match them.
- Reconcile the random cache hash, original export hash, missing-supervised-EOS count, and both arms' token/row receipts with the final artifacts.
- Retain the complete matched primary audit, both macro definitions, paired uncertainty and generation diagnostics; add any recovery result only through its separate audit and identity.
- Preserve the measured conclusion: advanced improves over this random control while both SFT models regress from original; retain the broad-source control, token-budget, one-seed and missing-ablation limitations.
- Verify publication links, cards, data/model hashes, final source commit, team details, README consistency, and the course submission receipt; update this snapshot's status and date.
