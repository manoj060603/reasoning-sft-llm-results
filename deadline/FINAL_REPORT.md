# Draft report: Cross-fitted Transfer Gain with Coverage for Reasoning SFT

**Snapshot: 20 October 2026, 11:24 a.m. Eastern / 15:24 UTC.** Both final full-SFT runs are complete: each performed 1,875 updates, used all 15,000 unique examples exactly once, and saved 434 verified finite FP32 tensors. The original-model evaluation is complete on the supplied eight benchmarks and supplementary AIME25. Both trained models have **8/9 completed tasks**: AMC, AIME24, MATH, CN_MATH_2024, Kaoyan, Minerva, OlympiadBench and supplementary AIME25. Their supplied-eight coverage is **7/8 each**. Only GPQA, complete macros, paired uncertainty, final Hugging Face links and course submission remain pending. Both unchanged evaluation continuations are running; final closeout and publication remain dependency-gated. Pending entries are not zero scores.

We investigate whether a source-only selection policy can improve reasoning SFT over a random control. Two small LoRA training runs measure how examples in the opposite source fold respond to adaptation. The final selector combines relative response-loss reduction with structural eligibility and topic/length coverage. Both final models start from the original Qwen2.5-3B-Instruct checkpoint. Previous team checkpoints and scores are not results of this experiment.

The original model achieves an equal-benchmark macro of **29.30267835% pass@1 and 52.86146036% pass@8** across the supplied eight tasks. These establish the fresh baseline; they do not establish an improvement from fine-tuning. The [assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM#8-submission) calls for original and fine-tuned performance, comparison and analysis, final hyperparameters, and the data-selection method. Each is covered below, with incomplete results explicitly marked.

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

When all three raw evaluations are complete and matched, uncertainty analysis will use 10,000 paired bootstrap draws of normalized-question clusters. Duplicate questions travel together; each draw retains record weights within a benchmark, and macros average independent within-benchmark draws equally. These percentile intervals condition on the fixed trained models and eight recorded responses. They do not estimate training-seed variation or uncertainty from new generation samples, and no multiple-comparison adjustment is applied. Partial or aggregate-only evidence cannot produce a complete uncertainty macro.

## 5. Current performance results

The [10:10 UTC partial audit](report/partial-20261020T1010Z/audit.json) verifies raw eight-sample predictions for all nine original-model tasks and seven tasks for each trained model, with no per-run audit errors. Completed records match across roles in protocol, benchmark data and hardware. The [15:24 UTC native audit](report/recovery-benchmark-partial-20261020T1524Z/additional-audit.json) additionally verifies OlympiadBench for all three primary roles. Both trained models still lack GPQA; the full comparison remains incomplete. The retained [original metrics](report/original_metrics.json) and [evaluation manifest](report/original_evaluation_manifest.json) preserve the complete baseline evidence. Values below are percentages; truncations count samples ending at the output/context limit, out of eight samples per problem.

| Benchmark | Problems | Original pass@1 | Original pass@8 | Original truncated samples | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| AIME 2024 | 30 | 7.92% | 23.33% | 10 | 0.42% / 3.33% | 2.50% / 13.33% |
| AMC 2023 | 40 | 42.50% | 70.00% | 3 | 22.50% / 47.50% | 25.00% / 45.00% |
| MATH | 500 | 65.90% | 83.80% | 19 | 49.73% / 78.20% | 51.23% / 78.20% |
| CN_MATH_2024 | 30 | 10.00% | 26.67% | 8 | 0.83% / 6.67% | 5.00% / 23.33% |
| Kaoyan | 199 | 21.11% | 49.75% | 3 | 14.07% / 41.71% | 15.45% / 47.74% |
| Minerva | 272 | 30.33% | 47.79% | 5 | 15.95% / 36.40% | 18.11% / 41.54% |
| OlympiadBench | 675 | 28.26% | 51.85% | 99 | 17.15% / 38.37% | 18.87% / 40.44% |
| GPQA | 198 | 28.41% | 69.70% | 5 | Pending | Pending |
| AIME25, supplementary | 30 | 0.83% | 3.33% | 0 | 2.92% / 16.67% | 1.67% / 13.33% |

| Complete benchmark set | Original pass@1 | Original pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
| --- | ---: | ---: | --- | --- |
| Supplied eight, equal-benchmark macro | 29.30267835% | 52.86146036% | Pending | Pending |
| Supplemented seven, internal macro | 23.06258478% | 45.58452613% | Pending | Pending |

Macros use unrounded benchmark values. AIME25 is supplementary and excluded from the supplied-eight macro. The audit correctly marks the full comparison **unmatched** because GPQA remains incomplete for both trained models. No trained-model full macro, paired confidence interval, overall winner or leaderboard position is available.

The completed tasks also show frequent output/context-limit termination:

| Task | Responses per role | Random truncated responses | Advanced truncated responses |
| --- | ---: | ---: | ---: |
| AMC | 320 | 190 | 169 |
| AIME24 | 240 | 204 | 192 |
| MATH | 4,000 | 1,508 (37.7%) | 1,235 (30.875%) |
| CN_MATH_2024 | 240 | 215 (89.5833%) | 159 (66.25%) |
| Kaoyan | 1,592 | 1,009 | 667 |
| Minerva | 2,176 | 1,269 | 1,088 (50%) |
| OlympiadBench | 5,400 | 3,526 | 3,062 |
| AIME25, supplementary | 240 | 203 | 185 |

These are raw finish-reason diagnostics, not a quantitative repetition metric or proof that truncation alone caused the accuracy differences.

## 6. Comparison and interpretation

The partial findings are **mixed**. Both trained models perform worse than the original on all seven completed supplied tasks: AMC, AIME24, MATH, CN_MATH_2024, Kaoyan, Minerva and OlympiadBench. Both perform better than the original on supplementary AIME25. Advanced improves over random in both metrics on AIME24, CN_MATH_2024, Kaoyan, Minerva and OlympiadBench, has higher pass@1 but lower pass@8 on AMC, and trails random on AIME25.

On the 500-problem MATH task, advanced pass@1 is **51.225%**, versus **49.725%** for random: **+1.50 percentage points**. Their pass@8 scores tie at **78.20%**. Both are below the original's **65.90% pass@1 / 83.80% pass@8**.

For CN_MATH_2024, advanced scores **5.00% / 23.3333%**, versus random **0.8333% / 6.6667%**, while the original scores **10.00% / 26.6667%** (pass@1 / pass@8). Kaoyan advanced scores **15.4523% / 47.7387%**, versus random **14.0704% / 41.7085%**, below the original's **21.1055% / 49.7487%**. The corresponding raw truncation counts are 159/240 advanced versus 215/240 random for CN and 667/1,592 versus 1,009/1,592 for Kaoyan. These are measured per-task differences, not a complete macro or evidence that the selection method is generally superior. Paired confidence intervals have not been computed, and significance is not established. AMC has 40 problems, each AIME/CN task has 30, and Kaoyan has 199.

On Minerva, advanced scores **18.1066% pass@1 / 41.5441% pass@8**, versus random **15.9467% / 36.3971%** and original **30.3309% / 47.7941%**. Advanced exceeds random in both measured metrics, while both remain below the original. Advanced has 1,088/2,176 responses truncated (50%), versus 1,269/2,176 for random. This is another per-task observation; complete macros, paired intervals and a general superiority claim remain unavailable.

The final comparison will report each SFT arm against the original, then advanced against random on every task and on the complete supplied-eight macro. Pass@1 and pass@8 should be considered together with output lengths and truncation. The **primary experiment's** selection method, trained checkpoints and scientific evaluation settings are unchanged in response to these partial scores. Its operational continuations address runtime only.

A later [accuracy-blind AMC diagnostic](operations/stopping-diagnostic-20261020-amc.json) quantifies generated repetition without inspecting answers or correctness. Across 320 responses per role, median output length is 765 tokens for the original and 32,526 for advanced. Repeated word eight-gram fraction exceeds 25% in 5 original responses and 162 advanced responses. Advanced has 169 length-limit finishes versus 3 for the original; only 6 of those 169 advanced length-limit responses contain any boxed-answer marker, which does not imply correctness. Actual EOS settings match across roles, and advanced has zero missing supervised EOS markers. These observations establish repetitive, unfinished generation on AMC; they do not prove that repetition alone caused the accuracy regression. Prior research also reports degradation from limited long-CoT supervision in models of at most 3B parameters, including Qwen2.5; that is context rather than a causal diagnosis of this run ([Through the Valley](https://arxiv.org/abs/2506.07712v2)).

The current pair tests the **whole advanced policy**: complete-target structural filtering, reservoir construction, transfer-gain ranking, and coverage allocation. It does not isolate the causal contribution of transfer gain. In particular, complete-target selection changes response lengths and EOS survival relative to the broad random control, while the supervised-token budgets differ substantially. Separate eligible-pool random and coverage-only arms on the same reservoir, quotas, and training recipe would be needed to separate these contributions. Neither ablation has been run.

Only one final training seed is used. Small benchmarks, duplicate questions, rule-based grading, lexical-only contamination controls, and the limited source domain constrain interpretation. All-positive probe gains also show why the score should be treated as a learning proxy: reduced teacher-forced loss does not establish solution correctness, improved free generation, or causal training utility. A negative or mixed result would still answer the experiment; superiority, statistical significance, novelty, and state-of-the-art performance are not assumed.

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

After exact terminal accounting reclaimed the unused development reservation, initial benchmark job `1214775` started at **14:54:26 UTC**, with a 45-minute allocation scheduled to reach its time limit at 15:39:26 UTC. Its [native entry preflight](operations/recovery-initial-eval-preflight-20261020.json) and [manifest audit](operations/recovery-evaluation-manifest-audit-20261020.json) bind the same fixed model and all-nine, eight-sample, 32K primary protocol, data, ordered IDs, grader, runtime and B200 hardware. The [admission receipt](operations/recovery-benchmark-initial-admission-20261020.json) establishes this bounded first stage, not completion of the benchmark suite. A later [15:02 UTC native partial audit](report/recovery-benchmark-partial-20261020T1502Z/audit.json) verifies the first recovery benchmark: AMC (40 problems, eight samples each). Recovery scores **39.0625% pass@1 / 75.00% pass@8**, versus original **42.50% / 70.00%**, random **22.50% / 47.50%**, and advanced **25.00% / 45.00%**. Recovery has 4 length-limit finishes out of 320 responses, versus advanced's 169 and original's 3. It improves over both trained arms on this task; the original retains higher pass@1 while recovery has higher pass@8. This one-task comparison establishes neither overall superiority nor significance. Full recovery macros and paired uncertainty remain pending. If it stops before all benchmarks finish, only completed atomic batches can be retained; a later identical-protocol continuation requires new verified GPU capacity and deadline reserves. The primary publisher still binds the original random/advanced trained checkpoints; any recovery publication requires its own identity and provenance.

The recovery snapshot now covers **seven of nine tasks**, using the [five-task native audit](report/recovery-benchmark-partial-20261020T1521Z/audit.json), [additional Kaoyan audit](report/recovery-benchmark-partial-20261020T1524Z/additional-audit.json), and [15:36 UTC Minerva audit](report/recovery-benchmark-partial-20261020T1535Z/minerva-audit.json). Values below are pass@1 / pass@8 percentages; OlympiadBench and GPQA remain outside this audited recovery snapshot.

| Recovery benchmark snapshot | Original | Random | Primary advanced | Recovery |
| --- | ---: | ---: | ---: | ---: |
| AMC | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% | 39.06% / 75.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% | 6.25% / 23.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% | 65.33% / 84.60% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% | 12.50% / 33.33% |
| Kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% | 21.11% / 54.27% |
| Minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% | 29.46% / 48.16% |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% | 2.50% / 10.00% |

Recovery approaches the original on MATH and Minerva, with slightly lower pass@1 and higher pass@8; CN exceeds the original in both recorded metrics, and Kaoyan ties original pass@1 with higher pass@8. Supplementary AIME25 remains mixed against the trained controls. These are measured partial-task differences, without full-suite macros or paired confidence intervals; general superiority and significance remain unestablished.

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

The reproducibility record includes selected question IDs, source reservations, actual probe and score receipts, source/model hashes, cache masks, finite-update logs, exact consumption receipts, final tensor checks and evaluation manifests. The [frozen plan](experiment_plan.json), [implementation overview](README.md), and [assignment checklist](assignment_checklist.json) describe the execution contract. Heavy training and verification run in ARC compute allocations. The original plan reserved 42 GPU-hours with a fixed 48 GPU-hour consumption cap.

After VPN access was restored and native checks passed, two unchanged [evaluation continuations](operations/evaluation-runtime-contingency.md) were admitted: random `1210309`, one GPU for 16 hours, and advanced `1210310`, one GPU for 12 hours, each after its parent evaluation terminated. Both continuations are **running** at the current snapshot. Start guards required 19 and 15 hours respectively before the assignment deadline, including three hours for closeout/publication. The [admission receipt](operations/evaluation-continuation-admission.json) records CPU closeout `1208135` depending on successful completion of both continuations and publisher `1209745` depending on closeout. Running evaluations and queued successors do not establish final completion or publication.

The [resume record](operations/evaluation-resume-20261020.json) preserves the operational history: six complete benchmark files per role were validated and retained; the interrupted, unsaved Minerva batch was regenerated in full under the frozen per-problem seeds. No partial response from the interrupted batch enters the scored results. The new 10:10 UTC raw audit establishes completion of both regenerated Minerva batches. Exact unsaved generated-token totals from the interrupted work are unknown; they are not estimated from progress counters. All interrupted allocation time remains included in GPU accounting.

| Initial evaluation parent | Terminal state | Elapsed seconds | Conservative charged GPU-seconds |
| --- | --- | ---: | ---: |
| Random `1208059` | TIMEOUT | 21,616 | 21,660 |
| Advanced `1208060` | TIMEOUT | 21,608 | 21,660 |

The [random reconciliation](operations/gpu-budget-reconciliation-d155f13d26ffca69.json) and [advanced reconciliation](operations/gpu-budget-reconciliation-637c6b2ed64206b8.json) charge each terminal allocation's GPU count times its elapsed time rounded up to a whole minute, including the small elapsed overruns beyond the requested six hours. Verified terminal charges now total **19.05 GPU-hours**. Adding the full **28-hour** reservation for the running continuations gives **47.05 / 48 GPU-hours (169,380 seconds)**. This is a conservative terminal-plus-open-reservation bound, **not actual elapsed GPU usage**. Historical admitted requests remain **70 hours**, distinct from the **88-hour** all-attempt request history, which includes 18 hours verified never admitted. No historical request is erased; open or unverified allocations retain their full reservation. Final consumed resources remain pending terminal continuation accounting.

The recovery development allocation completed in 316 seconds and is conservatively charged 360 GPU-seconds (0.10 hour), replacing its original 0.75-hour reservation only after the [terminal accounting proof](operations/gpu-budget-reconciliation-e85de9ae41202753.json). This reduced the bound to 47.15 hours. The later 45-minute initial benchmark admission makes the current bound **47.90 / 48 GPU-hours (172,440 seconds)**, leaving six minutes of uncommitted reservation capacity. Historical admitted requests are 71.50 hours and all-attempt requests 89.50 hours. CPU preparation allocated no GPU; any benchmark continuation must pass a new capacity check. The 48-hour cap covers both primary and recovery work.

**Built with Qwen.** The [3B base model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/README.md) declares `license: other`, `license_name: qwen-research`, and links to the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). The [NVIDIA AceReason-1.1-SFT card](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT#license) declares CC BY 4.0. Dataset attribution must credit NVIDIA and the actual redsgnaoh class subset separately; a separate license for that subset has not been verified or asserted here.

**Source publication update, 20 October 2026:** the implementation is public in [manoj060603/reasoning-sft-llm-results](https://github.com/manoj060603/reasoning-sft-llm-results). The historical [45e4d38 snapshot](https://github.com/manoj060603/reasoning-sft-llm-results/commit/45e4d38784f1fdad4c033031f3b2d4c32c6c94b2) and its [receipt](report/source_publication_receipt.json) preserve earlier source evidence. A later anonymous Git fetch verified that public snapshot [3003d58](https://github.com/manoj060603/reasoning-sft-llm-results/commit/3003d5860e829135ebda53f7e1826a92927eb1f1) has the entire tree `386e161cc13632530f171c055158e0f56f27506c`, equal to local commit `62e98f15e3e6775e0502ee00079e7701a74562ea`. That snapshot includes the primary 10:10 UTC audit, separately labeled recovery code, and sealed recovery checkpoint/source-screen provenance. Source publication does not establish completed recovery generation, benchmark improvement, Hugging Face artifact publication or course submission.

Two trained-model URLs, two original-dataset URLs, final comparisons and this completed report remain pending. The publication packager will recheck completed selection, training, weights, and matched evaluation receipts before staging cards and a manifest. Staging metadata is not an upload or course submission. The final deliverables must include usable repository and model/dataset links, a current README, and this completed report. The team must submit them by **20 October 2026 at 11:59 p.m. Eastern**.

## Final reconciliation before submission

- Retain both completed training receipts, checkpoint hashes, exact consumption, timing, memory and runtime evidence; check that the final published artifacts match them.
- Reconcile the random cache hash, original export hash, missing-supervised-EOS count, and both arms' token/row receipts with the final artifacts.
- Replace every pending score only from complete matched evaluations; update per-task results, both macro definitions, paired uncertainty, and length/truncation diagnostics.
- Write the measured comparison and conclusion, including negative or mixed outcomes; retain the broad-source control, token-budget, one-seed, and missing-ablation limitations.
- Verify publication links, cards, data/model hashes, final source commit, team details, README consistency, and the course submission receipt; update this snapshot's status and date.
