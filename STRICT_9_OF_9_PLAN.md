# Plan for the strict nine-column objective

Reviewed October 16, 2026 (America/New_York). This is a research and execution plan, not a measured claim of success or a launch-ready successor configuration. It supersedes the older ranked queues where they recommend another completed method or automatically replacing original targets.

## Decision

**Concrete successor design:** [DEMONSTRATION_RESPONSE_SFT_DESIGN.md](DEMONSTRATION_RESPONSE_SFT_DESIGN.md)
specifies original-demonstration assistance measurements, a 3K/6K/6K retention/acquisition selection,
and a medium-to-currently-missed curriculum. [The detailed literature review](ARXIV_DETAILED_REVIEW_20261016.md)
records methods, ablations, appendix findings, and pinned implementation checks for nine papers
plus LIMOPro. This refines the general selection stage below; its implementation and CPU checks are recorded in that design document. Full B200 preparation job 1183868 is submitted; GPU scoring and training remain unmeasured.

Keep the evaluated model a descendant of the pinned Qwen2.5-3B-Instruct base. Preserve SkillGraph ASFT job 1180491 results and advance the implemented successor as authorized by the user; never overlap B200 work. Prepare one new intervention around **measured student solvability and original-target quality**, rather than another loss, merge, or unverified rewrite. A teacher may supply preparation evidence; it is not the submitted model. The primary path selects original AceReason examples and retains their original complete responses.

Success remains one fixed final checkpoint clearing all nine previously frozen solve-count gates. A macro improvement or a subset of wins is a failed candidate. This means exceeding the nine columns of the prespecified first-place Fall 2025 row, not silently changing the target to the best value from every historical team. Exact historical equivalence is still unverified.

## What the assignment actually prescribes

The [official README](https://github.com/reds-lab/Project-Reasoning-SFT-LLM) centers the project on Qwen2.5-3B-Instruct, SFT, a random 15K baseline, an advanced 15K selection, comparison with base, and publication of the model, both datasets, code, and report. It explicitly permits additional dataset sources. Its example uses a 16,384-token cutoff, learning rate 5e-6, and three epochs; those values are examples, not evidence of the historical winner's recipe.

The session adds stricter requirements: exactly 15,000 unique sealed prompt identities, no public-benchmark-driven selection or tuning, no GPU pilots/development gates/checkpoint selection, sequential Owl B200 jobs only, complete nine-benchmark evaluation, and strict clearance of every column. The current experiment's 8K cutoff and 1,875 updates are frozen for that experiment; they are not assignment-wide rules. Any successor must state its schedule before launch and cannot silently inherit or change those values.

The README does not explicitly establish permission to replace the supplied solutions with newly generated responses. My earlier recommendation presented that extension too confidently. The primary next design therefore uses generated answers only to assess original training examples. On-policy replacement targets and multi-response sets remain conditional extensions whose assignment fit must be established before launch. No instructor contact is authorized by this document.

## Measured size of the problem

The table below comes from the preserved matched TIES comparison, the strongest completed seven-benchmark macro at this review. It is one model, not a per-benchmark mixture of checkpoints.

| Benchmark | Matched base solves | TIES solves | Frozen strict target | Additional TIES solves needed |
|---|---:|---:|---:|---:|
| AIME24 | 7/30 | 5/30 | 8/30 | 3 |
| AIME25 | 4/30 | 5/30 | 8/30 | 3 |
| MATH | 426/500 | 420/500 | 451/500 | 31 |
| CN Math 2024 | 7/30 | 10/30 | 14/30 | 4 |
| Kaoyan | 113/199 | 116/199 | 138/199 | 22 |
| AMC | 31/40 | 30/40 | 33/40 | 3 |
| Minerva | 135/272 | 134/272 | 95/272 | 0 |
| OlympiadBench | 334/675 | 343/675 | 385/675 | 42 |
| GPQA | 152/198 | 154/198 | 121/198 | 0 |

The strict integer targets are frozen conservative project gates derived from a rounded historical table. They are not recovered historical prediction receipts. Do not adjust them to make a candidate pass.

SkillGraph ASFT is still evaluating. Its first three completed counts are AMC 25/40, AIME25 4/30, and AIME24 4/30, already excluding strict success for that checkpoint. The user subsequently authorized moving on without waiting. Preserve available results and distinguish any cancelled evaluation tail from an operational failure; never relaunch this trained candidate unchanged.

## Why the routes have not solved it

1. **Small macro gains have concealed regressions.** Ordinary full SFT scored 51.52% seven-task pass@8 versus 49.55% base, but its paired interval included zero and pass@1 regressed. TIES scored 51.59% while passing only two historical columns. Neither is evidence of a broad reasoning improvement sufficient for the target.
2. **Stopping and reasoning are separate.** Compact targets reduced truncations from 141 to 44 but lowered the macro to 44.83%. Correct termination alone cannot establish the missing MATH/OlympiadBench solves.
3. **Proxy selection has not established learnability.** Base teacher-forced NLL is not empirical solve probability. Skill names and coverage counts are not demonstrations that the student can learn the selected derivations. Model-aware GEM scored 48.32% despite complete NLL scoring.
4. **Generated targets already have negative evidence.** The completed 2K Gen-SSD confirmation was below base and overgenerated. A stronger teacher is therefore not a sufficient new hypothesis by itself. The SkillGraph 47/256 final-answer agreement audit also cannot certify source errors: teacher failure, incomplete generations, extraction/verification failures, and genuine disagreement must be separated.
5. **Training exposure and serialization remain possible confounders.** The completed ordinary full-SFT run used a 4K cutoff; current loss experiments use complete 8K-safe examples. Finite loss and 1,875 updates do not prove the same final-answer/EOS supervision or equivalent exposure. There is no basis to label every weak result a setup bug, or to claim the setup has been fully ruled out.
6. **Multiple variables changed together.** SkillGraph changes selection and loss simultaneously. Its terminal result will measure the candidate, but cannot isolate the cause. Future designs must specify the distinguishing intervention and avoid an arbitrary stack of loss modifications.

## Evidence from primary literature and its limits

| Source | Useful result or mechanism | Limit for this project |
|---|---|---|
| [Select2Reason, 2505.17266](https://arxiv.org/html/2505.17266v1) | Joint difficulty/response-length selection; argues against treating every long trace as useless | Uses Qwen2.5-Math-7B and different data/metrics; neither a 3B guarantee nor justification to truncate targets |
| [AceReason-Nemotron 1.1, 2506.13284](https://arxiv.org/html/2506.13284v1) | Studies unique-prompt coverage, multiple responses, and repeated SFT exposure | Far larger data and different models; its epoch observations cannot select our checkpoint using public scores |
| [Efficient Long CoT in Small Models, 2505.18440](https://arxiv.org/abs/2505.18440) | Student-compatible curation and removing redundant reasoning | Our compact-target result shows shortening alone can regress accuracy |
| [On-policy SFT, 2602.13407](https://arxiv.org/html/2602.13407v1) | Learns from correct, complete self-generated solutions; discusses rollout temperature and length bias | Primarily improves efficiency while preserving accuracy in DeepSeek-R1 1.5B/7B; does not demonstrate the large broad gains needed here |
| [Reasoning Path Divergence, 2510.26122](https://arxiv.org/abs/2510.26122) | Diverse trajectories improve multi-sample coverage | Main Qwen3-4B setup plus a Qwen2.5-3B appendix; distinct small-data/multi-response protocol and no established assignment permission for extra targets |
| [OpenMathReasoning, 2504.16891](https://arxiv.org/abs/2504.16891) | Large verified solution collections and quality filtering underpin strong reasoning models | Millions of solutions, tools, and candidate selection are outside our unchanged evaluation interface |

The common requirement is usable, correct supervision that expands transferable skills while preserving reliable answers. A paper title or an efficiency gain does not justify forecasting 9/9. One-path selection should not be described as preserving within-prompt multi-path diversity; that claim requires actual multiple paths and a compatible training design.

## Execution sequence

### 1. Close the current candidate and audit the setup

Finish all nine predictions, check the exact 15K data hash, 1,875 finite updates, model shard index/files, tokenizer, final-answer and EOS supervision, and full manifests. Fetch non-destructively, update all required reports, and push the actual terminal result. Do not submit another GPU job until the queue confirms the current project job is terminal.

Audit the data loader on selected training records without a GPU pilot: compare the original response with decoded supervised labels; count target/EOS loss from cutoff or template handling; distinguish length-limit generation from repetition, missing boxes, unsupported answer forms, and completed wrong answers. Use synthetic answer-format fixtures for parser/grader equivalence checks. Benchmark error categories may inform the scientific report, but benchmark prompt content and error labels cannot flow into a selector, teacher prompt, quota, or training example.

A fresh metadata audit found **500 MATH and 199 Kaoyan source records both locally and on Owl**, with equal local/remote source hashes. The earlier statement that Owl had only 195 Kaoyan records was incorrect. The README lists 499 MATH; raw historical prediction/data receipts remain unavailable. The prepared MATH file has a different byte hash from the source file, so compare normalized identities and answers as a read-only protocol audit before asserting equivalence. Distinguish record counts from unique normalized prompts. Keep the current evaluator unchanged.

### 2. Prepare the next dataset using training-only solve evidence

The primary new hypothesis is **empirical-solvability selection with complete original targets**. Start from the full sealed approved pool, not public benchmark failures. Reuse the fixed skills/clusters as broad coverage controls. Pin the original 3B student for scoring rather than choose a scorer from public-suite results.

Use independent student generations on a deterministic, coverage-preserving candidate pool to estimate which examples are already mastered, partially solvable, or unsolved. Preserve per-generation finish reasons, verifier support, final-answer correctness, length, and exact prompt IDs. A length-limited wrong answer is censored evidence, not proof the problem is beyond the student. NLL remains a diagnostic rather than the difficulty label.

Freeze the candidate-pool sampling rule, rollout count/budget, seed, ranking, quotas, and fallback before GPU preparation. Prefer examples with reliable original solutions and evidence of learnable difficulty, while explicitly retaining coverage of unsolved skills and mastered skills; selecting only self-solved examples can remove the very capabilities we need to acquire. These allocation choices are design hypotheses, not published optimal constants. Choose them using training-only coverage/capacity and compute constraints, never leaderboard scores.

Export exactly 15,000 unique IDs and their **unchanged original complete responses**, with content/identity hashes, full Qwen serialization lengths, contamination receipts, and counts of mastered/frontier/unsolved examples. Verification failure is reported honestly; final-answer agreement alone does not prove every derivation step. Do not call unverified fallback targets verified. If eligible candidate coverage is inadequate, stop preparation and revise/report the design instead of silently relaxing checks or adding benchmark-derived examples.

### 3. Freeze a simple full SFT run and cost it before submission

Use ordinary response-only full SFT from the pinned base to avoid combining a new selector with another untested loss. Select a cutoff that contains every exported target and EOS; count supervised tokens and exposure rather than infer equivalence from update count. The current 8K/one-epoch schedule is the conservative reference, not a proven optimum. Any change in cutoff, learning rate, or number of epochs must have a written assignment/literature/training-data rationale and be frozen before evaluation. Evaluate only the final checkpoint. No public-score checkpoint choice or GPU pilot.

Preparation can dominate the cost: 83,037 prompts × four rollouts × 3,500 output tokens is a maximum **1,162,518,000 output tokens**, excluding prefill and overhead. At a hypothetical sustained 2,000 output tokens/s that alone is about 161.5 hours, not a short one-B200 experiment. Therefore use measured throughput from existing logs and a frozen coverage-preserving preparation pool/budget; record the approximation error. Account for the October 20, 11:59 p.m. Eastern assignment deadline, queue time, training, nine evaluations, and required uploads. Do not claim a start/completion ETA from queue availability alone. Resumable preparation allocations must be sequential, B200-only, and preserve immutable ledgers.

The next job is not launch-ready until its selection algorithm, schedule, token budget, CPU end-to-end export/data-loader checks, and provenance are implemented. This document authorizes no overlapping job and records no successor submission.

### 4. Evaluate and make the next decision

Run the unchanged sampler once on all nine datasets: temperature .6, top-p .95, eight samples, 32,768 maximum new tokens, original system prompt and rule-based grader. Require matching manifests to base, all eight samples per record, all nine integer gates, pass@1/pass@8, paired cluster-aware confidence intervals, output length, repetition, and truncation. Report counts and rates; a threshold crossing on 30 questions is not certainty about generalization. Never rerun a successful weak candidate unchanged or select a lucky seed.

If it fails, distinguish a demonstrated operational bug from a negative method result. Do not automatically proceed to rewritten-target or multi-path training: first establish the extension's assignment fit, verified training-data coverage, and credible single-B200 cost. Keep all weak outcomes and fulfill the assignment deliverables even if strict historical domination has not been achieved.

## Traceable evidence

- `results/arc-20261011-snapshot-2316/evaluations/lmf-ties-sft-gem-15k-suite-comparison.json`
- `results/arc-20261011-snapshot-2316/evaluations/lmf-ties-sft-gem-15k-domination.json`
- `LMF_CONTROL_EXPERIMENT.md`, `GEN_SSD_EXPERIMENT.md`, `GEM_EXPERIMENT.md`, `MODEL_AWARE_GEM_EXPERIMENT.md`, `SKILLGRAPH_ASFT_EXPERIMENT.md`
- Current preparation job: 1183868, initially pending B200 resources. SkillGraph 1180491 evaluation was cancelled at the user-authorized handoff after preserving seven complete benchmarks; training and weights are complete.

## Preparation recovery on October 17

Current preparation job is **1186667**, the single corrected resume of failed 1183868. All 49,416 saved requests validated; 60,900 remain. The prior broad fatal guard did not retain whether it saw empty text or an invalid finish reason. Empty stop/length responses now persist as unsupported/censored; unexpected reasons remain fatal with explicit diagnostics. All method AST nodes outside generation handling and all plan method fields remain unchanged. The original code/plan/log and the old/new hashes are preserved. Twenty-nine local and ten staged CPU tests passed, no GPU pilot was run, and the queue was empty before submission. Initial resume status: pending B200 resources. No new benchmark result or success claim follows.

## New candidate after negative demonstration preparation

The demonstration-response selector completed scoring but failed fixed category capacity; it is retired before training, not relaunched. [FULL_LENGTH_SFT_EXPERIMENT.md](FULL_LENGTH_SFT_EXPERIMENT.md) freezes the distinct original-target 16K, three-epoch, 5,625-update candidate. Its selector reads none of the failed student rollouts or public scores. The actual converter audit found that previous file-name-only dataset entries ignored the system column; explicit mapping and converter-level checks are now required. There is no causal claim that this mismatch explains all old weak scores. The new candidate uses the assignment reference recipe, not claimed historical winner settings; final all-nine evaluation remains unchanged. CPU export/checks must finish before one B200 submission.
