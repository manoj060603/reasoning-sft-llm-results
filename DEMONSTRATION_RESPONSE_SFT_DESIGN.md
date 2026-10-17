# SFT curriculum selected by response to original demonstrations

Design proposal, October 16, 2026 (America/New_York). Status: **implemented and CPU-verified; final preparation plan frozen before GPU generation; no measured win**. The distinguishing intervention is to select missing skills for which the original demonstration gives the student useful assistance. Its score measures conditional answering, not actual improvement after a gradient update.

## Research question

Can one fixed 3B model acquire more transferable skills by training on original problems that it currently fails, but can complete after seeing an answer-withheld prefix of their original solutions, while preserving reliable skills through retention examples?

Current solve rate is not the same as training utility. A problem can be unsolved yet teachable. A high-NLL response can instead be noisy or inaccessible. This design measures a third quantity: the student's response to information already present in the original demonstration. It selects the demonstration; it does not shorten or replace it.

For a fixed problem with per-draw correctness p and independent draws, eight-sample solve probability is `1 - (1 - p)^8`. As an illustration, increasing p from 0 to .1 changes this probability from 0 to about .57, whereas increasing p from .8 to .9 barely changes its already saturated eight-sample probability. These are mathematical examples, not project forecasts. The motivation is to bring previously missed skills within sampling reach, rather than merely make already-reliable answers more confident. A dataset-wide mean pass@1 cannot be substituted for each problem's p in this formula.

The rationale is informed by [prefix sufficiency](https://arxiv.org/html/2505.18440v2) and [acquisition curricula](https://arxiv.org/html/2606.04466v1). The former motivates a conditional answering measurement; the latter motivates preserving hard acquisition alongside manageable examples. Neither paper validates the combined selector below. Its constants are explicit design choices, not paper defaults or leaderboard-tuned values.

## Preserved protocol and assignment fit

- Pinned original Qwen2.5-3B-Instruct is both the preparation scorer and the training initialization. No publicly evaluated candidate is chosen as scorer.
- Exactly 15,000 unique prompt identities from the existing sealed approved AceReason pool. One unchanged original response per prompt; no additional local questions, rewritten targets, duplicate response rows, or fork tokens.
- Original system prompt and Qwen template; response-only full-parameter SFT. All original response and EOS tokens must fit inside 8,192 tokens. Preserve target content hashes.
- Ordinary cross-entropy, learning rate 2e-6, effective batch eight, cosine schedule and 3% warmup, seed 42, one complete prompt epoch, exactly 1,875 updates. The main intervention is data selection/order; no new probability-weighting loss, KL term, merge, or RL stage.
- One final checkpoint evaluated once on all nine datasets at temperature .6, top-p .95, eight samples, and 32,768 maximum new tokens. No public-suite gate, checkpoint choice, seed choice, inference hints, or verifier-based answer selection.
- Owl B200 only. Job 1180491 must terminate before any GPU preparation starts. Sequential resumable preparation allocations may preserve the immutable ledger; never overlap or relaunch an unchanged operational failure repeatedly.

The one-epoch/8K recipe is retained here to respect the existing experiment discipline and limit simultaneous changes. It is not asserted optimal. It differs from the assignment's example 16K/three-epoch recipe. More training exposure is an unresolved variable, rather than an automatic rescue after public evaluation.

## Preparation and measurements

### 1. Freeze an eligible candidate pool

Use the pool and static-feature hashes already recorded in the project. Apply existing contamination exclusions, unique-ID checks, complete-answer, repetition, and exact-template 8K eligibility. No public benchmark text, score, error label, or per-benchmark quota enters this stage.

Select 30,000 eligible candidate IDs with the existing frozen 128-cluster coverage rule, stratifying within clusters by original response-length quartile and stable ID hash. Allocate quartile budgets proportionally to available eligible records; distribute remainders deterministically. If fewer than 30,000 eligible records exist, report capacity before requesting a GPU. Preserve a deterministic reserve ordering for record validation failures, with every replacement recorded before scoring begins.

All selection and reserve rules must be implemented and frozen before generation. The final training size remains 15,000; the 30K pool is preparation-only and is not a GPU pilot.

### 2. Construct one answer-withheld demonstration prefix

Take the original response prefix ending at the last paragraph boundary before the smaller of one quarter of its response tokens and 768 tokens. A usable prefix must be nonempty, precede the first explicit final-answer/boxed-answer declaration, and preserve whole text units. The original exported training response remains untouched.

Reject prefixes containing an extracted boxed answer, explicit final-answer statement, or detectable rendering of the normalized target answer. Very short/common answers can cause conservative false exclusions; report this. These mechanical checks cannot prove that an unboxed equivalent answer is absent. A fixed training-only audit below addresses residual leakage. Prompts without usable prefixes cannot enter the acquisition category through this measurement.

The conditional generation uses the same system prompt and original user problem, with the partial original assistant response supplied as a continuation prefix. It does not request a paraphrase, supply the target answer, introduce a teacher-generated hint, or create an evaluated inference protocol. Incomplete thinking delimiters must be handled consistently with the tokenizer; the implementation must prove the exact prefix token span.

Use a tokenizer-supported continuation of the final assistant message (or a verified equivalent token construction). A serialized assistant EOS placed before the prefix continuation would invalidate the measurement. Check decoded token spans and the absence of an unintended end-of-message marker in the CPU preflight.

### 3. Compare answering with and without the prefix

For each scored candidate, generate two unassisted and two prefix-conditioned completions using the pinned base, T=.6, top-p=.95, and at most 2,048 new tokens. Derive stable seeds from the global seed, prompt ID, condition, and sample index. The short budget is a preparation measurement and does not change the final evaluator.

Record original ID, prompt/response/prefix hashes, exact prefix length, model/tokenizer identity, seed, completion, finish reason, extracted answer, verification support, symbolic match, and repetition. Do not execute generated code. Unsupported answer forms and generation failures are distinct from incorrect answers.

Let c0 and ch be the numbers of complete, supported, answer-matching outputs in the two unassisted and two assisted draws. Define the observed demonstration gain as:

`gain = (ch - c0) / 2`.

Use this only as a coarse ranking signal. Two samples cannot identify a true solve probability or certify that a problem is intrinsically hard. In particular, 0/2 does not mean zero success probability. Length-limited or invalid unassisted draws are censored; they cannot establish acquisition eligibility. Keep their receipts and report how many examples/skills the short preparation budget excludes. Do not silently convert them into failures or increase the budget after inspecting public scores.

### 4. Select exactly 15K original examples

Frozen disjoint categories:

| Category | Count | Observed requirement | Learning role |
|---|---:|---|---|
| Retention | 3,000 | 2/2 complete unassisted answers match | Preserve reliable skills and normal answer completion |
| Acquisition, manageable | 6,000 | 1/2 unassisted matches, 2/2 assisted matches, all draws complete/supported | Strengthen accessible reasoning with useful original demonstrations |
| Acquisition, currently missed | 6,000 | 0/2 unassisted matches, at least 1/2 assisted matches, all draws complete/supported | Introduce skills with observable demonstration assistance |

For each category, preserve cluster representation through the existing square-root quota allocation, subject to category capacity. Within a cluster, select greater observed gain first; resolve ties by marginal skill/skill-pair coverage from the frozen taxonomy, then stable ID. Retention ties use coverage and hash. No extra mixture weights are optimized.

Record every selected ID, category, score, quota, coverage increment, and tie-break. If the candidate pool cannot fill a category, the design has a preparation feasibility failure. Do not fill acquisition slots with arbitrary targets, duplicate identities, quietly change proportions, or launch a nominally identical full run. Revise a new training-only design and report that change explicitly. No 9/9 forecast is justified by passing preparation.

This selection preserves broad cross-prompt coverage. It does **not** reproduce RPD multi-path training or prove output-path diversity. One original path per prompt can still yield correlated wrong answers at evaluation.

### 5. Audit the actual mechanism before training readiness

Choose 256 selected examples deterministically: 86 retention, 85 manageable, and 85 currently missed, round-robin across language/length strata and prioritizing composition depth within strata. Audit final-answer extraction, mechanical prefix answer exposure, and original-target agreement. The implemented symbolic grader verifies final answers only; intermediate arithmetic/equation transitions are **not** independently verified, and this candidate must not be described as proof-verified.

For this fixed audit subset, generate **six fresh completions per prompt**: two unassisted, two with its correct prefix, and two with a mismatched prefix from a different training problem of the same broad cluster and length quartile. All audit seeds differ from selection seeds. Exclude mismatched prefixes that mechanically expose the current target answer. Compare the three conditions using these independent outcomes, rather than reusing selection draws that were chosen for an observed gain. This tests whether a supposed gain appears to depend on relevant demonstration information, rather than any added text. A mismatched prefix may itself be distracting, so this is a negative-control diagnostic, not a causal estimate of learning.

Require at least 32 question IDs with all six outputs complete and symbolically supported. At the question level, compare the two-draw correct-prefix solve count with each control using one-sided exact paired sign tests, ignoring ties, with Bonferroni alpha .025 per comparison. Both comparisons must pass. This is a prespecified training-data mechanism check, not a public benchmark, GPU pilot, accuracy forecast, or checkpoint selection. Insufficient category/control capacity or a negative/inconclusive audit prevents a training-ready receipt; preserve the negative preparation result.

Any direct answer leak must be fixed in the mechanical prefix construction and all affected scores invalidated/recomputed with preserved logs before export. Do not optimize the audit sample, use public benchmark feedback, or call a weak final-answer match rate a proof-quality certificate. If the audit cannot distinguish the proposed signal from leakage or parser failure, report the mechanism as unresolved and do not claim readiness.

## Curriculum and full training

Use original complete targets for every selected identity. Partition the 3,000 retention examples into two disjoint, coverage-balanced halves of 1,500.

1. Present the 6,000 manageable acquisition examples with the first 1,500 retention examples interleaved.
2. Continue with the 6,000 currently missed acquisition examples and the other 1,500 retention examples interleaved.

Shuffle within each acquisition block using the frozen seed. Distribute each retention half evenly through its block. The model sees each of the 15,000 identities exactly once. There is one optimizer/scheduler trajectory, no reset at the 7,500-example boundary. Gradient accumulation may span that boundary; the total remains 15,000/8 = 1,875 updates. A dataloader that shuffles the entire dataset would erase this intervention and must be rejected by the CPU preflight.

The supervised labels must reconstruct the full original response and train the final answer plus EOS. No conditioning prefix is added to the user prompt during SFT: training begins at the normal question and teaches the original full demonstration.

## Why this differs from the completed routes

| Completed route | Unresolved limitation | Proposed change |
|---|---|---|
| Raw-NLL model-aware GEM | Surprising teacher tokens did not predict downstream value | Observe response to the original demonstration on actual answering |
| SkillGraph ASFT | Taxonomy coverage did not show student acquisition; loss and data changed together | Keep coverage safeguards and add a specific student-response signal with plain CE |
| Compact-target SFT | Better stopping accompanied worse accuracy | Keep the entire original reasoning and answer |
| Gen-SSD | Rewritten generated supervision did not transfer on confirmation | Generate preparation evidence only; preserve original targets |
| TIES/interpolation | Combining learned parameters left substantial solve deficits | Change which missing skills receive supervision and their order |

The desired transfer is from demonstration-responsive training problems to new problems requiring similar operations. That transfer is the unmeasured hypothesis. Conditional answering cannot prove SFT will acquire the same skill, and retention examples cannot guarantee no forgetting.

The proposed full run changes selection and order together. It tests the combined candidate, not the isolated causal effect of either component. Report actual supervised-token exposure by category and stage; equal example counts do not imply equal token budgets. No ablation claim follows without separate matched full-run evidence.

## Compute and readiness

Four scoring completions for 30K candidates at a 2,048-token limit yield at most 245,760,000 output tokens. The fresh three-condition 256-example audit adds at most 3,145,728. Total upper bound: 248,905,728 output tokens, excluding prefill, template tokens, sampling overhead, scoring, training, and final evaluation. At an illustrative sustained 2,000 output tokens/s, output generation alone is about 34.6 hours. At 1,000/s it is about 69.1 hours. These are arithmetic scenarios, not measured B200 ETAs.

Use existing B200 logs for a measured throughput estimate, then account for current-job completion, queue time, resumable eight-hour preparation allocations, training, nine benchmarks, uploads, and the October 20, 11:59 p.m. Eastern deadline. Do not launch a budget that cannot plausibly finish. The frozen preparation budget is at most five sequential eight-hour B200 allocations (40 GPU-hours). The allocation receipt binds their IDs to the same plan hash. Reaching a time limit with a progressing ledger permits the next planned chunk; it does not authorize changed selection rules. A recurring operational error is different and must not be repeatedly relaunched. Stop and report capacity/audit failure or budget exhaustion rather than consume indefinite GPU time.

Before submission, implement and CPU-verify category membership, answer/finish-reason handling, stable resume seeds, leakage rejection, quota capacity, unique IDs, exact token spans, original response hashes, curriculum order, and the 1,875-update accounting. No GPU pilot or development benchmark. The implementation passed 28 local tests covering selection, censoring, leakage, deterministic resume, and generation-ledger handling. On Owl, the actual LLaMA-Factory tokenizer, supervised processor, and native sequential sampler passed a synthetic 15K-row CPU path check: complete original responses/EOS, masked prompts, exact curriculum order, and 1,875 updates. The sealed full-source CPU pass found 30,000 exact eligible candidates and 25,158 usable prefixes. This is pre-generation capacity only; category capacity, the independent audit, and the final selected 15K real training rows remain unmeasured until preparation. The real-row LLaMA-Factory check is mandatory again after export.

## Measurement and decision

The unchanged all-nine evaluator alone decides the frozen integer gates: AIME24 8/30; AIME25 8/30; MATH 451/500; CN Math 14/30; Kaoyan 138/199; AMC 33/40; Minerva 95/272; OlympiadBench 385/675; GPQA 121/198. Also report pass@1, matched pass@8, paired confidence intervals, output lengths, repetition, truncation, and all manifests.

The hardest measured TIES deficits include 42 OlympiadBench, 31 MATH, and 22 Kaoyan solves. This design is intended to acquire missing operations while preserving easier successes; it has no validated forecast for any of these counts. A partial win remains a failed candidate. The user authorized moving on without waiting for this candidate. Preserve its available results and make the B200 available once the verified successor is ready; never overlap GPU jobs. Prepare this mechanism as the next complete candidate, with no claim that it will clear the nine gates. Preserve negative results without rerunning an unchanged method.

## Implementation receipt before GPU submission

The nine staged demonstration-response tests passed in ARC’s installed runtime (test discovery avoids an unrelated installed `tests` package). The real source CPU plan and synthetic 15K training serialization preflight run on CPU, never as a GPU pilot. Runtime modules are `reasoning/demonstration_response.py` and `reasoning/demonstration_preflight.py`; preparation and full training entry points are `scripts/demonstration_prepare.slurm` and `scripts/demonstration_sft_full.slurm`. Preparation creates no training-ready receipt unless the categories fill, the fresh controls pass, and all real selected rows pass the actual training processor.

## Actual handoff and launch

At 2026-10-17 04:20 UTC, the user-authorized handoff cancelled only the unfinished evaluation tail of SkillGraph job 1180491 (training had completed); seven complete benchmark results and all original model files were preserved. The queue was confirmed empty before submission. Full B200 preparation job **1183868** was submitted with `scripts/demonstration_prepare.slurm`; its initial state is **PENDING (Resources)**, not running. No GPU jobs overlapped. The final CPU-plan receipt is `research/demonstration_response_prelaunch_1183868.json`; submission receipt is `research/demonstration_response_submission_1183868.json`. The existing 15-minute heartbeat now follows this preparation job and will submit full training only after all frozen readiness checks pass.


## Operational recovery of preparation job 1183868

Owl access recovered on October 17 after repeated SSH connection timeouts. The complete 659-line job log and saved state were inspected: job 1183868 exited after 2:03:06 with 49,416 valid unique saved requests, leaving 60,900 of 110,316. Its broad guard rejected an empty **or** invalid completion without recording which condition triggered. There was no earlier engine/OOM error in the complete log; engine shutdown followed the parent exception. The exact failed output was not retained, so immediate EOS is a possible explanation, not a recovered fact.

The concrete guard defect was corrected: an empty/whitespace-only response ending with `stop` or `length` is saved as unsupported/censored, with correctness false used only as a storage flag; existing category logic excludes unsupported responses from solve evidence. No longer treating such responses as fatal avoids repeatedly failing a legitimate terminal outcome. Other finish reasons remain fatal and now report finish reason, text length and token count. There is no increased generation budget, retry-to-correct selection, changed seed, taxonomy, quota, target or model. The original code, plan, allocation receipt and complete log are preserved remotely under `artifacts/demonstration-response/recovery-1183868/`.

All method AST nodes outside `run_requests` must match before and after the correction; all frozen plan fields except the preparation-module digest remain identical. The recovery receipt records old/new code and plan hashes plus the saved ledger hash. The allocation receipt keeps the failed job in the five-allocation budget. CPU tests cover immediate EOS, whitespace-only output, length-limited empty output, censorship, and resume without regeneration. The local suite passed 29 tests; the staged demonstration tests must pass in ARC before one non-overlapping B200 resume. No GPU pilot is used. A recurrence is not authorization for another unchanged relaunch.

The recovery CPU tests passed on ARC (10 tests), and exactly one resume was submitted as **1186667** after confirming an empty queue. Its initial status is **PENDING (Resources)**. Recovery/submission receipts are `research/demonstration_response_recovery_1183868.json` and `research/demonstration_response_resume_1186667.json`. The previous plan receipt remains an immutable record of initial submission; the recovery receipt explicitly records the new operational code/plan binding. No candidate data, category requirement, seed, target or evaluation setting changed.
