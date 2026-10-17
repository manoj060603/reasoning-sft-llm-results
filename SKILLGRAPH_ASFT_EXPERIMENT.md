# SkillGraph-Verified Anchored SFT experiment

## Objective and status

The only success condition is one fixed model exceeding **every** prespecified Fall 2025 solve-count threshold in the unchanged nine-benchmark internal evaluator. Historical dataset/runtime equivalence is not established. The preceding fixed TIES merge cleared only Minerva and GPQA (2/9), so this is a new data-and-objective intervention, not a TIES retune.

Preparation job **1176835** failed in its four-example preflight before writing labels because vLLM 0.19 rejected the schema's `uniqueItems` keyword. Replacement Owl job **1177248** was cancelled while pending to use an earlier Falcon L40S slot. Falcon job **654015** then failed in the same preflight before writing labels because vLLM requires a configured reasoning parser when `thinking_token_budget` is set. That unused option was removed. Falcon job **654075** reached structured generation but failed the same preflight when the teacher repeated a skill despite the prompt asking for distinct skills. Since xgrammar cannot enforce `uniqueItems`, repeated skill names are now deduplicated deterministically; a resulting single skill is retained rather than inventing another. Falcon job **654091** was cancelled while still pending at the user's request to run only on B200. The corrected official preparation is now Owl B200 job **1179718**. None of the prior failures produced training data or a benchmark result. The full training/evaluation job will be submitted on Owl B200 only after a successful preparation receipt and an idle project GPU queue.

## Literature and frozen design

- [MASS](https://arxiv.org/abs/2503.14917) motivates mathematical skill graphs and coverage-aware selection. Its published results concern pretraining; application to this 15K SFT assignment is experimental.
- [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284) motivates preserving unique-prompt coverage.
- [Anchored Supervised Fine-Tuning](https://arxiv.org/abs/2509.23753) and the [authors' implementation](https://github.com/zhuchichi56/ASFT) motivate detached gold-token probability weighting plus frozen-reference KL. The authors recommend KL weight **0.03** for BF16. This project implements that response-only objective with memory-bounded vocabulary chunks.

The pinned 7B teacher labels all 83,037 sealed training prompts using a constrained JSON schema: domain, two to five fixed-vocabulary mathematical skills, composition depth, representation, language, and answer form. Only the instruction prefix enters labeling. A deterministic, benchmark-blind greedy objective maximizes inverse-frequency-weighted concave coverage of skills, skill pairs, domains, languages, representations, answer forms, depth, and the pre-existing 128 clusters. The static eligibility screen requires a complete boxed answer, at least 128 response tokens, repetition at most 0.35, and a conservative 8K limit. Exact Qwen chat-template counts must then be at most 8,192. It exports exactly 15,000 unique sealed prompt IDs; all original target responses remain unchanged.

For 256 high-composition selected prompts, a separate teacher generation is compared symbolically with the original final boxed answer. This is a diagnostic receipt, not a source of training targets, and matching final answers do not establish the correctness of intermediate steps. The four-example live label and answer-generation checks run before the expensive stages. Public benchmark prompts, answers, scores, errors, and leader columns never enter labeling, selection, training, or hyperparameter choice.

The full model is pinned Qwen2.5-3B-Instruct with full-parameter response-only anchored SFT, original base as frozen reference, KL weight 0.03, learning rate 2e-6, effective batch eight, 8K cutoff, one 15K-prompt epoch, and exactly 1,875 updates. The single final checkpoint is evaluated on all nine benchmarks at temperature 0.6, top-p 0.95, eight samples, and 32,768 maximum new tokens. No development gate, public-score tuning, or checkpoint selection is included.

## Required terminal checks

The preparation receipt must prove source hashes `c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9` and `039cd63692d528dcce2caf2be2bfed85be917a66298a2532933ef1107f9db979`, 83,037 ordered unique skill labels, exact 15,000 unique selected IDs, finite selection gains, 8K token bounds, and 256 alternative-generation audit records. The final run must prove 1,875 finite updates, complete model files, eight predictions per question on all nine matched manifests, pass@1/pass@8, truncation, repetition, output length, paired confidence intervals, macro, and the strict per-column integer gate.

The preparation and training code passed 58 local unit tests, ARC loss/schema tests, shell/Python syntax checks, and an ARC CPU end-to-end export/token-count/answer-check preflight before job submission. After the second compatibility fix, the vLLM sampling parameters constructed successfully and the three skill-graph unit tests passed on Falcon. The live four-example structured-generation preflight has not yet passed.


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
