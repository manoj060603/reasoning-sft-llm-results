# SkillGraph-Verified Anchored SFT experiment

## Objective and status

The only success condition is one fixed model exceeding **every** prespecified Fall 2025 solve-count threshold in the unchanged nine-benchmark internal evaluator. Historical dataset/runtime equivalence is not established. The preceding fixed TIES merge cleared only Minerva and GPQA (2/9), so this is a new data-and-objective intervention, not a TIES retune.

Preparation job **1176835** failed in its four-example preflight before writing labels because vLLM 0.19 rejected the schema's `uniqueItems` keyword. The schema was fixed while retaining duplicate-skill validation in Python. Replacement Owl job **1177248** was cancelled while pending to use an earlier Falcon L40S slot. Falcon job **654015** then failed in the same preflight before writing labels because vLLM requires a configured reasoning parser when `thinking_token_budget` is set. That unused option was removed; the structured JSON constraint remains. Neither failure produced training data or a benchmark result. The full training/evaluation job will be submitted only after a successful preparation receipt and an idle project GPU queue.

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
