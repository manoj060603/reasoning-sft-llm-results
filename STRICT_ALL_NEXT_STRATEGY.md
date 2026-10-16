# Strict all-nine successor strategy

## Success criterion

One fixed Qwen2.5-3B-Instruct descendant must clear every prespecified Fall 2025 solve-count threshold under the unchanged assignment evaluator. A higher macro, fewer truncations, or a partial set of wins is a failed candidate. Historical runtime and dataset equivalence remain unverified, so a numerical clearance would still be reported as an internal reproduction result.

## Parallel literature review

Three independent searches converged on the same diagnosis: the next run must change verified skill and reasoning-path coverage, rather than apply another loss or merge to the same targets.

- [MASS](https://arxiv.org/abs/2503.14917) and its [official code](https://github.com/lijiazheng0917/MASS) use a mathematical skill graph for coverage-aware data selection and report gains at a fixed token budget.
- [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284) emphasizes unique-prompt coverage over repeatedly sampling the same prompts.
- [OpenMathReasoning](https://arxiv.org/abs/2504.16891) uses generated solutions, quality filtering, and verifier-based solution selection.
- [Anchored SFT](https://arxiv.org/abs/2509.23753) adds a frozen-base constraint intended to reduce capability drift.
- [Reasoning Path Divergence](https://arxiv.org/abs/2510.26122) and its [official code](https://github.com/fengjujf/Reasoning-Path-Divergence) report pass@k gains from selecting semantically different correct paths.
- [Set Supervised Fine-Tuning](https://arxiv.org/abs/2510.05132) and its [official code](https://github.com/Sheng-J/SSFT) show that naive multi-trace SFT can collapse modes and introduce a set-level alternative.
- [Inference-Aware Fine-Tuning](https://arxiv.org/abs/2412.15287) directly optimizes best-of-N behavior, but its RL variants are less aligned with the assignment's SFT focus and cost more on one B200.

These papers motivate the design; none guarantees clearing all nine heterogeneous thresholds.

## Next full candidate: SkillGraph-Verified Anchored SFT

1. Freeze the hashes of the complete 83,037-row sealed pool, decontamination records, tokenizer, base model, and source code.
2. Use the pinned 7B teacher to label every prompt from its first 2,048 instruction characters with constrained JSON containing broad domain, two to five skills from a fixed 46-skill vocabulary, composition depth, representation type, language, and answer form. Cap this classification output at 128 tokens and require schema-valid decoding.
3. Normalize skill aliases deterministically and build a benchmark-blind skill and skill-pair co-occurrence graph. No benchmark prompt, answer, score, error label, or leaderboard column may enter the graph.
4. Apply the existing sealed-pool contamination and unique-ID checks, then the 8K serialization, complete-box, response-length, and repetition filters. Independently generate one alternative solution for 256 high-composition selected prompts and audit symbolic final-answer agreement and exact token bounds. These alternatives are diagnostic only; they do not replace the original training targets, and matching final answers do not prove intermediate reasoning.
5. Select exactly 15,000 unique prompt IDs by deterministic greedy inverse-frequency-weighted square-root coverage over skills, skill pairs, broad labels, and the frozen 128-cluster IDs. Repetition enters only a small fixed quality term. No public-score signal enters the selection.
6. Train full-parameter response-only anchored SFT from the pinned original base, using detached gold-token probability weighting and KL(reference || online) against the frozen original base at the authors' BF16 weight 0.03. Use an 8K cutoff, effective batch eight, one prompt epoch, and exactly 1,875 updates.
7. Evaluate the single final checkpoint once on all nine benchmarks at temperature 0.6, top-p 0.95, eight samples, and 32,768 maximum new tokens.

## Required receipts

- finite skill labels for exactly 83,037 unique IDs;
- raw and normalized skill-tag hashes, taxonomy version, graph edge list, and graph hash;
- exact selection objective, quotas, marginal gains, and 15,000 unique selected IDs;
- source, teacher, tokenizer, model, code, configuration, and final dataset hashes;
- alternative-generation manifests, symbolic final-answer agreement, text-divergence scores, and 8K checks for the diagnostic subset;
- complete chat-template token counts with a maximum of 8,192;
- exactly 1,875 finite updates and complete final model files;
- complete matched predictions and manifests for all nine benchmarks, pass@1/pass@8, truncation, repetition, output length, paired confidence intervals, macro, and strict per-column gate.

## Escalation route if skill coverage remains insufficient

The next evidence-backed escalation is RPD-curated SSFT over the same 15,000 unique prompt identities, with two independently answer-verified paths stored as a set per prompt. This is held behind an implementation and protocol gate because paper-faithful fork tokens can change the evaluation interface, and candidate generation plus set-level training is estimated at roughly 26–58 single-B200 hours.
