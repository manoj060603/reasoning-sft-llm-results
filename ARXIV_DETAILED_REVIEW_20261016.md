# Detailed literature review for the strict nine-column objective

Review date: October 16, 2026, America/New_York. Scope: the methods, experimental settings, ablations, limitations, and relevant appendices of nine primary papers, plus the assignment-linked LIMOPro implementation. This is a focused review, not an exhaustive reading of arXiv. Full HTML copies were downloaded to a temporary reading directory; versioned URLs and content hashes are recorded in `research/arxiv_review_20261016_sources.json`. No external training dataset or model weights were downloaded.

The implementation review pinned four author repositories. Select2Reason's linked repository returned HTTP 404 during this review, so its code was not inspected. The distinction matters: available code and an abstract are not the same as a reproducible result in this assignment.

## 1. Select2Reason — difficulty and trace length

[Paper, v1](https://arxiv.org/html/2505.17266v1), particularly sections 3–6 and limitations. It ranks difficulty and response length jointly, using Qwen2.5-Math-7B on OpenR1 data. Its preliminary hard-example selection uses actual base-model solving; domain balancing alone does not reliably improve results. Metrics include pass@1 and majority@16, not our complete pass@8 protocol. Equation 4 appears to repeat the zero-token probability in the denominator; do not transcribe it without resolving the intended normalization. The important lesson is that difficulty, length, and utility differ. This does not validate our failed 512-token-NLL selector or prove that longer is always better.

## 2. Efficient Long CoT Reasoning in Small Language Models — prefix sufficiency

[Paper, v2](https://arxiv.org/html/2505.18440v2), section 3, experiment setup, main results, and prompt appendix. A student is asked to answer from a teacher-reasoning prefix, then prefix search constructs shorter targets. Experiments use 7B/8B students and about 25K retained samples. Large length savings coexist with accuracy losses relative to full-trace SFT in several cells. We borrow the *measurement* of whether a demonstration helps the student, without its pruning or target-replacement intervention. Successful prefix-conditioned answering is not proof that the student will learn to generate that prefix after SFT.

## 3. Learning What to Learn — acquisition differs from consolidation

[Paper, v1](https://arxiv.org/html/2606.04466v1), sections 3–4 and appendices A–D. It estimates difficulty with 64 samples, studies a medium-to-hard SFT curriculum at matched token budgets, and uses rewritten bridge supervision plus subsequent RL/recycling. Students are Qwen2.5-0.5B-Instruct and Llama3.2-1B-Instruct, largely trained from GSM8K. Our adaptation keeps original responses and omits RL, local extra prompts, and bridge rewriting. It motivates retaining hard acquisition examples; it does not establish our proposed low-sample classification, mixture proportions, or nine-column performance.

## 4. On-Policy SFT — the generation/update loop is essential

[Paper, v1](https://arxiv.org/html/2602.13407v1), sections 3.2, 5.2–5.3 and appendices E–G. The algorithm refreshes current-policy generations during training, filters correctness/completion, and corrects length bias. Its fixed-dataset variant loses accuracy; one-time generation followed by offline SFT is not a faithful reproduction. Experiments primarily preserve accuracy while improving efficiency on DeepSeek-R1 1.5B/7B. The released trainer generates and filters within the update loop. This corrects the earlier proposal to call a frozen curated dataset "on-policy SFT." It is insufficient evidence for a large cross-benchmark capability gain.

## 5. Reasoning Path Divergence — relevant, but the small-model appendix matters

[Paper, v2](https://arxiv.org/html/2510.26122v2), sections 3–4 and appendices B, D, E. It compares step summaries using asymmetric semantic distances, then selects problems and multiple paths. Appendix E.2 includes Qwen2.5-3B: AIME24 pass@8 is 20.00% for RPD versus 13.33% for one-path SFT, under its distinct protocol. The main experiments use small multi-response datasets and many epochs. Screening includes LLM judgments rather than universal proof verification. Its 3B evidence was omitted from the earlier short review. A multi-path reproduction cannot silently replace 15K unique prompts with fewer repeated prompts.

## 6. SSFT — strong interface assumptions

[Paper, v1](https://arxiv.org/html/2510.05132v1), method, inference protocol, experiments, and ablations. It matches several traces to reserved global forking tokens using bipartite matching. Inference explicitly inserts these tokens; headline results emphasize a 32B model and majority-vote consistency metrics. This interface is incompatible with simply dropping SSFT into our unchanged prompting and sampling pipeline. It remains a conditional research extension, not the next assignment submission. Naively storing several solutions is also not equivalent to its set-level loss.

## 7. AceReason-Nemotron 1.1 — scale and exposure cannot be ignored

[Paper, v1](https://arxiv.org/html/2506.13284v1), SFT curation, sections 4.4–4.5, and appendices. It studies unique prompts, responses per prompt, and repeated exposure; SFT gains continue across multiple epochs in its setting. Major headline gains also involve staged RL on far larger data and a Qwen2.5-Math-7B base. This supports checking our supervised-token budget and preserving prompt breadth. It does not justify importing its data, changing the base model, or assuming more epochs will repair our results. One epoch is a project recipe, not an assignment-wide limit.

## 8. Selecting Diverse SFT Traces Improves Post-RL Generalization — avoid crediting RL gains to SFT alone

[Paper, v1](https://arxiv.org/html/2609.33780v1), sections 3–4 and selection/diagnostic appendices. It uses procedural fingerprints and matched-budget selection, with principal coverage improvements measured *after RL*. Pre-RL mixed-outcome diagnostics explain possible exploration benefits. The project's existing lexical route signatures are an adaptation, not its full topology extractor. We retain broad route coverage as a safeguard, not as proof that a new selector will improve standalone SFT pass@8. More samples are useful only if they recover additional correct solutions.

## 9. LARK — within-prompt trajectory choice is a different problem

[Paper, v1](https://arxiv.org/html/2605.30651v1), sections 4–5 and appendices B, C, E. It ranks 33 answer-checked trajectories per problem, using a forward-pass learnability surrogate and a top-B weighting rule. Experiments use 5K problems, several students, five epochs, and a different metric. Its guarantees require structural assumptions and concern a surrogate, not downstream nine-column success. With one original trajectory per prompt, within-prompt selection is vacuous. The released scorer approximates the Brier term with a renormalized top-50 distribution, so it is not the exact full-vocabulary expression. We do not relabel another cross-prompt NLL ranking as LARK.

## 10. Assignment-linked LIMOPro — implementation evidence

[Author repository](https://github.com/GAIR-NLP/LIMOPro), README and `CIE/cie.py`, `CIE/calculate_cie.py`, `CIE/filter.py`. It measures answer-perplexity changes when reasoning groups are removed, with explicit attempts to skip answer-bearing groups. It refines responses, rather than simply selecting difficult questions. This provides an implementation reference for answer-leakage checks and local contribution measurements, but it is not permission to rewrite assignment responses. Its utility code is not a ready-made replacement for our tokenization, devices, provenance, or grader.

## Implementation revisions reviewed

| Repository | Pinned revision | Files inspected |
|---|---|---|
| `EIT-NLP/On-Policy-SFT` | `d788bc027a4bd6aea18051ce75081870fa3ac6b6` | README, update actor, generation/filter/update trainer |
| `fengjujf/Reasoning-Path-Divergence` | `8e82b0b7475ba89a714f41c02494371fb0e4262f` | configuration, distance matrix, problem selection, answer selection |
| `Tianrun-Yu/LARK` | `38cfd4f54707b19e5b4abd3e8a00051636fef19e` | README, forward scorer, top-B selector, full-SFT configuration |
| `GAIR-NLP/LIMOPro` | `7f18718bdeb2ae99bca7dd7f3a72095bfe3ee394` | README, perplexity scorer, group removal scoring, filtering |

Only source inspection was performed. These external implementations were not executed on the GPU, and this review does not certify them as bug-free or reproduce their results.

## Synthesis for this project

The next candidate should test whether an original demonstration supplies a usable missing step, preserve reliable skills, and leave room for genuinely hard acquisition. A selector based solely on current success preferentially rehearses existing capability; a selector based solely on surprise can admit incoherent or inaccessible targets. Skill counts do not resolve that distinction.

The concrete proposed adaptation is specified in [DEMONSTRATION_RESPONSE_SFT_DESIGN.md](DEMONSTRATION_RESPONSE_SFT_DESIGN.md). It measures assistance from an original reasoning prefix, selects unchanged original examples, and uses an acquisition curriculum with retention examples. This is a new hypothesis assembled from the reviewed mechanisms. No paper has validated the combined design, its allocation constants, or its ability to reach all nine gates.

The highlighted supervision audit, selection, and full evaluation checklist remains necessary execution discipline. It was not itself a research contribution. No new candidate was submitted during this design review; job 1180491 continues its full B200 evaluation.
