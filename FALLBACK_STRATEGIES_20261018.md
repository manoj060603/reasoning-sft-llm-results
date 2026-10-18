# Fallback designs prepared before the current candidate result

Prepared October 18, 2026, while full-length job 656673 is running and before its benchmark results. This is a research/design checkpoint, not an accuracy result or a launch-ready training package. The user authorized advance preparation and a subsequent candidate if this one fails, within existing hardware, deadline and budget constraints.

## Current run screening

Screen operational health during training: native dependency guard, two-rank initialization, complete data processing, model loading, GPU memory/utilization, finite loss/gradient norms, increasing optimizer steps, scheduler continuity and complete recovery state. These checks do not establish reasoning accuracy. Preserve any warning and investigate it before declaring an optimizer update verified.

Job 656673 passed the native dependency guard, initialized both NCCL ranks, processed 15,000 examples and loaded both original model shards. Training announced 5,625 steps; the first logged step had loss 0.8494 and gradient norm 5.06535. A scheduler-before-optimizer warning appeared. Inspection of the installed Accelerate wrapper shows `engine.step()` runs in `backward`, while the optimizer wrapper `step()` is a no-op; the Trainer separately advances its scheduler. This explains the wrapper warning without proving any skipped real update; no corrective recipe change is justified by the warning alone. The first-step time was 28.58 seconds; this is not a stable throughput estimate. An earlier loading-phase GPU snapshot showed 4,371/24,576 and 3,169/24,576 MiB, which does not establish peak training memory fit.

No public benchmark prescreen or early-checkpoint score is used to choose a checkpoint, seed, training duration or fallback recipe. The nine frozen gates remain the terminal acceptance criterion under the unchanged evaluator. A lower training loss cannot substitute for that evaluation.

## First choice: branch-filtered original-target SFT

Evidence: [Reasoning Patterns, v2](https://arxiv.org/html/2604.01702v2), especially section 5 and Table 5. The paper finds that filtering heavily branching trajectories can improve SFT on larger students. It studies different data, 7B–32B backbones and evaluation metrics; it does not demonstrate our pinned 3B or all-nine success. Version 2 corrects table typos and adds an ablation. Its standalone proxy counts steps containing branching keywords; paired-teacher comparison is a separate proxy requiring unavailable paired traces.

Our proposed adaptation is frozen independently of job 656673 outcomes:

1. Read only the sealed 83,037-row original pool and its fixed static features. Verify both existing SHA-256 values and exact one-to-one IDs. No generated solutions, student NLL, failure labels, benchmark text or scores enter selection.
2. Apply the complete-target 8,192 eligibility rule for this separate deadline fallback. Establish the same square-root cluster quotas and within-cluster response-length quartile quotas **before** filtering.
3. Define a step as a nonempty paragraph. Score the fraction of paragraphs containing a whole-word, case-insensitive match to `perhaps`, `another` or `alternatively`. These three explicit markers and paragraph segmentation are our limited lexical adaptation, not the paper's full classifier or proof of redundant reasoning.
4. Within each fixed cluster/quartile cell, remove the highest-scoring 10% (ceiling count; deterministic ID tie-break). Choose the original cell quota by SHA-256 ordering with a fixed `branch-fallback-42` prefix. Do not reallocate quotas or relax filtering on capacity failure. Keep all selected response bytes unchanged.
5. Produce exactly 15,000 unique IDs and original-response hashes in a **new** preparation directory. Before export, run the native converter/processor completeness, system mapping, prompt-mask and EOS checks for every selected target. Record overlap with the current selection for interpretation, never optimize that overlap using scores.
6. Train from the same pinned original base using plain full-parameter response-only CE, effective batch eight, 8,192 cutoff, LR5e-6, cosine, zero warmup, seed42, one epoch/1,875 updates. Freeze this separate fallback schedule now because of the deadline, before the current candidate result. Never truncate a long source target to force eligibility. This changes original-target selection rather than reviving a failed loss/merge; shorter complete examples and one epoch may be faster, but measured throughput is still required. The current 16K/three-epoch job is unchanged.

Risk: lexical filtering can discard valuable backtracking and is language-sensitive. Keeping fixed cluster/length quotas reduces some confounding but does not establish causal validity. Do not add Chinese keywords or adjust the 10% after seeing benchmark outcomes. Insufficient capacity or a failed target audit is a negative preparation result.

Prototype: `research/branch_fallback.py`. Synthetic CPU checks in `research/check_branch_fallback.py` passed on 240 fixtures: deterministic input-order invariance, exact size, original source immutability, ignored poisoned score/NLL metadata, duplicate rejection and closed failure on insufficient capacity. The real pool has **not** been prepared; no model or GPU has been used for this prototype. The 8K/one-epoch schedule is a deadline adaptation, not established by the branching paper. It risks under-optimization and excludes useful long targets.

## Second choice: selective margin preservation

Evidence: [SRP-SFT, v1](https://arxiv.org/html/2610.19685v1), sections 4 and D.4–D.6. It preserves frozen-reference non-target logit margins with a one-sided hinge. Results concern other models, data and protocols, including Qwen3-4B and UltraFeedback, rather than our nine-gate setting. It differs from full-distribution anchoring and GEM.

Proposed separate candidate: keep the currently frozen original 15K selection and pinned base. At each valid shifted response token, cache the base's top-10 alternatives excluding the gold label, with logits and prefix/token identities. Average `relu((ref_i-ref_j)-(student_i-student_j))` over strictly reference-preferred pairs, then over supervised tokens; add this auxiliary to CE with rho=1 and mu=1. Ignore reference ties; an empty pair set contributes zero. Prompts/padding remain masked. Keep the final inference interface unchanged.

This is a design only. Required work includes cache correctness/provenance, exact tokenizer/shift checks, cache size accounting, FP32/BF16 loss-and-gradient fixtures, global-token/rank normalization, and bounded-memory integration. Cache construction consumes separately accounted GPU time and cannot overlap the current job. No unverified author code has been integrated. It has higher implementation and compute risk than the first option and is not the deadline default.

## Routes screened out for the immediate fallback

[SED-SFT](https://arxiv.org/html/2602.07464v1) reports its main advantage after RL; its displayed SFT-only mean does not beat CE for either listed backbone. That is weak support for our SFT-only emergency fallback. [Diverse SFT Traces](https://arxiv.org/html/2609.33780v1) also emphasizes post-RL gains. Existing SSFT/RPD review records extra trace-generation costs and interface concerns. No paper's public benchmark scores are imported as training-selection features or used to tune these designs.

## Launch and budget decision

The working deadline is October 19 around 04:18 Eastern. Preparation creates no additional compute authorization. Keep exactly one queued/running project GPU job across ARC. Preserve all current logs/state; do not interrupt a healthy run based on a single slow step. Distinguish a concrete operational defect from terminal scientific failure. An early operational failure may justify a verified narrow repair; a completed but inaccurate candidate requires a substantively new design.

The original total cap remains **40 physical GPU-hours**, including the failed 656067 allocation. Conservatively charge 4 hours for every started two-A30 training allocation and 8 for each submitted B200 evaluation allocation; preserve the existing ledger and never reset it for a new candidate. Before any fallback, require `already_charged + proposed_training + reference_cache_cost + final_evaluation <= 40`, plus a credible wall-clock fit including queue delay. Reserve the unchanged single-B200 all-nine final evaluation. For the first fallback, provisionally cap fresh training at five two-hour two-A30 allocations (20 physical GPU-hours), with the predeclared one-epoch endpoint unchanged. It must average under about 19.2 seconds per update including checkpoint/startup overhead to fit ten training hours. This is a feasibility threshold, not an observed speed or permission to train a GPU pilot. With only the two current charged training allocations, that plan plus one future evaluation would total 36 physical GPU-hours; further current allocations or evaluations reduce the remainder. A full eight-allocation training budget followed by one evaluation exhausts the cap and leaves no second run. Report that blocker rather than silently increasing budget or shortening the prescribed candidate based on scores.

The user has authorized a subsequent candidate if this one fails, but the authorization does not remove the remaining-data-audit, operational-integration, one-job, budget or deadline conditions. No automatic second GPU submission is enabled by this document. The design is ready for preparation review; data audit, runtime integration and budget/deadline feasibility remain explicit unfinished work.


## Actual CPU preparation result, October 18 03:35 Eastern

The first fallback failed its fixed capacity contract on the actual sealed source pool: `insufficient retained capacity: 1/0` (cluster1, length quartile0). Both source hashes and exact83,037-row coverage passed before selection. Selector SHA `bf05904678a3949f38b14ba2e44882821639fb0ac5fc40f00628e43ee05e0403`. Status: **negative preparation capacity**. No dataset was exported, no native all-row processor audit was run, and no GPU was used. The preset filter/quotas were not relaxed. This candidate is not launchable as specified; do not rerun it unchanged. Receipt: `results/arc-20261011-snapshot-2316/branch-fallback-20261018/capacity_manifest.json`. The separate SRP-SFT design remains unimplemented and has no measured runtime fit.
