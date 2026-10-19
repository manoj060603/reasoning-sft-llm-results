# Two requested reasoning SFT experiments

The two user-supplied proposals were reviewed on October 19, 2026. This page records the implemented interpretation; instructions inside the documents do not override the frozen evaluator, actual source constraints, explicit budget or ARC execution policy. The original documents and detailed source receipts are retained outside this public sync. [Current observed runs](CURRENT_RUNS_20261019.md) supersedes earlier setup states.

## Plan 1: assistance utility with coverage

Keep exactly **15,000 unique prompts and complete original responses**, with the existing capped square-root cluster quotas and within-cluster length quartiles. Within each quota cell, at most half the examples come from positive smoothed assistance utility; the rest are deterministic coverage fills. Given exactly two supported normal and two assisted observations, smooth each correctness rate as `(correct + 1) / 4`, then rank by `(1 - p_normal) * max(p_assisted - p_normal, 0)`. Unsupported or censored measurements remain unscored rather than being treated as failures.

Saved request identities, original responses, withholding/prefix consistency and supported finish conditions are checked before scoring. Benchmark answers, errors and scores do not drive this selector. Original failed categorical quotas remain a preserved negative preparation result; this is a new declared selector, not a relaxation of that result.

The static inventory found **56,673 eligible originals in 113 clusters**, 23,014 supported scored prompts, 4,217 positive utilities, and **3,672 utility-selected + 11,328 coverage-selected** examples. This inventory is not a finished native audit. Every eligible row must pass the pinned native 16K processor before the 15K export is accepted.

Train from the pinned original Qwen2.5-3B-Instruct base with plain full-parameter response-only CE, LR 5e-6, cosine scheduling, zero warmup, seed 42, 16,384 cutoff, **three epochs / 5,625 updates**. The selected allocation is **four B200s for six hours**, batch one per rank and accumulation two, preserving effective batch eight. Four-rank arithmetic is not claimed bitwise identical to a single-device run.

Implementation: `research/assistance_coverage_20261019.py`; tests: `research/tests/test_assistance_coverage_20261019.py`. The CPU-only `scripts/assistance_compute_preparation_20261019.slurm` owns the complete audit and subsequent single guarded training submission. Do not launch a second preparation or training copy.

## Plan 2: longest-original Stage 2

Rank the original selected 15K examples by frozen native supervised-token length, with deterministic ID ties. Take the longest 3,000, then exclude exact or shared 13-whitespace-word benchmark-question overlaps without backfill. Only question text is used for this contamination check; no benchmark answers or outcomes enter selection.

The audit flagged 11 potential 13-word matches among the full 15K, with zero exact prompt matches; three flags occurred in the longest 3K. After removing those three, the curriculum has **2,997 originals** and **40,212,735 supervised tokens per epoch**. These are conservative textual overlap flags, not proof of semantic leakage. Whitespace matching has limited Chinese coverage and does not replace the earlier character-based audit.

Initialize from the verified final original model, with a **new optimizer**, LR 5e-6, warmup ratio .03, cosine scheduling, seed 42 and two epochs. Request **two B200s for six hours**, batch one per rank / accumulation four / effective batch eight. The installed Transformers 4.55 CPU Trainer and distributed sampler give **750 updates**, 5,994 dataset-row presentations and **5,996 padded sampled presentations**. Startup must confirm the GPU dataloader schedule; the padding is not hidden or counted as new unique data.

Implementation: `research/stage2_plan_20261019.py` and `research/two_plan_contamination_20261019.py`. Native checks preserve original system, input, target, prompt masks, complete response and EOS. The prepared source remains unchanged and the subset does not replace the original 15K dataset.

The final parent's fixed **200-training-row** teacher-forced shifted/masked token-accuracy diagnostic completed at **83.8123%**. The document's <95% condition is retained as an explicitly experimental admission gate; it is not established as a universal convergence test. The implementation uses the native converter and masks, including nonempty input, rather than the document's illustrative manual serialization.

Only checkpoints 5,600 and 5,625 remain from the original run. Missing epoch-one / step-1,875 and step-3,750 checkpoints are not reconstructed, substituted or used for early/final interpolation. The optional LR 1e-5 restart and C5 merge experiment are excluded.

## Frozen development and terminal evaluation

A new **200-question held-out panel** is selected from the original validation source before outcomes exist. It excludes the entire 83,037-prompt training pool, prior development questions, and exact public benchmark-question identities. Original grader-supported reference answers are preserved. The earlier 100-row development panel remains unchanged.

Run one matched eight-sample control on the original final parent, then one on the completed Stage 2 candidate only if the parent qualifies. Both must have **at most 80 truncated samples out of 1,600**; Stage 2 must solve **at least two additional questions out of 200**. An unqualified parent leaves no qualified control and fails closed. No repeated screen, seed selection or invented winner is allowed. The development helper uses the pinned evaluation environment.

A final candidate requires complete finite training, model completeness and every final tensor finite. Stage 2 also requires the actual passed development gate before terminal submission. Each terminal evaluation retains the unchanged internal all-nine protocol: one B200, TP1, temperature .6, top-p .95, eight samples per problem, maximum 32,768 generated tokens, batch 1,024, maximum sequences 128 and GPU memory utilization .85. There are no simultaneous per-benchmark engine changes or altered historical gates.

## Bound compute and operational checks

| Envelope | Physical GPU-hours |
| --- | ---: |
| Assistance training: four B200s × six hours, one allocation | 24 |
| Stage 2 training: two B200s × six hours, one allocation | 12 |
| Two unchanged single-B200 terminal evaluations × eight hours | 16 |
| Final-parent token diagnostic plus two development screens | 5 |
| Additional authorized maximum | **57** |
| Prior committed hours plus new maximum | **97** |

Distinct plans may overlap. One training allocation per plan, one terminal evaluation per plan, and one allocation per auxiliary purpose are permitted. No automatic retry or requeue is enabled. Actual scheduler IDs are reserved under one shared lock immediately after successful submission; uncertain submission outcomes block further submission until reconciliation. Failed allocations stay charged under the existing rules. Reserved evaluation time and the October 20, 11:59 p.m. Eastern deadline are checked before admission; a queue estimate is not a start or completion guarantee.

Both draft YAMLs intentionally keep `do_train: false` and non-overwriting outputs. The guarded runtime produces an effective enabled configuration only after source, preparation, authorization and allocation checks. It never invents an approval record. Four-rank checkpoint checks require all four RNG files plus model, optimizer, scheduler and finite per-update history. Before-start operational GPU-visibility changes were CPU-verified with the same submitted IDs; original scripts and receipts were preserved privately. Guards inspect assigned PyTorch CUDA devices rather than host-wide NVML line counts.

**ARC execution correction:** after a staff warning, the remaining intensive login-node Python audit was stopped. The unchanged preparation now runs in a CPU compute allocation. All heavy audits, native checks, tensor scans, data/model hashing, tests and guarded submission validation must execute on compute nodes. Login nodes are used only for lightweight scheduler, file and log operations. This correction does not authorize additional GPU hours.

Validation: 28 actual pinned ARC execution/convergence CPU checks passed, including real FP32/BF16 four-rank token-normalized gradients. At public-sync preparation, 36 local execution/convergence/development tests also passed, plus shell syntax and whitespace checks. CPU checks do not prove GPU fit, throughput or accuracy. The strict objective remains one fixed model passing all nine gates; no such success is claimed.
