# Current plans and runs — October 19, 2026

Snapshot: **October 19, 1:43 p.m. Eastern**. These are observed states, not a live dashboard. Deadline: **October 20, 11:59 p.m. Eastern**. [Machine-readable sanitized snapshot](research/current_runs_public_20261019.json).

## Completed original training and ongoing evaluation

The original full-length model completed **5,625 finite updates / three epochs**, with **15,000 unique prompts, 45,000 example presentations**, and all saved final model tensors CPU-verified finite. Training job 1203514 completed. Evaluation job **1206228 is running** under the unchanged matched single-B200 nine-benchmark protocol.

| Benchmark | Solved at pass@8 | Frozen required gate | Result |
| --- | ---: | ---: | --- |
| AMC | 24/40 | 33/40 | Failed |
| AIME 2025 | 5/30 | 8/30 | Failed |
| AIME 2024 | 5/30 | 8/30 | Failed |
| MATH | 428/500 | 451/500 | Failed |
| CN Math 2024 | 11/30 | 14/30 | Failed |
| Kaoyan | 135/199 | 138/199 | Failed |

Minerva, OlympiadBench, and GPQA are unfinished. The fixed candidate has already failed the strict all-nine objective. No complete-suite macro, final gate count, or causal explanation is inferred. The evaluation continues for an honest complete report. All six completed artifacts have been checked for complete problem identities and eight response/correctness/finish/token entries per question; Kaoyan was checked locally to keep computation off login nodes.

## Two new plans

- **Assistance plus coverage:** new source-only selection of exactly 15K complete original responses, with assistance utility capped at half each quota cell and deterministic coverage filling the remainder. CPU preparation job **1207446 is running on a compute node**. Its full 56,673-eligible-row native audit is incomplete, so the **four-B200 training job has not been submitted**. The CPU job alone owns the subsequent guarded submission. Training envelope: one six-hour four-B200 allocation, batch one / accumulation two / effective batch eight, three epochs / 5,625 updates.
- **Longest-example Stage 2:** 2,997 complete originals after three conservative overlap exclusions without backfill. Initialize from the verified final original model with a new optimizer; two epochs, LR 5e-6, effective batch eight. The pinned native CPU schedule is 750 updates, 5,994 dataset presentations and 5,996 padded distributed samples. **Job 1207378 is pending resources**, requesting two B200s for six hours. Its provisional scheduler start was 5:55 p.m. Eastern; this is not a guarantee. GPU dataloader and finite-update checks remain startup requirements.

The fixed 200-training-row native token-accuracy diagnostic **1207375 completed**: **1,109,712 / 1,324,044 = 83.8123%**, passing the predeclared experimental <95% admission rule for Stage 2. This threshold is not a validated convergence guarantee. The separate source-only **200-question held-out parent development control 1207417 is running**. Stage 2 requires complete finite training and a qualified development comparison before any final all-nine evaluation. No missing early checkpoint is substituted, and no optional hotter restart or C5 merge is included.

## Resources, safety of execution, and publication scope

The approved cap carries **40 prior committed GPU-hours + 57 additional = 97 total**. Extra envelopes: assistance training 24, Stage 2 training 12, two terminal evaluations 16, and diagnostic/development allocations 5. Distinct plans may overlap; duplicate jobs for the same candidate and unapproved repeats are forbidden. GPU allocations remain charged under the existing conservative rules even if they end early.

After an ARC staff warning, the remaining intensive login-node audit was terminated and replaced by CPU-only compute job 1207446. Heavy audits, native checks, model/data hashing, tests and guarded submission validation must run inside compute allocations. Login-node work is limited to lightweight scheduler, file and log operations.

This public snapshot includes code and sanitized status only. Original proposal documents, raw datasets and generations, exact question/source identifiers, authorization records, detailed provenance receipts and raw logs remain outside this public sync. The separate ordinary-SFT submission fallback is preserved. No Hugging Face upload is authorized by this GitHub request.

[Implemented plan details](TWO_PLAN_SETUP_20261019.md). Earlier dated entries in other reports are historical; this snapshot supersedes their live-state claims. Queue times, six-hour training fit and strict 9/9 success remain unproven.
