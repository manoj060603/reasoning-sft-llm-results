# Fresh reasoning SFT experiment

**Cross-fitted Transfer Gain with Coverage** selects 15,000 original AceReason
examples and trains a new model from the original `Qwen/Qwen2.5-3B-Instruct`.
It compares that model with the original and a matched 15,000-example random
control. Earlier team models are not used for initialization or probe scoring.

This project addresses the [Virginia Tech reasoning SFT assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM),
due **October 20, 2026, at 11:59 p.m. Eastern**.

## Verified progress

Last successful scheduler check: **October 19, 2026, at 9:42 p.m. Eastern**.
Subsequent ARC SSH and OnDemand connections timed out. This is an access failure;
it does not establish that the running cluster jobs failed or stopped.
The Cisco VT Traffic VPN session expired; its reconnect now awaits Virginia
Tech sign-in and Duo in Brave.

| Stage | Verified state |
|---|---|
| Source preparation and both probes | Complete; 30,000 opposite-fold candidate scores |
| New-method selection | Complete; 15,000 unchanged examples, zero training truncation |
| Random-control full SFT | Complete; 1,875 updates, 15,000 unique examples, finite final FP32 weights |
| New-method full SFT | Running at last check; 1,461 / 1,875 updates |
| Original benchmark evaluation | All nine complete, including the supplementary AIME25 task |
| Random-control evaluation | Running at last check; full benchmark suite incomplete |
| New-method evaluation | Queued behind its training; no benchmark results verified yet |
| Final comparisons and artifact publication | Pending |

The original model's **official-eight macro is 29.30% pass@1 and 52.86% pass@8**.
These are fresh matched-protocol baseline measurements. Neither a complete
trained-model comparison nor a state-of-the-art result has been established.
Missing results are pending, not zero.

The control's first benchmark generated unusually long responses. A
[16-hour evaluation continuation](deadline/operations/evaluation-runtime-contingency.md)
is prepared with the same evaluation settings, but has **not been admitted**.
It requires restored ARC access, native budget checks, verified GPU accounting
and an updated final-results dependency. Deadline completion remains uncertain.

## Read the experiment

- [Method, controls and execution workflow](deadline/README.md)
- [Report draft and assignment answers](deadline/FINAL_REPORT.md)
- [Measured result audit](deadline/report/results.md)
- [Requirement-by-requirement checklist](deadline/assignment_checklist.json)
- [Frozen scientific plan](deadline/experiment_plan.json)

The method ranks response loss reductions measured by a probe trained on the
opposite question fold, then combines positive-gain leaders with seeded topic
and length coverage. Both final models use the same fresh base, full-parameter
training recipe, example count, update count and evaluation protocol. The report
discloses differences in eligibility, truncation and supervised token counts;
the comparison tests the complete selection method rather than an isolated
transfer-gain ablation.

## Reproducibility and delivery

The [fresh source snapshot](https://github.com/manoj060603/reasoning-sft-llm-results/commit/45e4d38784f1fdad4c033031f3b2d4c32c6c94b2)
is public and verified against the entire local Git tree; see the
[publication receipt](deadline/report/source_publication_receipt.json).

Heavy preparation, training, evaluation and native checks run in ARC compute
allocations. Exact source revisions, selected IDs, actual example consumption,
model hashes and evaluation manifests are retained. The original reservation
plan was 42 GPU-hours; the fresh experiment's consumption cap remains 48 hours.
Any unused-reservation reconciliation preserves historical requests and charges
the full reservation for every open or unverified allocation.

The final deliverables require matched original/random/new-method scores,
uncertainty estimates, the completed report, two public model repositories and
two public dataset repositories with verified contents. Source availability,
queued jobs and upload code do not establish completed publication. Course
submission remains a human action.

The code license is in [LICENSE](LICENSE). Published derivative model weights
must carry the [pinned upstream Qwen Research License](deadline/licenses/Qwen2.5-3B-Instruct.LICENSE),
attribution and modification notices. Dataset provenance and upstream license
details are recorded in the report and generated dataset cards.

## Preserved earlier work

[UPSTREAM_PROGRESS_README.md](UPSTREAM_PROGRESS_README.md) preserves the previous
README, earlier team progress and the assignment tutorial. Its model scores and
experiment status refer to earlier work. Other existing root-level reports are
also historical unless explicitly linked as fresh experiment evidence above.
