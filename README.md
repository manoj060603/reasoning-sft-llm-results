# Fresh reasoning SFT experiment

**Cross-fitted Transfer Gain with Coverage** selects 15,000 original AceReason
examples and trains a new model from the original `Qwen/Qwen2.5-3B-Instruct`.
It compares that model with the original and a matched 15,000-example random
control. Earlier team models are not used for initialization or probe scoring.

This project addresses the [Virginia Tech reasoning SFT assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM),
due **October 20, 2026, at 11:59 p.m. Eastern**.

## Verified progress

Verified snapshot: **October 20, 2026, at 1:25 a.m. Eastern / 05:25 UTC**.
VPN and ARC access are restored. Both final training runs are complete. The
random control has five audited completed tasks and the advanced model has
four, each count including supplementary AIME25.

| Stage | Verified state |
|---|---|
| Source preparation and both probes | Complete; 30,000 opposite-fold candidate scores |
| New-method selection | Complete; 15,000 unchanged examples, zero training truncation |
| Both final full-SFT models | Complete; 1,875 updates each, 15,000 unique examples used once, 434 finite FP32 tensors per model |
| Original benchmark evaluation | All nine complete, including supplementary AIME25 |
| Random-control evaluation | 5/9 complete: AMC, AIME24, MATH, CN_MATH_2024 and supplementary AIME25; four tasks pending |
| New-method evaluation | 4/9 complete: AMC, AIME24, MATH and supplementary AIME25; five tasks pending |
| Evaluation continuations | Admitted: random `1210309` (16 hours), advanced `1210310` (12 hours), after their respective parent evaluations terminate |
| Final comparisons and Hugging Face publication | Pending; CPU closeout `1208135` depends on both continuations succeeding; publisher `1209745` depends on closeout |

The original model's **official-eight macro is 29.30% pass@1 and 52.86% pass@8**.
The [partial raw-result audit](deadline/report/partial-20261020T0525Z/results.md)
reports the following percentages; every completed task has eight responses per
problem and verified identities. Protocol, benchmark data and hardware agree;
the full matched comparison remains incomplete.

| Task with measured results | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
|---|---:|---:|---:|
| AMC 2023 | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | Pending |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% |

Both trained models trail the original on AMC, AIME24 and MATH and improve on
supplementary AIME25. On MATH, advanced pass@1 is **51.225% versus 49.725%** for
random, a **+1.50 percentage-point** difference; pass@8 ties at **78.20%**.
Both remain below the original's 65.90% / 83.80%. MATH truncations are
1,235/4,000 advanced responses (30.875%) versus 1,508/4,000 random (37.7%).
The random CN_MATH_2024 result is also below the original, with 215/240
responses truncated; the advanced CN result is not yet in this audit.

Advanced-versus-random findings remain mixed across tasks and metrics. No
paired confidence intervals, significance claim or overall winner is available.
Both trained-model full macros and uncertainty remain pending; missing results
are not zero. The selection method, models and evaluation protocol are unchanged
in response to these scores.

The [runtime continuation plan](deadline/operations/evaluation-runtime-contingency.md)
retains the original evaluation settings and saved results. Native budget and
pipeline checks passed **30 tests with no failures, errors or skips** in the
running evaluation allocation. The admitted reservation/accounting bound is
**47.0167 of 48 GPU-hours**, combining verified closed allocations with full
open reservations; this is **not actual elapsed GPU usage**. Historical
admitted requests total 70 hours; the 88-hour all-attempt history additionally
includes 18 hours verified never admitted. These figures describe distinct
accounting quantities. Continuation success and deadline completion remain
uncertain.

## Read the experiment

- [Method, controls and execution workflow](deadline/README.md)
- [Report draft and assignment answers](deadline/FINAL_REPORT.md)
- [Current partial result audit](deadline/report/partial-20261020T0525Z/results.md)
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
is public and verified against its corresponding local source tree; see the
[publication receipt](deadline/report/source_publication_receipt.json).
The [latest verified source update](https://github.com/manoj060603/reasoning-sft-llm-results/commit/d00576f0125c009ed28db557a4a0e4d8b3aff9c6)
matches local commit `9e6477fac49c0187e7d02a5027acaa5f50d70b07` and entire tree
`5dabfa8d4010779ec3eff4c8690836e1ad1874ac`. It includes the continuation wrapper and
the earlier training/partial-result evidence. The new 05:25 UTC partial audit
and this documentation update are prepared for the next source publication.
Source publication does not establish publication of model or dataset artifacts.

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
