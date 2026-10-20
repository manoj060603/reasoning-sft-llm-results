# Fresh reasoning SFT experiment

**Cross-fitted Transfer Gain with Coverage** selects 15,000 original AceReason
examples and trains a new model from the original `Qwen/Qwen2.5-3B-Instruct`.
It compares that model with the original and a matched 15,000-example random
control. Earlier team models are not used for initialization or probe scoring.

This project addresses the [Virginia Tech reasoning SFT assignment](https://github.com/reds-lab/Project-Reasoning-SFT-LLM),
due **October 20, 2026, at 11:59 p.m. Eastern**.

## Verified progress

Verified snapshot: **October 20, 2026, at 6:10 a.m. Eastern / 10:10 UTC**.
Both final training runs are complete. Each trained model has seven audited
completed tasks, including supplementary AIME25: **7/9 total, 6/8 supplied
tasks**. Only OlympiadBench and GPQA remain pending for both models.

| Stage | Verified state |
|---|---|
| Source preparation and both probes | Complete; 30,000 opposite-fold candidate scores |
| New-method selection | Complete; 15,000 unchanged examples, zero training truncation |
| Both final full-SFT models | Complete; 1,875 updates each, 15,000 unique examples used once, 434 finite FP32 tensors per model |
| Original benchmark evaluation | All nine complete, including supplementary AIME25 |
| Random-control evaluation | 7/9 complete: AMC, AIME24, MATH, CN_MATH_2024, Kaoyan, Minerva and supplementary AIME25 |
| New-method evaluation | 7/9 complete: AMC, AIME24, MATH, CN_MATH_2024, Kaoyan, Minerva and supplementary AIME25 |
| Evaluation continuations | Running: random `1210309` (16-hour allocation), advanced `1210310` (12-hour allocation); both initial six-hour parents timed out |
| Final comparisons and Hugging Face publication | Pending; CPU closeout `1208135` depends on both continuations succeeding; publisher `1209745` depends on closeout |

The original model's **official-eight macro is 29.30% pass@1 and 52.86% pass@8**.
The [partial raw-result audit](deadline/report/partial-20261020T1010Z/results.md)
reports the following percentages; every completed task has eight responses per
problem and verified identities. Protocol, benchmark data and hardware agree;
the full matched comparison remains incomplete.

| Task with measured results | Original pass@1 / pass@8 | Random pass@1 / pass@8 | Advanced pass@1 / pass@8 |
|---|---:|---:|---:|
| AMC 2023 | 42.50% / 70.00% | 22.50% / 47.50% | 25.00% / 45.00% |
| AIME24 | 7.92% / 23.33% | 0.42% / 3.33% | 2.50% / 13.33% |
| MATH | 65.90% / 83.80% | 49.73% / 78.20% | 51.23% / 78.20% |
| CN_MATH_2024 | 10.00% / 26.67% | 0.83% / 6.67% | 5.00% / 23.33% |
| Kaoyan | 21.11% / 49.75% | 14.07% / 41.71% | 15.45% / 47.74% |
| Minerva | 30.33% / 47.79% | 15.95% / 36.40% | 18.11% / 41.54% |
| AIME25, supplementary | 0.83% / 3.33% | 2.92% / 16.67% | 1.67% / 13.33% |

Both trained models trail the original on all six currently completed supplied
tasks and improve on supplementary AIME25. Advanced exceeds random on CN,
Kaoyan and Minerva in both metrics. On MATH, advanced pass@1 is **51.225% versus 49.725%**
for random, a **+1.50 percentage-point** difference; pass@8 ties at **78.20%**.
Both remain below the original's 65.90% / 83.80%. Advanced also exceeds random
on AIME24, has higher pass@1 but lower pass@8 on AMC, and trails random on
AIME25. Findings remain mixed; no overall winner or significance is established.

MATH truncations are 1,235/4,000 advanced responses (30.875%) versus
1,508/4,000 random (37.7%). CN has 159/240 advanced versus 215/240 random
truncations; Kaoyan has 667/1,592 advanced versus 1,009/1,592 random. Minerva
has 1,088/2,176 advanced responses truncated (50%), versus 1,269/2,176 random.
These finish-reason counts do not establish what caused the accuracy differences.
Full trained-model macros and paired uncertainty remain pending. Missing
results are not zero. The selection method, models and evaluation protocol
are unchanged in response to these scores.

The [runtime continuation record](deadline/operations/evaluation-resume-20261020.json)
documents retaining six completed benchmarks and fully regenerating the unsaved
Minerva batch under the frozen settings. No interrupted partial response enters
the scores. Interrupted work is included in GPU accounting; its exact unsaved
generated-token total is unknown. Both continuations are now running and the
new audit verifies completed regenerated Minerva results.

Native budget/pipeline checks previously passed **30 tests with no failures,
errors or skips** in a compute allocation. Both initial evaluation parents
ended at their time limit after 21,616 and 21,608 seconds; each is conservatively
charged 21,660 GPU-seconds. The [random accounting proof](deadline/operations/gpu-budget-reconciliation-d155f13d26ffca69.json)
and [advanced proof](deadline/operations/gpu-budget-reconciliation-637c6b2ed64206b8.json)
therefore update the terminal-plus-open-reservation bound to **47.05 of 48
GPU-hours (169,380 seconds)**. This is **not actual elapsed GPU usage**.
Historical admitted requests remain 70 hours; the 88-hour all-attempt history
includes 18 hours verified never admitted. Final resource use, continuation
success, comparisons and publication remain pending.

## Read the experiment

- [Method, controls and execution workflow](deadline/README.md)
- [Report draft and assignment answers](deadline/FINAL_REPORT.md)
- [Current partial result audit](deadline/report/partial-20261020T1010Z/results.md)
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
A [verified source snapshot](https://github.com/manoj060603/reasoning-sft-llm-results/commit/c8aea2fd0d66ff9d3ef46461984d08c10bf57433)
matches local commit `90b861fa847f395b2ca5cc580eb89c8223c990c8` and entire tree
`2c2761cb009d9ddd951e22811ceff46a8c2ac70a`. This records verified source history. The new
10:10 UTC partial audit and this documentation update are prepared for a
subsequent source publication. Source publication does not establish
publication of model or dataset artifacts.

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
