# Evaluation runtime contingency — October 19, 2026

**Current status, October 20 at 03:26 UTC:** both continuations are admitted.
Random job **1210309** reserves 16 GPU-hours after parent 1208059 terminates;
advanced job **1210310** reserves 12 GPU-hours after parent 1208060 terminates.
The [native guard receipt](runtime-contingency-native-qa.json) records 30 passing
checks and identical evaluation argument tails. The [admission receipt](evaluation-continuation-admission.json)
records the 47.0167-hour conservative bound and verified downstream dependencies.
The sections below preserve the preparation history.

The first random-control AMC batch produced 6,634,502 generated tokens across
320 responses; 190 responses (59.375%) stopped at the length limit. Generation
took 1,111 seconds and the first saved result arrived about 1,242 seconds after
job start. These runtime measurements make the six-hour evaluation request a
substantial risk. They do not justify changing the model, selected data,
generation limit, sample count, prompts, grader, benchmark population or seeds.
Accuracy outcomes are not used to choose a replacement scientific method.

The prepared operational response is a 16-hour continuation with the unchanged
evaluator and output directory, dependent on termination (`afterany`) of the
existing six-hour random evaluation. Complete saved benchmark batches are
reused by the existing evaluator; an unfinished batch must be generated again.
There is no concurrent writer to that role's result directory. The continuation
checks for at least 19 hours before the assignment deadline on both admission
and actual start, reserving its full 16-hour request, one hour of CPU result
checks, and two hours for Hugging Face publication. The latest permissible start
is October 20 at 04:59 Eastern. Queue admission and actual start are uncertain.

The original frozen plan reserved 42 GPU-hours. Historical requested hours are
preserved exactly. To stay under the unchanged **48 GPU-hour consumption cap**,
`gpu_budget.py` charges each verified terminal allocation by its generic
allocated GPU count times elapsed seconds rounded up to a minute. Every open,
missing, ambiguous or unverified allocation retains its entire reservation.
Generic and GPU-type TRES counts are not added together. Allocation accounting
includes duplicate/restarted records and rejects ambiguous extra allocations;
all scheduler states are included in the active-identity check. Reconciliation
retains exact accounting text, timestamps, immutable original reservations and
an append-only evidence audit under the existing ledger lock.

The last verified terminal allocations were:

| Allocation | Elapsed seconds | Conservative GPU-hours |
|---|---:|---:|
| Original evaluation 1207982 | 3,564 | 1.0000 |
| Probe task 1208021_0 / raw1208041 | 5,189 | 1.4500 |
| Probe task 1208021_1 / raw1208021 | 4,299 | 1.2000 |
| Random training 1208033 | 6,734 | 1.8833 |

These total 5.5333 hours. Keeping the full 18 hours reserved for the remaining
three GPU roles gives a 23.5333-hour bound. A new 16-hour continuation would
increase the bound to **39.5333 hours**, below 48, while cumulative admitted
historical requests would become **58 hours**. Those two quantities must not
be conflated. Final consumed hours must be reported from completed allocation
records. Any later continuation must independently pass this same locked cap
and deadline check; this note does not authorize an unbounded extension.

**Prepared, not yet admitted:** ARC SSH access timed out through Owl and both
TinkerCliffs login hosts during preparation. No receipt reconciliation, scheduler
submission, or final-result dependency update is inferred from this document.
The new guard tests must pass inside a compute allocation before admission.
After successful admission, the final CPU results job must depend on successful
completion of the continuation as well as the original and advanced results.
The four-artifact publisher remains gated on successful final CPU checks.

## Access-restoration update: paired continuation requests

ARC access has been restored. Fresh scheduler accounting reports advanced
training **COMPLETED** after **5,294 seconds**. Its one-GPU allocation rounds
up to 89 minutes, or 1.4833 GPU-hours. Together with the four terminal
allocations listed above, all verified closed allocations total **421 minutes
(7.0167 GPU-hours)**. Both existing six-hour trained-model evaluations are
still running and retain their full reservations. This supersedes the earlier
three-open-role estimate; it does not infer complete evaluation results.

The revised operational plan prepares a **16-hour random continuation** and a
**12-hour advanced continuation**, each after termination of its own existing
evaluation. The conservative bound is:

| Budget component | GPU-hours |
|---|---:|
| Verified closed allocations, rounded up separately | 7.0167 |
| Two existing six-hour evaluation parents, full reservations | 12.0000 |
| Proposed random continuation, one GPU for 16 hours | 16.0000 |
| Proposed advanced continuation, one GPU for 12 hours | 12.0000 |
| **Allocated-plus-open bound** | **47.0167 / 48.0000** |

Historical admitted requested hours would be **42 + 16 + 12 = 70 GPU-hours** after both
admissions. That cumulative request history remains distinct from the
conservative consumption bound; it is preserved rather than rewritten.

The advanced wrapper is `deadline/gpu_eval_extended_12h.slurm`. It reserves
15 hours before the deadline on actual start: 12 hours for evaluation and
three hours for final CPU checks and publication. Its admission must also
require at least 15 hours. The random wrapper retains its 16-hour limit and
19-hour reserve. Both reuse the original role-specific output directories
and unchanged scientific protocol, including all benchmarks, eight samples,
prompts, seed, generation limits, grader and model identity.

**Prepared, not yet admitted:** no continuation job IDs or scheduler submissions
are recorded by this update. Admission still requires native guard checks,
locked ledger reconciliation against exact terminal accounting and fresh
expanded queue identities, and independent cap and deadline checks. Final CPU
closeout and publication dependencies must be updated to the successful
continuation jobs. Model weights, selected datasets and scientific settings remain unchanged.

## Verified admission and downstream repair

Both jobs were admitted after native compute-allocation guard checks and locked
terminal-accounting reconciliation. Their actual scheduler records confirm one
B200, eight CPU cores, 64GB RAM, the authorized short QoS and respective
16-hour/12-hour limits. Each has `afterany` on its own existing parent, so there
is no concurrent writer to a role's result directory. No model, benchmark,
sample-count, context, generation, seed or grader setting was changed.

Final CPU job 1208135 now requires successful jobs 1210309 and 1210310. Original
evaluation 1207982 had already completed successfully and its fulfilled scheduler
dependency was not re-added. Publisher 1209745 still requires successful final
CPU checks. The original requested submission commands remain in the ledger;
the dependency update is recorded separately with its previous value.

The 70-hour admitted request history is separate from two earlier verified
unadmitted attempts totaling 18 requested hours. All-attempt request history is
therefore 88 hours, with zero allocated GPU time charged for those proven
unadmitted attempts. The hard consumption cap remains 48 hours; completed-use
ceilings plus all open reservations total 47.0167 hours at admission. Final
consumption remains to be reconciled after all allocations terminate.

Both final training runs are complete. At the partial native result audit,
each trained model had three saved benchmarks; complete comparisons and
Hugging Face publication remain pending. Queue/start uncertainty and the
deadline guards still apply.
