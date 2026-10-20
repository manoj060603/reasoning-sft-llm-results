# Evaluation runtime contingency — October 19, 2026

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
