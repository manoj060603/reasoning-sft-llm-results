# Fresh experiment: Cross-fitted Transfer Gain with Coverage

This is a new experiment initialized from `Qwen/Qwen2.5-3B-Instruct` revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. It does not initialize from any of
the team's previous trained models. The existing repository supplies utility
code and the assignment's grader; previous reported scores are not results of
this experiment.

## Method

The proposed selector measures transfer rather than ranking raw teacher loss.
After reserving source-only held-out questions and excluding public benchmark
question overlap, split prompt identities into two deterministic folds. Train
one rank-eight LoRA probe on 512 complete original examples from each fold.
Each probe performs 64 updates, with batch eight and learning rate `5e-5`.

Score a frozen candidate reservoir only with the probe trained on the opposite
fold. For each original target, compute response-token-mean cross entropy:

`gain_i = (CE_base(i) - CE_opposite_probe(i)) / max(CE_base(i), 1e-8)`.

Within topic clusters and response-length cells, fill at most half the quota
from positive-gain leaders, and fill the rest by seeded coverage. Export exactly
15,000 unique original prompt/response pairs. The selected solutions must fit
the 8K training context in full, including the native supervised end marker.
Probe training examples and every held-out source question are excluded from
both final training arms. The final model starts from the original base, not a
probe checkpoint.

The motivation combines relative loss reduction ([DavIR](https://arxiv.org/abs/2310.13008))
and learning-trajectory selection ([S2L](https://arxiv.org/abs/2403.07384)).
Cross-fitting and coverage constraints are our proposed experimental synthesis.
Global novelty and benchmark superiority have not been established. Relative
teacher-forced loss reduction is a selection proxy, not proof of mathematical
correctness or causal usefulness.

## Controlled experiment

Compare the original model, one uniformly sampled broad prepared-pool 15K control,
and the new selected 15K model. Both trained arms use full-parameter SFT, one
pass / 1,875 updates, global batch eight, learning rate `2e-6`, cosine scheduling,
3% warmup, and the same system message and response masks. Trainable parameters
and AdamW moments use FP32; forward computation uses BF16 autocasting.

Original dataset exports are unchanged in both arms. The random control samples
the broad prepared source pool after shared deduplication, benchmark exclusions,
source-heldout reservations, question-capacity bounds and probe exclusions.
It is not restricted to complete short traces or boxed/low-repetition solutions.
Both arms use the same 8K training cutoff. Random tokenized targets that exceed
it are prefix-truncated without adding an invented EOS; counts and missing-EOS
rates are recorded. No selected training row is silently dropped. Advanced
selection requires the original target to fit completely. The random control
is uniform over the declared prepared pool, not the untouched original 100K.
The comparison tests the whole eligibility-plus-transfer-plus-coverage method.
Separate eligible-random and coverage-only ablations would be needed to isolate
the contribution of transfer gain alone. This control-population correction was
made before any probe training, score computation or benchmark evaluation.
The initial 4K capacity audit found only 15,938 eligible complete targets,
insufficient for 15K final rows plus 1,024 disjoint probe rows. Both arms were
therefore moved to 8K before any GPU training or scoring. The failed CPU audit
produced no trained model or benchmark accuracy.

Final evaluation uses eight samples, temperature 0.6, top-p 0.95, maximum 32,768
generated tokens and native 32,768-token total context. Report both pass@1 and
pass@8, each benchmark, the official-eight macro, and the supplementary
historical-seven macro separately. No benchmark outcomes enter selection.

## Execution and evidence

The frozen design is in [experiment_plan.json](experiment_plan.json).
The configured ARC deployment is
`/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019`.

1. `prepare_transfer.slurm` downloads the pinned source and prepares native tokens.
2. `finalize_preparation.slurm` runs native tests, verifies source/model identities,
   applies supplementary AIME25 question exclusions, freezes the random control,
   and submits the bounded experiment.
3. `gpu_probe.slurm` trains the two probes and scores opposite folds.
4. `select_transfer.slurm` validates complete score artifacts and exports 15K pairs.
5. `gpu_train.slurm` trains fresh random and advanced models.
6. `gpu_eval.slurm` evaluates each complete model with the same protocol.
7. `finalize_results.slurm` waits for all three evaluations, then creates the
   matched audit, paired uncertainty, and verified publication metadata.
8. `publish_huggingface.slurm` waits for final checks and the CPU publication
   preflight. It uploads the two models and two unchanged datasets under
   `adversarialbandit`, then checks anonymous access and remote file hashes.

Execution overlap keeps the scientific design fixed. `bootstrap_evaluation.slurm`
runs native checks and starts the original-model evaluation while source token
caches are still being prepared. A locked benchmark-readiness receipt prevents
either preparation stage from rewriting those benchmark files. The finalizer
seals all source exclusions, probe/candidate IDs, transfer metadata, and random
IDs in `probe_inputs_ready.json` before starting the probe array. It then exports
the random token cache using eight ordered CPU workers. The complete finalization
receipt still gates random training and downstream selection. Repeated admission
of an already submitted role reuses its ledger job rather than submitting twice.
Parallel export is checked for byte equivalence against serial native tokenization.
The later advanced export also uses up to eight allocated CPU workers. GPU jobs
use the account-authorized `owl_b200_normal_short` QoS, which permits their
unchanged six-hour requests and has higher priority than the partition default.
The final training and evaluation jobs may be queued in advance with `afterok`
dependencies on their producers. Each job still checks the required completion
receipt before loading a model; producer-side admission reuses the recorded job.

Submissions are recorded under a shared lock in `job_ledger.json`. Ambiguous
submissions are not retried automatically. The planned conservative GPU budget
is 42 hours, with a hard fresh-experiment cap of 48 hours and explicit deadline
reserves. Heavy audits execute only inside compute allocations.

The [evaluation runtime contingency](operations/evaluation-runtime-contingency.md)
records a prepared, unchanged-protocol 16-hour continuation after unusually long
random-control generations. Its admission can reclaim only verified terminal
unused reservations, with exact scheduler evidence retained by
`reconcile_gpu_budget.py`. Historical requested hours remain visible separately
from the conservative actual-plus-outstanding bound; the 48-hour cap is unchanged.
The continuation requires native guard checks and fresh ARC access before it
can be admitted. Its existence in source does not establish a queued job.

Runtime evidence stays in `artifacts/`, `runs/`, `logs/`, `results/`, and `report/`
on ARC. A completion receipt is written only after the corresponding stage has
passed its checks. `report/results.md` leaves missing results blank. No benchmark
accuracy for this new method is claimed until those actual evaluations finish.

After the three evaluations complete, run the following in a CPU compute
allocation to create uncertainty estimates and publication metadata:

```bash
python -m deadline.compare_uncertainty \
  --original deadline/results/original --random deadline/results/random \
  --advanced deadline/results/advanced --output deadline/report/uncertainty.json
python -m deadline.package_deliverables \
  --root deadline --output deadline/publication-metadata
```

Uncertainty uses 10,000 paired bootstrap draws of normalized-question clusters,
keeping duplicate benchmark questions together. Within each benchmark, scores
retain the official record weights; complete macros weight benchmarks equally.
These intervals condition on the fixed models and eight recorded responses and
do not estimate training-seed or additional generation-sampling variability.
Partial or aggregate-only evidence cannot produce a complete uncertainty macro.

The packager rechecks source, selection, training, model, and evaluation hashes
before staging model cards, dataset cards, and a file-reference manifest. It
does not copy model weights, upload artifacts, or infer publication success.
Both commands refuse to replace an existing output; retain earlier evidence
and choose a new output path when producing a revised artifact.

The separate publisher requires a CPU allocation and the explicit expected
account. Its normal invocation is:

```bash
python -m deadline.publish_huggingface --expected-owner adversarialbandit
```

Publication preserves the evaluated FP32 weight bytes and complete original
dataset exports. It rejects conflicting destinations, binds every uploaded
file to an immutable publication identity, and supports resuming matching
partial uploads. The models include the pinned upstream Qwen Research License,
attribution notice, and modification notice. A successful upload call alone
does not establish completion: `hf/publication/publication_complete.json`
requires all four public repositories and their exact file-hash proofs.
No GitHub publication or course submission is inferred from that receipt.

The [report draft](FINAL_REPORT.md) is reconciled with measured results and
verified publication links before final delivery.
