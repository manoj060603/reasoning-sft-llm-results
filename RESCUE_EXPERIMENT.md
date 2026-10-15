# Capacity-matched rescue experiment

The first 15K route-ASFT run failed its practical objective: completed public
benchmarks showed far more length-limited generations than the matched base.
The run was stopped during MATH evaluation, and its partial scores remain
diagnostic rather than a complete leaderboard result.

## Measured failure

| Benchmark | Base pass@8 | Route-ASFT pass@8 | Base truncations | Route-ASFT truncations |
|---|---:|---:|---:|---:|
| AMC | 77.50% | 67.50% | 2/320 | 52/320 |
| AIME25 | 13.33% | 13.33% | 4/240 | 66/240 |
| AIME24 | 23.33% | 20.00% | 10/240 | 83/240 |

The 512-example pilot used a 4,096-token complete-sequence cutoff and about
1,783 response tokens per training example. The failed full run changed that
regime to an 8,192-token cutoff and about 4,191 response tokens per example,
while retaining a 5e-5 learning rate. That extrapolation was not validated.

## Replacement

The replacement is implemented in `scripts/arc_rescue_prepare.sh` and
`scripts/arc_rescue.sh`:

- 15,000 unique route-diverse examples under a 4,096 complete-sequence limit.
- A fixed capacity mixture: at least 10,000 examples under 3,040 tokens and
  5,000 additional examples under 4,064 tokens.
- Measured dataset: 36,706,060 response tokens, mean 2,447.1, versus 62,866,732
  and 4,191.1 in the failed full run.
- Route-DFT, the simpler pilot objective whose accuracy was statistically tied
  with route-ASFT and which avoids an extra full-vocabulary reference pass.
- Learning rate 1e-5 instead of 5e-5.
- Adapter-only milestones every 300 updates.
- A new 200-question held-out gate. Public evaluation is skipped if pass@8 is
  below the matched base or candidate truncations exceed base by more than 40.

The design follows the observed small-model stopping failure and the evidence
in [Making Small Language Models Efficient Reasoners](https://arxiv.org/abs/2505.07961),
[Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143),
and [Efficient Long CoT Reasoning in Small Language Models](https://arxiv.org/abs/2505.18440).
It is a testable repair, not a guaranteed leaderboard win.

Dataset preparation completed in ARC job `7950879` in 113 seconds. All 25 tests
passed locally and on ARC.

## Terminal result

The user approved the additional allocation and ARC Owl job `1123493` ran the
replacement on one B200. Training completed all **1,875** updates in 6,500.6
seconds, used all **15,000** selected rows, dropped zero rows, and processed
38,012,362 complete-sequence tokens. All 1,875 logged objective losses and
gradient norms are finite. The final LoRA adapter and merged model were written;
the merged weights remain on ARC and the 479,005,064-byte adapter is preserved in
the local result snapshot.

The prespecified 200-question development gate rejected the checkpoint:

| Development gate (200 questions × 8 samples) | Base | Candidate | Delta |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 19.938% | -3.438 pp, 95% CI [-6.125, -0.563] |
| Pass@8 | 47.500% | 43.500% | -4.000 pp, 95% CI [-10.500, +2.500] |
| Length-limited generations | 31/1,600 | 408/1,600 | +377 |

The job exited with code 1 because the gate intentionally raises a terminal
error when pass@8 falls below base or truncations exceed base by more than 40.
Both rejection conditions were met. Public evaluation was therefore skipped,
so this rescue has no nine-benchmark macro or leaderboard AVG. The outcome does
not beat the base model or the historical leaderboard.

The candidate-only evaluation emitted a tokenizer-regex warning while loading
the merged export. The matched manifests agree on data, prompts, sampling,
grader, versions, and hardware, but this warning is a remaining export/evaluation
confound. It does not overturn the gate result: the checkpoint also missed the
accuracy threshold and exceeded the truncation threshold by a wide margin.
