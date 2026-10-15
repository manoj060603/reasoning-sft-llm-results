# Full-SFT weight interpolation experiment

## Motivation

The full-parameter 15K model increased the matched seven-benchmark pass@8 macro
from 49.55% to 51.52%, while pass@1 fell and length-limit terminations increased
from 141 to 1,964. The measured gain is concentrated in AIME25 and CN Math.
This experiment tests whether a smaller parameter displacement retains some
pass@8 benefit while reducing generation damage.

## Models

Starting from the exact base and final full-SFT checkpoints, construct

`theta_alpha = theta_base + alpha * (theta_full_sft - theta_base)`

for `alpha` 0.25, 0.50, and 0.75. Interpolation is tensor-by-tensor in float32;
the resulting checkpoints are saved as BF16 with hash-bound provenance. No new
training occurs.

## Prespecified evaluation and selection

Evaluate base, final full SFT, and all three interpolated models on the same
training-disjoint 200-question development panel using temperature 0.6, top-p
0.95, eight samples, and the 32,768-token budget. Among interpolation models,
select by pass@8, then pass@1, fewer truncations, and lower repeated-8-gram
fraction. Evaluate only that selected interpolation and matched base on the
untouched 200-question confirmation panel at the same protocol.

This is exploratory post-public-suite evidence. It does not retroactively turn
the public benchmark result into a clean confirmatory leaderboard claim.

## Measured outcome: job 1144222

Job `1144222` completed in 1:25:18. All three BF16 interpolations contain 434
matching tensors with finite weights and complete model indices. The selected
alpha-0.50 model is 6,171,877,376 bytes; its model-index hash is
`45e07d77bbeb02675d70133f336cb9e5d86f54929ff1357453a787115fe257b8`.

| Development model | Pass@1 | Pass@8 | Truncations | Mean tokens | Repeated 8-gram |
|---|---:|---:|---:|---:|---:|
| Base | 23.31% | 47.50% | 29 | 1,415 | 0.058 |
| Full SFT | 23.00% | **53.00%** | 275 | 6,979 | 0.188 |
| Alpha 0.25 | 23.19% | 45.00% | **25** | 1,417 | 0.059 |
| Alpha 0.50 | **23.25%** | 51.50% | 55 | 2,141 | 0.075 |
| Alpha 0.75 | 22.88% | 50.50% | 119 | 3,678 | 0.115 |

The preregistered interpolation-only selector chose alpha 0.50 by pass@8. On
the fresh 200-question confirmation panel, base scored 26.75% pass@1 and 53.50%
pass@8; alpha 0.50 scored 26.19% and 55.50%. The paired deltas were −0.56
percentage points for pass@1 (95% CI −2.94 to +1.75) and +2.00 points for
pass@8 (95% CI −2.50 to +7.00). Neither interval excludes zero.

The smaller update retained a numerical pass@8 gain while reducing damage
relative to final full SFT, but it did not restore base-like generation:
confirmation truncations increased from 21 to 56, mean output length from 1,290
to 2,265 tokens, and repeated-8-gram fraction from 0.060 to 0.079. Alpha 0.50
was not evaluated on the public nine-benchmark suite and does not establish a
leaderboard gain. Complete artifacts are under
`results/arc-20261011-snapshot-2316/interpolation`.

## Complete public-suite follow-up: job 1145532

The fixed alpha-0.50 checkpoint was evaluated without further tuning on all nine
benchmarks. Job `1145532` completed in 1:01:58 with all 1,974 records and 15,792
responses present.

| Seven-benchmark macro | Base | Alpha 0.50 | Delta | Paired cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Pass@1 | 23.55% | 23.18% | -0.37 pp | [-1.75, +0.94] pp |
| Pass@8 | 49.55% | 50.63% | +1.08 pp | [-2.34, +4.52] pp |

The all-nine pass@8 macro is 51.00%, versus 50.60% for base. Alpha 0.50 produced
343 truncations, compared with 141 for base and 1,964 for full SFT. It therefore
recovers most of the termination damage but gives up 0.89 points of the full-SFT
seven-benchmark pass@8 score. The public result is below both the 51.52% full-SFT
result and the illustrative historical 52.5% leader. Its interval includes zero,
and the run remains exploratory because alpha was selected after the full-SFT
public result was known.
