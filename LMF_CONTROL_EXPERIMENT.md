# LLaMA-Factory trainer control

Date prepared: 2026-10-13

## Question

The concise 2K experiment showed severe repetition after ordinary SFT and much
smaller damage after DFT. Because both arms used the custom LoRA runner, the
experiment did not distinguish a supervision problem from a trainer-specific
problem. Arm A reproduces the concise ordinary-SFT condition in the vendored
LLaMA-Factory implementation.

## Matched Arm A

- Original `Qwen/Qwen2.5-3B-Instruct` base snapshot.
- Exact sealed 2,000-example concise dataset, unchanged order and content.
- Response-only SFT with Qwen template and 4,096-token cutoff.
- Unquantized LoRA rank 64, alpha 128, dropout 0.05.
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`.
- Batch size one, accumulation eight, 250 optimizer updates.
- AdamW, learning rate `1e-5`, weight decay 0.01, gradient norm 1.0.
- Cosine schedule, 3% warmup, BF16, gradient checkpointing, seed 42.
- Checkpoints at 50, 100, 150, 200, and 250 updates; the prespecified diagnostic
  evaluates steps 50, 100, and 250.

The control intentionally changes the training implementation and its Qwen data
processor. It cannot be bitwise identical to the custom loop, but the model,
examples, adaptation rank, target modules, optimizer-scale settings, update
count, and response-only objective are matched.

## Evaluation rule

All three checkpoints first run on the existing 200-question development panel
at eight samples and an 8,192-token limit. A checkpoint is excluded as clearly
unstable if it has more than 40 additional truncations or more than 0.05 extra
mean repeated-8-gram fraction relative to base. Among safe checkpoints, choose
pass@8, then pass@1, then fewer truncations.

Only the selected safe checkpoint proceeds to matched base-versus-candidate
development evaluation at the official 32,768-token generation setting. This is
still development evidence. A fresh confirmation panel is reserved until Arms
A, B, and C have identified one candidate.

## Interpretation

- If LLaMA-Factory LoRA avoids the SFT collapse, investigate the custom trainer
  and serialization path before changing supervision.
- If it reproduces the collapse, the evidence shifts toward update strength,
  adaptation limits, or teacher-target mismatch.
- Arm A cannot show whether full fine-tuning or rewritten targets help. Those
  are separate Arms B and C and should follow only after this control resolves
  the trainer uncertainty.

No public benchmark is opened by this experiment, and no leaderboard claim can
come from it.

## Actual execution and superseding full run

The user stopped Arm A after 4 minutes 37 seconds and requested that no further
pilot or development-gate runs be used. Arm A therefore has no result and must
not be interpreted as a trainer comparison.

The replacement was job `1137895`, a full-parameter LLaMA-Factory SFT run on
all 15,000 selected examples. It used response-only loss, a 4,096-token training
cutoff, batch size one with accumulation eight, learning rate `2e-6`, and 1,875
updates. Training completed successfully in 1:06:29. All 15,000 examples were
loaded, the final global step was 1,875, and all recorded losses and gradient
norms were finite. Final full-model weights and checkpoints remain preserved on
ARC under `runs/lmf-full-15000-sft`.

The first allocation completed eight benchmarks and timed out during GPQA. Job
`1142866` resumed the exact manifest-bound result directory, validated the
cached predictions, completed all 198 GPQA problems, and wrote the final matched
comparison. Every benchmark now has eight samples for every problem.

The matched seven-benchmark pass@8 macro improved from **49.55% to 51.52%**
(+1.97 points), while pass@1 fell from **23.55% to 21.85%** (-1.70 points).
A stratified paired cluster bootstrap gives a 95% interval of **[-1.89, +5.74]
points** for the pass@8 macro change and **[-3.16, -0.22] points** for pass@1.
The pass@8 gain is therefore uncertain, while the pass@1 regression is supported.

Generation behavior remains substantially worse: total length-limit terminations
rose from 141 to 1,964 across the nine benchmarks. Standard LLaMA-Factory
full-parameter SFT therefore does not remove the target-induced overgeneration
seen with the custom LoRA trainer. The complete receipt is
`results/arc-20261011-snapshot-2316/lmf-full-15000-final-report.json`.
