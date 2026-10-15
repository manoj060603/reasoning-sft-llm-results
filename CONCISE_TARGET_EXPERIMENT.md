# Concise-target 2K SFT versus DFT experiment

Date prepared: 2026-10-12

## Motivation

The 15K route-ASFT run and capacity-matched route-DFT rescue both produced severe
nontermination. The rescue candidate had 408 length-limited responses on the
200-question gate, compared with 31 for the base. The prespecified diagnostic
classifies those 408 samples as:

- 321 high-repetition loops without an extracted answer;
- 75 nonrepetitive but unfinished solutions; and
- 12 responses that emitted an extractable answer and then reached the length
  limit, only one of which was correct.

Stopping after an answer could recover at most a small part of the failure. The
dominant hypothesis is that the model learned long, unstable reasoning behavior.

## Training-target audit

All 15,000 rescue targets contain one complete teacher answer and both thinking
tags. Only 30 have a repeated-word-8-gram fraction above 0.10 and none exceeds
0.35. In a deterministic 256-row serialization sample, the prompt is an exact
prefix and the assistant `<|im_end|>` token is supervised. Qwen's chat template
places a newline token after `<|im_end|>`, so EOS is penultimate rather than the
last serialized token. Base and candidate development generation settings match
exactly apart from model identity.

The failure is therefore not explained by missing answer labels, omitted
assistant-end supervision, mismatched samplers, or obviously repetitive teacher
targets. The remaining target-level problem is length and excess deliberation:
the rescue target mean is about 2,447 response tokens.

## Sealed comparison

Select exactly 2,000 unchanged teacher records from the approved AceReason pool.
Every record must have:

- one `<think>...</think>` block;
- exactly one balanced final `\boxed{...}`;
- at most 16 characters after the closing box;
- no more than 1,536 response tokens;
- repeated-word-8-gram fraction at most 0.10; and
- a complete serialized sequence below 4,096 tokens.

Route-coverage selection is applied within the eligible pool. Both arms consume
the identical ordered 2,000 records and start from the original base model.

| Arm | Objective | Examples | Updates | Accumulation | LR | LoRA |
|---|---|---:|---:|---:|---:|---:|
| Concise SFT | response-token cross entropy | 2,000 | 250 | 8 | 1e-5 | rank 64 |
| Concise DFT | detached-probability weighted cross entropy | 2,000 | 250 | 8 | 1e-5 | rank 64 |

The experiment selects on the existing 200-question rescue gate because that
panel has already guided design choices. An arm passes only if pass@8 is at least
the matched base and it produces no more than 40 additional truncations. Ties use
pass@8, pass@1, and then fewer truncations. Only a passing winner is evaluated on
a fresh 200-question confirmation set drawn from the unused held-out records.

Report pass@1, pass@8, length-limited samples, generated tokens, mean response
length, boxed-answer count, and repeated-8-gram fraction. Do not scale to 15K or
open the public benchmark suite unless the fresh confirmation shows improved
accuracy with controlled truncation.

## Intermediate-checkpoint diagnostic

Evaluate rescue milestones 300, 900, and 1,500 on the existing development gate
using vLLM LoRA inference. These early, middle, and late points were chosen before
reading their scores. The final 1,875-step checkpoint and base are already
measured. This diagnostic tests whether useful behavior existed before the final
checkpoint; it cannot select the concise experiment's objective or data.

## Evidence boundary

The compact dataset preserves teacher outputs unchanged and verifies structural
completion, low repetition, token length, and answer-box presence. The teacher
answers are not independently re-solved, so “verified” here means serialization
and structural verification rather than a new mathematical proof check.

## Terminal outcome of job 1133003

ARC Owl job `1133003` started on 2026-10-12 and terminated after 10 minutes 47
seconds with exit code `1:0`. This was an orchestration failure before either
concise-target arm began training. It is therefore not evidence for or against
concise SFT or concise DFT.

The prespecified step-300 rescue-adapter diagnostic completed on all 200
development questions and all 1,600 sampled responses:

| Model | Pass@1 | Pass@8 | Length-limited samples |
|---|---:|---:|---:|
| Matched base | 23.375% | 47.500% | 31/1,600 |
| Rescue milestone step 300 | 21.5625% | 48.000% | 245/1,600 |
| Difference | -1.8125 pp | +0.500 pp | +214 |

The step-300 checkpoint has essentially unchanged pass@8 and substantially worse
termination. It would fail the experiment's stopping gate because it exceeds the
base truncation count by more than 40. This result is diagnostic only and does
not select between the concise objectives.

After writing that evaluation, `reasoning.compare` rejected the otherwise
matched manifests with `ValueError: incomparable protocol: lora_identity`.
Model identity must differ in any base-versus-LoRA comparison, so treating
`lora_identity` as a protocol field is an implementation error. The script
stopped before evaluating steps 900 and 1,500 and before training the 2,000-row
SFT and DFT arms. Consequently there are no concise adapters, training receipts,
selection result, or fresh-confirmation scores from this job. The completed
evaluation and full job log were fetched non-destructively under
`results/arc-20261011-snapshot-2316/`.

## Fast-track completion: job 1133616

The repaired fast-track job completed successfully in 38 minutes 13 seconds. It
trained both matched arms on the sealed 2,000-row dataset and intentionally
skipped the remaining legacy milestone diagnostics. Each arm used all 2,000
rows, dropped zero rows, completed all 250 updates, and logged finite losses and
gradient norms. Both final LoRA adapters and their receipts are preserved in the
local ARC snapshot.

| Development: 200 questions × 8 samples | Base | Concise SFT | Concise DFT |
|---|---:|---:|---:|
| Pass@1 | 23.375% | 15.3125% | 23.4375% |
| Pass@8 | 47.500% | 40.000% | 46.000% |
| Pass@1 delta, paired 95% CI | — | -8.0625 pp [-11.000, -5.125] | +0.0625 pp [-2.500, +2.4375] |
| Pass@8 delta, paired 95% CI | — | -7.500 pp [-14.500, -1.000] | -1.500 pp [-7.000, +4.000] |
| Length-limited samples | 31 | 596 | 93 |
| Mean output tokens | 963.25 | 3,879.33 | 1,437.09 |
| Boxed-answer samples | 1,569 | 998 | 1,494 |
| Mean repeated-8-gram fraction | 0.0567 | 0.3027 | 0.0903 |

Neither arm passed the preregistered development gate. SFT failed both accuracy
and stopping by a large margin. DFT preserved pass@1 and substantially reduced
the SFT failure, but its pass@8 was 1.5 points below base and it produced 62
additional truncations, exceeding the allowed 40. The selection receipt records
`selected: null`, so fresh confirmation and public-suite evaluation were
correctly skipped.

This result rejects the hypothesis that selecting unchanged complete traces
below 1,536 tokens is sufficient to improve this model. It also strengthens the
evidence that DFT is safer than ordinary SFT for these targets, while providing
no evidence that either trained model beats the base. Because no candidate
passed development, there is no leaderboard or Fall 2025 comparison to report.
