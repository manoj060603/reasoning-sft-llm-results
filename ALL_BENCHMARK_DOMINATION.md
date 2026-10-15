# Strict all-benchmark objective

The next objective is stronger than improving the seven-benchmark macro: one fixed model must strictly exceed the Fall 2025 leader in every displayed pass@8 column under the assignment sampler.

| Benchmark | Historical leader | Minimum strict target | Current full SFT | Additional solves needed |
|---|---:|---:|---:|---:|
| AIME24 | 7/30 (0.233 shown) | 8/30 = 0.267 | 5/30 | 3 |
| AIME25 | 7/30 (0.233 shown) | 8/30 = 0.267 | 6/30 | 2 |
| MATH | 450/500 = 0.900 | 451/500 = 0.902 | 425/500 | 26 |
| CN Math 2024 | 13/30 (0.433 shown) | 14/30 = 0.467 | 10/30 | 4 |
| Kaoyan | about 137/199 = 0.688 | 138/199 = 0.693 | 119/199 | 19 |
| AMC | 32/40 = 0.800 | 33/40 = 0.825 | 28/40 | 5 |
| Minerva | about 94/272 = 0.346 | 95/272 = 0.349 | 133/272 | already above |
| OlympiadBench | about 384/675 = 0.569 | 385/675 = 0.570 | 333/675 | 52 |
| GPQA | about 120/198 = 0.606 | 121/198 = 0.611 | 157/198 | already above |

The printed historical decimals do not expose their exact underlying counts. The strict targets above conservatively convert each displayed score to the next integer solve count. Historical dataset and runtime equivalence remain unverified.

## Full-run recipe

The current full SFT learned useful mathematical behavior but also learned the teacher's long hidden deliberation style, producing 1,964 truncations. The new treatment changes the supervision rather than merely adding more optimization:

1. Start from the original Qwen2.5-3B-Instruct model.
2. Use 15,000 unique, decontaminated math questions from the sealed AceReason pool.
3. Remove the complete `<think>...</think>` section and supervise the polished derivation after `</think>`; do not truncate targets.
4. Require a complete boxed answer, 96–2,048 response tokens, no malformed thinking tags, repeated 8-gram fraction at most 0.08, and complete Qwen serialization within 4,096 tokens.
5. Allocate the dataset across geometry, number theory, algebra, combinatorics, calculus/linear algebra, probability, and other mathematics with fixed quotas. Classification uses general mathematical vocabulary and never public benchmark text.
6. Train all 15,000 examples for 1,875 updates with full-parameter response-only SFT at `1e-6`, half the previous learning rate, to reduce base-capability drift.
7. Run the complete nine-benchmark suite directly. No pilot, public-benchmark checkpoint selection, or partial result may be presented as success.
8. Apply `reasoning.domination` to require every column to pass independently. Also report pass@1, truncation, repetition, and confidence intervals.

This recipe is motivated by evidence that long-chain distillation can burden small models with redundant steps, while pruning and student-compatible trajectories can retain reasoning with shorter outputs ([Efficient Long CoT Reasoning in Small Language Models](https://arxiv.org/abs/2505.18440)). Rejection-sampling distillation from larger models is established, but the completed Gen-SSD experiment showed that stronger generated targets alone did not transfer reliably here. The present run therefore isolates a different intervention: complete visible solutions, broader competition-math coverage, and a smaller parameter update.

There is no defensible guarantee that a 3B model will clear all nine thresholds. The result is successful only if the final nine-column gate says so.

## Execution

ARC Owl job `1156820` was submitted on October 15, 2026 with an eight-hour B200 limit. The CPU preflight loaded exactly 15,000 unique rows, verified the dataset SHA-256 `fb1a2dfe71f239be755d0bceb03e443888ecfd895ce9cbb3b1465b36bc091e48`, and confirmed the fixed domain counts. The full job has no pilot, development gate, checkpoint selection, or public-suite tuning; it trains all 1,875 updates and evaluates all nine benchmarks.

