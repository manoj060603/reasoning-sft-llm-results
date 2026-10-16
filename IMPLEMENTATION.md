# Reproducible reasoning SFT experiments

This implementation adds data preparation, selection, full SFT/QLoRA configuration, SLURM launchers, resumable evaluation, and paired score comparisons around the supplied LLaMA-Factory and grading code. The original assignment README and vendor code are retained.

**Current status:** See [DETAILED_PROJECT_REPORT_20261014.md](DETAILED_PROJECT_REPORT_20261014.md) for the final consolidated evidence. All 31 tests pass locally. Full-parameter SFT on all 15,000 rows has the strongest complete public result: 51.52% seven-benchmark pass@8 versus 49.55% for matched base, with a confidence interval crossing zero and severe overgeneration. Fixed alpha-0.50 interpolation completed the same suite at 50.63% with 343 truncations, versus 1,964 for full SFT and 141 for base. The repaired 2K student-guided experiment completed both 250-update arms, but guided targets scored 51.0% pass@8 on fresh confirmation versus 52.0% for the existing-target control and 53.5% for base, with worse termination. No official historical leaderboard win is established.

The next prespecified full experiment is documented in
[`TRIMSFT_EXPERIMENT.md`](TRIMSFT_EXPERIMENT.md). It adds detached Gaussian
logit-gap token weighting to the vendored LLaMA-Factory trainer and otherwise
reuses the exact 15K dataset and official nine-benchmark protocol. The public
suite will be evaluated only at the final 1,875-update checkpoint.

The follow-up recipe in [`PRIFT_EXPERIMENT.md`](PRIFT_EXPERIMENT.md) was staged
after the first three TrimSFT benchmark columns rejected that weighting rule.
It implements the official PriFT-mass frozen-reference objective and restores
the selection pipeline's 8,192-token bound. It remains unrun until the active
TrimSFT evaluation releases the GPU.

## Strategy and evidence

The proposed default is **quality-filtered, diverse, student-aware selection** of exactly 15,000 unique prompts. It is an experimental synthesis, not an exact reproduction of a paper or a guaranteed best strategy.

| Evidence reviewed | Implementation decision | Limitation |
|---|---|---|
| [AceReason 1.1](https://arxiv.org/abs/2506.13284) found prompt scaling more useful than adding responses to the same prompts | One trace per normalized prompt; broad prompt coverage | Reported results are not a promise for this 3B model |
| [s1 paper](https://arxiv.org/abs/2501.19393) and [GitHub](https://github.com/simplescaling/s1) emphasize quality, difficulty, diversity | Quality gates, clustering, student-loss proxy | No budget forcing: preserve the evaluation protocol |
| [LIMO paper](https://arxiv.org/abs/2502.03387) and [GitHub](https://github.com/GAIR-NLP/LIMO) motivate careful reasoning-data curation | Preserve complete teacher solutions and final answers | Large-model small-data results need testing at 3B |
| [LIMOPro GitHub](https://github.com/GAIR-NLP/LIMOPro) describes perplexity-based step-importance refinement | Do not arbitrarily trim reasoning paragraphs | PIR requires step-removal counterfactual scoring; this implementation does not claim to implement PIR |
| [Step-length confounding](https://arxiv.org/abs/2604.06834) warns of confounded likelihood scores | Rank student loss within response-length bins | This is a partial control, not ASLEC or a complete correction for step length |
| [OpenThoughts](https://arxiv.org/abs/2506.04178) motivates controlled data-recipe experiments | Include random, filtered-random, and diversity-only ablations | No external data silently added |

Pipeline details:

1. Canonicalize Alpaca or single-turn chat records. Hash normalized questions; retain numerical values/operators. Reject exact and high character-shingle overlap with the supplied benchmarks. Reserve 1,000 unique prompts for validation before any selection; exclude their near matches from the training pool. The overlap screen is lexical and cannot guarantee absence of semantic paraphrases or undisclosed benchmark contamination.
2. Count tokens using the actual Qwen chat template. Require a balanced final `\boxed{...}`, at least 128 response tokens, limited repeated 8-grams, and a complete sequence fitting the 16,384-token training cutoff with a 32-token template margin. These checks establish formatting/completeness, not correctness. They deliberately favor boxed-answer math/science over code, which is a substantive domain-selection change relative to random sampling.
3. Cluster question representations: character TF-IDF + SVD is the default; `--embedding-model sentence-transformers/all-MiniLM-L6-v2` enables semantic embeddings after installing sentence-transformers. Allocate samples approximately by square root of cluster size, respecting small-cluster capacity.
4. For the hybrid variant, measure the base student's mean teacher-forced negative log likelihood on the first 512 response tokens, with prompt tokens masked. Prefer approximately the 65th percentile within power-of-two response-length bins, plus low repetition and seeded tie-breaking. This is a cheap difficulty **proxy**, not measured solve rate or verified trace quality. The percentile/weights are hypotheses that must be ablated. Features checkpoint every 1,000 records.
5. Export `random` from the entire clean pool, `filtered_random` from the exact same eligible pool as advanced, and `advanced`. The first follows the assignment-style random baseline and may contain traces that training truncates. The second isolates selection from formatting, domain, and length filtering. All variants use the same reserved validation pool.

The best practical experiment sequence is random → filtered-random → diversity-only → hybrid, holding the training recipe fixed. Start with a 100-step smoke run (`reasoning.train_config --max-steps 100` in a separate output directory), measure throughput, then estimate the full cost from actual tokens. A 24-GPU-hour budget is not assumed sufficient for all variants or 3 epochs. Compare full SFT at 5e-6 versus 1e-5 only if held-out development results justify the extra run. Do not select checkpoints or tune heuristics on public test scores.

## Installation on Linux/CUDA

For the original LLaMA-Factory route, use two fresh Python 3.10 environments. The vendored training dependency bounds and the original `eval/requirements.txt` are incompatible with blindly installing the newest packages together. These proposed version pins remain unvalidated. The actual advanced ARC experiments use the separate `scripts/arc_*.sh` path and an existing Torch 2.10.0 / Transformers 4.57.6 / vLLM 0.19.0 runtime, recorded in the advanced guide.

```bash
cd /absolute/path/Project-Reasoning-SFT-LLM-main
python3.10 -m venv .venv-train
source .venv-train/bin/activate
bash scripts/setup_train.sh
# Optional: pip install sentence-transformers==3.4.1

deactivate
python3.10 -m venv .venv-eval
source .venv-eval/bin/activate
bash scripts/setup_eval.sh
```

The evaluator retains the assignment's symbolic grader. Actual ARC testing exposed an incompatibility in the assignment's suggested ANTLR workaround. `latex2sympy2==1.9.1` needs ANTLR 4.7.2; the private evaluator now uses that version. SymPy's separate ANTLR parser is unavailable in this combination, and the original grader's existing fallback uses latex2sympy2. Six timed and untimed equivalence checks gate evaluation, and exact grading-library versions are recorded. Historical runs using another parser stack are not an exact comparison. Training setup runs `pip check`; both environments save their installed versions. Follow site documentation for loading CUDA/compiler modules first.

## Prepare data

Run from the project root, in the training environment; the default hybrid scorer needs a CUDA GPU:

```bash
bash scripts/prepare.sh
```

The downloader resolves and records a Hugging Face dataset commit. For immutable model provenance, pass the same Hub commit via `features --revision`, `train_config --revision`, and baseline `evaluate --revision`. The 100K source file and model weights are downloaded only when this command is run. Allow enough disk and host RAM for the full JSON dataset and tokenization. Public dataset schema was checked against a small remote sample; the full dataset was not downloaded here.

For the CPU-capable diversity ablation, use a distinct artifact directory or retain/copy the previous exports before rerunning:

```bash
STRATEGY=diversity bash scripts/prepare.sh
```

Individual commands offer explicit input/output paths. For example, after hybrid features exist, export the diversity ablation without rescoring:

```bash
python -m reasoning.select --strategy diversity --output artifacts/datasets-diversity
python -m reasoning.devset
```

The development set contains 200 frozen questions from the reserved validation pool. Answers are extracted from teacher traces and are not independently verified. It is useful for selecting variants; it is not a new public benchmark.

The benchmark audit reads valid JSONL correctly even when a string contains U+2028. The source GPQA file is valid and needs no repair. Canonical copies and hashes are under `artifacts/benchmarks`. Counts: AIME24 30, MATH 500, CN Math 30, Kaoyan 199, AMC 40, Minerva 272, OlympiadBench 675, GPQA 198. **AIME25 is absent from the supplied archive.** The ARC continuation now adds the pinned `math-ai/aime25` supplement in a separate directory and audits it against the unchanged split; see the advanced experiment guide. For other additional benchmarks, use `reasoning.prepare --extra-exclusion PATH` before preparing new splits, or `reasoning.extend_exclusions` to filter an existing split without moving held-out examples. The historical table is illustrative Fall 2025 data, not measured results of these runs.

## Train

Generate configs after preparing data:

```bash
python -m reasoning.train_config --dataset random --output runs/random-s42
python -m reasoning.train_config --dataset filtered_random --output runs/filtered-random-s42
python -m reasoning.train_config --dataset advanced --output runs/hybrid-s42
python -m reasoning.train_config --dataset advanced --dataset-dir artifacts/datasets-diversity --output runs/diversity-s42
```

Default: full SFT, BF16, FlashAttention 2, DeepSpeed ZeRO-2, sequence length 16,384, per-device batch 1, accumulation 8, learning rate 5e-6, cosine schedule, 3% warmup, 3 epochs, response-only loss, gradient checkpointing. Prefer a large-memory GPU such as an H200 or A100 80GB for this profile; actual memory depends on sequence lengths. No packing is used, avoiding cross-example attention ambiguity. Save/evaluate every 100 steps; retain the best validation-loss checkpoint. Validation loss is only a proxy for reasoning accuracy, so use the frozen development set for final variant selection.

For a smaller GPU:

```bash
python -m reasoning.train_config --dataset advanced --profile qlora --output runs/hybrid-qlora-s42
```

QLoRA uses NF4 via LLaMA-Factory defaults, rank 64/alpha 128, dropout .05, and learning rate 5e-5. It is an alternative experiment, not score-equivalent to full SFT. A 24GB GPU may still need a shorter cutoff; regenerate features/selection/configs consistently rather than silently truncating advanced traces. Before evaluation, merge the adapter using `llamafactory-cli export configs/hybrid-qlora-s42-export.yaml` in the training environment and evaluate the `-merged` directory.

ARC launch (replace account/partition and paths with your available allocation):

```bash
export PROJECT_ROOT="$PWD"
export TRAIN_PYTHON="$PWD/.venv-train/bin/python"
export CONFIG="$PWD/configs/hybrid-s42.yaml"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/train.slurm
```

For direct single-GPU execution: `llamafactory-cli train configs/hybrid-s42.yaml`. To resume, regenerate the same configuration with `--resume /absolute/path/to/checkpoint-N`. No automatic submission, uploading, or paid resource provisioning occurs.

## Evaluate and compare

Use the evaluation environment. Development evaluation can run before public benchmarks:

```bash
python -m reasoning.evaluate --model runs/hybrid-s42 --data-dir artifacts/dev \
  --benchmarks acereason_dev --output results/hybrid-dev

python -m reasoning.evaluate --model Qwen/Qwen2.5-3B-Instruct --output results/base
python -m reasoning.evaluate --model runs/random-s42 --output results/random
python -m reasoning.evaluate --model runs/hybrid-s42 --output results/hybrid
python -m reasoning.compare --baseline results/random --candidate results/hybrid
```

Alternatively set `PROJECT_ROOT`, `EVAL_PYTHON`, `MODEL`, and `RUN_OUTPUT`, then submit `scripts/eval.slurm` with your account/partition. A smoke evaluation uses `--benchmarks amc` and its own output directory. Do not reuse that directory for a different benchmark list/configuration.

Protocol: eight samples, temperature 0.6, top-p 0.95, maximum generation 32,768, assignment system prompt, original extraction and correctness functions. The native model has **32,768 total positions**, including the prompt; vLLM can end output at that limit. Truncation counts are reported. Other sampling parameters are explicit vLLM defaults (no top-k filter, min-p zero, repetition penalty one), so model-specific generation files cannot change the comparison. These choices and per-question seeds mean results are matched internally, not bitwise replicas of historical runs.

Each atomic batch caches complete generations and grading. The manifest binds resume to checkpoint file hashes or Hub commit, dataset hashes/IDs/counts, grader hashes, package versions, and generation settings. A changed run requires a new directory. Comparisons reject incomplete or mismatched runs. Score outputs include average sample accuracy (pass@1), any-correct accuracy (pass@8), and paired bootstrap intervals over questions. Repeat training/evaluation seeds before trusting small gains, especially on 30-question AIME. The available-suite macro average is explicitly **not** the historical leaderboard AVG, because that average includes AIME25 and excludes AIME24/MATH.

## Local verification

```bash
python3 -m unittest discover -s tests -v
python3 -m reasoning.audit
# Optional: downloads tokenizer files, not model weights; uses synthetic examples
python3 scripts/smoke_cpu.py
```

No leaderboard values are prefilled as experiment results. After GPU runs, preserve selection manifests, environment freezes, YAMLs, trainer logs, and result manifests alongside the comparison JSON. The remaining decisive test is whether the hybrid or diversity variant improves held-out accuracy under the matched recipe.

## Full-parameter 15K LLaMA-Factory execution

The final full-run configuration is `configs/lmf-full-15000-sft.yaml`, launched
by `scripts/lmf_full_15000.slurm` through
`scripts/arc_lmf_full_15000.sh`. It uses all 15,000 `advanced` rows, response-only
SFT, a 4,096-token cutoff, learning rate `2e-6`, effective batch size eight, and
1,875 updates. Checkpoints are written every 250 steps and the last two are
retained. The script verifies selection IDs, data and source hashes, final step,
and finite training logs before evaluation.

Job `1137895` completed training and eight benchmarks. Job `1142866` resumed the
same manifest-bound output directory and completed GPQA. The evaluator validates
cached batch identities and all eight samples per problem before reuse, so the
combined result is one protocol-consistent nine-benchmark run. The complete
comparison is
`results/arc-20261011-snapshot-2316/evaluations/lmf-full-15000-sft-suite-comparison.json`.
The final model remains on ARC as three safetensors shards under
`runs/lmf-full-15000-sft`.
# Gen-SSD-style generation runner

`reasoning.gen_ssd` implements resumable chunk-level teacher generation with
base-student perplexity selection and symbolic answer verification. Job
`1144570` exposed a vLLM edge case in which an already-complete prefix returns
only zero-token `stop` candidates. The runner now records that as normal
termination rather than dropping every candidate. The failed run produced no
training or benchmark result.

## Weight interpolation implementation and result

`reasoning.interpolate_weights` streams matching base and full-SFT tensors and
writes BF16 checkpoints for `base + alpha * (full_sft - base)`. It rejects key,
shape, or nonfinite-value mismatches and records the model-index hashes. The
completed run tested alpha 0.25, 0.50, and 0.75, selecting alpha 0.50. Fresh
confirmation measured 55.50% pass@8 versus 53.50% for base, with paired 95% CI
for the +2.00-point delta of [−2.50, +7.00]. Generation-cost diagnostics still
regressed, and no public-suite evaluation was performed for the interpolation.


## GEM and model-aware GEM terminal results (October 16, 2026)

Both full 15K GEM experiments completed exactly 1,875 finite updates and all nine benchmarks. Ordinary GEM scored **50.53%** seven-benchmark pass@8 versus **49.55%** for matched base, with **417/15,792 truncations**. Model-aware GEM scored all 83,037 approved pool rows, selected exactly 15,000 unique 8K-safe examples, and scored **48.32%** pass@8 with **284/15,792 truncations**. Its matched pass@1/pass@8 values were **21.00% / 48.32%**, versus **23.55% / 49.55%** for base.

Ordinary GEM improved the target macro by 0.98 points but failed the strict per-column objective. Model-aware reselection improved Kaoyan by 5.03 points and GPQA by 3.54 points over base, tied base on AIME25 and CN Math, and regressed most other columns. Both cleared only the contextual Minerva and GPQA Fall 2025 thresholds. The ordinary full-SFT model remains the highest measured seven-benchmark candidate at **51.52%**, with its previously documented pass@1 and truncation costs.

This evidence rejects teacher-forced base NLL as a sufficient standalone selection signal for this pool. NLL measures surprising teacher tokens and can favor noisy or unsupported trajectories; it does not establish correctness or learnability. Complete receipts and predictions are under `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-15000-8k-*` and `results/arc-20261011-snapshot-2316/evaluations/lmf-gem-modelaware-15000-8k-*`. Historical runtime equivalence remains unverified, so these are internal matched results rather than official leaderboard claims.
