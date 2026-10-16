# ArXiv-backed score-improvement plan

## Objective and constraints

The target is one fixed Qwen2.5-3B-Instruct descendant that is evaluated with the assignment sampler on all nine benchmarks and strictly clears every displayed Fall 2025 column. The comparison remains contextual because historical dataset and runtime equivalence are unverified.

The strict integer targets are 8/30 AIME24, 8/30 AIME25, 451/500 MATH, 14/30 CN Math 2024, 138/199 Kaoyan, 33/40 AMC, 95/272 Minerva, 385/675 OlympiadBench, and 121/198 GPQA. The strongest complete model currently clears Minerva and GPQA but needs 3, 2, 26, 4, 19, 5, and 52 additional solved questions on the other seven tasks.

Every proposed final experiment must use exactly 15,000 approved examples, preserve the sealed benchmark exclusions, start from the pinned original base unless explicitly testing a continuation, train the full prespecified schedule, and evaluate all nine benchmarks with temperature 0.6, top-p 0.95, eight samples, and a 32,768-token generation limit. Public benchmarks cannot be used for training, hyperparameter selection, checkpoint selection, or data selection.

## Evidence from this project

- Ordinary full SFT reached 51.52% seven-benchmark pass@8, but pass@1 fell and truncations rose to 1,964/15,792. It improved only four of nine benchmarks.
- Alpha-0.50 weight interpolation reduced drift and truncation but also reduced the macro to 50.63%.
- Compact targets nearly fixed termination but reduced the macro to 44.83%.
- Teacher-guided 2K targets did not beat the matched existing-target arm or base on fresh confirmation.
- TrimSFT was stopped after seven completed benchmarks because every completed column was below base.

The repeated pattern is useful: the project can trade accuracy for termination or vice versa, but single-trajectory cross-entropy has not produced broad coverage. Any next method should directly address pretrained-capability retention, distributional diversity, or both.

## Ranked experiment queue

### 0. Fixed TIES merge of full SFT and GEM — running next

**Sources:** [TIES-Merging](https://arxiv.org/abs/2306.01708) and [Model Soups](https://arxiv.org/abs/2203.05482).

The ordinary full-SFT checkpoint has the strongest complete macro, while GEM has far fewer truncations and a different output distribution. A fixed layerwise TIES merge can retain aligned task-vector changes and discard small or sign-conflicting changes without training on or selecting against public benchmarks. The prespecified merge keeps the top 20% task-vector entries per tensor, uses magnitude sign election and a disjoint mean, and applies scale 1.0. It is evaluated once on all nine benchmarks. This is a low-cost test of whether the two completed objectives learned complementary transferable directions; it is not a public-score-tuned soup.

### 1. PriFT-mass with the full 8K-approved targets — completed, negative

**Paper and code:** [PriFT: Prior-Support Guided Supervised Fine-Tuning](https://arxiv.org/abs/2606.09396), [official implementation](https://github.com/wang-kee/PriFT).

PriFT computes token weights from a frozen copy of the pretrained model. This avoids the self-reinforcing weight drift of online probability weighting and preferentially learns target tokens supported by the base distribution. It directly addresses the broad regressions observed after ordinary SFT. The current run also uses the dataset's actual 8,192-token bound; the earlier 4K trainer could cut otherwise approved targets before their final answer and end token.

PriFT completed all nine benchmarks at 45.96% seven-benchmark pass@8 versus 49.55% for the matched base. It cleared the contextual Fall 2025 columns only on Minerva and GPQA. This is a completed negative method result and will not be rerun unchanged.

### 2. GEM on the same sealed 15K set — completed, partial gain

**Paper and code:** [Preserving Diversity in Supervised Fine-Tuning of Large Language Models](https://arxiv.org/abs/2408.16673), [official GEM repository](https://github.com/liziniu/GEM).

GEM replaces ordinary cross-entropy distribution matching with an entropy-regularized reverse-KL objective. The paper reports less overfitting and gains of up to ten points on math/code best-of-N evaluation. This is unusually well aligned with the assignment's pass@8 metric and with our observed collapse into long repetitive trajectories.

**Prespecified full-run design:** original base; exact same 15,000 identities and 8K serialization; response-only supervision; official GEM loss and published default first; 1,875 updates; one final checkpoint; direct nine-benchmark evaluation. Log cross-entropy as a diagnostic in addition to GEM loss, as recommended by the official implementation. No public-suite tuning of the entropy parameter.

**Why ranked second:** it optimizes the output-distribution property most clearly missing from our full-SFT model, has public reference code, and does not require new teacher generations or duplicate prompts.

Training completed all 1,875 finite updates and all nine benchmarks. GEM scored 50.53% seven-benchmark pass@8 versus 49.55% for base, but pass@1 fell to 21.81% and the strict goal passed only Minerva and GPQA.

### 3. Model-aware selection from the complete approved pool — completed, negative

**Sources:** [Select2Reason](https://arxiv.org/abs/2505.17266), [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284), and [GEM](https://arxiv.org/abs/2408.16673).

The current 15K identities were selected using route diversity and static quality checks. The 83,037-row approved pool was never scored relative to the pinned student: `features_extended.json` contains clusters, lengths, answer completeness, and repetition, but no base-model loss. This leaves a concrete data-selection hypothesis untested while repeated loss-only changes reuse the same examples.

The staged successor computes teacher-forced base response NLL over 512 tokens for every approved row, ranks difficulty within response-length buckets, and selects exactly 15,000 unique 8K-safe rows with frozen 128-cluster quotas around the prespecified 65th difficulty percentile. It then applies the already tested GEM objective. NLL is treated only as a model-relative difficulty proxy; it does not prove correctness or solve probability. Public benchmark data and scores are excluded from selection.

The run completed full-pool scoring, selection, 1,875 finite updates, and all nine benchmarks. It scored 48.32% seven-benchmark pass@8 and 21.00% pass@1, both below base, while reducing truncations to 284. The hypothesis is rejected for this selector: teacher-forced NLL can emphasize surprising teacher tokens without identifying correct, transferable, student-learnable skills.

### 4. Base-anchored diversity SFT — conservative hybrid only after standalone evidence

**Sources:** [Anchored Supervised Fine-Tuning](https://arxiv.org/abs/2509.23753) and [CurioSFT](https://arxiv.org/abs/2602.02244) with its [official implementation](https://github.com/HaoooWang/CurioSFT).

ASFT adds a KL anchor to stabilize probability-weighted fine-tuning; CurioSFT uses adaptive self-distillation to preserve useful exploration instead of flattening all token distributions. Both target the failure pattern we measured: SFT acquires some new solves while losing base capability and sampling diversity.

The project already ran one ASFT-style route experiment that overgenerated badly, so simply repeating it is low priority. A defensible revisit would use the approved 8K-complete dataset, a frozen original-base anchor, explicit EOS supervision, and a fixed published coefficient chosen without public scores. CurioSFT is scientifically attractive but its released training stack targets eight H800 GPUs; a faithful one-B200 port is a larger engineering and runtime risk than GEM.

**Decision:** implement only if PriFT shows base retention but inadequate pass@8 coverage, or GEM improves coverage while drifting on GPQA/AMC. Do not invent an unvalidated PriFT+GEM+ASFT compound objective before measuring the standalone methods.

### 5. Multiple-solution diversity curation — high upside, larger data change

**Sources:** [Reasoning Path Divergence](https://arxiv.org/abs/2510.26122), [official RPD code](https://github.com/fengjujf/Reasoning-Path-Divergence), and [Set Supervised Fine-Tuning](https://arxiv.org/abs/2510.05132) with [official SSFT code](https://github.com/Sheng-J/SSFT).

RPD reports that selecting semantically different correct paths per problem improves pass@k, including a reported +4.99 points on AIME24. SSFT explicitly matches multiple correct reasoning traces to global forking tokens to preserve parallel reasoning modes. These are more directly targeted at pass@8 than selecting one canonical solution per question.

**Assignment-compatible design:** keep exactly 15,000 approved training rows and the sealed exclusions; form multiple-solution groups only from approved AceReason responses or newly verified solutions allowed by the assignment; verify every final answer symbolically; cap serialized length at 8K; allocate the 15,000 rows across prompt groups without exposing benchmark content. Report both unique prompt count and total row count. If the instructor requires 15,000 unique prompts, use RPD only as a selection score among available candidate responses and do not duplicate prompts.

**Why not immediate:** the current pool is primarily one response per selected prompt, so faithful SSFT/RPD needs additional correct trajectories and substantially more generation, verification, and training integration.

### 6. Direct Coverage Optimization or verifier-guided RFT — closest objective, highest compliance/compute risk

**Source:** [Direct Coverage Optimization](https://arxiv.org/abs/2502.07154).

DCO directly targets pass@N coverage rather than token likelihood. This is conceptually the closest match to the leaderboard metric. A verifier-guided rejection-sampling variant could generate multiple student rollouts, keep correct diverse solutions, and train on those solutions.

**Gate before implementation:** confirm that the assignment permits optimization beyond SFT and permits generated responses in the 15K subset. The main assignment asks for data selection and SFT; therefore this route could weaken submission compliance even if it improves scores. It also requires costly online or offline sampling and reliable reward/verifier behavior.

### 7. Further data scaling and verifier-based selection — useful, but more expensive

**Sources:** [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284), [OpenThoughts](https://arxiv.org/abs/2506.04178), and [LIMO](https://arxiv.org/abs/2502.03387).

AceReason-Nemotron reports that increasing unique prompt coverage matters more than adding responses to the same prompts. OpenThoughts emphasizes controlled data-generation and filtering recipes. LIMO shows that small, carefully curated reasoning datasets can be effective. These support model-aware selection using base solvability, prompt diversity, verified answers, complete target serialization, and domain coverage.

The compact-balanced failure shows that coarse domain quotas and short targets alone are insufficient. A future 15K selector should rank examples by a joint criterion: clean verified final answer; full 8K fit; intermediate base solvability rather than impossible targets; embedding-based novelty; and underrepresented skill coverage. It must freeze all weights and quotas before the public suite.

## Methods not worth repeating unchanged

- TrimSFT with the same margin/bandwidth and dataset: decisively worse on all seven completed columns.
- Compact post-thinking targets with the same quotas: termination improved but accuracy regressed broadly.
- The same final full-SFT checkpoint or a denser public-suite interpolation sweep: that would tune on public results and cannot establish a clean improvement.
- The same 2K guided-teacher recipe: it already lost to both base and matched existing targets on fresh confirmation.
- Longer chains merely for their length: this project measured large truncation and repetition increases, and length alone did not transfer reasoning skill.

## Execution policy

1. Preserve ordinary GEM and model-aware GEM as completed results.
2. Do not rerun either method unchanged.
3. Any successor must introduce verified skill coverage or multiple correct reasoning paths rather than another loss-only change or raw-NLL selector.
4. Keep only one GPU job active. Do not submit pilots or overlap full jobs.
5. Every completed candidate must produce all nine metrics, raw predictions, manifest matches, truncation/repetition diagnostics, paired uncertainty, seven-benchmark macro, and the strict integer domination receipt.
6. Treat paper-reported gains as motivation, never as expected project outcomes. No method is claimed to beat the historical leaderboard until this exact evaluator completes and the nine-column gate passes.

## Practical ranking

| Rank | Method | Direct fit to pass@8 | Base-retention mechanism | One-B200 practicality | Assignment fit |
|---:|---|---|---|---|---|
| 1 | GEM 8K | strong | entropy-preserving objective | completed partial gain | strong |
| 2 | Model-aware 15K + GEM | strong | model-relative difficulty plus diversity | completed negative | strongest data-selection fit |
| 3 | Anchored/CurioSFT | strong | KL or adaptive self-distillation | moderate to difficult | strong if kept as SFT |
| 4 | RPD/SSFT multi-solution | very strong | diversity by construction | data-generation heavy | conditional on unique-prompt rule |
| 5 | DCO/verifier RFT | strongest objective match | reward/KL dependent | expensive | requires instructor-rule check |
| 6 | PriFT-mass 8K | medium | strong frozen-prior support | completed negative | strong |

