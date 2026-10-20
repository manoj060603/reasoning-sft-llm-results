---
task_categories:
- text-generation
size_categories:
- 10K<n<100K
---

# Reasoning SFT — random 15K original pairs

Source: [redsgnaoh/acereason11_100k](https://huggingface.co/datasets/redsgnaoh/acereason11_100k), revision `ce815bd726c0c4ffdc0912789754646910188195`. The dataset has 15,000 original instruction/input/output/system records. The original exported responses remain unchanged; this file is the dataset to upload, not a shortened token cache.

Uniform sample from the broad prepared source pool, without advanced boxed-answer, repetition, complete-target length, topic or transfer-gain filters.

Dataset SHA-256: `3709a2ece0468d5346130dc4e6075610fb11234cd5f407c2bf26a64d791abec7`. Selection seed: 20261019. Held-out source questions, benchmark-question overlap and reserved probe training examples are excluded.

At the shared 8K training cutoff, 5806 selected rows use prefixes and 5806 lack supervised EOS; these training effects do not rewrite this exported dataset.

The random control is uniform within the broad prepared source pool after deduplication, benchmark/source-heldout, prompt-capacity, and reserved-probe exclusions; it is not a literal uniform draw over the untouched original 100K rows. Original exported responses are unchanged. Random training may use declared 8K prefixes without an invented EOS. Advanced targets must fit completely. The advanced method combines eligibility filtering, reservoir construction, transfer gain, and coverage. Separate eligible-random and coverage-only ablations are needed to isolate their contributions. Relative teacher-forced loss reduction is a proxy, not mathematical verification or proof of causal data utility. Global novelty, state-of-the-art performance, and equivalence to a historical leaderboard are not established.

Upstream data attribution: [NVIDIA AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) lists CC BY 4.0. The actual class subset used here is credited separately to redsgnaoh/acereason11_100k. A separate subset license has not been verified or asserted.
