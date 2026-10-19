# Assistance candidate readiness — October 19, 2026

This is the new assistance-utility plus coverage experiment from Plan 1. It preserves the earlier categorical selector's negative result and does not relax its quotas.

Implementation: `research/assistance_coverage_20261019.py`; six focused selector tests: `research/tests/test_assistance_coverage_20261019.py`. The selector allocates capped cluster and response-length quotas before scoring, limits utility selection to half each quota cell, and fills remaining slots by deterministic source-ID hash. It uses exactly two supported normal and assisted observations per condition, Laplace smoothing, provenance checks and utility `(1 - p_normal) * max(p_assisted - p_normal, 0)`. Missing or censored measurements remain unscored. Full original targets are retained.

The completed static inventory established **56,673 eligible originals, 113 clusters, 23,014 supported scored prompts, 4,217 positive utilities, and 3,672 utility + 11,328 coverage selections**. This is not a complete native processor audit or accepted export.

At the public snapshot, CPU-only compute job **1207446** runs the full native audit. The stopped login-node attempt and its partial output were preserved; no partial output is treated as ready. The compute job alone may run the heavy `--prepare` validation and single guarded four-B200 training submission after all checks pass. Do not duplicate either operation or run them on a login node.

Training is one four-B200 six-hour allocation, original pinned base, 15K unique original prompts, 16K, plain response-only CE, LR 5e-6, three epochs / 5,625 updates, effective batch eight. Four ranks use batch one and accumulation two. That allocation is within the user-approved 97-hour total cap, which preserves prior charges and reserves final evaluation. No training job for this candidate had been submitted at the snapshot.

Mechanical prefix withholding and assistance association do not prove absence of semantic leakage or a causal learning benefit. Token exposure and final diagnostics must be reported. Actual GPU fit, throughput, completed training, final evaluation and strict nine-gate success remain unverified. See [current runs](../CURRENT_RUNS_20261019.md) and [implemented plans](../TWO_PLAN_SETUP_20261019.md).
