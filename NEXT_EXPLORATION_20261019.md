# Successor exploration — October 19, 2026

This exploration began after the original full-length candidate missed its first frozen gate. Its purpose was to assess whether the sealed source pool already provided multiple original responses for each prompt, without using benchmark outcomes to select training examples.

`research/original_path_inventory.py` reads only original source fields, verifies the sealed source bytes and row count, groups exact prompts and NFC-normalized prompts separately, hashes response text, and exports no training data. It ignores unrelated outcome metadata and rejects malformed or duplicate source identities. Its synthetic checks cover order invariance, duplicate responses, variant counts, Unicode composition, poisoned metadata and invalid rows.

The actual completed source inventory found **83,037 prompt groups with one original response each and zero groups with two original response variants**, under both grouping modes. This is a negative feasibility result for constructing a multiple-original-response candidate from that pool. It does not establish semantic reasoning diversity, target correctness or a causal diagnosis of the original model's benchmark failures. Detailed receipts and source identifiers remain private.

Generating replacement responses would require a separately specified target policy, preparation and budget. No such generated-target candidate was launched. The later **57 additional / 97 total GPU-hour authorization applies to the two original-target plans described in [TWO_PLAN_SETUP_20261019.md](TWO_PLAN_SETUP_20261019.md)**, not a third generated-target experiment. Their current observed state is recorded in [CURRENT_RUNS_20261019.md](CURRENT_RUNS_20261019.md).

Do not rerun retired methods unchanged, relax earlier negative preparation contracts, select checkpoints or data from benchmark errors, or reinterpret an unmeasured result as zero. Original responses, the fixed evaluator and declared exposure remain the contracts for the two active plans. No strict nine-gate success or speed guarantee is inferred from this inventory.
