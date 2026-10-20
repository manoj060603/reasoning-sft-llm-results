"""Finalize producer schemas and the frozen random control in a CPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import random

from reasoning import features as feature_module
from reasoning.data import digest, question, read_rows, write_json
from reasoning.prepare import ExclusionIndex
from deadline.compatible_coverage import _heldout_exclusions, _validate_pool, eligible
from deadline.export_training import export
from deadline.prepare_transfer import order_key
from deadline.transfer_probe import model_identity
from deadline.bootstrap_evaluation import ensure_benchmarks
from deadline.pipeline import submit


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use a compute allocation for model/data auditing.")
    root = Path("deadline/artifacts")
    if (root / "finalized.json").exists():
        print("Preparation already finalized; retained unchanged.")
        return
    preparation = json.loads((root / "preparation_complete.json").read_text())
    cache_meta = json.loads((root / "cache_meta.json").read_text())
    model = Path(cache_meta["model_path"])
    identity = model_identity(model)
    base_hash = identity["identity_sha256"]
    write_json(root / "model_identity.json", identity)
    ensure_benchmarks(root)
    extra_index = ExclusionIndex([question(row) for row in read_rows(root / "benchmarks/aime25/test.jsonl")])
    original_pool = read_rows(root / "pool.json")
    extra_excluded = {row["id"] for row in original_pool if extra_index.matches(row["instruction"])}
    pool = [row for row in original_pool if row["id"] not in extra_excluded]
    heldout = [row for row in read_rows(root / "heldout.json") if not extra_index.matches(row["instruction"])]
    write_json(root / "pool.json", pool)
    write_json(root / "heldout.json", heldout)
    feature_path = root / "features.json"
    features = [feature for feature in read_rows(feature_path) if feature["id"] not in extra_excluded]
    # A noneligible row has no fitted topic. Give it a nonnegative sentinel
    # outside the 64 real cluster IDs; eligibility still forbids its acquisition.
    for feature in features:
        if feature["cluster"] < 0:
            feature["cluster"] = 64
    # Assignment control: sample the broad prepared source pool uniformly.
    # Complete-target/quality eligibility applies only to advanced acquisition.
    # Original random exports remain unchanged; the common 8K training budget
    # may truncate their tokenized targets, which is counted and disclosed.
    write_json(feature_path, features)
    meta_path = root / "features.json.meta.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("provenance", {}).setdefault("producer_code_sha256", meta["provenance"].get("code_sha256"))
    meta["provenance"]["code_sha256"] = digest(feature_module.__file__)
    meta["provenance"]["finalizer_code_sha256"] = digest(__file__)
    meta["input_sha256"] = digest(root / "pool.json")
    write_json(meta_path, meta)
    transfer_meta = json.loads((root / "transfer_meta.json").read_text())
    # A scheduler-only retry may start from the already-pruned source pool.
    # Retain the original exclusion audit instead of replacing it with [].
    extra_excluded |= set(transfer_meta.get("provenance", {}).get("supplementary_question_excluded_ids", []))
    transfer_meta["base_model_sha256"] = base_hash
    transfer_meta["input_sha256"] = digest(root / "pool.json")
    transfer_meta["base_model_revision"] = cache_meta["model_revision"]
    transfer_meta["fold_assignments"] = {key: str(value) for key, value in transfer_meta["fold_assignments"].items()}
    pool_by_id = {row["id"]: row for row in pool}
    feature_by_id = {feature["id"]: feature for feature in features}
    def acquisition_valid(identifier):
        return identifier in pool_by_id and eligible(pool_by_id[identifier], feature_by_id[identifier], 8192)
    reservoir = [identifier for identifier in transfer_meta["reservoir_ids"] if acquisition_valid(identifier)]
    all_probes = {identifier for ids in transfer_meta["probe_training_ids"].values() for identifier in ids}
    for name, old_ids in transfer_meta["probe_training_ids"].items():
        retained = [identifier for identifier in old_ids if acquisition_valid(identifier)]
        fold = "0" if name == "probeA" else "1"
        replacements = sorted((identifier for identifier in reservoir
                               if identifier not in all_probes and transfer_meta["fold_assignments"][identifier] == fold),
                              key=lambda identifier: order_key(identifier, name, preparation["arguments"]["seed"]))
        retained.extend(replacements[:len(old_ids) - len(retained)])
        if len(retained) != len(old_ids):
            raise ValueError("Supplementary benchmark exclusion leaves insufficient probe capacity.")
        transfer_meta["probe_training_ids"][name] = retained
        write_json(root / f"{name}_ids.json", retained)
    all_probes = {identifier for ids in transfer_meta["probe_training_ids"].values() for identifier in ids}
    reservoir = [identifier for identifier in reservoir if identifier not in all_probes]
    if len(reservoir) < 15000:
        raise ValueError("Supplementary benchmark exclusions leave fewer than 15K acquisition candidates.")
    transfer_meta["reservoir_ids"] = reservoir
    used = set(reservoir) | all_probes
    transfer_meta["fold_assignments"] = {key: value for key, value in transfer_meta["fold_assignments"].items() if key in used}
    for fold in (0, 1):
        write_json(root / f"fold{fold}_ids.json", [identifier for identifier in reservoir
                     if transfer_meta["fold_assignments"][identifier] == str(fold)])
    transfer_meta["provenance"]["supplementary_question_excluded_ids"] = sorted(extra_excluded)
    transfer_meta["probe_initialization_sha256"] = {key: base_hash for key in transfer_meta["probe_training_ids"]}
    write_json(root / "transfer_meta.json", transfer_meta)
    cache_meta["base_model_sha256"] = base_hash
    write_json(root / "cache_meta.json", cache_meta)
    by_id, _ = _validate_pool(pool, features, meta, digest(root / "pool.json"))
    excluded, _ = _heldout_exclusions(pool, heldout)
    probes = {identifier for identifiers in transfer_meta["probe_training_ids"].values() for identifier in identifiers}
    available = [by_id[key] for key in sorted(by_id) if key not in excluded | probes]
    selected = random.Random(preparation["arguments"]["seed"]).sample(available, 15000)
    identifiers = [row["id"] for row in selected]
    write_json(root / "random_ids.json", identifiers)
    # Freeze every scientific input and both arms' IDs before any probe starts.
    # Only the deterministic random token-cache export remains after this seal.
    write_json(root / "probe_inputs_ready.json", {
        "base_model_sha256": base_hash, "cutoff": 8192,
        "input_sha256": digest(root / "pool.json"),
        "feature_sha256": digest(feature_path),
        "transfer_meta_sha256": digest(root / "transfer_meta.json"),
        "cache_meta_sha256": digest(root / "cache_meta.json"),
        "cache_sha256": cache_meta["cache_sha256"],
        "random_ids_sha256": digest(root / "random_ids.json"),
        "random_selection_frozen_before_probe_training": True})
    submit("probes_and_scores", Path("deadline/gpu_probe.slurm"), gpu_hours=12, required_hours=20)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    workers = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    random_meta = export(pool, identifiers, tokenizer, root / "training/random", "random", cutoff=8192,
                         workers=workers, tokenizer_path=model)
    random_meta.update(pool_sha256=digest(root / "pool.json"), model=str(model), base_model_sha256=base_hash)
    write_json(root / "training/random/cache_meta.json", random_meta)
    write_json(root / "finalized.json", {"base_model_sha256": base_hash,
        "random_rows": 15000, "random_truncated_rows": random_meta["truncated_rows"],
        "supplementary_question_excluded_ids": sorted(extra_excluded),
        "source_pool_after_all_exclusions": len(pool), "source_heldout_after_all_exclusions": len(heldout),
        "random_population": "uniform broad prepared source pool after shared deduplication, benchmark/source-heldout, prompt-capacity, and probe exclusions",
        "uniform_over_original_100k": False,
        "complete_target_eligibility_applies_only_to_advanced": True,
        "same_training_cutoff_for_both_arms": 8192,
        "feature_sha256": digest(feature_path), "transfer_meta_sha256": digest(root / "transfer_meta.json")})
    print(json.dumps({"base_model_sha256": base_hash, "random_rows": 15000,
                      "random_truncated_rows": random_meta["truncated_rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
