"""Seal fresh cross-fitted source selection after both score folds complete.

CPU compute allocation only. Default layout: deadline/artifacts and
deadline/runs/{probeA,probeB,scores_fold0,scores_fold1}. Inputs and prior random
exports are read-only; new selected data and advanced training cache are staged
and published only after all identity, reservation and completeness checks pass.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

from deadline.compatible_coverage import content_sha256, select_datasets
from deadline.export_training import export
from deadline.transfer_probe import SCHEMA, model_identity, value_digest
from reasoning.data import alpaca, digest, read_rows, write_json


SEED = 20261019
SIZE = 15000
CUTOFF = 8192


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compute_only():
    _require(bool(os.environ.get("SLURM_JOB_ID")) and
             not os.environ.get("SLURM_JOB_PARTITION", "").lower().startswith("login"),
             "Selection construction must run in a Slurm CPU compute allocation")
    for key in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        _require(os.environ.get(key, "") in ("", "0", "(null)", "N/A"),
                 "Use a CPU allocation without assigned GPUs for selection construction")
    # No CUDA computation is needed, including during local tokenizer loading.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _validate_model_identity(identity, model):
    expected = dict(identity)
    recorded = expected.pop("identity_sha256", None)
    _require(recorded == value_digest(expected), "Frozen base identity checksum is inconsistent")
    _require(model_identity(model) == identity, "Local original base differs from the frozen model identity")
    return recorded


def _validate_probe(path, name, identity, expected_ids):
    receipt_path = path / "probe_receipt.json"
    receipt, manifest = _read(receipt_path), _read(path / "run_manifest.json")
    _require(receipt.get("complete") is True and receipt.get("schema") == SCHEMA,
             f"{name}: probe receipt is not complete")
    _require(receipt.get("probe_id") == name, f"{name}: probe identity differs")
    _require(receipt.get("base") == identity and manifest.get("base") == identity,
             f"{name}: probe did not initialize from the frozen original base")
    _require(receipt.get("manifest_sha256") == value_digest(manifest),
             f"{name}: probe manifest checksum differs")
    _require(all(receipt.get(key) == value for key, value in manifest.items()),
             f"{name}: receipt differs from its actual training manifest")
    _require(receipt.get("training_ids") == expected_ids,
             f"{name}: actual probe training IDs differ from frozen source reservations")
    _require(receipt.get("completed_updates") == manifest.get("steps") and
             receipt.get("unique_training_rows") == len(expected_ids),
             f"{name}: not every reserved probe row received complete training")
    expected_fold = 0 if name == "probeA" else 1
    _require(receipt.get("training_fold") == expected_fold, f"{name}: training fold differs")
    files = receipt.get("adapter", {}).get("files", {})
    _require(isinstance(files, dict) and {"adapter_model.safetensors", "adapter_config.json"}.issubset(files),
             f"{name}: missing saved adapter weights/configuration identity")
    actual_names = {file.name for file in (path / "adapter").iterdir() if file.is_file()}
    _require(actual_names == set(files), f"{name}: saved adapter file set changed")
    for filename, info in files.items():
        _require(Path(filename).name == filename, f"{name}: invalid adapter filename")
        file = path / "adapter" / filename
        _require(file.stat().st_size == info.get("bytes") and digest(file) == info.get("sha256"),
                 f"{name}: saved adapter file changed: {filename}")
    return receipt, digest(receipt_path), digest(path / "adapter" / "adapter_model.safetensors")


def _validate_score_fold(path, fold, probe_name, probe_receipt_sha, identity, artifacts, cache_sha):
    manifest, receipt = _read(path / "run_manifest.json"), _read(path / "score_receipt.json")
    manifest_sha = value_digest(manifest)
    _require(receipt.get("schema") == SCHEMA and receipt.get("complete") is True and
             receipt.get("manifest_sha256") == manifest_sha,
             f"fold {fold}: scoring receipt is incomplete or has changed provenance")
    _require(manifest.get("schema") == SCHEMA and manifest.get("probe_id") == probe_name and
             manifest.get("scoring_fold") == fold and receipt.get("scoring_fold") == fold and
             receipt.get("probe_id") == probe_name,
             f"fold {fold}: scoring fold/probe differs")
    _require(manifest.get("training_fold") == 1 - fold,
             f"fold {fold}: same-fold probe scoring is forbidden")
    _require(manifest.get("probe_receipt_sha256") == probe_receipt_sha,
             f"fold {fold}: scoring used a different actual probe receipt")
    _require(manifest.get("base") == identity and manifest.get("cache", {}).get("cache_sha256") == cache_sha,
             f"fold {fold}: scoring base/cache differs from frozen preparation")
    id_file = artifacts / f"fold{fold}_ids.json"
    expected_ids = _read(id_file)
    candidate_ids = manifest.get("candidate_ids", [])
    _require(isinstance(expected_ids, list) and isinstance(candidate_ids, list) and
             len(set(candidate_ids)) == len(candidate_ids) == len(expected_ids) and
             set(candidate_ids) == set(expected_ids) and
             manifest.get("candidate_ids_file_sha256") == digest(id_file),
             f"fold {fold}: candidate identities differ from the frozen fold file")
    scores = read_rows(path / "scores.jsonl")
    _require(receipt.get("scores_sha256") == digest(path / "scores.jsonl") and
             receipt.get("records") == len(scores) == len(candidate_ids) and
             [score.get("id") for score in scores] == candidate_ids,
             f"fold {fold}: final score file is changed or incomplete")
    batch_size = manifest.get("batch_size")
    _require(type(batch_size) is int and batch_size > 0, f"fold {fold}: invalid scoring batch size")
    expected_parts = [f"{index:08d}.json" for index in range((len(scores) + batch_size - 1) // batch_size)]
    part_hashes = receipt.get("parts_sha256", {})
    _require(set(part_hashes) == set(expected_parts), f"fold {fold}: incomplete score-part seal")
    for index, filename in enumerate(expected_parts):
        part_path = path / "parts" / filename
        _require(digest(part_path) == part_hashes[filename], f"fold {fold}: score part changed")
        part = _read(part_path)
        expected_records = scores[index * batch_size:(index + 1) * batch_size]
        _require(part.get("manifest_sha256") == manifest_sha and
                 part.get("records") == expected_records and
                 part.get("records_sha256") == value_digest(expected_records),
                 f"fold {fold}: score parts do not reproduce the sealed score file")
    return scores, {"receipt_sha256": digest(path / "score_receipt.json"),
                    "manifest_sha256": digest(path / "run_manifest.json"),
                    "scores_sha256": digest(path / "scores.jsonl"), "rows": len(scores)}


def build_selection(root=Path("deadline"), *, expected_rows=SIZE, tokenizer_factory=None,
                    export_workers=1):
    _require(type(export_workers) is int and export_workers >= 1,
             "Export workers must be a positive integer")
    _compute_only()
    root = Path(root).resolve()
    artifacts, runs = root / "artifacts", root / "runs"
    destination, training = artifacts / "selected", artifacts / "training" / "advanced"
    complete = artifacts / "selection_complete.json"
    _require(not any(path.exists() for path in (destination, training, complete)),
             "Refusing to overwrite existing selected data, advanced training cache, or completion seal")
    _require((artifacts / "finalized.json").is_file(), "Source preparation must be finalized before selection")
    cache_meta = _read(artifacts / "cache_meta.json")
    model = Path(cache_meta["model_path"]).resolve()
    identity = _read(artifacts / "model_identity.json")
    base_sha = _validate_model_identity(identity, model)
    _require(cache_meta.get("cutoff") == CUTOFF and cache_meta.get("complete_original_targets") is True and
             cache_meta.get("prefix_truncated_count") == 0,
             "Scoring cache must contain audited complete original 8K targets")
    cache_sha = digest(artifacts / "transfer_cache.jsonl")
    _require(cache_meta.get("cache_sha256") == cache_sha, "Frozen source scoring cache has changed")
    transfer_meta = _read(artifacts / "transfer_meta.json")
    _require(transfer_meta.get("base_model_sha256") == base_sha,
             "Frozen transfer metadata names another original base")
    pool_path, feature_path = artifacts / "pool.json", artifacts / "features.json"
    finalized = _read(artifacts / "finalized.json")
    _require(finalized.get("base_model_sha256") == base_sha and
             finalized.get("feature_sha256") == digest(feature_path) and
             finalized.get("transfer_meta_sha256") == digest(artifacts / "transfer_meta.json"),
             "Finalized source features/transfer design changed after freezing")
    pool_sha = digest(pool_path)
    _require(transfer_meta.get("input_sha256") == pool_sha,
             "Frozen transfer metadata names another source pool")
    heldout_path = artifacts / "heldout.json"
    rows, heldout = read_rows(pool_path), read_rows(heldout_path)
    reservations = transfer_meta.get("probe_training_ids", {})
    _require(set(reservations) == {"probeA", "probeB"}, "Both frozen probe reservations are required")
    probes, probe_files, checkpoints = {}, {}, {}
    for name in ("probeA", "probeB"):
        probe, receipt_sha, checkpoint_sha = _validate_probe(runs / name, name, identity, reservations[name])
        _require(probe.get("cache", {}).get("cache_sha256") == cache_sha,
                 f"{name}: actual probe trained from a different source cache")
        probes[name], checkpoints[name] = probe, checkpoint_sha
        probe_files[name] = {"receipt_sha256": receipt_sha, "checkpoint_sha256": checkpoint_sha}
    score_rows, score_files = [], {}
    for fold in (0, 1):
        name = "probeB" if fold == 0 else "probeA"
        scores, files = _validate_score_fold(runs / f"scores_fold{fold}", fold, name,
                                            probe_files[name]["receipt_sha256"], identity, artifacts, cache_sha)
        score_rows.extend(scores)
        score_files[f"fold{fold}"] = files
    # Values come from real, complete probe artifacts, never from placeholder metadata.
    transfer_meta["probe_initialization_sha256"] = {
        name: probe["base"]["identity_sha256"] for name, probe in probes.items()}
    transfer_meta["probe_checkpoint_sha256"] = checkpoints
    transfer_meta["verified_probe_artifacts"] = probe_files
    transfer_meta["verified_score_artifacts"] = score_files
    result = select_datasets(rows, read_rows(feature_path),
        feature_meta=_read(artifacts / "features.json.meta.json"), pool_sha256=pool_sha,
        heldout_rows=heldout, size=expected_rows, cutoff=CUTOFF, seed=SEED,
        strategy="transfer_gain", transfer_scores=score_rows, transfer_meta=transfer_meta)
    manifest = result["manifest"]
    random_ids = manifest["datasets"]["random"]["selected_ids"]
    _require(random_ids == _read(artifacts / "random_ids.json"),
             "Final random IDs differ from the already exported frozen random control")
    random_training_ids = artifacts / "training" / "random" / "selected_ids.json"
    _require(random_training_ids.is_file() and _read(random_training_ids) == random_ids,
             "Previously exported random training IDs differ from the final control")
    source_files = [pool_path, feature_path, artifacts / "features.json.meta.json", heldout_path,
                    artifacts / "random_ids.json", artifacts / "finalized.json",
                    artifacts / "cache_meta.json", artifacts / "transfer_meta.json",
                    artifacts / "model_identity.json"]
    manifest["build_source_file_sha256"] = {str(path.relative_to(root)): digest(path) for path in source_files}
    manifest["builder_code_sha256"] = digest(__file__)
    manifest.pop("selection_sha256")
    manifest["selection_sha256"] = content_sha256(manifest)
    if tokenizer_factory is None:
        from transformers import AutoTokenizer
        tokenizer_factory = lambda path: AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False)
    tokenizer = tokenizer_factory(model)
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".selection-stage-", dir=artifacts) as temporary:
        staged = Path(temporary)
        staged_selected, staged_training = staged / "selected", staged / "advanced"
        for name, records in result["datasets"].items():
            write_json(staged_selected / f"{name}.json", alpaca(records))
        write_json(staged_selected / "dataset_info.json", result["dataset_info"])
        write_json(staged_selected / "selection_manifest.json", manifest)
        advanced_ids = manifest["datasets"]["advanced"]["selected_ids"]
        exported = export(rows, advanced_ids, tokenizer, staged_training, "advanced",
                          cutoff=CUTOFF, expected_rows=expected_rows,
                          workers=export_workers, tokenizer_path=model)
        exported.update(pool_sha256=pool_sha, model=str(model), model_path=str(model),
                        selection_source_sha256=digest(staged_selected / "selection_manifest.json"),
                        selection_sha256=manifest["selection_sha256"],
                        base_model_sha256=base_sha, tokenizer_sha256=transfer_meta["tokenizer_sha256"])
        write_json(staged_training / "cache_meta.json", exported)
        _require(exported["rows"] == expected_rows and exported["truncated_rows"] == 0 and
                 exported["unique_rows_dropped"] == 0 and exported["target_rewriting"] is False,
                 "Advanced training export lost or truncated an original target")
        for name in ("advanced", "random"):
            _require(content_sha256(read_rows(staged_selected / f"{name}.json")) ==
                     manifest["datasets"][name]["exported_rows_sha256"],
                     f"{name}: staged target export differs from selected originals")
        _require(_read(staged_training / "selected_ids.json") == advanced_ids and
                 digest(staged_training / "training.jsonl") == exported["cache_sha256"] and
                 digest(staged_training / "dataset.json") == exported["dataset_sha256"],
                 "Advanced training cache identity verification failed")
        output_hashes = {
            **{f"selected/{path.name}": digest(path) for path in sorted(staged_selected.iterdir()) if path.is_file()},
            **{f"training/advanced/{path.name}": digest(path) for path in sorted(staged_training.iterdir()) if path.is_file()}}
        _require(all(digest(root / relative) == checksum
                     for relative, checksum in manifest["build_source_file_sha256"].items()),
                 "Frozen selection input changed while the training cache was being exported")
        (artifacts / "training").mkdir(exist_ok=True)
        _require(not destination.exists() and not training.exists(), "Selection output appeared during construction")
        staged_selected.rename(destination)
        staged_training.rename(training)
    completion = dict(schema="deadline-crossfit-selection-complete-v1", complete=True,
        method="cross_fitted_transfer_gain_with_coverage", rows_per_arm=expected_rows,
        seed=SEED, cutoff=CUTOFF, selection_sha256=manifest["selection_sha256"],
        base_model_sha256=base_sha, random_control_ids_unchanged=True,
        advanced_training_rows=exported["rows"], advanced_truncated_rows=0,
        files_sha256=output_hashes, probe_artifacts=probe_files, score_artifacts=score_files)
    _require(not complete.exists(), "Selection completion seal appeared during construction")
    write_json(complete, completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("deadline"))
    args = parser.parse_args()
    workers = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK") or "1"))
    print(json.dumps(build_selection(args.root, export_workers=workers), indent=2), flush=True)


if __name__ == "__main__":
    main()
