"""Synthetic artifact-chain checks for CPU-only selection publication."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deadline.build_selection import CUTOFF, SEED, build_selection, main
from deadline.compatible_coverage import select_random_control
from deadline.export_training import export as serial_export
from deadline.transfer_probe import SCHEMA, ScoreJournal, model_identity, transfer_gain, value_digest
from reasoning.data import digest, write_json
from test_deadline_selection import fixture, transfer_fixture


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [11, 12, 13, 14, 15] if messages[-1]["role"] == "assistant" else [11, 12]


def make_artifacts(root):
    artifacts, runs = root / "artifacts", root / "runs"
    artifacts.mkdir()
    model = root / "original-base"
    model.mkdir()
    write_json(model / "config.json", {"model_type": "qwen2"})
    (model / "model.safetensors").write_bytes(b"synthetic-identity-test-only")
    identity = model_identity(model)
    rows, features, feature_meta, heldout = fixture()
    scores, transfer = transfer_fixture(rows)
    transfer["probe_training_ids"] = {"probeA": transfer["probe_training_ids"]["probe_a"],
                                       "probeB": transfer["probe_training_ids"]["probe_b"]}
    transfer["fold_to_probe"] = {"0": "probeB", "1": "probeA"}
    for score in scores:
        score["probe_id"] = "probeB" if score["fold"] == "0" else "probeA"
        score["gain"] = transfer_gain(score["base_ce"], score["probe_ce"])
        score["supervised_tokens"] = 3
        score["fold"] = int(score["fold"])
    transfer.pop("probe_checkpoint_sha256")
    transfer.pop("probe_initialization_sha256")
    write_json(artifacts / "pool.json", rows)
    pool_sha = digest(artifacts / "pool.json")
    transfer.update(input_sha256=pool_sha, base_model_sha256=identity["identity_sha256"])
    feature_meta["input_sha256"] = pool_sha
    for name, value in (("features.json", features), ("features.json.meta.json", feature_meta),
                        ("heldout.json", heldout), ("transfer_meta.json", transfer),
                        ("model_identity.json", identity)):
        write_json(artifacts / name, value)
    write_json(artifacts / "finalized.json", dict(base_model_sha256=identity["identity_sha256"],
        feature_sha256=digest(artifacts / "features.json"),
        transfer_meta_sha256=digest(artifacts / "transfer_meta.json")))
    (artifacts / "transfer_cache.jsonl").write_text('{"synthetic_fixture":true}\n')
    cache_sha = digest(artifacts / "transfer_cache.jsonl")
    write_json(artifacts / "cache_meta.json", dict(model_path=str(model), cutoff=CUTOFF,
        cache_sha256=cache_sha, complete_original_targets=True, prefix_truncated_count=0))
    reserved = [identifier for identifiers in transfer["probe_training_ids"].values() for identifier in identifiers]
    random_rows = select_random_control(rows, heldout_rows=heldout, probe_ids=reserved, size=12, seed=SEED)["rows"]
    identifiers = [row["id"] for row in random_rows]
    write_json(artifacts / "random_ids.json", identifiers)
    write_json(artifacts / "training" / "random" / "selected_ids.json", identifiers)
    for name in ("probeA", "probeB"):
        path = runs / name
        adapter = path / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(name.encode())
        write_json(adapter / "adapter_config.json", {"synthetic_fixture": True})
        manifest = dict(schema=SCHEMA, probe_id=name, base=identity,
                        cache={"cache_sha256": cache_sha}, training_ids=transfer["probe_training_ids"][name],
                        training_fold=0 if name == "probeA" else 1, steps=64)
        receipt = dict(manifest, manifest_sha256=value_digest(manifest), complete=True,
                       completed_updates=64, unique_training_rows=1,
                       adapter={"files": {file.name: {"sha256": digest(file), "bytes": file.stat().st_size}
                                            for file in adapter.iterdir()}})
        write_json(path / "run_manifest.json", manifest)
        write_json(path / "probe_receipt.json", receipt)
    for fold in (0, 1):
        fold_scores = [score for score in scores if score["fold"] == fold]
        ids = [score["id"] for score in fold_scores]
        write_json(artifacts / f"fold{fold}_ids.json", ids)
        probe_name = "probeB" if fold == 0 else "probeA"
        manifest = dict(schema=SCHEMA, probe_id=probe_name, scoring_fold=fold, training_fold=1 - fold,
                        probe_receipt_sha256=digest(runs / probe_name / "probe_receipt.json"),
                        base=identity, cache={"cache_sha256": cache_sha}, candidate_ids=ids,
                        candidate_ids_file_sha256=digest(artifacts / f"fold{fold}_ids.json"), batch_size=4)
        batches = [ids[i:i + 4] for i in range(0, len(ids), 4)]
        journal = ScoreJournal(runs / f"scores_fold{fold}", manifest, batches)
        for index in range(len(batches)):
            journal.commit(index, fold_scores[index * 4:(index + 1) * 4])
        journal.finish(dict(probe_id=probe_name, scoring_fold=fold))
    return rows


class BuildSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = make_artifacts(self.root)
        self.environ = patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-cpu-test",
                                               "SLURM_JOB_PARTITION": "cpu",
                                               "SLURM_CPUS_PER_TASK": "8",
                                               "SLURM_JOB_GPUS": "", "SLURM_GPUS": "",
                                               "SLURM_GPUS_ON_NODE": ""})
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def run_builder(self, **kwargs):
        return build_selection(self.root, expected_rows=12,
                               tokenizer_factory=lambda _: FakeTokenizer(), **kwargs)

    def assert_unpublished(self):
        artifacts = self.root / "artifacts"
        self.assertFalse((artifacts / "selection_complete.json").exists())
        self.assertFalse((artifacts / "selected").exists())
        self.assertFalse((artifacts / "training" / "advanced").exists())

    def test_complete_artifact_chain_and_immutable_original_inputs(self):
        originals = {path: digest(path) for path in (self.root / "artifacts").rglob("*") if path.is_file()}
        receipt = self.run_builder()
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["rows_per_arm"], 12)
        self.assertEqual(receipt["cutoff"], 8192)
        self.assertTrue(receipt["random_control_ids_unchanged"])
        artifacts = self.root / "artifacts"
        manifest = json.loads((artifacts / "selected" / "selection_manifest.json").read_text())
        for name in ("probeA", "probeB"):
            provenance = manifest["transfer_provenance"]
            self.assertEqual(provenance["probe_initialization_sha256"][name], receipt["base_model_sha256"])
            self.assertEqual(provenance["probe_checkpoint_sha256"][name],
                             digest(self.root / "runs" / name / "adapter" / "adapter_model.safetensors"))
        meta = json.loads((artifacts / "training" / "advanced" / "cache_meta.json").read_text())
        self.assertEqual(meta["model_path"], str((self.root / "original-base").resolve()))
        self.assertEqual(meta["truncated_rows"], 0)
        self.assertEqual(meta["cutoff"], 8192)
        self.assertEqual(meta["selection_sha256"], receipt["selection_sha256"])
        for filename, checksum in receipt["files_sha256"].items():
            self.assertEqual(digest(artifacts / filename), checksum)
        self.assertTrue(all(digest(path) == checksum for path, checksum in originals.items()))

    def test_api_default_stays_serial_with_eight_allocated_cpus(self):
        with patch("deadline.build_selection.export", wraps=serial_export) as exporter:
            self.run_builder()
        self.assertEqual(exporter.call_args.kwargs["workers"], 1)
        self.assertEqual(exporter.call_args.kwargs["tokenizer_path"], (self.root / "original-base").resolve())

    def test_explicit_worker_count_reaches_advanced_export(self):
        def export_locally(*args, **kwargs):
            return serial_export(*args, **dict(kwargs, workers=1))
        with patch("deadline.build_selection.export", side_effect=export_locally) as exporter:
            self.run_builder(export_workers=3)
        self.assertEqual(exporter.call_args.kwargs["workers"], 3)
        self.assertEqual(exporter.call_args.args[4], "advanced")

    def test_cli_workers_follow_cpu_allocation_up_to_eight(self):
        for cpus, expected in (("", 1), ("1", 1), ("3", 3), ("16", 8)):
            with self.subTest(cpus=cpus), patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": cpus}), \
                    patch("sys.argv", ["build_selection", "--root", str(self.root)]), \
                    patch("deadline.build_selection.build_selection", return_value={}) as builder, \
                    patch("builtins.print"):
                main()
            builder.assert_called_once_with(self.root, export_workers=expected)

    def test_invalid_export_worker_count_fails_before_publication(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "positive integer"):
                self.run_builder(export_workers=workers)
        self.assert_unpublished()

    def test_login_or_gpu_allocation_rejected(self):
        for changes in ({"SLURM_JOB_ID": ""}, {"SLURM_JOB_PARTITION": "login"}, {"SLURM_JOB_GPUS": "0,1"}):
            with patch.dict(os.environ, changes), self.assertRaisesRegex(ValueError, "CPU|compute"):
                self.run_builder()
        self.assert_unpublished()

    def test_changed_base_or_probe_weights_rejected(self):
        (self.root / "original-base" / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original base"):
            self.run_builder()
        self.assert_unpublished()

    def test_changed_adapter_rejected(self):
        (self.root / "runs" / "probeA" / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "adapter file changed"):
            self.run_builder()
        self.assert_unpublished()

    def test_old_4k_scoring_cache_metadata_is_rejected(self):
        path = self.root / "artifacts" / "cache_meta.json"
        meta = json.loads(path.read_text())
        meta["cutoff"] = 4096
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, "complete original 8K"):
            self.run_builder()
        self.assert_unpublished()

    def test_changed_frozen_feature_file_rejected(self):
        path = self.root / "artifacts" / "features.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed after freezing"):
            self.run_builder()
        self.assert_unpublished()

    def test_incomplete_probe_rejected(self):
        path = self.root / "runs" / "probeA" / "probe_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["complete"] = False
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "not complete"):
            self.run_builder()
        self.assert_unpublished()

    def test_changed_score_output_rejected(self):
        path = self.root / "runs" / "scores_fold1" / "scores.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "score file is changed"):
            self.run_builder()
        self.assert_unpublished()

    def test_changed_score_part_rejected(self):
        path = self.root / "runs" / "scores_fold0" / "parts" / "00000000.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "score part changed"):
            self.run_builder()
        self.assert_unpublished()

    def test_scoring_bound_to_current_actual_probe_receipt(self):
        path = self.root / "runs" / "probeA" / "probe_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["extra_audit_note"] = "changed receipt after scoring"
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "different actual probe receipt"):
            self.run_builder()
        self.assert_unpublished()

    def test_random_order_must_match_already_exported_control(self):
        path = self.root / "artifacts" / "random_ids.json"
        write_json(path, list(reversed(json.loads(path.read_text()))))
        with self.assertRaisesRegex(ValueError, "random IDs differ"):
            self.run_builder()
        self.assert_unpublished()

    def test_staged_export_failure_publishes_nothing(self):
        with patch("deadline.build_selection.export", side_effect=ValueError("synthetic export failure")):
            with self.assertRaisesRegex(ValueError, "synthetic export failure"):
                self.run_builder()
        self.assert_unpublished()
        self.assertFalse(list((self.root / "artifacts").glob(".selection-stage-*")))

    def test_completed_artifacts_are_never_overwritten(self):
        self.run_builder()
        seal = self.root / "artifacts" / "selection_complete.json"
        old_hash = digest(seal)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.run_builder()
        self.assertEqual(digest(seal), old_hash)


if __name__ == "__main__":
    unittest.main()
