"""Offline publication fixtures: tiny dummy files, no Hub/model/tokenizer calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from deadline import publish_huggingface as module


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


class FakeHub:
    def __init__(self):
        self.name, self.repos, self.writes, self.sequence = module.OWNER, {}, [], 0
        self.fail_commit_once = False
        self.fail_on_commit = None
        self.fail_create_once = False
        self.corrupt_public = False
        self.hide_public = False

    def whoami(self):
        return self.name

    def snapshot(self, repo_id, repo_type, *, public=False):
        if public and self.hide_public:
            return None
        value = copy.deepcopy(self.repos.get((repo_type, repo_id)))
        if value and public and self.corrupt_public:
            for name, info in value["files"].items():
                if name.endswith(".safetensors"):
                    info["lfs"]["sha256"] = "0" * 64
        return value

    def create(self, repo_id, repo_type):
        key = repo_type, repo_id
        if key in self.repos:
            raise RuntimeError("already exists")
        self.writes.append(("create", key))
        self.repos[key] = {"commit": "0" * 40, "files": {}}
        if self.fail_create_once:
            self.fail_create_once = False
            raise TimeoutError("synthetic lost repository creation response")

    def commit(self, repo_id, repo_type, paths, parent):
        key = repo_type, repo_id
        repo = self.repos[key]
        if repo["commit"] != parent:
            raise RuntimeError("parent mismatch")
        for name, path in paths.items():
            data = Path(path).read_bytes()
            large = name.endswith(".safetensors") or name == "data/train.json"
            repo["files"][name] = {"size": len(data), "blob_id": blob(data),
                "lfs": {"sha256": module.sha(data), "size": len(data)} if large else None}
        self.sequence += 1
        repo["commit"] = f"{self.sequence:040x}"
        self.writes.append(("commit", key, tuple(paths)))
        if self.fail_commit_once or self.fail_on_commit == self.sequence:
            self.fail_commit_once = False
            raise TimeoutError("synthetic accepted commit with lost response")
        return repo["commit"]


def fixture(root):
    """Tiny already-packaged stand-in; upstream scientific audits are mocked."""
    package = root / "publication-metadata"
    package.mkdir()
    refs, inventory = {}, {}
    module.write(root / "experiment_plan.json", {"base_revision": module.BASE_REVISION,
                                                "source_revision": "fixture-source"})
    module.write(root / "artifacts/selection_complete.json", {"selection_sha256": "selection-fixture"})
    module.write(root / "artifacts/selected/selection_manifest.json", {"fixture_only": True})
    for role in ("random", "advanced"):
        model, data, run = root / "runs" / role / "model", root / "artifacts/training" / role, root / "runs" / role
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(("synthetic-not-a-model-" + role).encode())
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            module.write(model / name, {"fixture_only": True})
        # These files must never be published even if present beside final files.
        (model / "training_args.bin").write_bytes(b"excluded")
        (run / "optimizer.pt").write_bytes(b"excluded")
        module.write(data / "dataset.json", [{"instruction": "fixture", "input": "", "output": "unchanged", "system": "fixture"}])
        for name in ("selected_ids.json", "cache_meta.json"):
            module.write(data / name, {"fixture_only": True})
        for name in ("manifest.json", "cache_audit.json", "consumption_final.json"):
            module.write(run / name, {"fixture_only": True})
        module.write(run / "complete.json", {"binding_sha256": role, "steps": 1875,
            "unique_examples": 15000, "example_presentations": 15000, "full_parameter": True,
            "final_model_verification": {"parameter_dtype": "float32"}})
        for name in ("manifest.json", "metrics.json"):
            module.write(root / "results" / role / name, {"fixture_only": True})
        refs[role] = {"folder": str(model.resolve()), "dataset": str((data / "dataset.json").resolve()),
                      "binding_sha256": role}
        for kind in ("models", "datasets"):
            card = package / kind / role / "README.md"
            card.parent.mkdir(parents=True)
            card.write_text("# Synthetic offline fixture\nBuilt with Qwen.\n")
    for path in root.rglob("*"):
        if path.is_file() and package not in path.parents and path.name not in ("optimizer.pt", "training_args.bin"):
            inventory[str(path.resolve())] = {"sha256": module.sha(path.read_bytes()), "bytes": path.stat().st_size}
    summary = {"matched_official_eight": True, "models": {r: {"fixture_only": True}
                for r in ("original", "random", "advanced")}, "optional_aime25_matched": False}
    module.write(package / "evaluation_summary.json", summary)
    module.write(package / "package_manifest.json", {"schema": "deadline-publication-metadata-v1",
        "status": "staged_metadata_not_uploaded", "uploaded": False, "rows_per_dataset": 15000,
        "cutoff": 8192, "selection_sha256": "selection-fixture", "verified_input_files": inventory,
        "model_and_dataset_references": refs})
    return package, summary


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package, summary = fixture(self.root)
        self.state, self.hub = self.root / "hf/publication", FakeHub()
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "offline-fixture",
            "SLURM_JOB_PARTITION": "normal_q", "SLURM_JOB_GPUS": "", "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.evaluation = patch.object(module, "verify_evaluations", return_value=summary)
        self.evaluation.start()
        self.addCleanup(self.evaluation.stop)

    def publish(self):
        return module.publish(self.root, self.package, self.state,
                              expected_owner=module.OWNER, hub=self.hub)

    def test_publish_exact_payload_license_and_four_public_receipts(self):
        receipt = self.publish()
        self.assertTrue(receipt["complete"])
        self.assertEqual(set(receipt["repositories"]), set(module.DESTINATIONS))
        self.assertTrue(all(r["public_access_verified"] for r in receipt["repositories"].values()))
        for key, entry in receipt["repositories"].items():
            names = set(entry["files"])
            self.assertNotIn("optimizer.pt", names)
            self.assertNotIn("training_args.bin", names)
            self.assertIn("provenance/selected_ids.json", names)
            if entry["repo_type"] == "model":
                self.assertEqual(entry["files"]["LICENSE"]["method"], "Git blob SHA1 and size")
                self.assertIn("NOTICE", names)
                self.assertIn("MODIFICATIONS.md", names)
                self.assertEqual(entry["files"]["model.safetensors"]["method"], "LFS SHA256 and size")
            else:
                self.assertEqual(entry["files"]["data/train.json"]["method"], "LFS SHA256 and size")
                self.assertIn("/datasets/", entry["url"])
        plan = module.read(self.state / "publication_plan.json")
        expected = dict(plan)
        expected.pop("publication_sha256")
        self.assertEqual(module.sha(module.canonical(expected)), plan["publication_sha256"])
        self.assertTrue(all(module.MANIFEST not in p["files"] for p in plan["destinations"].values()))

    def test_second_call_verifies_without_writing_remote_again(self):
        first = self.publish()
        writes = len(self.hub.writes)
        second = self.publish()
        self.assertEqual(first["publication_sha256"], second["publication_sha256"])
        self.assertEqual(len(self.hub.writes), writes)

    def test_lost_commit_response_resumes_by_remote_identity(self):
        self.hub.fail_commit_once = True
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())
        self.assertTrue(self.publish()["complete"])
        self.assertEqual(sum(w[0] == "create" for w in self.hub.writes), 4)

    def test_lost_large_weight_commit_resumes_without_reupload(self):
        self.hub.fail_on_commit = 3
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertTrue(self.publish()["complete"])
        random_weights = [w for w in self.hub.writes if w[0] == "commit" and
                          w[1][1].endswith(module.DESTINATIONS["random_model"][2]) and
                          "model.safetensors" in w[2]]
        self.assertEqual(len(random_weights), 1)

    def test_ambiguous_repo_creation_requires_manual_reconciliation(self):
        self.hub.fail_create_once = True
        with self.assertRaises(TimeoutError):
            self.publish()
        writes = len(self.hub.writes)
        with self.assertRaisesRegex(ValueError, "reconcile manually"):
            self.publish()
        self.assertEqual(len(self.hub.writes), writes)

    def test_conflict_in_last_destination_prevents_all_remote_writes(self):
        kind, _, name = module.DESTINATIONS["advanced_dataset"]
        self.hub.repos[(kind, module.OWNER + "/" + name)] = {"commit": "1" * 40,
            "files": {"foreign.txt": {"size": 0}}}
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_unknown_existing_empty_repo_is_not_adopted(self):
        kind, _, name = module.DESTINATIONS["random_model"]
        self.hub.repos[(kind, module.OWNER + "/" + name)] = {"commit": "1" * 40, "files": {}}
        with self.assertRaisesRegex(ValueError, "no matching publication identity"):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_conflicting_existing_owned_file_is_never_overwritten(self):
        self.publish()
        key = "model", module.OWNER + "/" + module.DESTINATIONS["random_model"][2]
        self.hub.repos[key]["files"]["model.safetensors"]["lfs"]["sha256"] = "f" * 64
        writes = len(self.hub.writes)
        with self.assertRaisesRegex(ValueError, "LFS SHA256"):
            self.publish()
        self.assertEqual(len(self.hub.writes), writes)

    def test_public_tree_mismatch_cannot_claim_success(self):
        self.hub.corrupt_public = True
        with self.assertRaisesRegex(ValueError, "LFS SHA256"):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())

    def test_authenticated_only_visibility_cannot_claim_success(self):
        self.hub.hide_public = True
        with self.assertRaisesRegex(ValueError, "Public remote tree"):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())

    def test_wrong_account_or_expected_owner_never_writes(self):
        self.hub.name = "another-account"
        with self.assertRaisesRegex(ValueError, "Authenticated"):
            self.publish()
        self.assertEqual(self.hub.writes, [])
        with self.assertRaisesRegex(ValueError, "explicitly"):
            module.publish(self.root, self.package, self.state, expected_owner="another-account", hub=self.hub)

    def test_login_or_gpu_allocation_rejected(self):
        for env in ({"SLURM_JOB_ID": ""}, {"SLURM_JOB_PARTITION": "login"}, {"SLURM_JOB_GPUS": "0,1"}):
            with self.subTest(env=env), patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "allocation"):
                self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_changed_sealed_input_prevents_network_writes(self):
        (self.root / "runs/random/model/model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_changed_card_cannot_reuse_same_publication_identity(self):
        self.publish()
        writes = len(self.hub.writes)
        frozen = {p: p.read_bytes() for p in (self.state / "generated").glob("*")}
        (self.package / "models/random/README.md").write_text("changed card. Built with Qwen.")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.publish()
        self.assertEqual(len(self.hub.writes), writes)
        self.assertTrue(all(p.read_bytes() == value for p, value in frozen.items()))

    def test_matched_summary_and_actual_audit_are_required(self):
        summary = module.read(self.package / "evaluation_summary.json")
        summary["matched_official_eight"] = False
        module.write(self.package / "evaluation_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "complete matched"):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_unsealed_model_sidecar_is_refused(self):
        (self.root / "runs/random/model/chat_template.jinja").write_text("new and unsealed")
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_regular_git_blob_and_mandatory_lfs_checks(self):
        data = b"small unchanged fixture"
        expected = {"sha256": module.sha(data), "bytes": len(data), "git_blob_sha1": blob(data), "require_lfs": False}
        self.assertEqual(module.verify_file({"size": len(data), "blob_id": blob(data)}, expected)["method"],
                         "Git blob SHA1 and size")
        with self.assertRaisesRegex(ValueError, "Git blob"):
            module.verify_file({"size": len(data), "blob_id": blob(data)}, dict(expected, require_lfs=True))

    def test_vendored_license_matches_exact_pinned_source(self):
        path = Path(module.__file__).parent / "licenses/Qwen2.5-3B-Instruct.LICENSE"
        self.assertEqual(module.sha(path.read_bytes()), module.LICENSE_SHA256)
        self.assertEqual(module.read(path.with_name(path.name + ".source.json"))["source_url"], module.LICENSE_URL)
        self.assertEqual(module.NOTICE.strip(), "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.")

    def test_sdk_adapter_refuses_redirected_canonical_namespace(self):
        hub = module.Hub.__new__(module.Hub)
        hub.not_found = FileNotFoundError
        hub.api = SimpleNamespace(repo_info=MagicMock(return_value=SimpleNamespace(
            id="another-account/moved-repository", private=False, gated=False, sha="a" * 40)),
            list_repo_tree=MagicMock(return_value=[]))
        with self.assertRaisesRegex(ValueError, "different canonical namespace"):
            hub.snapshot(module.OWNER + "/fixed-destination", "model")
        hub.api.list_repo_tree.assert_not_called()

    def test_sdk_anonymous_snapshot_uses_pinned_commit_and_no_token(self):
        repo_id = module.OWNER + "/fixed-destination"
        hub = module.Hub.__new__(module.Hub)
        hub.not_found = FileNotFoundError
        hub.api = SimpleNamespace(repo_info=MagicMock(return_value=SimpleNamespace(
            id=repo_id, private=False, gated=False, sha="a" * 40)),
            list_repo_tree=MagicMock(return_value=[]))
        hub.snapshot(repo_id, "model", public=True)
        hub.api.repo_info.assert_called_once_with(repo_id, repo_type="model", token=False)
        hub.api.list_repo_tree.assert_called_once_with(repo_id, repo_type="model", revision="a" * 40,
                                                     recursive=True, token=False)

    def test_cli_sdk_failure_does_not_echo_credentials(self):
        secret = "synthetic-hf-token-and-presigned-url"
        argv = ["publish_huggingface", "--expected-owner", module.OWNER, "--state", str(self.state)]
        with patch("sys.argv", argv), patch.object(module, "publish", side_effect=RuntimeError(secret)), \
                patch.object(module.signal, "signal"), patch.object(module.signal, "alarm"):
            with self.assertRaises(SystemExit) as error:
                module.main()
        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, (self.state / "failure.json").read_text())

    def test_cli_internal_conflict_reason_remains_actionable(self):
        argv = ["publish_huggingface", "--expected-owner", module.OWNER, "--state", str(self.state)]
        with patch("sys.argv", argv), patch.object(module, "publish", side_effect=module.PublicationError("Different canonical namespace")), \
                patch.object(module.signal, "signal"), patch.object(module.signal, "alarm"):
            with self.assertRaisesRegex(SystemExit, "Different canonical namespace"):
                module.main()
        self.assertEqual(module.read(self.state / "failure.json")["reason"], "Different canonical namespace")


class RealPackageHandoffTests(unittest.TestCase):
    def test_real_packager_seals_optional_sidecars_for_publisher(self):
        # The established fixture uses tiny dummy weight files and synthetic
        # aggregate scores; this is a schema handoff test, never real accuracy.
        from test_deadline_package import fixture as make_package_inputs
        from deadline.package_deliverables import build_package
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_package_inputs(root)
            plan = module.read(root / "experiment_plan.json")
            plan["base_revision"] = module.BASE_REVISION
            module.write(root / "experiment_plan.json", plan)
            preparation = module.read(root / "artifacts/preparation_complete.json")
            preparation["model_revision"] = module.BASE_REVISION
            module.write(root / "artifacts/preparation_complete.json", preparation)
            for role in ("random", "advanced"):
                model = root / "runs" / role / "model"
                for name in ("generation_config.json", "special_tokens_map.json"):
                    module.write(model / name, {"synthetic_fixture": True})
                manifest = module.read(root / "results" / role / "manifest.json")
                manifest["identity"].update({p.name: module.sha(p.read_bytes()) for p in model.iterdir() if p.is_file()})
                module.write(root / "results" / role / "manifest.json", manifest)
            package = root / "publication-metadata"
            build_package(root, package)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "offline-fixture", "SLURM_JOB_PARTITION": "normal_q",
                    "SLURM_JOB_GPUS": "", "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": ""}):
                receipt = module.publish(root, package, root / "hf/publication",
                                         expected_owner=module.OWNER, hub=FakeHub())
            for role in ("random", "advanced"):
                self.assertIn("generation_config.json", receipt["repositories"][role + "_model"]["files"])
                self.assertIn("special_tokens_map.json", receipt["repositories"][role + "_model"]["files"])


if __name__ == "__main__":
    unittest.main()
