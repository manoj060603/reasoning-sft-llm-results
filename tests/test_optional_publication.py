"""Offline optional-publication fixtures; no model, native runtime or network."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deadline import publish_huggingface as primary
from deadline import publish_optional_model as module
from deadline.audit_results import EXPECTED
from deadline.package_deliverables import Files


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def primary_receipt():
    repositories = {}
    for role, (kind, _, name) in primary.DESTINATIONS.items():
        repo_id = primary.OWNER + "/" + name
        prefix = "datasets/" if kind == "dataset" else ""
        repositories[role] = dict(repo_id=repo_id, repo_type=kind,
            url="https://huggingface.co/" + prefix + repo_id, commit="a" * 40,
            public_access_verified=True, complete=True,
            files={"README.md": {"method": "Git blob SHA1 and size", "bytes": 1}})
    return dict(schema="deadline-hf-publication-complete-v1", complete=True,
        owner=primary.OWNER, publication_sha256="1" * 64,
        evaluated_and_published_weight_bytes_identical=True, repositories=repositories)


def raw_run():
    return dict(errors=[], manifest={"n": 8}, benchmarks={name: dict(
        status="complete", evidence="raw eight-sample predictions verified", errors=[],
        metrics=dict(problems=count, samples_per_problem=8, pass_at_1=0., pass_at_8=0.))
        for name, count in EXPECTED.items()})


def merge_receipt():
    weights = {f"model-{i:05d}-of-00006.safetensors": f"{i:064x}" for i in range(1, 7)}
    sidecars = {name: f"{i:064x}" for i, name in enumerate((
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "LICENSE", "NOTICE", "model.safetensors.index.json"), start=10)}
    return dict(schema="deadline-base-anchor-merge-v1", complete=True,
        alpha_advanced=.25, alpha_base=.75,
        posthoc_after_primary_benchmark_observations=True,
        primary_plan_sha256="2" * 64, recovery_plan_sha256="3" * 64,
        merge_inputs_sha256="4" * 64,
        inputs={"base": {"identity_sha256": "5" * 64},
                "advanced": {"training_receipt_sha256": "6" * 64}},
        output=dict(finite_fp32_tensors=434, all_finite=True, parameter_dtype="float32",
            weight_files=weights,
            files={**weights, **sidecars}))


class EvidenceGatesTests(unittest.TestCase):
    def test_complete_primary_publication_is_accepted_without_mutation(self):
        value = primary_receipt()
        before = copy.deepcopy(value)
        module.validate_primary_receipt(value)
        self.assertEqual(value, before)

    def test_primary_receipt_rejects_missing_or_substituted_destination(self):
        value = primary_receipt()
        del value["repositories"]["advanced_dataset"]
        with self.assertRaises(ValueError):
            module.validate_primary_receipt(value)
        value = primary_receipt()
        value["repositories"]["random_model"]["repo_id"] = module.REPO_ID
        with self.assertRaises(ValueError):
            module.validate_primary_receipt(value)

    def test_primary_receipt_cannot_swap_same_type_roles(self):
        value = primary_receipt()
        a, b = value["repositories"]["random_model"], value["repositories"]["advanced_model"]
        value["repositories"]["random_model"], value["repositories"]["advanced_model"] = b, a
        with self.assertRaises(ValueError):
            module.validate_primary_receipt(value)

    def test_primary_receipt_requires_public_complete_exact_weight_evidence(self):
        mutations = (
            lambda v: v.update(schema="another-publication-schema"),
            lambda v: v.update(complete=False),
            lambda v: v.update(owner="another-account"),
            lambda v: v.update(evaluated_and_published_weight_bytes_identical=False),
            lambda v: v["repositories"]["random_model"].update(public_access_verified=False),
            lambda v: v["repositories"]["advanced_model"].update(complete=False),
            lambda v: v["repositories"]["random_dataset"].update(repo_type="model"),
            lambda v: v["repositories"]["advanced_dataset"].update(files={}),
        )
        for mutate in mutations:
            value = primary_receipt()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_primary_receipt(value)

    def test_all_nine_raw_benchmarks_are_accepted(self):
        value = raw_run()
        before = copy.deepcopy(value)
        module.require_all_nine_raw(value)
        self.assertEqual(value, before)

    def test_missing_supplementary_ninth_task_is_rejected(self):
        value = raw_run()
        del value["benchmarks"]["aime25"]
        with self.assertRaises(ValueError):
            module.require_all_nine_raw(value)

    def test_aggregate_only_receipt_cannot_substitute_for_raw_predictions(self):
        value = raw_run()
        value["benchmarks"]["gpqa"]["evidence"] = "aggregate receipt only; raw predictions unavailable"
        with self.assertRaises(ValueError):
            module.require_all_nine_raw(value)

    def test_errors_or_wrong_eight_sample_coverage_are_rejected(self):
        mutations = (
            lambda v: v.update(errors=["manifest mismatch"]),
            lambda v: v["benchmarks"]["gpqa"].update(errors=["raw mismatch"]),
            lambda v: v["benchmarks"]["gpqa"].update(status="invalid"),
            lambda v: v["benchmarks"]["gpqa"]["metrics"].update(samples_per_problem=7),
            lambda v: v["benchmarks"]["gpqa"]["metrics"].update(problems=197),
        )
        for mutate in mutations:
            value = raw_run()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.require_all_nine_raw(value)

    def test_fixed_complete_posthoc_merge_is_accepted(self):
        value = merge_receipt()
        before = copy.deepcopy(value)
        module.validate_recovery_receipt(value)
        self.assertEqual(value, before)

    def test_changed_merge_policy_or_incomplete_fp32_output_is_rejected(self):
        mutations = (
            lambda v: v.update(schema="another-merge-schema"),
            lambda v: v.update(complete=False),
            lambda v: v.update(alpha_advanced=.5),
            lambda v: v.update(alpha_base=.5),
            lambda v: v.update(posthoc_after_primary_benchmark_observations=False),
            lambda v: v.update(primary_plan_sha256=""),
            lambda v: v.update(recovery_plan_sha256=""),
            lambda v: v["output"].update(finite_fp32_tensors=433),
            lambda v: v["output"].update(all_finite=False),
            lambda v: v["output"].update(parameter_dtype="bfloat16"),
            lambda v: v["output"]["weight_files"].pop("model-00006-of-00006.safetensors"),
            lambda v: v["output"]["weight_files"].update({"model-00001-of-00006.safetensors": ""}),
            lambda v: v["output"]["files"].pop("model.safetensors.index.json"),
            lambda v: v["output"]["files"].pop("tokenizer.json"),
            lambda v: v["output"]["files"].pop("NOTICE"),
        )
        for mutate in mutations:
            value = merge_receipt()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_recovery_receipt(value)

    def test_publication_notice_adds_upstream_attribution_and_preserves_original_bytes(self):
        original = b"Built with Qwen.\nOriginal post-hoc merge notice.\r\n\xff"
        self.assertEqual(module.publication_notice(original), primary.NOTICE.encode() + b"\n" + original)
        self.assertEqual(original, b"Built with Qwen.\nOriginal post-hoc merge notice.\r\n\xff")


class FakeHub:
    """In-memory remote trees, including accepted requests with lost responses."""
    def __init__(self):
        self.name = primary.OWNER
        self.repos, self.writes, self.sequence = {}, [], 0
        self.fail_on_commit = None
        self.fail_create_once = False
        self.hide_public = False
        self.corrupt_public = False

    def whoami(self):
        return self.name

    def snapshot(self, repo_id, repo_type, *, public=False):
        if public and self.hide_public:
            return None
        value = copy.deepcopy(self.repos.get((repo_type, repo_id)))
        if public and value and self.corrupt_public:
            for name, info in value["files"].items():
                if name.endswith(".safetensors"):
                    info["lfs"]["sha256"] = "0" * 64
        return value

    def create(self, repo_id, repo_type):
        key = repo_type, repo_id
        if key in self.repos:
            raise RuntimeError("already exists")
        self.repos[key] = {"commit": "0" * 40, "files": {}}
        self.writes.append(("create", key))
        if self.fail_create_once:
            self.fail_create_once = False
            raise TimeoutError("synthetic lost creation response")

    def commit(self, repo_id, repo_type, paths, parent):
        key = repo_type, repo_id
        repo = self.repos[key]
        if repo["commit"] != parent:
            raise RuntimeError("parent mismatch")
        for name, path in paths.items():
            data = Path(path).read_bytes()
            large = name.endswith(".safetensors")
            repo["files"][name] = dict(size=len(data), blob_id=blob(data),
                lfs={"sha256": primary.sha(data), "size": len(data)} if large else None)
        self.sequence += 1
        repo["commit"] = f"{self.sequence:040x}"
        self.writes.append(("commit", key, tuple(paths)))
        if self.fail_on_commit == self.sequence:
            raise TimeoutError("synthetic accepted commit with lost response")
        return repo["commit"]


class OptionalTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recovery = self.root / "optional-input"
        self.recovery.mkdir()
        self.state = self.root / "optional-publication-state"
        self.primary_state = self.root / "hf/publication"
        primary.write(self.primary_state / "publication_complete.json", primary_receipt())
        self.primary_path = self.primary_state / "publication_complete.json"
        self.primary_bytes = self.primary_path.read_bytes()
        self.destinations_before = copy.deepcopy(primary.DESTINATIONS)
        self.files = Files()
        paths = {}
        for name, data in {
            "README.md": b"# Offline optional fixture\nBuilt with Qwen.\nPost-hoc.\n",
            "NOTICE": module.publication_notice(b"Original recovery notice.\n"),
            "model-00001-of-00002.safetensors": b"synthetic-first-weight-not-a-model",
            "model-00002-of-00002.safetensors": b"synthetic-second-weight-not-a-model",
            module.MANIFEST: b'{"offline_optional_identity": true}\n',
        }.items():
            path = self.recovery / name
            path.write_bytes(data)
            paths[name] = str(path.resolve())
        descriptors = {name: dict(primary.fingerprint(path, self.files),
            require_lfs=name.endswith(".safetensors")) for name, path in paths.items()}
        self.payload = dict(repo_id=module.REPO_ID, repo_type="model", role="optional_recovery",
                            files=descriptors, local_paths=paths)
        self.identity = dict(schema="deadline-optional-recovery-publication-v1",
            publication_sha256="8" * 64, merge_receipt_sha256="9" * 64,
            evaluation_summary_sha256="a" * 64, model_identity_sha256="b" * 64)
        self.hub = FakeHub()
        env = dict(SLURM_JOB_ID="offline-optional-fixture", SLURM_JOB_PARTITION="normal_q",
                   SLURM_JOB_GPUS="", SLURM_GPUS="", SLURM_GPUS_ON_NODE="")
        environment = patch.dict(os.environ, env)
        environment.start()
        self.addCleanup(environment.stop)
        prepared = patch.object(module, "prepare", side_effect=lambda *args: (
            copy.deepcopy(self.identity), copy.deepcopy(self.payload), self.files))
        prepared.start()
        self.addCleanup(prepared.stop)

    def publish(self, expected_owner=None):
        return module.publish(self.root, self.recovery, self.state,
            expected_owner=module.OWNER if expected_owner is None else expected_owner,
            primary_receipt=self.primary_path, hub=self.hub)

    def assert_primary_untouched(self):
        self.assertEqual(primary.DESTINATIONS, self.destinations_before)
        self.assertEqual(self.primary_path.read_bytes(), self.primary_bytes)
        self.assertEqual(set(self.primary_state.iterdir()), {self.primary_path})
        self.assertTrue(all(write[1] == ("model", module.REPO_ID) for write in self.hub.writes))

    def disk_contents(self):
        return {str(path.relative_to(self.root)): path.read_bytes() if path.is_file() else None
                for path in self.root.rglob("*")}

    def test_separate_optional_state_path_is_accepted_without_creating_it(self):
        before = self.disk_contents()
        self.assertEqual(module.validate_state_path(self.root, self.recovery, self.state),
                         self.state.resolve())
        self.assertEqual(self.disk_contents(), before)
        self.assertFalse(self.state.exists())

    def test_unsafe_state_paths_are_rejected_before_any_directory_or_file_write(self):
        unsafe = (
            self.primary_state,
            self.primary_state / "optional-child",
            self.root / "publication-metadata/optional-child",
            self.root / "runs/random/model",
            self.root / "runs/advanced/model/optional-child",
            self.recovery / "model/optional-child",
        )
        for state in unsafe:
            before = self.disk_contents()
            with self.subTest(state=state), self.assertRaises(ValueError):
                module.publish(self.root, self.recovery, state, expected_owner=module.OWNER,
                    primary_receipt=self.primary_path, hub=self.hub)
            self.assertEqual(self.disk_contents(), before)
            self.assertEqual(self.hub.writes, [])
        self.assert_primary_untouched()

    def test_optional_destination_and_state_are_isolated_from_all_four_primary_repositories(self):
        self.assertEqual(module.OWNER, "adversarialbandit")
        self.assertEqual(module.REPO_ID, "adversarialbandit/qwen25-3b-base75-advanced25-posthoc-20261020")
        self.assertNotIn(module.REPO_ID, {primary.OWNER + "/" + value[2]
                                        for value in primary.DESTINATIONS.values()})
        self.assertTrue(self.publish()["complete"])
        self.assertEqual(set(self.hub.repos), {("model", module.REPO_ID)})
        self.assert_primary_untouched()

    def test_identical_retry_reverifies_without_any_additional_remote_writes(self):
        first = self.publish()
        writes = copy.deepcopy(self.hub.writes)
        second = self.publish()
        self.assertEqual(first["publication_sha256"], second["publication_sha256"])
        self.assertEqual(self.hub.writes, writes)
        self.assert_primary_untouched()

    def test_lost_accepted_large_commit_resumes_without_reuploading_that_shard(self):
        self.hub.fail_on_commit = 3  # identity, small files, first large shard
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())
        self.assertTrue(self.publish()["complete"])
        shard = "model-00001-of-00002.safetensors"
        self.assertEqual(sum(write[0] == "commit" and shard in write[2]
                             for write in self.hub.writes), 1)
        self.assertEqual(sum(write[0] == "create" for write in self.hub.writes), 1)
        self.assert_primary_untouched()

    def test_conflicting_optional_repository_prevents_all_remote_writes(self):
        self.hub.repos[("model", module.REPO_ID)] = dict(commit="c" * 40,
                                                       files={"foreign.txt": {"size": 0}})
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.hub.writes, [])
        self.assert_primary_untouched()

    def test_unknown_existing_empty_repository_is_not_adopted(self):
        self.hub.repos[("model", module.REPO_ID)] = dict(commit="c" * 40, files={})
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.hub.writes, [])

    def test_ambiguous_lost_creation_response_requires_reconciliation(self):
        self.hub.fail_create_once = True
        with self.assertRaises(TimeoutError):
            self.publish()
        writes = copy.deepcopy(self.hub.writes)
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.hub.writes, writes)

    def test_private_or_authenticated_only_visibility_cannot_claim_completion(self):
        self.hub.hide_public = True
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())
        self.assert_primary_untouched()

    def test_public_lfs_hash_mismatch_cannot_claim_completion(self):
        self.hub.corrupt_public = True
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse((self.state / "publication_complete.json").exists())

    def test_wrong_expected_owner_or_authenticated_account_never_writes(self):
        with self.assertRaises(ValueError):
            self.publish(expected_owner="another-account")
        self.assertEqual(self.hub.writes, [])
        self.hub.name = "another-account"
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.hub.writes, [])
        self.assert_primary_untouched()

    def test_login_or_gpu_allocation_never_writes(self):
        for env in ({"SLURM_JOB_ID": ""}, {"SLURM_JOB_PARTITION": "login"},
                    {"SLURM_JOB_GPUS": "0,1"}):
            with self.subTest(env=env), patch.dict(os.environ, env), self.assertRaises(ValueError):
                self.publish()
        self.assertEqual(self.hub.writes, [])


if __name__ == "__main__":
    unittest.main()
