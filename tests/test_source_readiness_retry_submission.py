"""Offline, metadata-only retry oracles; execute later in native CPU QA."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import source_readiness_retry_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12361;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value(command, kwargs) if callable(value) else value


class SourceReadinessRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        project = Path(module.__file__).resolve().parents[1]
        # Public metadata and source only; no cache, tokenizer, model or data.
        old_evidence = json.loads((project / module.previous.DIRECTORY / "unit_qa.json").read_text())["initial_evidence"]
        parents = set(old_evidence) | {module.previous.DIRECTORY + "/" + name for name in module.previous.RECEIPTS}
        parents |= {module.DIRECTORY + "/" + name for name in module.RECEIPTS}
        for name in set(module.SOURCE_PATHS) | parents:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project / name).read_bytes())
        self.prior_ledger = self.read(module.DIRECTORY + "/job_ledger_snapshot.json")
        self.prior_jobs = deepcopy(self.prior_ledger["jobs"])
        self.write("extension/job_ledger.json", self.prior_ledger)
        self.now = module.DEADLINE - timedelta(days=5)
        self.script = self.root / module.SCRIPT
        options = ["--job-name=" + module.NAME, "--account=tml_2026", "--partition=normal_q", "--qos=owl_normal_short",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=02:00:00", "--no-requeue",
            "--output=" + str(self.root / "extension/logs/retry-%j.out"),
            "--error=" + str(self.root / "extension/logs/retry-%j.err")]
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + item for item in options) + "\nset -euo pipefail\n")
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(module._canonical(value) + b"\n")

    def ledger(self):
        return self.read("extension/job_ledger.json")

    def submit(self, scheduler=None, *, now=None):
        return module.submit(self.root, runner=Scheduler() if scheduler is None else scheduler,
            now=self.now if now is None else now)

    def test_one_fixed_zero_gpu_retry_binds71_sources30_assets_and_nine_unchanged_jobs(self):
        scheduler = Scheduler()
        self.assertEqual(self.submit(scheduler), "12361")
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.script)])
        entry = self.ledger()["jobs"][module.ROLE]
        self.assertEqual(len(entry["source_sha256"]), 71)
        self.assertEqual(len(entry["initial_evidence"]), 30)
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        self.assertEqual(entry["requested_gpu_hours"], 0)
        self.assertEqual(entry["plan_sha256"], module.PLAN_SHA256)
        self.assertEqual(entry["binding"]["fixture_only_repair"], {"path": module.FIXTURE,
            "before_sha256": module.OLD_FIXTURE_SHA256, "after_sha256": module.NEW_FIXTURE_SHA256})
        for role, prior in self.prior_jobs.items():
            self.assertEqual(self.ledger()["jobs"][role], prior)

    def test_exact_fixture_repair_is_the_only_allowed_delta_from67_executed_sources(self):
        inventory, jobs = module.initial_evidence(self.root)
        self.assertEqual(len(inventory), 30)
        self.assertEqual(jobs, self.prior_jobs)
        installed = self.read(module.DIRECTORY + "/source_install_receipt.json")["source_inventory"]
        current = module.previous.source_inventory(self.root)
        self.assertEqual([name for name in installed if current[name] != installed[name]], [module.FIXTURE])
        self.assertEqual(current[module.FIXTURE]["sha256"], module.NEW_FIXTURE_SHA256)
        self.assertEqual(installed[module.FIXTURE]["sha256"], module.OLD_FIXTURE_SHA256)

    def test_other66_sources_including_scorer_driver_and_v1_adapter_are_immutable(self):
        for name in ("extension/source_evaluation.py", "extension/pilot_driver.py", "tests/test_pilot_driver.py",
            "extension/source_readiness.py", "extension/source_readiness_submission.py", "eval/utils/grader.py"):
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(name=name), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
            path.write_bytes(before)

    def test_old_failing_fixture_or_any_unapproved_third_version_cannot_admit_retry(self):
        path = self.root / module.FIXTURE
        corrected = path.read_bytes()
        for changed in ((self.root / module.DIRECTORY / "executed_test_source_evaluation.py").read_bytes(),
            corrected + b"\n"):
            path.write_bytes(changed)
            with self.assertRaisesRegex(module.SubmissionError, "explicitly pinned"):
                self.submit(Scheduler([]))
        path.write_bytes(corrected)

    def test_all_five_failure_assets_are_pinned_including_executed_fixture(self):
        for name in module.RECEIPTS:
            path = self.root / module.DIRECTORY / name
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(module.SubmissionError, "evidence changed"):
                self.submit(Scheduler([]))
            path.write_bytes(before)

    def test_original_plan_is_unchanged_even_a_cosmetic_byte_edit_is_rejected(self):
        path = self.root / module.PLAN
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(module.SubmissionError, "byte-identical"):
            self.submit(Scheduler([]))

    def test_all_nine_entries_including_failed_stage_and_noid_dns_attempt_are_preserved(self):
        for role in self.prior_jobs:
            ledger = deepcopy(self.prior_ledger)
            ledger["jobs"][role]["status"] = "changed"
            self.write("extension/job_ledger.json", ledger)
            with self.subTest(role=role), self.assertRaisesRegex(module.SubmissionError, "nine entries"):
                self.submit(Scheduler([]))

    def test_unknown_ledger_roles_cannot_be_silently_removed_or_adopted(self):
        ledger = deepcopy(self.prior_ledger)
        ledger["jobs"]["another_untracked_stage"] = {"status": "submitted", "job_id": "999"}
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaisesRegex(module.SubmissionError, "unknown or changed"):
            self.submit(Scheduler([]))

    def test_proven_failed_v1_and_existing_b200_can_remain_visible_without_new_gpu_admission(self):
        for state in ("PENDING", "RUNNING", "COMPLETED", "FAILED"):
            self.write("extension/job_ledger.json", self.prior_ledger)
            queue = "1225252|FAILED|sft-extension-source-readiness\n1224577|" + state + "|sft-extension-native-model-admission\n"
            self.assertEqual(self.submit(Scheduler([result(queue), result("12361;owl")])), "12361")

    def test_prior_active_unknown_renamed_array_or_duplicate_failure_job_is_rejected(self):
        for queue in ("1225252|RUNNING|sft-extension-source-readiness\n",
            "1225252|COMPLETING|sft-extension-source-readiness\n", "1225252|FAILED|different-name\n",
            "1225252_1|FAILED|sft-extension-source-readiness\n", "999|FAILED|sft-extension-source-readiness\n"):
            with self.subTest(queue=queue), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(queue)]))

    def test_submitting_retry_state_is_durable_before_scheduler_write(self):
        def submitted(command, kwargs):
            self.assertEqual(self.ledger()["jobs"][module.ROLE]["status"], "submitting")
            self.assertEqual({role: self.ledger()["jobs"][role] for role in self.prior_jobs}, self.prior_jobs)
            return result("12361;owl")
        self.submit(Scheduler([result(), submitted]))

    def test_exact_existing_submission_is_idempotent_without_second_sbatch(self):
        self.submit()
        self.write("extension/runs/mutable-runtime.json", {"mutable": True})
        scheduler = Scheduler([result("12361|PENDING|" + module.NAME)])
        self.assertEqual(self.submit(scheduler), "12361")
        self.assertEqual(len(scheduler.calls), 1)

    def test_changed_uncertain_or_failed_existing_retry_never_resubmits(self):
        for status in ("submitting", "uncertain", "submission_failed", "complete", None):
            ledger = deepcopy(self.prior_ledger)
            ledger["jobs"][module.ROLE] = {"status": status}
            self.write("extension/job_ledger.json", ledger)
            with self.subTest(status=status), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(Scheduler([result()]))

    def test_scheduler_timeout_rejection_and_ambiguous_ids_are_preserved_without_retry(self):
        for response, status in ((subprocess.TimeoutExpired("sbatch", 60), "uncertain"),
            (result("not-an-id"), "uncertain"), (result("", "controller rejected", 1), "submission_failed"),
            (result("12361;tc"), "uncertain")):
            self.write("extension/job_ledger.json", self.prior_ledger)
            scheduler = Scheduler([result(), response])
            with self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(self.ledger()["jobs"][module.ROLE]["status"], status)
            self.assertEqual(len(scheduler.calls), 2)

    def test_scheduler_queue_failure_prevents_any_ledger_admission(self):
        scheduler = Scheduler([result("", "controller unavailable", 1)])
        with self.assertRaises(module.SubmissionError):
            self.submit(scheduler)
        self.assertNotIn(module.ROLE, self.ledger()["jobs"])
        self.assertEqual(len(scheduler.calls), 1)

    def test_new_source_change_during_queue_snapshot_blocks_scheduler_submission(self):
        def changed(command, kwargs):
            path = self.root / "extension/source_readiness_retry_qa.py"
            path.write_bytes(path.read_bytes() + b"\n")
            return result()
        scheduler = Scheduler([changed])
        with self.assertRaisesRegex(module.SubmissionError, "changed during"):
            self.submit(scheduler)
        self.assertEqual(len(scheduler.calls), 1)
        self.assertNotIn(module.ROLE, self.ledger()["jobs"])

    def test_scheduler_overrides_are_removed_but_slurm_configuration_is_retained(self):
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:2", "SLURM_JOB_ID": "999", "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_CONF": "/fixture/slurm.conf", "SLURM_CONF_SERVER": "owl-controller"}):
            scheduler = Scheduler()
            self.submit(scheduler)
        environment = scheduler.calls[1][1]["env"]
        for key in ("SBATCH_GRES", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["SLURM_CONF"], "/fixture/slurm.conf")
        self.assertEqual(environment["SLURM_CONF_SERVER"], "owl-controller")

    def test_fixed_cpu_resources_no_gpu_array_or_late_scheduler_directive(self):
        before = self.script.read_text()
        for changed in (before.replace("--mem=64G", "--mem=128G"), before.replace("02:00:00", "01:00:00"),
            before + "#SBATCH --gres=gpu:1\n", before + "#SBATCH --array=0-1\n"):
            self.script.write_text(changed)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
        self.script.write_text(before)

    def test_deadline_keeps_full_two_hours_plus48_hour_reserve(self):
        with self.assertRaises(module.SubmissionError):
            self.submit(Scheduler([]), now=module.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(self.submit(now=module.DEADLINE - timedelta(seconds=180000)), "12361")

    def test_admission_never_opens_private_cache_source_rows_tokenizer_or_weights(self):
        original = module.metadata._small_bytes
        def bounded(path, root):
            relative = str(Path(path).relative_to(root))
            self.assertNotIn("/runs/", relative)
            self.assertNotIn("artifacts/", relative)
            self.assertNotIn(".safetensors", relative)
            return original(path, root)
        with patch.object(module.metadata, "_small_bytes", side_effect=bounded):
            self.assertEqual(self.submit(), "12361")


if __name__ == "__main__":
    unittest.main()
