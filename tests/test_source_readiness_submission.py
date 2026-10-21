"""Authored metadata admission oracles; execute only in native CPU QA."""
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

from extension import source_readiness_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12360;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value(command, kwargs) if callable(value) else value


class SourceReadinessAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        project = Path(module.__file__).resolve().parents[1]
        old_evidence = json.loads((project / module.DIRECTORY / "unit_qa.json").read_text())["initial_evidence"]
        parent_files = set(old_evidence) | {module.DIRECTORY + "/" + name for name in module.RECEIPTS}
        for name in set(module.SOURCE_PATHS) | parent_files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((project / name).read_bytes())
        self.parent_ledger = self.read(module.DIRECTORY + "/job_ledger_snapshot.json")
        self.write("extension/job_ledger.json", self.parent_ledger)
        self.now = module.DEADLINE - timedelta(days=5)
        self.script = self.root / module.SCRIPT
        directives = ["--job-name=" + module.NAME, "--account=tml_2026", "--partition=normal_q", "--qos=owl_normal_short",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=02:00:00", "--no-requeue",
            "--output=" + str(self.root / "extension/logs/cpu-%j.out"),
            "--error=" + str(self.root / "extension/logs/cpu-%j.err")]
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + item for item in directives) + "\nset -euo pipefail\n")
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(module._canonical(value) + b"\n")

    def submit(self, scheduler=None, now=None):
        return module.submit(self.root, runner=Scheduler() if scheduler is None else scheduler,
            now=self.now if now is None else now)

    def test_exact_zero_gpu_admission_binds67_sources25_receipts_and_preserves_eight_jobs(self):
        scheduler = Scheduler()
        self.assertEqual(self.submit(scheduler), "12360")
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.script)])
        ledger = self.read("extension/job_ledger.json")
        entry = ledger["jobs"][module.ROLE]
        self.assertEqual(len(entry["source_sha256"]), 67)
        self.assertEqual(len(entry["initial_evidence"]), 25)
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        self.assertEqual(entry["requested_gpu_hours"], 0)
        self.assertEqual({role: ledger["jobs"][role] for role in self.parent_ledger["jobs"]}, self.parent_ledger["jobs"])

    def test_submitting_state_is_durable_before_contacting_scheduler(self):
        def callback(command, kwargs):
            self.assertEqual(self.read("extension/job_ledger.json")["jobs"][module.ROLE]["status"], "submitting")
            return result("12360;owl")
        self.assertEqual(self.submit(Scheduler([result(), callback])), "12360")

    def test_known_b200_pending_running_or_terminal_does_not_block_zero_gpu_work(self):
        for state in ("PENDING", "RUNNING", "COMPLETING", "COMPLETED", "FAILED"):
            self.write("extension/job_ledger.json", self.parent_ledger)
            queue = "1224577|" + state + "|sft-extension-native-model-admission\n"
            self.assertEqual(self.submit(Scheduler([result(queue), result("12360;owl")])), "12360")

    def test_duplicate_or_changed_prior_native_job_is_rejected(self):
        for queue in ("99999|PENDING|sft-extension-native-model-admission\n",
            "1224577_1|RUNNING|sft-extension-native-model-admission\n",
            "1224577|RUNNING|different-job\n", "1224497|RUNNING|sft-extension-smoke-preparation\n",
            "99999|PENDING|sft-extension-source-readiness\n"):
            with self.subTest(queue=queue), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(queue)]))

    def test_idempotent_existing_role_returns_exact_id_without_second_submission(self):
        self.submit()
        scheduler = Scheduler([result("12360|PENDING|" + module.NAME)])
        self.assertEqual(self.submit(scheduler), "12360")
        self.assertEqual(len(scheduler.calls), 1)

    def test_uncertain_failed_or_changed_role_never_resubmits(self):
        self.submit()
        saved = self.read("extension/job_ledger.json")
        for status in ("submitting", "uncertain", "submission_failed", "complete"):
            ledger = deepcopy(saved)
            ledger["jobs"][module.ROLE]["status"] = status
            self.write("extension/job_ledger.json", ledger)
            with self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(Scheduler([result()]))

    def test_all_prior_entries_unknown_roles_and_parent_receipts_are_immutable(self):
        for role in self.parent_ledger["jobs"]:
            ledger = deepcopy(self.parent_ledger)
            ledger["jobs"][role]["status"] = "changed"
            self.write("extension/job_ledger.json", ledger)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
        self.write("extension/job_ledger.json", self.parent_ledger)
        ledger = deepcopy(self.parent_ledger)
        ledger["jobs"]["unknown-role"] = {"status": "submitted"}
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaises(module.SubmissionError):
            self.submit(Scheduler([]))

    def test_changed_prior_code_grader_or_native_receipt_blocks_before_scheduler(self):
        for name in ("extension/preserving_trainer.py", "eval/utils/grader.py",
            module.DIRECTORY + "/complete.json", module.DIRECTORY + "/native_qa.json",
            module.DIRECTORY + "/accounting.json", module.DIRECTORY + "/benchmark_audit.json"):
            path = self.root / name
            raw = path.read_bytes()
            path.write_bytes(raw + b"\n")
            with self.subTest(name=name), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
            path.write_bytes(raw)

    def test_metadata_gate_never_reads_raw_cohorts_pool_or_model(self):
        original = module.metadata._small_bytes
        def small(path, root):
            relative = str(Path(path).relative_to(root))
            self.assertNotIn("/runs/", relative)
            self.assertNotIn("artifacts/", relative)
            self.assertNotIn(".safetensors", relative)
            return original(path, root)
        with patch.object(module.metadata, "_small_bytes", side_effect=small):
            self.assertEqual(self.submit(), "12360")

    def test_deadline_reserves_full_two_hours_and48_hours(self):
        with self.assertRaises(module.SubmissionError):
            self.submit(Scheduler([]), now=module.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(self.submit(now=module.DEADLINE - timedelta(seconds=180000)), "12360")

    def test_timeout_rejection_and_ambiguous_scheduler_results_remain_recorded(self):
        for response, status in ((subprocess.TimeoutExpired("sbatch", 60), "uncertain"),
            (result("not-a-job"), "uncertain"), (result("", "controller rejected", 1), "submission_failed")):
            self.write("extension/job_ledger.json", self.parent_ledger)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(), response]))
            self.assertEqual(self.read("extension/job_ledger.json")["jobs"][module.ROLE]["status"], status)

    def test_submitted_resource_override_environment_is_removed_preserving_configuration(self):
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:9", "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_CONF": "/fixture/slurm.conf", "SLURM_CONF_SERVER": "owl-controller"}):
            scheduler = Scheduler()
            self.submit(scheduler)
        environment = scheduler.calls[1][1]["env"]
        self.assertNotIn("SBATCH_GRES", environment)
        self.assertNotIn("SLURM_JOB_ID", environment)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["SLURM_CONF"], "/fixture/slurm.conf")
        self.assertEqual(environment["SLURM_CONF_SERVER"], "owl-controller")

    def test_mutated_cpu_scope_unknown_scheduler_options_and_late_directives_are_forbidden(self):
        plan = self.read(module.PLAN)
        for name in ("training", "model_inference", "automatic_downstream"):
            changed = deepcopy(plan)
            changed[name] = True
            self.write(module.PLAN, changed)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
        self.write(module.PLAN, plan)
        raw = self.script.read_text()
        for changed in (raw.replace("--mem=64G", "--mem=128G"), raw + "#SBATCH --gres=gpu:1\n",
            raw + "#SBATCH --array=1-2\n", raw + "#SBATCH --time=01:00:00\n"):
            self.script.write_text(changed)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
        self.script.write_text(raw)

    def test_ledger_and_lock_symlinks_are_refused(self):
        for name in ("extension/job_ledger.json", "extension/job_ledger.lock"):
            path = self.root / name
            if path.exists():
                path.unlink()
            other = self.root / "fixture-symlink-target"
            other.write_text("{}")
            path.symlink_to(other)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
            path.unlink()
            self.write("extension/job_ledger.json", self.parent_ledger)


if __name__ == "__main__":
    unittest.main()
