"""Offline metadata-only CPU-retry oracles; run later in native CPU QA."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import development_retry_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12348;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(command, kwargs) if callable(value) else value


class RetrySubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extension = self.root / "extension"
        project = Path(module.__file__).resolve().parents[1]
        # Only small public native metadata and immutable code are fixtures.
        # There is no raw data, token cache, tokenizer or model access here.
        for name in module.SOURCE_PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            old = name in module.previous.SOURCE_PATHS or name.startswith("extension/operations/")
            path.write_bytes((project / name).read_bytes() if old else b"new retry source placeholder\n")
        earlier = module.previous.labeled
        for directory, pins in ((earlier.EVIDENCE_DIRECTORY, earlier.INITIAL_RECEIPT_SHA256),
            (module.previous.EVIDENCE_DIRECTORY, module.previous.LABELED_RECEIPT_SHA256),
            (module.FAILED_DIRECTORY, module.FAILED_RECEIPT_SHA256),
            (module.DIAGNOSTIC_DIRECTORY, module.DIAGNOSTIC_RECEIPT_SHA256)):
            for name in pins:
                path = self.root / directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((project / directory / name).read_bytes())
        self.prior_ledger = json.loads((self.root / module.DIAGNOSTIC_DIRECTORY / "job_ledger_snapshot.json").read_text())
        self.prior_jobs = copy.deepcopy(self.prior_ledger["jobs"])
        self.write("extension/job_ledger.json", self.prior_ledger)
        self.plan = dict(schema="extension-development-seal-plan-v2", cpu_only=True,
            model_inference=False, training=False, automatic_downstream=False, choose_splits=True,
            deadline_utc=module.DEADLINE.isoformat(), wall_time_seconds=7200, reserve_seconds=172800,
            seed=20261021, targets=copy.deepcopy(module.previous.TARGETS), near_duplicate=dict(module.CRITERION),
            labeled_audit=dict(directory=module.previous.LABELED_RUN_DIRECTORY,
                complete_sha256=module.previous.LABELED_RECEIPT_SHA256["complete.json"],
                qa_sha256=module.previous.LABELED_RECEIPT_SHA256["native_qa.json"]),
            pool=dict(path="deadline/artifacts/pool.json", sha256=module.previous.POOL_SHA256),
            level_diagnostic=dict(job_id=module.DIAGNOSTIC_JOB_ID, directory=module.DIAGNOSTIC_RUN_DIRECTORY,
                complete_sha256=module.DIAGNOSTIC_RECEIPT_SHA256["complete.json"]),
            math_unknown_levels=dict(module.OBSERVED_UNKNOWN_LEVELS))
        self.write("extension/development_retry_plan.json", self.plan)
        self.script = self.extension / "development_retry.slurm"
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + value for value in (
            "--job-name=sft-extension-development-seal-v2", "--account=tml_2026", "--partition=normal_q",
            "--qos=owl_normal_short", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=02:00:00", "--no-requeue", f"--output={self.extension}/logs/retry-%j.out",
            f"--error={self.extension}/logs/retry-%j.err")) + "\nset -euo pipefail\n")
        self.now = module.DEADLINE - timedelta(days=5)
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def ledger(self):
        return json.loads((self.extension / "job_ledger.json").read_text())

    def submit(self, scheduler=None, **kwargs):
        return module.submit_development_retry(self.root, now=kwargs.pop("now", self.now),
            runner=Scheduler() if scheduler is None else scheduler, **kwargs)

    def test_fixed_retry_preserves_all_five_entries_and_binds38_sources(self):
        scheduler = Scheduler()
        self.assertEqual(self.submit(scheduler), "12348")
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.script)])
        ledger = self.ledger()
        for role, entry in self.prior_jobs.items():
            self.assertEqual(ledger["jobs"][role], entry)
        entry = ledger["jobs"][module.ROLE]
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        self.assertEqual(entry["requested_gpu_hours"], 0)
        self.assertEqual(len(entry["source_sha256"]), 38)
        self.assertEqual(entry["source_sha256"], module.source_inventory(self.root))
        self.assertEqual(len(entry["binding"]["initial_evidence"]), 15)
        self.assertEqual(entry["binding"]["observed_unknown_levels"], {"Level ?": 1})
        self.assertEqual(entry["binding"]["wall_time_seconds"], 7200)
        self.assertEqual(entry["binding"]["closeout_reserve_seconds"], 172800)

    def test_proven_terminal_failed_and_completed_rows_can_remain_in_queue(self):
        queue = ("1223566|FAILED|sft-extension-development-seal\n"
                 "1223713|COMPLETED|sft-extension-level-diagnostic\n")
        self.assertEqual(self.submit(Scheduler([result(queue), result("12348;owl")])), "12348")
        # The no-ID rejected diagnostic is preserved, never assigned a fake ID.
        self.assertNotIn("job_id", self.ledger()["jobs"][module.REJECTED_ROLE])

    def test_active_unknown_mismatched_or_duplicate_prior_rows_fail_closed(self):
        for row in ("1223566|RUNNING|sft-extension-development-seal\n",
                    "1223713|REQUEUED|sft-extension-level-diagnostic\n",
                    "1223713|COMPLETED|different-name\n",
                    "999|COMPLETED|sft-extension-level-diagnostic\n",
                    "1223566_0|FAILED|sft-extension-development-seal\n"):
            scheduler = Scheduler([result(row)])
            with self.subTest(row=row), self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_submitting_receipt_durable_before_sbatch_and_history_unchanged(self):
        def submitted(command, kwargs):
            ledger = self.ledger()
            self.assertEqual(ledger["jobs"][module.ROLE]["status"], "submitting")
            for role, entry in self.prior_jobs.items():
                self.assertEqual(ledger["jobs"][role], entry)
            return result("12348;owl")
        self.submit(Scheduler([result(), submitted]))

    def test_exact_successful_retry_is_idempotent_and_runtime_not_source(self):
        self.submit()
        self.write("extension/runs/unrelated.json", {"mutable": True})
        self.write("extension/runtime.json", {"mutable": True})
        scheduler = Scheduler([result("12348|PENDING|sft-extension-development-seal-v2\n")])
        self.assertEqual(self.submit(scheduler), "12348")
        self.assertEqual(len(scheduler.calls), 1)

    def test_newrole_ambiguous_or_failed_records_never_resubmit(self):
        for status in ("uncertain", "submitting", "submission_failed", "unknown", "complete", None):
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][module.ROLE] = dict(status=status)
            self.write("extension/job_ledger.json", ledger)
            scheduler = Scheduler([result()])
            with self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_the_specific_dns_exception_does_not_allow_any_other_uncertain_entry(self):
        ledger = copy.deepcopy(self.prior_ledger)
        ledger["jobs"][module.REJECTED_ROLE]["scheduler_stderr"] = "generic controller timeout"
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaisesRegex(module.SubmissionError, "changed five-entry history"):
            self.submit(Scheduler([]))
        self.assertNotIn(module.ROLE, self.ledger()["jobs"])

    def test_all_parent_entries_including_noid_attempt_are_exactly_preserved(self):
        for role in self.prior_jobs:
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][role]["requested_gpu_hours"] = 1
            self.write("extension/job_ledger.json", ledger)
            with self.subTest(role=role), self.assertRaisesRegex(module.SubmissionError, "changed five-entry history"):
                self.submit(Scheduler([]))

    def test_immutable28sources_and_both_diagnostic_versions_are_verified(self):
        for name in ("extension/development_seal.py", "extension/preserving_objectives.py",
                     "extension/operations/development_metadata_diagnostic.slurm",
                     "extension/operations/submit_development_diagnostic_v2.py"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))
            path.write_bytes(original)

    def test_new_retry_source_or_plan_changes_prevent_idempotent_resubmission(self):
        self.submit()
        for name in ("extension/development_retry_plan.json", "extension/development_seal_v2.py",
                     "tests/test_preservation_retry_submission.py"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(Scheduler([result()]))
            path.write_bytes(original)

    def test_each_native_failure_diagnostic_and_dns_receipt_is_pinned(self):
        for directory, filename in ((module.FAILED_DIRECTORY, "native_qa.json"),
            (module.FAILED_DIRECTORY, "development-seal-1223566-failure.json"),
            (module.DIAGNOSTIC_DIRECTORY, "complete.json"),
            (module.DIAGNOSTIC_DIRECTORY, "development-diagnostic-dns-rejection-20261021.json")):
            path = self.root / directory / filename
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(module.SubmissionError, "digest mismatch"):
                self.submit(Scheduler([]))
            path.write_bytes(original)

    def test_unready_diagnostic_pins_block_before_any_scheduler_request(self):
        with patch.dict(module.DIAGNOSTIC_RECEIPT_SHA256, {}, clear=True):
            with self.assertRaisesRegex(module.SubmissionError, "unready"):
                self.submit(Scheduler([]))

    def test_exact_target_seed_unknown_level_map_and_criterion_are_required(self):
        for change in (lambda value: value.update(seed=1),
                       lambda value: value["targets"]["calibration"].update(math=15),
                       lambda value: value["near_duplicate"].update(jaccard_numerator=3),
                       lambda value: value.update(math_unknown_levels={"Level ?": 2}),
                       lambda value: value["level_diagnostic"].update(job_id="999")):
            value = copy.deepcopy(self.plan)
            change(value)
            self.write("extension/development_retry_plan.json", value)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_scope_and_48hour_reserve_guard(self):
        with self.assertRaisesRegex(module.SubmissionError, "48-hour"):
            self.submit(Scheduler([]), now=module.DEADLINE - timedelta(seconds=179999))
        for field in ("model_inference", "training", "automatic_downstream"):
            value = copy.deepcopy(self.plan)
            value[field] = True
            self.write("extension/development_retry_plan.json", value)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_exact_deadline_boundary_accepts_finite_cpu_reservation(self):
        self.assertEqual(self.submit(now=module.DEADLINE - timedelta(seconds=180000)), "12348")

    def test_gpu_unknown_or_unbounded_scheduler_directives_are_forbidden(self):
        original = self.script.read_text()
        for addition in ("--gpus=1", "--gres=gpu:1", "--array=0-9"):
            self.script.write_text(original.replace("set -euo pipefail", "#SBATCH " + addition + "\nset -euo pipefail"))
            with self.assertRaisesRegex(module.SubmissionError, "GPU/unknown"):
                self.submit(Scheduler([]))
        self.script.write_text(original.replace("02:00:00", "UNLIMITED"))
        with self.assertRaisesRegex(module.SubmissionError, "fixed 8-CPU"):
            self.submit(Scheduler([]))

    def test_scheduler_environment_preserves_configuration_but_removes_request_overrides(self):
        scheduler = Scheduler()
        with patch.dict(os.environ, {"SLURM_CONF": "/etc/slurm.conf", "SLURM_CONF_SERVER": "scheduler",
                "SLURM_JOB_ID": "999", "SBATCH_GRES": "gpu:8", "CUDA_VISIBLE_DEVICES": "0"}):
            self.submit(scheduler)
        env = scheduler.calls[1][1]["env"]
        self.assertEqual(env["SLURM_CONF"], "/etc/slurm.conf")
        self.assertEqual(env["SLURM_CONF_SERVER"], "scheduler")
        self.assertNotIn("SLURM_JOB_ID", env)
        self.assertNotIn("SBATCH_GRES", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_timeout_and_nonzero_responses_preserve_uncertainty_without_retry(self):
        for response in (subprocess.TimeoutExpired("sbatch", 60, output=b"12348;owl\n"),
                         result("12348;owl", "controller failure", 1), result("12348;other")):
            self.write("extension/job_ledger.json", self.prior_ledger)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(), response]))
            self.assertIn(self.ledger()["jobs"][module.ROLE]["status"], ("uncertain", "submission_failed"))
            with self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(Scheduler([result()]))

    def test_duplicate_newrole_or_bad_scheduler_snapshot_blocks(self):
        for response in (result("999|RUNNING|sft-extension-development-seal-v2\n"),
                         result("9_[0-1]|PENDING|other\n"), result("", "controller down", 1)):
            scheduler = Scheduler([response])
            with self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_metadata_admission_never_reads_raw_data_or_historical_ledger(self):
        for name in ("extension/runs/labeled-feasibility-20261020-1222926/surviving.jsonl",
                     "deadline/artifacts/pool.json", "deadline/job_ledger.json"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not JSON; metadata admission must not read this")
        self.submit()
        self.assertNotIn("job_id", self.ledger()["jobs"][module.REJECTED_ROLE])

    def test_symlink_or_missing_ledger_cannot_replace_history(self):
        ledger = self.extension / "job_ledger.json"
        ledger.unlink()
        with self.assertRaisesRegex(module.SubmissionError, "five-entry"):
            self.submit(Scheduler([]))
        history = self.root / "deadline/job_ledger.json"
        history.parent.mkdir()
        history.write_text("immutable")
        ledger.symlink_to(history)
        with self.assertRaisesRegex(module.SubmissionError, "symlinks"):
            self.submit(Scheduler([]))
        self.assertEqual(history.read_text(), "immutable")


if __name__ == "__main__":
    unittest.main()
