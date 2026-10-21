"""Metadata/scheduler oracles; execute later in native CPU QA, never submit."""
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import development_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12347;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(command, kwargs) if callable(value) else value


class DevelopmentSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extension = self.root / "extension"
        project = Path(module.__file__).resolve().parents[1]
        # The already-delivered small public native receipts and 18 historical
        # sources form exact metadata fixtures. No raw data/model paths are read.
        for name in module.SOURCE_PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project / name).read_bytes() if name in module.labeled.SOURCE_PATHS
                             else b"new source placeholder for binding oracle\n")
        for directory, names in ((module.labeled.EVIDENCE_DIRECTORY, module.labeled.INITIAL_RECEIPT_SHA256),
                                  (module.EVIDENCE_DIRECTORY, module.LABELED_RECEIPT_SHA256)):
            for name in names:
                target = self.root / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((project / directory / name).read_bytes())
        archive = self.root / module.EVIDENCE_DIRECTORY / "job_ledger_snapshot.json"
        self.prior_ledger = json.loads(archive.read_text())
        self.prior_jobs = copy.deepcopy(self.prior_ledger["jobs"])
        self.write("extension/job_ledger.json", self.prior_ledger)
        self.plan = dict(schema="extension-development-seal-plan-v1", cpu_only=True,
            model_inference=False, training=False, automatic_downstream=False, choose_splits=True,
            deadline_utc=module.DEADLINE.isoformat(), wall_time_seconds=7200, reserve_seconds=172800,
            seed=20261021, targets=copy.deepcopy(module.TARGETS),
            labeled_audit=dict(directory=module.LABELED_RUN_DIRECTORY,
                complete_sha256=module.LABELED_RECEIPT_SHA256["complete.json"],
                qa_sha256=module.LABELED_RECEIPT_SHA256["native_qa.json"]),
            pool=dict(path="deadline/artifacts/pool.json", sha256=module.POOL_SHA256))
        self.write("extension/development_plan.json", self.plan)
        self.script = self.extension / "development_seal.slurm"
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + value for value in (
            "--job-name=sft-extension-development-seal", "--account=tml_2026", "--partition=normal_q",
            "--qos=owl_normal_short", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=02:00:00", "--no-requeue", f"--output={self.extension}/logs/development-%j.out",
            f"--error={self.extension}/logs/development-%j.err")) + "\nset -euo pipefail\n")
        self.now = module.DEADLINE - timedelta(days=5)
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def ledger(self):
        return json.loads((self.extension / "job_ledger.json").read_text())

    def submit(self, scheduler=None, **kwargs):
        return module.submit_development_seal(self.root, now=kwargs.pop("now", self.now),
            runner=Scheduler() if scheduler is None else scheduler, **kwargs)

    def test_fixed_cpu_admission_binds_28_sources_eight_receipts_and_data_hashes(self):
        scheduler = Scheduler([result("CLUSTER: owl\n900_1|RUNNING|unrelated\n"), result("12347;owl\n")])
        self.assertEqual(self.submit(scheduler), "12347")
        self.assertEqual(scheduler.calls[0][0], ["squeue", "--clusters=owl", "--array", "--states=all",
            "--user=fixture_user", "--noheader", "--format=%64i|%64T|%128j"])
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.script)])
        ledger = self.ledger()
        for role, entry in self.prior_jobs.items():
            self.assertEqual(ledger["jobs"][role], entry)
        entry = ledger["jobs"][module.ROLE]
        self.assertEqual(entry["requested_gpu_hours"], 0)
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        self.assertEqual(entry["binding"]["wall_time_seconds"], 7200)
        self.assertEqual(entry["binding"]["closeout_reserve_seconds"], 172800)
        self.assertEqual(len(entry["source_sha256"]), 28)
        self.assertEqual(entry["source_sha256"], module.source_inventory(self.root))
        self.assertEqual(len(entry["binding"]["initial_evidence"]), 8)
        self.assertEqual(entry["binding"]["labeled_output_sha256"], module.LABELED_OUTPUT_SHA256)
        self.assertEqual(entry["plan_sha256"], hashlib.sha256((self.extension / "development_plan.json").read_bytes()).hexdigest())

    def test_durable_submitting_and_both_prior_entries_before_sbatch(self):
        def submitted(command, kwargs):
            ledger = self.ledger()
            self.assertEqual(ledger["jobs"][module.ROLE]["status"], "submitting")
            for role, entry in self.prior_jobs.items():
                self.assertEqual(ledger["jobs"][role], entry)
            return result("12347;owl")
        self.submit(Scheduler([result(), submitted]))

    def test_exact_submitted_retry_is_idempotent_and_mutable_runtime_not_bound(self):
        self.submit()
        self.write("extension/runtime.json", {"status": "changed"})
        self.write("extension/runs/unrelated/receipt.json", {"status": "changed"})
        scheduler = Scheduler([result("12347|PENDING|sft-extension-development-seal\n")])
        self.assertEqual(self.submit(scheduler), "12347")
        self.assertEqual(len(scheduler.calls), 1)
        for role, entry in self.prior_jobs.items():
            self.assertEqual(self.ledger()["jobs"][role], entry)

    def test_metadata_admission_never_reads_raw_labeled_data_pool_or_history(self):
        raw = self.root / module.LABELED_RUN_DIRECTORY / "surviving.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text("invalid JSON; admission must not open or hash this raw file")
        pool = self.root / "deadline/artifacts/pool.json"
        pool.parent.mkdir(parents=True)
        pool.write_text("invalid JSON; compute seal owns this check")
        historical = self.root / "deadline/job_ledger.json"
        historical.write_text("immutable history; never open this")
        self.submit()
        self.assertEqual(raw.read_text(), "invalid JSON; admission must not open or hash this raw file")
        self.assertEqual(pool.read_text(), "invalid JSON; compute seal owns this check")
        self.assertEqual(historical.read_text(), "immutable history; never open this")

    def test_changed_new_source_plan_or_pending_preserving_source_blocks_retry(self):
        self.submit()
        for name in ("extension/development_plan.json", "extension/development_seal.py",
                     "extension/preserving_trainer.py", "tests/test_preserving_objectives.py"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            scheduler = Scheduler([result()])
            with self.subTest(name=name), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)
            path.write_bytes(original)

    def test_any_changed_historical_source_blocks_even_first_admission(self):
        path = self.extension / "objectives.py"
        path.write_bytes(path.read_bytes() + b"\n")
        scheduler = Scheduler([])
        with self.assertRaisesRegex(module.SubmissionError, "18 tested sources"):
            self.submit(scheduler)
        self.assertEqual(scheduler.calls, [])

    def test_unknown_or_modified_prior_ledger_entries_block(self):
        for role in self.prior_jobs:
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][role]["status"] = "complete"
            self.write("extension/job_ledger.json", ledger)
            scheduler = Scheduler([])
            with self.subTest(role=role), self.assertRaisesRegex(module.SubmissionError, "changed prior"):
                self.submit(scheduler)
            self.assertEqual(scheduler.calls, [])
        ledger = copy.deepcopy(self.prior_ledger)
        ledger["jobs"]["unadmitted_gpu"] = {"job_id": "999"}
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaisesRegex(module.SubmissionError, "Unknown role"):
            self.submit(Scheduler([]))

    def test_missing_prior_ledger_is_not_recreated(self):
        path = self.extension / "job_ledger.json"
        path.unlink()
        with self.assertRaisesRegex(module.SubmissionError, "two-stage"):
            self.submit(Scheduler([]))
        self.assertFalse(path.exists())

    def test_failed_unknown_or_ambiguous_state_never_retries(self):
        for status in ("submitting", "uncertain", "submission_failed", "unknown", "complete", None):
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][module.ROLE] = dict(status=status)
            self.write("extension/job_ledger.json", ledger)
            scheduler = Scheduler([result()])
            with self.subTest(status=status), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_timeout_preserves_uncertain_evidence_and_no_duplicate(self):
        with self.assertRaisesRegex(module.SubmissionError, "uncertain"):
            self.submit(Scheduler([result(), subprocess.TimeoutExpired("sbatch", 60, output=b"12347;owl\n")]))
        entry = self.ledger()["jobs"][module.ROLE]
        self.assertEqual(entry["status"], "uncertain")
        self.assertEqual(entry["scheduler_stdout"], "12347;owl\n")
        retry = Scheduler([result()])
        with self.assertRaises(module.SubmissionError):
            self.submit(retry)
        self.assertEqual(len(retry.calls), 1)

    def test_nonzero_and_malformed_scheduler_responses_preserved(self):
        for response in (result("12347;owl", "connection failed", 1), result("12347;other"),
                         result("Submitted batch job 12347"), result("12347\n12348")):
            self.write("extension/job_ledger.json", self.prior_ledger)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(), response]))
            entry = self.ledger()["jobs"][module.ROLE]
            self.assertIn(entry["status"], ("submission_failed", "uncertain"))
            if response.returncode:
                self.assertEqual(entry["possible_job_id"], "12347")
            for role, value in self.prior_jobs.items():
                self.assertEqual(self.ledger()["jobs"][role], value)

    def test_scheduler_environment_cannot_add_gpus_arrays_or_inherited_allocation(self):
        scheduler = Scheduler()
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:8", "SBATCH_ARRAY_INX": "0-9",
                "SLURM_JOB_ID": "999", "SLURM_JOB_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0"}):
            self.submit(scheduler)
        environment = scheduler.calls[1][1]["env"]
        self.assertFalse(any(key.startswith("SBATCH_") for key in environment))
        for name in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
            self.assertNotIn(name, environment)

    def test_exact_runtime_plus_48_hour_reserve_and_new_deadline(self):
        with self.assertRaisesRegex(module.SubmissionError, "48-hour"):
            self.submit(Scheduler([]), now=module.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(self.submit(now=module.DEADLINE - timedelta(seconds=180000)), "12347")

    def test_fixed_scope_targets_seed_and_input_pins_are_enforced(self):
        changes = [lambda value: value.update(model_inference=True), lambda value: value.update(training=True),
            lambda value: value.update(choose_splits=False), lambda value: value.update(automatic_downstream=True),
            lambda value: value.update(seed=1), lambda value: value.update(deadline_utc="2026-10-21T03:59:00+00:00"),
            lambda value: value["targets"]["confirmation"].update(math=1023),
            lambda value: value["labeled_audit"].update(directory="extension/runs/arbitrary"),
            lambda value: value["labeled_audit"].update(complete_sha256="0" * 64),
            lambda value: value["pool"].update(path="arbitrary.json")]
        for change in changes:
            value = copy.deepcopy(self.plan)
            change(value)
            self.write("extension/development_plan.json", value)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_gpu_unknown_resource_or_inactive_scheduler_header_is_rejected(self):
        original = self.script.read_text()
        for addition in ("--gpus=1", "--gres=gpu:1", "--array=0-9", "-G 1"):
            self.script.write_text(original.replace("set -euo pipefail", "#SBATCH " + addition + "\nset -euo pipefail"))
            with self.assertRaisesRegex(module.SubmissionError, "GPU/unknown"):
                self.submit(Scheduler([]))
        for old, new in (("02:00:00", "UNLIMITED"), ("--cpus-per-task=8", "--cpus-per-task=16"),
                         ("--mem=64G", "--mem=128G"), ("--qos=owl_normal_short", "--qos=other")):
            self.script.write_text(original.replace(old, new))
            with self.assertRaisesRegex(module.SubmissionError, "fixed 8-CPU"):
                self.submit(Scheduler([]))
        self.script.write_text(original.replace("#!/bin/bash", "#!/bin/bash\necho before-header"))
        with self.assertRaisesRegex(module.SubmissionError, "ignored"):
            self.submit(Scheduler([]))

    def test_prior_receipt_digest_change_blocks_without_raw_data_read(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "complete.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(module.SubmissionError, "digest mismatch"):
            self.submit(Scheduler([]))

    def test_terminal_accounting_requires_54_seconds_complete_and_zero_gpu(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "accounting.json"
        original = json.loads(path.read_text())
        for transform in (lambda value: value.update(elapsed_seconds=55),
                          lambda value: value.update(stdout=value["stdout"].replace("COMPLETED", "RUNNING")),
                          lambda value: value.update(stdout=value["stdout"].replace("cpu=8", "gres/gpu=1,cpu=8")),
                          lambda value: value.update(stdout=value["stdout"] + value["stdout"])):
            value = copy.deepcopy(original)
            transform(value)
            path.write_text(json.dumps(value))
            new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict(module.LABELED_RECEIPT_SHA256, {"accounting.json": new_sha}):
                with self.assertRaises(module.SubmissionError):
                    self.submit(Scheduler([]))

    def test_prior_qa_requires_exact108_and_valid_admission_binding(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "native_qa.json"
        original = json.loads(path.read_text())
        for key, replacement in (("tests_run", 107), ("skipped", 1), ("success", False),
                                 ("admission_binding_sha256", "0" * 64)):
            value = copy.deepcopy(original)
            value[key] = replacement
            path.write_text(json.dumps(value))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            self.plan["labeled_audit"]["qa_sha256"] = checksum
            self.write("extension/development_plan.json", self.plan)
            with patch.dict(module.LABELED_RECEIPT_SHA256, {"native_qa.json": checksum}):
                with self.assertRaisesRegex(module.SubmissionError, "QA108"):
                    self.submit(Scheduler([]))

    def test_live_prior_or_untracked_development_job_blocks(self):
        for row in ("1222926|SUSPENDED|sft-extension-labeled-feasibility\n",
                    "1222225|RUNNING|sft-extension-feasibility\n",
                    "999|PENDING|sft-extension-development-seal\n"):
            scheduler = Scheduler([result(row)])
            with self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_bad_or_nonexpanded_scheduler_snapshot_blocks(self):
        for response in (result("", "controller error", 1), result("9_[0-1]|PENDING|other\n"),
                         result("9|RUNNING|a\n9|RUNNING|a\n")):
            scheduler = Scheduler([response])
            with self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_source_mutation_during_queue_snapshot_blocks_before_sbatch(self):
        def change(command, kwargs):
            (self.extension / "development_seal.py").write_text("changed during query")
            return result()
        scheduler = Scheduler([change])
        with self.assertRaisesRegex(module.SubmissionError, "during admission"):
            self.submit(scheduler)
        self.assertEqual(len(scheduler.calls), 1)

    def test_symlinked_ledger_or_escaping_logs_cannot_modify_historical_state(self):
        historical = self.root / "deadline/job_ledger.json"
        historical.parent.mkdir()
        historical.write_text("immutable historical state")
        path = self.extension / "job_ledger.json"
        path.unlink()
        path.symlink_to(historical)
        with self.assertRaisesRegex(module.SubmissionError, "symlinks"):
            self.submit(Scheduler([]))
        self.assertEqual(historical.read_text(), "immutable historical state")
        path.unlink()
        self.write("extension/job_ledger.json", self.prior_ledger)
        self.script.write_text(self.script.read_text().replace(str(self.extension / "logs"), str(self.root / "deadline/logs")))
        with self.assertRaisesRegex(module.SubmissionError, "below extension"):
            self.submit(Scheduler([]))

    def test_missing_static_source_is_not_silently_dropped(self):
        (self.root / "tests/test_preserving_trainer.py").unlink()
        with self.assertRaisesRegex(module.SubmissionError, "Missing reviewed"):
            self.submit(Scheduler([]))


if __name__ == "__main__":
    unittest.main()
