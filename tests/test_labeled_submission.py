"""Offline metadata/scheduler oracles; run only in the native CPU allocation."""
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

from extension import labeled_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12346;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(command, kwargs) if callable(value) else value


class LabeledSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extension = self.root / "extension"
        for name in module.SOURCE_PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("small reviewed source\n")
        self.campaign = dict(schema="reasoning-sft-extension-campaign-v1", preserve_completed_experiments=True,
            authorization=dict(new_deadline_utc=module.DEADLINE.isoformat(),
                new_deadline_eastern="2026-10-27T23:59:00-04:00", latest_user_total_gpu_hour_limit=None),
            budget=dict(new_user_global_cap_gpu_hours=None, historical_cap_and_ledger_remain_unchanged=True,
                historical_ledger="deadline/job_ledger.json", new_extension_ledger="extension/job_ledger.json"))
        self.write("extension/campaign.json", self.campaign)
        self.plan = dict(schema="extension-labeled-feasibility-plan-v1", cpu_only=True,
            model_inference=False, choose_splits=False, automatic_downstream=False,
            deadline_utc=module.DEADLINE.isoformat(), wall_time_seconds=7200, reserve_seconds=172800,
            source_metadata="extension/development_source_metadata.json", sources=[
                dict(repo_id=name, revision=value["revision"], files=sorted(value["files"]))
                for name, value in module.SOURCE_PINS.items()])
        self.write("extension/labeled_plan.json", self.plan)
        self.script = self.extension / "labeled_feasibility.slurm"
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + value for value in (
            "--job-name=sft-extension-labeled-feasibility", "--account=tml_2026", "--partition=normal_q",
            "--qos=owl_normal_short", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=02:00:00", "--no-requeue", f"--output={self.extension}/logs/labeled-%j.out",
            f"--error={self.extension}/logs/labeled-%j.err")) + "\nset -euo pipefail\n")
        initial_binding = dict(role=module.INITIAL_ROLE, requested_gpu_hours=0, requested_gpu_seconds=0,
            command=["sbatch", "--clusters=owl", "--parsable", str(self.extension / "cpu_feasibility.slurm")])
        self.initial_entry = dict(status="submitted", job_id=module.INITIAL_JOB_ID,
            requested_gpu_hours=0, requested_gpu_seconds=0, binding=initial_binding,
            binding_sha256=module._sha(module._canonical(initial_binding)),
            immutable_history={"source": "original binding remains unchanged", "requested_at": "2026-10-20"})
        self.initial_ledger = dict(schema=module.SCHEMA, historical_ledger_unchanged="deadline/job_ledger.json",
                                  gpu_hour_cap=None, jobs={module.INITIAL_ROLE: self.initial_entry})
        self.write("extension/job_ledger.json", self.initial_ledger)
        qa = dict(schema="extension-feasibility-native-qa-v1", job_id=module.INITIAL_JOB_ID,
            success=True, gpu_disabled=True, wrapper_syntax_passed=True, tests_run=64,
            errors=0, failures=0, skipped=0, expected_failures=0, unexpected_successes=0,
            compute_proof=dict(gpu_requested_or_allocated=False))
        receipt_root = module.EVIDENCE_DIRECTORY
        self.write(receipt_root + "/native_qa.json", qa)
        self.qa_sha = hashlib.sha256((self.root / receipt_root / "native_qa.json").read_bytes()).hexdigest()
        complete = dict(schema="extension-cpu-feasibility-v1", complete=True,
            gpu_seconds_requested_or_used=0, input_stats_unchanged_at_closeout=True,
            native_qa=dict(tests_run=64, sha256=self.qa_sha))
        accounting = dict(schema="extension-terminal-cpu-accounting-v1", complete=True,
            job_id=module.INITIAL_JOB_ID, elapsed_seconds=221, gpu_seconds=0,
            no_gpu_in_alloc_tres=True, stderr="", stdout=module.INITIAL_JOB_ID +
            "|sft-extension-feasibility|COMPLETED|0:0|221|2026-10-20T22:59:54|2026-10-20T23:03:35|cpu=8,mem=64G,node=1\n")
        self.write(receipt_root + "/complete.json", complete)
        self.write(receipt_root + "/accounting.json", accounting)
        self.write(receipt_root + "/job_ledger_snapshot.json", self.initial_ledger)
        # Synthetic receipts remain cryptographically bound. The production map
        # contains the four exact native job-1222225 hashes, never fixture data.
        self.receipt_hashes = {name: hashlib.sha256((self.root / receipt_root / name).read_bytes()).hexdigest()
                               for name in module.INITIAL_RECEIPT_SHA256}
        patcher = patch.object(module, "INITIAL_RECEIPT_SHA256", self.receipt_hashes)
        patcher.start()
        self.addCleanup(patcher.stop)
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)
        self.now = module.DEADLINE - timedelta(days=5)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def ledger(self):
        return json.loads((self.extension / "job_ledger.json").read_text())

    def submit(self, scheduler=None, **kwargs):
        return module.submit_labeled_feasibility(self.root, now=kwargs.pop("now", self.now),
            runner=Scheduler() if scheduler is None else scheduler, **kwargs)

    def replace_receipt(self, name, value):
        self.write(module.EVIDENCE_DIRECTORY + "/" + name, value)
        self.receipt_hashes[name] = hashlib.sha256((self.root / module.EVIDENCE_DIRECTORY / name).read_bytes()).hexdigest()

    def test_fixed_command_zero_gpu_plan_and_prior_evidence_binding(self):
        scheduler = Scheduler([result("CLUSTER: owl\n900_0|RUNNING|unrelated\n"), result("12346;owl\n")])
        self.assertEqual(self.submit(scheduler), "12346")
        self.assertEqual(scheduler.calls[0][0], ["squeue", "--clusters=owl", "--array", "--states=all",
            "--user=fixture_user", "--noheader", "--format=%64i|%64T|%128j"])
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.script)])
        value = self.ledger()
        self.assertEqual(value["jobs"][module.INITIAL_ROLE], self.initial_entry)
        entry = value["jobs"][module.ROLE]
        self.assertEqual(entry["requested_gpu_hours"], 0)
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        self.assertEqual(entry["binding"]["wall_time_seconds"], 7200)
        self.assertEqual(entry["binding"]["closeout_reserve_seconds"], 172800)
        self.assertEqual(entry["source_sha256"], module.source_inventory(self.root))
        self.assertEqual(entry["plan_sha256"], hashlib.sha256((self.extension / "labeled_plan.json").read_bytes()).hexdigest())
        self.assertEqual(len(entry["binding"]["initial_evidence"]), 4)

    def test_durable_submitting_record_and_old_entry_precede_sbatch(self):
        def submitted(command, kwargs):
            ledger = self.ledger()
            self.assertEqual(ledger["jobs"][module.ROLE]["status"], "submitting")
            self.assertEqual(ledger["jobs"][module.INITIAL_ROLE], self.initial_entry)
            return result("12346;owl")
        self.submit(Scheduler([result(), submitted]))

    def test_exact_successful_retry_is_idempotent_and_excludes_runtime(self):
        self.submit()
        (self.extension / "runtime.json").write_text("mutable runtime")
        (self.extension / "runs").mkdir()
        (self.extension / "runs/large_dataset.json").write_text("must never enter source binding")
        scheduler = Scheduler([result("12346|PENDING|sft-extension-labeled-feasibility\n")])
        self.assertEqual(self.submit(scheduler), "12346")
        self.assertEqual(len(scheduler.calls), 1)
        self.assertEqual(self.ledger()["jobs"][module.INITIAL_ROLE], self.initial_entry)

    def test_source_or_full_plan_byte_change_blocks_retry(self):
        self.submit()
        for relative in ("extension/labeled_plan.json", "extension/labeled_feasibility.py",
                         "tests/test_labeled_submission.py", "extension/submission.py"):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            scheduler = Scheduler([result()])
            with self.subTest(relative=relative), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)
            path.write_bytes(original)

    def test_unknown_submitting_failed_and_uncertain_records_never_retry(self):
        for status in ("unknown", "submitting", "uncertain", "submission_failed", "complete", None):
            ledger = copy.deepcopy(self.initial_ledger)
            ledger["jobs"][module.ROLE] = dict(status=status)
            self.write("extension/job_ledger.json", ledger)
            scheduler = Scheduler([result()])
            with self.subTest(status=status), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_timeout_preserves_possible_submission_and_initial_entry(self):
        scheduler = Scheduler([result(), subprocess.TimeoutExpired("sbatch", 60, output=b"12346;owl\n")])
        with self.assertRaisesRegex(module.SubmissionError, "uncertain"):
            self.submit(scheduler)
        ledger = self.ledger()
        self.assertEqual(ledger["jobs"][module.INITIAL_ROLE], self.initial_entry)
        self.assertEqual(ledger["jobs"][module.ROLE]["status"], "uncertain")
        self.assertEqual(ledger["jobs"][module.ROLE]["scheduler_stdout"], "12346;owl\n")
        retry = Scheduler([result()])
        with self.assertRaises(module.SubmissionError):
            self.submit(retry)
        self.assertEqual(len(retry.calls), 1)

    def test_nonzero_or_malformed_response_is_preserved_without_retry(self):
        for response in (result("12346;owl", "controller failure", 1), result("12346;other"),
                         result("Submitted batch job 12346"), result("12346\n12347")):
            self.write("extension/job_ledger.json", self.initial_ledger)
            with self.subTest(response=response.stdout), self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([result(), response]))
            entry = self.ledger()["jobs"][module.ROLE]
            self.assertIn(entry["status"], ("uncertain", "submission_failed"))
            if response.returncode:
                self.assertEqual(entry["possible_job_id"], "12346")

    def test_sanitized_environment_cannot_override_cpu_request(self):
        scheduler = Scheduler()
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:8", "SBATCH_ARRAY_INX": "0-8",
                "SLURM_JOB_ID": "999", "SLURM_JOB_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0"}):
            self.submit(scheduler)
        environment = scheduler.calls[1][1]["env"]
        self.assertFalse(any(key.startswith("SBATCH_") for key in environment))
        for key in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
            self.assertNotIn(key, environment)

    def test_48_hour_reserve_and_exact_new_deadline_required(self):
        scheduler = Scheduler([])
        with self.assertRaisesRegex(module.SubmissionError, "48-hour"):
            self.submit(scheduler, now=module.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(scheduler.calls, [])
        self.plan["deadline_utc"] = "2026-10-21T03:59:00+00:00"
        self.write("extension/labeled_plan.json", self.plan)
        with self.assertRaisesRegex(module.SubmissionError, "deadline"):
            self.submit(Scheduler([]))

    def test_exact_boundary_admits_finite_cpu_job(self):
        self.assertEqual(self.submit(now=module.DEADLINE - timedelta(seconds=180000)), "12346")

    def test_scope_flags_and_pinned_train_sources_are_enforced(self):
        for flag in ("cpu_only", "model_inference", "choose_splits", "automatic_downstream"):
            value = copy.deepcopy(self.plan)
            value[flag] = not value[flag]
            self.write("extension/labeled_plan.json", value)
            with self.subTest(flag=flag), self.assertRaisesRegex(module.SubmissionError, "CPU feasibility only"):
                self.submit(Scheduler([]))
        for edit in (lambda value: value["sources"][0].update(revision="main"),
                     lambda value: value["sources"][1]["files"].append("ARC-Easy/train-00000-of-00001.parquet"),
                     lambda value: value["sources"].append(copy.deepcopy(value["sources"][0]))):
            value = copy.deepcopy(self.plan)
            edit(value)
            self.write("extension/labeled_plan.json", value)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_gpu_other_resource_or_hidden_scheduler_directive_refused(self):
        original = self.script.read_text()
        for addition in ("--gpus=1", "--gres=gpu:1", "--array=0-9", "-G 1"):
            self.script.write_text(original.replace("set -euo pipefail", "#SBATCH " + addition + "\nset -euo pipefail"))
            with self.subTest(addition=addition), self.assertRaisesRegex(module.SubmissionError, "GPU/unknown"):
                self.submit(Scheduler([]))
        for before, after in (("02:00:00", "UNLIMITED"), ("--cpus-per-task=8", "--cpus-per-task=16"),
                              ("--mem=64G", "--mem=128G"), ("--qos=owl_normal_short", "--qos=other")):
            self.script.write_text(original.replace(before, after))
            with self.subTest(after=after), self.assertRaisesRegex(module.SubmissionError, "fixed 8-CPU"):
                self.submit(Scheduler([]))
        self.script.write_text(original.replace("#!/bin/bash", "#!/bin/bash\necho executable"))
        with self.assertRaisesRegex(module.SubmissionError, "ignored"):
            self.submit(Scheduler([]))

    def test_prior_receipt_digest_is_strict(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "accounting.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(module.SubmissionError, "digest mismatch"):
            self.submit(Scheduler([]))

    def test_prior_qa_requires_64_without_skip(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "native_qa.json"
        original = json.loads(path.read_text())
        for key, value in (("tests_run", 63), ("skipped", 1), ("success", False), ("gpu_disabled", False)):
            qa = copy.deepcopy(original)
            qa[key] = value
            self.replace_receipt("native_qa.json", qa)
            with self.subTest(key=key), self.assertRaisesRegex(module.SubmissionError, "64 tests"):
                self.submit(Scheduler([]))

    def test_prior_accounting_rejects_nonterminal_gpu_or_duplicate_allocations(self):
        path = self.root / module.EVIDENCE_DIRECTORY / "accounting.json"
        original = json.loads(path.read_text())
        for transformation in (lambda text: text.replace("COMPLETED", "RUNNING"),
                               lambda text: text.replace("cpu=8", "gres/gpu=1,cpu=8"),
                               lambda text: text + text):
            account = copy.deepcopy(original)
            account["stdout"] = transformation(account["stdout"])
            self.replace_receipt("accounting.json", account)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_unknown_role_or_changed_initial_entry_blocks_scheduler(self):
        for edit in (lambda value: value["jobs"].update(gpu_stage={"job_id": "999"}),
                     lambda value: value["jobs"][module.INITIAL_ROLE].update(status="complete")):
            ledger = copy.deepcopy(self.initial_ledger)
            edit(ledger)
            self.write("extension/job_ledger.json", ledger)
            scheduler = Scheduler([])
            with self.assertRaisesRegex(module.SubmissionError, "changed initial entry"):
                self.submit(scheduler)
            self.assertEqual(scheduler.calls, [])

    def test_missing_initial_ledger_does_not_create_one(self):
        (self.extension / "job_ledger.json").unlink()
        with self.assertRaisesRegex(module.SubmissionError, "initial-stage"):
            self.submit(Scheduler([]))
        self.assertFalse((self.extension / "job_ledger.json").exists())

    def test_untracked_new_or_live_old_cpu_job_blocks(self):
        for row in ("999|SUSPENDED|sft-extension-labeled-feasibility\n",
                    module.INITIAL_JOB_ID + "|RUNNING|sft-extension-feasibility\n"):
            scheduler = Scheduler([result(row)])
            with self.subTest(row=row), self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)

    def test_snapshot_error_nonexpanded_or_duplicate_rows_blocks(self):
        for response in (result("", "scheduler error", 1), result("9_[0-2]|PENDING|other\n"),
                         result("9|RUNNING|a\n9|RUNNING|a\n")):
            scheduler = Scheduler([response])
            with self.assertRaises(module.SubmissionError):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls), 1)
            self.assertNotIn(module.ROLE, self.ledger()["jobs"])

    def test_source_change_during_scheduler_query_blocks(self):
        def change(command, kwargs):
            (self.extension / "labeled_feasibility.py").write_text("mutated")
            return result()
        scheduler = Scheduler([change])
        with self.assertRaisesRegex(module.SubmissionError, "during admission"):
            self.submit(scheduler)
        self.assertEqual(len(scheduler.calls), 1)
        self.assertNotIn(module.ROLE, self.ledger()["jobs"])

    def test_logs_and_ledger_cannot_escape_extension(self):
        historical = self.root / "deadline/job_ledger.json"
        historical.parent.mkdir()
        historical.write_text("immutable historical data")
        ledger = self.extension / "job_ledger.json"
        ledger.unlink()
        ledger.symlink_to(historical)
        with self.assertRaisesRegex(module.SubmissionError, "symlinks"):
            self.submit(Scheduler([]))
        self.assertEqual(historical.read_text(), "immutable historical data")
        ledger.unlink()
        self.write("extension/job_ledger.json", self.initial_ledger)
        self.script.write_text(self.script.read_text().replace(str(self.extension / "logs"), str(self.root / "deadline/logs")))
        with self.assertRaisesRegex(module.SubmissionError, "below extension"):
            self.submit(Scheduler([]))

    def test_duplicate_or_nonfinite_metadata_rejected(self):
        for text in ('{"schema":"one","schema":"two"}', '{"schema":NaN}'):
            (self.extension / "labeled_plan.json").write_text(text)
            with self.assertRaises(module.SubmissionError):
                self.submit(Scheduler([]))

    def test_missing_required_source_is_not_silently_dropped(self):
        (self.root / "tests/test_labeled_feasibility.py").unlink()
        scheduler = Scheduler([])
        with self.assertRaisesRegex(module.SubmissionError, "Missing required"):
            self.submit(scheduler)
        self.assertEqual(scheduler.calls, [])


if __name__ == "__main__":
    unittest.main()
