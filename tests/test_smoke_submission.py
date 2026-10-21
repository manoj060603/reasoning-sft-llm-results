"""Offline admission oracles. Execute later only inside native CPU QA."""
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

from extension import smoke_submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses = list([result(), result("12349;owl\n")] if responses is None else responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(command, kwargs) if callable(value) else value


class SmokeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.extension = self.root / "extension"
        project = Path(module.__file__).resolve().parents[1]
        for name in module.SOURCE_PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            old = name in module.previous.SOURCE_PATHS or name in (module.CPU_PLAN, module.GPU_PLAN)
            path.write_bytes((project / name).read_bytes() if old else b"bounded synthetic reviewed source fixture\n")
        earlier = module.previous.previous
        for directory, pins in (
            (earlier.labeled.EVIDENCE_DIRECTORY, earlier.labeled.INITIAL_RECEIPT_SHA256),
            (earlier.EVIDENCE_DIRECTORY, earlier.LABELED_RECEIPT_SHA256),
            (module.previous.FAILED_DIRECTORY, module.previous.FAILED_RECEIPT_SHA256),
            (module.previous.DIAGNOSTIC_DIRECTORY, module.previous.DIAGNOSTIC_RECEIPT_SHA256),
            (module.EVIDENCE_DIRECTORY, module.RECEIPT_SHA256)):
            for name in pins:
                path = self.root / directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((project / directory / name).read_bytes())
        self.prior_ledger = self.read(module.EVIDENCE_DIRECTORY + "/job_ledger_snapshot.json")
        self.prior_jobs = copy.deepcopy(self.prior_ledger["jobs"])
        self.write("extension/job_ledger.json", self.prior_ledger)
        self.cpu_plan = self.read(module.CPU_PLAN)
        self.cpu_plan["native_plan"]["sha256"] = module._sha((self.root / module.GPU_PLAN).read_bytes())
        self.write(module.CPU_PLAN, self.cpu_plan)
        self.cpu_script = self.wrapper(gpu=False)
        self.gpu_script = self.wrapper(gpu=True)
        self.now = module.DEADLINE - timedelta(days=5)
        user = patch.object(module.metadata.getpass, "getuser", return_value="fixture_user")
        user.start()
        self.addCleanup(user.stop)

    def read(self, relative):
        return json.loads((self.root / relative).read_text())

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = module._canonical(value) + b"\n"
        path.write_bytes(raw)
        return {"path": relative, "sha256": module._sha(raw)}

    def wrapper(self, gpu):
        script = self.root / (module.GPU_SCRIPT if gpu else module.CPU_SCRIPT)
        directives = ["--job-name=" + (module.GPU_NAME if gpu else module.CPU_NAME), "--account=tml_2026",
            "--partition=" + ("b200_normal_q" if gpu else "normal_q"),
            "--qos=" + ("owl_b200_normal_short" if gpu else "owl_normal_short"), "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=" + ("128G" if gpu else "64G"), "--time=02:00:00", "--no-requeue",
            "--output=" + str(self.extension / "logs/stage-%j.out"),
            "--error=" + str(self.extension / "logs/stage-%j.err")]
        if gpu:
            directives.append("--gres=gpu:1")
        script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH " + item for item in directives) + "\nset -euo pipefail\n")
        return script

    def ledger(self):
        return self.read("extension/job_ledger.json")

    def cpu(self, scheduler=None, **kwargs):
        return module.submit_cpu_preparation(self.root, runner=Scheduler() if scheduler is None else scheduler,
            now=kwargs.pop("now", self.now), **kwargs)

    def gpu(self, scheduler=None, **kwargs):
        default = Scheduler([result(), result(), result("12350;owl")])
        return module.submit_gpu_diagnostic(self.root, cpu_run=kwargs.pop("cpu_run", self.cpu_run),
            runner=default if scheduler is None else scheduler, now=kwargs.pop("now", self.now), **kwargs)

    def ready_cpu(self):
        self.assertEqual(self.cpu(), "12349")
        self.cpu_run = "extension/runs/native-smoke-preparation-12349"
        sources = module.source_inventory(self.root)
        code = {name: item["sha256"] for name, item in sources.items()}
        compute = {"job_id": "12349", "node": "owl060", "gpu_requested_or_allocated": False}
        common = {"success": True, "tests_run": 300, "failures": 0, "errors": 0, "skipped": 0,
            "expected_failures": 0, "unexpected_successes": 0, "gpu_disabled": True,
            "job_id": "12349", "node": "owl060", "compute_proof": compute,
            "source_inventory": sources, "code_sha256": code}
        unit = self.write(self.cpu_run + "/unit_qa.json", {**common,
            "schema": "extension-native-smoke-preparation-unit-qa-v1",
            "plan_sha256": sources[module.CPU_PLAN]["sha256"], "native_plan_sha256": sources[module.GPU_PLAN]["sha256"]})
        prepared = {"path": self.cpu_run + "/prepared.json", "sha256": "a" * 64}
        # This intentionally is not a prepared cache. Admission must never open it.
        self.write(prepared["path"], {"private": "no cache/model/data reads on login"})
        qa = self.write(self.cpu_run + "/native_qa.json", {**common,
            "schema": "extension-native-model-admission-native-qa-v1", "plan_sha256": sources[module.GPU_PLAN]["sha256"],
            "cpu_preparation_plan_sha256": sources[module.CPU_PLAN]["sha256"], "prepared_sha256": prepared["sha256"]})
        preparation = self.write(self.cpu_run + "/preparation.json", {
            "schema": "extension-native-smoke-preparation-v1", "complete": True, "cpu_only": True,
            "gpu_seconds": 0, "plan_sha256": sources[module.GPU_PLAN]["sha256"], "code_sha256": code,
            "native_qa": {**unit, "path": str(self.root / unit["path"]), "schema": "extension-native-smoke-preparation-unit-qa-v1",
                "tests_run": 300, "code_sha256": code}, "input_snapshot_unchanged": True,
            "no_model_or_benchmark_outcomes": True})
        self.write(self.cpu_run + "/complete.json", {
            "schema": "extension-native-smoke-preparation-complete-v1", "complete": True,
            "cpu_only": True, "job_id": "12349", "gpu_seconds": 0, "input_snapshot_unchanged": True,
            "plan_sha256": sources[module.CPU_PLAN]["sha256"], "native_plan_sha256": sources[module.GPU_PLAN]["sha256"],
            "code_sha256": code, "source_inventory": sources, "prepared": prepared, "native_qa": qa,
            "unit_qa": unit, "preparation": preparation, "compute_start": compute, "compute_end": compute,
            "output_sha256": {"prepared.json": prepared["sha256"], "native_qa.json": qa["sha256"],
                "unit_qa.json": unit["sha256"], "preparation.json": preparation["sha256"]}})
        self.write(self.cpu_run + "/accounting.json", {
            "schema": "extension-terminal-cpu-accounting-v1", "complete": True, "job_id": "12349",
            "gpu_seconds": 0, "no_gpu_in_alloc_tres": True, "elapsed_seconds": 60, "stderr": "",
            "stdout": "".join(job + "|fixture|COMPLETED|0:0|60|start|end|cpu=8,mem=64G,node=1\n"
                for job in ("12349", "12349.batch", "12349.extern"))})

    def test_cpu_fixed_zero_gpu_command_and_all_six_entries_preserved(self):
        scheduler = Scheduler()
        self.assertEqual(self.cpu(scheduler), "12349")
        self.assertEqual(scheduler.calls[1][0], ["sbatch", "--clusters=owl", "--parsable", str(self.cpu_script)])
        entry = self.ledger()["jobs"][module.CPU_ROLE]
        self.assertEqual(entry["source_sha256"], module.source_inventory(self.root))
        self.assertEqual(len(entry["source_sha256"]), 51)
        self.assertEqual(len(entry["initial_evidence"]), 19)
        self.assertEqual(entry["requested_gpu_seconds"], 0)
        for role, prior in self.prior_jobs.items():
            self.assertEqual(self.ledger()["jobs"][role], prior)

    def test_submitting_entry_is_durable_before_scheduler_call(self):
        def submit(command, kwargs):
            self.assertEqual(self.ledger()["jobs"][module.CPU_ROLE]["status"], "submitting")
            return result("12349;owl")
        self.cpu(Scheduler([result(), submit]))

    def test_source_inventory_excludes_every_mutable_runtime_receipt(self):
        before = module.source_inventory(self.root)
        self.cpu()
        self.write("extension/runs/unrelated.json", {"mutable": True})
        self.write("extension/operations/runtime-proof.json", {"mutable": True})
        self.assertEqual(before, module.source_inventory(self.root))

    def test_all_38_prior_sources_and_19_receipts_remain_pinned(self):
        for relative in ("extension/preserving_objectives.py", module.EVIDENCE_DIRECTORY + "/complete.json",
            module.EVIDENCE_DIRECTORY + "/native_qa.json", module.EVIDENCE_DIRECTORY + "/accounting.json"):
            path = self.root / relative
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(relative=relative), self.assertRaises(module.SubmissionError):
                self.cpu(Scheduler([]))
            path.write_bytes(before)

    def test_changed_or_unknown_prior_history_is_never_normalized(self):
        for role in self.prior_jobs:
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][role]["status"] = "unknown"
            self.write("extension/job_ledger.json", ledger)
            with self.subTest(role=role), self.assertRaises(module.SubmissionError):
                self.cpu(Scheduler([]))
        self.write("extension/job_ledger.json", self.prior_ledger)
        ledger = self.ledger()
        ledger["jobs"]["arbitrary_role"] = {"status": "submitted"}
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaises(module.SubmissionError):
            self.cpu(Scheduler([]))

    def test_proven_terminal_prior_may_remain_visible_but_active_unknown_duplicate_cannot(self):
        queue = "1223943|COMPLETED|sft-extension-development-seal-v2\n1223566|FAILED|sft-extension-development-seal\n"
        self.assertEqual(self.cpu(Scheduler([result(queue), result("12349;owl")])), "12349")
        self.write("extension/job_ledger.json", self.prior_ledger)
        for row in ("1223943|RUNNING|sft-extension-development-seal-v2\n",
            "1223943|REQUEUED|sft-extension-development-seal-v2\n", "999|PENDING|sft-extension-development-seal-v2\n"):
            with self.subTest(row=row), self.assertRaises(module.SubmissionError):
                self.cpu(Scheduler([result(row)]))

    def test_exact_cpu_submission_idempotence_never_sends_second_sbatch(self):
        self.cpu()
        scheduler = Scheduler([result("12349|PENDING|" + module.CPU_NAME)])
        self.assertEqual(self.cpu(scheduler), "12349")
        self.assertEqual(len(scheduler.calls), 1)

    def test_failed_uncertain_submitting_existing_roles_never_resubmit(self):
        for status in ("uncertain", "submitting", "submission_failed", "complete", None):
            ledger = copy.deepcopy(self.prior_ledger)
            ledger["jobs"][module.CPU_ROLE] = {"status": status}
            self.write("extension/job_ledger.json", ledger)
            with self.subTest(status=status), self.assertRaisesRegex(module.SubmissionError, "never resubmit"):
                self.cpu(Scheduler([result()]))

    def test_scheduler_timeout_or_ambiguous_response_is_retained_without_retry(self):
        for value, status in ((subprocess.TimeoutExpired("sbatch", 60), "uncertain"),
            (result("not-an-id"), "uncertain"), (result("", "controller rejected", 1), "submission_failed")):
            self.write("extension/job_ledger.json", self.prior_ledger)
            scheduler = Scheduler([result(), value])
            with self.assertRaises(module.SubmissionError):
                self.cpu(scheduler)
            self.assertEqual(self.ledger()["jobs"][module.CPU_ROLE]["status"], status)
            self.assertEqual(len(scheduler.calls), 2)

    def test_deadline_reserves_48_hours_plus_two_hour_work(self):
        with self.assertRaises(module.SubmissionError):
            self.cpu(Scheduler([]), now=module.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(self.cpu(now=module.DEADLINE - timedelta(seconds=180000)), "12349")

    def test_scheduler_environment_strips_overrides_but_preserves_slurm_configuration(self):
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:9", "SLURM_JOB_ID": "999", "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_CONF": "/fixture/slurm.conf", "SLURM_CONF_SERVER": "owl-controller"}):
            scheduler = Scheduler()
            self.cpu(scheduler)
        environment = scheduler.calls[1][1]["env"]
        for key in ("SBATCH_GRES", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["SLURM_CONF"], "/fixture/slurm.conf")
        self.assertEqual(environment["SLURM_CONF_SERVER"], "owl-controller")

    def test_fixed_cpu_and_gpu_resources_unknown_options_and_late_directives(self):
        for script, gpu, before, after in (
            (self.cpu_script, False, "--mem=64G", "--mem=128G"),
            (self.gpu_script, True, "--gres=gpu:1", "--gres=gpu:2"),
            (self.gpu_script, True, "b200_normal_q", "normal_q")):
            raw = script.read_text()
            script.write_text(raw.replace(before, after))
            with self.assertRaises(module.SubmissionError):
                module.validate_wrapper(script, self.root, gpu=gpu)
            script.write_text(raw)
        raw = self.cpu_script.read_text()
        for suffix in ("#SBATCH --gres=gpu:1\n", "#SBATCH --array=0-1\n"):
            self.cpu_script.write_text(raw + suffix)
            with self.assertRaises(module.SubmissionError):
                module.validate_wrapper(self.cpu_script, self.root)
        self.cpu_script.write_text(raw)

    def test_cpu_plan_scope_and_gpu_resource_deadline_are_fail_closed(self):
        sources = module.source_inventory(self.root)
        cpu, gpu = self.read(module.CPU_PLAN), self.read(module.GPU_PLAN)
        for change in (lambda x: x.update(model_inference=True), lambda x: x.update(automatic_downstream=True),
            lambda x: x["native_plan"].update(sha256="0" * 64)):
            changed = copy.deepcopy(cpu)
            change(changed)
            with self.assertRaises(module.SubmissionError):
                module.validate_plans(changed, gpu, sources)
        for change in (lambda x: x["resources"].update(gpus=2),
            lambda x: x["resources"].update(maximum_initial_concurrent_gpu_allocations=3),
            lambda x: x.update(deadline_utc="2026-10-29T03:59:00+00:00")):
            changed = copy.deepcopy(gpu)
            change(changed)
            with self.assertRaises(module.SubmissionError):
                module.validate_plans(cpu, changed, sources)

    def test_gpu_requires_exact_terminal_cpu_run_and_never_reads_private_cache(self):
        self.ready_cpu()
        original = module.metadata._small_bytes
        def guarded(path, root):
            self.assertNotEqual(Path(path), self.root / self.cpu_run / "prepared.json")
            self.assertNotIn(str(path), (str(self.root / "deadline/artifacts/pool.json"),
                str(self.root / "deadline/artifacts/features.json")))
            return original(path, root)
        with patch.object(module.metadata, "_small_bytes", side_effect=guarded):
            self.assertEqual(self.gpu(), "12350")
        entry = self.ledger()["jobs"][module.GPU_ROLE]
        self.assertEqual(entry["requested_gpu_seconds"], 7200)
        self.assertEqual(entry["requested_gpu_hours"], 2)
        self.assertEqual(entry["command"], ["sbatch", "--clusters=owl", "--parsable", str(self.gpu_script), self.cpu_run])
        receipt = self.read("extension/operations/native-model-admission-job-12350.json")
        self.assertEqual(receipt["schema"], "extension-native-model-admission-job-v1")
        self.assertEqual(receipt["prepared_sha256"], "a" * 64)
        self.assertEqual(receipt["binding_sha256"], entry["binding_sha256"])
        self.assertFalse(receipt["automatic_downstream"])

    def test_gpu_cannot_precede_cpu_or_use_arbitrary_run_path(self):
        self.cpu_run = "extension/runs/native-smoke-preparation-12349"
        with self.assertRaises(module.SubmissionError):
            self.gpu(Scheduler([]))
        self.ready_cpu()
        for relative in ("/tmp/private", "extension/runs/native-smoke-preparation-999", "extension/../prepared.json"):
            with self.subTest(relative=relative), self.assertRaises(module.SubmissionError):
                self.gpu(Scheduler([]), cpu_run=relative)

    def test_gpu_rejects_cpu_partial_failed_or_gpu_accounting_before_scheduler(self):
        self.ready_cpu()
        path = self.cpu_run + "/accounting.json"
        original = self.read(path)
        for change in (lambda x: x.update(complete=False), lambda x: x.update(gpu_seconds=1),
            lambda x: x.update(stdout=x["stdout"].replace("COMPLETED", "FAILED", 1)),
            lambda x: x.update(stdout=x["stdout"].splitlines()[1] + "\n"),
            lambda x: x.update(stdout=x["stdout"] + x["stdout"].splitlines()[0] + "\n"),
            lambda x: x.update(stdout=x["stdout"].replace("cpu=8", "cpu=8,gres/gpu=1", 1))):
            value = copy.deepcopy(original)
            change(value)
            self.write(path, value)
            with self.assertRaises(module.SubmissionError):
                self.gpu(Scheduler([]))
        self.write(path, original)

    def test_gpu_rejects_mutated_bound_receipts_and_unknown_cpu_binding(self):
        self.ready_cpu()
        for name in ("native_qa.json", "unit_qa.json", "preparation.json"):
            path = self.root / self.cpu_run / name
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.assertRaises(module.SubmissionError):
                self.gpu(Scheduler([]))
            path.write_bytes(before)
        ledger = self.ledger()
        ledger["jobs"][module.CPU_ROLE]["binding"] = {}
        ledger["jobs"][module.CPU_ROLE]["binding_sha256"] = module._sha(module._canonical({}))
        self.write("extension/job_ledger.json", ledger)
        with self.assertRaises(module.SubmissionError):
            self.gpu(Scheduler([]))

    def test_gpu_requires_same_51_tested_source_bytes_after_cpu_complete(self):
        self.ready_cpu()
        for relative in ("extension/native_model_admission.py", "extension/native_model_admission_plan.json",
            "reasoning/features.py"):
            path = self.root / relative
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.assertRaises(module.SubmissionError):
                self.gpu(Scheduler([]))
            path.write_bytes(before)

    def test_gpu_concurrency_includes_pending_suspended_and_expanded_arrays(self):
        self.ready_cpu()
        rows = "41_0|PENDING|other|gpu:1\n41_1|SUSPENDED|other|gpu:b200:1\n"
        scheduler = Scheduler([result(), result(rows)])
        with self.assertRaisesRegex(module.SubmissionError, "Two initial"):
            self.gpu(scheduler)
        self.assertEqual(len(scheduler.calls), 2)
        rows = "41_0|COMPLETED|other|gpu:1\n41_1|RUNNING|other|gpu:b200:1\n42|PENDING|cpu|(null)\n"
        self.assertEqual(self.gpu(Scheduler([result(), result(rows), result("12350;owl")])), "12350")

    def test_gpu_queue_unknown_gres_duplicate_or_snapshot_failure_never_admits(self):
        for response in (result("1|PENDING|other|unknown\n"), result("1|RUNNING|x|gpu:1\n1|RUNNING|x|gpu:1\n"),
            result("", "controller unavailable", 1), result("1_[0-2]|PENDING|other|gpu:1\n")):
            with self.assertRaises(module.SubmissionError):
                module.gpu_queue_snapshot(Scheduler([response]))

    def test_gpu_requires_cpu_terminal_even_if_accounting_receipt_exists(self):
        self.ready_cpu()
        with self.assertRaises(module.SubmissionError):
            self.gpu(Scheduler([result("12349|RUNNING|" + module.CPU_NAME)]))

    def test_exact_gpu_retry_can_repair_missing_receipt_without_second_sbatch(self):
        self.ready_cpu()
        self.gpu()
        receipt = self.root / "extension/operations/native-model-admission-job-12350.json"
        expected = receipt.read_bytes()
        receipt.unlink()
        scheduler = Scheduler([result("12350|PENDING|" + module.GPU_NAME)])
        self.assertEqual(self.gpu(scheduler), "12350")
        self.assertEqual(len(scheduler.calls), 1)
        self.assertEqual(receipt.read_bytes(), expected)
        for role, entry in self.prior_jobs.items():
            self.assertEqual(self.ledger()["jobs"][role], entry)

    def test_conflicting_gpu_receipt_cannot_be_overwritten_or_resubmitted(self):
        self.ready_cpu()
        self.gpu()
        self.write("extension/operations/native-model-admission-job-12350.json", {"status": "conflict"})
        with self.assertRaises(module.SubmissionError):
            self.gpu(Scheduler([result()]))

    def test_gpu_timeout_records_uncertainty_without_admission_receipt(self):
        self.ready_cpu()
        with self.assertRaises(module.SubmissionError):
            self.gpu(Scheduler([result(), result(), subprocess.TimeoutExpired("sbatch", 60)]))
        self.assertEqual(self.ledger()["jobs"][module.GPU_ROLE]["status"], "uncertain")
        self.assertFalse((self.root / "extension/operations/native-model-admission-job-12350.json").exists())


if __name__ == "__main__":
    unittest.main()
