from datetime import timedelta
import json
import os
import subprocess
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from deadline import pipeline
from deadline.pipeline import DEADLINE, guard_start
from deadline.gpu_budget import make_terminal_receipt
from reasoning.evaluate import checkpoint_identity


class DeadlineStartTests(unittest.TestCase):
    def test_sufficient_reserve(self):
        guard_start(6, DEADLINE - timedelta(hours=7))

    def test_late_queued_job_cannot_start(self):
        with self.assertRaisesRegex(RuntimeError, "Queue delay"):
            guard_start(6, DEADLINE - timedelta(hours=5))

    def test_nonpositive_reserve_rejected(self):
        with self.assertRaises(ValueError):
            guard_start(0, DEADLINE - timedelta(hours=7))


class EvaluationIdentityTests(unittest.TestCase):
    def test_changed_chat_template_changes_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / 'config.json').write_text('{}')
            template = model / 'chat_template.jinja'
            template.write_text('original template')
            before = checkpoint_identity(directory, None)
            template.write_text('changed template')
            after = checkpoint_identity(directory, None)
            self.assertIn('chat_template.jinja', before)
            self.assertNotEqual(before, after)


class LoginSubmissionTests(unittest.TestCase):
    def test_afterany_continuation_uses_verified_budget_and_preserves_request_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = {"status": "submitted", "job_id": "111", "requested_gpu_hours": 42}
            prior["gpu_accounting_receipt"] = make_terminal_receipt("prior", prior,
                "111|111|COMPLETED|3600|gres/gpu=1,gres/gpu:b200=1\n",
                live_job_ids=(), observed_at_utc="2026-10-20T00:00:00+00:00")
            path = root / "job_ledger.json"
            path.write_text(json.dumps({"jobs": {"prior": prior}}))
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                scheduler.side_effect = [subprocess.CompletedProcess([], 0, "222\n", ""),
                                         subprocess.CompletedProcess([], 0, "333;owl\n", "")]
                pipeline.submit("continuation", root / "extended.slurm", dependency="222",
                    dependency_kind="afterany", gpu_hours=18, required_hours=19)
                self.assertIn("--array", scheduler.call_args_list[0].args[0])
                self.assertIn("--states=all", scheduler.call_args_list[0].args[0])
                self.assertIn("--dependency=afterany:222", scheduler.call_args_list[1].args[0])
            ledger = json.loads(path.read_text())
            self.assertEqual(ledger["jobs"]["prior"]["requested_gpu_hours"], 42)
            self.assertEqual(sum(job["requested_gpu_hours"] for job in ledger["jobs"].values()), 60)

    def test_live_reconciled_job_and_failed_scheduler_snapshot_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = {"status": "submitted", "job_id": "111", "requested_gpu_hours": 42}
            prior["gpu_accounting_receipt"] = make_terminal_receipt("prior", prior,
                "111|111|COMPLETED|3600|gres/gpu=1\n", live_job_ids=(),
                observed_at_utc="2026-10-20T00:00:00+00:00")
            path = root / "job_ledger.json"
            path.write_text(json.dumps({"jobs": {"prior": prior}}))
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                scheduler.return_value = subprocess.CompletedProcess([], 0, "111\n", "")
                with self.assertRaisesRegex(RuntimeError, "48-GPU-hour"):
                    pipeline.submit("continuation", root / "extended.slurm", gpu_hours=18)
                scheduler.assert_called_once()
                scheduler.reset_mock()
                scheduler.side_effect = subprocess.CalledProcessError(1, ["squeue"])
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.submit("continuation", root / "extended.slurm", gpu_hours=18)
                scheduler.assert_called_once()
            self.assertNotIn("continuation", json.loads(path.read_text())["jobs"])

    def test_invalid_dependency_kind_never_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pipeline, "ROOT", Path(directory)), patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                with self.assertRaisesRegex(ValueError, "dependency kind"):
                    pipeline.submit("role", "extended.slurm", dependency="222", dependency_kind="singleton")
                scheduler.assert_not_called()

    def test_single_and_all_success_dependencies_are_preserved_with_no_gpu_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "job_ledger.json"
            path.write_text(json.dumps({"jobs": {"prior": {"status": "submitted", "job_id": "1",
                                                             "requested_gpu_hours": 42}}}))
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                scheduler.return_value = subprocess.CompletedProcess([], 0, "333;owl\n", "")
                for index, dependency in enumerate((123, "123", "1207982:1208059:1208060")):
                    with self.subTest(dependency=dependency):
                        pipeline.submit(f"cpu_closeout_{index}", root / "finalize_results.slurm",
                                        dependency=dependency, required_hours=1)
                        self.assertEqual(scheduler.call_args.args[0],
                            ["sbatch", "--clusters=owl", "--parsable",
                             "--dependency=afterok:" + str(dependency), str(root / "finalize_results.slurm")])
            ledger = json.loads(path.read_text())
            self.assertEqual(sum(job["requested_gpu_hours"] for job in ledger["jobs"].values()), 42)
            self.assertTrue(all(ledger["jobs"][f"cpu_closeout_{index}"]["requested_gpu_hours"] == 0
                                for index in range(3)))

    def test_invalid_dependency_never_dispatches_or_records_a_submission(self):
        invalid = ("", " ", ":123", "123:", "123::456", "123: :456", "123;owl",
                   "123:456;owl", "123,456", "afterok:123", "123\n", [], True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                for dependency in invalid:
                    with self.subTest(dependency=dependency), self.assertRaisesRegex(ValueError, "Invalid scheduler dependency"):
                        pipeline.submit("cpu_closeout", root / "finalize_results.slurm",
                                        dependency=dependency, required_hours=1)
                scheduler.assert_not_called()
            self.assertFalse((root / "job_ledger.json").exists())

    def test_login_submission_is_once_only_and_counts_budget_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = {"jobs": {"prior_training": {"status": "submitted", "job_id": "111",
                                                  "requested_gpu_hours": 12}}, "gpu_hours_cap": 48}
            path = root / "job_ledger.json"
            path.write_text(json.dumps(ledger))
            with patch.dict(os.environ, {}, clear=True), patch.object(pipeline, "ROOT", root), \
                    patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                scheduler.return_value = subprocess.CompletedProcess([], 0, "222;owl\n", "")
                self.assertNotIn("SLURM_JOB_ID", os.environ)
                first = pipeline.submit("original_evaluation", root / "gpu_eval.slurm",
                                        arguments=["original"], gpu_hours=6)
                second = pipeline.submit("original_evaluation", root / "gpu_eval.slurm",
                                         arguments=["original"], gpu_hours=6)
                self.assertEqual((first, second), ("222", "222"))
                scheduler.assert_called_once()
                self.assertEqual(scheduler.call_args.args[0], ["sbatch", "--clusters=owl", "--parsable",
                                                              str(root / "gpu_eval.slurm"), "original"])
            recorded = json.loads(path.read_text())
            self.assertEqual(sum(job["requested_gpu_hours"] for job in recorded["jobs"].values()), 18)
            self.assertEqual(recorded["jobs"]["original_evaluation"]["status"], "submitted")

    def test_login_keeps_cap_deadline_and_uncertain_outcome_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "job_ledger.json"
            with patch.dict(os.environ, {}, clear=True), patch.object(pipeline, "ROOT", root), \
                    patch.object(pipeline, "datetime") as clock, \
                    patch.object(pipeline.subprocess, "run") as scheduler:
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                path.write_text(json.dumps({"jobs": {"prior": {"status": "submitted", "job_id": "111",
                                                               "requested_gpu_hours": 43}}}))
                with self.assertRaisesRegex(RuntimeError, "48-GPU-hour"):
                    pipeline.submit("original_evaluation", root / "gpu_eval.slurm", gpu_hours=6)
                path.write_text(json.dumps({"jobs": {}}))
                clock.now.return_value = DEADLINE - timedelta(hours=5)
                with self.assertRaisesRegex(RuntimeError, "deadline reserve"):
                    pipeline.submit("original_evaluation", root / "gpu_eval.slurm", gpu_hours=6)
                clock.now.return_value = DEADLINE - timedelta(hours=24)
                path.write_text(json.dumps({"jobs": {"original_evaluation": {"status": "uncertain",
                                                                            "requested_gpu_hours": 6}}}))
                with self.assertRaisesRegex(RuntimeError, "reconcile"):
                    pipeline.submit("original_evaluation", root / "gpu_eval.slurm", gpu_hours=6)
                scheduler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
