"""Immutable benchmark readiness and early-base-evaluation admission checks."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from deadline import bootstrap_evaluation as module
from deadline.audit_results import OFFICIAL
from reasoning.data import digest


class BenchmarkBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.originals = self.root / "originals"
        self.artifacts = self.root / "artifacts"
        for name, count in OFFICIAL.items():
            path = self.originals / name / "test.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps({"question": f"{name} original question {index}",
                                                "answer": "2"}) + "\n" for index in range(count)))

    def downloader(self, command, **kwargs):
        self.assertEqual(command[1:3], ["-m", "reasoning.download_aime25"])
        self.assertEqual(command[command.index("--revision") + 1], module.AIME25_REVISION)
        destination = Path(command[command.index("--output") + 1])
        hashes = {}
        for name in OFFICIAL:
            source = self.originals / name / "test.jsonl"
            target = destination / name / "test.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[name] = digest(source)
        target = destination / "aime25/test.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps({"id": str(index),
                      "question": f"Pinned supplementary question {index}", "answer": str(index)})
                                + "\n" for index in range(30)))
        (destination / "supplement.json").write_text(json.dumps({"repo": "math-ai/aime25",
            "revision": module.AIME25_REVISION, "problems": 30, "source_sha256": "a" * 64,
            "canonical_sha256": digest(target), "supplied_benchmark_hashes": hashes}))
        return types.SimpleNamespace(returncode=0)

    def prepare(self):
        with patch.object(module.subprocess, "run", side_effect=self.downloader) as downloader:
            result = module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
            self.assertEqual(downloader.call_count, 1)
            return result

    def test_verified_publication_is_reused_without_download_or_rewrite(self):
        receipt = self.prepare()
        self.assertTrue(receipt["complete"])
        self.assertEqual(len(receipt["evidence"]["benchmark_hashes"]), 9)
        paths = list((self.artifacts / "benchmarks").glob("*/test.jsonl"))
        before = {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths}
        with patch.object(module.subprocess, "run", side_effect=AssertionError("must not redownload")):
            second = module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
        self.assertEqual(second, receipt)
        self.assertEqual(before, {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths})

    def test_changed_benchmark_or_readiness_seal_fails_closed(self):
        self.prepare()
        target = self.artifacts / "benchmarks/math/test.jsonl"
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
        target.write_bytes(original)
        ready = self.artifacts / module.READINESS_NAME
        receipt = json.loads(ready.read_text())
        receipt["evidence"]["benchmark_counts"]["math"] = 499
        ready.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)

    def test_existing_partial_directory_is_never_overwritten(self):
        target = self.artifacts / "benchmarks/math/test.jsonl"
        target.parent.mkdir(parents=True)
        target.write_text("preserve partial forensic bytes")
        with patch.object(module.subprocess, "run", side_effect=AssertionError("must not redownload")):
            with self.assertRaisesRegex(ValueError, "complete nine"):
                module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
        self.assertEqual(target.read_text(), "preserve partial forensic bytes")
        self.assertFalse((self.artifacts / module.READINESS_NAME).exists())

    def test_first_download_is_audited_before_any_publication(self):
        def wrong_revision(*args, **kwargs):
            result = self.downloader(*args, **kwargs)
            destination = Path(args[0][args[0].index("--output") + 1])
            path = destination / "supplement.json"
            receipt = json.loads(path.read_text())
            receipt["revision"] = "wrong"
            path.write_text(json.dumps(receipt))
            return result
        with patch.object(module.subprocess, "run", side_effect=wrong_revision):
            with self.assertRaisesRegex(ValueError, "exact pinned"):
                module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
        self.assertFalse((self.artifacts / "benchmarks").exists())
        self.assertFalse((self.artifacts / module.READINESS_NAME).exists())

    def test_complete_existing_suite_is_adopted_without_rewriting(self):
        self.prepare()
        (self.artifacts / module.READINESS_NAME).unlink()
        target = self.artifacts / "benchmarks/aime25/test.jsonl"
        before = target.stat().st_mtime_ns
        with patch.object(module.subprocess, "run", side_effect=AssertionError("must not redownload")):
            module.ensure_benchmarks(self.artifacts, base_benchmarks=self.originals)
        self.assertEqual(before, target.stat().st_mtime_ns)

    def test_bootstrap_admits_only_the_ledger_original_role_after_preflight(self):
        receipt = self.prepare()
        events = []
        def checks():
            events.append("preflight")
            return {"tests_run": 120, "skipped": 0, "grader": "passed"}
        def ready(root):
            events.append("readiness")
            return receipt
        def submit(role, script, **kwargs):
            events.append("submit")
            self.assertEqual(role, "original_evaluation")
            self.assertEqual(str(script), "deadline/gpu_eval.slurm")
            self.assertEqual(kwargs, {"arguments": ["original"], "gpu_hours": 6})
            return "12345"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "99999"}), \
                patch.object(module, "run_native_preflight", side_effect=checks), \
                patch.object(module, "ensure_benchmarks", side_effect=ready), \
                patch("deadline.pipeline.submit", side_effect=submit):
            output = module.bootstrap(self.artifacts)
        self.assertEqual(events, ["preflight", "readiness", "submit"])
        self.assertEqual(output["job_id"], "12345")

    def test_missing_compute_allocation_cannot_submit(self):
        with patch.dict(os.environ, {}, clear=True), patch("deadline.pipeline.submit") as submit:
            with self.assertRaisesRegex(RuntimeError, "CPU compute"):
                module.bootstrap(self.artifacts)
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
