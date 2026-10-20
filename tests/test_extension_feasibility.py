"""Lightweight synthetic oracles; no real Slurm command, corpus or model access."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extension import cpu_feasibility as audit
from reasoning.data import SYSTEM, alpaca, canonical


def row(question, output=r"A solution. \boxed{2}"):
    return canonical({"instruction": question, "output": output})


def feature(value, total=8192, response=100):
    return {"id": value["id"], "total_tokens": total, "response_tokens": response,
            "cluster": 0, "complete_box": True, "repetition": 0.}


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("Question one"), row("Question two")]
        self.by_id = audit.canonical_map(self.rows)

    def test_canonical_identity_and_uniqueness(self):
        self.assertEqual(set(self.by_id), {item["id"] for item in self.rows})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.canonical_map(self.rows + [self.rows[0]])
        altered = deepcopy(self.rows)
        altered[0]["instruction"] = "A changed prompt"
        with self.assertRaisesRegex(ValueError, "incorrect"):
            audit.canonical_map(altered)

    def test_feature_coverage_rejects_duplicates_missing_and_boolean_counts(self):
        values = [feature(value) for value in self.rows]
        self.assertEqual(len(audit.feature_map(values, self.by_id)), 2)
        for broken in (values[:1], values + [values[0]], [dict(values[0], total_tokens=True), values[1]]):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                audit.feature_map(broken, self.by_id)

    def test_feature_stale_box_and_nonfinite_repetition_rejected(self):
        values = [feature(value) for value in self.rows]
        for changes in ({"complete_box": False}, {"repetition": math.nan}, {"response_tokens": 8192}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.feature_map([dict(values[0], **changes), values[1]], self.by_id)

    def test_exact_unfitted_cluster_sentinel_allowed_without_topic_claim(self):
        values = [feature(value) for value in self.rows]
        values[0]["cluster"] = -1
        values[1]["cluster"] = 64
        result = audit.feature_map(values, self.by_id)
        self.assertEqual(result[self.rows[0]["id"]]["cluster"], -1)
        with self.assertRaises(ValueError):
            audit.feature_map([dict(values[0], cluster=-2), values[1]], self.by_id)

    def test_subset_requires_exact_original_order_and_unique_ids(self):
        identifiers = [item["id"] for item in self.rows]
        exported = alpaca(self.rows)
        self.assertEqual(audit.subset_ids(identifiers, exported, self.by_id), identifiers)
        with self.assertRaises(ValueError):
            audit.subset_ids(identifiers, list(reversed(exported)), self.by_id)
        with self.assertRaises(ValueError):
            audit.subset_ids([identifiers[0]] * 2, exported, self.by_id)
        changed = deepcopy(exported)
        changed[0]["output"] += " rewritten"
        with self.assertRaises(ValueError):
            audit.subset_ids(identifiers, changed, self.by_id)

    def test_original_pairs_bind_first_canonical_occurrence(self):
        source = [{"instruction": item["instruction"], "output": item["output"]} for item in self.rows]
        source += [{"instruction": self.rows[0]["instruction"], "output": "a later conflicting duplicate"}, {}]
        result = audit.verify_originals(source, (self.by_id,))
        self.assertEqual(result, {"raw_source_rows": 4, "valid_unique_canonical_source_prompts": 2,
            "invalid_source_rows": 1, "duplicate_canonical_source_rows": 1,
            "prepared_and_heldout_original_pairs_verified": 2})
        with self.assertRaisesRegex(ValueError, "first canonical"):
            audit.verify_originals(list(reversed(source)), (self.by_id,))
        with self.assertRaisesRegex(ValueError, "absent"):
            audit.verify_originals(source[1:2], (self.by_id,))
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit.verify_originals(source, (self.by_id, self.by_id))


class DiagnosticTests(unittest.TestCase):
    def test_quantile_linear_interpolation_oracle(self):
        result = audit.quantiles([30, 0, 20, 10])
        self.assertEqual(result["median"], 15)
        self.assertAlmostEqual(result["p10"], 3)
        self.assertEqual(result["min"], 0)
        self.assertEqual(result["max"], 30)
        self.assertEqual(audit.quantiles([8])["p99"], 8)
        for values in ([], [1, math.inf], [True]):
            with self.assertRaises(ValueError):
                audit.quantiles(values)

    def test_branch_steps_whole_words_case_insensitive_no_empty_lines(self):
        result = audit.text_diagnostics("Perhaps try this.\n\nAnother plan, alternatively.\nAnotherness is not a branch.")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["branch_steps"], 2)
        self.assertAlmostEqual(result["branch_step_density"], 2 / 3)

    def test_think_order_is_checked_not_just_equal_counts(self):
        valid = audit.text_diagnostics("<think>Reason.\n</think>\nAnswer.")
        self.assertTrue(valid["starts_think_open"])
        self.assertTrue(valid["think_balanced_ordered"])
        for text in ("</think><think>", "<think>unfinished", "only </think>"):
            with self.subTest(text=text):
                self.assertFalse(audit.text_diagnostics(text)["think_balanced_ordered"])
        no_tags = audit.text_diagnostics("ordinary source answer")
        self.assertTrue(no_tags["no_think_tags"])
        self.assertTrue(no_tags["think_balanced_ordered"])

    def test_eight_gram_repetition_known_oracle(self):
        self.assertEqual(audit.text_diagnostics("one two three four five six seven")["repetition"], 0)
        self.assertAlmostEqual(audit.text_diagnostics(" ".join(["x"] * 16))["repetition"], 8 / 9)

    def test_exact_fit_boundaries_full_span_and_prefix_token_oracle(self):
        totals = [8192, 8193, 16384, 16385, 32768, 32769]
        values = [row("Unique question " + str(index)) for index in range(len(totals))]
        identifiers = [item["id"] for item in values]
        fs = {value["id"]: feature(value, total=total, response=100) for value, total in zip(values, totals)}
        ds = {value["id"]: audit.text_diagnostics(value["output"]) for value in values}
        result = audit.summarize_population(identifiers, fs, ds)
        self.assertEqual(result["total_response_span_tokens"], 600)
        self.assertEqual(result["total_serialized_tokens"], sum(totals))
        self.assertEqual(result["exact_cached_token_fit"]["8192"]["complete_serialized_rows"], 1)
        self.assertEqual(result["exact_cached_token_fit"]["16384"]["complete_serialized_rows"], 3)
        self.assertEqual(result["exact_cached_token_fit"]["32768"]["complete_serialized_rows"], 5)
        self.assertEqual(result["exact_cached_token_fit"]["8192"]["supervised_tokens_if_prefix_limited"], 199)
        self.assertEqual(result["exact_cached_token_fit"]["8192"]["rows_fitting_with_32_token_reserve"], 0)
        self.assertEqual(result["exact_cached_token_fit"]["32768"]["supervised_tokens_if_prefix_limited"], 599)

    def test_cache_totals_reconcile_and_do_not_infer_eos(self):
        values = [row("First"), row("Second")]
        ids = [item["id"] for item in values]
        fs = {values[0]["id"]: feature(values[0], 8000, 100),
              values[1]["id"]: feature(values[1], 8250, 100)}
        meta = {"rows": 2, "expected_rows": 2, "unique_rows_dropped": 0, "target_rewriting": False,
                "cutoff": 8192, "truncated_ids": ids[1:], "truncated_rows": 1,
                "original_serialized_tokens": 16250, "supervised_training_tokens": 142,
                "truncation_policy": "prefix_truncate"}
        result = audit.validate_cache_meta(meta, ids, fs)
        self.assertEqual(result["supervised_training_tokens"], 142)
        self.assertIn("No EOS ID is inferred", result["eos_note"])
        for change in ({"truncated_rows": 0}, {"truncated_ids": []}, {"supervised_training_tokens": 200},
                       {"expected_rows": 3}, {"target_rewriting": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_cache_meta(dict(meta, **change), ids, fs)


class GuardTests(unittest.TestCase):
    TEXT = ("JobId=123 UserId=tester(456) JobState=RUNNING Partition=normal_q "
            "NumNodes=1 NumCPUs=8 NodeList=owl-sr003 "
            "ReqTRES=cpu=8,mem=64G,node=1 AllocTRES=cpu=8,mem=64G,node=1 Gres=(null)")

    def test_running_cpu_allocation_positive_proof(self):
        result = audit.validate_allocation(self.TEXT, "123", "owl-sr003", 456)
        self.assertEqual(result["allocated_cpus"], 8)
        self.assertFalse(result["gpu_requested_or_allocated"])

    def test_allocation_gpu_requests_and_incomplete_proof_fail_closed(self):
        invalid = (self.TEXT + " TresPerNode=gres/gpu:b200:1",
                   self.TEXT.replace("ReqTRES=cpu=8", "ReqTRES=gres/gpu=0,cpu=8"),
                   self.TEXT.replace("JobState=RUNNING", "JobState=PENDING"),
                   self.TEXT.replace("AllocTRES=cpu=8,mem=64G,node=1", "AllocTRES=(null)"),
                   self.TEXT.replace("NodeList=owl-sr003", "NodeList=owl-sr[003-004]"),
                   self.TEXT.replace("NumNodes=1", "NumNodes=2"))
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.validate_allocation(text, "123", "owl-sr003", 456)
        with self.assertRaises(ValueError):
            audit.validate_allocation(self.TEXT, "123", "owl-sr003", 999)

    def test_deadline_is_extended_october_27_eastern(self):
        self.assertEqual(audit.DEADLINE, datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc))
        audit.deadline_guard(1800, datetime(2026, 10, 28, 3, 29, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            audit.deadline_guard(1800, datetime(2026, 10, 28, 3, 30, tzinfo=timezone.utc))

    def test_sha_evidence_detects_modified_input_and_bad_expected_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"count":1}')
            files = audit.Evidence()
            checksum = files.check(path)
            self.assertEqual(files.read(path, checksum), {"count": 1})
            with self.assertRaises(ValueError):
                files.check(path, "not-a-sha")
            path.write_text('{"count":22}')
            with self.assertRaisesRegex(ValueError, "changed"):
                files.unchanged()

    def test_duplicate_json_and_nonfinite_values_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    audit.read_json(path)

    def test_exclusive_output_and_existing_complete_refused_before_data_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            output = repo / "extension/runs/feasibility-20261020-123"
            output.mkdir(parents=True)
            (output / "native_qa.json").write_text("{}")
            audit.write_new_json(output / "complete.json", {"complete": True})
            with self.assertRaises(FileExistsError):
                audit.write_new_json(output / "complete.json", {"complete": False})
            with patch.object(audit, "REPOSITORY", repo), patch.object(audit, "require_cpu_allocation", return_value={"job_id": "123"}):
                with self.assertRaisesRegex(ValueError, "Output must contain only"):
                    audit.audit(repo / "no-artifacts-read", output, output / "native_qa.json")

    def test_native_qa_bound_to_code_and_cpu_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            for name in audit.CODE_FILES:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source " + name)
            output = repo / "extension/runs/feasibility-20261020-123"
            output.mkdir(parents=True)
            path = output / "native_qa.json"
            with patch.object(audit, "REPOSITORY", repo):
                qa = {"schema": "extension-feasibility-native-qa-v1", "success": True,
                    "tests_run": 20, "failures": 0, "errors": 0, "skipped": 0,
                    "code_sha256": audit.code_hashes(), "job_id": "123", "node": "owl-sr003", "gpu_disabled": True}
                compute = {"job_id": "123", "node": "owl-sr003"}
                path.write_text(json.dumps(qa))
                self.assertEqual(audit.validate_qa(path, output, compute, audit.Evidence())["tests_run"], 20)
                for changes in ({"skipped": 1}, {"node": "login"}, {"code_sha256": {}}, {"tests_run": 0}):
                    path.write_text(json.dumps(dict(qa, **changes)))
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        audit.validate_qa(path, output, compute, audit.Evidence())


if __name__ == "__main__":
    unittest.main()
