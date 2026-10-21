"""Synthetic retry oracles; no data/model execution, network or submission."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extension.cpu_feasibility import Evidence, digest, value_digest, write_new_json
from extension.labeled_feasibility import CRITERION, MATH_REVISION, ExactIndex, group_and_filter, source_record
from extension import development_seal as original
from extension import development_seal_v2 as retry
from extension.development_retry_submission import SOURCE_PATHS


def native_record(ordinal=0, level="Level 1", subject="algebra", question="A native question."):
    source = {"repository": "EleutherAI/hendrycks_math", "revision": MATH_REVISION,
        "file": subject + "/train/0000.parquet", "row_ordinal": ordinal}
    raw = {"problem": question, "level": level, "type": "Algebra", "solution": "Proof. \\boxed{1}"}
    row, quarantine = source_record("math", raw, source)
    assert quarantine is None
    return row


def plan():
    return {"schema": retry.PLAN_SCHEMA, "cpu_only": True, "model_inference": False,
        "choose_splits": True, "automatic_downstream": False, "training": False,
        "selection_uses_labels_or_model_outcomes": False, "historical_inputs_read_only": True,
        "seed": 20261021, "split_order": list(retry.SPLITS), "targets": copy.deepcopy(retry.TARGETS),
        "hash_template": retry.HASH_TEMPLATE, "near_duplicate": CRITERION,
        "deadline_utc": "2026-10-28T03:59:00+00:00", "wall_time_seconds": 7200, "reserve_seconds": 172800,
        "labeled_audit": {"directory": retry.LABELED_DIRECTORY, "complete_sha256": "a" * 64, "qa_sha256": "b" * 64},
        "pool": {"path": retry.POOL_PATH, "sha256": retry.POOL_SHA256},
        "level_diagnostic": copy.deepcopy(retry.DIAGNOSTIC), "math_unknown_levels": {"Level ?": 1}}


def diagnostic():
    return {"schema": "extension-development-level-diagnostic-v1", "complete": True,
        "compute": {"job_id": "1223713", "gpu_requested_or_allocated": False, "state": "RUNNING", "node": "owl001"},
        "input_sha256": "f1304725d60087666fa2c3995fe7b4b344067ad28d1f2af35e8dbc5dab4e0551",
        "rows": 7132, "math_rows": 6013, "non_1_to_5_levels": {"Level ?": 1}, "gpu_seconds": 0,
        "no_labels_or_solutions_inspected": True, "no_split_assignment_or_model_outcomes": True,
        "subject_level_counts": [{"subject": "algebra", "official_level": "Level 1", "rows": 6012},
            {"subject": "algebra", "official_level": "Level ?", "rows": 1}]}


def actual_counts_fixture(records):
    math = [row for row in records if row["source_kind"] == "math"]
    counts = Counter((row["source"]["file"].split("/")[0], row["original_record"]["level"]) for row in math)
    return {"rows": len(records), "math_rows": len(math), "non_1_to_5_levels": {"Level ?": 1},
        "subject_level_counts": [{"subject": subject, "official_level": level, "rows": count}
            for (subject, level), count in sorted(counts.items())]}


class UnknownLevelTests(unittest.TestCase):
    def test_only_observed_unknown_maps_to_explicit_stratum_without_mutation(self):
        row = native_record(level="Level ?")
        before = copy.deepcopy(row)
        self.assertEqual(retry.row_stratum(row), 'math:["algebra","unknown:Level ?"]')
        self.assertEqual(row, before)
        self.assertEqual(row["original_record"]["level"], "Level ?")

    def test_every_numeric_level_retains_v1_stratum_exactly(self):
        for level in range(1, 6):
            row = native_record(level="Level " + str(level))
            self.assertEqual(retry.row_stratum(row), original.row_stratum(row))

    def test_any_other_unknown_still_fails(self):
        for value in ("", "Level 0", "Level 6", "Level ??", "unknown", "level ?", "Level ? "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                retry.row_stratum(native_record(level=value))

    def test_gold_answer_never_affects_stratum(self):
        row = native_record(level="Level ?")
        changed = copy.deepcopy(row)
        changed["original_record"]["solution"] = "Different solution. \\boxed{99}"
        changed["label"]["answer"] = "99"
        self.assertEqual(retry.row_stratum(row), retry.row_stratum(changed))

    def test_mixed_native_group_keeps_every_member_and_raw_level(self):
        rows = [native_record(0, "Level ?"), native_record(1, "Level 1")]
        surviving, _, groups = group_and_filter(rows, ExactIndex())
        by_id, eligible, _ = retry.validate_population(surviving, groups)
        self.assertEqual(len(by_id), 2)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["stratum"], "mixed")
        self.assertEqual(eligible[0]["rows"], 2)
        self.assertEqual(sorted(row["original_record"]["level"] for row in by_id.values()), ["Level 1", "Level ?"])


class DiagnosticTests(unittest.TestCase):
    def test_valid_pinned_metadata_only_diagnostic_schema(self):
        value = diagnostic()
        self.assertIs(retry.validate_diagnostic_receipt(value), value)

    def test_wrong_job_gpu_input_or_outcome_flags_rejected(self):
        mutations = [("input_sha256", "0" * 64), ("gpu_seconds", 1),
            ("no_split_assignment_or_model_outcomes", False), ("no_labels_or_solutions_inspected", False),
            ("non_1_to_5_levels", {"Level ?": 2}), ("math_rows", 6012)]
        for key, value in mutations:
            changed = diagnostic()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                retry.validate_diagnostic_receipt(changed)
        changed = diagnostic()
        changed["compute"]["job_id"] = "other"
        with self.assertRaises(ValueError):
            retry.validate_diagnostic_receipt(changed)

    def test_subject_level_totals_and_unique_entries_required(self):
        for alteration in ("total", "duplicate", "unknown"):
            changed = diagnostic()
            if alteration == "total":
                changed["subject_level_counts"][0]["rows"] -= 1
            elif alteration == "duplicate":
                changed["subject_level_counts"].append(copy.deepcopy(changed["subject_level_counts"][0]))
            else:
                changed["subject_level_counts"][0]["official_level"] = "Level !"
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                retry.validate_diagnostic_receipt(changed)

    def test_actual_raw_counts_match_without_rewriting(self):
        rows = [native_record(0, "Level ?"), native_record(1, "Level 1")]
        before = copy.deepcopy(rows)
        result = retry.validate_level_counts(rows, actual_counts_fixture(rows), plan())
        self.assertEqual(result["unknown_levels"], {"Level ?": 1})
        self.assertTrue(result["raw_records_unchanged"])
        self.assertEqual(rows, before)

    def test_missing_or_second_unknown_is_not_silently_quarantined(self):
        base = [native_record(0, "Level ?"), native_record(1, "Level 1")]
        fixed = actual_counts_fixture(base)
        for rows in ([native_record(0, "Level 1"), native_record(1, "Level 1")],
                base + [native_record(2, "Level ?")]):
            with self.assertRaisesRegex(ValueError, "Actual native levels"):
                retry.validate_level_counts(rows, fixed, plan())

    def test_all_subject_counts_are_recomputed_not_only_unknown_total(self):
        rows = [native_record(0, "Level ?"), native_record(1, "Level 1")]
        fixed = actual_counts_fixture(rows)
        changed = copy.deepcopy(rows)
        changed[1]["original_record"]["level"] = "Level 2"
        with self.assertRaises(ValueError):
            retry.validate_level_counts(changed, fixed, plan())


class VersionedContractTests(unittest.TestCase):
    def test_v2_plan_keeps_all_original_scientific_constants(self):
        self.assertEqual(retry.validate_plan(plan()), plan())
        for name in ("TARGETS", "SPLITS", "DOMAINS", "HASH_TEMPLATE", "LABELED_DIRECTORY", "POOL_PATH", "POOL_SHA256"):
            self.assertEqual(getattr(retry, name), getattr(original, name))
        self.assertNotEqual(retry.PLAN_SCHEMA, original.PLAN_SCHEMA)
        self.assertNotEqual(retry.QA_SCHEMA, original.QA_SCHEMA)
        self.assertNotEqual(retry.SCHEMA, original.SCHEMA)

    def test_wrong_diagnostic_unknown_seed_or_targets_rejected(self):
        for alteration in ("diagnostic", "unknown", "seed", "target"):
            changed = plan()
            if alteration == "diagnostic":
                changed["level_diagnostic"]["complete_sha256"] = "0" * 64
            elif alteration == "unknown":
                changed["math_unknown_levels"] = {"Level ?": 2}
            elif alteration == "seed":
                changed["seed"] += 1
            else:
                changed["targets"]["development"]["math"] -= 1
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                retry.validate_plan(changed)

    def test_numerical_allocation_and_hash_order_unchanged(self):
        groups = []
        for domain in retry.DOMAINS:
            for index in range(16):
                identifier = hashlib.sha256((domain + str(index)).encode()).hexdigest()
                group_id = value_digest({"criterion": CRITERION, "member_ids": [identifier]})
                groups.append({"group_id": group_id, "member_ids": [identifier], "rows": 1,
                    "domain": domain, "stratum": domain + ":fixture"})
        targets = {split: {domain: value for domain in retry.DOMAINS}
            for split, value in zip(retry.SPLITS, (2, 3, 4))}
        self.assertEqual(retry.allocate_groups(groups, targets), original.allocate_groups(groups, targets))
        self.assertEqual(retry.reachable_sums(Counter({1: 4, 3: 2}), 11),
            original.reachable_sums(Counter({1: 4, 3: 2}), 11))

    def qa_fixture(self, root):
        output = root / "extension/runs/development-seal-v2-20261021-999"
        output.mkdir(parents=True)
        inventory = {}
        for name in SOURCE_PATHS:
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture\n", encoding="utf-8")
            inventory[name] = {"sha256": digest(file), "bytes": file.stat().st_size}
        compute = {"job_id": "999", "node": "owl001", "gpu_requested_or_allocated": False}
        counts = {"test_extension_*.py": 64, "test_labeled_*.py": 44,
            "test_preserving_*.py": 36, "test_development_*.py": 48,
            "test_preservation_development_seal_v2.py": 17, "test_preservation_retry_submission.py": 21}
        qa = {"schema": retry.QA_SCHEMA, "success": True, "tests_run": sum(counts.values()),
            "test_counts": counts, "failures": 0, "errors": 0, "skipped": 0,
            "expected_failures": 0, "unexpected_successes": 0, "gpu_disabled": True,
            "wrapper_syntax_passed": True, "job_id": "999", "node": "owl001", "compute_proof": compute,
            "source_inventory": inventory, "code_sha256": {name: row["sha256"] for name, row in inventory.items()}}
        return output, compute, qa

    def test_exact_38_source_and_six_suite_native_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, compute, qa = self.qa_fixture(root)
            self.assertEqual(len(qa["code_sha256"]), 38)
            write_new_json(output / "native_qa.json", qa)
            result = retry.validate_native_qa(output / "native_qa.json", root, output, compute,
                Evidence(), root / "extension/development_retry_plan.json")
            self.assertEqual(result["tests_run"], 230)

    def test_missing_source_or_broad_suite_or_old_qa_rejected(self):
        for alteration in ("missing", "broad", "old"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output, compute, qa = self.qa_fixture(root)
                if alteration == "missing":
                    name = "extension/development_seal.py"
                    del qa["code_sha256"][name]
                    del qa["source_inventory"][name]
                elif alteration == "broad":
                    qa["test_counts"]["test_preservation_*.py"] = qa["test_counts"].pop("test_preservation_development_seal_v2.py")
                else:
                    qa["schema"] = original.QA_SCHEMA
                write_new_json(output / "native_qa.json", qa)
                with self.assertRaises(ValueError):
                    retry.validate_native_qa(output / "native_qa.json", root, output, compute,
                        Evidence(), root / "extension/development_retry_plan.json")

    def test_diagnostic_hash_failure_cannot_allocate_or_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "extension/runs/development-seal-v2-20261021-999"
            output.mkdir(parents=True)
            write_new_json(output / "native_qa.json", {})
            plan_path = root / "extension/development_retry_plan.json"
            write_new_json(plan_path, plan())
            diagnostic_path = root / retry.DIAGNOSTIC["directory"] / "complete.json"
            diagnostic_path.parent.mkdir(parents=True)
            write_new_json(diagnostic_path, diagnostic())
            compute = {"job_id": "999", "node": "owl001"}
            with patch.object(retry, "require_cpu_allocation", return_value=compute), \
                 patch.object(retry, "deadline_guard", return_value={}), \
                 patch.object(retry, "validate_native_qa", return_value={}), \
                 patch.object(retry, "allocate_groups") as allocation:
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    retry.seal(root, "extension/development_retry_plan.json", output, output / "native_qa.json")
            allocation.assert_not_called()
            self.assertFalse((output / "complete.json").exists())
            self.assertFalse((output / "calibration_ids.json").exists())


if __name__ == "__main__":
    unittest.main()
