"""Synthetic CPU-only oracles; no downloads, model inference or native audit."""
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extension.cpu_feasibility import Evidence, digest, value_digest, write_new_json
from extension.labeled_feasibility import (CRITERION, MATH_REVISION, ARC_REVISION,
    ExactIndex, group_and_filter, near_duplicate, source_record)
from extension import development_seal as seal
from reasoning.data import prompt_id


def record(kind, ordinal, question=None, subject="algebra", level="Level 1", answer="1", labels=None):
    if kind == "math":
        raw = {"problem": question or "Find the value of variable " + str(ordinal),
            "level": level, "type": "Algebra", "solution": "A solution. \\boxed{" + answer + "}"}
        source = {"repository": "EleutherAI/hendrycks_math", "revision": MATH_REVISION,
            "file": subject + "/train/0000.parquet", "row_ordinal": ordinal}
    else:
        labels = labels or ["A", "B", "C", "D"]
        raw = {"id": "arc-" + str(ordinal), "question": question or "What happens to object " + str(ordinal) + "?",
            "choices": {"label": labels, "text": ["Choice " + str(index) for index in range(len(labels))]}, "answerKey": answer if answer in labels else labels[0]}
        source = {"repository": "allenai/ai2_arc", "revision": ARC_REVISION,
            "file": "ARC-Challenge/train-00000-of-00001.parquet", "row_ordinal": ordinal}
    result, unavailable = source_record(kind, raw, source)
    assert unavailable is None
    return result


def component(members, domain="math", stratum="math:a"):
    members = sorted(members)
    identifier = value_digest({"criterion": CRITERION, "member_ids": members})
    return {"group_id": identifier, "member_ids": members, "rows": len(members), "domain": domain, "stratum": stratum}


def synthetic_groups(sizes, domain="math", stratum="math:a"):
    output, ordinal = [], 0
    for size in sizes:
        members = [hashlib.sha256((domain + ":" + stratum + ":" + str(index)).encode()).hexdigest()
            for index in range(ordinal, ordinal + size)]
        ordinal += size
        output.append(component(members, domain, stratum))
    return output


def tiny_targets(first=2, second=3, third=4):
    return {name: {domain: count for domain in seal.DOMAINS}
        for name, count in zip(seal.SPLITS, (first, second, third))}


class SubsetSumTests(unittest.TestCase):
    def test_bitset_matches_exhaustive_subsets_and_multiplicities(self):
        for sizes in ([], [1], [2, 2, 2], [1, 3, 3, 5], [2, 4, 6], [1] * 9):
            oracle = {sum(sizes[index] for index in range(len(sizes)) if mask & (1 << index))
                for mask in range(1 << len(sizes))}
            for limit in (0, 4, 11, 30):
                bits = seal.reachable_sums(Counter(sizes), limit)
                self.assertEqual({index for index in range(limit + 1) if bits & (1 << index)},
                    {index for index in oracle if index <= limit})

    def test_invalid_multiplicity_rejected(self):
        for counts in ({0: 1}, {2: -1}, {True: 2}, {2: 1.0}):
            with self.assertRaises(ValueError):
                seal.reachable_sums(counts, 10)

    def test_acceptance_guard_rejects_current_dead_end(self):
        # Taking size3 leaves one unreachable row; taking size2 permits size2.
        self.assertFalse(seal.acceptance_feasible(Counter({3: 1, 2: 2}), 3, 1, []))
        self.assertTrue(seal.acceptance_feasible(Counter({3: 1, 2: 2}), 2, 2, []))

    def test_future_individual_aggregate_and_capacity_guards(self):
        self.assertFalse(seal.acceptance_feasible(Counter({1: 1, 2: 4}), 1, 0, [3, 3]))
        # Remaining sizes2,3 have enough capacity and each future2 is reachable,
        # but their aggregate4 is not reachable.
        self.assertFalse(seal.acceptance_feasible(Counter({1: 1, 2: 1, 3: 1}), 1, 0, [2, 2]))
        self.assertFalse(seal.acceptance_feasible(Counter({2: 3}), 2, 0, [2, 2, 2]))
        self.assertTrue(seal.acceptance_feasible(Counter({1: 12}), 1, 1, [3, 4]))


class AllocationTests(unittest.TestCase):
    def groups(self):
        return synthetic_groups([1] * 20, "math", "a") + synthetic_groups([1] * 20, "arc_challenge", "b")

    def test_exact_targets_whole_groups_and_disjointness(self):
        groups = self.groups()
        assigned, unused, stats = seal.allocate_groups(groups, tiny_targets())
        selected = []
        for split in seal.SPLITS:
            self.assertEqual(Counter({domain: sum(group["rows"] for group in assigned[split]
                if group["domain"] == domain) for domain in seal.DOMAINS}), Counter(tiny_targets()[split]))
            selected.extend(group["group_id"] for group in assigned[split])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected) + len(unused), len(groups))
        self.assertEqual(stats["splits"]["development"]["math"]["achieved_rows"], 3)

    def test_input_order_invariance_and_declared_hash_template(self):
        groups = self.groups()
        a = seal.allocate_groups(groups, tiny_targets())
        b = seal.allocate_groups(list(reversed(groups)), tiny_targets())
        self.assertEqual(a, b)
        identifier = groups[0]["group_id"]
        expected = hashlib.sha256(("20261021:development-seal-v1:math:calibration:" + identifier).encode()).hexdigest()
        self.assertEqual(seal.group_order("math", "calibration", identifier), expected)

    def test_proportional_stratum_deficit_is_integer_and_source_only(self):
        groups = synthetic_groups([1] * 30, "math", "a") + synthetic_groups([1] * 10, "math", "b")
        groups += synthetic_groups([1] * 40, "arc_challenge", "arc")
        assigned, _, stats = seal.allocate_groups(groups, tiny_targets(4, 4, 4))
        for split in seal.SPLITS:
            counts = Counter(group["stratum"] for group in assigned[split] if group["domain"] == "math")
            self.assertEqual(counts, {"a": 3, "b": 1})
            self.assertEqual(stats["splits"][split]["math"]["desired_stratum_rows"]["b"],
                {"numerator": 40, "denominator": 40})

    def test_overshoot_reachability_prevents_stranding(self):
        groups = synthetic_groups([3, 2, 2, 1, 1, 1, 1], "math", "a")
        groups += synthetic_groups([1] * 11, "arc_challenge", "b")
        assigned, _, _ = seal.allocate_groups(groups, tiny_targets(4, 3, 3))
        for split, target in zip(seal.SPLITS, (4, 3, 3)):
            self.assertEqual(sum(group["rows"] for group in assigned[split] if group["domain"] == "math"), target)

    def test_cross_domain_groups_unused_even_if_capacity_needed(self):
        groups = self.groups() + [component(["f" * 64], "cross_domain", "mixed")]
        assigned, unused, stats = seal.allocate_groups(groups, tiny_targets())
        self.assertTrue(any(row["unused_reason"] == "cross_domain_component" for row in unused))
        self.assertEqual(stats["cross_domain_unused_rows"], 1)
        self.assertTrue(all(group["domain"] != "cross_domain" for rows in assigned.values() for group in rows))

    def test_impossible_exact_rows_fail_without_reducing_targets(self):
        groups = synthetic_groups([2] * 8, "math") + synthetic_groups([1] * 16, "arc_challenge")
        with self.assertRaisesRegex(ValueError, "not a global multi-split impossibility"):
            seal.allocate_groups(groups, tiny_targets(3, 3, 3))

    def test_insufficient_population_fails(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            seal.allocate_groups(synthetic_groups([1], "math"), tiny_targets())


class PopulationAndStrataTests(unittest.TestCase):
    def test_native_record_components_reconcile_and_answers_never_stratify(self):
        first = record("math", 0, question="A distinct algebra problem.", answer="1")
        second = record("math", 1, question="A completely different geometry problem.", answer="999")
        records, _, groups = group_and_filter([first, second], ExactIndex())
        by_id, eligible, excluded = seal.validate_population(records, groups)
        self.assertEqual(len(by_id), 2)
        self.assertEqual(excluded, 0)
        self.assertTrue(all(group["stratum"] == seal.row_stratum(first) for group in eligible))
        changed = copy.deepcopy(first)
        changed["original_record"]["solution"] = "An unrelated solution. \\boxed{other}"
        changed["label"] = {"answer": "arbitrary"}
        self.assertEqual(seal.row_stratum(first), seal.row_stratum(changed))

    def test_arc_word_bins_and_choice_count(self):
        for words, bucket in ((15, "<=15"), (16, "<=30"), (30, "<=30"), (31, "<=60"), (60, "<=60"), (61, ">60")):
            row = record("arc_challenge", 0, question=" ".join(["word"] * words), labels=["1", "2", "3"])
            self.assertEqual(seal.row_stratum(row), "arc_challenge:" + json.dumps([bucket, 3], separators=(",", ":")))

    def test_nonofficial_math_level_rejected(self):
        for level in ("", "Level 0", "Level 6", "level 1", "unknown"):
            with self.assertRaisesRegex(ValueError, "official level"):
                seal.row_stratum(record("math", 0, level=level))

    def test_mixed_strata_group_whole_and_cross_domain_unused(self):
        q = "The same exact question used to test transitive components."
        rows = [record("math", 0, question=q, level="Level 1"), record("math", 1, question=q, level="Level 2")]
        surviving, _, groups = group_and_filter(rows, ExactIndex())
        _, eligible, _ = seal.validate_population(surviving, groups)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["stratum"], "mixed")
        rows.append(record("arc_challenge", 0, question=q))
        surviving, _, groups = group_and_filter(rows, ExactIndex())
        _, eligible, _ = seal.validate_population(surviving, groups)
        self.assertEqual(eligible[0]["domain"], "cross_domain")

    def test_label_or_question_tampering_and_missing_member_fail(self):
        rows, _, groups = group_and_filter([record("math", 0)], ExactIndex())
        for field, value in (("question", "altered"), ("label_sha256", "0" * 64), ("group_id", "f" * 64)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                seal.validate_population(changed, groups)
        with self.assertRaises(ValueError):
            seal.validate_population(rows, groups + groups)


class ReservationTests(unittest.TestCase):
    def test_prior_category_intersections_fail_and_overlapping_prior_categories_are_allowed(self):
        first, second = "a" * 64, "b" * 64
        self.assertEqual(seal.assert_no_prior_overlap([], {"random": [first], "advanced": [first, second]}),
            {"advanced": {"prior_ids": 2, "reserved_overlap": 0}, "random": {"prior_ids": 1, "reserved_overlap": 0}})
        with self.assertRaisesRegex(ValueError, "prior excluded"):
            seal.assert_no_prior_overlap([first], {"random": [first]})

    def test_exact_and_lexical_pool_reservation_matches_bruteforce(self):
        chosen = [record("math", 0, question="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"),
            record("arc_challenge", 0, question="Which material conducts electricity?")]
        texts = [chosen[0]["question"].upper(), chosen[0]["question"] + " nu",
            chosen[1]["prompt"], "unrelated question", "alpha + 2"]
        questions = {prompt_id(text): text for text in texts}
        identifiers, matches = seal.reserve_pool(questions, chosen)
        expected = sorted(identifier for identifier, text in questions.items() if any(near_duplicate(text, variant)
            for row in chosen for variant in row["lexical_screen_variants"]))
        self.assertEqual(identifiers, expected)
        self.assertTrue(any("near" in row["match_kinds"] for row in matches))
        self.assertTrue(any("exact" in row["match_kinds"] for row in matches))

    def test_answers_and_solution_text_are_not_reserved_variants(self):
        row = record("math", 0, question="Find the unknown integer.", answer="secret-token-answer")
        questions = {prompt_id("secret-token-answer"): "secret-token-answer", prompt_id("A solution."): "A solution."}
        self.assertEqual(seal.reserve_pool(questions, [row]), ([], []))


class GuardsTests(unittest.TestCase):
    def plan(self):
        return {"schema": seal.PLAN_SCHEMA, "cpu_only": True, "model_inference": False,
            "choose_splits": True, "automatic_downstream": False, "training": False,
            "selection_uses_labels_or_model_outcomes": False, "historical_inputs_read_only": True, "seed": 20261021,
            "split_order": list(seal.SPLITS), "targets": copy.deepcopy(seal.TARGETS),
            "hash_template": seal.HASH_TEMPLATE, "near_duplicate": CRITERION,
            "deadline_utc": "2026-10-28T03:59:00+00:00", "wall_time_seconds": 3600, "reserve_seconds": 172800,
            "labeled_audit": {"directory": seal.LABELED_DIRECTORY, "complete_sha256": "a" * 64, "qa_sha256": "b" * 64},
            "pool": {"path": seal.POOL_PATH, "sha256": seal.POOL_SHA256, "expected_rows": 83029}}

    def test_plan_forbids_target_seed_and_source_changes(self):
        self.assertEqual(seal.validate_plan(self.plan()), self.plan())
        for name, value in (("seed", 7), ("model_inference", True), ("hash_template", "different"),
                ("split_order", list(reversed(seal.SPLITS)))):
            plan = self.plan()
            plan[name] = value
            with self.assertRaises(ValueError):
                seal.validate_plan(plan)
        plan = self.plan()
        plan["targets"]["calibration"]["math"] -= 1
        with self.assertRaises(ValueError):
            seal.validate_plan(plan)

    def test_wrong_allocation_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "extension/runs/development-seal-999"
            output.mkdir(parents=True)
            write_new_json(output / "native_qa.json", {})
            with patch.object(seal, "require_cpu_allocation", side_effect=ValueError("not compute")):
                with self.assertRaisesRegex(ValueError, "not compute"):
                    seal.seal(root, "extension/development_plan.json", output, output / "native_qa.json")
            self.assertEqual({path.name for path in output.iterdir()}, {"native_qa.json"})
            self.assertFalse((output / "complete.json").exists())

    def test_existing_output_refused_before_data_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "extension/runs/development-seal-999"
            output.mkdir(parents=True)
            write_new_json(output / "native_qa.json", {})
            write_new_json(output / "complete.json", {})
            with patch.object(seal, "require_cpu_allocation", return_value={"job_id": "999", "node": "owl001"}):
                with self.assertRaisesRegex(ValueError, "exclusive"):
                    seal.seal(root, "extension/development_plan.json", output, output / "native_qa.json")

    def test_prior_absolute_receipt_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                seal._project_receipt_path(root, "/elsewhere/source.json")

    def qa_fixture(self, root):
        output = root / "extension/runs/development-seal-999"
        output.mkdir(parents=True)
        names = ("extension/development_seal.py", "tests/test_development_seal.py",
            "extension/development_seal.slurm", "extension/labeled_feasibility.py",
            "extension/cpu_feasibility.py", "reasoning/data.py", "extension/development_plan.json")
        inventory = {}
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source\n", encoding="utf-8")
            inventory[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        compute = {"job_id": "999", "node": "owl001", "gpu_requested_or_allocated": False}
        qa = {"schema": seal.QA_SCHEMA, "success": True, "tests_run": 167,
            "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
            "gpu_disabled": True, "wrapper_syntax_passed": True, "job_id": "999", "node": "owl001",
            "compute_proof": compute, "source_inventory": inventory,
            "code_sha256": {name: entry["sha256"] for name, entry in inventory.items()},
            "test_counts": {"test_extension_*.py": 64, "test_labeled_*.py": 44,
                "test_preserving_*.py": 36, "test_development_*.py": 23}}
        return output, compute, qa

    def test_native_qa_requires_same_job_all_regressions_and_no_skips(self):
        for mutation in ({"job_id": "other"}, {"skipped": 1}, {"tests_run": 1}, {"wrapper_syntax_passed": False}):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output, compute, qa = self.qa_fixture(root)
                qa.update(mutation)
                write_new_json(output / "native_qa.json", qa)
                with self.assertRaises(ValueError):
                    seal.validate_native_qa(output / "native_qa.json", root, output, compute, Evidence(),
                        root / "extension/development_plan.json")

    def test_native_qa_binds_every_source_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, compute, qa = self.qa_fixture(root)
            write_new_json(output / "native_qa.json", qa)
            good = seal.validate_native_qa(output / "native_qa.json", root, output, compute, Evidence(),
                root / "extension/development_plan.json")
            self.assertEqual(good["tests_run"], 167)
            (root / "extension/development_seal.py").write_text("different bytes\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                seal.validate_native_qa(output / "native_qa.json", root, output, compute, Evidence(),
                    root / "extension/development_plan.json")


if __name__ == "__main__":
    unittest.main()
