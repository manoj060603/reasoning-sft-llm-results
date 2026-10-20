"""Native CPU synthetic oracles; no network, original corpus or model access."""
from copy import deepcopy
import itertools
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from extension import labeled_feasibility as module
from extension.cpu_feasibility import Evidence, value_digest
from reasoning.data import canonical


def math_raw(problem="Find the integer x.", solution=r"It is \boxed{2}."):
    return {"problem": problem, "level": "Level 1", "type": "Algebra", "solution": solution}


def arc_raw():
    return {"id": "ARC-1", "question": "Which material conducts electricity?",
        "choices": {"label": ["1", "2", "3", "4"], "text": ["copper", "wood", "rubber", "glass"]}, "answerKey": "1"}


def record(text, ordinal=0):
    return module.source_record("math", math_raw(text), {"repository": "EleutherAI/hendrycks_math",
        "revision": module.MATH_REVISION, "file": "algebra/train/0000.parquet", "row_ordinal": ordinal})[0]


def plan():
    return {"schema": "extension-labeled-feasibility-plan-v1", "cpu_only": True,
        "model_inference": False, "choose_splits": False, "automatic_downstream": False,
        "deadline_utc": "2026-10-28T03:59:00+00:00", "wall_time_seconds": 7200, "reserve_seconds": 172800,
        "source_metadata": "extension/development_source_metadata.json",
        "near_duplicate": dict(module.CRITERION),
        "grouping_policy": {
            "within_labeled": "connected components across all label-valid records",
            "exclusion_propagation": "Exclude entire labeled component if any member directly matches a declared prior or benchmark exclusion"},
        "sources": [{"repo_id": repo, "revision": values[1], "files": list(values[2]),
            "kind": values[0], "expected_rows": 7500 if values[0] == "math" else 1119,
            "license_card": deepcopy(module.LICENSE_CARDS[repo])} for repo, values in module.SOURCE_PINS.items()],
        "exclusions": [{"path": "questions.jsonl", "format": "benchmark_jsonl", "reason": "test", "question_field": "question"}]}


class LexicalTests(unittest.TestCase):
    def test_nfkc_casefold_whitespace_preserves_numbers_and_operators(self):
        self.assertEqual(module.lexical(" Ｘ  + １２ =  3 ")[0], "x + 12 = 3")
        self.assertEqual(module.lexical("x+12=3")[1], ("x", "+", "12", "=", "3"))
        self.assertNotEqual(module.lexical("2+3")[1], module.lexical("2-3")[1])
        self.assertNotEqual(module.lexical("2+3")[1], module.lexical("2+4")[1])

    def test_integer_threshold_boundary_and_short_exact_only(self):
        self.assertTrue(module.near_duplicate("a b c d e f", "a b c d e f g"))
        self.assertFalse(module.near_duplicate("a b c d e f", "a b c d e f g h"))
        self.assertTrue(module.near_duplicate("A  B", "a b"))
        self.assertFalse(module.near_duplicate("a b", "a b c"))

    def test_inverted_index_matches_bruteforce_complete_oracle(self):
        values = [" ".join(tokens) for size in range(1, 7) for tokens in itertools.product(("a", "b"), repeat=size)]
        values += ["a b c d e f", "a b c d e f g", "a b c d e f g h", "2+3=5", "2-3=-1"]
        index = module.ExactIndex()
        for position, text in enumerate(values):
            index.add(text, str(position))
        for query in values:
            expected = {str(position) for position, text in enumerate(values) if module.near_duplicate(query, text)}
            self.assertEqual(set(index.query(query)), expected, query)

    def test_rare_prefix_and_repeated_shingles_match_random_oracle(self):
        rng = random.Random(9)
        values = [" ".join(rng.choice(["x", "y", "z", "+", "17"]) for _ in range(rng.randint(3, 18))) for _ in range(150)]
        index = module.ExactIndex()
        for position, text in enumerate(values):
            index.add(text, str(position))
        for query in values[::3]:
            self.assertEqual(set(index.query(query)), {str(position) for position, text in enumerate(values)
                if module.near_duplicate(query, text)})

    def test_duplicate_exclusion_questions_merge_owners(self):
        index = module.ExactIndex()
        index.add("Repeated question?", "benchmark")
        index.add(" REPEATED  question? ", "prior_training")
        self.assertEqual(len(index.entries), 1)
        self.assertEqual(index.query("Repeated question?"), {"benchmark": ["exact"], "prior_training": ["exact"]})


class LabelTests(unittest.TestCase):
    def test_math_final_nested_box_and_original_preservation(self):
        raw = math_raw(solution=r"Earlier \boxed{8}; final \boxed{\frac{1}{2}}.")
        value, reason = module.source_record("math", raw, {"row_ordinal": 1})
        self.assertIsNone(reason)
        self.assertEqual(value["label"]["answer"], r"\frac{1}{2}")
        self.assertEqual(value["original_record"], raw)
        self.assertNotIn("frac", value["prompt"])

    def test_missing_empty_or_unbalanced_last_math_box_quarantined(self):
        for solution in ("No boxed final answer", r"\boxed{}", r"Earlier \boxed{2}; final \boxed{3"):
            value, reason = module.source_record("math", math_raw(solution=solution), {})
            self.assertEqual(reason, "unextractable_final_boxed_answer")
            self.assertIsNone(value["label"]["answer"])

    def test_unsupported_math_schema_is_fatal(self):
        for raw in (dict(math_raw(), extra="unknown"), {"problem": "q", "solution": "a"},
                    dict(math_raw(), level=2), dict(math_raw(), problem="")):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                module.source_record("math", raw, {})

    def test_arc_numeric_labels_match_without_remapping(self):
        value, reason = module.source_record("arc_challenge", arc_raw(), {})
        self.assertIsNone(reason)
        self.assertEqual(value["label"]["choice_index"], 0)
        raw = arc_raw()
        raw["answerKey"] = "A"
        value, reason = module.source_record("arc_challenge", raw, {})
        self.assertEqual(reason, "missing_or_unmatched_answer_key")

    def test_arc_choice_order_and_question_label_separation(self):
        raw = arc_raw()
        first, _ = module.source_record("arc_challenge", raw, {"row_ordinal": 0})
        raw["answerKey"] = "2"
        second, _ = module.source_record("arc_challenge", raw, {"row_ordinal": 0})
        self.assertEqual(first["question_sha256"], second["question_sha256"])
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertNotEqual(first["label_id"], second["label_id"])
        self.assertIn("1. copper\n2. wood\n3. rubber\n4. glass", first["prompt"])
        raw["choices"]["text"][0] = "silver"
        third, _ = module.source_record("arc_challenge", raw, {"row_ordinal": 0})
        self.assertEqual(first["question_sha256"], third["question_sha256"])
        self.assertNotEqual(first["prompt_sha256"], third["prompt_sha256"])
        self.assertEqual(first["lexical_screen_variants"][0], raw["question"])

    def test_arc_malformed_choices_are_fatal(self):
        for choices in ({"label": ["A", "A"], "text": ["one", "two"]},
                        {"label": ["A", "B"], "text": ["one", ""]},
                        {"label": ["A", "B"], "text": ["one"]}):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                module.source_record("arc_challenge", dict(arc_raw(), choices=choices), {})


class GroupAndExclusionTests(unittest.TestCase):
    def test_transitive_group_propagation_blocks_excluded_bridge(self):
        a, b, c = [record(text, number) for number, text in enumerate(
            ("a b c d e f", "a b c d e f g", "b c d e f g"))]
        self.assertFalse(module.near_duplicate(a["question"], c["question"]))
        index = module.ExactIndex()
        index.add(a["question"], "prior_train")
        kept, excluded, groups = module.group_and_filter([a, b, c], index)
        self.assertEqual(kept, [])
        self.assertEqual(len(groups), 1)
        self.assertTrue(next(row for row in excluded if row["id"] == c["id"])["propagated_group_exclusion"])
        self.assertEqual(sum(row["propagated_group_exclusion"] for row in excluded), 1)

    def test_no_arbitrary_near_duplicate_drop_and_order_invariant_groups(self):
        values = [record("a b c d e f", 0), record("a b c d e f g", 1), record("different science question", 2)]
        first = module.group_and_filter(values, module.ExactIndex())
        second = module.group_and_filter(list(reversed(values)), module.ExactIndex())
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 3)
        self.assertEqual(len(first[2]), 2)

    def test_reason_union_and_id_resolution_never_use_solutions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [canonical({"instruction": "Only question text", "output": "secret solution words"})]
            (root / "pool.json").write_text(json.dumps(rows))
            (root / "ids.json").write_text(json.dumps([rows[0]["id"]]))
            (root / "bench.jsonl").write_text(json.dumps({"question": "Only question text", "answer": "another secret"}) + "\n")
            specs = [{"path": "ids.json", "format": "id_list", "lookup": "pool.json", "reason": "train", "expected_rows": 1},
                     {"path": "bench.jsonl", "format": "benchmark_jsonl", "question_field": "question", "reason": "benchmark", "expected_rows": 1}]
            index, receipts = module.load_exclusions(root, specs, Evidence())
            self.assertEqual(set(index.query("Only question text")), {"train", "benchmark"})
            self.assertEqual(index.query("secret solution words"), {})
            self.assertEqual(len(receipts), 2)
            specs[0]["expected_rows"] = 2
            with self.assertRaises(ValueError):
                module.load_exclusions(root, specs, Evidence())

    def test_ids_pointer_hash_and_duplicate_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [canonical({"instruction": "Question", "output": "Answer"})]
            (root / "pool.json").write_text(json.dumps(rows))
            (root / "meta.json").write_text(json.dumps({"reservoir_ids": [rows[0]["id"]]}))
            spec = {"path": "meta.json", "format": "id_list", "lookup": "pool.json",
                    "json_pointer": "/reservoir_ids", "reason": "scored"}
            index, _ = module.load_exclusions(root, [spec], Evidence())
            self.assertEqual(index.query("Question"), {"scored": ["exact"]})
            with self.assertRaises(ValueError):
                module.load_exclusions(root, [dict(spec, lookup_sha256="0" * 64)], Evidence())
            (root / "meta.json").write_text(json.dumps({"reservoir_ids": [rows[0]["id"]] * 2}))
            with self.assertRaises(ValueError):
                module.load_exclusions(root, [spec], Evidence())

    def test_viewed_categories_disjoint_but_training_reason_overlap_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [canonical({"instruction": "Question", "output": "Answer"})]
            (root / "pool.json").write_text(json.dumps(rows))
            (root / "ids.json").write_text(json.dumps([rows[0]["id"]]))
            common = {"path": "ids.json", "format": "id_list", "lookup": "pool.json"}
            module.load_exclusions(root, [dict(common, reason="training"), dict(common, reason="reservoir")], Evidence())
            with self.assertRaisesRegex(ValueError, "viewed.*overlap"):
                module.load_exclusions(root, [dict(common, reason="viewed_old"), dict(common, reason="viewed_new")], Evidence())


class MetadataGuardTests(unittest.TestCase):
    def test_plan_accepts_scope_metadata_but_rejects_policy_drift(self):
        value = plan()
        value["grouping_policy"]["scope"] = "Explicit lexical scope"
        module.validate_plan(value)
        for change in ({"jaccard_numerator": 3}, {"shingle_tokens": 2}):
            bad = deepcopy(value)
            bad["near_duplicate"].update(change)
            with self.assertRaises(ValueError):
                module.validate_plan(bad)
        bad = deepcopy(value)
        bad["grouping_policy"]["within_labeled"] = "drop duplicates"
        with self.assertRaises(ValueError):
            module.validate_plan(bad)
        bad = deepcopy(value)
        bad["sources"][1]["files"] = ["ARC-Easy/train-00000-of-00001.parquet"]
        with self.assertRaises(ValueError):
            module.validate_plan(bad)

    def test_license_card_scalar_list_absence_and_mismatch(self):
        self.assertTrue(module.card_license("---\nlicense: mit\n---\nbody", "mit")["field_present"])
        self.assertTrue(module.card_license("---\nlicense:\n- cc-by-sa-4.0\n---", "cc-by-sa-4.0")["field_present"])
        self.assertFalse(module.card_license("No YAML header", "mit")["field_present"])
        with self.assertRaises(ValueError):
            module.card_license("---\nlicense: apache-2.0\n---", "mit")

    def test_all_nine_readiness_and_both_amc_fields_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "deadline/artifacts/benchmark_readiness.json"
            path.parent.mkdir(parents=True)
            hashes = {name: value_digest(name) for name in module.BENCHMARK_COUNTS}
            receipt = {"schema": "deadline-benchmark-readiness-v1", "complete": True, "evidence": {
                "benchmark_hashes": hashes, "benchmark_counts": module.BENCHMARK_COUNTS,
                "supplied_benchmark_hashes": {key: value for key, value in hashes.items() if key != "aime25"}}}
            receipt["binding_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            path.write_text(json.dumps(receipt))
            value = plan()
            value["benchmark_readiness"] = {"path": str(path.relative_to(root)), "sha256": module.digest(path)}
            value["exclusions"] = [{"path": "deadline/artifacts/benchmarks/" + name + "/test.jsonl",
                "format": "benchmark_jsonl", "question_field": "question", "reason": "benchmark_" + name,
                "expected_rows": count, "sha256": hashes[name]} for name, count in module.BENCHMARK_COUNTS.items()]
            value["exclusions"].append(dict(next(item for item in value["exclusions"] if item["reason"] == "benchmark_amc"), question_field="problem"))
            module.validate_benchmarks(root, value, Evidence())
            original_amc = next(item for item in value["exclusions"] if item["reason"] == "benchmark_amc")
            original_amc["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash/count"):
                module.validate_benchmarks(root, value, Evidence())

    def test_native_qa_requires_current_sources_and_matching_cpu_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "extension/runs/labeled-123"
            output.mkdir(parents=True)
            names = ("extension/labeled_feasibility.py", "tests/test_labeled_feasibility.py",
                "extension/cpu_feasibility.py", "reasoning/data.py", "extension/labeled_feasibility.slurm",
                "extension/labeled_plan.json", "extension/development_source_metadata.json")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("native source fixture " + name)
            inventory = {name: {"sha256": module.digest(root / name), "bytes": (root / name).stat().st_size} for name in names}
            compute = {"job_id": "123", "node": "test-cpu", "gpu_requested_or_allocated": False}
            qa = {"schema": "extension-labeled-feasibility-native-qa-v1", "success": True, "tests_run": 90,
                "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
                "gpu_disabled": True, "wrapper_syntax_passed": True, "job_id": "123", "node": "test-cpu",
                "compute_proof": compute, "source_inventory": inventory,
                "code_sha256": {name: value["sha256"] for name, value in inventory.items()}}
            path = output / "native_qa.json"
            def check(value):
                path.write_text(json.dumps(value))
                return module.validate_qa(path, root, output, compute, Evidence(),
                    root / "extension/labeled_plan.json", root / "extension/development_source_metadata.json")
            self.assertEqual(check(qa)["tests_run"], 90)
            for change in ({"expected_failures": 1}, {"wrapper_syntax_passed": False}, {"gpu_disabled": False},
                           {"compute_proof": dict(compute, job_id="999")}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    check(dict(qa, **change))
            (root / "reasoning/data.py").write_text("changed source")
            with self.assertRaises(ValueError):
                check(qa)

    def test_failure_never_creates_complete_and_download_adapter_is_fake(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "extension/runs/labeled-123"
            output.mkdir(parents=True)
            (output / "native_qa.json").write_text("{}")
            value = plan()
            (root / "extension/labeled_plan.json").write_text(json.dumps(value))
            metadata = {"schema": "reasoning-sft-fresh-development-source-pins-v1", "sources": [
                {"repo_id": repo, "resolved_revision": values[1], "files": list(values[2])}
                for repo, values in module.SOURCE_PINS.items()]}
            (root / "extension/development_source_metadata.json").write_text(json.dumps(metadata))
            calls = []
            def fake_download(spec, filename, destination):
                calls.append((spec["repo_id"], filename))
                path = destination / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("---\nlicense: " + spec["license_card"]["declared_license"] + "\n---" if filename == "README.md" else "fake parquet bytes")
                return path
            with patch.object(module, "require_cpu_allocation", return_value={"job_id": "123", "node": "fake-cpu"}), \
                 patch.object(module, "validate_qa", return_value={"synthetic": True}), \
                 patch.object(module, "validate_benchmarks", return_value={}), \
                 patch.object(module, "load_exclusions", return_value=(module.ExactIndex(), [])):
                with self.assertRaisesRegex(ValueError, "Unsupported MATH"):
                    module.audit(root, "extension/labeled_plan.json", output, output / "native_qa.json",
                        downloader=fake_download, reader=lambda _: iter([{"unsupported": "schema"}]))
            self.assertTrue(calls)
            self.assertTrue((output / "started.json").exists())
            self.assertFalse((output / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
