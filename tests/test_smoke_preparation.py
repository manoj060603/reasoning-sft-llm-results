"""Unexecuted offline CPU-preparation oracles; no real data/models/network."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import smoke_preparation as module
from extension.cpu_feasibility import Evidence, digest, value_digest, write_new_json
from extension.labeled_feasibility import near_duplicate
from reasoning.data import SYSTEM, prompt_id


class NativeTokenizer:
    eos_token_id = 151645
    chat_template = "native-fixture:unmodified-system-user-assistant;EosThenNewline"

    def __init__(self, fault=None):
        self.fault = fault

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Never apply a second special-token pass")
        return [self.eos_token_id if character == "\ue000" else ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        if return_dict or [message["role"] for message in messages] not in (
                ["system", "user"], ["system", "user", "assistant"]):
            raise AssertionError("Unexpected native role/return schema")
        if add_generation_prompt != (len(messages) == 2):
            raise AssertionError("Wrong assistant prefix policy")
        text = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
        if len(messages) == 3:
            body = messages[2]["content"]
            if self.fault == "rewrite":
                body += " rewritten"
            text += body + ("\n" if self.fault == "missing_eos" else "\ue000\n")
        return self.encode(text, add_special_tokens=False) if tokenize else text


def canonical_rows():
    result = []
    for question, response in (("Short question one?", "a"), ("Short question two?", "bb"),
                               ("Short question three?", "ccc"), ("Longest eligible?", "x" * 700),
                               ("Reserved source overlap?", "z" * 1000), ("Final benchmark overlap?", "d")):
        result.append({"id": prompt_id(question), "instruction": question, "input": "", "output": response, "system": SYSTEM})
    return result


def features(rows, tokenizer):
    result = []
    for row in rows:
        messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["instruction"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["output"]}],
            tokenize=True, add_generation_prompt=False)
        result.append({"id": row["id"], "total_tokens": len(full), "response_tokens": len(full) - len(prompt)})
    return result


def variant(question, task="math", field="problem", ordinal=0):
    return {"task": task, "field": field, "row_ordinal": ordinal, "question": question}


class LexicalAndSelectionTests(unittest.TestCase):
    def test_reverse_lexical_scan_matches_bruteforce_and_excludes_answer_only_text(self):
        questions = [" ".join("word" + str(index) for index in range(20)),
            " ".join("word" + str(index) for index in range(19)), "2 + 3", "2 - 3", "A", "Only in a gold answer"]
        variants = [variant(questions[0]), variant("2+3", "aime", "problem"), variant("a", "amc", "question")]
        pool = {prompt_id(text): text for text in questions}
        scan = module.benchmark_matches(pool, variants)
        expected = sorted(identifier for identifier, text in pool.items()
            if any(near_duplicate(text, item["question"]) for item in variants))
        self.assertEqual(scan["ids"], expected)
        self.assertNotIn(prompt_id("2 - 3"), scan["ids"])
        self.assertNotIn(prompt_id("Only in a gold answer"), scan["ids"])
        reverse = module.ExactIndex()
        for identifier, text in pool.items():
            reverse.add(text, identifier)
        self.assertEqual(sorted({identifier for item in variants for identifier in reverse.query(item["question"])}), expected)

    def test_question_variant_identities_include_both_amc_fields_and_never_raw_text(self):
        variants = [variant("AMC native question", "amc", "question"), variant("AMC rendered problem", "amc", "problem")]
        questions = {prompt_id(item["question"]): item["question"] for item in variants}
        result = module.benchmark_matches(questions, variants)
        self.assertEqual(result["variant_counts"], {"amc": 2})
        self.assertEqual(set(result["question_variant_sha256"]), {"amc"})
        self.assertEqual(len(result["ids"]), 2)
        self.assertNotIn("AMC native question", json.dumps(result))
        self.assertEqual(module.benchmark_matches(questions, list(reversed(variants))), result)

    def test_benchmark_input_rejects_nonquestion_fields_duplicate_owners_and_wrong_pool_ids(self):
        questions = {prompt_id("native question"): "native question"}
        for variants in ([dict(variant("native question"), answer="do not index")],
                         [variant("native question"), variant("changed text same identity")],
                         [variant("native question", field="answer")]):
            with self.assertRaises(ValueError):
                module.benchmark_matches(questions, variants)
        with self.assertRaises(ValueError):
            module.benchmark_matches({"0" * 64: "native question"}, [variant("native question")])

    def test_benchmark_loader_accesses_only_explicit_ten_question_variants(self):
        class QuestionOnly(dict):
            def get(self, key, default=None):
                if key not in ("question", "problem"):
                    raise AssertionError("Gold or model outcome field inspected")
                return super().get(key, default)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            specs, raw = [], {}
            for task in module.BENCHMARK_COUNTS:
                path = root / "deadline/artifacts/benchmarks" / task / "test.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("fixture\n")
                raw[path] = [QuestionOnly(question=task + " question", problem=task + " rendered problem",
                    answer="Only in a gold answer", prediction="Do not inspect", correctness=True)]
                for field in (("question", "problem") if task == "amc" else ("question",)):
                    specs.append({"format": "benchmark_jsonl", "path": str(path.relative_to(root)),
                        "sha256": digest(path), "question_field": field, "expected_rows": 1})
            binding = {"benchmark_hashes": {task: "a" * 64 for task in module.BENCHMARK_COUNTS}}
            with patch.object(module, "validate_benchmarks", return_value=binding) as readiness, \
                 patch.object(module, "_jsonl", side_effect=lambda path, evidence: raw[path]):
                variants, actual = module.load_benchmark_variants(root, {"exclusions": specs}, Evidence())
            self.assertEqual(len(variants), 10)
            self.assertEqual(actual, binding)
            self.assertEqual({item["field"] for item in variants if item["task"] == "amc"}, {"question", "problem"})
            self.assertNotIn("Only in a gold answer", json.dumps(variants))
            readiness.assert_called_once()

    def test_union_keeps_both_audit_identities_and_all_overlapping_ids_once(self):
        ids = sorted(prompt_id(text) for text in ("first", "second", "third"))
        receipt = module.exclusion_union(ids, ids[:2], ids[1:], "a" * 64, "b" * 64, "c" * 64, "d" * 64)
        self.assertEqual(receipt["population_ids_sha256"], value_digest(ids))
        self.assertEqual(receipt["components"]["reserved_source_evaluation"]["audit_receipt_sha256"], "a" * 64)
        self.assertEqual(receipt["components"]["final_benchmark_questions"]["audit_receipt_sha256"], "b" * 64)
        self.assertEqual(set(ids), set(receipt["components"]["reserved_source_evaluation"]["pool_ids"]) |
                         set(receipt["components"]["final_benchmark_questions"]["pool_ids"]))
        with self.assertRaises(ValueError):
            module.exclusion_union(ids, ["0" * 64], [], "a" * 64, "b" * 64, "c" * 64, "d" * 64)

    def test_selection_uses_both_exclusions_and_only_cached_lengths(self):
        rows, tokenizer = canonical_rows(), NativeTokenizer()
        cached = features(rows, tokenizer)
        eligible = {row["id"] for row in rows[:4]}
        choice = module.choose_witnesses(eligible, cached, [row["id"] for row in rows])
        expected = sorted(eligible, key=lambda identifier: (next(row["total_tokens"] for row in cached if row["id"] == identifier), identifier))
        self.assertEqual(choice["short"], expected[:3])
        self.assertEqual(choice["long"], rows[3]["id"])
        self.assertNotIn(rows[4]["id"], choice["short"] + [choice["long"]])
        self.assertNotIn(rows[5]["id"], choice["short"] + [choice["long"]])
        altered = [dict(row, nll=float("nan"), repetition=99, complete_box=False, correctness=False) for row in cached]
        changed = module.choose_witnesses(eligible, altered, [row["id"] for row in rows])
        self.assertEqual(changed["short"], choice["short"])
        self.assertEqual(changed["long"], choice["long"])

    def test_short_distinct_active_lengths_and_long_ties_follow_ascending_id(self):
        ids = sorted(prompt_id(str(index)) for index in range(7))
        cached = [{"id": identifier, "total_tokens": total, "response_tokens": active}
            for identifier, total, active in zip(ids, (100, 100, 101, 102, 103, 1000, 1000), (2, 2, 3, 4, 5, 900, 901))]
        choice = module.choose_witnesses(set(ids), cached, ids)
        self.assertEqual(choice["short"], [ids[0], ids[2], ids[3]])
        self.assertEqual(choice["long"], ids[5])

    def test_no_short_capacity_or_overlength_or_missing_feature_order_fails_without_refill(self):
        rows = canonical_rows()
        cached = features(rows, NativeTokenizer())
        order = [row["id"] for row in rows]
        for alteration in ("short", "long", "order", "duplicate", "bool"):
            changed, eligible = deepcopy(cached), set(order[:4])
            if alteration == "short":
                for row in changed:
                    row["response_tokens"] = 1
            elif alteration == "long":
                changed[3]["total_tokens"] = 32769
            elif alteration == "order":
                changed.reverse()
            elif alteration == "duplicate":
                changed[1]["id"] = changed[0]["id"]
            else:
                changed[0]["response_tokens"] = True
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                module.choose_witnesses(eligible, changed, order)

    def test_native_length_mismatch_is_fatal_and_has_no_reselection(self):
        example = {"id": "a" * 64, "provenance": {"total_tokens": 40, "active_tokens": 12}}
        for cached in (dict(total_tokens=41, response_tokens=12), dict(total_tokens=40, response_tokens=11)):
            with self.assertRaisesRegex(ValueError, "do not replace or refill"):
                module.assert_native_counts([example], {example["id"]: cached})

    def test_terminal_comes_from_exact_native_template_and_never_appends_missing_eos(self):
        tokenizer = NativeTokenizer()
        text, suffix = module.derive_terminal(tokenizer, deepcopy(module.TERMINAL_PROBE))
        self.assertEqual(text, "\ue000\n")
        self.assertEqual(suffix, (151645, 1010))
        for fault in ("rewrite", "missing_eos"):
            with self.assertRaises(ValueError):
                module.derive_terminal(NativeTokenizer(fault), module.TERMINAL_PROBE)
        with self.assertRaises(ValueError):
            module.derive_terminal(tokenizer, dict(module.TERMINAL_PROBE, user="a different probe"))

    def test_tokenizer_loader_is_local_only_without_remote_code_or_model_loader(self):
        calls = []
        token = object()
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs:
            (calls.append((args, kwargs)) or token)))
        with patch.dict(sys.modules, {"transformers": fake}):
            self.assertIs(module._load_tokenizer(Path("/private/native")), token)
        self.assertEqual(calls, [(("/private/native",), {"local_files_only": True, "trust_remote_code": False})])


def cohort_fixture():
    cohorts, ids_by_split, groups_by_split, receipt = {}, {}, {}, {"splits": {}}
    targets = {split: {"math": 2, "arc_challenge": 2} for split in module.SPLITS}
    for split in module.SPLITS:
        records, groups = [], []
        for domain in ("math", "arc_challenge"):
            for ordinal in range(2):
                identifier = value_digest({"source": split + domain, "row": ordinal})
                group_id = value_digest({"criterion": module.CRITERION, "member_ids": [identifier]})
                records.append({"id": identifier, "group_id": group_id, "source_kind": domain,
                    "lexical_screen_variants": [split + domain + str(ordinal)],
                    "label": {"answer": "DO_NOT_PRINT_GOLD"}, "original_record": {"solution": "PRIVATE_SOLUTION"}})
                groups.append({"group_id": group_id, "member_ids": [identifier], "rows": 1, "domain": domain, "stratum": domain})
        records.sort(key=lambda row: row["id"])
        groups.sort(key=lambda row: row["group_id"])
        cohorts[split], groups_by_split[split] = records, groups
        identifiers = ids_by_split[split] = [row["id"] for row in records]
        receipt["splits"][split] = {"rows": 4, "group_count": 4, "domain_rows": targets[split],
            "ids_sha256": value_digest(identifiers), "group_ids_sha256": value_digest([group["group_id"] for group in groups])}
    return cohorts, ids_by_split, groups_by_split, receipt, targets


class SourceSealAndPlanTests(unittest.TestCase):
    def unit_fixture(self, root):
        output = root / "extension/runs/native-smoke-preparation-999"
        output.mkdir(parents=True)
        inventory = {}
        for name in module.REQUIRED_CODE:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic-source-byte-fixture\n")
            inventory[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        cpu_plan, native_plan = root / "cpu_plan.json", root / "native_plan.json"
        write_new_json(cpu_plan, {"scope": "synthetic CPU plan"})
        write_new_json(native_plan, {"scope": "synthetic native plan"})
        compute = {"job_id": "999", "node": "owl001", "gpu_requested_or_allocated": False}
        qa = {"schema": module.UNIT_SCHEMA, "success": True, "tests_run": 300, "failures": 0,
            "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
            "gpu_disabled": True, "job_id": "999", "node": "owl001", "compute_proof": compute,
            "plan_sha256": digest(cpu_plan), "native_plan_sha256": digest(native_plan),
            "source_inventory": inventory, "code_sha256": {name: item["sha256"] for name, item in inventory.items()}}
        return output, cpu_plan, native_plan, compute, qa

    def test_current_unit_qa_binds_both_plans_same_cpu_job_and_all_new_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, cpu_plan, native_plan, compute, qa = self.unit_fixture(root)
            write_new_json(output / "unit_qa.json", qa)
            result = module.validate_unit_qa(output / "unit_qa.json", root, output, compute,
                Evidence(), cpu_plan, native_plan)
            self.assertEqual(result["path"], str((output / "unit_qa.json").relative_to(root)))
            self.assertEqual(result["code_sha256"], qa["code_sha256"])
            self.assertEqual(result["sha256"], digest(output / "unit_qa.json"))

    def test_old_failed_wrongjob_incomplete_or_changed_source_unit_qa_is_rejected(self):
        for alteration in ("old", "skipped", "job", "native_plan", "missing_source", "old_count", "source_bytes"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output, cpu_plan, native_plan, compute, qa = self.unit_fixture(root)
                if alteration == "old":
                    qa["schema"] = "extension-development-seal-native-qa-v2"
                elif alteration == "skipped":
                    qa["skipped"] = 1
                elif alteration == "job":
                    qa["job_id"] = "1000"
                elif alteration == "native_plan":
                    qa["native_plan_sha256"] = "0" * 64
                elif alteration == "missing_source":
                    del qa["source_inventory"]["extension/preserving_data.py"]
                    del qa["code_sha256"]["extension/preserving_data.py"]
                elif alteration == "old_count":
                    qa["tests_run"] = 230
                else:
                    (root / "extension/preserving_data.py").write_text("changed after QA\n")
                write_new_json(output / "unit_qa.json", qa)
                with self.assertRaises(ValueError):
                    module.validate_unit_qa(output / "unit_qa.json", root, output, compute,
                        Evidence(), cpu_plan, native_plan)

    def test_private_cohorts_remain_disjoint_whole_groups_and_return_only_aggregates(self):
        cohorts, identifiers, groups, receipt, targets = cohort_fixture()
        with patch.object(module, "TARGETS", targets):
            result = module.validate_cohorts(cohorts, identifiers, groups, receipt)
        self.assertEqual((result["records"], result["groups"]), (12, 12))
        self.assertNotIn("DO_NOT_PRINT_GOLD", json.dumps(result))
        self.assertNotIn("PRIVATE_SOLUTION", json.dumps(result))
        changed = deepcopy(cohorts)
        for records in changed.values():
            for record in records:
                record["label"]["answer"] = "A_DIFFERENT_GOLD"
        with patch.object(module, "TARGETS", targets):
            self.assertEqual(module.validate_cohorts(changed, identifiers, groups, receipt), result)

    def test_missing_group_member_duplicate_split_or_tampered_seal_fails(self):
        for alteration in ("group_member", "record_group", "overlap", "count", "hash"):
            cohorts, identifiers, groups, receipt, targets = cohort_fixture()
            if alteration == "group_member":
                groups["calibration"][0]["member_ids"] = ["0" * 64]
            elif alteration == "record_group":
                cohorts["calibration"][0]["group_id"] = "0" * 64
            elif alteration == "overlap":
                cohorts["development"] = deepcopy(cohorts["calibration"])
                identifiers["development"] = list(identifiers["calibration"])
            elif alteration == "count":
                receipt["splits"]["development"]["rows"] -= 1
            else:
                receipt["splits"]["confirmation"]["ids_sha256"] = "0" * 64
            with self.subTest(alteration=alteration), patch.object(module, "TARGETS", targets), self.assertRaises(ValueError):
                module.validate_cohorts(cohorts, identifiers, groups, receipt)

    def test_current_root_plan_is_fixed_cpu_only_and_independent_native_plan(self):
        project = Path(module.__file__).resolve().parents[1]
        plan = module.read_json(project / "extension/smoke_preparation_plan.json")
        self.assertIs(module.validate_plan(plan), plan)
        for field in ("model_inference", "training", "automatic_downstream"):
            altered = deepcopy(plan)
            altered[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_plan(altered)

    def test_changed_population_probe_runtime_or_context_is_not_accepted(self):
        project = Path(module.__file__).resolve().parents[1]
        original = module.read_json(project / "extension/smoke_preparation_plan.json")
        for alteration in ("pool", "probe", "reserve", "context", "base"):
            plan = deepcopy(original)
            if alteration == "pool":
                plan["pool"]["sha256"] = "0" * 64
            elif alteration == "probe":
                plan["terminal_probe"]["assistant"] += "changed"
            elif alteration == "reserve":
                plan["reserve_seconds"] = 1
            elif alteration == "context":
                plan["witness_selection"]["cutoffs"]["short"] = 1024
            else:
                del plan["base"]["files"]["tokenizer.json"]
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                module.validate_plan(plan)


class PreparationIntegrationTests(unittest.TestCase):
    """Mock external admission/base audits; exercise pure orchestration and bytes."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "extension/runs/native-smoke-preparation-999"
        self.output.mkdir(parents=True)
        self.tokenizer, self.rows = NativeTokenizer(), canonical_rows()
        self.compute = {"job_id": "999", "node": "owl001", "gpu_requested_or_allocated": False}
        self.write(self.output / "unit_qa.json", {"synthetic": True})
        project = Path(module.__file__).resolve().parents[1]
        native_plan = module.read_json(project / "extension/native_model_admission_plan.json")
        self.native_plan_path = self.root / "extension/native_model_admission_plan.json"
        self.write(self.native_plan_path, native_plan)
        self.pool_path = self.root / "pool.json"
        self.write(self.pool_path, self.rows)
        self.feature_path = self.root / "features.json"
        self.feature_rows = features(self.rows, self.tokenizer)
        self.write(self.feature_path, self.feature_rows)
        meta_path = self.root / "feature_meta.json"
        self.write(meta_path, {"input_sha256": digest(self.pool_path), "source_sha256": module.SOURCE_SHA256,
            "resolved_tokenizer_commit": module.native.REVISION})
        source_path = self.root / "original_source_audit.json"
        self.write(source_path, {"schema": "extension-cpu-feasibility-v1", "complete": True,
            "original_source": {"repository": "redsgnaoh/acereason11_100k", "revision": module.SOURCE_REVISION,
                "sha256": module.SOURCE_SHA256, "prepared_and_heldout_original_pairs_verified": 84029}})
        labeled = self.root / "labeled_plan.json"
        self.write(labeled, {"exclusions": []})
        self.reserved_path = self.root / "reserved.json"
        self.write(self.reserved_path, [self.rows[4]["id"]])
        seal_path = self.root / "source_seal.json"
        self.write(seal_path, {"complete": True})
        self.base = self.root / "private-base"
        self.base.mkdir()
        self.plan = {"wall_time_seconds": 7200, "reserve_seconds": 172800,
            "native_plan": self.pin(self.native_plan_path), "development_seal": self.pin(seal_path),
            "pool": dict(self.pin(self.pool_path), expected_rows=len(self.rows)),
            "features": self.pin(self.feature_path), "feature_meta": self.pin(meta_path),
            "original_source_audit": self.pin(source_path), "labeled_plan": self.pin(labeled),
            "terminal_probe": deepcopy(module.TERMINAL_PROBE),
            "base": {"path": str(self.base), "identity_receipt": {"path": "identity.json", "sha256": "a" * 64}}}
        self.plan_path = self.root / "extension/smoke_preparation_plan.json"
        self.write(self.plan_path, self.plan)
        self.unit = {"path": str((self.output / "unit_qa.json").relative_to(self.root)), "sha256": digest(self.output / "unit_qa.json"),
            "schema": module.UNIT_SCHEMA, "tests_run": 300, "code_sha256": {"fixture.py": "b" * 64}}
        self.variants = [variant(self.rows[5]["instruction"], task="math")]
        self.binding = {"benchmark_hashes": {task: "c" * 64 for task in module.BENCHMARK_COUNTS}}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new_json(path, value)

    def pin(self, path):
        return {"path": str(path.relative_to(self.root)), "sha256": digest(path)}

    def run_prepare(self, *, mismatch=False):
        if mismatch:
            self.feature_rows[0]["total_tokens"] += 1
            self.feature_path.unlink()
            self.write(self.feature_path, self.feature_rows)
            self.plan["features"] = self.pin(self.feature_path)
            self.plan_path.unlink()
            self.write(self.plan_path, self.plan)
        with patch.object(module, "require_cpu_allocation", return_value=self.compute), \
             patch.object(module, "deadline_guard", return_value={}), \
             patch.object(module, "validate_plan", side_effect=lambda value: value), \
             patch.object(module, "validate_unit_qa", return_value=self.unit), \
             patch.object(module, "load_development", return_value=({}, [self.rows[4]["id"]],
                 {"path": str(self.reserved_path.relative_to(self.root)), "sha256": digest(self.reserved_path)}, {"question_inventory_sha256": "d" * 64})), \
             patch.object(module, "load_benchmark_variants", return_value=(self.variants, self.binding)), \
             patch.object(module, "_validate_base", return_value=self.base), \
             patch.object(module, "_load_tokenizer", return_value=self.tokenizer) as tokenizer_loader, \
             patch.object(module, "POOL_SHA256", digest(self.pool_path)), \
             patch.object(module, "FEATURES_SHA256", digest(self.feature_path)), \
             patch.object(module, "FEATURE_META_SHA256", self.plan["feature_meta"]["sha256"]):
            result = module.prepare(self.root, self.plan_path.relative_to(self.root), self.output, self.output / "unit_qa.json")
        tokenizer_loader.assert_called_once_with(self.base)
        return result

    def test_acyclic_preparation_binds_exact_four_rows_and_both_exclusions(self):
        prepared = self.run_prepare()
        preparation = module.read_json(self.output / "preparation.json")
        self.assertEqual(preparation["rows_sha256"], value_digest(prepared["rows"]))
        self.assertNotIn("prepared_sha256", preparation)
        self.assertEqual(prepared["source_audit"]["preparation"]["sha256"], digest(self.output / "preparation.json"))
        self.assertEqual(preparation["plan_sha256"], digest(self.native_plan_path))
        self.assertEqual(preparation["cpu_plan_sha256"], digest(self.plan_path))
        self.assertEqual(preparation["native_qa"]["sha256"], digest(self.output / "unit_qa.json"))
        all_rows = prepared["rows"]["short"] + [prepared["rows"]["long"]]
        self.assertEqual({row["id"] for row in all_rows}, {row["id"] for row in self.rows[:4]})
        self.assertEqual(prepared["rows"]["long"]["id"], self.rows[3]["id"])
        self.assertEqual({row["provenance"]["cutoff"] for row in prepared["rows"]["short"]}, {512})
        self.assertEqual(prepared["rows"]["long"]["provenance"]["cutoff"], 32768)
        self.assertEqual(set(prepared["source_audit"]), module.native.AUDIT_FIELDS)
        self.assertFalse((self.output / "native_qa.json").exists())
        self.assertFalse((self.output / "complete.json").exists())
        self.assertFalse(preparation["gpu_stage_admitted"])

    def test_byte_hash_and_value_hash_of_exclusion_receipt_are_distinct_and_bound(self):
        prepared = self.run_prepare()
        union = module.read_json(self.output / "exclusion_receipt.json")
        preparation = module.read_json(self.output / "preparation.json")
        self.assertEqual(preparation["exclusion_receipt_sha256"], digest(self.output / "exclusion_receipt.json"))
        self.assertEqual(preparation["exclusion_receipt_value_sha256"], value_digest(union))
        self.assertNotEqual(preparation["exclusion_receipt_sha256"], preparation["exclusion_receipt_value_sha256"])
        self.assertEqual(prepared["rows"]["long"]["provenance"]["exclusion_receipt_sha256"], value_digest(union))

    def test_actual_cached_length_failure_keeps_no_prepared_cache_and_never_refills(self):
        with self.assertRaisesRegex(ValueError, "do not replace or refill"):
            self.run_prepare(mismatch=True)
        self.assertFalse((self.output / "prepared.json").exists())
        self.assertFalse((self.output / "preparation.json").exists())
        self.assertTrue((self.output / "benchmark_audit.json").exists())

    def test_exclusive_output_and_cpu_allocation_gate_precede_input_work(self):
        write_new_json(self.output / "unexpected.json", {})
        with patch.object(module, "require_cpu_allocation", return_value=self.compute), \
             patch.object(module, "_pin") as reads, self.assertRaisesRegex(ValueError, "exclusive current-job"):
            module.prepare(self.root, self.plan_path.relative_to(self.root), self.output, self.output / "unit_qa.json")
        reads.assert_not_called()
        with patch.object(module, "require_cpu_allocation", side_effect=ValueError("not compute")), \
             patch.object(module, "_pin") as reads, self.assertRaisesRegex(ValueError, "not compute"):
            module.prepare(self.root, self.plan_path.relative_to(self.root), self.output, self.output / "unit_qa.json")
        reads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
