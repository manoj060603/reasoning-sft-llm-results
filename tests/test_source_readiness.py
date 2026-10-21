"""Authored CPU readiness integration oracles; no original data/model loading."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extension import source_readiness as module
from extension.cpu_feasibility import value_digest
from extension.labeled_feasibility import CRITERION, source_record
from extension.preserving_data import text_sha256


class Tokenizer:
    eos_token_id = 151645
    chat_template = "source-fixture-system-user-assistant-v1"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert add_generation_prompt is True and [row["role"] for row in messages] == ["system", "user"]
        rendered = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
        return self.encode(rendered) if tokenize else rendered


def records():
    rows = []
    for index in range(16):
        native, problem = {"source": "fixture", "row": index}, "Compute " + str(index) + "+1."
        raw = {"problem": problem, "level": "Level 1", "type": "Algebra", "solution": "A proof. \\boxed{" + str(index + 1) + "}"}
        row, rejected = source_record("math", raw, native)
        assert rejected is None
        row["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [row["id"]]})
        rows.append(row)
    for index in range(16):
        raw = {"id": "arc-" + str(index), "question": "Select fixture " + str(index),
            "choices": {"label": ["2", "1"], "text": ["correct fixture", "other fixture"]}, "answerKey": "2"}
        row, rejected = source_record("arc_challenge", raw, {"source": "fixture-arc", "row": index})
        assert rejected is None
        row["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [row["id"]]})
        rows.append(row)
    return sorted(rows, key=lambda row: row["id"])


def identity(tokenizer):
    return {"files": {"fixture.json": "a" * 64}, "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "eos_token_id": tokenizer.eos_token_id}


def correct(row, text, **kwargs):
    assert kwargs == {"timeout_seconds": 5}
    return {"correct": True, "rankable": True, "status": "graded", "denominator": 1}


class SourceReadinessTests(unittest.TestCase):
    def test_all_32_native_prompts_keep_exact_question_choice_order_and_seeds(self):
        tokenizer, population = Tokenizer(), records()
        cache, prechecks = module.build_calibration_cache(population, tokenizer, identity(tokenizer), grader=correct)
        self.assertEqual((len(cache), len(prechecks)), (32, 32))
        self.assertEqual([row["record"] for row in cache], population)
        self.assertEqual({row["record"]["group_id"] for row in cache}, {row["group_id"] for row in population})
        for row in cache:
            self.assertEqual(row["sample_indices"], list(range(8)))
            self.assertEqual(row["request_seed"], (20261021 + int(row["record"]["id"][:8], 16)) % (2 ** 31))
            self.assertEqual(row["prompt"]["prompt_token_ids"], tokenizer.encode(row["prompt"]["rendered_prompt"]))
            if row["record"]["source_kind"] == "arc_challenge":
                prompt = row["prompt"]["messages"][1]["content"]
                self.assertLess(prompt.index("2. correct fixture"), prompt.index("1. other fixture"))

    def test_missing_extra_duplicate_wrong_domain_and_split_groups_are_not_refilled(self):
        population, tokenizer = records(), Tokenizer()
        for changed in (population[:-1], population + [population[0]], list(reversed(population)),
                        [population[0]] + population[:-1]):
            with self.assertRaises(ValueError):
                module.build_calibration_cache(changed, tokenizer, identity(tokenizer), grader=correct)
        changed = deepcopy(population)
        changed[1]["group_id"] = changed[0]["group_id"]
        with self.assertRaises(ValueError):
            module.build_calibration_cache(changed, tokenizer, identity(tokenizer), grader=correct)

    def test_self_grading_timeout_or_crash_blocks_everything_instead_of_dropping_record(self):
        tokenizer = Tokenizer()
        for bad in ({"correct": False, "rankable": True}, {"correct": False, "rankable": False, "status": "timeout"},
                    {"correct": False, "rankable": False, "status": "worker_crash"}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "self-comparison"):
                module.build_calibration_cache(records(), tokenizer, identity(tokenizer), grader=lambda *a, **k: bad)

    def test_unsupported_gold_fails_before_prompt_or_self_comparison(self):
        tokenizer = Tokenizer()
        with patch.object(module.source, "validate_gold", return_value={"supported": False}), \
             patch.object(module.source, "serialize_prompt") as serializer, \
             patch.object(module.source, "grade_response") as grader:
            with self.assertRaisesRegex(ValueError, "no refill"):
                module.build_calibration_cache(records(), tokenizer, identity(tokenizer))
        serializer.assert_not_called()
        grader.assert_not_called()

    def test_native_gold_leak_is_detected_by_actual_mutation_pair(self):
        tokenizer = Tokenizer()
        original = module.source.serialize_prompt
        def leaking(record, *args, **kwargs):
            value = original(record, *args, **kwargs)
            value["rendered_prompt"] += str(record["label"])
            return value
        with patch.object(module.source, "serialize_prompt", side_effect=leaking), \
             self.assertRaisesRegex(ValueError, "Changing gold"):
            module.build_calibration_cache(records(), tokenizer, identity(tokenizer), grader=correct)

    def test_gold_mutation_preserves_question_choices_and_does_not_mutate_input(self):
        for row in records():
            before = deepcopy(row)
            changed = module.gold_mutation(row)
            self.assertEqual(row, before)
            self.assertEqual(changed["question"], row["question"])
            self.assertEqual(changed["id"], row["id"])
            if row["source_kind"] == "arc_challenge":
                self.assertEqual(changed["original_record"]["choices"], row["original_record"]["choices"])
            self.assertNotEqual(changed["label"], row["label"])

    def test_overlong_prompt_stops_without_cropping_or_refilling(self):
        population, tokenizer = records(), Tokenizer()
        index = next(index for index, row in enumerate(population) if row["source_kind"] == "math")
        old = population[index]
        raw = deepcopy(old["original_record"])
        raw["problem"] = "long question " * 8192
        changed, rejected = source_record("math", raw, old["source"])
        self.assertIsNone(rejected)
        changed["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [changed["id"]]})
        population[index] = changed
        population.sort(key=lambda row: row["id"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            module.build_calibration_cache(population, tokenizer, identity(tokenizer), grader=correct)

    def test_exact_cpu_scope_sampling_context_and_gold_population_are_fixed(self):
        plan = json.loads((Path(module.__file__).resolve().parent / "source_readiness_plan.json").read_text())
        self.assertIs(module.validate_plan(plan), plan)
        changes = (lambda v: v.update(model_inference=True), lambda v: v.update(training=True),
            lambda v: v.update(automatic_downstream=True), lambda v: v.update(reserve_seconds=0),
            lambda v: v["calibration"].update(split="confirmation"), lambda v: v["calibration"].update(record_count=31),
            lambda v: v["calibration"].update(maximum_prompt_tokens=16384),
            lambda v: v["source_generation_profile"].update(max_new_tokens=32768),
            lambda v: v["source_generation_profile"].update(additional_stop_strings=["answer"]),
            lambda v: v["source_generation_profile"].update(n=1))
        for change in changes:
            changed = deepcopy(plan)
            change(changed)
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_math_worker_identity_cannot_be_replaced_by_an_unrelated_installed_utils_module(self):
        expected = {"loaded_modules": {"utils.grader": "/fixture/eval/utils/grader.py"},
            "runtime_versions": {"sympy": "1.14.0"}}
        provenance = {**deepcopy(expected), "include_percentage": False, "is_close": True, "internal_timeout": False,
            "timeout_owner": "external_spawn_process", "normalization_order": "explicit_percentage_then_pinned_strip_string"}
        module.validate_comparator_result({"comparator_provenance": provenance}, expected)
        for key, value in (("include_percentage", True), ("loaded_modules", {"utils.grader": "/unrelated/site-packages/utils/grader.py"}),
            ("runtime_versions", {"sympy": "different"}), ("internal_timeout", True), ("timeout_owner", "unbounded_queue")):
            changed = deepcopy(provenance)
            changed[key] = value
            with self.assertRaises(ValueError):
                module.validate_comparator_result({"comparator_provenance": changed}, expected)


if __name__ == "__main__":
    unittest.main()
