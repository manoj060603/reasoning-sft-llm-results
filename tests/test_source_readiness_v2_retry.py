"""Versioned v2 fixture repair; execution belongs only to new native CPU QA.

No model, downloaded dataset, confirmation outcome, or real calibration record
is loaded by these fixtures. Native execution is a separate admission decision.
The old 24 test methods and assertions are retained with retry-controller identity
changes and one relative-plan fixture correction; four new tests bind the retry
history. This does not relabel the frozen 472-pass/two-error native attempt.
"""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch

from extension import source_evaluation as v1
from extension import source_readiness_v2_retry as module
from extension.cpu_feasibility import value_digest
from extension.labeled_feasibility import CRITERION, source_record
from extension.preserving_data import text_sha256


class Tokenizer:
    eos_token_id = 151645
    chat_template = "v2-readiness-fixture-envelope"

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
        raw = {"problem": "Compute fixture " + str(index) + "+1.", "level": "Level 1", "type": "Algebra",
            "solution": r"A public synthetic proof. \boxed{" + str(index + 1) + "}"}
        row, rejected = source_record("math", raw, {"source": "v2-math", "row": index})
        assert rejected is None
        row["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [row["id"]]})
        rows.append(row)
    for index in range(16):
        raw = {"id": "arc-v2-" + str(index), "question": "Select synthetic fixture " + str(index),
            "choices": {"label": ["2", "A", "1"], "text": ["correct fixture", "second fixture", "other fixture"]},
            "answerKey": "2"}
        row, rejected = source_record("arc_challenge", raw, {"source": "v2-arc", "row": index})
        assert rejected is None
        row["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [row["id"]]})
        rows.append(row)
    return sorted(rows, key=lambda row: row["id"])


def identity(tokenizer):
    return {"files": {"fixture.json": "a" * 64}, "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "eos_token_id": tokenizer.eos_token_id}


def expected_comparator():
    return {"loaded_modules": {name: "/fixture/" + name.replace(".", "/") + ".py"
            for name in module.source.MODULE_NAMES},
        "source_sha256": {name: "a" * 64 for name in module.source.MODULE_NAMES},
        "runtime_versions": {name: "fixture-1" for name in module.source.PACKAGE_NAMES}}


def proof(expected=None):
    return {**deepcopy(expected or expected_comparator()), "include_percentage": False, "is_close": True,
        "internal_timeout": False, "timeout_owner": "external_two_phase_spawn_process",
        "worker_protocol": module.source.PROTOCOL, "setup_deadline_seconds": 10., "comparison_deadline_seconds": 5.,
        "normalization_order": "explicit_percentage_then_pinned_strip_string"}


def successful_grade(row, text, *, timeout_seconds=5, expected_comparator=None):
    assert timeout_seconds == 5
    expected = expected_comparator
    result = {"record_id": row["id"], "source_kind": row["source_kind"], "denominator": 1,
        "status": "graded", "correct": True, "rankable": True, "error": None,
        "response_sha256": text_sha256(text), "worker_protocol": module.source.PROTOCOL, "caller_must_abort": False}
    if row["source_kind"] == "math":
        assert expected is not None
        result["comparator_provenance"] = proof(expected)
        result["worker"] = {"status": "graded", "correct": True, "go_sent": True,
            "worker_protocol": module.source.PROTOCOL, "comparator_provenance": proof(expected),
            "cleanup": {"successful": True}, "timings": {"declared_setup_deadline_seconds": 10.,
                "declared_comparison_deadline_seconds": 5., "declared_cleanup_join_budget_seconds": .55,
                "declared_total_budget_seconds": 15.55, "setup_elapsed_seconds": .2,
                "comparison_elapsed_seconds": .1, "total_elapsed_including_cleanup_seconds": .3}}
    return result


def build(population=None, grader=successful_grade, expected=None, tokenizer=None):
    tokenizer = tokenizer or Tokenizer()
    return module.build_calibration_cache(records() if population is None else population, tokenizer, identity(tokenizer),
        expected_comparator=expected_comparator() if expected is None else expected, grader=grader)


def plans():
    root = Path(module.__file__).resolve().parents[1]
    return (root, json.loads((root / module.PLAN).read_text()),
        json.loads((root / module.LEGACY_PLAN["path"]).read_text()),
        json.loads((root / module.PROTOCOL_PIN["path"]).read_text()))


class PreparationFixture:
    """Tiny orchestration seam; pure tests separately exercise real file pins.

    The synthetic source inventory and prior files intentionally replace the
    full historical evidence graph. This fixture establishes control flow and
    output qualification, never real cohort, code, model, or GPU admission.
    """
    def __init__(self, temporary):
        _, self.plan, self.old, self.protocol = plans()
        self.root = Path(temporary)
        self.compute = {"job_id": "880", "node": "owl123", "gpu_requested_or_allocated": False}
        self.output = self.root / "extension/runs/source-readiness-v2-retry-880"
        self.output.mkdir(parents=True)
        self.module_file = self.root / "extension/source_readiness_v2_retry.py"
        self.module_file.write_text("# Synthetic control-flow source witness.\n")
        self.plan_path = self.root / module.PLAN
        self.plan_path.write_text(json.dumps(self.plan, sort_keys=True))
        self.sources = {"extension/source_readiness_v2_retry.py": {"sha256": module.digest(self.module_file),
            "bytes": self.module_file.stat().st_size}}
        self.admission = ({"binding_sha256": "b" * 64}, "c" * 64)
        patterns = list(module.joint.PATTERNS) + self.plan["native_qa"]["new_patterns"]
        counts = list(module.joint.COUNTS) + [40, 28]
        started = {"schema": module.QA_SCHEMA, "job_id": "880", "source_inventory": self.sources,
            "old465_scope_replayed": False, "old_readiness24_executed": False}
        (self.output / "started.json").write_text(json.dumps(started))
        qa = {"schema": module.QA_SCHEMA, "success": True, "job_id": "880", "node": "owl123",
            "gpu_disabled": True, "tests_run": sum(counts), "test_counts": dict(zip(patterns, counts)),
            "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
            "source_inventory": self.sources, "code_sha256": {key: value["sha256"] for key, value in self.sources.items()},
            "plan_sha256": module.digest(self.plan_path), "legacy_plan_sha256": module.LEGACY_PLAN["sha256"],
            "scientific_plan_sha256": module.SCIENTIFIC_PLAN["sha256"], "old_readiness24_executed": False,
            "worker_protocol_sha256": module.PROTOCOL_PIN["sha256"], "old465_scope_replayed": False,
            "admission_binding_sha256": self.admission[0]["binding_sha256"], "compute_proof": self.compute,
            "started_sha256": module.digest(self.output / "started.json")}
        (self.output / "unit_qa.json").write_text(json.dumps(qa))
        self.population = records()

    def run(self, grader=successful_grade, *, ending=None, source_change=False):
        fixture = self
        class MetadataEvidence:
            def __init__(self):
                self.files = {}

            def check(self, path, checksum=None):
                path = Path(path)
                actual = module.digest(path) if path.is_file() else checksum or "a" * 64
                if path.is_file() and checksum is not None and actual != checksum:
                    raise ValueError("Synthetic metadata checksum differs")
                self.files[str(path)] = {"sha256": actual, "bytes": path.stat().st_size if path.is_file() else 0}
                return actual

            def read(self, path, checksum=None):
                self.check(path, checksum)
                if Path(path).is_file():
                    return json.loads(Path(path).read_text())
                if str(path).endswith(fixture.old["prior_cpu"]["complete.json"]["path"]):
                    return {"complete": True, "cpu_only": True, "job_id": "1224497", "input_snapshot_unchanged": True}
                raise AssertionError("Unexpected synthetic metadata read: " + str(path))

            def unchanged(self):
                return None
        inventory_calls = 0
        def inventory(root):
            nonlocal inventory_calls
            inventory_calls += 1
            result = deepcopy(self.sources)
            if source_change and inventory_calls > 1:
                result[next(iter(result))]["sha256"] = "0" * 64
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "__file__", str(self.module_file)))
            stack.enter_context(patch.object(module, "require_cpu_allocation",
                side_effect=[self.compute, self.compute if ending is None else ending]))
            stack.enter_context(patch.object(module.legacy.native, "_host_resources"))
            stack.enter_context(patch.object(module, "deadline_guard"))
            stack.enter_context(patch.object(module, "Evidence", MetadataEvidence))
            stack.enter_context(patch.object(module, "_read_plans", return_value=(self.plan, self.old, self.protocol)))
            stack.enter_context(patch.object(module, "source_inventory", side_effect=inventory))
            stack.enter_context(patch.object(module, "current_admission", return_value=self.admission))
            stack.enter_context(patch.object(module.importlib.metadata, "version", side_effect=self.old["runtime_versions"].__getitem__))
            seal = {"splits": {"calibration": {"ids_sha256": value_digest([row["id"] for row in self.population])}}}
            stack.enter_context(patch.object(module.legacy.prepared_source, "load_development",
                return_value=(seal, None, None, {"synthetic_metadata_only": True})))
            stack.enter_context(patch.object(module.legacy.prepared_source, "_jsonl", return_value=self.population))
            stack.enter_context(patch.object(module.legacy.prepared_source, "_validate_base", return_value=Path("/fixture/base")))
            stack.enter_context(patch.object(module.legacy.prepared_source, "_load_tokenizer", return_value=Tokenizer()))
            stack.enter_context(patch.object(module.source, "grade_response", side_effect=grader))
            return module.prepare(self.root, self.plan_path.relative_to(self.root).as_posix(), self.output, self.output / "unit_qa.json")


class CalibrationPreparationTests(unittest.TestCase):
    def test_exact_sorted_32_rows_prompts_choices_seeds_and_sample_indices(self):
        population, tokenizer = records(), Tokenizer()
        before = deepcopy(population)
        cache, prechecks = build(population, tokenizer=tokenizer)
        self.assertEqual((len(cache), len(prechecks)), (32, 32))
        self.assertEqual([row["record"] for row in cache], population)
        self.assertEqual([row["record_id"] for row in prechecks], [row["id"] for row in population])
        for row in cache:
            self.assertEqual(row["sample_indices"], list(range(8)))
            self.assertEqual(row["request_seed"], (20261021 + int(row["record"]["id"][:8], 16)) % 2 ** 31)
            self.assertEqual(row["prompt"]["prompt_token_ids"], tokenizer.encode(row["prompt"]["rendered_prompt"]))
            self.assertFalse(row["prompt"]["provenance"]["gold_fields_used"])
            self.assertFalse(row["prompt"]["provenance"]["truncation_performed"])
            if row["record"]["source_kind"] == "arc_challenge":
                prompt = row["prompt"]["messages"][1]["content"]
                self.assertLess(prompt.index("2. correct fixture"), prompt.index("A. second fixture"))
                self.assertLess(prompt.index("A. second fixture"), prompt.index("1. other fixture"))
        self.assertEqual(population, before)

    def test_missing_extra_reordered_duplicate_and_shared_group_fail_without_grade(self):
        population = records()
        shared_group = deepcopy(population)
        shared_group[1]["group_id"] = shared_group[0]["group_id"]
        for changed in (population[:-1], population + [population[0]], list(reversed(population)),
                        [population[0]] + population[:-1], shared_group):
            with patch.object(module.source, "grade_response") as grader, self.assertRaises(ValueError):
                module.build_calibration_cache(changed, Tokenizer(), identity(Tokenizer()),
                    expected_comparator=expected_comparator())
            grader.assert_not_called()

    def test_wrong_domain_mix_is_not_replaced_by_additional_math_rows(self):
        population = records()
        index = next(index for index, row in enumerate(population) if row["source_kind"] == "arc_challenge")
        raw = {"problem": "Seventeenth math fixture", "level": "Level 1", "type": "Algebra", "solution": r"\boxed{1}"}
        row, rejected = source_record("math", raw, {"source": "additional", "row": 0})
        self.assertIsNone(rejected)
        row["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [row["id"]]})
        population[index] = row
        population.sort(key=lambda row: row["id"])
        with self.assertRaises(ValueError):
            build(population)

    def test_gold_mutation_preserves_questions_choices_and_native_requests(self):
        tokenizer = Tokenizer()
        for row in records():
            before = deepcopy(row)
            mutated = module.gold_mutation(row)
            self.assertEqual(row, before)
            self.assertEqual((row["id"], row["question"]), (mutated["id"], mutated["question"]))
            self.assertNotEqual(row["label"], mutated["label"])
            self.assertEqual(module.source.serialize_prompt(row, tokenizer, identity(tokenizer)),
                module.source.serialize_prompt(mutated, tokenizer, identity(tokenizer)))
            if row["source_kind"] == "arc_challenge":
                self.assertEqual(row["original_record"]["choices"], mutated["original_record"]["choices"])

    def test_gold_dependent_serialization_is_detected_before_self_comparison(self):
        original = v1.serialize_prompt
        calls = []
        def leaking(row, *args, **kwargs):
            value = original(row, *args, **kwargs)
            value["rendered_prompt"] += str(row["label"])
            return value
        with patch.object(v1, "serialize_prompt", side_effect=leaking), self.assertRaisesRegex(ValueError, "Changing gold"):
            build(grader=lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])

    def test_unsupported_gold_stops_before_prompt_grading_and_refill(self):
        with patch.object(v1, "validate_gold", return_value={"supported": False}), \
                patch.object(v1, "serialize_prompt") as serializer, patch.object(module.source, "grade_response") as grader:
            with self.assertRaisesRegex(ValueError, "no refill"):
                module.build_calibration_cache(records(), Tokenizer(), identity(Tokenizer()),
                    expected_comparator=expected_comparator())
        serializer.assert_not_called()
        grader.assert_not_called()

    def test_overlong_native_prompt_stops_without_truncation_or_refill(self):
        population = records()
        index = next(index for index, row in enumerate(population) if row["source_kind"] == "math")
        old = population[index]
        raw = deepcopy(old["original_record"])
        raw["problem"] = "long fixture " * 8192
        changed, rejected = source_record("math", raw, old["source"])
        self.assertIsNone(rejected)
        changed["group_id"] = value_digest({"criterion": CRITERION, "member_ids": [changed["id"]]})
        population[index] = changed
        population.sort(key=lambda row: row["id"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build(population)

    def test_v2_wrapper_never_calls_legacy_timeout_owner_validator(self):
        from extension import source_readiness as legacy
        with patch.object(legacy, "validate_comparator_result", side_effect=AssertionError("v1 owner reached")):
            cache, prechecks = build()
        self.assertEqual((len(cache), len(prechecks)), (32, 32))
        self.assertTrue(all(row["self_comparison"]["worker_protocol"] == module.source.PROTOCOL for row in prechecks))

    def test_expected_five_module_identity_is_required_before_any_self_grade(self):
        for expected in ({}, {"loaded_modules": {}}, {**expected_comparator(), "source_sha256": {}}):
            calls = []
            with self.assertRaises(ValueError):
                build(expected=expected, grader=lambda *args, **kwargs: calls.append(args))
            self.assertEqual(calls, [])

    def test_each_grade_binds_the_exact_record_protocol_and_single_denominator(self):
        mutations = ({"record_id": "0" * 64}, {"source_kind": "other"}, {"denominator": 0},
            {"worker_protocol": "v1"}, {"caller_must_abort": True})
        for mutation in mutations:
            calls = []
            def corrupt(row, text, **kwargs):
                calls.append(row["id"])
                return {**successful_grade(row, text, **kwargs), **mutation}
            with self.assertRaises((ValueError, module.source.SourceGraderCleanupError)):
                build(grader=corrupt)
            self.assertEqual(calls, [records()[0]["id"]])

    def test_first_timeout_stops_ordered_processing_without_partial_return_or_refill(self):
        population, calls = records(), []
        target = 3
        def timed_out(row, text, **kwargs):
            calls.append(row["id"])
            result = successful_grade(row, text, **kwargs)
            if len(calls) == target:
                result.update(status="comparison_timeout", correct=False, rankable=False,
                    error={"type": "ComparisonDeadlineExceeded", "phase": "comparison"})
            return result
        with self.assertRaises(ValueError):
            build(population, grader=timed_out)
        self.assertEqual(calls, [row["id"] for row in population[:target]])

    def test_cleanup_error_propagates_and_never_continues_with_next_record(self):
        calls = []
        def cleanup_error(row, text, **kwargs):
            calls.append(row["id"])
            failure = successful_grade(row, text, **kwargs)
            failure.update(status="cleanup_error", correct=False, rankable=False, caller_must_abort=True,
                error={"type": "BoundedWorkerCleanupFailed", "phase": "cleanup"})
            raise module.source.SourceGraderCleanupError(failure)
        with self.assertRaises(module.source.SourceGraderCleanupError) as caught:
            build(grader=cleanup_error)
        self.assertEqual(calls, [records()[0]["id"]])
        self.assertEqual(caught.exception.result["denominator"], 1)
        self.assertFalse(caught.exception.result["rankable"])


class ComparatorBindingTests(unittest.TestCase):
    def successful_math(self):
        row = next(row for row in records() if row["source_kind"] == "math")
        return successful_grade(row, r"\boxed{1}", expected_comparator=expected_comparator())

    def test_success_requires_pinned_v2_ready_and_complete_two_phase_receipt(self):
        result = self.successful_math()
        before = deepcopy(result)
        module.validate_comparator_result(result, expected_comparator())
        self.assertEqual(result, before)
        self.assertEqual(result["comparator_provenance"]["timeout_owner"], "external_two_phase_spawn_process")

    def test_any_loaded_path_source_hash_version_or_v1_owner_change_is_rejected(self):
        for field in ("loaded_modules", "source_sha256", "runtime_versions"):
            result = self.successful_math()
            key = next(iter(result["comparator_provenance"][field]))
            result["comparator_provenance"][field][key] = "unexpected"
            with self.assertRaises(ValueError):
                module.validate_comparator_result(result, expected_comparator())
        result = self.successful_math()
        result["comparator_provenance"]["timeout_owner"] = "external_spawn_process"
        with self.assertRaises(ValueError):
            module.validate_comparator_result(result, expected_comparator())

    def test_worker_proof_go_cleanup_and_correctness_must_match_top_receipt(self):
        for mutation in ({"status": "comparison_timeout"}, {"correct": False}, {"go_sent": False},
                         {"worker_protocol": "v1"}, {"cleanup": {"successful": False}},
                         {"comparator_provenance": {}}):
            result = self.successful_math()
            result["worker"].update(mutation)
            with self.assertRaises(ValueError):
                module.validate_comparator_result(result, expected_comparator())

    def test_every_declared_budget_and_finite_elapsed_field_is_bound(self):
        for field in ("declared_setup_deadline_seconds", "declared_comparison_deadline_seconds",
                      "declared_cleanup_join_budget_seconds", "declared_total_budget_seconds"):
            result = self.successful_math()
            result["worker"]["timings"][field] = 999.
            with self.assertRaises(ValueError):
                module.validate_comparator_result(result, expected_comparator())
        for field in ("setup_elapsed_seconds", "comparison_elapsed_seconds", "total_elapsed_including_cleanup_seconds"):
            for value in (-1, True, float("nan"), float("inf"), "0"):
                result = self.successful_math()
                result["worker"]["timings"][field] = value
                with self.assertRaises(ValueError):
                    module.validate_comparator_result(result, expected_comparator())

    def test_scalar_self_check_false_or_nonrankable_cannot_be_promoted(self):
        for mutation in ({"correct": False}, {"rankable": False}, {"status": "setup_error", "correct": False,
                "rankable": False, "error": {"type": "Setup"}}):
            def bad(row, text, **kwargs):
                return {**successful_grade(row, text, **kwargs), **mutation}
            with self.assertRaises(ValueError):
                build(grader=bad)


class PlanAndOutputTests(unittest.TestCase):
    def test_exact_v2_plan_reuses_immutable_cohort_sampling_and_separate_protocol(self):
        root, plan, old, protocol = plans()
        self.assertIs(module.validate_plan(plan), plan)
        self.assertIs(module.validate_protocol(protocol, old), protocol)
        self.assertEqual(module.digest(root / plan["legacy_plan"]["path"]), plan["legacy_plan"]["sha256"])
        self.assertEqual(module.digest(root / plan["worker_protocol"]["path"]), plan["worker_protocol"]["sha256"])
        self.assertEqual(plan["generation_profile"], "unchanged_legacy_plan.source_generation_profile")
        self.assertEqual(old["source_generation_profile"]["n"], 8)
        self.assertEqual(old["source_generation_profile"]["max_new_tokens"], 8192)
        self.assertEqual(old["source_generation_profile"]["max_model_len"], 16384)
        self.assertEqual(plan["native_qa"]["legacy_tests"] + plan["native_qa"]["v2_helper_tests"] +
            plan["native_qa"]["new_readiness_tests"], 478)
        self.assertFalse(plan["native_qa"]["old465_scope_replayed"])
        self.assertEqual(plan["native_qa"]["source_inventory_files"], 91)
        self.assertFalse(plan["outputs"]["raw_labels_prompts_or_ids_public"])

    def test_cpu_resource_scope_reference_calibration_and_qa_mutations_fail_closed(self):
        _, plan, _, _ = plans()
        mutations = (lambda value: value.update(training=True), lambda value: value.update(model_inference=True),
            lambda value: value.update(automatic_downstream=True), lambda value: value.update(reserve_seconds=0),
            lambda value: value.update(wall_time_seconds=3600), lambda value: value.update(deadline_utc="2026-10-29T03:59:00+00:00"),
            lambda value: value["resources"].update(gpus=1), lambda value: value["resources"].update(cpus=1),
            lambda value: value["calibration"].update(split="confirmation"),
            lambda value: value["calibration"].update(record_count=31),
            lambda value: value["calibration"].update(drop_refill_or_replace=True),
            lambda value: value["legacy_plan"].update(sha256="0" * 64),
            lambda value: value["worker_protocol"].update(sha256="0" * 64),
            lambda value: value["predecessor_joint_qa"].update(sha256="0" * 64),
            lambda value: value["new_frozen_sources"].update({"extension/source_evaluation_v2.py": "0" * 64}),
            lambda value: value.update(generation_profile="rewritten"),
            lambda value: value.update(source_generated_answers_used=True),
            lambda value: value.update(confirmation_outputs_opened=True),
            lambda value: value["native_qa"].update(legacy_tests=465),
            lambda value: value["native_qa"].update(v2_helper_tests=39),
            lambda value: value["native_qa"].update(new_readiness_tests=0),
            lambda value: value["native_qa"].update(source_inventory_files=90),
            lambda value: value["native_qa"].update(old465_scope_replayed=True))
        for mutate in mutations:
            changed = deepcopy(plan)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_protocol_phase_rules_comparator_and_runtime_cannot_be_rewritten(self):
        _, _, old, protocol = plans()
        for key, value in (("setup_seconds", 5), ("work_seconds", 10), ("cleanup_seconds", 1),
                           ("maximum_total_seconds", 16), ("go_messages", 2), ("worker_reuse", True),
                           ("automatic_retry", True), ("late_results_accepted", True),
                           ("cleanup_error_blocks_further_grading_and_ranking", False)):
            changed = deepcopy(protocol)
            changed["worker_protocol"][key] = value
            with self.assertRaises(ValueError):
                module.validate_protocol(changed, old)
        for mutate in (lambda value: value["runtime_versions"].update(sympy="different"),
                       lambda value: value["comparator_source"]["grader.py"].update(sha256="0" * 64),
                       lambda value: value.update(historical_and_final_benchmark_grading="changed")):
            changed = deepcopy(protocol)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.validate_protocol(changed, old)

    def test_expected_ready_identity_uses_all_five_exact_root_paths_hashes_and_versions(self):
        root, _, old, _ = plans()
        expected = module.comparator_identity(root, old)
        module.source._expected_identity(expected)
        self.assertEqual(set(expected["loaded_modules"]), module.source.MODULE_NAMES)
        self.assertEqual(set(expected["runtime_versions"]), module.source.PACKAGE_NAMES)
        for name, spec in old["grader_source"].items():
            module_name = "utils." + Path(name).stem
            self.assertEqual(expected["loaded_modules"][module_name], str((root / spec["path"]).resolve()))
            self.assertEqual(expected["source_sha256"][module_name], spec["sha256"])
            self.assertEqual(hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest(), spec["sha256"])
        self.assertEqual(expected["runtime_versions"], {key: old["runtime_versions"][key] for key in module.source.PACKAGE_NAMES})

    def test_actual_metadata_file_pins_refuse_modified_legacy_protocol_or_qa_receipt(self):
        root, plan, old, protocol = plans()
        pins = [module.LEGACY_PLAN, module.PROTOCOL_PIN, module.PRIOR_QA, module.SCIENTIFIC_PLAN]
        pins += [{"path": module.FAILED_DIRECTORY + "/" + name, "sha256": checksum}
            for name, checksum in module.FAILURE_PINS.items()]
        for altered in pins:
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary)
                for spec in pins:
                    path = target / spec["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((root / spec["path"]).read_bytes())
                plan_path = target / module.PLAN
                plan_path.write_text(json.dumps(plan))
                evidence = module.Evidence()
                parsed = module._read_plans(target, plan_path, evidence)
                self.assertEqual(parsed, (plan, old, protocol))
                path = target / altered["path"]
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    module._read_plans(target, plan_path, module.Evidence())

    def test_success_cache_is_private_source_only_and_does_not_admit_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PreparationFixture(temporary)
            complete = fixture.run()
            self.assertEqual((complete["record_count"], complete["gold_self_comparisons_passed"], complete["tests_run"]), (32, 32, 478))
            self.assertEqual(complete["gpu_seconds"], 0)
            for key in ("generation", "training_pilot", "automatic_downstream", "confirmation_outputs_opened",
                        "old465_scope_replayed", "old_five_second_end_to_end_contract_validated", "old_readiness24_executed"):
                self.assertIs(complete[key], False, key)
            self.assertTrue(complete["raw_cache_private"])
            cache = json.loads((fixture.output / "calibration_prompt_cache.json").read_text())
            self.assertEqual(cache["schema"], module.CACHE_SCHEMA)
            self.assertTrue(cache["raw_records_private"])
            self.assertFalse(cache["generation_performed"] or cache["training_performed"])
            self.assertEqual(cache["source_generation_profile"], fixture.old["source_generation_profile"])
            self.assertEqual([row["record"]["id"] for row in cache["records"]], [row["id"] for row in fixture.population])
            public = json.dumps(complete)
            for row in fixture.population:
                self.assertNotIn(row["id"], public)
                self.assertNotIn(row["question"], public)
            self.assertFalse((fixture.output / "failure.json").exists())

    def test_error_or_changed_inputs_preserve_failure_without_any_partial_cache(self):
        def timeout(row, text, **kwargs):
            value = successful_grade(row, text, **kwargs)
            value.update(status="comparison_timeout", correct=False, rankable=False,
                error={"type": "ComparisonDeadlineExceeded", "phase": "comparison"})
            return value
        def cleanup(row, text, **kwargs):
            value = successful_grade(row, text, **kwargs)
            value.update(status="cleanup_error", correct=False, rankable=False, caller_must_abort=True,
                error={"type": "BoundedWorkerCleanupFailed", "phase": "cleanup"})
            raise module.source.SourceGraderCleanupError(value)
        cases = [(timeout, {}), (cleanup, {}), (successful_grade, {"source_change": True}),
                 (successful_grade, {"ending": {"job_id": "881", "node": "owl123"}})]
        for grader, options in cases:
            with tempfile.TemporaryDirectory() as temporary:
                fixture = PreparationFixture(temporary)
                with self.assertRaises((ValueError, module.source.SourceGraderCleanupError)):
                    fixture.run(grader, **options)
                self.assertFalse((fixture.output / "complete.json").exists())
                self.assertFalse((fixture.output / "calibration_prompt_cache.json").exists())
                self.assertFalse((fixture.output / "calibration_gold_prechecks.json").exists())
                failure = json.loads((fixture.output / "failure.json").read_text())
                self.assertFalse(failure["complete"] or failure["calibration_prepared"] or failure["automatic_downstream"])
                self.assertTrue(failure["no_drop_refill_retry"])
                private = json.loads((fixture.output / "calibration_failure_private.json").read_text())
                self.assertTrue(private["raw_grades_private"])
                self.assertGreaterEqual(len(private["grading_results"]), 1)
                public = json.dumps(failure)
                for row in fixture.population:
                    self.assertNotIn(row["id"], public)
                    self.assertNotIn(row["question"], public)
        secret = "PRIVATE-GOLD-PROMPT:" + records()[0]["id"]
        with patch.object(module, "prepare", side_effect=ValueError(secret)):
            with self.assertRaises(RuntimeError) as caught:
                module.main(["--root", ".", "--output", "private-fixture", "--qa-receipt", "private-qa.json"])
        self.assertEqual(str(caught.exception), "Source readiness v2 retry failed. See private failure evidence.")
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertIsNone(caught.exception.__cause__)
        rendered = "".join(traceback.format_exception(type(caught.exception), caught.exception, caught.exception.__traceback__))
        self.assertNotIn(secret, rendered)


class VersionedRetryBoundaryTests(unittest.TestCase):
    def test_new_role_keeps_exact_failed_attempt_and_original_scientific_plan(self):
        root, plan, _, _ = plans()
        self.assertEqual(module.ROLE, "source_readiness_v2_20261021_v2")
        self.assertEqual(module.FAILED_ROLE, "source_readiness_v2_20261021_v1")
        self.assertEqual(module.FAILED_JOB, "1225820")
        self.assertEqual(plan["scientific_plan"], module.SCIENTIFIC_PLAN)
        self.assertEqual(module.digest(root / module.SCIENTIFIC_PLAN["path"]),
            "40479982fe3b43e27f51ec27f28a2cf39f76a464ceffdc576e1f38268f0a6a3b")
        directory = root / module.FAILED_DIRECTORY
        qa = json.loads((directory / "unit_qa_failure.json").read_text())
        accounting = json.loads((directory / "accounting.json").read_text())
        self.assertEqual((qa["tests_run"], qa["failures"], qa["errors"], qa["skipped"]), (474, 0, 2, 0))
        self.assertEqual(qa["test_counts"]["test_source_evaluation_v2.py"], 40)
        self.assertEqual(qa["test_counts"]["test_source_readiness_v2.py"], 24)
        self.assertEqual((accounting["job_id"], accounting["state"], accounting["exit_code"],
            accounting["elapsed_seconds"], accounting["gpu_seconds"]), ("1225820", "FAILED", "1:0", 71, 0))
        self.assertNotIn("test_source_readiness_v2.py", plan["native_qa"]["new_patterns"])
        self.assertIn("test_source_readiness_v2_retry.py", plan["native_qa"]["new_patterns"])

    def test_retry_parent_failure_history_and_fixture_correction_are_not_mutable_choices(self):
        _, plan, _, _ = plans()
        for field in ("retry_of", "scientific_plan", "failure_evidence", "prior_entries_sha256", "fixture_correction"):
            changed = deepcopy(plan)
            changed[field] = "unreviewed-replacement"
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_retry_inventory_preserves_all_failed_86_bytes_and_adds_exactly_five_files(self):
        root, _, _, _ = plans()
        before = module.parent.source_inventory(root)
        current = module.source_inventory(root)
        self.assertEqual((len(before), len(current)), (86, 91))
        self.assertEqual({key: current[key] for key in before}, before)
        self.assertEqual(set(current) - set(before), {"extension/source_readiness_v2_retry.py",
            "extension/source_readiness_v2_retry_plan.json", "extension/source_readiness_v2_retry_qa.py",
            "extension/source_readiness_v2_retry.slurm", "tests/test_source_readiness_v2_retry.py"})
        self.assertEqual(current["tests/test_source_readiness_v2.py"]["sha256"],
            "5971cfc15b30255e1b7528ad276de92293510d30980ad91935c6c730b63c008a")
        failed = json.loads((root / module.FAILED_DIRECTORY / "unit_qa_failure.json").read_text())
        self.assertEqual(failed["source_inventory"], before)

    def test_current_retry_admission_requires_every_prior_entry_and_its_own_binding(self):
        root, _, _, _ = plans()
        prior = json.loads((root / module.FAILED_DIRECTORY / "job_ledger_snapshot.json").read_text())["jobs"]
        self.assertEqual(len(prior), 15)
        self.assertEqual(value_digest(prior), module.PRIOR_ENTRIES_SHA256)
        sources = {"extension/source_readiness_v2_retry.py": {"sha256": "a" * 64, "bytes": 1}}
        checksum = "d" * 64
        binding = {"role": module.ROLE, "source_sha256": sources, "plan_sha256": checksum,
            "scientific_plan_sha256": module.SCIENTIFIC_PLAN["sha256"],
            "legacy_plan_sha256": module.LEGACY_PLAN["sha256"], "worker_protocol_sha256": module.PROTOCOL_PIN["sha256"],
            "prior_entries_sha256": value_digest(prior), "wall_time_seconds": 7200,
            "closeout_reserve_seconds": 172800, "automatic_downstream": False}
        entry = {"status": "submitted", "job_id": "990", "requested_gpu_seconds": 0, "requested_gpu_hours": 0,
            "source_sha256": sources, "binding": binding, "binding_sha256": value_digest(binding)}
        ledger = {"schema": module.joint.previous.SCHEMA, "gpu_hour_cap": None,
            "historical_ledger_unchanged": "deadline/job_ledger.json", "jobs": {**deepcopy(prior), module.ROLE: entry}}
        with patch.object(module, "read_json", return_value=ledger):
            self.assertEqual(module.current_admission(root, {"job_id": "990"}, sources, checksum),
                (entry, module.PRIOR_ENTRIES_SHA256))
        variants = []
        changed = deepcopy(ledger)
        changed["jobs"].pop(module.FAILED_ROLE)
        variants.append(changed)
        changed = deepcopy(ledger)
        changed["jobs"][module.FAILED_ROLE]["job_id"] = "991"
        changed_prior = {key: value for key, value in changed["jobs"].items() if key != module.ROLE}
        changed["jobs"][module.ROLE]["binding"]["prior_entries_sha256"] = value_digest(changed_prior)
        changed["jobs"][module.ROLE]["binding_sha256"] = value_digest(changed["jobs"][module.ROLE]["binding"])
        variants.append(changed)
        for change in ({"status": "uncertain"}, {"job_id": module.FAILED_JOB}, {"requested_gpu_seconds": 1}):
            changed = deepcopy(ledger)
            changed["jobs"][module.ROLE].update(change)
            variants.append(changed)
        changed = deepcopy(ledger)
        changed["jobs"][module.ROLE]["binding"]["scientific_plan_sha256"] = "0" * 64
        changed["jobs"][module.ROLE]["binding_sha256"] = value_digest(changed["jobs"][module.ROLE]["binding"])
        variants.append(changed)
        for value in variants:
            with patch.object(module, "read_json", return_value=value), self.assertRaises(ValueError):
                module.current_admission(root, {"job_id": "990"}, sources, checksum)


if __name__ == "__main__":
    unittest.main()
