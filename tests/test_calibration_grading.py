"""32 unexecuted synthetic grading oracles; no real cache/outputs/workers.

Fixture file pins differ explicitly from native pins. Every grading call uses
an injected fixture grader; no SymPy worker, model, tokenizer download, scheduler
or production generation is started. Native CPU QA/admission is still required.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from extension import calibration_grading as module
from extension import source_evaluation_v2 as source
from extension import source_generation_v3 as generation
from extension.labeled_feasibility import source_record


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


class Tokenizer:
    eos_token_id = 151645
    chat_template = "independent-calibration-grading-fixture"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        text = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
        return self.encode(text) if tokenize else text


def comparator():
    return {"loaded_modules": {name: "/fixture/eval/utils/" + name.split(".")[-1] + ".py" for name in module.COMPARATOR_SOURCES},
        "source_sha256": dict(module.COMPARATOR_SOURCES), "runtime_versions": dict(module.COMPARATOR_VERSIONS)}


def worker_proof(expected):
    return {**deepcopy(expected), "include_percentage": False, "is_close": True, "internal_timeout": False,
        "timeout_owner": "external_two_phase_spawn_process", "worker_protocol": source.PROTOCOL,
        "setup_deadline_seconds": 10., "comparison_deadline_seconds": 5.,
        "normalization_order": "explicit_percentage_then_pinned_strip_string"}


def fixture_grade(row, text, *, timeout_seconds=5, expected_comparator=None):
    """Independent exact integer/percentage fixture comparison, never symbolic truth."""
    assert timeout_seconds == 5
    if row["source_kind"] == "arc_challenge":
        return source.grade_response(row, text, timeout_seconds=5, expected_comparator=expected_comparator)
    extraction, gold = source.extract_final_box(text), source.validate_gold(row)
    result = {"record_id": row["id"], "source_kind": "math", "denominator": 1, "correct": False, "rankable": True,
        "error": None, "status": extraction["status"], "extraction": extraction,
        "response_sha256": module.file_sha256(text.encode()), "worker_protocol": source.PROTOCOL, "caller_must_abort": False}
    if not gold["supported"]:
        result.update(status="unsupported_gold", rankable=False, error={"type": "UnsupportedGold"})
    elif extraction["status"] == "ok":
        normalized = source.normalize_percentage(extraction["answer"])
        if not normalized["supported"]:
            result["status"] = "unsupported_prediction_percentage"
        else:
            correct = normalized["normalized"] == gold["percent_normalized_gold"]
            proof = worker_proof(expected_comparator)
            result.update(status="graded", correct=correct, comparator_provenance=proof,
                worker={"status": "graded", "correct": correct, "go_sent": True, "worker_protocol": source.PROTOCOL,
                    "comparator_provenance": deepcopy(proof), "cleanup": {"successful": True},
                    "timings": {"declared_setup_deadline_seconds": 10., "declared_comparison_deadline_seconds": 5.,
                        "declared_cleanup_join_budget_seconds": .55, "declared_total_budget_seconds": 15.55,
                        "setup_elapsed_seconds": .2, "comparison_elapsed_seconds": .1, "total_elapsed_including_cleanup_seconds": .3}})
    return result


class Bundle:
    def __init__(self, response=None):
        tokenizer = Tokenizer()
        identity = {"repository": generation.native.REPOSITORY, "revision": generation.native.REVISION,
            "files": dict(generation.native.TOKENIZER_FILES), "eos_token_id": 151645,
            "chat_template_sha256": module.file_sha256(tokenizer.chat_template.encode())}
        records = []
        for index in range(16):
            raw = {"problem": "Synthetic private MATH " + str(index) + "+1?", "level": "Level 1", "type": "Algebra",
                "solution": r"Synthetic solution only. \boxed{" + str(index + 1) + "}"}
            row, rejected = source_record("math", raw, {"fixture": "math", "row": index})
            assert rejected is None
            row["group_id"] = module.value_sha256([row["id"]])
            records.append(row)
        for index in range(16):
            raw = {"id": "fixture-arc-" + str(index), "question": "Synthetic private ARC " + str(index) + "?",
                "choices": {"label": ["2", "A", "1"], "text": ["first option", "second option", "third option"]}, "answerKey": "2"}
            row, rejected = source_record("arc_challenge", raw, {"fixture": "arc", "row": index})
            assert rejected is None
            row["group_id"] = module.value_sha256([row["id"]])
            records.append(row)
        records.sort(key=lambda row: row["id"])
        prepared = [{"record": row, "prompt": source.serialize_prompt(row, tokenizer, identity),
            "request_seed": source.sample_seed(row["id"]), "sample_indices": list(range(8))} for row in records]
        self.cache = {"schema": generation.prepared_stage.CACHE_SCHEMA, "complete": True,
            "plan_sha256": generation.prepared_stage.SCIENTIFIC_PLAN["sha256"],
            "execution_plan_sha256": "7526ff1b21dd2a8a53c39f47c91981ffe91496cd5f1270b9cddf50974cf3c39b",
            "source_generation_profile": deepcopy(generation.PROFILE), "generation_performed": False,
            "training_performed": False, "raw_records_private": True,
            "model": {"repository": generation.native.REPOSITORY, "revision": generation.native.REVISION,
                "files": dict(generation.native.ORIGINAL_FILES), "path": "/fixture/pinned-original"},
            "tokenizer_identity": identity, "records": prepared,
            "calibration_ids_sha256": module.value_sha256([row["id"] for row in records])}
        proof = generation.validate_cache(self.cache)
        self.records = []
        for index, request in enumerate(generation.build_requests(self.cache)):
            row = records[index]
            samples = []
            for sample_index in range(8):
                gold = row["label"]["answer"] if row["source_kind"] == "math" else row["label"]["answer_key"]
                text = response(row, sample_index) if response else r"PRIVATE-RESPONSE-MARKER \boxed{" + (gold if sample_index < 4 else "999") + "}"
                tokens = [100 + sample_index, 151645]
                samples.append({"engine_output_position": sample_index, "engine_output_index": sample_index,
                    "sample_index": sample_index, "text": text, "token_ids": tokens, "output_tokens": 2,
                    "finish_reason": "stop", "stop_reason": None, "text_sha256": module.file_sha256(text.encode()),
                    "token_ids_sha256": module.value_sha256(tokens)})
            self.records.append({"schema": "extension-source-generation-record-v1", **request, "engine_request_id": "fixture-" + str(index),
                "finished": True, "samples": samples, "native_timing": {"arrival_time": 1.}, "native_metrics_type": "FixtureMetrics",
                "raw_engine_observation_sha256": module.value_sha256([index, "fixture"]), "grading_performed": False})
        sources = {generation.PLAN: {"sha256": "a" * 64, "bytes": 1},
            "extension/source_generation_v3.py": {"sha256": "b" * 64, "bytes": 1}, "extension/gpu_identity_v2.py": {"sha256": "c" * 64, "bytes": 1}}
        sources.update({"extension/fixture-source-" + str(index) + ".py": {"sha256": "d" * 64, "bytes": 1} for index in range(125)})
        self.complete = {"schema": generation.SCHEMA, "complete": True, "job_id": "fixture-only", "model_role": "original",
            "request_count": 32, "sample_count": 256, "cache_sha256": "updated-by-raw", "profile_sha256": generation.PROFILE_SHA256,
            "parent_complete_sha256": generation.PARENT_PINS["complete.json"], "source_inventory": sources,
            "model_identity_sha256": proof["model_identity_sha256"], "tokenizer_identity_sha256": proof["tokenizer_identity_sha256"],
            "plan_sha256": "a" * 64, "native_interface_established": False, "fixture_factory_injected": True,
            "input_snapshot_unchanged": True, "raw_requests_outputs_private": True, "engine_adapter": deepcopy(generation.ENGINE_ADAPTER),
            "grading": False, "ranking": False, "training": False, "automatic_retry": False, "automatic_downstream": False,
            "length_finishes": 0, "output_tokens": 512, "private_output_sha256": {}}

    def raw(self):
        cache = encoded(self.cache)
        records = {name: encoded(row) for name, row in zip(module.RECORD_NAMES, self.records)}
        self.complete["cache_sha256"] = module.file_sha256(cache)
        self.complete["private_output_sha256"] = {name: module.file_sha256(raw) for name, raw in records.items()}
        complete = encoded(self.complete)
        return cache, complete, records, {"cache_json_sha256": module.file_sha256(cache), "generation_complete_json_sha256": module.file_sha256(complete)}

    def validate(self):
        cache, complete, records, pins = self.raw()
        return module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)

    def grade(self, grader=fixture_grade):
        cache, complete, records, pins = self.raw()
        return module.grade_calibration(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator(), grader=grader, fixture=True)


class CalibrationGradingTests(unittest.TestCase):
    def test_pure_source_has_no_cli_admission_or_file_io(self):
        source_text = Path(module.__file__).read_text()
        self.assertNotIn("def main(", source_text)
        self.assertNotIn(".open(", source_text)
        self.assertNotIn("read_bytes(", source_text)
        self.assertNotIn("subprocess", source_text)
        self.assertEqual(len(module.RECORD_NAMES), 32)

    def test_valid_proof_distinguishes_real_file_bytes_and_canonical_value_hashes(self):
        bundle = Bundle(); context = bundle.validate()
        self.assertEqual(context["proof"]["sample_count"], 256)
        self.assertEqual(context["proof"]["record_count"], 32)
        self.assertNotEqual(context["proof"]["cache_file_sha256"], context["proof"]["cache_value_sha256"])
        self.assertTrue(context["proof"]["fixture_inputs"])
        self.assertFalse(context["proof"]["physical_paths_permissions_current_job_runtime_verified_here"])

    def test_fixture_file_pins_never_implicitly_admit_native_generation(self):
        cache, complete, records, pins = Bundle().raw()
        with self.assertRaises(module.CalibrationInputError):
            module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator())
        native = {"cache_json_sha256": module.CACHE_FILE_SHA256, "generation_complete_json_sha256": module.GENERATION_COMPLETE_FILE_SHA256}
        with self.assertRaises(module.CalibrationInputError):
            module.validate_inputs(cache, complete, records, expected_file_pins=native, expected_comparator=comparator())

    def test_fixture_and_native_interface_flags_are_explicit_and_strict(self):
        for key, value in (("native_interface_established", True), ("fixture_factory_injected", False), ("complete", False)):
            bundle = Bundle(); bundle.complete[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): bundle.validate()
        cache, complete, records, pins = Bundle().raw()
        with self.assertRaises(ValueError):
            module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=1)

    def test_duplicate_nonfinite_malformed_and_nonobject_json_inputs_fail_closed(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'[]', b'{broken', b'\xff'):
            bundle = Bundle(); _, complete, records, pins = bundle.raw(); pins["cache_json_sha256"] = module.file_sha256(raw)
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                module.validate_inputs(raw, complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)

    def test_input_bytes_and_size_bounds_are_enforced_before_grade_calls(self):
        cache, complete, records, pins = Bundle().raw()
        with self.assertRaises(ValueError):
            module.validate_inputs(cache.decode(), complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)
        for limit in ("MAX_CACHE_BYTES", "MAX_COMPLETE_BYTES", "MAX_RECORD_BYTES", "MAX_ALL_RECORD_BYTES"):
            with patch.object(module, limit, 1), self.subTest(limit=limit), self.assertRaises(ValueError):
                module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)

    def test_exact_record_file_set_and_explicit_pin_key_set_are_required(self):
        cache, complete, records, pins = Bundle().raw()
        for changed in ({key: value for key, value in records.items() if key != module.RECORD_NAMES[0]}, {**records, "record-32.private.json": b"{}"}):
            with self.assertRaises(ValueError): module.validate_inputs(cache, complete, changed, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)
        for changed in ({}, {**pins, "implicit": "a" * 64}):
            with self.assertRaises(ValueError): module.validate_inputs(cache, complete, records, expected_file_pins=changed, expected_comparator=comparator(), fixture=True)

    def test_even_whitespace_change_in_a_private_record_breaks_its_file_seal(self):
        cache, complete, records, pins = Bundle().raw(); records[module.RECORD_NAMES[0]] += b" "
        with self.assertRaises(ValueError): module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)

    def test_cache_or_complete_byte_tampering_cannot_self_repin(self):
        cache, complete, records, pins = Bundle().raw()
        for first, second in ((cache + b" ", complete), (cache, complete + b" ")):
            with self.assertRaises(ValueError): module.validate_inputs(first, second, records, expected_file_pins=pins, expected_comparator=comparator(), fixture=True)

    def test_cache_membership_order_domain_and_complete_population_are_fixed(self):
        for kind in ("missing", "extra", "order", "duplicate", "domain"):
            bundle = Bundle()
            if kind == "missing": bundle.cache["records"].pop()
            elif kind == "extra": bundle.cache["records"].append(deepcopy(bundle.cache["records"][0]))
            elif kind == "order": bundle.cache["records"].reverse()
            elif kind == "duplicate": bundle.cache["records"][1] = deepcopy(bundle.cache["records"][0])
            else: bundle.cache["records"][0]["record"]["source_kind"] = "other"
            with self.subTest(kind=kind), self.assertRaises(ValueError): bundle.validate()

    def test_source_plan_profile_snapshot_and_generation_scope_bindings_are_required(self):
        for key, value in (("source_inventory", {}), ("plan_sha256", "0" * 64), ("profile_sha256", "0" * 64),
                           ("input_snapshot_unchanged", False), ("grading", True), ("model_role", "advanced"), ("sample_count", 255)):
            bundle = Bundle(); bundle.complete[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): bundle.validate()

    def test_original_model_and_tokenizer_identities_are_bound_to_the_prepared_cache(self):
        for key in ("model_identity_sha256", "tokenizer_identity_sha256"):
            bundle = Bundle(); bundle.complete[key] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError): bundle.validate()
        bundle = Bundle(); bundle.cache["model"]["revision"] = "other"
        with self.assertRaises(ValueError): bundle.validate()

    def test_record_order_seed_prompt_and_loaded_identity_mutations_fail_before_grading(self):
        for key, value in (("record_id", "f" * 64), ("request_seed", 1), ("prompt_token_ids", [1]),
                           ("prompt_token_ids_sha256", "0" * 64), ("prompt_sha256", "0" * 64),
                           ("model_identity_sha256", "0" * 64), ("tokenizer_identity_sha256", "0" * 64)):
            bundle = Bundle(); bundle.records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): bundle.validate()

    def test_eight_sample_indices_must_be_complete_unique_ordered_plain_integers(self):
        for kind in ("missing", "extra", "duplicate", "order", "bool"):
            bundle = Bundle(); samples = bundle.records[0]["samples"]
            if kind == "missing": samples.pop()
            elif kind == "extra": samples.append(deepcopy(samples[0]))
            elif kind == "duplicate": samples[1]["sample_index"] = 0
            elif kind == "order": samples.reverse()
            else: samples[0]["sample_index"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError): bundle.validate()

    def test_opaque_request_ids_finished_state_and_record_schema_are_not_inferred(self):
        for kind in ("duplicate", "missing", "unfinished", "extra"):
            bundle = Bundle()
            if kind == "duplicate": bundle.records[1]["engine_request_id"] = bundle.records[0]["engine_request_id"]
            elif kind == "missing": bundle.records[0]["engine_request_id"] = ""
            elif kind == "unfinished": bundle.records[0]["finished"] = False
            else: bundle.records[0]["extra_inferred_profile"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): bundle.validate()

    def test_raw_text_tokens_hashes_native_eos_and_vocabulary_are_validated(self):
        for key, value in (("text", "changed"), ("token_ids", [False, 151645]), ("token_ids", [151936, 151645]),
                           ("text_sha256", "0" * 64), ("token_ids_sha256", "0" * 64), ("output_tokens", 3), ("stop_reason", 151643)):
            bundle = Bundle(); bundle.records[0]["samples"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): bundle.validate()
        bundle = Bundle(); sample = bundle.records[0]["samples"][0]; sample["token_ids"] = [1, 151643]; sample["token_ids_sha256"] = module.value_sha256(sample["token_ids"])
        with self.assertRaises(ValueError): bundle.validate()

    def test_length_finishes_and_native_stops_use_only_frozen_completion_definitions(self):
        bundle = Bundle(); sample = bundle.records[0]["samples"][0]
        sample.update(finish_reason="length", token_ids=[1] * 8192, output_tokens=8192, token_ids_sha256=module.value_sha256([1] * 8192))
        bundle.complete.update(length_finishes=1, output_tokens=512 - 2 + 8192)
        result = bundle.grade(); domain = bundle.cache["records"][0]["record"]["source_kind"]
        summary = result["public"]["domains"][domain]
        self.assertEqual(summary["length_finishes"], 1); self.assertEqual(summary["native_eos_observed"], 127)
        self.assertEqual(summary["stop_finishes"], 127)
        bundle.records[0]["samples"][0]["token_ids"] = [1]
        with self.assertRaises(ValueError): bundle.validate()

    def test_abort_or_unknown_engine_finish_prevents_all_grading_calls(self):
        for finish in ("abort", "unknown", None):
            bundle = Bundle(); bundle.records[-1]["samples"][-1]["finish_reason"] = finish
            calls = []
            with self.assertRaises(ValueError): bundle.grade(lambda *args, **kwargs: calls.append(True))
            self.assertEqual(calls, [])

    def test_complete_output_and_length_totals_must_match_every_private_sample(self):
        for key in ("length_finishes", "output_tokens"):
            bundle = Bundle(); bundle.complete[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): bundle.validate()

    def test_expected_comparator_five_paths_hashes_and_versions_are_pinned(self):
        cache, complete, records, pins = Bundle().raw()
        for field in ("source_sha256", "runtime_versions", "loaded_modules"):
            changed = comparator(); name = next(iter(changed[field])); changed[field][name] = "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_inputs(cache, complete, records, expected_file_pins=pins, expected_comparator=changed, fixture=True)

    def test_domain_pass_estimators_match_an_independent_full_denominator_oracle(self):
        def response(row, index):
            ordinal = row["source"]["row"]
            count = ordinal % 9
            gold = row["label"]["answer"] if row["source_kind"] == "math" else "2"
            return r"\boxed{" + (gold if index < count else "wrong") + "}"
        result = Bundle(response).grade()["public"]
        expected_correct = sum(min(index % 9, 8) for index in range(16))
        expected_nonzero = sum(index % 9 > 0 for index in range(16))
        for domain in ("math", "arc_challenge"):
            actual = result["domains"][domain]
            self.assertEqual((actual["record_count"], actual["sample_count"], actual["denominator"]), (16, 128, 128))
            self.assertEqual(actual["correct_samples"], expected_correct)
            self.assertEqual(actual["pass_at_1"], expected_correct / 128)
            self.assertEqual(actual["pass_at_8"], expected_nonzero / 16)
        self.assertFalse(result["method_ranking_permitted"])

    def test_nonrankable_worker_errors_preserve_denominator_one_and_all_128_domain_samples(self):
        def grader(row, text, **kwargs):
            result = fixture_grade(row, text, **kwargs)
            if row["source_kind"] == "math":
                result.update(status="comparison_timeout", correct=False, rankable=False,
                    error={"type": "FixtureTimeout", "phase": "comparison"}, worker={"cleanup": {"successful": True}})
            return result
        result = Bundle().grade(grader)
        domain = result["public"]["domains"]["math"]
        self.assertEqual((domain["pass_at_1"], domain["pass_at_8"], domain["denominator"], domain["grading_errors"]), (0., 0., 128, 128))
        self.assertFalse(domain["all_grades_rankable"])
        self.assertEqual(domain["status_counts"], {"comparison_timeout": 128})
        self.assertTrue(all(row["grade"]["denominator"] == 1 for row in result["private"]["observed_grades"]))

    def test_strict_last_box_percentage_and_original_arc_labels_are_not_adapted(self):
        texts = (r"\boxed{2} trailing \boxed", r"\boxed{}", r"\boxed{", "no final box")
        for text in texts:
            result = Bundle(lambda row, index: text).grade()["public"]
            self.assertTrue(all(domain["correct_samples"] == 0 for domain in result["domains"].values()))
        bundle = Bundle(lambda row, index: r"\boxed{a}" if row["source_kind"] == "arc_challenge" else r"\boxed{100%}")
        result = bundle.grade()["public"]
        self.assertEqual(result["domains"]["arc_challenge"]["correct_samples"], 0)
        self.assertEqual(result["domains"]["math"]["correct_samples"], 8)  # Only synthetic gold1 equals100%.

    def test_grading_calls_are_exactly_sequential32_by8_with_original_text_and_seed_order(self):
        bundle = Bundle(); calls = []
        def grader(row, text, **kwargs):
            calls.append((row["id"], text, deepcopy(kwargs)))
            return fixture_grade(row, text, **kwargs)
        bundle.grade(grader)
        expected = [(record["record_id"], sample["text"]) for record in bundle.records for sample in record["samples"]]
        self.assertEqual([(identifier, text) for identifier, text, _ in calls], expected)
        self.assertEqual(len(calls), 256)
        self.assertTrue(all(kwargs == {"timeout_seconds": 5, "expected_comparator": comparator()} for _, _, kwargs in calls))

    def test_mutable_grader_arguments_are_isolated_and_raw_inputs_never_change(self):
        bundle = Bundle(); cache, complete, records, pins = bundle.raw(); original = deepcopy((records, pins)); expected = comparator()
        def grader(row, text, **kwargs):
            result = fixture_grade(row, text, **kwargs)
            row["original_record"].clear(); kwargs["expected_comparator"]["source_sha256"].clear()
            return result
        result = module.grade_calibration(cache, complete, records, expected_file_pins=pins, expected_comparator=expected, grader=grader, fixture=True)
        self.assertEqual((records, pins), original); self.assertEqual(expected, comparator())
        self.assertEqual(len(result["private"]["observed_grades"]), 256)

    def test_public_result_contains_aggregate_hashes_without_questions_answers_or_record_ids(self):
        bundle = Bundle(); result = bundle.grade(); public = json.dumps(result["public"])
        for row in bundle.cache["records"]:
            self.assertNotIn(row["record"]["id"], public)
            self.assertNotIn(row["record"]["question"], public)
        for secret in ("PRIVATE-RESPONSE-MARKER", "original_record", "extraction", "gold_text", "token_ids", "observed_grades"):
            self.assertNotIn(secret, public)
        self.assertFalse(result["public"]["raw_records_answers_or_grades_included"])
        self.assertFalse(result["public"]["intermediate_reasoning_verified"])

    def test_private_grade_inventory_hash_and_fixed_repetition_definition_are_reproducible(self):
        text = " ".join(["repeat"] * 10) + r" \boxed{2}"
        bundle = Bundle(lambda row, index: text)
        first, second = bundle.grade(), bundle.grade()
        self.assertEqual(first, second)
        self.assertEqual(first["public"]["private_grades_value_sha256"], module.value_sha256(first["private"]))
        words = text.split(); grams = [tuple(words[index:index + 8]) for index in range(len(words) - 7)]
        expected = 1 - len(set(grams)) / len(grams)
        for domain in first["public"]["domains"].values(): self.assertAlmostEqual(domain["mean_repetition_8gram"], expected)
        self.assertTrue(first["public"]["fixture_grader_injected"]); self.assertFalse(first["public"]["frozen_grader_callable_used"])

    def test_cleanup_exception_carries_its_denominator_and_partial_inventory_then_stops(self):
        bundle = Bundle(); calls = []
        def grader(row, text, **kwargs):
            calls.append(row["id"])
            result = fixture_grade(row, text, **kwargs)
            if len(calls) == 10:
                result.update(status="cleanup_error", correct=False, rankable=False, error={"type": "FixtureCleanup"}, caller_must_abort=True,
                    worker={"cleanup": {"successful": False}})
                raise source.SourceGraderCleanupError(result)
            return result
        with self.assertRaises(module.CalibrationGradingAborted) as raised: bundle.grade(grader)
        abort = raised.exception
        self.assertEqual(len(calls), 10); self.assertEqual(len(abort.completed_grades), 10)
        self.assertEqual(abort.completed_grades[-1]["grade"]["denominator"], 1)
        self.assertTrue(abort.completed_grades[-1]["grade"]["caller_must_abort"])
        self.assertFalse(abort.result["public"]["complete"]); self.assertNotIn("domains", abort.result["public"])
        self.assertEqual(len(abort.result["private"]["record_summaries"]), 1)
        json.dumps(abort.result, allow_nan=False)

    def test_returned_cleanup_error_or_unfinished_cleanup_also_blocks_next_sample(self):
        for kind in ("status", "flag", "worker"):
            calls = []
            def grader(row, text, **kwargs):
                calls.append(True); result = fixture_grade(row, text, **kwargs)
                if kind == "status": result.update(status="cleanup_error", correct=False, rankable=False, error={"type": "UnsafeCleanup"})
                elif kind == "flag": result["caller_must_abort"] = True
                else: result["worker"] = {"cleanup": {"successful": False}}
                return result
            with self.subTest(kind=kind), self.assertRaises(module.CalibrationGradingAborted): Bundle().grade(grader)
            self.assertEqual(len(calls), 1)

    def test_unexpected_grader_exception_preserves_prior_observations_without_manufacturing_a_grade(self):
        calls = []
        def grader(row, text, **kwargs):
            calls.append(True)
            if len(calls) == 4: raise RuntimeError("synthetic worker orchestration failure")
            return fixture_grade(row, text, **kwargs)
        with self.assertRaises(module.CalibrationGradingAborted) as raised: Bundle().grade(grader)
        self.assertEqual(len(calls), 4); self.assertEqual(len(raised.exception.completed_grades), 3)
        self.assertEqual(raised.exception.result["private"]["abort"]["reason"], "unexpected_grader_exception")
        self.assertNotIn("domains", raised.exception.result["public"])

    def test_malformed_status_denominator_identity_or_nonjson_receipt_aborts_without_retry(self):
        for kind in ("denominator", "status", "identity", "hash", "correct-error", "object", "nonfinite"):
            calls = []
            def grader(row, text, **kwargs):
                calls.append(True); result = fixture_grade(row, text, **kwargs)
                if kind == "denominator": result["denominator"] = 0
                elif kind == "status": result["status"] = "invented"
                elif kind == "identity": result["record_id"] = "0" * 64
                elif kind == "hash": result["response_sha256"] = "0" * 64
                elif kind == "correct-error": result.update(status="comparison_timeout", correct=True, rankable=False, error={"type": "Bad"})
                elif kind == "object": return object()
                else: result["nonfinite"] = float("nan")
                return result
            with self.subTest(kind=kind), self.assertRaises(module.CalibrationGradingAborted) as raised: Bundle().grade(grader)
            self.assertEqual(len(calls), 1); json.dumps(raised.exception.result, allow_nan=False)

    def test_injected_grader_requires_fixture_inputs_and_no_default_worker_is_started_in_fixtures(self):
        cache, complete, records, pins = Bundle().raw()
        with self.assertRaises(ValueError): module.grade_calibration(cache, complete, records,
            expected_file_pins=pins, expected_comparator=comparator(), grader=fixture_grade)
        with self.assertRaises(ValueError): module.grade_calibration(cache, complete, records,
            expected_file_pins=pins, expected_comparator=comparator(), fixture=True)


if __name__ == "__main__":
    unittest.main()
