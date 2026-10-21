"""Authored source-only oracles; execution belongs to a future CPU allocation.

Fixtures use no downloaded data, model weights or native tokenizer. A few worker
integration oracles import the pinned local math grader and its installed CPU
dependencies; no experiment model is loaded. All spawned targets are importable.
"""
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from extension import source_evaluation as module
from extension.labeled_feasibility import CRITERION, source_record
from extension.preserving_data import text_sha256, value_sha256
from reasoning.data import SYSTEM


def math_row(answer="2", question="What is 1+1?"):
    raw = {"problem": question, "level": "Level 1", "type": "Algebra", "solution": r"A proof. \boxed{" + answer + "}"}
    row, rejected = source_record("math", raw, {"source": "fixture-math", "row": 0})
    assert rejected is None
    row["group_id"] = value_sha256({"criterion": CRITERION, "member_ids": [row["id"]]})
    return row


def arc_row():
    raw = {"id": "fixture-arc", "question": "Which option is first?",
        "choices": {"label": ["2", "A", "1"], "text": ["first", "second", "third"]}, "answerKey": "2"}
    row, rejected = source_record("arc_challenge", raw, {"source": "fixture-arc", "row": 0})
    assert rejected is None
    row["group_id"] = value_sha256({"criterion": CRITERION, "member_ids": [row["id"]]})
    return row


class FixtureTokenizer:
    eos_token_id = 151645
    chat_template = "fixture-native-system-user-assistant"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        text = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
        return self.encode(text) if tokenize else text


def identity(tokenizer):
    return {"files": {"fixture.json": "a" * 64}, "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "eos_token_id": tokenizer.eos_token_id}


def fake_comparison(prediction, gold, timeout_seconds):
    # A narrow numeric test oracle, deliberately independent from the real grader.
    return {"status": "graded", "correct": Fraction(prediction) == Fraction(gold),
        "comparator_provenance": {"fixture": True}}


def worker_hang(sender, prediction, gold):
    time.sleep(30)


def worker_crash(sender, prediction, gold):
    os._exit(17)


def worker_invalid(sender, prediction, gold):
    sender.send({"status": "graded", "correct": "true"})
    sender.close()


def worker_error(sender, prediction, gold):
    sender.send({"status": "grader_error", "correct": False, "error_type": "FixtureFailure"})
    sender.close()


class PromptAndRecordTests(unittest.TestCase):
    def test_complete_record_rebuild_is_exact_and_does_not_mutate(self):
        for row in (math_row(), arc_row()):
            before = deepcopy(row)
            proof = module.validate_record(row)
            self.assertEqual(proof["record_sha256"], value_sha256(row))
            self.assertFalse(proof["complete_group_audit_performed_here"])
            self.assertEqual(row, before)

    def test_changed_gold_hash_original_or_extra_field_fails_complete_validation(self):
        for mutate in (lambda row: row.update(label_sha256="0" * 64),
                       lambda row: row["original_record"].update(solution=r"\boxed{99}"),
                       lambda row: row.update(unsealed="extra")):
            row = math_row()
            mutate(row)
            with self.assertRaises(ValueError):
                module.validate_record(row)

    def test_math_question_is_byte_preserved_and_contains_no_solution(self):
        row = math_row(question="  Preserve\n $1+1$ exactly.  ")
        messages = module.render_prompt(row)
        self.assertEqual(messages, [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["original_record"]["problem"]}])
        self.assertNotIn("A proof", messages[1]["content"])

    def test_arc_options_labels_and_order_are_original(self):
        row = arc_row()
        user = module.render_prompt(row)[1]["content"]
        self.assertEqual(user, "Which option is first?\n\n2. first\nA. second\n1. third\n\n" + module.ARC_INSTRUCTION)
        self.assertNotIn("answerKey", user)

    def test_gold_mutation_and_deletion_cannot_change_prompt_or_tokens(self):
        tokenizer = FixtureTokenizer()
        for row in (math_row(), arc_row()):
            original = module.serialize_prompt(row, tokenizer, identity(tokenizer))
            changed = deepcopy(row)
            changed["label"] = {"answer": "leaked-gold", "answer_key": "leaked-gold"}
            changed["label_sha256"] = changed["label_id"] = "0" * 64
            changed["original_record"].pop("solution", None)
            changed["original_record"].pop("answerKey", None)
            rewritten = module.serialize_prompt(changed, tokenizer, identity(tokenizer))
            self.assertEqual(original, rewritten)
            with self.assertRaises(ValueError):
                module.validate_record(changed)

    def test_native_render_parity_hashes_and_exact_envelope(self):
        row, tokenizer = math_row(), FixtureTokenizer()
        output = module.serialize_prompt(row, tokenizer, identity(tokenizer))
        proof, tokens = output["provenance"], output["prompt_token_ids"]
        self.assertEqual(tokens, tokenizer.encode(output["rendered_prompt"]))
        self.assertEqual(proof["messages_sha256"], value_sha256(output["messages"]))
        self.assertEqual(proof["prompt_token_ids_sha256"], value_sha256(tokens))
        self.assertEqual(proof["rendered_prompt_sha256"], text_sha256(output["rendered_prompt"]))
        self.assertEqual(proof["record_id_namespace"], "independent_labeled_source_record")
        self.assertFalse(proof["gold_fields_used"])
        self.assertFalse(proof["truncation_performed"])
        self.assertFalse(proof["tokenizer_files_verified_here"])
        self.assertEqual(module.serialize_prompt(row, tokenizer, identity(tokenizer), len(tokens))["prompt_token_ids"], tokens)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            module.serialize_prompt(row, tokenizer, identity(tokenizer), len(tokens) - 1)

    def test_wrong_template_eos_identity_and_nonplain_ids_are_rejected(self):
        tokenizer = FixtureTokenizer()
        for change in ({"chat_template_sha256": "0" * 64}, {"eos_token_id": 0}, {"files": {}}):
            proof = dict(identity(tokenizer), **change)
            with self.assertRaises(ValueError):
                module.serialize_prompt(math_row(), tokenizer, proof)
        with patch.object(tokenizer, "apply_chat_template", return_value=[True]):
            with self.assertRaises(ValueError):
                module.serialize_prompt(math_row(), tokenizer, identity(tokenizer))

    def test_rendered_text_token_mismatch_is_not_repaired(self):
        tokenizer = FixtureTokenizer()
        original = tokenizer.apply_chat_template
        def mismatch(messages, *, tokenize, add_generation_prompt):
            output = original(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
            return output + [7] if tokenize else output
        with patch.object(tokenizer, "apply_chat_template", side_effect=mismatch), self.assertRaisesRegex(ValueError, "sequence differs"):
            module.serialize_prompt(math_row(), tokenizer, identity(tokenizer))

    def test_question_source_and_original_choice_identity_guards(self):
        for mutate in (lambda row: row.update(id="0" * 64), lambda row: row.update(group_id="not-sha"),
                       lambda row: row.update(question="rewritten"),
                       lambda row: row["original_record"].update(problem="rewritten")):
            row = math_row()
            mutate(row)
            with self.assertRaises(ValueError):
                module.render_prompt(row)
        row = arc_row()
        row["original_record"]["choices"]["label"] = ["A", "A", "B"]
        with self.assertRaises(ValueError):
            module.render_prompt(row)


class ExtractionAndPercentTests(unittest.TestCase):
    def test_last_balanced_literal_box_handles_nested_braces(self):
        text = r"First \boxed{1}. Final \boxed{\frac{1}{2}} done."
        extracted = module.extract_final_box(text)
        self.assertEqual(extracted["answer"], r"\frac{1}{2}")
        self.assertEqual(extracted["attempts"], 2)
        self.assertEqual(text[extracted["start"]:extracted["end"]], r"\boxed{\frac{1}{2}}")

    def test_later_malformed_box_never_falls_back(self):
        for suffix, status in ((r" \boxed", "malformed_final_box"), (r" \boxed {2}", "malformed_final_box"),
                               (r" \boxed{2", "unbalanced_final_box"), (r" \boxed{}", "empty_final_box"),
                               (r" \boxedx", "malformed_final_box")):
            extracted = module.extract_final_box(r"\boxed{1}" + suffix)
            self.assertEqual(extracted["status"], status)
            self.assertIsNone(extracted["answer"])

    def test_missing_box_and_plain_numbers_are_not_answer_fallbacks(self):
        for text in ("2", "The answer is A", "", r"\fbox{2}"):
            self.assertEqual(module.extract_final_box(text)["status"], "missing_box")

    def test_supported_scalar_percentages_use_exact_fraction_arithmetic(self):
        cases = {"50%": "1/2", r"50\%": "1/2", "-50%": "-1/2", "+.5%": "1/200",
            "0%": "0", "200%": "2", "1/2%": "1/200", "1/-2%": "-1/200",
            r"\frac{1}{2}%": "1/200", r"-\dfrac{-1}{2}\%": "1/200",
            r"\tfrac{.25}{.5}%": "1/200", "- 12.5 %": "-1/8"}
        for text, expected in cases.items():
            result = module.normalize_percentage(text)
            self.assertTrue(result["supported"], text)
            self.assertEqual(result["normalized"], expected, text)

    def test_unsupported_symbolic_mixed_or_zero_denominator_percent_is_explicit(self):
        for text in ("x%", "50%+1", "50%%", "1e2%", "$50%$", "one percent%", "1/0%",
                     r"\frac{1}{0}%", r"\frac{x}{2}%", r"\frac{\frac{1}{2}}{3}%", "50% of 2"):
            result = module.normalize_percentage(text)
            self.assertFalse(result["supported"], text)
            self.assertEqual(result["status"], "unsupported_percentage")
            self.assertIsNone(result["normalized"])

    def test_nonpercent_math_passes_through_unchanged_except_outer_whitespace(self):
        for text in (r"\sqrt{2}", "10", "1000", "0.1", "(1,2)"):
            self.assertEqual(module.normalize_percentage(" " + text + " ")["normalized"], text)

    def test_gold_preflight_blocks_unsupported_percent_without_refill(self):
        for answer in ("x%", "1/0%"):
            result = module.validate_gold(math_row(answer))
            self.assertFalse(result["supported"])
            self.assertEqual(result["gold_text"], answer)
            self.assertEqual(result["gold_sha256"], text_sha256(answer))
        raw = {"problem": "Empty official gold fixture", "level": "Level 1", "type": "Algebra",
               "solution": r"A proof. \boxed{}"}
        empty, quarantine = source_record("math", raw, {"source": "fixture-empty", "row": 0})
        self.assertEqual(quarantine, "unextractable_final_boxed_answer")
        self.assertIsNone(empty["label"]["answer"])
        self.assertEqual(module.validate_gold(math_row("50%"))["percent_normalized_gold"], "1/2")


class GradingTests(unittest.TestCase):
    def test_arc_requires_exact_original_label_without_case_or_index_remapping(self):
        row = arc_row()
        self.assertTrue(module.grade_response(row, r"\boxed{2}")["correct"])
        for text in (r"\boxed{A}", r"\boxed{a}", r"\boxed{B}", r"\boxed{(2)}", r"\boxed{2.}", "2"):
            result = module.grade_response(row, text)
            self.assertFalse(result["correct"], text)
            self.assertTrue(result["rankable"])
            self.assertEqual(result["denominator"], 1)
            self.assertIsNone(result["error"])

    def test_percent_is_normalized_before_comparator_for_both_sides(self):
        with patch.object(module, "_compare_math", side_effect=fake_comparison) as compare:
            result = module.grade_response(math_row("50%"), r"\boxed{.5}")
            self.assertTrue(result["correct"])
            compare.assert_called_once_with(".5", "1/2", 5.)
        with patch.object(module, "_compare_math", side_effect=fake_comparison) as compare:
            result = module.grade_response(math_row(".5"), r"\boxed{50\%}")
            self.assertTrue(result["correct"])
            compare.assert_called_once_with("1/2", ".5", 5.)

    def test_no_100x_equivalence_is_added(self):
        with patch.object(module, "_compare_math", side_effect=fake_comparison):
            for gold, prediction in (("50", "50%"), ("10", "1000"), ("10", "0.1"), ("50%", "50")):
                self.assertFalse(module.grade_response(math_row(gold), r"\boxed{" + prediction + "}")["correct"])

    def test_unsupported_gold_is_not_ordinary_incorrect(self):
        with patch.object(module, "_compare_math") as compare:
            result = module.grade_response(math_row("x%"), r"\boxed{2}")
            compare.assert_not_called()
        self.assertEqual(result["status"], "unsupported_gold")
        self.assertFalse(result["rankable"])
        self.assertEqual(result["denominator"], 1)

    def test_unsupported_prediction_is_explicit_incorrect_without_worker(self):
        with patch.object(module, "_compare_math") as compare:
            result = module.grade_response(math_row("2"), r"\boxed{x%}")
            compare.assert_not_called()
        self.assertEqual(result["status"], "unsupported_prediction_percentage")
        self.assertTrue(result["rankable"])
        self.assertIsNone(result["error"])

    def test_extraction_failure_never_calls_comparator(self):
        with patch.object(module, "_compare_math") as compare:
            result = module.grade_response(math_row(), r"\boxed{2} \boxed{")
            compare.assert_not_called()
        self.assertEqual(result["status"], "unbalanced_final_box")
        self.assertTrue(result["rankable"])

    def test_worker_errors_keep_denominator_and_block_ranking(self):
        for status in ("timeout", "worker_crash", "grader_error", "worker_protocol_error", "worker_error"):
            with patch.object(module, "_compare_math", return_value={"status": status, "correct": False, "error_type": "Fixture"}):
                result = module.grade_response(math_row(), r"\boxed{2}")
            self.assertFalse(result["rankable"])
            self.assertFalse(result["correct"])
            self.assertEqual(result["status"], status)
            self.assertEqual(result["error"], {"type": "Fixture"})
            self.assertEqual(result["denominator"], 1)

    def test_invalid_deadline_is_rejected_before_worker(self):
        for timeout in (0, -1, True, float("nan"), float("inf"), 61):
            with self.assertRaises(ValueError):
                module.grade_response(math_row(), r"\boxed{2}", timeout_seconds=timeout)

    def test_spawn_timeout_and_crash_return_without_queue_hang(self):
        for target, status in ((worker_hang, "timeout"), (worker_crash, "worker_crash")):
            with patch.object(module, "_math_worker", target):
                begun = time.monotonic()
                result = module._compare_math("1", "1", 1.)
                self.assertLess(time.monotonic() - begun, 6.)
            self.assertEqual(result["status"], status)
            self.assertFalse(result["correct"])

    def test_spawn_invalid_payload_and_explicit_error_are_distinct(self):
        for target, status in ((worker_invalid, "worker_protocol_error"), (worker_error, "grader_error")):
            with patch.object(module, "_math_worker", target):
                result = module._compare_math("1", "1", 5.)
            self.assertEqual(result["status"], status)

    def test_bounded_cleanup_never_closes_a_still_live_process_or_masks_error(self):
        class Connection:
            def close(self):
                pass
            def poll(self, timeout):
                return False
        class Process:
            pid = 7
            def __init__(self):
                self.joins, self.closed = [], False
            def start(self):
                pass
            def join(self, seconds):
                self.joins.append(seconds)
            def is_alive(self):
                return True
            def terminate(self):
                pass
            def kill(self):
                pass
            def close(self):
                self.closed = True
                raise ValueError("Cannot close a live process")
        class Context:
            def __init__(self):
                self.process = Process()
            def Pipe(self, duplex):
                assert duplex is False
                return Connection(), Connection()
            def Process(self, **kwargs):
                return self.process
        context = Context()
        with patch.object(module.multiprocessing, "get_context", return_value=context):
            result = module._compare_math("1", "1", 1.)
        self.assertEqual(result["status"], "worker_cleanup_error")
        self.assertEqual(result["prior_status"], "timeout")
        self.assertEqual(context.process.joins, [.05, .25, .25])
        self.assertFalse(context.process.closed)

    def test_actual_pinned_worker_percentage_and_module_identity(self):
        result = module.grade_response(math_row(".5"), r"\boxed{50\%}", timeout_seconds=5)
        self.assertTrue(result["rankable"], result)
        self.assertTrue(result["correct"], result)
        proof = result["comparator_provenance"]
        directory = Path(module.__file__).resolve().parents[1] / "eval" / "utils"
        self.assertEqual(proof["loaded_modules"], {name: str((directory / filename).resolve())
            for name, filename in module._GRADER_FILES.items()})
        self.assertEqual(set(proof["runtime_versions"]), set(module._PACKAGES))
        self.assertFalse(proof["include_percentage"])
        self.assertFalse(proof["internal_timeout"])
        self.assertEqual(proof["timeout_owner"], "external_spawn_process")

    def test_actual_pinned_worker_does_not_accept_100x_scaling(self):
        for gold, prediction in (("50", "50%"), ("10", "1000"), ("10", ".1")):
            result = module.grade_response(math_row(gold), r"\boxed{" + prediction + "}", timeout_seconds=5)
            self.assertTrue(result["rankable"], result)
            self.assertFalse(result["correct"], result)


class SamplingMetricsTests(unittest.TestCase):
    def samples(self, row):
        result = []
        for index in range(8):
            text = r"\boxed{2}" if index < 3 else r"\boxed{A}"
            result.append({"sample_index": index, "request_seed": module.sample_seed(row["id"]),
                "grade": module.grade_response(row, text),
                "metrics": module.completion_metrics(text, [10, 151645], "stop", None, 151645)})
        return result

    def test_seed_uses_record_identity_and_declared_global_seed(self):
        row = arc_row()
        self.assertEqual(module.sample_seed(row["id"]), (20261021 + int(row["id"][:8], 16)) % 2 ** 31)
        self.assertNotEqual(module.sample_seed(row["id"]), module.sample_seed(row["id"], 7))
        for identifier, seed in (("bad", 0), (row["id"], True), (row["id"], -1)):
            with self.assertRaises(ValueError):
                module.sample_seed(identifier, seed)

    def test_completion_metrics_distinguish_length_stop_and_native_eos(self):
        stopped = module.completion_metrics(r"\boxed{2}", [3], "stop", None, 151645)
        self.assertTrue(stopped["stop_finished"])
        self.assertFalse(stopped["native_eos_observed"])
        self.assertTrue(module.completion_metrics("", [3], "stop", 151645, 151645)["native_eos_observed"])
        self.assertTrue(module.completion_metrics("", [151645], "stop", None, 151645)["native_eos_observed"])
        self.assertTrue(module.completion_metrics("", [3], "length", None, 151645)["length_finished"])

    def test_repetition_is_whitespace_8gram_duplication_fraction(self):
        text = " ".join(["word"] * 10)
        self.assertAlmostEqual(module.completion_metrics(text, [], "length", None, 151645)["repetition_8gram"], 2 / 3)
        self.assertEqual(module.completion_metrics("one two", [], "stop", None, 151645)["repetition_8gram"], 0)

    def test_exact_eight_samples_report_mean_success_and_any_success(self):
        row = arc_row()
        samples = self.samples(row)
        summary = module.summarize_samples(row, list(reversed(samples)))
        self.assertEqual((summary["pass_at_1"], summary["pass_at_8"]), (3 / 8, 1.))
        self.assertEqual((summary["denominator"], summary["sample_count"], summary["generated_token_sum"]), (8, 8, 16))
        self.assertTrue(summary["rankable"])
        self.assertEqual(summary, module.summarize_samples(row, samples))

    def test_missing_duplicate_wrong_seed_or_response_mismatch_rejected(self):
        row = arc_row()
        samples = self.samples(row)
        bad = [samples[:-1], samples + [samples[0]]]
        for field, value in (("sample_index", 1), ("request_seed", 0)):
            changed = deepcopy(samples)
            changed[0][field] = value
            bad.append(changed)
        changed = deepcopy(samples)
        changed[0]["metrics"]["response_sha256"] = "0" * 64
        bad.append(changed)
        for inventory in bad:
            with self.assertRaises(ValueError):
                module.summarize_samples(row, inventory)

    def test_error_keeps_all_eight_in_denominator_and_blocks_selection(self):
        row = arc_row()
        samples = self.samples(row)
        samples[0]["grade"].update(correct=False, rankable=False, status="timeout", error={"type": "Timeout"})
        result = module.summarize_samples(row, samples)
        self.assertEqual(result["denominator"], 8)
        self.assertEqual(result["pass_at_1"], 2 / 8)
        self.assertEqual(result["grading_errors"], 1)
        self.assertFalse(result["rankable"])

    def test_aborted_completion_blocks_ranking_even_with_parseable_correct_box(self):
        row = arc_row()
        samples = self.samples(row)
        samples[0]["metrics"]["finish_reason"] = "abort"
        result = module.summarize_samples(row, samples)
        self.assertEqual(result["denominator"], 8)
        self.assertEqual(result["aborted_samples"], 1)
        self.assertFalse(result["rankable"])

    def test_invalid_hash_repetition_and_inconsistent_grade_states_fail_closed(self):
        row = arc_row()
        cases = []
        for value in (float("nan"), float("inf"), -1, 1.01, True):
            samples = self.samples(row)
            samples[0]["metrics"]["repetition_8gram"] = value
            cases.append(samples)
        samples = self.samples(row)
        samples[0]["metrics"]["response_sha256"] = samples[0]["grade"]["response_sha256"] = "not-sha"
        cases.append(samples)
        for field, value in (("status", "unknown"), ("status", "missing_box"), ("rankable", False)):
            samples = self.samples(row)
            samples[0]["grade"][field] = value
            cases.append(samples)
        for samples in cases:
            with self.assertRaises(ValueError):
                module.summarize_samples(row, samples)


if __name__ == "__main__":
    unittest.main()
