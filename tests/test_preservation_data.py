"""Unexecuted pure native-tokenizer fixtures; no real models, data or downloads."""
from copy import deepcopy
import unittest

from reasoning.data import prompt_id
from extension.preserving_data import (CompleteContextError, EXCLUSION_SCHEMA,
    POOL_ID_NAMESPACE, PreservationDataError, prepare_pool, serialize_selected,
    text_sha256, value_sha256, verify_serialized_examples)


EOS, TAIL, SUFFIX, VOCAB = 2, "\ue000\n", (2, 20), 2_000_000


class NativeTokenizer:
    """A deterministic native frame with an EOS followed by a newline token."""
    chat_template = "fixture-v1:system/user/assistant; unchanged content; EOS then newline"
    eos_token_id = EOS

    def __init__(self, fault=None):
        self.fault, self.calls = fault, []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Native template must not receive another special-token pass")
        result = [EOS if character == "\ue000" else ord(character) + 10 for character in text]
        if self.fault == "boundary_merge" and "\ue000" in text and "<assistant>" in text:
            boundary = text.index("<assistant>") + len("<assistant>") - 1
            result[boundary] += 1
        return result

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        self.calls.append({"messages": deepcopy(messages), "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt, "return_dict": return_dict})
        roles = [message["role"] for message in messages]
        if roles not in (["system", "user"], ["system", "user", "assistant"]):
            raise AssertionError("Unexpected role sequence")
        if add_generation_prompt != (len(messages) == 2) or return_dict:
            raise AssertionError("Unexpected native serialization flags")
        system = messages[0]["content"]
        if self.fault == "strip_system":
            system = system.strip()
        prompt = "<system>" + system + "</system>\n<user>" + messages[1]["content"] + "</user>\n<assistant>"
        text = prompt
        if len(messages) == 3:
            output = messages[2]["content"]
            if self.fault == "rewrite_output":
                output = output.strip()
            text += output + ("\n" if self.fault == "missing_eos" else TAIL)
        if not tokenize:
            return {} if self.fault == "nontext_render" else text
        tokens = self.encode(text, add_special_tokens=False)
        if len(messages) == 3 and self.fault == "token_text_disagreement":
            tokens[-1] += 1
        if self.fault == "boolean_token":
            tokens[0] = True
        if self.fault == "out_of_vocab":
            tokens[0] = VOCAB
        return tuple(tokens) if self.fault == "nonlist_tokens" else tokens


def pool():
    result = []
    for instruction, output in (("  What is 1 + 1?\n", "  <think>work</think> \\boxed{2} \n"),
                                ("A second question?", "Another original answer."),
                                ("A reserved source question?", "Do not select."),
                                ("A benchmark-overlapping question?", "Also do not select.")):
        result.append({"id": prompt_id(instruction), "instruction": instruction, "input": "",
                       "output": output, "system": "  Native system text.\n"})
    return result


def audit(rows, source_ids=(), benchmark_ids=()):
    components = {}
    for index, (purpose, scope, identifiers) in enumerate((
            ("reserved_source_evaluation", "all_reserved_source_splits", source_ids),
            ("final_benchmark_questions", "all_nine_final_benchmarks", benchmark_ids))):
        components[purpose] = {"scope": scope, "pool_ids": sorted(identifiers),
            "audit_receipt_sha256": str(index + 1) * 64, "criterion_sha256": "3" * 64,
            "question_inventory_sha256": str(index + 4) * 64}
    return {"schema": EXCLUSION_SCHEMA, "id_namespace": POOL_ID_NAMESPACE,
            "population_ids_sha256": value_sha256(sorted(row["id"] for row in rows)), "components": components}


def source_identity():
    return {"repository": "fixture/original", "revision": "a" * 40, "canonical_pool_file_sha256": "b" * 64}


def tokenizer_identity(tokenizer):
    return {"repository": "fixture/native", "revision": "c" * 40, "eos_token_id": EOS,
            "chat_template_sha256": text_sha256(tokenizer.chat_template), "files": {"tokenizer.json": "d" * 64}}


class PreservationDataTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.tokenizer = pool(), NativeTokenizer()
        self.excluded = {self.rows[2]["id"], self.rows[3]["id"]}
        self.receipt = audit(self.rows, [self.rows[2]["id"]], [self.rows[3]["id"]])

    def prepare(self, **changes):
        values = dict(excluded_pool_ids=self.excluded, exclusion_receipt=self.receipt,
                      source_identity=source_identity())
        values.update(changes)
        return prepare_pool(self.rows, **values)

    def serialize(self, ids=None, tokenizer=None, **changes):
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        values = dict(excluded_pool_ids=self.excluded, exclusion_receipt=self.receipt,
            source_identity=source_identity(), tokenizer_identity=tokenizer_identity(tokenizer),
            cutoff=100_000, context_role="full", eos_token_id=EOS, terminal_suffix=SUFFIX,
            terminal_text=TAIL, vocab_size=VOCAB)
        values.update(changes)
        return serialize_selected(self.rows, [self.rows[0]["id"]] if ids is None else ids, tokenizer, **values)

    def replay(self, examples, **changes):
        values = dict(source_identity=source_identity(), tokenizer_identity=tokenizer_identity(self.tokenizer),
            exclusion_receipt_sha256=value_sha256(self.receipt), cutoff=100_000, context_role="full",
            eos_token_id=EOS, terminal_suffix=SUFFIX, terminal_text=TAIL, vocab_size=VOCAB)
        values.update(changes)
        return verify_serialized_examples(examples, self.tokenizer, **values)

    def test_prepare_preserves_exact_strings_order_and_inputs(self):
        before, receipt_before = deepcopy(self.rows), deepcopy(self.receipt)
        prepared = self.prepare()
        self.assertEqual(prepared["rows"], self.rows[:2])
        self.assertEqual(self.rows, before)
        self.assertEqual(self.receipt, receipt_before)
        prepared["rows"][0]["output"] = "changed by downstream caller"
        self.assertEqual(self.rows, before)
        self.assertEqual(prepared["provenance"]["excluded_rows"], 2)
        self.assertFalse(prepared["provenance"]["selection_or_scoring_performed"])
        self.assertFalse(prepared["provenance"]["semantic_decontamination_claimed"])

    def test_both_audits_and_exact_union_are_required(self):
        for alteration in ("missing_component", "wrong_scope", "wrong_criterion", "missing_id"):
            receipt = deepcopy(self.receipt)
            if alteration == "missing_component":
                del receipt["components"]["final_benchmark_questions"]
            elif alteration == "wrong_scope":
                receipt["components"]["final_benchmark_questions"]["scope"] = "one_benchmark_only"
            elif alteration == "wrong_criterion":
                receipt["components"]["final_benchmark_questions"]["criterion_sha256"] = "5" * 64
            else:
                receipt["components"]["final_benchmark_questions"]["pool_ids"] = []
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.prepare(exclusion_receipt=receipt)

    def test_full_population_and_correct_id_namespace_required(self):
        for alteration in ("population", "namespace", "independent_labeled_id"):
            receipt = deepcopy(self.receipt)
            excluded = set(self.excluded)
            if alteration == "population":
                receipt["population_ids_sha256"] = "0" * 64
            elif alteration == "namespace":
                receipt["id_namespace"] = "independent_labeled_source_record"
            else:
                excluded.add(value_sha256({"source": "independent", "question_sha256": self.rows[0]["id"]}))
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.prepare(exclusion_receipt=receipt, excluded_pool_ids=excluded)

    def test_empty_exclusion_results_are_allowed_only_with_both_audit_bindings(self):
        prepared = self.prepare(excluded_pool_ids=set(), exclusion_receipt=audit(self.rows))
        self.assertEqual(prepared["rows"], self.rows)
        self.assertEqual(prepared["provenance"]["excluded_rows"], 0)

    def test_all_excluded_pool_fails_explicitly(self):
        with self.assertRaisesRegex(PreservationDataError, "no eligible"):
            self.prepare(excluded_pool_ids={row["id"] for row in self.rows},
                         exclusion_receipt=audit(self.rows, [row["id"] for row in self.rows]))

    def test_bad_exclusion_schema_duplicates_or_nonsha_fail_closed(self):
        for alteration in ("extra", "duplicate", "bad_sha", "wrong_union_type"):
            receipt, excluded = deepcopy(self.receipt), set(self.excluded)
            if alteration == "extra":
                receipt["unverified_extra"] = True
            elif alteration == "duplicate":
                receipt["components"]["reserved_source_evaluation"]["pool_ids"] *= 2
            elif alteration == "bad_sha":
                receipt["components"]["reserved_source_evaluation"]["audit_receipt_sha256"] = "not-a-hash"
            else:
                excluded = list(excluded)
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.prepare(excluded_pool_ids=excluded, exclusion_receipt=receipt)

    def test_noncanonical_rows_are_never_rewritten_or_silently_accepted(self):
        for alteration in ("id", "input", "missing", "extra", "output", "instruction", "system"):
            original = deepcopy(self.rows)
            if alteration == "id":
                self.rows[0]["id"] = "0" * 64
            elif alteration == "input":
                self.rows[0]["input"] = "must not be concatenated"
            elif alteration == "missing":
                del self.rows[0]["system"]
            elif alteration == "extra":
                self.rows[0]["label"] = "a foreign schema"
            else:
                self.rows[0][alteration] = " \n"
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.prepare()
            self.rows = original

    def test_duplicate_normalized_prompt_id_is_not_dropped(self):
        self.rows.append(dict(self.rows[0], output="A different answer to the same canonical prompt"))
        with self.assertRaisesRegex(PreservationDataError, "Duplicate"):
            self.prepare()

    def test_reserved_missing_duplicate_or_labeled_selected_ids_fail_before_tokenization(self):
        bad_lists = [[self.rows[2]["id"]], [self.rows[3]["id"]], ["0" * 64],
                     [self.rows[0]["id"], self.rows[0]["id"]], [],
                     [value_sha256({"source": "independent", "question": self.rows[0]["id"]})]]
        for ids in bad_lists:
            tokenizer = NativeTokenizer()
            with self.subTest(ids=ids), self.assertRaises(PreservationDataError):
                self.serialize(ids, tokenizer)
            self.assertEqual(tokenizer.calls, [])

    def test_response_mask_includes_first_answer_eos_and_post_eos_tail_without_shift(self):
        result = self.serialize()
        example = result["examples"][0]
        count = example["provenance"]["prompt_tokens"]
        body = self.tokenizer.encode(self.rows[0]["output"], add_special_tokens=False)
        self.assertEqual(example["labels"][:count], [-100] * count)
        self.assertEqual(example["input_ids"][count:], body + list(SUFFIX))
        self.assertEqual(example["labels"][count:], body + list(SUFFIX))
        self.assertEqual(example["attention_mask"], [1] * len(example["input_ids"]))
        self.assertEqual(example["provenance"]["active_tokens"], len(body) + len(SUFFIX))
        self.assertEqual(example["provenance"]["active_eos_tokens"], 1)
        self.assertFalse(example["provenance"]["causal_shift_applied"])
        self.assertEqual(example["original_row"], self.rows[0])

    def test_explicit_requested_order_and_no_replacement_rows(self):
        ids = [self.rows[1]["id"], self.rows[0]["id"]]
        result = self.serialize(ids)
        self.assertEqual([example["id"] for example in result["examples"]], ids)
        self.assertEqual(result["provenance"]["ordered_ids_sha256"], value_sha256(ids))
        for key in ("truncated_rows", "rewritten_rows", "requested_rows_dropped", "replacement_rows_added"):
            self.assertEqual(result["provenance"][key], 0)

    def test_native_messages_preserve_text_and_never_place_target_in_prompt_message(self):
        self.serialize()
        for call in self.tokenizer.calls:
            self.assertEqual(call["messages"][:2], [{"role": "system", "content": self.rows[0]["system"]},
                                                   {"role": "user", "content": self.rows[0]["instruction"]}])
            if call["add_generation_prompt"]:
                self.assertEqual(len(call["messages"]), 2)
            else:
                self.assertEqual(call["messages"][2], {"role": "assistant", "content": self.rows[0]["output"]})

    def test_boundary_token_merge_fails_even_when_text_and_token_encodings_agree(self):
        with self.assertRaisesRegex(PreservationDataError, "exact token prefix"):
            self.serialize(tokenizer=NativeTokenizer("boundary_merge"))

    def test_native_and_rendered_token_disagreement_fails(self):
        with self.assertRaisesRegex(PreservationDataError, "exact rendered text"):
            self.serialize(tokenizer=NativeTokenizer("token_text_disagreement"))

    def test_native_output_rewrite_missing_eos_and_prompt_stripping_fail(self):
        for fault in ("rewrite_output", "missing_eos", "strip_system", "nontext_render"):
            with self.subTest(fault=fault), self.assertRaises(PreservationDataError):
                self.serialize(tokenizer=NativeTokenizer(fault))

    def test_noninteger_outofvocab_or_nonlist_native_tokens_fail(self):
        for fault in ("boolean_token", "out_of_vocab", "nonlist_tokens"):
            with self.subTest(fault=fault), self.assertRaises(PreservationDataError):
                self.serialize(tokenizer=NativeTokenizer(fault))

    def test_actual_template_eos_and_tokenizer_file_identity_must_match(self):
        for alteration in ("template", "eos", "files", "native_eos"):
            tokenizer = NativeTokenizer()
            identity = tokenizer_identity(tokenizer)
            if alteration == "template":
                identity["chat_template_sha256"] = "0" * 64
            elif alteration == "eos":
                identity["eos_token_id"] = 3
            elif alteration == "files":
                identity["files"] = {}
            else:
                tokenizer.eos_token_id = 3
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.serialize(tokenizer=tokenizer, tokenizer_identity=identity)

    def test_exact_terminal_text_and_suffix_are_supplied_not_inferred_or_appended(self):
        for changes in (dict(terminal_suffix=(EOS,)), dict(terminal_suffix=(20, EOS)),
                        dict(terminal_suffix=(EOS, EOS, 20)), dict(terminal_text="\n"),
                        dict(terminal_suffix=[EOS, 20])):
            with self.subTest(changes=changes), self.assertRaises(PreservationDataError):
                self.serialize(**changes)

    def test_complete_context_boundary_passes_and_one_less_fails_with_exact_full_hash(self):
        baseline = self.serialize()["examples"][0]
        length = len(baseline["input_ids"])
        self.assertEqual(self.serialize(cutoff=length)["examples"][0]["input_ids"], baseline["input_ids"])
        with self.assertRaises(CompleteContextError) as caught:
            self.serialize(cutoff=length - 1)
        failure = caught.exception.failure
        self.assertEqual(failure["reason"], "complete_pair_exceeds_cutoff")
        self.assertEqual(failure["total_tokens"], length)
        self.assertEqual(failure["full_token_ids_sha256"], value_sha256(baseline["input_ids"]))
        self.assertEqual(failure["labels_sha256"], value_sha256(baseline["labels"]))
        self.assertFalse(failure["truncated"])
        self.assertEqual(failure["row_id"], baseline["id"])

    def test_prompt_capacity_failure_is_distinct_and_never_returns_truncated_response(self):
        count = self.serialize()["examples"][0]["provenance"]["prompt_tokens"]
        with self.assertRaises(CompleteContextError) as caught:
            self.serialize(cutoff=count)
        self.assertEqual(caught.exception.failure["reason"], "prompt_has_no_response_capacity")
        self.assertGreater(caught.exception.failure["active_tokens"], len(SUFFIX))

    def test_one_overlength_selected_pair_fails_entire_cohort_without_refill(self):
        short = self.serialize([self.rows[1]["id"]])["examples"][0]
        self.assertLess(len(short["input_ids"]), len(self.serialize()["examples"][0]["input_ids"]))
        with self.assertRaises(CompleteContextError) as caught:
            self.serialize([self.rows[1]["id"], self.rows[0]["id"]], cutoff=len(short["input_ids"]))
        self.assertEqual(caught.exception.failure["row_id"], self.rows[0]["id"])

    def test_raw_pair_output_and_tokens_are_answer_sensitive_but_prompt_identity_is_not(self):
        first = self.serialize()["examples"][0]["provenance"]
        self.rows[0]["output"] += " Extra original response text."
        second = self.serialize()["examples"][0]["provenance"]
        for key in ("row_id", "prompt_messages_sha256", "rendered_prompt_sha256", "prompt_token_ids_sha256"):
            self.assertEqual(first[key], second[key])
        for key in ("row_sha256", "pair_sha256", "output_text_sha256", "rendered_full_sha256", "full_token_ids_sha256"):
            self.assertNotEqual(first[key], second[key])

    def test_identities_and_diagnostic_pilot_full_envelopes_are_separately_bound(self):
        baseline = self.serialize()["provenance"]
        for role in ("diagnostic", "pilot"):
            changed = self.serialize(context_role=role)["provenance"]
            self.assertEqual(changed["ordered_token_sha256"], baseline["ordered_token_sha256"])
            self.assertNotEqual(changed["context_role"], baseline["context_role"])
        source = source_identity()
        source["revision"] = "e" * 40
        changed = self.serialize(source_identity=source)["examples"][0]["provenance"]
        self.assertNotEqual(changed["source_identity_sha256"], baseline["pool_provenance"]["source_identity_sha256"])
        self.assertFalse(baseline["model_memory_admission_established"])
        self.assertFalse(baseline["tokenizer_files_independently_verified_here"])

    def test_stable_hashes_ignore_dictionary_insertion_order_but_not_raw_whitespace(self):
        self.assertEqual(value_sha256({"b": 2, "a": [1]}), value_sha256({"a": [1], "b": 2}))
        self.assertNotEqual(text_sha256(" answer"), text_sha256("answer"))
        self.assertEqual(self.serialize(), self.serialize())

    def test_nonfinite_nonjson_or_empty_provenance_rejected(self):
        for identity in ({}, {"nonfinite": float("nan")}, {"nonfinite": float("inf")}, {1: "nonstring key"},
                         {"set": {1}}, {"tuple": (1,)}):
            with self.subTest(identity=str(identity)), self.assertRaises(PreservationDataError):
                self.prepare(source_identity=identity)

    def test_invalid_context_and_vocabulary_arguments_rejected(self):
        for changes in (dict(cutoff=True), dict(cutoff=1), dict(context_role="implicit"), dict(context_role={}),
                        dict(vocab_size=True), dict(eos_token_id=True), dict(vocab_size=0)):
            with self.subTest(changes=changes), self.assertRaises(PreservationDataError):
                self.serialize(**changes)

    def test_bounded_replay_checks_all_four_private_witnesses_without_pool_scan(self):
        receipt = audit(self.rows)
        examples = self.serialize([row["id"] for row in self.rows], excluded_pool_ids=set(),
                                  exclusion_receipt=receipt)["examples"]
        result = self.replay(examples, exclusion_receipt_sha256=value_sha256(receipt))
        self.assertEqual(result["examples_verified"], 4)
        self.assertEqual(result["examples_sha256"], value_sha256(examples))
        self.assertFalse(result["full_pool_audit_reestablished"])
        self.assertFalse(result["selection_performed"])
        self.assertTrue(result["raw_witnesses_remain_private"])

    def test_replay_rejects_changed_tokens_masks_originals_or_provenance(self):
        original = self.serialize()["examples"]
        for alteration in ("id", "input_ids", "labels", "attention_mask", "original_row", "provenance"):
            changed = deepcopy(original)
            if alteration == "id":
                changed[0]["id"] = "0" * 64
            elif alteration == "original_row":
                changed[0]["original_row"]["output"] += " Rewritten after CPU preparation."
            elif alteration == "provenance":
                changed[0]["provenance"]["active_tokens"] += 1
            else:
                changed[0][alteration][-1] += 1
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.replay(changed)
        self.assertEqual(original, self.serialize()["examples"])

    def test_replay_rejects_changed_context_source_tokenizer_or_exclusion_binding(self):
        examples = self.serialize()["examples"]
        source = dict(source_identity(), revision="e" * 40)
        tokenizer = dict(tokenizer_identity(self.tokenizer), revision="f" * 40)
        for changes in (dict(source_identity=source), dict(tokenizer_identity=tokenizer),
                        dict(exclusion_receipt_sha256="0" * 64), dict(cutoff=100_001), dict(context_role="pilot")):
            with self.subTest(changes=changes), self.assertRaises(PreservationDataError):
                self.replay(examples, **changes)

    def test_replay_bound_shape_namespace_and_exact_json_types_fail_closed(self):
        original = self.serialize()["examples"]
        for alteration in ("empty", "too_many", "duplicate", "extra_key", "tuple_tokens", "boolean_mask"):
            changed = deepcopy(original)
            if alteration == "empty":
                changed = []
            elif alteration == "too_many":
                changed *= 5
            elif alteration == "duplicate":
                changed *= 2
            elif alteration == "extra_key":
                changed[0]["foreign"] = True
            elif alteration == "tuple_tokens":
                changed[0]["input_ids"] = tuple(changed[0]["input_ids"])
            else:
                changed[0]["attention_mask"][0] = True
            with self.subTest(alteration=alteration), self.assertRaises(PreservationDataError):
                self.replay(changed)


if __name__ == "__main__":
    unittest.main()
