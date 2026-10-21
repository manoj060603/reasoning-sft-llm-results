"""Prospective v2 worker oracles; execution belongs to native CPU QA only.

The scripted process/duplex transport tests use no child process or comparator.
The explicitly named public-scalar integration oracle uses the pinned local
grader in fresh children when a future compute allocation executes this suite.
No data download, source cohort, model, or generation is used by these fixtures.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extension import source_evaluation as v1
from extension import source_evaluation_v2 as module
from extension.labeled_feasibility import CRITERION, source_record
from extension.preserving_data import text_sha256, value_sha256
from reasoning.data import SYSTEM


REQUEST_ID = "d" * 64


def math_row(answer="2", question="What is 1+1?"):
    row, rejected = source_record("math", {"problem": question, "level": "Level 1", "type": "Algebra",
        "solution": r"A public scalar fixture. \boxed{" + answer + "}"}, {"source": "v2-fixture", "row": 0})
    assert rejected is None
    row["group_id"] = value_sha256({"criterion": CRITERION, "member_ids": [row["id"]]})
    return row


def arc_row():
    row, rejected = source_record("arc_challenge", {"id": "v2-arc", "question": "Which option is first?",
        "choices": {"label": ["2", "A", "1"], "text": ["first", "second", "third"]}, "answerKey": "2"},
        {"source": "v2-fixture-arc", "row": 0})
    assert rejected is None
    row["group_id"] = value_sha256({"criterion": CRITERION, "member_ids": [row["id"]]})
    return row


class FixtureTokenizer:
    eos_token_id = 151645
    chat_template = "v2-fixture-native-envelope"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        text = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
        return self.encode(text) if tokenize else text


def tokenizer_identity(tokenizer):
    return {"files": {"fixture.json": "a" * 64}, "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "eos_token_id": tokenizer.eos_token_id}


def expected_identity():
    return {"loaded_modules": {name: "/fixture/eval/" + name.replace(".", "/") + ".py"
            for name in module.MODULE_NAMES},
        "source_sha256": {name: "a" * 64 for name in module.MODULE_NAMES},
        "runtime_versions": {name: "fixture-1.0" for name in module.PACKAGE_NAMES}}


def comparator_proof(expected=None):
    return {**deepcopy(expected or expected_identity()), "include_percentage": False, "is_close": True,
        "internal_timeout": False, "timeout_owner": "external_two_phase_spawn_process",
        "worker_protocol": module.PROTOCOL, "setup_deadline_seconds": 10., "comparison_deadline_seconds": 5.,
        "normalization_order": "explicit_percentage_then_pinned_strip_string"}


def frame(phase, **values):
    if phase == "ready":
        defaults = {"comparator_provenance": comparator_proof(),
            "worker_setup_elapsed_seconds": .25, "worker_setup_cpu_seconds": .125}
    elif phase == "result":
        defaults = {"correct": True, "worker_comparison_elapsed_seconds": .25,
            "worker_comparison_cpu_seconds": .125}
    elif phase in {"setup_error", "comparator_error"}:
        defaults = {"error_type": "FixtureError"}
    else:
        defaults = {}
    return {"schema": module.FRAME_SCHEMA, "request_id": REQUEST_ID, "phase": phase, **defaults, **values}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class DuplexEnd:
    """Events are (absolute-or-after-GO time, bytes/exception, requires_GO)."""
    def __init__(self, context, events=()):
        self.context, self.events, self.sent, self.polls = context, list(events), [], []
        self.closed, self.receive_delay, self.send_delay = False, 0., 0.

    def poll(self, seconds=0.):
        self.polls.append(seconds)
        if not self.events or (self.events[0][2] and self.context.go_time is None):
            self.context.clock.advance(seconds)
            return False
        when, _, after_go = self.events[0]
        available = when + self.context.go_time if after_go else when
        if available <= self.context.clock() + seconds:
            self.context.clock.now = max(self.context.clock(), available)
            return True
        self.context.clock.advance(seconds)
        return False

    def recv_bytes(self, maxlength):
        _, payload, _ = self.events.pop(0)
        self.context.clock.advance(self.receive_delay)
        if isinstance(payload, BaseException):
            raise payload
        if len(payload) > maxlength:
            raise OSError("bad message length")
        return payload

    def send_bytes(self, payload):
        self.sent.append(payload)
        self.context.clock.advance(self.send_delay)
        if json.loads(payload)["phase"] == "go":
            self.context.go_time = self.context.clock()

    def close(self):
        self.closed = True


class ScriptedProcess:
    def __init__(self, context):
        self.context, self.pid, self.exitcode = context, None, None
        self.joins, self.actions, self.closed, self.alive = [], [], False, False

    def start(self):
        self.context.starts += 1
        self.context.clock.advance(self.context.start_delay)
        if self.context.start_error:
            raise self.context.start_error
        self.pid, self.alive = 123, True

    def join(self, seconds):
        self.joins.append(seconds)
        if self.context.stubborn:
            self.context.clock.advance(seconds)
        else:
            self.alive, self.exitcode = False, self.context.exitcode

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")

    def close(self):
        if self.alive:
            raise AssertionError("A still-live process must not be closed")
        self.closed = True


class Context:
    def __init__(self, events=None, *, start_delay=0., start_error=None, stubborn=False, exitcode=0):
        self.clock, self.go_time = Clock(), None
        self.start_delay, self.start_error, self.stubborn, self.exitcode = start_delay, start_error, stubborn, exitcode
        self.starts, self.process_calls, self.pipe_calls, self.process_kwargs = 0, 0, [], None
        events = events if events is not None else [(1., encoded(frame("ready")), False), (.5, encoded(frame("result")), True)]
        self.parent, self.child = DuplexEnd(self, events), DuplexEnd(self)
        self.process = ScriptedProcess(self)

    def Pipe(self, *, duplex):
        self.pipe_calls.append(duplex)
        return self.parent, self.child

    def Process(self, **kwargs):
        self.process_calls += 1
        self.process_kwargs = kwargs
        return self.process


def compare(context=None, expected=None):
    context = context or Context()
    result = module._compare_math("2", "2", expected_identity() if expected is None else expected,
        context=context, clock=context.clock, process_clock=lambda: 0., request_id=REQUEST_ID)
    return result, context


class SemanticCompatibilityTests(unittest.TestCase):
    def test_pure_semantics_are_exact_v1_callable_aliases(self):
        for name in ("validate_record", "render_prompt", "serialize_prompt", "normalize_percentage", "validate_gold",
                     "extract_final_box", "sample_seed", "completion_metrics"):
            self.assertIs(getattr(module, name), getattr(v1, name), name)
        self.assertEqual((module.SETUP_SECONDS, module.WORK_SECONDS, module.CLEANUP_SECONDS), (10., 5., .55))

    def test_record_rebuild_gold_mutation_and_original_bytes(self):
        row = math_row(question="  Preserve\n $1+1$ exactly.  ")
        before = deepcopy(row)
        self.assertEqual(module.validate_record(row)["record_sha256"], value_sha256(row))
        self.assertEqual(module.render_prompt(row), [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["original_record"]["problem"]}])
        self.assertEqual(row, before)
        changed = deepcopy(row)
        changed["original_record"]["solution"] = r"\boxed{99}"
        with self.assertRaises(ValueError):
            module.validate_record(changed)

    def test_gold_changes_cannot_change_prompt_or_native_tokens(self):
        tokenizer = FixtureTokenizer()
        for row in (math_row(), arc_row()):
            original = module.serialize_prompt(row, tokenizer, tokenizer_identity(tokenizer))
            changed = deepcopy(row)
            changed["label"] = {"answer": "LEAK", "answer_key": "LEAK"}
            changed["label_sha256"] = changed["label_id"] = "0" * 64
            changed["original_record"].pop("solution", None)
            changed["original_record"].pop("answerKey", None)
            self.assertEqual(original, module.serialize_prompt(changed, tokenizer, tokenizer_identity(tokenizer)))
            self.assertFalse(original["provenance"]["gold_fields_used"])
            self.assertFalse(original["provenance"]["truncation_performed"])
            with self.assertRaises(ValueError):
                module.serialize_prompt(row, tokenizer, tokenizer_identity(tokenizer), len(original["prompt_token_ids"]) - 1)

    def test_last_box_no_fallback_and_exact_percent_scalar_semantics(self):
        self.assertEqual(module.extract_final_box(r"\boxed{1} final \boxed{\frac{1}{2}}")["answer"], r"\frac{1}{2}")
        for suffix, status in ((r" \boxed", "malformed_final_box"), (r" \boxed{", "unbalanced_final_box"),
                               (r" \boxed{}", "empty_final_box")):
            self.assertEqual(module.extract_final_box(r"\boxed{1}" + suffix)["status"], status)
        for scalar, expected in (("50%", "1/2"), (r"50\%", "1/2"), (".5%", "1/200"),
                                 (r"\frac{1}{2}%", "1/200"), ("-12.5%", "-1/8")):
            self.assertEqual(module.normalize_percentage(scalar)["normalized"], expected)
        for scalar in ("1/0%", "x%", "50%%", "50%+1"):
            self.assertFalse(module.normalize_percentage(scalar)["supported"])

    def test_arc_keeps_original_order_and_exact_label_without_worker(self):
        row = arc_row()
        self.assertEqual(module.render_prompt(row)[1]["content"],
            "Which option is first?\n\n2. first\nA. second\n1. third\n\n" + module.ARC_INSTRUCTION)
        with patch.object(module, "_compare_math") as worker:
            self.assertTrue(module.grade_response(row, r"\boxed{2}")["correct"])
            for prediction in ("A", "a", "B", "(2)", "2."):
                result = module.grade_response(row, r"\boxed{" + prediction + "}")
                self.assertFalse(result["correct"])
                self.assertTrue(result["rankable"])
                self.assertEqual(result["denominator"], 1)
            worker.assert_not_called()

    def test_percent_normalization_precedes_worker_without_100x_equivalence(self):
        def scalar(prediction, gold, expected):
            self.assertEqual(expected, expected_identity())
            return {"status": "graded", "correct": Fraction(prediction) == Fraction(gold)}
        with patch.object(module, "_compare_math", side_effect=scalar) as worker:
            self.assertTrue(module.grade_response(math_row("50%"), r"\boxed{.5}", expected_comparator=expected_identity())["correct"])
            worker.assert_called_with(".5", "1/2", expected_identity())
            self.assertTrue(module.grade_response(math_row(".5"), r"\boxed{50\%}", expected_comparator=expected_identity())["correct"])
            for gold, prediction in (("50%", "50"), ("10", "1000"), ("10", ".1")):
                self.assertFalse(module.grade_response(math_row(gold), r"\boxed{" + prediction + "}",
                    expected_comparator=expected_identity())["correct"])

    def test_extraction_and_unsupported_paths_never_start_worker(self):
        with patch.object(module, "_compare_math") as worker:
            self.assertEqual(module.grade_response(math_row(), "2")["status"], "missing_box")
            self.assertEqual(module.grade_response(math_row(), r"\boxed{2} \boxed{")["status"], "unbalanced_final_box")
            self.assertFalse(module.grade_response(math_row("x%"), r"\boxed{2}")["rankable"])
            prediction = module.grade_response(math_row(), r"\boxed{x%}")
            self.assertTrue(prediction["rankable"])
            self.assertEqual(prediction["status"], "unsupported_prediction_percentage")
            worker.assert_not_called()

    def test_only_exact_five_second_public_comparison_budget_is_allowed(self):
        for deadline in (0, -1, True, 4.99, 5.01, 10, float("nan"), float("inf"), "5"):
            with self.assertRaises(ValueError):
                module.grade_response(arc_row(), r"\boxed{2}", timeout_seconds=deadline)


class ParentProtocolTests(unittest.TestCase):
    def test_one_fresh_duplex_process_one_payload_free_go_and_cleanup(self):
        result, context = compare()
        self.assertEqual(result["status"], "graded")
        self.assertTrue(result["correct"])
        self.assertEqual((context.process_calls, context.starts, context.pipe_calls), (1, 1, [True]))
        self.assertIs(context.process_kwargs["target"], module._math_worker)
        self.assertEqual(context.process_kwargs["args"], (context.child, "2", "2", REQUEST_ID))
        self.assertEqual([json.loads(raw) for raw in context.parent.sent], [frame("go")])
        self.assertTrue(result["go_sent"])
        self.assertIsNone(result["error_phase"])
        self.assertTrue(result["cleanup"]["successful"])
        self.assertTrue(context.parent.closed and context.child.closed and context.process.closed)

    def test_spawn_setup_and_comparison_have_separate_exact_clocks(self):
        events = [(9.8, encoded(frame("ready")), False), (4.9, encoded(frame("result")), True)]
        result, context = compare(Context(events, start_delay=9.))
        self.assertEqual(result["status"], "graded")
        self.assertAlmostEqual(result["timings"]["setup_elapsed_seconds"], 9.8)
        self.assertAlmostEqual(result["timings"]["comparison_elapsed_seconds"], 4.9)
        self.assertAlmostEqual(result["timings"]["total_elapsed_including_cleanup_seconds"], 14.7)
        self.assertEqual(result["timings"]["declared_total_budget_seconds"], 15.55)
        self.assertIn("not an OS syscall timing guarantee", result["timings"]["budget_qualification"])
        self.assertEqual(context.starts, 1)

    def test_exact_arrival_boundaries_are_accepted_but_late_frames_are_not(self):
        for ready, work, status in ((10., 5., "graded"), (10.001, 0., "setup_timeout"),
                                    (1., 5.001, "comparison_timeout")):
            result, _ = compare(Context([(ready, encoded(frame("ready")), False),
                (work, encoded(frame("result")), True)]))
            self.assertEqual(result["status"], status)
            self.assertEqual(result["go_sent"], status != "setup_timeout")

    def test_start_time_is_in_setup_budget_and_no_go_after_setup_expiry(self):
        result, context = compare(Context(start_delay=10.01))
        self.assertEqual(result["status"], "setup_timeout")
        self.assertFalse(result["go_sent"])
        self.assertEqual(context.parent.sent, [])
        self.assertEqual(context.starts, 1)

    def test_identity_validation_is_inside_setup_clock(self):
        context = Context([(9.9, encoded(frame("ready")), False)])
        original = module._validate_proof
        def verify(proof, expected):
            original(proof, expected)
            context.clock.advance(.2)
        with patch.object(module, "_validate_proof", side_effect=verify):
            result, _ = compare(context)
        self.assertEqual(result["status"], "setup_timeout")
        self.assertEqual(context.parent.sent, [])

    def test_frame_receiving_and_go_sending_are_inside_their_deadlines(self):
        context = Context([(9.9, encoded(frame("ready")), False)])
        context.parent.receive_delay = .2
        result, _ = compare(context)
        self.assertEqual(result["status"], "setup_timeout")
        context = Context()
        context.parent.send_delay = 5.1
        result, _ = compare(context)
        self.assertEqual(result["status"], "comparison_timeout")
        self.assertTrue(result["go_sent"])

    def test_missing_inexact_or_unpinned_expected_identity_fails_before_spawn(self):
        variants = [None, {}, {**expected_identity(), "extra": 1}]
        for key in ("loaded_modules", "source_sha256", "runtime_versions"):
            changed = expected_identity()
            changed[key].pop(next(iter(changed[key])))
            variants.append(changed)
        changed = expected_identity()
        changed["loaded_modules"][next(iter(module.MODULE_NAMES))] = "relative.py"
        variants.append(changed)
        changed = expected_identity()
        changed["source_sha256"][next(iter(module.MODULE_NAMES))] = "not-a-sha"
        variants.append(changed)
        for expected in variants:
            context = Context()
            result = module._compare_math("2", "2", expected, context=context,
                clock=context.clock, process_clock=lambda: 0., request_id=REQUEST_ID)
            self.assertEqual(result["status"], "setup_identity_error")
            self.assertFalse(result["correct"])
            self.assertEqual((context.process_calls, context.starts), (0, 0))

    def test_ready_must_match_every_pinned_path_hash_version_and_flag(self):
        mutations = []
        for key in ("loaded_modules", "source_sha256", "runtime_versions"):
            proof = comparator_proof()
            proof[key][next(iter(proof[key]))] = "unexpected"
            mutations.append(proof)
        for key, value in (("include_percentage", True), ("is_close", False), ("internal_timeout", True),
                           ("timeout_owner", "old-owner"), ("worker_protocol", "v1"),
                           ("setup_deadline_seconds", 5), ("comparison_deadline_seconds", 10),
                           ("normalization_order", "changed")):
            mutations.append({**comparator_proof(), key: value})
        for proof in mutations:
            result, context = compare(Context([(1., encoded(frame("ready", comparator_provenance=proof)), False)]))
            self.assertEqual(result["status"], "setup_identity_error")
            self.assertEqual(context.parent.sent, [])

    def test_setup_error_and_wrong_phase_before_ready_never_send_go(self):
        for payload, status in ((frame("setup_error"), "setup_error"), (frame("result"), "protocol_error"),
                                 (frame("go"), "protocol_error"), (frame("comparator_error"), "protocol_error")):
            result, context = compare(Context([(1., encoded(payload), False)]))
            self.assertEqual(result["status"], status)
            self.assertEqual(context.parent.sent, [])
            self.assertEqual(context.starts, 1)

    def test_premature_or_duplicate_frame_before_go_is_protocol_error(self):
        for extra in (frame("result"), frame("ready"), frame("go")):
            result, context = compare(Context([(1., encoded(frame("ready")), False), (1., encoded(extra), False)]))
            self.assertEqual(result["status"], "protocol_error")
            self.assertFalse(result["go_sent"])
            self.assertEqual(context.parent.sent, [])

    def test_crash_or_eof_is_classified_in_either_phase_without_retry(self):
        for events, phase in (([(1., EOFError(), False)], "setup"),
                             ([(1., encoded(frame("ready")), False), (.1, EOFError(), True)], "comparison")):
            result, context = compare(Context(events, exitcode=17))
            self.assertEqual(result["status"], "worker_crash")
            self.assertEqual(result["error_phase"], phase)
            self.assertFalse(result["correct"])
            self.assertEqual((context.starts, context.process_calls), (1, 1))
        result, context = compare(Context(start_error=RuntimeError("spawn failed")))
        self.assertEqual(result["status"], "worker_error")
        self.assertEqual(context.starts, 1)

    def test_post_go_requires_boolean_result_or_explicit_comparator_error(self):
        for payload, status in ((frame("result", correct=False), "graded"),
                (frame("comparator_error", error_type="NormalizerFailure"), "comparator_error"),
                (frame("result", correct="true"), "protocol_error"), (frame("ready"), "protocol_error"),
                (frame("go"), "protocol_error")):
            result, context = compare(Context([(1., encoded(frame("ready")), False), (.1, encoded(payload), True)]))
            self.assertEqual(result["status"], status)
            self.assertFalse(result["correct"])
            self.assertEqual(len(context.parent.sent), 1)

    def test_duplicate_result_is_rejected_but_post_result_eof_is_allowed(self):
        for extra, status in ((encoded(frame("result")), "protocol_error"), (EOFError(), "graded")):
            result, _ = compare(Context([(1., encoded(frame("ready")), False),
                (.1, encoded(frame("result")), True), (.1, extra, True)]))
            self.assertEqual(result["status"], status)

    def test_malformed_oversized_wrong_request_or_extra_fields_fail_closed(self):
        wrong_id = frame("ready", request_id="e" * 64)
        wrong_schema = frame("ready", schema="old-schema")
        missing = frame("ready")
        missing.pop("worker_setup_cpu_seconds")
        for payload in (b"not-json", b"[]", encoded(wrong_id), encoded(wrong_schema), encoded(missing),
                        encoded(frame("ready", unexpected="payload")), b"x" * (module.MAX_FRAME_BYTES + 1)):
            result, context = compare(Context([(1., payload, False)]))
            self.assertEqual(result["status"], "protocol_error")
            self.assertEqual(context.parent.sent, [])

    def test_duplicate_json_keys_and_nonfinite_or_invalid_timings_rejected(self):
        raw = encoded(frame("ready"))
        duplicate = raw[:-1] + b',"phase":"ready"}'
        payloads = [duplicate]
        for key in ("worker_setup_elapsed_seconds", "worker_setup_cpu_seconds"):
            for value in (True, -1, float("nan"), float("inf"), "0"):
                payloads.append(encoded(frame("ready", **{key: value})))
        for payload in payloads:
            result, _ = compare(Context([(1., payload, False)]))
            self.assertEqual(result["status"], "protocol_error")
        for value in (True, -1, float("nan"), "0"):
            result, _ = compare(Context([(1., encoded(frame("ready")), False),
                (.1, encoded(frame("result", worker_comparison_elapsed_seconds=value)), True)]))
            self.assertEqual(result["status"], "protocol_error")

    def test_nonzero_exit_after_valid_result_cannot_be_reported_as_graded(self):
        result, _ = compare(Context(exitcode=17))
        self.assertEqual(result["status"], "worker_crash")
        self.assertFalse(result["correct"])

    def test_bounded_cleanup_escalates_without_closing_live_worker_or_masking_prior(self):
        result, context = compare(Context(events=[], stubborn=True))
        self.assertEqual(result["status"], "cleanup_error")
        self.assertEqual(result["prior_status"], "setup_timeout")
        self.assertTrue(result["caller_must_abort"])
        self.assertFalse(result["correct"])
        self.assertEqual(context.process.actions, ["terminate", "kill"])
        self.assertEqual(len(context.process.joins), 3)
        for actual, expected in zip(context.process.joins, [.05, .25, .25]):
            self.assertAlmostEqual(actual, expected)
        self.assertLessEqual(sum(context.process.joins), module.CLEANUP_SECONDS + 1e-12)
        self.assertFalse(context.process.closed)
        self.assertTrue(result["cleanup"]["worker_alive"])
        self.assertEqual(result["error_phase"], "cleanup")

    def test_connection_close_error_cannot_skip_process_cleanup_or_hide_abort(self):
        context = Context()
        def close_error():
            raise OSError("fixture close failure")
        context.parent.close = close_error
        result, _ = compare(context)
        self.assertEqual((result["status"], result["prior_status"]), ("cleanup_error", "graded"))
        self.assertEqual(result["cleanup"]["connection_close_error_type"], "OSError")
        self.assertTrue(result["caller_must_abort"])
        self.assertTrue(context.process.closed)

    def test_invalid_request_identity_is_rejected_without_any_process(self):
        for request in ("bad", "D" * 64, "d" * 63, 17, True):
            context = Context()
            with self.assertRaises(ValueError):
                module._compare_math("2", "2", expected_identity(), context=context,
                    clock=context.clock, process_clock=lambda: 0., request_id=request)
            self.assertEqual(context.process_calls, 0)

    def test_pipe_construction_error_has_no_worker_and_preserves_public_denominator(self):
        class BrokenPipe(Context):
            def Pipe(self, *, duplex):
                self.pipe_calls.append(duplex)
                raise OSError("fixture pipe construction failure")
        context = BrokenPipe()
        result, _ = compare(context)
        self.assertEqual((result["status"], result["error_phase"]), ("worker_error", "setup"))
        self.assertEqual(result["error_type"], "OSError")
        self.assertEqual((context.process_calls, context.starts, context.parent.sent), (0, 0, []))
        self.assertFalse(result["go_sent"])
        public_context = BrokenPipe()
        with patch.object(module.multiprocessing, "get_context", return_value=public_context):
            grade = module.grade_response(math_row(), r"\boxed{2}", expected_comparator=expected_identity())
        self.assertEqual((grade["denominator"], grade["rankable"], grade["correct"]), (1, False, False))
        self.assertEqual(grade["error"], {"type": "OSError", "phase": "setup"})

    def test_process_construction_error_closes_both_pipe_ends_and_keeps_denominator(self):
        class BrokenProcess(Context):
            def Process(self, **kwargs):
                self.process_calls += 1
                raise RuntimeError("fixture process construction failure")
        context = BrokenProcess()
        result, _ = compare(context)
        self.assertEqual((result["status"], result["error_phase"]), ("worker_error", "setup"))
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertTrue(context.parent.closed and context.child.closed)
        self.assertEqual((context.process_calls, context.starts, context.parent.sent), (1, 0, []))
        self.assertFalse(result["go_sent"])
        public_context = BrokenProcess()
        with patch.object(module.multiprocessing, "get_context", return_value=public_context):
            grade = module.grade_response(math_row(), r"\boxed{2}", expected_comparator=expected_identity())
        self.assertEqual((grade["denominator"], grade["rankable"], grade["correct"]), (1, False, False))
        self.assertEqual(grade["error"], {"type": "RuntimeError", "phase": "setup"})
        self.assertTrue(public_context.parent.closed and public_context.child.closed)


class WorkerPhaseTests(unittest.TestCase):
    def worker(self, controls=None, *, setup_error=None, comparator_error=None):
        context = Context()
        controls = controls if controls is not None else [frame("go")]
        connection = DuplexEnd(context, [(0., encoded(control), False) for control in controls])
        calls = []
        def normalizer(text):
            self.assertEqual([json.loads(raw)["phase"] for raw in connection.sent], ["ready"])
            self.assertEqual(connection.events, [], "Answer work started before the sole GO was consumed")
            calls.append(("normalize", text))
            return "normalized:" + text
        def comparator(prediction, gold, **kwargs):
            calls.append(("compare", prediction, gold, kwargs))
            if comparator_error:
                raise comparator_error
            return True
        def load():
            calls.append(("load",))
            if setup_error:
                raise setup_error
            return comparator, normalizer, comparator_proof()
        with patch.object(module, "_actual_comparator", side_effect=load), \
                patch.object(module.time, "monotonic", context.clock), patch.object(module.time, "process_time", context.clock):
            module._math_worker(connection, "prediction", "gold", REQUEST_ID)
        return [json.loads(raw) for raw in connection.sent], calls, connection

    def test_normalization_and_comparison_wait_for_one_valid_go(self):
        messages, calls, connection = self.worker()
        self.assertEqual([message["phase"] for message in messages], ["ready", "result"])
        self.assertEqual(calls, [("load",), ("normalize", "prediction"), ("normalize", "gold"),
            ("compare", "normalized:prediction", "normalized:gold",
             {"include_percentage": False, "is_close": True, "timeout": False})])
        self.assertTrue(connection.closed)

    def test_setup_failure_never_sends_ready_or_does_answer_work(self):
        messages, calls, connection = self.worker(setup_error=ImportError("fixture"))
        self.assertEqual([message["phase"] for message in messages], ["setup_error"])
        self.assertEqual(messages[0]["error_type"], "ImportError")
        self.assertEqual(calls, [("load",)])
        self.assertTrue(connection.closed)

    def test_missing_wrong_or_duplicate_go_never_normalizes_answers(self):
        for controls in ([], [frame("result")], [frame("go", unexpected=1)], [frame("go"), frame("go")]):
            messages, calls, _ = self.worker(controls)
            self.assertEqual([message["phase"] for message in messages], ["ready", "setup_error"])
            self.assertEqual(calls, [("load",)])

    def test_comparator_exception_is_distinct_from_cold_setup_error(self):
        messages, calls, _ = self.worker(comparator_error=ArithmeticError("fixture"))
        self.assertEqual([message["phase"] for message in messages], ["ready", "comparator_error"])
        self.assertEqual(messages[-1]["error_type"], "ArithmeticError")
        self.assertEqual(calls[-1][0], "compare")

    def test_actual_comparator_hashes_exact_loaded_module_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = {}
            for index, name in enumerate(sorted(module.MODULE_NAMES)):
                path = Path(temporary) / (name.replace(".", "_") + ".py")
                path.write_bytes(("# fixture " + str(index)).encode())
                files[name] = str(path)
            old = comparator_proof()
            old["loaded_modules"] = files
            old.pop("source_sha256")
            with patch.object(v1, "_load_pinned_comparator", return_value=("comparator", "normalizer", old)):
                comparator, normalizer, proof = module._actual_comparator()
            self.assertEqual((comparator, normalizer), ("comparator", "normalizer"))
            self.assertEqual(proof["source_sha256"], {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in files.items()})
            module._validate_proof(proof, {key: proof[key] for key in expected_identity()})


class DenominatorAndIntegrationTests(unittest.TestCase):
    def samples(self, row):
        return [{"sample_index": index, "request_seed": module.sample_seed(row["id"]),
            "grade": module.grade_response(row, r"\boxed{2}" if index < 3 else r"\boxed{A}"),
            "metrics": module.completion_metrics(r"\boxed{2}" if index < 3 else r"\boxed{A}",
                [7, 151645], "stop", None, 151645)} for index in range(8)]

    def test_each_operational_error_preserves_one_denominator_and_blocks_ranking(self):
        for status in sorted(module.ERROR_STATUSES - {"cleanup_error"}):
            failure = {"status": status, "correct": False, "error_type": "FixtureFailure", "error_phase": "setup"}
            with patch.object(module, "_compare_math", return_value=failure):
                result = module.grade_response(math_row(), r"\boxed{2}", expected_comparator=expected_identity())
            self.assertEqual((result["denominator"], result["rankable"], result["correct"]), (1, False, False))
            self.assertEqual(result["error"], {"type": "FixtureFailure", "phase": "setup"})

    def test_cleanup_failure_raises_complete_private_receipt_and_cannot_be_ignored(self):
        comparison = {"status": "cleanup_error", "correct": False, "error_type": "BoundedWorkerCleanupFailed",
            "prior_status": "graded", "caller_must_abort": True}
        with patch.object(module, "_compare_math", return_value=comparison):
            with self.assertRaises(module.SourceGraderCleanupError) as caught:
                module.grade_response(math_row(), r"\boxed{2}", expected_comparator=expected_identity())
        result = caught.exception.result
        self.assertEqual((result["denominator"], result["rankable"], result["correct"]), (1, False, False))
        self.assertTrue(result["caller_must_abort"])
        self.assertEqual(result["worker"]["prior_status"], "graded")
        self.assertEqual(result["response_sha256"], text_sha256(r"\boxed{2}"))
        comparison["prior_status"] = "changed"
        self.assertEqual(result["worker"]["prior_status"], "graded")

    def test_sample_error_adapter_preserves_exact_inventory_and_eight_denominators(self):
        row = arc_row()
        for status in sorted(module.ERROR_STATUSES - {"cleanup_error"}):
            samples = self.samples(row)
            samples[0]["grade"].update(status=status, correct=False, rankable=False, error={"type": "Fixture"})
            before = deepcopy(samples)
            result = module.summarize_samples(row, list(reversed(samples)))
            self.assertEqual((result["denominator"], result["sample_count"], result["grading_errors"]), (8, 8, 1))
            self.assertEqual((result["pass_at_1"], result["pass_at_8"]), (2 / 8, 1.))
            self.assertFalse(result["rankable"])
            self.assertEqual(result["sample_inventory_sha256"], value_sha256(samples))
            self.assertEqual(samples, before)

    def test_cleanup_recorded_sample_is_fatal_and_malformed_errors_cannot_count(self):
        row = arc_row()
        samples = self.samples(row)
        samples[0]["grade"].update(status="cleanup_error", correct=False, rankable=False,
            error={"type": "Cleanup"}, caller_must_abort=True)
        with self.assertRaises(module.SourceGraderCleanupError) as caught:
            module.summarize_samples(row, samples)
        self.assertEqual(caught.exception.result["denominator"], 1)
        for change in ({"correct": True}, {"rankable": True}, {"denominator": 0}, {"error": None}):
            samples = self.samples(row)
            samples[0]["grade"].update(status="setup_timeout", correct=False, rankable=False, error={"type": "Fixture"})
            samples[0]["grade"].update(change)
            with self.assertRaises(ValueError):
                module.summarize_samples(row, samples)

    def test_eight_sample_identity_seed_hash_abort_and_length_guards_are_retained(self):
        row = arc_row()
        samples = self.samples(row)
        summary = module.summarize_samples(row, samples)
        self.assertEqual((summary["pass_at_1"], summary["pass_at_8"]), (3 / 8, 1.))
        for mutate in (lambda value: value.pop(), lambda value: value[0].update(sample_index=1),
                       lambda value: value[0].update(request_seed=0),
                       lambda value: value[0]["metrics"].update(response_sha256="0" * 64)):
            changed = deepcopy(samples)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.summarize_samples(row, changed)
        samples[0]["metrics"]["finish_reason"] = "abort"
        self.assertFalse(module.summarize_samples(row, samples)["rankable"])
        samples[0]["metrics"] = module.completion_metrics(r"\boxed{2}", [7], "length", None, 151645)
        summary = module.summarize_samples(row, samples)
        self.assertTrue(summary["rankable"])
        self.assertEqual(summary["length_finishes"], 1)

    def test_actual_pinned_public_scalar_workers_use_new_protocol_without_scaling(self):
        # These four public constants are not labeled-source or benchmark data.
        # This integration oracle must execute only in the future native CPU QA.
        directory = Path(module.__file__).resolve().parents[1] / "eval" / "utils"
        expected = {"loaded_modules": {name: str((directory / filename).resolve())
                for name, filename in v1._GRADER_FILES.items()},
            "source_sha256": {name: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                for name, filename in v1._GRADER_FILES.items()},
            "runtime_versions": {name: version(name) for name in module.PACKAGE_NAMES}}
        # Preserve the exact four declared controls from the cold-start
        # diagnostic, including the formerly timing-sensitive fraction-vs-50.
        for gold, prediction, correct in ((".5", "50%", True), ("50", "50%", False),
                                         ("10", "1000", False), ("10", ".1", False)):
            result = module.grade_response(math_row(gold), r"\boxed{" + prediction + "}", expected_comparator=expected)
            self.assertTrue(result["rankable"], result)
            self.assertEqual(result["status"], "graded", result)
            self.assertIs(result["correct"], correct, result)
            self.assertTrue(result["worker"]["go_sent"])
            proof = result["comparator_provenance"]
            for key in expected:
                self.assertEqual(proof[key], expected[key])
            self.assertEqual(proof["worker_protocol"], module.PROTOCOL)
            self.assertFalse(proof["include_percentage"])
            self.assertFalse(proof["internal_timeout"])
            self.assertEqual(result["worker"]["timings"]["declared_setup_deadline_seconds"], 10.)
            self.assertEqual(result["worker"]["timings"]["declared_comparison_deadline_seconds"], 5.)


if __name__ == "__main__":
    unittest.main()
