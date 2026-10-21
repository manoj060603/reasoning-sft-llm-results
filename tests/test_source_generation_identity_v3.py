"""Unadmitted synthetic collector oracles; native CPU QA owns execution.

No original weights, real calibration row/cache, vLLM, or model generation is
used by the fixture path. Passing these tests cannot prove the native interface.
"""
from contextlib import ExitStack
from copy import deepcopy
import ast
import builtins
import hashlib
import json
from pathlib import Path
import stat
import traceback
from types import SimpleNamespace
from types import ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

from extension import source_generation_v3 as module
from extension import source_generation as predecessor
from extension.cpu_feasibility import value_digest
from extension.preserving_data import text_sha256
from extension.submission import SubmissionError
from test_source_readiness_v2_retry import Tokenizer, records, successful_grade, expected_comparator


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_ASSIGNMENTS = {"ROLE", "PLAN", "PLAN_SCHEMA", "QA_SCHEMA", "ADMISSION_SCHEMA", "SCHEMA", "NEW_SOURCES", "METADATA_SOURCES"}
ADDED_ASSIGNMENTS = {"SCIENTIFIC_PARENT", "FROZEN_COLLECTOR_SHA256", "IDENTITY_HELPER", "IDENTITY_DIRECTORY", "IDENTITY_PINS",
                     "OPERATIONAL_PARENT", "RETRY_FAILURE"}
OPERATIONAL_FUNCTIONS = {"validate_plan", "source_inventory", "_current_admission", "load_inputs"}
ADDED_IMPORTS = {"from extension import gpu_identity_v2", "from extension import joint_identity_replay_v3_qa as identity_parent",
                 "from extension import source_generation as scientific_parent",
                 "from extension import source_generation_v2 as operational_parent"}


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


def assignment_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None


def scientific_projection(source, *, versioned):
    """Remove only named metadata additions, then inverse the two runtime edits.

    Operational function bodies are explicitly reviewed separately; their
    signatures stay exact. Every other top-level body has exact AST AND text.
    """
    result = []
    for node in ast.parse(source).body:
        segment = ast.get_source_segment(source, node)
        name = assignment_name(node)
        if versioned and (name in ADDED_ASSIGNMENTS or segment in ADDED_IMPORTS or
                          isinstance(node, ast.FunctionDef) and node.name in {"_identity_parent_evidence", "retry_failure_evidence"}):
            continue
        if name in OPERATIONAL_ASSIGNMENTS:
            result.append(("operational_assignment", name))
            continue
        if isinstance(node, ast.FunctionDef) and node.name in OPERATIONAL_FUNCTIONS:
            signature = deepcopy(node)
            signature.body = [ast.Pass()]
            result.append(("operational_signature", node.name, ast_dump(signature)))
            continue
        if versioned and isinstance(node, ast.ClassDef) and node.name == "NativeEngine":
            old = "self.hardware = gpu_identity_v2.assigned_gpu_identity(torch)"
            if segment.count(old) != 1:
                raise AssertionError("Exactly one production identity delegation is required")
            segment = segment.replace(old, "self.hardware = native.assigned_gpu_identity(torch)", 1)
        if versioned and isinstance(node, ast.FunctionDef) and node.name == "run":
            old = '"source-generation-original-v3-"'
            if segment.count(old) != 1:
                raise AssertionError("Exactly one output namespace change is required")
            segment = segment.replace(old, '"source-generation-original-"', 1)
        result.append(("unchanged", ast_dump(ast.parse(segment)), segment))
    return result


def assert_scientific_replay(original, candidate):
    if scientific_projection(original, versioned=False) != scientific_projection(candidate, versioned=True):
        raise AssertionError("A non-allowlisted collector function, constant, import, or inference path changed")


def plan():
    return json.loads((Path(module.__file__).resolve().parents[1] / module.PLAN).read_text())


def cache():
    tokenizer = Tokenizer()
    identity = {"repository": module.native.REPOSITORY, "revision": module.native.REVISION,
        "files": dict(module.native.TOKENIZER_FILES), "eos_token_id": 151645,
        "chat_template_sha256": text_sha256(tokenizer.chat_template)}
    rows, _ = module.prepared_stage.build_calibration_cache(records(), tokenizer, identity,
        expected_comparator=expected_comparator(), grader=successful_grade)
    return {"schema": module.prepared_stage.CACHE_SCHEMA, "complete": True,
        "plan_sha256": module.prepared_stage.SCIENTIFIC_PLAN["sha256"],
        "execution_plan_sha256": "7526ff1b21dd2a8a53c39f47c91981ffe91496cd5f1270b9cddf50974cf3c39b",
        "source_generation_profile": deepcopy(module.PROFILE), "generation_performed": False,
        "training_performed": False, "raw_records_private": True,
        "model": {"repository": module.native.REPOSITORY, "revision": module.native.REVISION,
            "files": dict(module.native.ORIGINAL_FILES), "path": "/fixture/pinned-original"},
        "tokenizer_identity": identity, "records": rows,
        "calibration_ids_sha256": value_digest([row["record"]["id"] for row in rows])}


def output(request, *, request_id=None):
    return SimpleNamespace(request_id=request_id or "engine-" + request["record_id"][:12], finished=True,
        prompt_token_ids=list(request["prompt_token_ids"]), metrics=SimpleNamespace(arrival_time=1.,
            first_scheduled_time=1.1, first_token_time=1.2, last_token_time=1.3, finished_time=1.4),
        outputs=[SimpleNamespace(index=index, text="  Raw fixture " + str(index) + r" \boxed{2}" + "\n",
            token_ids=[100 + index, 151645], finish_reason="stop", stop_reason=None) for index in range(8)])


class FixtureEvidence:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def unchanged(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("Synthetic input snapshot changed")


class RunFixture:
    """Guarded-run seam, not an attestation of the actual sealed cache/model."""
    def __init__(self, temporary):
        self.root = Path(temporary).resolve()
        self.module_file = self.root / "extension/source_generation_v3.py"
        self.module_file.parent.mkdir(parents=True)
        self.module_file.write_text("# Synthetic generation orchestration witness.\n")
        self.allocation = {"job_id": "990", "node": "owl123", "gpu_name": "NVIDIA B200"}
        self.output = self.root / "extension/runs/source-generation-original-v3-990"
        self.value = cache()
        self.requests = module.build_requests(self.value)
        self.sources = {"extension/source_generation_v3.py": {"sha256": module.digest(self.module_file),
            "bytes": self.module_file.stat().st_size}}
        self.admission = {"binding_sha256": "d" * 64}
        self.evidence = FixtureEvidence()
        self.context = {"root": self.root, "plan": plan(), "plan_sha256": "a" * 64,
            "cache": self.value, "cache_proof": module.validate_cache(self.value), "requests": self.requests,
            "base": Path("/fixture/pinned-original"), "sources": self.sources, "qa_sha256": "b" * 64,
            "admission": self.admission, "admission_sha256": "c" * 64, "evidence": self.evidence,
            "versions": {"vllm": "fixture-only-never-imported"}}
        self.factory_calls, self.generate_calls, self.engines = [], [], []

    def factory(self, config):
        fixture = self
        self.factory_calls.append(deepcopy(config))
        class Engine:
            def generate(self, requests):
                fixture.generate_calls.append(deepcopy(requests))
                print("PRIVATE-ENGINE-MARKER")
                return [output(row, request_id="engine-" + str(index)) for index, row in enumerate(requests)]

            def proof(self):
                return {"native_engine": True, "one_generate_call": True, "request_count": 32,
                    "native_eos_token_id": 151645, "labels_entered_engine": False, "fixture_proof_only": True}
        engine = Engine()
        self.engines.append(engine)
        return engine

    def run(self, *, factory=None, ending=None, source_change=False, admission_change=False, proof_error=None):
        source_calls, admission_calls = 0, 0
        def sources(root):
            nonlocal source_calls
            source_calls += 1
            value = deepcopy(self.sources)
            if source_change:
                value[next(iter(value))]["sha256"] = "0" * 64
            return value
        def admission(*args):
            nonlocal admission_calls
            admission_calls += 1
            if admission_change and admission_calls > 1:
                return {"binding_sha256": "0" * 64}
            return self.admission
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "__file__", str(self.module_file)))
            allocation = stack.enter_context(patch.object(module.native, "require_gpu_allocation",
                side_effect=proof_error if proof_error else [self.allocation, self.allocation if ending is None else ending]))
            stack.enter_context(patch.object(module.native, "_host_resources"))
            stack.enter_context(patch.object(module, "deadline_guard"))
            loader = stack.enter_context(patch.object(module, "load_inputs", return_value=self.context))
            stack.enter_context(patch.object(module, "source_inventory", side_effect=sources))
            stack.enter_context(patch.object(module, "_current_admission", side_effect=admission))
            stack.enter_context(patch.object(module, "replay_prompts", return_value={"requests_replayed": 32,
                "native_eos_token_id": 151645, "labels_used_for_generation": False}))
            self.loader, self.allocation_mock = loader, allocation
            return module.run(self.root, module.PLAN, self.output, "extension/runs/unit-qa.json",
                "extension/operations/admission.json", engine_factory=self.factory if factory is None else factory)


class PlanAndCacheTests(unittest.TestCase):
    def test_plan_keeps_pinned_original_cohort_profile_resources_and_no_downstream(self):
        value = plan()
        self.assertIs(module.validate_plan(value), value)
        self.assertEqual(value["model_role"], "original")
        self.assertEqual((value["request_count"], value["sample_count"]), (32, 256))
        self.assertEqual(value["profile"], module.PROFILE)
        self.assertEqual((value["profile"]["n"], value["profile"]["max_new_tokens"],
            value["profile"]["max_model_len"], value["profile"]["batch_size"]), (8, 8192, 16384, 32))
        self.assertEqual(value["resources"], module.RESOURCE)
        self.assertEqual(value_digest(value["profile"]), module.PROFILE_SHA256)
        self.assertEqual(value["engine_adapter"], module.ENGINE_ADAPTER)
        self.assertTrue(value["one_generate_call"] and value["first_native_interface_check"])
        self.assertFalse(any(value[key] for key in ("training", "grading", "ranking", "automatic_retry", "automatic_downstream")))

    def test_plan_rejects_other_models_seeds_stops_sampling_caps_or_gpu_scope(self):
        original = plan()
        for change in (lambda value: value.update(model_role="recovery"), lambda value: value.update(request_count=31),
            lambda value: value.update(sample_count=255), lambda value: value["profile"].update(seed=7),
            lambda value: value["profile"].update(n=1), lambda value: value["profile"].update(max_new_tokens=4096),
            lambda value: value["profile"].update(max_model_len=32768),
            lambda value: value["profile"].update(additional_stop_strings=["answer"]),
            lambda value: value["profile"].update(ignore_eos=True), lambda value: value["resources"].update(gpus=2),
            lambda value: value["resources"].update(gpu_type="A100"), lambda value: value.update(automatic_retry=True),
            lambda value: value.update(grading=True), lambda value: value.update(automatic_downstream=True),
            lambda value: value.update(engine_adapter={})):
            changed = deepcopy(original)
            change(changed)
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_plan_binds_exact_parent_cache_source_and_new_deadline(self):
        original = plan()
        for change in (lambda value: value["cache"].update(sha256="0" * 64),
            lambda value: value["source_plan"].update(sha256="0" * 64),
            lambda value: value["parent_evidence"]["complete.json"].update(sha256="0" * 64),
            lambda value: value.update(deadline_utc="2026-10-29T03:59:00+00:00"),
            lambda value: value.update(one_generate_call=False), lambda value: value.update(first_native_interface_check=False)):
            changed = deepcopy(original)
            change(changed)
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_synthetic_cache_has_exact32_native_prompts_and256_reserved_samples(self):
        value = cache()
        before = deepcopy(value)
        proof = module.validate_cache(value, plan())
        self.assertEqual((proof["record_count"], proof["sample_count"]), (32, 256))
        self.assertEqual(proof["record_ids_sha256"], value["calibration_ids_sha256"])
        self.assertEqual(value, before)

    def test_cache_missing_extra_duplicate_reordered_domain_and_group_rows_fail(self):
        original = cache()
        cases = []
        for rows in (original["records"][:-1], original["records"] + [original["records"][0]],
                     list(reversed(original["records"])), [original["records"][0]] + original["records"][:-1]):
            changed = deepcopy(original)
            changed["records"] = deepcopy(rows)
            cases.append(changed)
        for field, value in (("group_id", original["records"][0]["record"]["group_id"]), ("source_kind", "other")):
            changed = deepcopy(original)
            changed["records"][1]["record"][field] = value
            cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):
                module.validate_cache(changed)

    def test_cache_prompt_token_text_native_identity_and_seed_tampering_fail(self):
        original = cache()
        mutations = (lambda value: value["records"][0]["prompt"]["prompt_token_ids"].append(8),
            lambda value: value["records"][0]["prompt"].update(rendered_prompt="rewritten"),
            lambda value: value["records"][0].update(request_seed=0),
            lambda value: value["records"][0].update(sample_indices=list(range(7))),
            lambda value: value["records"][0]["prompt"]["provenance"].update(gold_fields_used=True),
            lambda value: value["records"][0]["prompt"]["provenance"].update(truncation_performed=True),
            lambda value: value["records"][0]["prompt"]["provenance"].update(tokenizer_identity_sha256="0" * 64),
            lambda value: value.update(calibration_ids_sha256="0" * 64))
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.validate_cache(changed)

    def test_cache_only_original_model_tokenizer_and_no_prior_generation_are_allowed(self):
        original = cache()
        mutations = (lambda value: value["model"].update(revision="different"),
            lambda value: value["model"]["files"].update({"config.json": "0" * 64}),
            lambda value: value["tokenizer_identity"].update(eos_token_id=0),
            lambda value: value["tokenizer_identity"]["files"].pop("vocab.json"),
            lambda value: value.update(complete=False), lambda value: value.update(generation_performed=True),
            lambda value: value.update(training_performed=True), lambda value: value.update(raw_records_private=False),
            lambda value: value.update(plan_sha256="0" * 64), lambda value: value.update(execution_plan_sha256="0" * 64))
        for mutate in mutations:
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.validate_cache(changed)


class BlindRequestTests(unittest.TestCase):
    def test_engine_request_whitelist_contains_only_native_ids_seeds_and_hashes(self):
        value = cache()
        requests = module.build_requests(value)
        expected = {"record_id", "source_kind", "prompt_token_ids", "request_seed", "prompt_sha256",
            "prompt_token_ids_sha256", "model_identity_sha256", "tokenizer_identity_sha256"}
        self.assertEqual(len(requests), 32)
        for request, row in zip(requests, value["records"]):
            self.assertEqual(set(request), expected)
            self.assertEqual(request["prompt_token_ids"], row["prompt"]["prompt_token_ids"])
            self.assertNotIn("label", request)
            self.assertNotIn("prompt", request)
            self.assertNotIn("messages", request)
            self.assertNotIn("rendered_prompt", request)

    def test_changing_or_deleting_gold_solution_and_answer_key_cannot_change_requests(self):
        original = cache()
        changed = deepcopy(original)
        for row in changed["records"]:
            row["record"]["label"] = {"answer": "PRIVATE-GOLD-MUST-NOT-REACH-ENGINE", "answer_key": "SECRET"}
            row["record"].pop("label_sha256", None)
            row["record"]["original_record"].pop("solution", None)
            row["record"]["original_record"].pop("answerKey", None)
        self.assertEqual(module.build_requests(changed), module.build_requests(original))
        self.assertNotIn("PRIVATE-GOLD", json.dumps(module.build_requests(changed)))

    def test_requests_keep_sorted_record_seeds_exact_context_and_defensive_token_copy(self):
        value = cache()
        before = deepcopy(value)
        requests = module.build_requests(value)
        self.assertEqual([row["record_id"] for row in requests], sorted(row["record_id"] for row in requests))
        for request in requests:
            self.assertEqual(request["request_seed"], (20261021 + int(request["record_id"][:8], 16)) % 2 ** 31)
            self.assertEqual(value_digest(request["prompt_token_ids"]), request["prompt_token_ids_sha256"])
            self.assertLessEqual(len(request["prompt_token_ids"]) + 8192, 16384)
        requests[0]["prompt_token_ids"].append(9)
        self.assertEqual(value, before)


class CompletionCollectionTests(unittest.TestCase):
    def request(self):
        return module.build_requests(cache())[0]

    def test_all_eight_raw_samples_text_tokens_eos_hashes_and_engine_indices_are_preserved(self):
        request = self.request()
        observed = output(request)
        raw = module.snapshot_output(observed)
        result = module.collect_record(request, observed)
        self.assertEqual([sample["sample_index"] for sample in result["samples"]], list(range(8)))
        self.assertEqual(result["record_id"], request["record_id"])
        self.assertFalse(result["grading_performed"])
        self.assertEqual(result["raw_engine_observation_sha256"], value_digest(raw))
        for sample, native in zip(result["samples"], observed.outputs):
            self.assertEqual(sample["text"], native.text)
            self.assertEqual(sample["token_ids"], native.token_ids)
            self.assertEqual(sample["text_sha256"], text_sha256(native.text))
            self.assertEqual(sample["token_ids_sha256"], value_digest(native.token_ids))
            self.assertEqual((sample["finish_reason"], sample["stop_reason"], sample["output_tokens"]), ("stop", None, 2))

    def test_reversed_native_order_sorts_by_index_and_retains_original_positions(self):
        request = self.request()
        observed = output(request)
        observed.outputs.reverse()
        result = module.collect_record(request, observed)
        self.assertEqual([sample["sample_index"] for sample in result["samples"]], list(range(8)))
        self.assertEqual([sample["engine_output_position"] for sample in result["samples"]], list(reversed(range(8))))
        self.assertEqual(result["raw_engine_observation_sha256"], value_digest(module.snapshot_output(observed)))

    def test_missing_extra_or_duplicate_samples_are_never_dropped_or_refilled(self):
        request = self.request()
        for mutate in (lambda value: value.outputs.pop(), lambda value: value.outputs.append(deepcopy(value.outputs[0])),
                       lambda value: setattr(value.outputs[0], "index", 1)):
            observed = output(request)
            mutate(observed)
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)

    def test_sample_index_must_be_plain_unique_integer_in_zero_to_seven(self):
        request = self.request()
        for index in (True, -1, 8, "0", None, .5):
            observed = output(request)
            observed.outputs[0].index = index
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)

    def test_abort_unknown_or_unfinished_outputs_never_count_as_success(self):
        request = self.request()
        for reason in ("abort", "error", "pending", None, "finished"):
            observed = output(request)
            observed.outputs[0].finish_reason = reason
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)
        observed = output(request)
        observed.finished = False
        with self.assertRaises(ValueError):
            module.collect_record(request, observed)

    def test_request_id_native_prompt_and_finished_identity_cannot_be_substituted(self):
        request = self.request()
        for mutate in (lambda value: setattr(value, "request_id", None), lambda value: setattr(value, "request_id", 1),
                       lambda value: setattr(value, "request_id", "x" * 1025),
                       lambda value: value.prompt_token_ids.append(9), lambda value: setattr(value, "finished", 1)):
            observed = output(request)
            mutate(observed)
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)

    def test_stop_requires_native_terminal_eos_without_changing_raw_text_or_stop_reason(self):
        request = self.request()
        for stop in (None, 151645):
            observed = output(request)
            observed.outputs[0].stop_reason = stop
            original_text = observed.outputs[0].text
            collected = module.collect_record(request, observed)["samples"][0]
            self.assertEqual(collected["stop_reason"], stop)
            self.assertEqual(collected["text"], original_text)
            self.assertEqual(collected["token_ids"][-1], 151645)
        for stop, tokens in (("answer", [9, 151645]), (1, [9, 151645]), (True, [9, 151645]),
                             (151643, [9, 151643]), (None, [9, 151643]), (None, []), (None, [9])):
            observed = output(request)
            observed.outputs[0].stop_reason, observed.outputs[0].token_ids = stop, tokens
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)

    def test_length_finish_keeps_all8192_tokens_and_cannot_hide_early_stopping(self):
        request = self.request()
        observed = output(request)
        observed.outputs[0].finish_reason = "length"
        observed.outputs[0].token_ids = [7] * 8192
        collected = module.collect_record(request, observed)["samples"][0]
        self.assertEqual((collected["output_tokens"], collected["finish_reason"], collected["token_ids"]), (8192, "length", [7] * 8192))
        for length in (0, 1, 8191, 8193):
            observed.outputs[0].token_ids = [7] * length
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)

    def test_tokens_are_plain_native_vocabulary_ids_and_text_is_never_repaired(self):
        request = self.request()
        for tokens in ([True, 151645], [-1, 151645], [151936, 151645], ["7", 151645]):
            observed = output(request)
            observed.outputs[0].token_ids = tokens
            with self.assertRaises(ValueError):
                module.collect_record(request, observed)
        observed = output(request)
        observed.outputs[0].text = None
        with self.assertRaises(ValueError):
            module.collect_record(request, observed)

    def test_raw_snapshot_preserves_abort_and_missing_samples_before_validation(self):
        request = self.request()
        observed = output(request)
        observed.outputs[0].finish_reason = "abort"
        observed.outputs.pop()
        raw = module.snapshot_output(observed)
        self.assertEqual(len(raw["outputs"]), 7)
        self.assertEqual(raw["outputs"][0]["finish_reason"], "abort")
        self.assertEqual(raw["outputs"][0]["text"], observed.outputs[0].text)
        self.assertEqual(raw["outputs"][0]["token_ids"], observed.outputs[0].token_ids)
        with self.assertRaises(ValueError):
            module.collect_record(request, observed)

    def test_observed_native_timing_is_preserved_with_explicit_unavailable_fields(self):
        request = self.request()
        observed = output(request)
        result = module.collect_record(request, observed)
        self.assertEqual(result["native_timing"]["first_token_time"], 1.2)
        self.assertIsNone(result["native_timing"]["model_forward_time"])
        self.assertEqual(result["native_metrics_type"], "SimpleNamespace")


class GenerationRunBoundaryTests(unittest.TestCase):
    def test_gpu_allocation_guard_precedes_inputs_factory_and_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            with self.assertRaisesRegex(ValueError, "No actual GPU allocation"):
                fixture.run(proof_error=ValueError("No actual GPU allocation"))
            fixture.loader.assert_not_called()
            self.assertEqual(fixture.factory_calls, [])
            self.assertFalse(fixture.output.exists())

    def test_reused_output_is_rejected_before_inputs_and_relative_input_contract_is_real(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            fixture.output.mkdir(parents=True)
            with self.assertRaises(ValueError):
                fixture.run()
            fixture.loader.assert_not_called()
            self.assertEqual(fixture.factory_calls, [])
            self.assertEqual(module.inside(fixture.root, module.PLAN), fixture.root / module.PLAN)
            for value in (str(fixture.root / module.PLAN), "../outside.json", ""):
                with self.assertRaises(ValueError):
                    module.inside(fixture.root, value)
            for index in range(3):
                arguments = [module.PLAN, "extension/runs/qa.json", "extension/operations/admission.json"]
                arguments[index] = str(fixture.root / arguments[index])
                with self.assertRaises(ValueError):
                    module.load_inputs(fixture.root, *arguments, fixture.allocation)

    def test_one32_batch_preserves256_unique_samples_and_cannot_claim_native_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            complete = fixture.run()
            self.assertEqual((len(fixture.factory_calls), len(fixture.generate_calls)), (1, 1))
            self.assertEqual(fixture.generate_calls[0], fixture.requests)
            self.assertEqual((complete["request_count"], complete["sample_count"], complete["output_tokens"]), (32, 256, 512))
            self.assertTrue(complete["fixture_factory_injected"])
            self.assertFalse(complete["native_interface_established"])
            self.assertEqual(complete["cache_sha256"], module.CACHE["sha256"])
            self.assertEqual(complete["profile_sha256"], module.PROFILE_SHA256)
            self.assertEqual(complete["plan_sha256"], fixture.context["plan_sha256"])
            self.assertEqual(complete["qa_sha256"], fixture.context["qa_sha256"])
            self.assertEqual(complete["admission_sha256"], fixture.context["admission_sha256"])
            record_files = sorted(fixture.output.glob("record-*.private.json"))
            self.assertEqual(len(record_files), 32)
            sample_keys = set()
            for path in record_files:
                row = json.loads(path.read_text())
                sample_keys.update((row["record_id"], sample["sample_index"]) for sample in row["samples"])
            self.assertEqual(len(sample_keys), 256)

    def test_factory_and_engine_receive_no_labels_context_or_native_framework_imports(self):
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "vllm", "transformers"}:
                raise AssertionError("Fixture path attempted a native framework import")
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            with patch("builtins.__import__", side_effect=guarded_import):
                fixture.run()
            config = fixture.factory_calls[0]
            self.assertEqual(set(config), {"base_path", "repository", "revision", "model_files", "tokenizer_identity", "profile", "generation_config"})
            self.assertEqual(config["model_files"], module.native.ORIGINAL_FILES)
            self.assertEqual(config["profile"], module.PROFILE)
            sealed = config["generation_config"]
            artifact = Path(sealed["directory"]) / sealed["filename"]
            self.assertEqual(artifact, fixture.output / "engine-config/generation_config.json")
            self.assertEqual(artifact.read_bytes(), b'{"eos_token_id":151645}\n')
            self.assertEqual(artifact.read_bytes(), module.EOS_CONFIG_BYTES)
            self.assertEqual(module.digest(artifact), module.EOS_CONFIG_SHA256)
            self.assertEqual((sealed["sha256"], sealed["content"]), (module.EOS_CONFIG_SHA256, {"eos_token_id": 151645}))
            self.assertEqual(module.validate_engine_config(sealed), sealed["directory"])
            for mutation in ({"sha256": "0" * 64}, {"content": {"eos_token_id": [151645, 151643]}},
                             {"filename": "different.json"}, {"directory": "relative/engine-config"}):
                with self.assertRaises(ValueError):
                    module.validate_engine_config({**sealed, **mutation})
            unexpected = artifact.parent / "unexpected.json"
            unexpected.write_text("{}")
            with self.assertRaises(ValueError):
                module.validate_engine_config(sealed)
            unexpected.unlink()
            before = artifact.read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                module.seal_engine_config(fixture.output)
            self.assertEqual(artifact.read_bytes(), before)
            payload = json.dumps([config, fixture.generate_calls[0]])
            for key in ('"label"', '"answerKey"', '"solution"', '"record"', '"context"', '"rendered_prompt"'):
                self.assertNotIn(key, payload)

    def test_missing_extra_reordered_or_duplicate_request_outputs_fail_without_retry(self):
        def variants(outputs):
            shortened = outputs[:-1]
            extra = outputs + [deepcopy(outputs[0])]
            reversed_rows = list(reversed(outputs))
            duplicate = deepcopy(outputs)
            duplicate[1].request_id = duplicate[0].request_id
            return shortened, extra, reversed_rows, duplicate
        for kind in range(4):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(temporary)
                def factory(config):
                    engine = fixture.factory(config)
                    native_generate = engine.generate
                    def generate(requests):
                        return variants(native_generate(requests))[kind]
                    engine.generate = generate
                    return engine
                with self.assertRaises(ValueError):
                    fixture.run(factory=factory)
                self.assertEqual((len(fixture.factory_calls), len(fixture.generate_calls)), (1, 1))
                self.assertFalse((fixture.output / "complete.json").exists())
                self.assertTrue((fixture.output / "engine_outputs.private.json").exists())
                failure = json.loads((fixture.output / "failure.json").read_text())
                self.assertFalse(failure["complete"] or failure["automatic_retry"] or failure["automatic_downstream"])
                self.assertTrue(failure["no_generation_refill_or_replay"])

    def test_aborted_sample_preserves_raw_observation_but_prevents_complete_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            def factory(config):
                engine = fixture.factory(config)
                native_generate = engine.generate
                def generate(requests):
                    observed = native_generate(requests)
                    observed[0].outputs[0].finish_reason = "abort"
                    return observed
                engine.generate = generate
                return engine
            with self.assertRaises(ValueError):
                fixture.run(factory=factory)
            self.assertEqual(len(fixture.generate_calls), 1)
            self.assertFalse((fixture.output / "complete.json").exists())
            raw = json.loads((fixture.output / "engine_outputs.private.json").read_text())
            self.assertEqual(raw[0]["outputs"][0]["finish_reason"], "abort")
            self.assertEqual(raw[0]["outputs"][0]["text"], output(fixture.requests[0]).outputs[0].text)

    def test_constructor_or_generate_exception_is_preserved_privately_with_no_retry(self):
        for phase in ("constructor", "generate"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(temporary)
                calls = []
                def factory(config):
                    calls.append("constructor")
                    if phase == "constructor":
                        raise RuntimeError("PRIVATE original constructor failure")
                    engine = fixture.factory(config)
                    def generate(requests):
                        calls.append("generate")
                        raise RuntimeError("PRIVATE original generate failure")
                    engine.generate = generate
                    return engine
                with self.assertRaisesRegex(RuntimeError, "PRIVATE original"):
                    fixture.run(factory=factory)
                self.assertEqual(calls.count("constructor"), 1)
                self.assertEqual(calls.count("generate"), int(phase == "generate"))
                self.assertFalse((fixture.output / "complete.json").exists())
                failure = json.loads((fixture.output / "failure.json").read_text())
                private = json.loads((fixture.output / "failure.private.json").read_text())
                self.assertEqual(private["exception"], "RuntimeError")
                self.assertNotIn("PRIVATE original", json.dumps(failure))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            state = {"llm_args": {"generation_config": "PRIVATE fixture engine directory"},
                     "hardware": {"name": "NVIDIA B200"},
                     "native_generation_fields": {"eos_token_id": [151645, 151643]}}
            expected_state = deepcopy(state)
            error = module.NativeInterfaceError(state)
            state["native_generation_fields"]["eos_token_id"].clear()
            calls = []
            def factory(config):
                calls.append("constructor")
                raise error
            with self.assertRaises(module.NativeInterfaceError) as raised:
                fixture.run(factory=factory)
            self.assertIs(raised.exception, error)
            self.assertEqual(calls, ["constructor"])
            self.assertEqual(fixture.generate_calls, [])
            self.assertFalse((fixture.output / "complete.json").exists())
            private = json.loads((fixture.output / "failure.private.json").read_text())
            failure = json.loads((fixture.output / "failure.json").read_text())
            self.assertEqual(private["exception"], "NativeInterfaceError")
            self.assertEqual(private["native_engine_state"], expected_state)
            self.assertEqual(private["validated_record_count"], 0)
            self.assertEqual(failure["exception"], "NativeInterfaceError")
            self.assertNotIn("native_engine_state", failure)
            self.assertNotIn("PRIVATE fixture engine directory", json.dumps(failure))

    def test_malformed_raw_snapshot_cannot_mask_original_collection_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            def factory(config):
                engine = fixture.factory(config)
                native_generate = engine.generate
                def generate(requests):
                    observed = native_generate(requests)
                    observed[0].outputs[0].token_ids = 17
                    return observed
                engine.generate = generate
                return engine
            with patch.object(module, "collect_record", side_effect=RuntimeError("original collection failure")):
                with self.assertRaisesRegex(RuntimeError, "original collection failure"):
                    fixture.run(factory=factory)
            self.assertFalse((fixture.output / "complete.json").exists())
            self.assertEqual(json.loads((fixture.output / "failure.json").read_text())["exception"], "RuntimeError")
            self.assertEqual(json.loads((fixture.output / "failure.private.json").read_text())["exception"], "RuntimeError")
            self.assertEqual(len(fixture.generate_calls), 1)

    def test_source_input_admission_or_ending_allocation_changes_prevent_success(self):
        cases = ({"source_change": True}, {"admission_change": True}, {"ending": {"job_id": "991", "node": "owl123"}},
                 {"evidence_fail": True}, {"eos_artifact_change": True})
        for options in cases:
            with tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(temporary)
                options = dict(options)
                if options.pop("evidence_fail", False):
                    fixture.evidence.fail_at = 2
                if options.pop("eos_artifact_change", False):
                    def factory(config):
                        engine = fixture.factory(config)
                        native_generate = engine.generate
                        def generate(requests):
                            observed = native_generate(requests)
                            artifact = Path(config["generation_config"]["directory"]) / "generation_config.json"
                            artifact.write_text('{"eos_token_id":[151645,151643]}\n')
                            return observed
                        engine.generate = generate
                        return engine
                    options["factory"] = factory
                with self.assertRaises(ValueError):
                    fixture.run(**options)
                self.assertFalse((fixture.output / "complete.json").exists())
                self.assertTrue((fixture.output / "failure.json").exists())
                self.assertEqual(len(fixture.generate_calls), 1)

    def test_missing_or_false_native_adapter_provenance_cannot_complete(self):
        for mutation in ({"one_generate_call": False}, {"request_count": 31}, {"native_eos_token_id": 0},
                         {"labels_entered_engine": True}):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = RunFixture(temporary)
                def factory(config):
                    engine = fixture.factory(config)
                    original_proof = engine.proof
                    engine.proof = lambda: {**original_proof(), **mutation}
                    return engine
                with self.assertRaises(ValueError):
                    fixture.run(factory=factory)
                self.assertFalse((fixture.output / "complete.json").exists())
                self.assertEqual(len(fixture.generate_calls), 1)

    def test_public_metadata_permissions_live_log_qualification_and_cli_privacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RunFixture(temporary)
            complete = fixture.run()
            self.assertEqual(stat.S_IMODE(fixture.output.stat().st_mode), 0o700)
            self.assertTrue(complete["raw_requests_outputs_private"])
            self.assertFalse(any(complete[key] for key in ("grading", "ranking", "training", "automatic_retry", "automatic_downstream")))
            for name in ("engine.stdout.private.log", "engine.stderr.private.log"):
                self.assertNotIn(name, complete["private_output_sha256"], "Worker logs are not immutable before terminal closeout")
                self.assertIn("terminal", complete["unsealed_private_logs"][name])
            for name, checksum in complete["private_output_sha256"].items():
                self.assertEqual(module.digest(fixture.output / name), checksum)
                self.assertIn(stat.S_IMODE((fixture.output / name).stat().st_mode), {0o400, 0o600})
            self.assertEqual(complete["private_output_sha256"]["engine-config/generation_config.json"], module.EOS_CONFIG_SHA256)
            public = json.dumps(complete)
            self.assertNotIn("PRIVATE-ENGINE-MARKER", public)
            for row in fixture.value["records"]:
                self.assertNotIn(row["record"]["id"], public)
                self.assertNotIn(row["record"]["question"], public)
            self.assertIn("PRIVATE-ENGINE-MARKER", (fixture.output / "engine.stdout.private.log").read_text())
        secret = "PRIVATE source prompt/gold/response"
        with patch.object(module, "run", side_effect=RuntimeError(secret)):
            with self.assertRaises(RuntimeError) as caught:
                module.main(["--root", ".", "--plan", module.PLAN, "--output", "private-output",
                    "--qa-receipt", "extension/private-qa.json", "--admission-receipt", "extension/private-job.json"])
        self.assertEqual(str(caught.exception), "Source calibration generation failed. See private failure evidence.")
        self.assertTrue(caught.exception.__suppress_context__)
        rendered = "".join(traceback.format_exception(type(caught.exception), caught.exception, caught.exception.__traceback__))
        self.assertNotIn(secret, rendered)


class VersionedIdentityIntegrationTests(unittest.TestCase):
    def test_frozen_collector_tests_helper_and_native65_source_namespace_are_unchanged(self):
        pins = {"extension/source_generation.py": module.FROZEN_COLLECTOR_SHA256,
                "tests/test_source_generation.py": "8fb6d9c48f438628a3a8f6484b6f0ecdc0933cf4b8554c7083687754de34a5cc",
                "extension/source_generation_v2.py": "69c4b09bc11845850951a137509b3b5d0f9b0d3e9047876426e661f88a0da05f",
                "extension/source_generation_v2_plan.json": module.OPERATIONAL_PARENT["sha256"],
                "extension/source_generation_v2.slurm": "f6ea4a4e723f80a3d1d01fef1618ba8c4e850cc433d0f1c7171c969cce530a69",
                "tests/test_source_generation_identity_v2.py": "735407038b0b1131a257fee6e8f35ab54ddcb8963759d0975ff58e423b6a440d",
                module.IDENTITY_HELPER["path"]: module.IDENTITY_HELPER["sha256"]}
        for name, expected in pins.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        old_source = (ROOT / "tests/test_source_generation.py").read_text()
        new_source = Path(__file__).read_text()
        old_classes = {node.name: node for node in ast.parse(old_source).body if isinstance(node, ast.ClassDef) and node.name.endswith("Tests")}
        new_classes = {node.name: node for node in ast.parse(new_source).body if isinstance(node, ast.ClassDef)}
        old_count = 0
        for name, old_class in old_classes.items():
            for old_test in (item for item in old_class.body if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")):
                old_count += 1
                actual = [item for item in new_classes[name].body if isinstance(item, ast.FunctionDef) and item.name == old_test.name]
                self.assertEqual(len(actual), 1)
                self.assertEqual(ast_dump(actual[0]), ast_dump(old_test))
        self.assertEqual(old_count, 32)
        proof = module.source_inventory(ROOT)
        frozen = json.loads((ROOT / module.IDENTITY_DIRECTORY / "complete.json").read_text())["source_inventory"]
        self.assertEqual((len(frozen), len(proof)), (120, 128))
        self.assertEqual(set(proof) - set(frozen), module.NEW_SOURCES)
        self.assertEqual(len(module.NEW_SOURCES), 8)
        self.assertEqual(len(module.NEW_SOURCES - module.operational_parent.NEW_SOURCES), 4)
        self.assertTrue(all(proof[name] == value for name, value in frozen.items()))

    def test_entire_module_text_and_ast_preserve_science_except_named_operational_deltas(self):
        original = (ROOT / "extension/source_generation.py").read_text()
        candidate = (ROOT / "extension/source_generation_v3.py").read_text()
        assert_scientific_replay(original, candidate)
        self.assertIs(module.native, predecessor.native)
        for name in ("CACHE", "PROFILE", "PROFILE_SHA256", "PARENT_PINS", "PARENT_DIRECTORY", "SOURCE_PLAN", "RESOURCE",
                     "EOS_CONFIG_BYTES", "EOS_CONFIG_SHA256", "ENGINE_ADAPTER", "ENGINE_STATE_FIELDS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(module, name), getattr(predecessor, name))
        self.assertEqual(module.NativeEngine.__module__, "extension.source_generation_v3")
        self.assertIsNot(module.NativeEngine, predecessor.NativeEngine)

    def test_sampling_prompt_native_flag_or_inference_mutations_fail_projection(self):
        original = (ROOT / "extension/source_generation.py").read_text()
        candidate = (ROOT / "extension/source_generation_v3.py").read_text()
        for old, new in (("temperature=.6", "temperature=.7"), ("max_tokens=8192", "max_tokens=4096"),
                         ("effective.all_stop_token_ids == {151645}", "effective.all_stop_token_ids == {151645, 151643}"),
                         ("engine_factory is None", "engine_factory is not None"),
                         ('"source-generation-original-v3-"', '"source-generation-original-"'),
                         ("gpu_identity_v2.assigned_gpu_identity(torch)", "native.assigned_gpu_identity(torch)"),
                         ('"grading_performed": False', '"grading_performed": True')):
            with self.subTest(old=old):
                self.assertIn(old, candidate)
                with self.assertRaises(AssertionError):
                    assert_scientific_replay(original, candidate.replace(old, new, 1))

    def test_versioned_plan_binds_scientific_parent_helper_and_exact_native65_evidence(self):
        original, value = json.loads((ROOT / module.SCIENTIFIC_PARENT["path"]).read_text()), plan()
        self.assertEqual(module.digest(ROOT / module.SCIENTIFIC_PARENT["path"]), module.SCIENTIFIC_PARENT["sha256"])
        with self.assertRaises(ValueError):
            module.validate_plan(original)
        self.assertEqual(value["role"], "source_generation_original_calibration_20261021_v3")
        self.assertEqual(value["native_qa"]["collector_expected_tests"], 40)
        self.assertEqual((value["native_qa"]["source_inventory_files"], value["native_qa"]["separate_metadata_source_files"]), (128, 4))
        self.assertFalse(value["native_qa"]["identity_parent_tests_replayed"])
        parent = json.loads((ROOT / module.OPERATIONAL_PARENT["path"]).read_text())
        self.assertEqual(value["admission_seals"], parent["admission_seals"])
        self.assertEqual(value["operational_parent"], module.OPERATIONAL_PARENT)
        self.assertEqual(value["retry_failure"], module.RETRY_FAILURE)
        self.assertEqual(value["retry_failure"]["prior_entries"], 25)
        for mutate in (lambda row: row["scientific_parent"].update(sha256="0" * 64),
                       lambda row: row["identity_helper"].update(sha256="0" * 64),
                       lambda row: row["identity_parent_evidence"]["complete.json"].update(sha256="0" * 64),
                       lambda row: row["native_qa"].update(identity_parent_tests_replayed=True),
                       lambda row: row["native_qa"].update(source_inventory_files=95),
                       lambda row: row["operational_parent"].update(sha256="0" * 64),
                       lambda row: row["retry_failure"].update(errors=0),
                       lambda row: row.update(role=predecessor.ROLE)):
            changed = deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                module.validate_plan(changed)

    def test_old_native48_qa_is_rejected_before_cache_model_or_current_admission_access(self):
        class MetadataEvidence:
            def __init__(self):
                self.reads = []
            def read(self, path, *args):
                self.reads.append(Path(path).name)
                if Path(path).name == Path(module.PLAN).name:
                    return plan()
                if Path(path).name == "old_qa.json":
                    return {"schema": predecessor.QA_SCHEMA,
                            "test_counts": {"test_source_generation.py": 32, "test_source_generation_submission.py": 16}}
                return {"schema": predecessor.ADMISSION_SCHEMA}
            def check(self, *args):
                raise AssertionError("Old QA must fail before any file/weight/cache checks")
        evidence = MetadataEvidence()
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, "Evidence", return_value=evidence), \
                patch.object(module, "source_inventory", return_value={}), patch.object(module, "_current_admission") as admission:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "collector40"):
                module.load_inputs(root, module.PLAN, "extension/runs/old_qa.json", "extension/operations/old_admission.json",
                                   {"job_id": "990", "node": "owl123"})
            admission.assert_not_called()
        self.assertEqual(evidence.reads, [Path(module.PLAN).name, "old_qa.json", "old_admission.json"])

    def test_current_job_metadata_accepts_only_new_role_and_parent_helper_bindings(self):
        sources = {"extension/source_generation_v3.py": {"sha256": "a" * 64, "bytes": 1}}
        allocation, plan_sha, qa_sha = {"job_id": "990"}, "b" * 64, "c" * 64
        binding = {"role": module.ROLE, "source_sha256": sources, "plan_sha256": plan_sha, "qa_sha256": qa_sha,
                   "cache_sha256": module.CACHE["sha256"], "profile_sha256": module.PROFILE_SHA256,
                   "scientific_parent_sha256": module.SCIENTIFIC_PARENT["sha256"],
                   "operational_parent_sha256": module.OPERATIONAL_PARENT["sha256"],
                   "retry_failure_sha256": value_digest(module.RETRY_FAILURE),
                   "failed_cpu_snapshot_sha256": module.RETRY_FAILURE["pins"]["job_ledger_snapshot.json"],
                   "identity_parent_complete_sha256": module.IDENTITY_PINS["complete.json"],
                   "identity_helper_sha256": module.IDENTITY_HELPER["sha256"],
                   "parent_complete_sha256": module.PARENT_PINS["complete.json"], "model_role": "original",
                   "authorize_first_interface_checking_generation": True, "requested_gpu_seconds": 7200,
                   "wall_time_seconds": 7200, "closeout_reserve_seconds": 172800,
                   "automatic_retry": False, "automatic_downstream": False}
        entry = {"status": "submitted", "job_id": "990", "source_sha256": sources, "binding": binding,
                 "binding_sha256": value_digest(binding), "requested_gpu_seconds": 7200, "requested_gpu_hours": 2}
        ledger = {"schema": module.prepared_stage.joint.previous.SCHEMA, "gpu_hour_cap": None,
                  "historical_ledger_unchanged": "deadline/job_ledger.json", "jobs": {module.ROLE: entry}}
        with patch.object(module, "read_json", return_value=ledger):
            self.assertIs(module._current_admission(ROOT, allocation, sources, plan_sha, qa_sha), entry)
        for field in ("scientific_parent_sha256", "identity_parent_complete_sha256", "identity_helper_sha256", "qa_sha256",
                      "operational_parent_sha256", "retry_failure_sha256", "failed_cpu_snapshot_sha256"):
            changed = deepcopy(ledger)
            changed["jobs"][module.ROLE]["binding"][field] = "0" * 64
            changed["jobs"][module.ROLE]["binding_sha256"] = value_digest(changed["jobs"][module.ROLE]["binding"])
            with self.subTest(field=field), patch.object(module, "read_json", return_value=changed):
                with self.assertRaises(ValueError):
                    module._current_admission(ROOT, allocation, sources, plan_sha, qa_sha)
        old = deepcopy(ledger)
        old["jobs"] = {predecessor.ROLE: entry}
        with patch.object(module, "read_json", return_value=old), self.assertRaises(ValueError):
            module._current_admission(ROOT, allocation, sources, plan_sha, qa_sha)

    def test_actual_cpu65_parent_proof_is_bound_without_claiming_generation_or_replaying_tests(self):
        sources = module.source_inventory(ROOT)
        values = module._identity_parent_evidence(ROOT, module.Evidence(), sources)
        self.assertEqual(values["complete.json"]["tests_run"], 65)
        self.assertFalse(values["unit_qa.json"]["actual_gpu_or_memory_admission_established"])
        failed = module.retry_failure_evidence(ROOT)
        self.assertEqual(len(failed["initial"]), 25)
        self.assertEqual(len(failed["protected_source_inventory"]), 139)
        self.assertEqual(failed["values"]["failure.json"]["tests_run"], 60)
        self.assertEqual(failed["values"]["failure.json"]["errors"], 1)
        self.assertFalse(failed["values"]["failure.json"]["success"])
        class MetadataEvidence:
            def __init__(self, value):
                self.value = value
            def read(self, path, expected):
                return deepcopy(self.value[Path(path).name])
        for filename, key, value in (("complete.json", "source_inventory", {}), ("unit_qa.json", "tests_run", 450),
                                     ("unit_qa.json", "actual_gpu_or_memory_admission_established", True),
                                     ("unit_qa.json", "experiment_model_or_tokenizer_loaded", True),
                                     ("accounting.json", "elapsed_seconds", 33)):
            with self.subTest(filename=filename, key=key):
                changed = deepcopy(values)
                changed[filename][key] = value
                expected = SubmissionError if filename == "accounting.json" else ValueError
                with self.assertRaises(expected):
                    module._identity_parent_evidence(ROOT, MetadataEvidence(changed), sources)

    def test_real_production_constructor_delegates_identity_before_any_engine_load(self):
        fake_torch, fake_vllm = ModuleType("torch"), ModuleType("vllm")
        fake_vllm.__path__ = []
        fake_outputs, fake_sampling = ModuleType("vllm.outputs"), ModuleType("vllm.sampling_params")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                         is_bf16_supported=lambda: True, get_device_name=lambda index: "NVIDIA B200")
        loads = []
        class FakeLLM:
            def __init__(self, **kwargs):
                loads.append(kwargs)
                raise AssertionError("No engine may be loaded in this CPU dispatch fixture")
            def generate(self, prompts, sampling_params, use_tqdm=False):
                raise AssertionError("No generation may run in this CPU dispatch fixture")
        fake_vllm.LLM, fake_vllm.SamplingParams = FakeLLM, object
        fake_vllm.outputs, fake_vllm.sampling_params = fake_outputs, fake_sampling
        fake_sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY=object())
        fixture_modules = {"torch": fake_torch, "vllm": fake_vllm, "vllm.outputs": fake_outputs, "vllm.sampling_params": fake_sampling}
        proof = {"fixture_only": True, "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff"}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, fixture_modules), \
                patch.object(module.importlib.metadata, "version", return_value="0.12.0"), \
                patch.object(module.gpu_identity_v2, "assigned_gpu_identity", return_value=proof) as identify, \
                patch.object(module.NativeEngine, "memory", side_effect=RuntimeError("fixture halt before engine load")):
            artifact = module.seal_engine_config(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "fixture halt before engine load"):
                module.NativeEngine({"generation_config": artifact})
        identify.assert_called_once_with(fake_torch)
        self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
