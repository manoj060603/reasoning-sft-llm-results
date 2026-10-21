"""Unexecuted CPU oracles for one prospective allocator-only engineering replay.

Every CUDA observation below is a synthetic object. No Torch import, model,
tokenizer, corpus, CUDA API, allocator mutation, scheduler, or real-memory claim
is made by these fixtures. Execute only through separately admitted native QA.
"""
import ast
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from extension import joint_memory_admission_v3 as parent
from extension import joint_memory_allocator_v4 as module


ROOT = Path(__file__).resolve().parents[1]
PARENT = "extension/joint_memory_admission_v3.py"
CANDIDATE = "extension/joint_memory_allocator_v4.py"
PARENT_SHA = "be1d4aa53b5e01755bcddfc380cf132af3ea0dca90f32b5418d1ecd28325dafa"
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:True"}
STATS = (
    "allocated_bytes.all.current", "allocated_bytes.all.peak", "active_bytes.all.current", "active_bytes.all.peak",
    "requested_bytes.all.current", "requested_bytes.all.peak", "reserved_bytes.all.current", "reserved_bytes.all.peak",
    "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak", "segment.all.current", "segment.all.peak",
    "num_alloc_retries", "num_ooms", "num_sync_all_streams", "num_device_alloc", "num_device_free",
)
ADDITIONS = frozenset({CANDIDATE, "tests/test_joint_allocator_v4.py", "extension/joint_allocator_v4_plan.json",
    "extension/joint_allocator_v4_qa.py", "extension/joint_allocator_v4_qa.slurm", "extension/joint_memory_allocator_v4.slurm"})
RAW_SENTINEL = "PRIVATE-RAW-SNAPSHOT-MUST-NOT-ESCAPE"


class FixtureTorchVersion(str):
    """Pinned Torch exposes a str subclass; this fixture imports no Torch."""


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def dump(node):
    return ast.dump(node, include_attributes=False)


def top(tree, name):
    nodes = [node for node in tree.body if getattr(node, "name", None) == name or isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(nodes) != 1:
        raise AssertionError("Expected one exact top-level node: " + name)
    return nodes[0]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("An allowed operational replacement is missing or ambiguous")
    return source.replace(old, new, 1)


def fake_snapshot():
    return {"allocator_settings": {**ENV, "expandable_segments": True, "unretained_extra": RAW_SENTINEL},
        "segments": [{"address": RAW_SENTINEL, "blocks": [{"frames": [RAW_SENTINEL]}]}],
        "device_traces": [[RAW_SENTINEL]], "external_annotations": [RAW_SENTINEL]}


def fake_torch(*, backend="native", version=FixtureTorchVersion("2.9.0+cu128"), snapshot=None,
               total=191495471104, allocated=149686226432, reserved=175040888832, driver=918224896):
    stats = OrderedDict((name, 0) for name in STATS)
    for stem in ("allocated", "active", "requested"):
        stats[stem + "_bytes.all.current"] = stats[stem + "_bytes.all.peak"] = allocated
    stats["reserved_bytes.all.current"] = stats["reserved_bytes.all.peak"] = reserved
    memory = SimpleNamespace(get_allocator_backend=Mock(return_value=backend),
        _snapshot=Mock(return_value=fake_snapshot() if snapshot is None else snapshot),
        _set_allocator_settings=Mock(side_effect=AssertionError("No runtime allocator mutation")))
    cuda = SimpleNamespace(memory=memory, synchronize=Mock(),
        mem_get_info=Mock(return_value=(total - reserved - driver, total)),
        memory_allocated=Mock(return_value=allocated), memory_reserved=Mock(return_value=reserved),
        max_memory_allocated=Mock(return_value=allocated), max_memory_reserved=Mock(return_value=reserved),
        reset_peak_memory_stats=Mock(), memory_stats=Mock(return_value=stats),
        empty_cache=Mock(side_effect=AssertionError("No cache emptying")))
    return SimpleNamespace(__version__=version, cuda=cuda), stats


def limits():
    return {"headroom_min_bytes": 16 << 30, "headroom_fraction": .1}


def operational_plan():
    return {"schema": "extension-joint-allocator-v4-operational-plan-v1", "allocator_policy": deepcopy(module.ALLOCATOR_POLICY),
        "scientific_parent": {"path": "extension/joint_memory_admission_plan.json",
            "sha256": "b926be22779d59af501afddd10e7129f105b68b723b9bc8386be0aa397ddfc24"},
        "prepared": {"path": "extension/runs/native-smoke-preparation-1224497/prepared.json",
            "sha256": "d1bb1875a293c71f2a108df4b287b3bb2ab35b3168cd6505b5bd91a98fd66457"},
        "settings": dict(parent.SETTINGS), "case_order": list(parent.MODES), "presentation_roles": list(parent.PRESENTATION_ROLES),
        "tolerances": {"logit_max_abs": .001, "ce_abs": 1e-5, "kl_abs": 1e-5, "reload_logit_max_abs": .001,
            "headroom_min_bytes": 16 << 30, "headroom_fraction": .1},
        "diagnostic_only": True, "automatic_retry": False, "automatic_downstream": False}


def allocator_samples():
    _, values = fake_torch()
    return [{"capture_index": index, "presentation_index": position,
        "active_phases": ["actual_seventeen_presentations_three_updates", "actual_backward", "external_objective_forward"],
        "stats": dict(values)} for index, position in enumerate((8, 9))]


def expected_existing_source(source):
    """Independent literal allowlist for every change to the prior worker.

    New top-level allocator helpers are checked by behavioral fixtures; this
    oracle protects the complete pre-existing module and scientific worker.
    """
    changes = [
        ('"extension/joint_memory_admission_v3.py", "tests/test_joint_identity_replay_v3.py"})',
         '"extension/joint_memory_admission_v3.py", "tests/test_joint_identity_replay_v3.py",\n'
         '    "extension/joint_memory_allocator_v4.py", "tests/test_joint_allocator_v4.py",\n'
         '    "extension/joint_allocator_v4_plan.json", "extension/joint_allocator_v4_qa.py",\n'
         '    "extension/joint_allocator_v4_qa.slurm", "extension/joint_memory_allocator_v4.slurm"})'),
        ('    plan_sha = evidence.check(paths["plan"])\n    old_plan =',
         '    plan_sha = evidence.check(paths["plan"])\n'
         '    operational = validate_allocator_plan(evidence.read(native._project_path(root, ALLOCATOR_PLAN)))\n'
         '    operational_sha = evidence.check(native._project_path(root, ALLOCATOR_PLAN))\n'
         '    require(plan_sha == operational["scientific_parent"]["sha256"], "The allocator replay must retain the exact original scientific plan")\n'
         '    old_plan ='),
        ('        type(codes) is dict and REQUIRED_CODE <= set(codes), "New joint-diagnostic CPU QA is required; the old67freeze is insufficient")',
         '        type(codes) is dict and REQUIRED_CODE <= set(codes) and qa.get("allocator_policy") == ALLOCATOR_POLICY and\n'
         '        qa.get("operational_plan_sha256") == operational_sha, "New allocator-only CPU QA is required; previous headroom failures remain failed")'),
        ('        admission.get("automatic_downstream") is False, "A new exact current-job admission is required")',
         '        admission.get("automatic_downstream") is False and admission.get("allocator_policy") == ALLOCATOR_POLICY and\n'
         '        admission.get("operational_plan_sha256") == operational_sha, "A new exact current-job allocator-only admission is required")'),
        ('"qa_sha256": evidence.check(paths["qa"]), "admission_sha256": evidence.check(paths["admission"]), "code_sha256": codes}',
         '"qa_sha256": evidence.check(paths["qa"]), "admission_sha256": evidence.check(paths["admission"]), "code_sha256": codes,\n'
         '        "operational_plan_sha256": operational_sha}'),
        ('def validate_case_receipt(receipt, context, mode, output, *, verify_outputs=True, output_evidence=None):\n',
         'def validate_case_receipt(receipt, context, mode, output, *, verify_outputs=True, output_evidence=None):\n'
         '    require(type(receipt) is dict, "Allocator diagnostic case receipt must be a dictionary")\n'
         '    require(receipt.get("operational_plan_sha256") == context["operational_plan_sha256"] and\n'
         '        receipt.get("allocator_policy") == ALLOCATOR_POLICY and receipt.get("allocator_preimport", {}).get("torch_not_imported_at_guard") is True and\n'
         '        receipt["allocator_preimport"].get("environment") == validate_allocator_environment({"PYTORCH_CUDA_ALLOC_CONF": ALLOCATOR_ENV}),\n'
         '        "Exact allocator-only operational/preimport provenance is required")\n'
         '    before, after = (validate_recorded_allocator_proof(receipt.get(name)) for name in ("allocator_before_model", "allocator_at_closeout"))\n'
         '    require(before == after, "Effective allocator policy changed during the diagnostic")\n'
         '    validate_allocator_samples(receipt.get("memory", {}).get("allocator_samples"))\n'),
        ('def run(root, plan_path, output, qa_receipt, admission_receipt, *, runner=None):\n',
         'def run(root, plan_path, output, qa_receipt, admission_receipt, *, runner=None):\n    allocator_preimport_proof()\n'),
        ('"extension.joint_memory_admission_v3"', '"extension.joint_memory_allocator_v4"'),
        ('            "plan_sha256": context["plan_sha256"], "prepared_sha256": context["prepared_sha256"],\n            "code_sha256":',
         '            "plan_sha256": context["plan_sha256"], "prepared_sha256": context["prepared_sha256"],\n'
         '            "operational_plan_sha256": context["operational_plan_sha256"], "allocator_policy": ALLOCATOR_POLICY,\n'
         '            "code_sha256":'),
        ('        "Case is outside its admitted running compute allocation")\n    import torch',
         '        "Case is outside its admitted running compute allocation")\n    preimport = allocator_preimport_proof()\n    import torch'),
        ('    base, begun, memory = Path(prepared["base"]["path"]), time.monotonic(), native.MemoryTrace(torch)',
         '    base, begun, memory = Path(prepared["base"]["path"]), time.monotonic(), None\n    allocator_before = allocator_after = None'),
        ('    ledger, journal, observations = None, None, []\n    try:\n        set_seed(settings["seed"])',
         '    ledger, journal, observations = None, None, []\n    try:\n'
         '        allocator_before = allocator_proof(torch)\n        memory = AllocatorMemoryTrace(torch)\n        set_seed(settings["seed"])'),
        ('                index = int(inputs["row_index"].item())\n                if index in (0, 9):',
         '                index = int(inputs["row_index"].item())\n                memory.presentation_index = index\n                if index in (0, 9):'),
        ('            trainer.train()\n        rng["after_train"]',
         '            trainer.train()\n        memory.presentation_index = None\n        rng["after_train"]'),
        ('            "Global loss or actual prepared external logits precision differs")\n        memory_result = memory.result(limits)',
         '            "Global loss or actual prepared external logits precision differs")\n'
         '        allocator_after = allocator_proof(torch)\n'
         '        require(allocator_after == allocator_before, "Effective allocator settings changed after model work")\n'
         '        memory_result = memory.result(limits)'),
        ('        receipt = {"schema": CASE_SCHEMA, "complete": True, "mode": mode, "binding_sha256": context["binding_sha256"],\n',
         '        receipt = {"schema": CASE_SCHEMA, "complete": True, "mode": mode, "binding_sha256": context["binding_sha256"],\n'
         '            "operational_plan_sha256": context["operational_plan_sha256"], "allocator_policy": ALLOCATOR_POLICY,\n'
         '            "allocator_preimport": preimport, "allocator_before_model": allocator_before, "allocator_at_closeout": allocator_after,\n'),
        ('            "memory_stages_observed": memory.stages, "allocator_peak_allocated_seen": memory.whole_allocated,\n'
         '            "allocator_peak_reserved_seen": memory.whole_reserved, "max_observed_nonallocator_bytes": memory.max_observed_nonallocator_bytes,',
         '            "memory_stages_observed": [] if memory is None else memory.stages,\n'
         '            "allocator_peak_allocated_seen": 0 if memory is None else memory.whole_allocated,\n'
         '            "allocator_peak_reserved_seen": 0 if memory is None else memory.whole_reserved,\n'
         '            "max_observed_nonallocator_bytes": 0 if memory is None else memory.max_observed_nonallocator_bytes,\n'
         '            "allocator_preimport": preimport, "allocator_before_model": allocator_before, "allocator_at_closeout": allocator_after,\n'
         '            "allocator_samples": [] if memory is None else memory.allocator_samples, "allocator_policy": ALLOCATOR_POLICY,'),
        ('def main():\n', 'def main():\n    allocator_preimport_proof()\n'),
    ]
    for old, new in changes:
        source = replace_once(source, old, new)
    return source


HELPER_NAMES = frozenset({"ALLOCATOR_ENV", "ALLOCATOR_PLAN", "ALLOCATOR_POLICY", "ALLOCATOR_STATS", "ALLOCATOR_QUALIFICATION",
    "validate_allocator_environment", "allocator_preimport_proof", "allocator_proof", "selected_allocator_stats",
    "AllocatorMemoryTrace", "validate_allocator_plan", "validate_recorded_allocator_proof", "validate_allocator_samples"})


def existing_ast(candidate):
    tree = ast.parse(candidate)
    kept, removed = [], []
    for node in tree.body:
        names = [getattr(node, "name", None)] if not isinstance(node, ast.Assign) else [
            target.id for target in node.targets if isinstance(target, ast.Name)]
        if set(names) & HELPER_NAMES:
            removed.extend(name for name in names if name in HELPER_NAMES)
        else:
            kept.append(node)
    if set(removed) != HELPER_NAMES or len(removed) != len(HELPER_NAMES):
        raise AssertionError("Missing or duplicated explicitly enumerated allocator helper")
    tree.body = kept
    return tree


def assert_scientific_replay(candidate):
    expected = ast.parse(expected_existing_source(read(PARENT)))
    if dump(existing_ast(candidate)) != dump(expected):
        raise AssertionError("Non-allowlisted scientific/worker code changed")


class AllocatorEnvironmentAndProofTests(unittest.TestCase):
    def test_exact_single_configuration_is_accepted_without_mutating_environment(self):
        environment = {**ENV, "UNRELATED_FIXTURE": "retained"}
        before = deepcopy(environment)
        proof = module.validate_allocator_environment(environment)
        self.assertEqual(environment, before)
        self.assertEqual(proof, {**ENV, "conflicting_or_cache_disabling_variables_present": False})

    def test_missing_altered_extra_knob_or_other_backend_configuration_fails(self):
        invalid = ({}, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
            {"PYTORCH_CUDA_ALLOC_CONF": ENV["PYTORCH_CUDA_ALLOC_CONF"] + ",max_split_size_mb:128"},
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:False"}, None, [])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_allocator_environment(value)

    def test_cache_disabling_variable_is_rejected_even_if_empty_or_zero(self):
        for value in ("", "0", "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_allocator_environment({**ENV, "PYTORCH_NO_CUDA_MEMORY_CACHING": value})

    def test_alternate_allocator_aliases_are_rejected_even_if_values_agree(self):
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
            for value in ("", ENV["PYTORCH_CUDA_ALLOC_CONF"]):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    module.validate_allocator_environment({**ENV, name: value})

    def test_preimport_guard_attests_only_explicit_absence_of_torch(self):
        modules = {"synthetic_unrelated_module": object()}
        result = module.allocator_preimport_proof(environ=ENV, modules=modules)
        self.assertEqual(result["schema"], "extension-joint-allocator-preimport-v1")
        self.assertIs(result["torch_not_imported_at_guard"], True)
        self.assertEqual(result["environment"], module.validate_allocator_environment(ENV))
        self.assertEqual(set(modules), {"synthetic_unrelated_module"})

    def test_preimport_guard_refuses_already_imported_torch_without_unloading_it(self):
        loaded = object()
        modules = {"torch": loaded}
        with self.assertRaisesRegex(ValueError, "before importing Torch"):
            module.allocator_preimport_proof(environ=ENV, modules=modules)
        self.assertIs(modules["torch"], loaded)

    def test_effective_proof_requires_actual_native_backend_and_true_pinned_setting(self):
        torch, _ = fake_torch()
        proof = module.allocator_proof(torch, environ=ENV)
        self.assertEqual(proof["schema"], "extension-joint-allocator-effective-v1")
        self.assertEqual(proof["allocator_backend"], "native")
        self.assertIs(type(proof["torch_version"]), str)
        self.assertEqual(proof["torch_version"], "2.9.0+cu128")
        self.assertEqual(proof["effective_settings"], {**ENV, "expandable_segments": True})
        self.assertIs(proof["private_api_version_pinned"], True)
        self.assertIn("private", proof["qualification"])
        torch.cuda.memory.get_allocator_backend.assert_called_once_with()
        torch.cuda.memory._snapshot.assert_called_once_with()

    def test_bad_environment_fails_before_any_allocator_observation(self):
        torch, _ = fake_torch()
        with self.assertRaises(ValueError):
            module.allocator_proof(torch, environ={})
        torch.cuda.memory.get_allocator_backend.assert_not_called()
        torch.cuda.memory._snapshot.assert_not_called()

    def test_non_pinned_version_fails_before_backend_or_snapshot_query(self):
        for version in (None, 2.9, "2.8.0", "2.10.0", "2.9.1", "2.9.0dev"):
            torch, _ = fake_torch(version=version)
            with self.subTest(version=version), self.assertRaises(ValueError):
                module.allocator_proof(torch, environ=ENV)
            torch.cuda.memory.get_allocator_backend.assert_not_called()
            torch.cuda.memory._snapshot.assert_not_called()

    def test_backend_mismatch_fails_without_attempting_snapshot_or_fallback(self):
        for backend in ("cudaMallocAsync", "", None):
            torch, _ = fake_torch(backend=backend)
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                module.allocator_proof(torch, environ=ENV)
            torch.cuda.memory._snapshot.assert_not_called()
            torch.cuda.memory._set_allocator_settings.assert_not_called()

    def test_missing_or_unsupported_effective_snapshot_fields_fail_closed(self):
        invalid = ([], {}, {"allocator_settings": []}, {"allocator_settings": {}},
            {"allocator_settings": {**ENV, "expandable_segments": False}},
            {"allocator_settings": {**ENV, "expandable_segments": 1}},
            {"allocator_settings": {**ENV, "expandable_segments": "True"}},
            {"allocator_settings": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "expandable_segments": True}})
        for snapshot in invalid:
            torch, _ = fake_torch(snapshot=snapshot)
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                module.allocator_proof(torch, environ=ENV)

    def test_raw_snapshot_is_not_retained_and_observation_errors_do_not_retry(self):
        snapshot = fake_snapshot()
        torch, _ = fake_torch(snapshot=snapshot)
        proof = module.allocator_proof(torch, environ=ENV)
        snapshot["allocator_settings"]["expandable_segments"] = False
        self.assertIs(proof["effective_settings"]["expandable_segments"], True)
        self.assertNotIn(RAW_SENTINEL, json.dumps(proof))
        for name in ("segments", "blocks", "addresses", "frames", "device_traces"):
            self.assertNotIn(name, proof)
        broken, _ = fake_torch()
        error = RuntimeError("synthetic private-snapshot API unavailable")
        broken.cuda.memory._snapshot.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            module.allocator_proof(broken, environ=ENV)
        self.assertIs(raised.exception, error)
        self.assertEqual(broken.cuda.memory._snapshot.call_count, 1)
        broken.cuda.memory._set_allocator_settings.assert_not_called()


class AllocatorStatisticsAndTraceTests(unittest.TestCase):
    def test_native_ordered_dict_stats_are_supported_and_only_fixed_aggregates_exported(self):
        _, values = fake_torch()
        values["private_segments"] = RAW_SENTINEL
        before = deepcopy(values)
        selected = module.selected_allocator_stats(values)
        self.assertEqual(type(selected), dict)
        self.assertEqual(set(selected), set(STATS))
        self.assertEqual(values, before)
        self.assertNotIn(RAW_SENTINEL, json.dumps(selected))
        values[STATS[0]] += 1
        self.assertNotEqual(selected[STATS[0]], values[STATS[0]])

    def test_missing_allocator_stat_or_non_mapping_is_rejected(self):
        _, values = fake_torch()
        for name in STATS:
            candidate = dict(values)
            candidate.pop(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.selected_allocator_stats(candidate)
        for candidate in ([], None, ()): 
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                module.selected_allocator_stats(candidate)

    def test_allocator_aggregates_require_nonnegative_plain_integers(self):
        _, values = fake_torch()
        for value in (True, False, -1, .5, float("nan"), float("inf"), "0", None):
            candidate = dict(values)
            candidate[STATS[0]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.selected_allocator_stats(candidate)
        self.assertEqual(module.selected_allocator_stats({name: 0 for name in STATS}), {name: 0 for name in STATS})

    def test_statistics_whitelist_has_no_snapshot_content_and_keeps_oom_and_retry_counts(self):
        self.assertEqual(tuple(module.ALLOCATOR_STATS), STATS)
        _, values = fake_torch()
        values["num_ooms"], values["num_alloc_retries"] = 2, 3
        selected = module.selected_allocator_stats(values)
        self.assertEqual((selected["num_ooms"], selected["num_alloc_retries"]), (2, 3))
        self.assertIn("excluded from inactive_split", module.ALLOCATOR_QUALIFICATION)
        self.assertIn("unchanged", module.ALLOCATOR_QUALIFICATION)

    def test_capture_preserves_native_synchronized_values_and_adds_safe_stats(self):
        torch, values = fake_torch()
        trace = module.AllocatorMemoryTrace(torch)
        trace.presentation_index = 9
        capture = trace._capture()
        self.assertEqual({name: capture[name] for name in ("allocated_bytes", "reserved_bytes", "free_device_bytes", "total_bytes")},
            {"allocated_bytes": 149686226432, "reserved_bytes": 175040888832,
             "free_device_bytes": 15536357376, "total_bytes": 191495471104})
        self.assertEqual(capture["allocator_stats"], dict(values))
        self.assertEqual(capture["presentation_index"], 9)
        self.assertGreaterEqual(torch.cuda.synchronize.call_count, 2)

    def test_samples_retain_capture_order_presentation_position_and_active_phase_only(self):
        torch, _ = fake_torch()
        trace = module.AllocatorMemoryTrace(torch)
        trace.presentation_index = 8
        frame = trace.begin("synthetic-long-before-forward")
        trace.presentation_index = 9
        trace._capture()
        trace.end(frame)
        samples = trace.allocator_samples
        self.assertEqual([value["capture_index"] for value in samples], list(range(len(samples))))
        self.assertEqual([value["presentation_index"] for value in samples], [8, 9, 9])
        self.assertEqual(samples[1]["active_phases"], ["synthetic-long-before-forward"])
        self.assertNotIn(RAW_SENTINEL, json.dumps(samples))

    def test_headroom_formula_and_failed_parent_numbers_remain_unchanged(self):
        torch, _ = fake_torch()
        trace = module.AllocatorMemoryTrace(torch)
        result = trace.result(limits())
        self.assertEqual(result["headroom"], {"peak_bytes": 175959113728, "total_bytes": 191495471104,
            "free_after_peak_bytes": 15536357376, "required_headroom_bytes": 19149547111, "passed": False})
        self.assertEqual(result["allocator_peak_allocated_bytes"], 149686226432)
        self.assertEqual(result["allocator_peak_reserved_bytes"], 175040888832)
        control_torch, _ = fake_torch()
        control = module.native.MemoryTrace(control_torch).result(limits())
        self.assertEqual({name: result[name] for name in control}, control)

    def test_large_headroom_passes_by_same_rule_without_cache_emptying_or_extra_resets(self):
        torch, _ = fake_torch(total=200 << 30, allocated=100 << 30, reserved=150 << 30, driver=2 << 30)
        trace = module.AllocatorMemoryTrace(torch)
        frame = trace.begin("synthetic-update")
        trace.end(frame)
        result = trace.result(limits())
        self.assertIs(result["headroom"]["passed"], True)
        self.assertEqual(result["headroom"]["free_after_peak_bytes"], 48 << 30)
        self.assertEqual(result["headroom"]["required_headroom_bytes"], 20 << 30)
        self.assertEqual(torch.cuda.reset_peak_memory_stats.call_count, 2)
        torch.cuda.empty_cache.assert_not_called()
        torch.cuda.memory._set_allocator_settings.assert_not_called()

    def test_malformed_stats_aborts_capture_instead_of_making_a_valid_sample(self):
        torch, _ = fake_torch()
        trace = module.AllocatorMemoryTrace(torch)
        torch.cuda.memory_stats.return_value = {"unexpected": RAW_SENTINEL}
        with self.assertRaises(ValueError):
            trace._capture()
        self.assertEqual(trace.allocator_samples, [])

    def test_trace_inherits_original_peak_and_headroom_implementation_without_global_patch(self):
        self.assertTrue(issubclass(module.AllocatorMemoryTrace, module.native.MemoryTrace))
        self.assertIs(module.native, parent.native)
        self.assertIs(module.pilot, parent.pilot)
        tree = ast.parse(read(CANDIDATE))
        trace = top(tree, "AllocatorMemoryTrace")
        self.assertEqual(dump(trace.bases[0]), dump(ast.parse("native.MemoryTrace", mode="eval").body))
        text = ast.get_source_segment(read(CANDIDATE), trace)
        self.assertNotIn("empty_cache(", text)
        self.assertNotIn("reset_peak_memory_stats(", text)
        self.assertNotIn("headroom_report(", text)


class AllocatorRecordedProofAndReplayTests(unittest.TestCase):
    def test_recorded_effective_proof_is_strict_and_never_accepts_raw_snapshot_fields(self):
        torch, _ = fake_torch()
        proof = module.allocator_proof(torch, environ=ENV)
        self.assertIs(module.validate_recorded_allocator_proof(proof), proof)
        for key, value in (("allocator_backend", "cudaMallocAsync"), ("private_api_version_pinned", False),
                           ("effective_settings", {**ENV, "expandable_segments": False}), ("torch_version", "2.9.1")):
            candidate = deepcopy(proof)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_recorded_allocator_proof(candidate)
        for key in ("segments", "blocks", "address", "frames"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_recorded_allocator_proof({**proof, key: RAW_SENTINEL})

    def test_recorded_samples_require_both_actual_long_positions_in_forward_phase(self):
        samples = allocator_samples()
        self.assertIs(module.validate_allocator_samples(samples), samples)
        for candidate in ([], samples[:1], [samples[1]],
                          [{**item, "active_phases": ["actual_backward"]} for item in samples]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                module.validate_allocator_samples(candidate)
        self.assertNotIn(RAW_SENTINEL, json.dumps(samples))

    def test_recorded_sample_payload_order_and_accumulated_counters_cannot_be_forged(self):
        mutations = (
            lambda values: values[0].update(capture_index=True),
            lambda values: values[0].update(presentation_index=17),
            lambda values: values[0].update(frames=[RAW_SENTINEL]),
            lambda values: values[0].update(active_phases=[""]),
            lambda values: values[0]["stats"].update(address=RAW_SENTINEL),
            lambda values: values[0]["stats"].update(num_ooms=1),
            lambda values: values[0]["stats"].update(num_alloc_retries=1),
        )
        for mutate in mutations:
            candidate = allocator_samples()
            mutate(candidate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                module.validate_allocator_samples(candidate)

    def test_operational_plan_pins_one_allocator_and_original_scientific_inputs(self):
        plan = operational_plan()
        self.assertIs(module.validate_allocator_plan(plan), plan)
        for key, value in (("automatic_retry", True), ("automatic_downstream", True), ("diagnostic_only", False),
                           ("prepared", {"path": "other.json", "sha256": "0" * 64})):
            candidate = deepcopy(plan)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_allocator_plan(candidate)
        for key, value in (("value", "expandable_segments:True"), ("backend", "cudaMallocAsync"),
                           ("empty_cache_introduced", True), ("configuration_sweep", True)):
            candidate = deepcopy(plan)
            candidate["allocator_policy"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_allocator_plan(candidate)

    def test_frozen_parent_and_every_unchanged_scientific_component_remain_exact(self):
        self.assertEqual(hashlib.sha256((ROOT / PARENT).read_bytes()).hexdigest(), PARENT_SHA)
        original, candidate = ast.parse(read(PARENT)), ast.parse(read(CANDIDATE))
        for name in ("SCHEMA", "COMPLETE_SCHEMA", "CASE_SCHEMA", "MODES", "PRESENTATION_ROLES", "SETTINGS",
                     "validate_plan", "presentation_rows", "DiagnosticLedger", "make_trainer_class",
                     "inspect_pre_forward", "validate_observations"):
            with self.subTest(name=name):
                self.assertEqual(dump(top(original, name)), dump(top(candidate, name)))

    def test_full_existing_module_ast_has_only_enumerated_allocator_and_provenance_deltas(self):
        assert_scientific_replay(read(CANDIDATE))
        self.assertEqual(module.REQUIRED_CODE, parent.REQUIRED_CODE | ADDITIONS)

    def test_scientific_context_precision_shape_headroom_and_hidden_worker_changes_fail_allowlist(self):
        source = read(CANDIDATE)
        changes = (("expected_parameters=434", "expected_parameters=433"), ('"chunk_tokens": 128', '"chunk_tokens": 64'),
                   ('"student_dtype": "float32"', '"student_dtype": "bfloat16"'), ('"accumulation": 8', '"accumulation": 2'),
                   (".1 <= limits[\"headroom_fraction\"]", ".05 <= limits[\"headroom_fraction\"]"),
                   ('"extension.joint_memory_allocator_v4"', '"extension.joint_memory_admission_v3"'),
                   ('model.to("cuda")', 'model.to("cpu")'))
        for old, new in changes:
            self.assertIn(old, source)
            with self.subTest(old=old), self.assertRaises(AssertionError):
                assert_scientific_replay(source.replace(old, new, 1))

    def test_seventeen_position_four_objective_schedule_and_all_existing_guards_are_preserved(self):
        self.assertEqual(module.MODES, ("sft", "ce_kl", "asft", "prior_anchored"))
        self.assertEqual(module.PRESENTATION_ROLES, parent.PRESENTATION_ROLES)
        self.assertEqual(len(module.PRESENTATION_ROLES), 17)
        self.assertEqual(module.PRESENTATION_ROLES[8:10], ("long", "long"))
        self.assertEqual(module.SETTINGS, parent.SETTINGS)
        self.assertEqual(module.SETTINGS["accumulation"], 8)
        self.assertIs(module.SETTINGS["resume"], False)
        for name in ADDITIONS:
            self.assertIn(name, module.REQUIRED_CODE)

    def test_child_dispatch_uses_new_module_without_retry_and_preserves_uncertain_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "extension/runs/synthetic-v4-dispatch"
            allocation = {"job_id": "990", "node": "owl123"}
            context = {"root": root, "binding_sha256": "a" * 64, "admission": {"job_id": "990"},
                "plan": {"resources": {"wall_time_seconds": 7200, "reserve_seconds": 172800, "scratch_free_gib": 160}},
                "paths": {"plan": "extension/joint_memory_admission_plan.json", "qa": "extension/runs/fixture/native_qa.json",
                    "admission": "extension/operations/synthetic-admission.json"}}
            runner = Mock(return_value=SimpleNamespace(returncode=1))
            with ExitStack() as stack:
                stack.enter_context(patch.object(module, "allocator_preimport_proof", return_value={"synthetic": True}))
                stack.enter_context(patch.object(module.native, "require_gpu_allocation", return_value=allocation))
                stack.enter_context(patch.object(module, "load_inputs", return_value=context))
                stack.enter_context(patch.object(module, "deadline_guard"))
                stack.enter_context(patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)))
                with self.assertRaisesRegex(ValueError, "failed without retry"):
                    module.run(root, context["paths"]["plan"], output, context["paths"]["qa"], context["paths"]["admission"], runner=runner)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], [sys.executable, "-m", "extension.joint_memory_allocator_v4", "--root", str(root),
                "--plan", context["paths"]["plan"], "--output", str(output), "--qa-receipt", context["paths"]["qa"],
                "--admission-receipt", context["paths"]["admission"], "--mode", "sft"])
            failure = json.loads((output / "failure.json").read_text())
            self.assertIs(failure["complete"], False)
            self.assertIs(failure["retry"], False)
            self.assertIs(failure["automatic_downstream"], False)
            self.assertEqual(failure["completed_cases"], {})
            self.assertFalse((output / "complete.json").exists())

    def test_preimport_and_effective_guards_precede_models_and_closeout_preserves_same_policy(self):
        source = read(CANDIDATE)
        tree = ast.parse(source)
        worker = ast.get_source_segment(source, top(tree, "run_case"))
        self.assertLess(worker.index("preimport = allocator_preimport_proof()"), worker.index("    import torch"))
        self.assertLess(worker.index("allocator_before = allocator_proof(torch)"), worker.index("student, reference = load_model(base)"))
        self.assertLess(worker.index("allocator_after = allocator_proof(torch)"), worker.index("memory_result = memory.result(limits)"))
        self.assertIn('require(allocator_after == allocator_before, "Effective allocator settings changed after model work")', worker)
        for name in ("run", "main"):
            body = top(tree, name).body
            self.assertEqual(dump(body[0]), dump(ast.parse("allocator_preimport_proof()").body[0]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertFalse(isinstance(node.func, ast.Attribute) and node.func.attr in
                    {"empty_cache", "_set_allocator_settings", "change_current_allocator", "caching_allocator_enable"})
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and
            any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names)
            for node in tree.body))


if __name__ == "__main__":
    unittest.main()

