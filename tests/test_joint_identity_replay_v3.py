"""Structural and synthetic CPU regressions for the joint-worker initializer.

The old/v2 source remains frozen and its observed failure is not relabeled.
Executing one extracted assignment below uses four tiny synthetic witnesses,
not Torch, models, private caches, a training update, or GPU admission evidence.
"""
import ast
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

from extension import gpu_identity_v2 as identity
from extension import joint_memory_admission_v2 as parent
from extension import joint_memory_admission_v3 as module


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["rows", "metrics", "lr_used", "lr_after", "gradient_norms"]
ADDITIONS = ("extension/joint_memory_admission_v3.py", "tests/test_joint_identity_replay_v3.py")
FROZEN = {
    "extension/joint_memory_admission.py": "4fcce77d633ea13baab8630c8485fccade52a5026eb51c4a5a75806c712f95fe",
    "extension/joint_memory_admission_v2.py": "382c661b7cfa5eb2ba18ee4bd91c553efe5d5a9349b1d6d3322b362795472abe",
    "tests/test_joint_memory_admission.py": "f996b0f6c9ecf39c42656c4120335ee53182e6a6a0d721d5c361058fc9b410b6",
    "tests/test_native_identity_replay.py": "b5c2e15a82717daafcde2dad4896a684cd30be224304463e5dfa720436323fe9",
}
OLD_INITIALIZER = "rows, metrics, lr_used, lr_after, gradient_norms = presentation_rows(prepared), [], [], [], [], []"
NEW_INITIALIZER = "rows, metrics, lr_used, lr_after, gradient_norms = presentation_rows(prepared), [], [], [], []"


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def dump(node):
    return ast.dump(node, include_attributes=False)


def top_node(tree, name):
    nodes = [node for node in tree.body if getattr(node, "name", None) == name or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                  for target in node.targets)]
    if len(nodes) != 1:
        raise AssertionError("Expected one top-level source node: " + name)
    return nodes[0]


def fixed_cardinality_errors(source):
    """Compare only statically fixed tuple/list destructuring, recursively.

    Calls, generators, and starred unpacking have runtime-dependent cardinality
    and are intentionally not asserted by this narrow structural oracle.
    """
    errors = []
    def compare(target, value, owner, line, position):
        if not (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))):
            return
        if any(isinstance(item, ast.Starred) for item in target.elts + value.elts):
            return
        if len(target.elts) != len(value.elts):
            errors.append({"owner": owner, "line": line, "position": position,
                           "targets": len(target.elts), "values": len(value.elts)})
            return
        for index, (child_target, child_value) in enumerate(zip(target.elts, value.elts)):
            compare(child_target, child_value, owner, line, position + [index])
    def visit(node, owner="<module>"):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = node.name if owner == "<module>" else owner + "." + node.name
        if isinstance(node, ast.Assign):
            for target in node.targets:
                compare(target, node.value, owner, node.lineno, [])
        for child in ast.iter_child_nodes(node):
            visit(child, owner)
    visit(ast.parse(source))
    return errors


def initializer(tree):
    matches = [node for node in ast.walk(top_node(tree, "run_case")) if isinstance(node, ast.Assign) and
               len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and
               all(isinstance(item, ast.Name) for item in node.targets[0].elts) and
               [item.id for item in node.targets[0].elts] == TARGETS]
    if len(matches) != 1:
        raise AssertionError("Expected one actual five-target worker initializer")
    return matches[0]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("Every allowed textual replacement must be unique")
    return source.replace(old, new, 1)


def expected_source(source):
    inventory = ast.get_source_segment(source, top_node(ast.parse(source), "REQUIRED_CODE"))
    if not inventory.endswith("})"):
        raise AssertionError("Frozen source-inventory expression changed")
    extension = ',\n    "extension/joint_memory_admission_v3.py", "tests/test_joint_identity_replay_v3.py"})'
    result = replace_once(source, inventory, inventory[:-2] + extension)
    result = replace_once(result, '"extension.joint_memory_admission_v2"', '"extension.joint_memory_admission_v3"')
    return replace_once(result, OLD_INITIALIZER, NEW_INITIALIZER)


def expected_ast(source):
    tree = deepcopy(ast.parse(source))
    inventory = top_node(tree, "REQUIRED_CODE")
    if not (isinstance(inventory.value, ast.BinOp) and isinstance(inventory.value.right, ast.Call) and
            len(inventory.value.right.args) == 1 and isinstance(inventory.value.right.args[0], ast.Set)):
        raise AssertionError("Frozen source-inventory AST changed")
    inventory.value.right.args[0].elts.extend(ast.Constant(value=name) for name in ADDITIONS)
    targets = [node for node in ast.walk(top_node(tree, "run")) if isinstance(node, ast.Constant) and
               node.value == "extension.joint_memory_admission_v2"]
    if len(targets) != 1:
        raise AssertionError("Expected one unchanged subprocess dispatch location")
    targets[0].value = "extension.joint_memory_admission_v3"
    assignment = initializer(tree)
    if not (isinstance(assignment.value, ast.Tuple) and len(assignment.value.elts) == 6 and
            isinstance(assignment.value.elts[-1], ast.List) and not assignment.value.elts[-1].elts):
        raise AssertionError("The frozen six-value regression differs")
    assignment.value.elts.pop()
    return tree


def assert_replay(source, candidate):
    if dump(expected_ast(source)) != dump(ast.parse(candidate)):
        raise AssertionError("Non-allowlisted joint-worker logic changed")


def synthetic_prepared():
    rows = []
    for index, length in enumerate((4, 5, 6, 513)):
        tokens = [1, 2] + [3] * (length - 4) + [151645, 198]
        rows.append({"id": "synthetic-witness-" + str(index), "input_ids": tokens,
                     "labels": [-100, -100] + tokens[2:], "attention_mask": [1] * length})
    return {"rows": {"short": rows[:3], "long": rows[3]}}


def execute_initializer(source, prepared, presentation_rows):
    node = deepcopy(initializer(ast.parse(source)))
    fragment = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"prepared": prepared, "presentation_rows": presentation_rows, "__builtins__": {}}
    exec(compile(fragment, "<actual-extracted-joint-initializer>", "exec"), namespace)
    return namespace


class DispatchFixture:
    def __init__(self, temporary):
        self.root = Path(temporary).resolve()
        self.output = self.root / "extension/runs/joint-v3-dispatch-fixture"
        self.allocation = {"job_id": "990", "node": "owl123"}
        self.context = {"root": self.root, "binding_sha256": "a" * 64, "admission": {"job_id": "990"},
                        "plan": {"resources": {"wall_time_seconds": 7200, "reserve_seconds": 172800,
                                                "scratch_free_gib": 160}},
                        "paths": {"plan": "extension/joint_memory_admission_plan.json",
                                  "qa": "extension/runs/sealed/joint_qa.json",
                                  "admission": "extension/operations/current-v3-admission.json"}}

    def call(self, runner, allocation_error=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(module.native, "require_gpu_allocation", return_value=self.allocation,
                                             side_effect=allocation_error))
            self.loader = stack.enter_context(patch.object(module, "load_inputs", return_value=self.context))
            stack.enter_context(patch.object(module, "deadline_guard"))
            stack.enter_context(patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)))
            return module.run(self.root, self.context["paths"]["plan"], self.output, self.context["paths"]["qa"],
                              self.context["paths"]["admission"], runner=runner)

    def expected_command(self):
        return [sys.executable, "-m", "extension.joint_memory_admission_v3", "--root", str(self.root),
                "--plan", self.context["paths"]["plan"], "--output", str(self.output),
                "--qa-receipt", self.context["paths"]["qa"], "--admission-receipt", self.context["paths"]["admission"],
                "--mode", "sft"]


class JointInitializerRegressionTests(unittest.TestCase):
    def test_frozen_original_v2_and_old_test_sources_remain_exact(self):
        for name, expected in FROZEN.items():
            with self.subTest(name=name):
                path = ROOT / name
                self.assertFalse(path.is_symlink())
                self.assertLess(path.stat().st_size, 2 << 20)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_generic_literal_cardinality_oracle_handles_nested_chain_and_unknown_shapes(self):
        valid = "a,b = (1,2)\n[a,b] = [1,2]\na,(b,c) = (1,(2,3))\na,b = dynamic()\na,*rest = (1,2,3)\n"
        self.assertEqual(fixed_cardinality_errors(valid), [])
        invalid = "def worker():\n a,b=(1,2,3)\n a,(b,c)=(1,(2,))\n (a,b)=(c,d)=(1,)\n"
        failures = fixed_cardinality_errors(invalid)
        self.assertEqual([(item["targets"], item["values"], item["position"]) for item in failures],
                         [(2, 3, []), (2, 1, [1]), (2, 1, []), (2, 1, [])])
        self.assertTrue(all(item["owner"] == "worker" for item in failures))

    def test_original_and_v2_worker_cardinality_failure_is_explicitly_preserved(self):
        for name in ("extension/joint_memory_admission.py", "extension/joint_memory_admission_v2.py"):
            with self.subTest(name=name):
                failures = fixed_cardinality_errors(read(name))
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["owner"], "run_case")
                self.assertEqual((failures[0]["targets"], failures[0]["values"]), (5, 6))

    def test_v3_and_native_worker_clones_have_no_fixed_literal_cardinality_errors(self):
        for name in ("extension/joint_memory_admission_v3.py", "extension/native_model_admission.py",
                     "extension/native_model_admission_v2.py"):
            with self.subTest(name=name):
                self.assertEqual(fixed_cardinality_errors(read(name)), [])

    def test_actual_v3_initializer_has_five_targets_and_four_distinct_empty_list_literals(self):
        node = initializer(ast.parse(read("extension/joint_memory_admission_v3.py")))
        self.assertEqual([target.id for target in node.targets[0].elts], TARGETS)
        self.assertIsInstance(node.value, ast.Tuple)
        self.assertEqual(len(node.value.elts), 5)
        first = node.value.elts[0]
        self.assertEqual(dump(first), dump(ast.parse("presentation_rows(prepared)", mode="eval").body))
        self.assertTrue(all(isinstance(value, ast.List) and value.elts == [] for value in node.value.elts[1:]))
        self.assertEqual(len({id(value) for value in node.value.elts[1:]}), 4)

    def test_execute_actual_extracted_initializer_preserves_all_seventeen_witness_positions(self):
        prepared = synthetic_prepared()
        before = deepcopy(prepared)
        namespace = execute_initializer(read("extension/joint_memory_admission_v3.py"), prepared, module.presentation_rows)
        self.assertEqual(set(namespace) - {"prepared", "presentation_rows", "__builtins__"}, set(TARGETS))
        lookup = {"short" + str(index): row for index, row in enumerate(prepared["rows"]["short"])}
        lookup["long"] = prepared["rows"]["long"]
        self.assertEqual(len(namespace["rows"]), 17)
        for position, role in enumerate(module.PRESENTATION_ROLES):
            self.assertIs(namespace["rows"][position], lookup[role])
        self.assertIs(namespace["rows"][8], namespace["rows"][9])
        arrays = [namespace[name] for name in TARGETS[1:]]
        self.assertEqual(arrays, [[], [], [], []])
        self.assertEqual(len({id(value) for value in arrays}), 4)
        arrays[0].append("synthetic-metric")
        self.assertEqual(arrays[1:], [[], [], []])
        self.assertEqual(prepared, before)

    def test_old_extracted_initializers_reproduce_unpack_error_without_loading_models(self):
        for name in ("extension/joint_memory_admission.py", "extension/joint_memory_admission_v2.py"):
            with self.subTest(name=name):
                prepared = synthetic_prepared()
                before = deepcopy(prepared)
                with self.assertRaisesRegex(ValueError, "too many values to unpack"):
                    execute_initializer(read(name), prepared, parent.presentation_rows)
                self.assertEqual(prepared, before)


class JointV3ReplayTests(unittest.TestCase):
    def test_exact_full_source_delta_is_initializer_inventory_and_child_module_only(self):
        self.assertEqual(read("extension/joint_memory_admission_v3.py"), expected_source(read("extension/joint_memory_admission_v2.py")))

    def test_independent_full_ast_delta_has_the_same_three_item_allowlist(self):
        assert_replay(read("extension/joint_memory_admission_v2.py"), read("extension/joint_memory_admission_v3.py"))

    def test_scientific_or_shape_changes_and_old_child_dispatch_fail_the_allowlist(self):
        source, candidate = read("extension/joint_memory_admission_v2.py"), read("extension/joint_memory_admission_v3.py")
        changes = (("expected_parameters=434", "expected_parameters=433"), ('"beta": .05', '"beta": .5'),
                   ('"accumulation": 8', '"accumulation": 2'), ('"finite_moments"', '"dropped_finite_moments"'),
                   ('"extension.joint_memory_admission_v3"', '"extension.joint_memory_admission_v2"'),
                   (NEW_INITIALIZER, OLD_INITIALIZER))
        for old, new in changes:
            with self.subTest(old=old):
                self.assertIn(old, candidate)
                with self.assertRaises(AssertionError):
                    assert_replay(source, candidate.replace(old, new, 1))

    def test_scientific_constants_validators_and_other_worker_components_are_unchanged(self):
        for name in ("SCHEMA", "COMPLETE_SCHEMA", "CASE_SCHEMA", "MODES", "PRESENTATION_ROLES", "SETTINGS"):
            with self.subTest(name=name):
                self.assertEqual(getattr(module, name), getattr(parent, name))
        original, repaired = ast.parse(read("extension/joint_memory_admission_v2.py")), ast.parse(read("extension/joint_memory_admission_v3.py"))
        for name in ("validate_plan", "presentation_rows", "DiagnosticLedger", "make_trainer_class", "inspect_pre_forward",
                     "validate_observations", "load_inputs", "validate_case_receipt", "main"):
            with self.subTest(name=name):
                self.assertEqual(dump(top_node(original, name)), dump(top_node(repaired, name)))

    def test_inventory_adds_only_v3_source_and_its_regression_tests(self):
        self.assertEqual(module.REQUIRED_CODE, parent.REQUIRED_CODE | frozenset(ADDITIONS))
        for name in module.REQUIRED_CODE:
            with self.subTest(name=name):
                path = ROOT / name
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertLess(path.stat().st_size, 2 << 20)

    def test_uuid_repair_and_frozen_pilot_bindings_are_preserved_without_global_patching(self):
        self.assertIs(module.native, parent.native)
        self.assertIs(module.pilot, parent.pilot)
        torch_fixture, proof = object(), {"fixture": "identity delegation only"}
        with patch.object(identity, "assigned_gpu_identity", return_value=proof) as identify:
            self.assertIs(module.native.assigned_gpu_identity(torch_fixture), proof)
        identify.assert_called_once_with(torch_fixture)

    def test_child_uses_v3_module_with_exact_old_arguments_and_no_retry_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DispatchFixture(temporary)
            runner = Mock(return_value=SimpleNamespace(returncode=1))
            with self.assertRaisesRegex(ValueError, "failed without retry"):
                fixture.call(runner)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], fixture.expected_command())
            self.assertEqual(runner.call_args.kwargs["cwd"], str(fixture.root))
            self.assertIs(runner.call_args.kwargs["check"], False)
            self.assertTrue(0 < runner.call_args.kwargs["timeout"] <= 7200)
            failure = json.loads((fixture.output / "failure.json").read_text())
            self.assertIs(failure["complete"], False)
            self.assertIs(failure["retry"], False)
            self.assertIs(failure["resume_supported"], False)
            self.assertIs(failure["automatic_downstream"], False)
            self.assertEqual(failure["completed_cases"], {})
            self.assertFalse((fixture.output / "complete.json").exists())

    def test_timeout_is_preserved_without_attempting_a_second_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DispatchFixture(temporary)
            error = subprocess.TimeoutExpired(["synthetic-v3-child"], 7200)
            runner = Mock(side_effect=error)
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                fixture.call(runner)
            self.assertIs(raised.exception, error)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], fixture.expected_command())
            self.assertEqual(json.loads((fixture.output / "failure.json").read_text())["exception"], "TimeoutExpired")
            self.assertFalse((fixture.output / "complete.json").exists())

    def test_allocation_failure_stops_before_inputs_output_or_child_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DispatchFixture(temporary)
            runner = Mock()
            with self.assertRaisesRegex(ValueError, "synthetic allocation blocked"):
                fixture.call(runner, ValueError("synthetic allocation blocked"))
            fixture.loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
