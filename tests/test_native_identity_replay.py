"""CPU-only source/dispatch oracles for the explicit GPU-identity replay.

Source equivalence and mocked process dispatch do not establish numerical,
memory, or training success for the v2 GPU programs. Those require new native
GPU receipts. No Torch/model/data dependency is imported by these fixtures.
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
from extension import joint_memory_admission as joint_parent
from extension import joint_memory_admission_v2 as joint
from extension import native_model_admission as native_parent
from extension import native_model_admission_v2 as native


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "extension/native_model_admission.py": "55c73185441a0ac893d8fc95cb5bc8d1a659f767a80d27fe9b6734f8ab8fbb7a",
    "extension/joint_memory_admission.py": "4fcce77d633ea13baab8630c8485fccade52a5026eb51c4a5a75806c712f95fe",
    "extension/pilot_driver.py": "e6b707537298d1b42f21e744dafef4e5edcd604a79ee95d37c2906738d2b9d1a",
    "extension/preserving_trainer.py": "3a46929eaff536f8165e8b1388fddc9c54fde0198c84aac3bcb69cf397d507d0",
    "extension/preserving_objectives.py": "05b079d2da89579436ce53b8a801c0fc51bd6cfef45628c50d23a5af93de3eb4",
    "extension/preserving_data.py": "a560e039d4af86739d16f4c3101eb820535ecebf0373571064db9c588bc69d03",
    "tests/test_native_model_admission.py": "cbbd53d76057badaa1a14afdc36ee422c9c8591ce84e90158dd007215adb2b28",
    "tests/test_joint_memory_admission.py": "f996b0f6c9ecf39c42656c4120335ee53182e6a6a0d721d5c361058fc9b410b6",
}
ADDITIONS = (
    "extension/gpu_identity_v2.py", "tests/test_gpu_identity_v2.py",
    "extension/native_model_admission_v2.py", "extension/joint_memory_admission_v2.py",
    "tests/test_native_identity_replay.py",
)
DELEGATE = '''def assigned_gpu_identity(torch):
    """Use the separately bound public-API identity repair; no ordinal fallback."""
    from extension.gpu_identity_v2 import assigned_gpu_identity as identify
    return identify(torch)'''
NATIVE_ADDITIONS = ''' | frozenset({
    "extension/gpu_identity_v2.py", "tests/test_gpu_identity_v2.py", "extension/native_model_admission_v2.py",
    "extension/joint_memory_admission_v2.py", "tests/test_native_identity_replay.py"})'''
JOINT_ADDITIONS = ''',
    "extension/gpu_identity_v2.py", "tests/test_gpu_identity_v2.py", "extension/native_model_admission_v2.py",
    "extension/joint_memory_admission_v2.py", "tests/test_native_identity_replay.py"})'''


def text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def dump(node):
    return ast.dump(node, include_attributes=False)


def top_node(tree, name):
    matches = [node for node in tree.body if getattr(node, "name", None) == name or
               isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                    for target in node.targets)]
    if len(matches) != 1:
        raise AssertionError("Expected one exact top-level node: " + name)
    return matches[0]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("Operational source replacement must be unique")
    return source.replace(old, new, 1)


def expected_text(parent, kind):
    tree = ast.parse(parent)
    inventory = ast.get_source_segment(parent, top_node(tree, "REQUIRED_CODE"))
    if kind == "native":
        result = replace_once(parent, inventory, inventory + NATIVE_ADDITIONS)
        old_identity = ast.get_source_segment(parent, top_node(tree, "assigned_gpu_identity"))
        result = replace_once(result, old_identity, DELEGATE)
    elif kind == "joint":
        if not inventory.endswith("})"):
            raise AssertionError("Frozen joint inventory expression changed")
        result = replace_once(parent, inventory, inventory[:-2] + JOINT_ADDITIONS)
        result = replace_once(result, "from extension import native_model_admission as native",
                              "from extension import native_model_admission_v2 as native")
    else:
        raise AssertionError("Unknown replay kind")
    return replace_once(result, '"extension.' + ("native_model_admission" if kind == "native" else "joint_memory_admission") + '"',
                        '"extension.' + ("native_model_admission_v2" if kind == "native" else "joint_memory_admission_v2") + '"')


def expected_ast(parent, kind):
    """Independently apply the small AST allowlist, without erasing functions."""
    tree = deepcopy(ast.parse(parent))
    inventory = top_node(tree, "REQUIRED_CODE")
    extra = ast.Set(elts=[ast.Constant(value=value) for value in ADDITIONS])
    if kind == "native":
        inventory.value = ast.BinOp(left=inventory.value, op=ast.BitOr(),
                                   right=ast.Call(func=ast.Name(id="frozenset", ctx=ast.Load()), args=[extra], keywords=[]))
        old_identity = top_node(tree, "assigned_gpu_identity")
        tree.body[tree.body.index(old_identity)] = ast.parse(DELEGATE).body[0]
        old_target, new_target = "extension.native_model_admission", "extension.native_model_admission_v2"
    elif kind == "joint":
        if not (isinstance(inventory.value, ast.BinOp) and isinstance(inventory.value.right, ast.Call) and
                len(inventory.value.right.args) == 1 and isinstance(inventory.value.right.args[0], ast.Set)):
            raise AssertionError("Frozen joint inventory AST changed")
        inventory.value.right.args[0].elts.extend(extra.elts)
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "extension" and
                   [(item.name, item.asname) for item in node.names] == [("native_model_admission", "native")]]
        if len(imports) != 1:
            raise AssertionError("Expected one exact frozen native import")
        imports[0].names[0].name = "native_model_admission_v2"
        old_target, new_target = "extension.joint_memory_admission", "extension.joint_memory_admission_v2"
    else:
        raise AssertionError("Unknown replay kind")
    targets = [node for node in ast.walk(top_node(tree, "run")) if isinstance(node, ast.Constant) and node.value == old_target]
    if len(targets) != 1:
        raise AssertionError("Expected one exact child-process module target")
    targets[0].value = new_target
    return tree


def assert_ast_replay(parent, candidate, kind):
    if dump(expected_ast(parent, kind)) != dump(ast.parse(candidate)):
        raise AssertionError("A non-allowlisted diagnostic implementation changed")


class ReplayDispatchFixture:
    def __init__(self, temporary, kind):
        self.root = Path(temporary).resolve()
        self.kind = kind
        self.module = native if kind == "native" else joint
        self.output = self.root / "extension/runs/identity-replay-fixture"
        self.allocation = {"job_id": "990", "node": "owl123"}
        resources = {"wall_time_seconds": 7200, "reserve_seconds": 172800, "scratch_free_gib": 160}
        self.context = {"root": self.root, "plan": {"resources": resources}, "binding_sha256": "a" * 64,
                        "admission": {"job_id": "990"}}
        if kind == "native":
            self.context.update({"plan_path": self.root / "extension/native_model_admission_plan.json",
                                 "prepared_path": self.root / "extension/runs/sealed/prepared.json",
                                 "qa_path": self.root / "extension/runs/sealed/native_qa.json",
                                 "admission_path": self.root / "extension/operations/admission.json"})
        else:
            self.context["paths"] = {"plan": "extension/joint_memory_plan.json", "qa": "extension/runs/sealed/native_qa.json",
                                     "admission": "extension/operations/admission.json"}

    def call(self, runner, allocation_error=None):
        with ExitStack() as stack:
            guard_owner = native if self.kind == "native" else joint.native
            self.guard = stack.enter_context(patch.object(guard_owner, "require_gpu_allocation",
                side_effect=allocation_error, return_value=self.allocation))
            self.loader = stack.enter_context(patch.object(self.module, "load_inputs", return_value=self.context))
            stack.enter_context(patch.object(self.module, "deadline_guard"))
            stack.enter_context(patch.object(self.module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30)))
            if self.kind == "native":
                return native.run(self.root, "extension/native_model_admission_plan.json", "extension/runs/sealed/prepared.json",
                                  self.output, "extension/runs/sealed/native_qa.json", "extension/operations/admission.json", runner=runner)
            return joint.run(self.root, "extension/joint_memory_plan.json", self.output,
                             "extension/runs/sealed/native_qa.json", "extension/operations/admission.json", runner=runner)

    def expected_command(self):
        if self.kind == "native":
            return [sys.executable, "-m", "extension.native_model_admission_v2", "--root", str(self.root),
                    "--plan", "extension/native_model_admission_plan.json", "--prepared", "extension/runs/sealed/prepared.json",
                    "--output", str(self.output), "--qa-receipt", "extension/runs/sealed/native_qa.json",
                    "--admission-receipt", "extension/operations/admission.json", "--case", "sft-short"]
        return [sys.executable, "-m", "extension.joint_memory_admission_v2", "--root", str(self.root),
                "--plan", self.context["paths"]["plan"], "--output", str(self.output),
                "--qa-receipt", self.context["paths"]["qa"], "--admission-receipt", self.context["paths"]["admission"],
                "--mode", "sft"]


class FrozenSourceReplayTests(unittest.TestCase):
    def test_all_frozen_parents_and_scientific_helpers_retain_exact_source_hashes(self):
        for name, expected in FROZEN.items():
            with self.subTest(name=name):
                path = ROOT / name
                self.assertFalse(path.is_symlink())
                self.assertLess(path.stat().st_size, 2 << 20)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_native_source_has_only_the_exact_declared_textual_changes(self):
        self.assertEqual(text("extension/native_model_admission_v2.py"),
                         expected_text(text("extension/native_model_admission.py"), "native"))

    def test_joint_source_has_only_the_exact_declared_textual_changes(self):
        self.assertEqual(text("extension/joint_memory_admission_v2.py"),
                         expected_text(text("extension/joint_memory_admission.py"), "joint"))

    def test_native_full_ast_diff_is_limited_to_inventory_dispatch_and_identity(self):
        assert_ast_replay(text("extension/native_model_admission.py"), text("extension/native_model_admission_v2.py"), "native")

    def test_joint_full_ast_diff_is_limited_to_inventory_dispatch_and_native_import(self):
        assert_ast_replay(text("extension/joint_memory_admission.py"), text("extension/joint_memory_admission_v2.py"), "joint")

    def test_scientific_mutations_or_old_dispatch_cannot_hide_inside_allowlist(self):
        mutations = (
            ("native", 'len(student_items) == 434', 'len(student_items) == 433'),
            ("native", 'settings.get("accumulation") == 2', 'settings.get("accumulation") == 8'),
            ("native", '"finite_moments"', '"omitted_finite_moments"'),
            ("native", '"extension.native_model_admission_v2"', '"extension.native_model_admission"'),
            ("joint", '"beta": .05', '"beta": .5'),
            ("joint", 'expected_parameters=434', 'expected_parameters=433'),
            ("joint", '"extension.joint_memory_admission_v2"', '"extension.joint_memory_admission"'),
        )
        for kind, old, new in mutations:
            with self.subTest(kind=kind, old=old):
                parent = text("extension/" + ("native_model_admission.py" if kind == "native" else "joint_memory_admission.py"))
                candidate = text("extension/" + ("native_model_admission_v2.py" if kind == "native" else "joint_memory_admission_v2.py"))
                self.assertIn(old, candidate)
                with self.assertRaises(AssertionError):
                    assert_ast_replay(parent, candidate.replace(old, new, 1), kind)

    def test_model_versions_masks_modes_checks_and_joint_schedule_constants_are_unchanged(self):
        for name in ("REVISION", "REPOSITORY", "ORIGINAL_FILES", "TOKENIZER_FILES", "MODES", "CASES", "VERSIONS",
                     "REQUIRED_CHECKS", "AUDIT_FIELDS"):
            with self.subTest(module="native", name=name):
                self.assertEqual(getattr(native, name), getattr(native_parent, name))
        for name in ("SCHEMA", "COMPLETE_SCHEMA", "CASE_SCHEMA", "MODES", "PRESENTATION_ROLES", "SETTINGS"):
            with self.subTest(module="joint", name=name):
                self.assertEqual(getattr(joint, name), getattr(joint_parent, name))

    def test_whole_validation_training_memory_and_checkpoint_bodies_are_identical(self):
        cases = (("native", ("validate_plan", "_example", "validate_prepared", "load_inputs", "validate_case_receipt",
                             "tensor_inventory", "assert_finite_model", "MemoryTrace", "check_parity", "headroom_report", "run_case")),
                 ("joint", ("validate_plan", "presentation_rows", "DiagnosticLedger", "make_trainer_class", "inspect_pre_forward",
                            "validate_observations", "load_inputs", "validate_case_receipt", "run_case")))
        for kind, names in cases:
            parent = ast.parse(text("extension/" + ("native_model_admission.py" if kind == "native" else "joint_memory_admission.py")))
            candidate = ast.parse(text("extension/" + ("native_model_admission_v2.py" if kind == "native" else "joint_memory_admission_v2.py")))
            for name in names:
                with self.subTest(kind=kind, name=name):
                    self.assertEqual(dump(top_node(parent, name)), dump(top_node(candidate, name)))

    def test_required_source_inventories_preserve_all_old_dependencies_and_add_exact_repair_files(self):
        self.assertEqual(native.REQUIRED_CODE, native_parent.REQUIRED_CODE | frozenset(ADDITIONS))
        self.assertEqual(joint.REQUIRED_CODE, joint_parent.REQUIRED_CODE | frozenset(ADDITIONS))
        for name in native.REQUIRED_CODE | joint.REQUIRED_CODE:
            with self.subTest(name=name):
                path = ROOT / name
                self.assertFalse(path.is_symlink())
                self.assertTrue(path.is_file())
                self.assertLess(path.stat().st_size, 2 << 20)
                self.assertIn(path.suffix, {".py", ".json", ".slurm"})
                self.assertEqual(len(hashlib.sha256(path.read_bytes()).hexdigest()), 64)

    def test_joint_import_uses_v2_identity_path_while_scientific_pilot_stays_frozen(self):
        self.assertIs(joint.native, native)
        self.assertIs(joint.pilot, joint_parent.pilot)
        self.assertIsNot(joint.native, native_parent)
        self.assertEqual(joint.native.ORIGINAL_FILES, joint_parent.native.ORIGINAL_FILES)


class OperationalReplayDispatchTests(unittest.TestCase):
    def test_native_identity_function_delegates_once_without_copying_or_changing_proof(self):
        sentinel_torch, sentinel_proof = object(), {"fixture": "identity only; no real CUDA"}
        with patch.object(identity, "assigned_gpu_identity", return_value=sentinel_proof) as identify:
            self.assertIs(native.assigned_gpu_identity(sentinel_torch), sentinel_proof)
        identify.assert_called_once_with(sentinel_torch)

    def test_joint_identity_path_reaches_the_same_versioned_helper(self):
        sentinel_torch, sentinel_proof = object(), {"fixture": "logical0 path only"}
        with patch.object(identity, "assigned_gpu_identity", return_value=sentinel_proof) as identify:
            self.assertIs(joint.native.assigned_gpu_identity(sentinel_torch), sentinel_proof)
        identify.assert_called_once_with(sentinel_torch)

    def test_native_first_child_dispatch_preserves_arguments_and_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayDispatchFixture(temporary, "native")
            runner = Mock(return_value=SimpleNamespace(returncode=1))
            with self.assertRaisesRegex(ValueError, "failed without retry"):
                fixture.call(runner)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], fixture.expected_command())
            self.assertEqual(runner.call_args.kwargs["cwd"], str(fixture.root))
            self.assertIs(runner.call_args.kwargs["check"], False)
            self.assertTrue(0 < runner.call_args.kwargs["timeout"] <= 7200)
            failure = json.loads((fixture.output / "failure.json").read_text())
            self.assertEqual(failure["completed_cases"], {})
            self.assertIs(failure["retry"], False)
            self.assertIs(failure["automatic_downstream"], False)
            self.assertFalse((fixture.output / "complete.json").exists())

    def test_joint_first_child_dispatch_preserves_arguments_and_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayDispatchFixture(temporary, "joint")
            runner = Mock(return_value=SimpleNamespace(returncode=1))
            with self.assertRaisesRegex(ValueError, "failed without retry"):
                fixture.call(runner)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], fixture.expected_command())
            self.assertEqual(runner.call_args.kwargs["cwd"], str(fixture.root))
            self.assertIs(runner.call_args.kwargs["check"], False)
            self.assertTrue(0 < runner.call_args.kwargs["timeout"] <= 7200)
            failure = json.loads((fixture.output / "failure.json").read_text())
            self.assertEqual(failure["completed_cases"], {})
            self.assertIs(failure["retry"], False)
            self.assertIs(failure["automatic_downstream"], False)
            self.assertIs(failure["resume_supported"], False)
            self.assertFalse((fixture.output / "complete.json").exists())

    def test_timed_out_child_is_preserved_as_failed_and_never_replayed(self):
        for kind in ("native", "joint"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                fixture = ReplayDispatchFixture(temporary, kind)
                error = subprocess.TimeoutExpired(["fixture child"], 7200)
                runner = Mock(side_effect=error)
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    fixture.call(runner)
                self.assertIs(raised.exception, error)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(runner.call_args.args[0], fixture.expected_command())
                self.assertEqual(json.loads((fixture.output / "failure.json").read_text())["exception"], "TimeoutExpired")
                self.assertFalse((fixture.output / "complete.json").exists())

    def test_allocation_rejection_precedes_loader_output_and_subprocess(self):
        for kind in ("native", "joint"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                fixture = ReplayDispatchFixture(temporary, kind)
                runner = Mock()
                with self.assertRaisesRegex(ValueError, "fixture allocation blocked"):
                    fixture.call(runner, ValueError("fixture allocation blocked"))
                fixture.loader.assert_not_called()
                runner.assert_not_called()
                self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
