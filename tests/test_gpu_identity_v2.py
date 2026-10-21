"""Offline fixtures for the separately versioned assigned-GPU identity helper.

These tests never import Torch, inspect a real GPU, or invoke nvidia-smi. The
runner and every CUDA observation are explicit fixtures; native execution is a
separate admission obligation.
"""
from copy import deepcopy
import builtins
import importlib
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from extension import gpu_identity_v2 as module


BARE = "00112233-4455-6677-8899-aabbccddeeff"
CANONICAL = "GPU-" + BARE
OTHER = "GPU-10112233-4455-6677-8899-aabbccddeeff"
DRIVER = "580.105.08"
NAME = "NVIDIA B200"
ABSENT = object()


class FixtureUUID:
    def __init__(self, text=BARE, byte_values=ABSENT):
        self.text = text
        if byte_values is not ABSENT:
            self.bytes = byte_values

    def __str__(self):
        return self.text


def fake_torch(raw_uuid=CANONICAL, *, name=NAME, property_name=NAME):
    properties = SimpleNamespace(uuid=raw_uuid, name=property_name)
    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        device_count=Mock(return_value=1),
        current_device=Mock(return_value=0),
        is_bf16_supported=Mock(return_value=True),
        get_device_properties=Mock(return_value=properties),
        get_device_name=Mock(return_value=name),
    )
    return SimpleNamespace(cuda=cuda), properties


def completed(stdout=CANONICAL + ", " + DRIVER + "\n", *, stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def observe(raw_uuid=CANONICAL, **kwargs):
    torch, properties = fake_torch(raw_uuid, **kwargs)
    runner = Mock(return_value=completed())
    proof = module.assigned_gpu_identity(torch, runner=runner)
    return proof, torch, properties, runner


class NormalizeAssignedUUIDTests(unittest.TestCase):
    def test_bare_torch_29_uuid_becomes_full_gpu_uuid(self):
        self.assertEqual(module.normalize_uuid(BARE), CANONICAL)

    def test_prefixed_uuid_remains_the_same_physical_identity(self):
        self.assertEqual(module.normalize_uuid(CANONICAL), CANONICAL)

    def test_hex_case_is_canonicalized_without_changing_uuid_bytes(self):
        for value in (BARE.upper(), "GPU-" + BARE.upper()):
            with self.subTest(value=value):
                result = module.normalize_uuid(value)
                self.assertEqual(result, CANONICAL)
                self.assertEqual(UUID(result[4:]).bytes, UUID(BARE).bytes)

    def test_uuid_object_string_protocol_is_supported_without_torch_import(self):
        self.assertEqual(module.normalize_uuid(FixtureUUID()), CANONICAL)
        self.assertEqual(module.normalize_uuid(UUID(BARE)), CANONICAL)

    def test_abbreviated_ordinal_mig_whitespace_and_non_uuid_inputs_fail(self):
        invalid = (None, True, 0, 1, b"00112233-4455-6677-8899-aabbccddeeff", "", "0", "GPU-0",
                   "00112233445566778899aabbccddeeff", "00112233-4455-6677-8899-aabbccddee",
                   "00112233-4455-6677-8899-aabbccddeefg", "MIG-" + BARE,
                   "MIG-" + CANONICAL + "/1/0", " " + BARE, BARE + " ", BARE + "\n",
                   CANONICAL + "," + OTHER, CANONICAL + "; echo unsafe", "GPU-GPU-" + BARE, "gpu-" + BARE)
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    module.normalize_uuid(value)

    def test_nil_uuid_is_not_an_assigned_device_identity(self):
        for value in ("00000000-0000-0000-0000-000000000000", "GPU-00000000-0000-0000-0000-000000000000"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    module.normalize_uuid(value)


class AssignedGPUIdentityTests(unittest.TestCase):
    def test_query_is_uuid_scoped_with_exact_bounded_subprocess_arguments(self):
        proof, torch, properties, runner = observe()
        command = ["nvidia-smi", "-i", CANONICAL, "--query-gpu=uuid,driver_version",
                   "--format=csv,noheader,nounits"]
        runner.assert_called_once_with(command, check=True, capture_output=True, text=True, timeout=15)
        self.assertEqual(proof["uuid"], CANONICAL)
        self.assertEqual(proof["driver_version"], DRIVER)
        self.assertEqual(proof["name"], NAME)
        self.assertEqual(proof["properties"], str(properties))
        self.assertEqual(proof["query_scope"], "CUDA logical0 assigned device UUID only; no ordinal fallback")
        self.assertEqual(proof["logical_cuda_device"], 0)
        self.assertEqual(proof["visible_cuda_device_count"], 1)
        self.assertIs(proof["bf16_supported"], True)
        self.assertEqual(proof["query"]["command"], command)
        self.assertEqual(proof["query"]["stdout"], CANONICAL + ", " + DRIVER + "\n")
        self.assertEqual(proof["query"]["stderr"], "")
        self.assertEqual(proof["query"]["timeout_seconds"], 15)
        self.assertEqual(proof["schema"], module.SCHEMA)
        self.assertEqual(proof["normalization"], "GPU-prefixed full dashed UUID")
        self.assertEqual(proof["torch_uuid_raw"], CANONICAL)
        self.assertEqual(proof["torch_uuid_type"], "builtins.str")
        self.assertIsNone(proof["torch_uuid_bytes"])
        self.assertEqual(proof["source_evidence"], module.SOURCE_EVIDENCE)
        self.assertIsNot(proof["source_evidence"], module.SOURCE_EVIDENCE)
        default_torch, _ = fake_torch()
        with patch.object(module.subprocess, "run", return_value=completed()) as default_runner:
            self.assertEqual(module.assigned_gpu_identity(default_torch)["uuid"], CANONICAL)
        default_runner.assert_called_once_with(command, check=True, capture_output=True, text=True, timeout=15)

    def test_bare_torch_uuid_object_bytes_and_raw_provenance_are_preserved(self):
        raw = FixtureUUID(BARE, UUID(BARE).bytes)
        proof, _, _, _ = observe(raw)
        self.assertEqual(proof["uuid"], CANONICAL)
        self.assertEqual(proof["torch_uuid_raw"], BARE)
        self.assertIn("FixtureUUID", proof["torch_uuid_type"])
        self.assertEqual(proof["normalization"], "bare full dashed UUID")
        self.assertEqual(proof["torch_uuid_bytes"], list(UUID(BARE).bytes))

    def test_bare_and_prefixed_object_spellings_bind_the_same_gpu(self):
        for text in (BARE, CANONICAL, BARE.upper(), "GPU-" + BARE.upper()):
            with self.subTest(text=text):
                proof, _, _, runner = observe(FixtureUUID(text, list(UUID(BARE).bytes)))
                self.assertEqual(proof["uuid"], CANONICAL)
                self.assertEqual(runner.call_args.args[0][2], CANONICAL)
                self.assertEqual(proof["torch_uuid_raw"], text)

    def test_only_logical_zero_properties_and_name_are_queried(self):
        _, torch, _, _ = observe()
        torch.cuda.get_device_properties.assert_called_once_with(0)
        torch.cuda.get_device_name.assert_called_once_with(0)
        torch.cuda.is_available.assert_called_once_with()
        torch.cuda.device_count.assert_called_once_with()
        torch.cuda.current_device.assert_called_once_with()
        torch.cuda.is_bf16_supported.assert_called_once_with()

    def test_unavailable_multiple_wrong_current_or_non_bf16_device_fails_before_query(self):
        for field, value in (("is_available", False), ("is_available", 1),
                             ("device_count", 0), ("device_count", 2), ("device_count", True), ("device_count", 1.0),
                             ("current_device", 1), ("current_device", False), ("current_device", 0.0),
                             ("is_bf16_supported", False), ("is_bf16_supported", 1)):
            with self.subTest(field=field, value=value):
                torch, _ = fake_torch()
                getattr(torch.cuda, field).return_value = value
                runner = Mock(return_value=completed())
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                runner.assert_not_called()

    def test_non_b200_or_empty_device_name_cannot_pass_identity(self):
        for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100", "", None):
            with self.subTest(name=name):
                torch, _ = fake_torch(name=name, property_name=name)
                runner = Mock(return_value=completed())
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                runner.assert_not_called()

    def test_properties_name_must_match_the_same_logical_device_name(self):
        for property_name in ("NVIDIA A100", "", None):
            with self.subTest(property_name=property_name):
                torch, _ = fake_torch(property_name=property_name)
                runner = Mock(return_value=completed())
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                runner.assert_not_called()

    def test_missing_or_invalid_property_uuid_never_falls_back_to_a_global_gpu_list(self):
        for raw in (ABSENT, None, "GPU-1234", "0", "MIG-" + BARE):
            with self.subTest(raw=repr(raw)):
                torch, properties = fake_torch(raw)
                if raw is ABSENT:
                    del properties.uuid
                runner = Mock(return_value=completed())
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                runner.assert_not_called()

    def test_uuid_byte_evidence_must_equal_the_full_string_identity(self):
        torch, _ = fake_torch(FixtureUUID(BARE, UUID(OTHER[4:]).bytes))
        runner = Mock(return_value=completed())
        with self.assertRaises(ValueError):
            module.assigned_gpu_identity(torch, runner=runner)
        runner.assert_not_called()

    def test_uuid_byte_evidence_requires_sixteen_plain_byte_values(self):
        valid = list(UUID(BARE).bytes)
        invalid = ([], valid[:-1], valid + [0], [True] + valid[1:], [-1] + valid[1:],
                   [256] + valid[1:], [0.0] + valid[1:], ["0"] + valid[1:], None, "0" * 16, 16)
        for values in invalid:
            with self.subTest(values=values):
                torch, _ = fake_torch(FixtureUUID(BARE, values))
                runner = Mock(return_value=completed())
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                runner.assert_not_called()

    def test_query_requires_exactly_one_nonempty_gpu_row(self):
        for stdout in ("", "\n \n", CANONICAL + ", " + DRIVER + "\n" + CANONICAL + ", " + DRIVER + "\n",
                       CANONICAL + ", " + DRIVER + "\n" + OTHER + ", " + DRIVER + "\n"):
            with self.subTest(stdout=stdout):
                torch, _ = fake_torch()
                runner = Mock(return_value=completed(stdout))
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                self.assertEqual(runner.call_count, 1)

    def test_other_physical_device_is_rejected_even_with_a_valid_driver(self):
        torch, _ = fake_torch()
        runner = Mock(return_value=completed(OTHER + ", " + DRIVER + "\n"))
        with self.assertRaises(ValueError):
            module.assigned_gpu_identity(torch, runner=runner)
        self.assertEqual(runner.call_count, 1)

    def test_query_uuid_must_be_full_prefixed_and_cannot_be_mig_or_abbreviated(self):
        for query_uuid in (BARE, "GPU-00112233", "MIG-" + BARE, "0", "GPU-00000000-0000-0000-0000-000000000000"):
            with self.subTest(query_uuid=query_uuid):
                torch, _ = fake_torch()
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=Mock(return_value=completed(query_uuid + ", " + DRIVER)))

    def test_driver_version_must_be_a_nonempty_dotted_numeric_version(self):
        for driver in ("", "unknown", "N/A", "580", "580..105", ".580.105", "580.105.",
                       "580.105beta", "580.105.08 extra", "580.105.08; command"):
            with self.subTest(driver=driver):
                torch, _ = fake_torch()
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=Mock(return_value=completed(CANONICAL + ", " + driver)))

    def test_query_fields_are_exact_and_outer_csv_whitespace_is_harmless(self):
        torch, _ = fake_torch()
        accepted = "\n  GPU-" + BARE.upper() + " , " + DRIVER + "  \n\n"
        proof = module.assigned_gpu_identity(torch, runner=Mock(return_value=completed(accepted)))
        self.assertEqual(proof["uuid"], CANONICAL)
        self.assertEqual(proof["query"]["stdout"], accepted)
        for stdout in (CANONICAL, CANONICAL + "," + DRIVER + ",unexpected", "," + DRIVER, CANONICAL + ","):
            with self.subTest(stdout=stdout):
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=Mock(return_value=completed(stdout)))

    def test_nonzero_completed_process_is_rejected_even_when_stdout_looks_valid(self):
        for response in (completed(stderr="fixture command failure", returncode=1),
                         completed(returncode=False), completed(returncode=0.0),
                         completed(stdout=None), completed(stdout=b"invalid response"),
                         completed(stderr=None), completed(stderr=b"invalid response")):
            with self.subTest(response=response):
                torch, _ = fake_torch()
                runner = Mock(return_value=response)
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(torch, runner=runner)
                self.assertEqual(runner.call_count, 1)

    def test_timeout_command_failure_or_missing_binary_is_not_retried(self):
        errors = (subprocess.TimeoutExpired(["nvidia-smi"], 15),
                  subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="fixture failure"),
                  FileNotFoundError("fixture nvidia-smi missing"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                torch, _ = fake_torch()
                runner = Mock(side_effect=error)
                with self.assertRaises(type(error)) as raised:
                    module.assigned_gpu_identity(torch, runner=runner)
                self.assertIs(raised.exception, error)
                self.assertEqual(runner.call_count, 1)

    def test_evidence_is_json_safe_detached_and_module_import_has_no_framework_work(self):
        raw_values = list(UUID(BARE).bytes)
        raw = FixtureUUID(BARE, raw_values)
        proof, _, _, _ = observe(raw)
        expected = deepcopy(proof)
        raw_values[0] = 255
        self.assertEqual(proof, expected)
        self.assertEqual(json.loads(json.dumps(proof)), proof)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "vllm", "numpy"}:
                raise AssertionError("GPU identity helper imported a model framework")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded_import):
            importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
