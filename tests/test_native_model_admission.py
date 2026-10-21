"""Offline numerical-admission oracles; run later in native CPU QA only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import native_model_admission as module
from extension import preserving_data as data
from reasoning.data import prompt_id


PIN = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
GIB = 1 << 30


class FixtureTokenizer:
    """Synthetic exact-rendering witness, never an actual Qwen tokenizer."""
    eos_token_id = 151645
    chat_template = "synthetic admission fixture template"
    tail = "<FIXTURE_EOS>\n"

    def encode(self, text, add_special_tokens=False):
        if text.endswith(self.tail):
            return [1 + ord(value) % 100 for value in text[:-len(self.tail)]] + [151645, 198]
        return [1 + ord(value) % 100 for value in text]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, add_generation_prompt=False):
        text = "SYSTEM:" + messages[0]["content"] + "\nUSER:" + messages[1]["content"] + "\nASSISTANT:"
        if len(messages) == 3:
            text += messages[2]["content"] + self.tail
        return self.encode(text) if tokenize else text


def make_plan():
    return {
        "schema": "extension-native-model-admission-plan-v1", "diagnostic_only": True,
        "automatic_downstream": False, "training_pilot": False, "base_revision": PIN,
        "objectives": list(module.MODES), "case_order": list(module.CASES),
        "deadline_utc": "2026-10-28T03:59:00+00:00",
        "resources": {"wall_time_seconds": 7200, "host_cpus": 8, "host_ram_gib": 128,
                      "gpus": 1, "gpu_type": "B200",
                      "scratch_free_gib": 160, "reserve_seconds": 172800},
        "settings": {"student_dtype": "float32", "reference_dtype": "float32", "bf16": True,
            "attention": "sdpa", "gradient_checkpointing": True, "checkpoint_use_reentrant": False,
            "chunk_tokens": 128, "checkpoint_chunks": True, "accumulation": 2,
            "learning_rate": .000002, "weight_decay": .01, "beta": .05,
            "clip_norm": 1., "seed": 20261021, "optimizer": "adamw_torch", "scheduler": "constant",
            "resume": False, "dropout_from_original_config_must_be_zero": True},
        "tolerances": {"logit_max_abs": .001, "ce_abs": .00001, "kl_abs": .00001,
            "reload_logit_max_abs": .001, "headroom_min_bytes": 16 * GIB, "headroom_fraction": .1},
    }


def make_prepared(plan, plan_sha256):
    tokenizer = FixtureTokenizer()
    identity = {"chat_template_sha256": data.text_sha256(tokenizer.chat_template),
                "eos_token_id": 151645, "files": dict(module.TOKENIZER_FILES)}
    rows = [{"id": prompt_id("Fixture question " + str(index)), "instruction": "Fixture question " + str(index),
             "input": "", "output": "a" * size, "system": "Fixture system"}
            for index, size in enumerate((3, 7, 11, 600))]
    components = {purpose: {"scope": scope, "pool_ids": [], "audit_receipt_sha256": "1" * 64,
        "criterion_sha256": "2" * 64, "question_inventory_sha256": "3" * 64}
        for purpose, scope in (("reserved_source_evaluation", "all_reserved_source_splits"),
                               ("final_benchmark_questions", "all_nine_final_benchmarks"))}
    exclusions = {"schema": data.EXCLUSION_SCHEMA, "id_namespace": data.POOL_ID_NAMESPACE,
                  "population_ids_sha256": data.value_sha256(sorted(row["id"] for row in rows)),
                  "components": components}
    source_identity = {"pool_sha256": "4" * 64, "features_sha256": "4" * 64}
    kwargs = dict(excluded_pool_ids=set(), exclusion_receipt=exclusions, source_identity=source_identity,
        tokenizer_identity=identity, context_role="diagnostic", eos_token_id=151645,
        terminal_suffix=(151645, 198), terminal_text=tokenizer.tail, vocab_size=151936)
    short = data.serialize_selected(rows, [row["id"] for row in rows[:3]], tokenizer, cutoff=512, **kwargs)
    long = data.serialize_selected(rows, [rows[3]["id"]], tokenizer, cutoff=32768, **kwargs)
    examples = short["examples"] + long["examples"]
    audit = {name: {"path": "extension/fixture/" + name + ".json", "sha256": "4" * 64}
        for name in ("preparation", "development_seal", "reserved_pool_ids", "benchmark_audit",
                     "benchmark_pool_ids", "exclusion_receipt", "pool", "features")}
    return {"schema": "extension-native-model-smoke-cache-v1", "complete": True, "plan_sha256": plan_sha256,
        "base": {"path": "/fixture/snapshots/" + PIN, "repository": "Qwen/Qwen2.5-3B-Instruct",
            "revision": PIN, "files": dict(module.ORIGINAL_FILES),
            "identity_receipt": {"path": "deadline/artifacts/model_identity.json",
                "sha256": "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2"}},
        "serialization": {"eos_token_id": 151645, "terminal_suffix": [151645, 198],
            "terminal_text": tokenizer.tail, "tokenizer_identity": identity},
        "rows": {"short": examples[:3], "long": examples[3]},
        "presentation_schedule": {"short": [row["id"] for row in examples[:3]],
                                  "long": [examples[3]["id"], examples[3]["id"]]},
        "source_audit": audit, "source_identity": source_identity}


class NativeAdmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.plan_sha = data.value_sha256(self.plan)
        self.prepared = make_prepared(self.plan, self.plan_sha)

    def test_valid_plan_and_complete_private_witness_structure(self):
        self.assertEqual(module.validate_plan(self.plan), self.plan)
        self.assertEqual(module.validate_prepared(self.prepared, self.plan, self.plan_sha), self.prepared)

    def test_exact_four_modes_and_eight_case_order_are_fixed(self):
        expected = tuple(mode + "-" + kind for mode in ("sft", "ce_kl", "asft", "prior_anchored")
                         for kind in ("short", "long"))
        self.assertEqual(module.CASES, expected)
        for key in ("case_order", "objectives"):
            value = copy.deepcopy(self.plan)
            value[key] = list(reversed(value[key]))
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_case_schedules_preserve_three_distinct_short_and_declared_long_repeat(self):
        schedules = module.case_schedule(self.prepared)
        self.assertEqual(tuple(schedules), module.CASES)
        for case in module.CASES:
            kind = case.rsplit("-", 1)[1]
            self.assertEqual(schedules[case], self.prepared["presentation_schedule"][kind])
        self.assertEqual(len(set(schedules["sft-short"])), 3)
        self.assertEqual(len(schedules["sft-long"]), 2)
        self.assertEqual(len(set(schedules["sft-long"])), 1)

    def test_unknown_scope_wrong_base_or_unsealed_settings_rejected(self):
        changes = [lambda value: value.update(diagnostic_only=False),
            lambda value: value.update(automatic_downstream=True), lambda value: value.update(training_pilot=True),
            lambda value: value.update(base_revision="main"),
            lambda value: value["settings"].update(reference_dtype="bfloat16"),
            lambda value: value["settings"].update(checkpoint_use_reentrant=True),
            lambda value: value["settings"].update(chunk_tokens=0),
            lambda value: value["settings"].update(accumulation=1)]
        for change in changes:
            value = copy.deepcopy(self.plan)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_nonfinite_or_missing_tolerances_and_unbounded_resources_rejected(self):
        for key in self.plan["tolerances"]:
            value = copy.deepcopy(self.plan)
            value["tolerances"][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_plan(value)
        for change in (lambda value: value["tolerances"].update(ce_abs=float("nan")),
                       lambda value: value["resources"].update(wall_time_seconds=0),
                       lambda value: value["resources"].update(reserve_seconds=0)):
            value = copy.deepcopy(self.plan)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_prepared_plan_base_files_and_input_bindings_rejected_when_changed(self):
        changes = [lambda value: value.update(plan_sha256="0" * 64),
            lambda value: value["base"].update(revision="main"),
            lambda value: value["base"]["files"].update({next(iter(module.ORIGINAL_FILES)): "0" * 64}),
            lambda value: value["serialization"]["tokenizer_identity"]["files"].update({next(iter(module.TOKENIZER_FILES)): "0" * 64}),
            lambda value: value["source_audit"].pop("development_seal"),
            lambda value: value["source_audit"]["benchmark_audit"].update(sha256="invalid")]
        for change in changes:
            value = copy.deepcopy(self.prepared)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_prepared(value, self.plan, self.plan_sha)

    def test_duplicate_ids_missing_originals_or_unequal_short_length_contract_rejected(self):
        changes = [lambda value: value["rows"]["short"].__setitem__(1, copy.deepcopy(value["rows"]["short"][0])),
            lambda value: value["rows"]["short"][0].pop("original_row"),
            lambda value: value["rows"]["short"][0]["original_row"].update(instruction="changed question"),
            lambda value: value["presentation_schedule"]["long"].pop(),
            lambda value: value["presentation_schedule"].update(short=list(reversed(value["presentation_schedule"]["short"])))]
        for change in changes:
            value = copy.deepcopy(self.prepared)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_prepared(value, self.plan, self.plan_sha)

    def test_cache_token_hash_complete_eos_and_masks_are_validated(self):
        changes = [lambda row: row["input_ids"].__setitem__(-1, 199),
            lambda row: row["labels"].__setitem__(-2, -100),
            lambda row: row["labels"].__setitem__(0, row["input_ids"][0]),
            lambda row: row["attention_mask"].__setitem__(-1, 0),
            lambda row: row["provenance"].update(full_token_ids_sha256="0" * 64),
            lambda row: row["provenance"].update(truncated=True),
            lambda row: row["provenance"].update(target_rewritten=True)]
        for change in changes:
            value = copy.deepcopy(self.prepared)
            change(value["rows"]["short"][0])
            with self.assertRaises(ValueError):
                module.validate_prepared(value, self.plan, self.plan_sha)

    def test_only_a_new_exclusive_output_directory_is_permitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            self.assertEqual(module.reserve_output(output), output.resolve())
            self.assertTrue(output.is_dir())
            for reuse in (output, output):
                with self.assertRaises((ValueError, FileExistsError)):
                    module.reserve_output(reuse)


class NativeAdmissionGateMathTests(unittest.TestCase):
    def test_gpu_identity_queries_only_the_cuda_assigned_uuid_and_rejects_other_devices(self):
        uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        cuda = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(uuid=uuid),
                               get_device_name=lambda index: "NVIDIA B200")
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(stdout=uuid + ", 580.105.08\n")) as run:
            proof = module.assigned_gpu_identity(SimpleNamespace(cuda=cuda))
        self.assertEqual(proof["uuid"], uuid)
        self.assertEqual(run.call_args.args[0], ["nvidia-smi", "--id=" + uuid,
            "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"])
        for stdout in ("GPU-00000000, 580.105.08\n", uuid + ", unknown\n", uuid + ", 580.105.08\n" + uuid + ", 580.105.08\n"):
            with patch.object(module.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
                with self.assertRaises(ValueError):
                    module.assigned_gpu_identity(SimpleNamespace(cuda=cuda))

    def test_memory_trace_preserves_original_oom_and_unwinds_unclosed_optimizer_phase(self):
        class FakeCuda:
            free = 90
            synchronize = staticmethod(lambda: None)
            reset_peak_memory_stats = staticmethod(lambda: None)
            memory_allocated = staticmethod(lambda: 4)
            memory_reserved = staticmethod(lambda: 8)
            max_memory_allocated = staticmethod(lambda: 5)
            max_memory_reserved = staticmethod(lambda: 9)

            def mem_get_info(self):
                return self.free, 100

        cuda = FakeCuda()
        trace = module.MemoryTrace(SimpleNamespace(cuda=cuda))
        with self.assertRaisesRegex(RuntimeError, "original OOM"):
            with trace.phase("outer"):
                with trace.phase("training"):
                    trace.begin("optimizer_unclosed")
                    cuda.free = 80
                    raise RuntimeError("original OOM")
        self.assertEqual(trace.stack, [])
        self.assertEqual([frame["stage"] for frame in trace.stages],
                         ["optimizer_unclosed", "training", "outer"])
        self.assertTrue(all(frame["aborted"] and frame["complete"] is False for frame in trace.stages))
        self.assertEqual(trace.max_observed_nonallocator_bytes, 12)
        result = trace.result({"headroom_min_bytes": 1, "headroom_fraction": .1})
        self.assertEqual(result["peak_reserved_bytes"], 21)
        self.assertEqual(result["baseline_driver_bytes"], 2)

    def test_headroom_uses_larger_of_fixed_bytes_and_device_fraction(self):
        fixed = module.headroom_report(80 * GIB, 96 * GIB, 16 * GIB, .1)
        self.assertTrue(fixed["passed"])
        fraction = module.headroom_report(180 * GIB, 200 * GIB, 16 * GIB, .1)
        self.assertTrue(fraction["passed"])
        self.assertFalse(module.headroom_report(180 * GIB + 1, 200 * GIB, 16 * GIB, .1)["passed"])
        self.assertFalse(module.headroom_report(80 * GIB + 1, 96 * GIB, 16 * GIB, .1)["passed"])

    def test_invalid_headroom_inputs_fail_closed(self):
        for arguments in ((-1, 96 * GIB, 16 * GIB, .1), (97 * GIB, 96 * GIB, 16 * GIB, .1),
                          (1, 0, 1, .1), (1, 10, -1, .1), (1, 10, 1, float("nan"))):
            with self.assertRaises(ValueError):
                module.headroom_report(*arguments)

    def test_parity_exact_boundary_and_each_independent_gate(self):
        limits = make_plan()["tolerances"]
        metrics = {"max_abs_logits": limits["logit_max_abs"], "ce_abs": limits["ce_abs"], "kl_abs": limits["kl_abs"]}
        self.assertTrue(module.check_parity(metrics, limits)["passed"])
        for name in metrics:
            value = dict(metrics)
            value[name] *= 2
            with self.assertRaises(ValueError):
                module.check_parity(value, limits)

    def test_parity_nonfinite_missing_or_negative_measurements_rejected(self):
        baseline = {"max_abs_logits": .0001, "ce_abs": .000001, "kl_abs": .000001}
        for name in baseline:
            for invalid in (float("nan"), float("inf"), -1.):
                value = dict(baseline)
                value[name] = invalid
                with self.assertRaises(ValueError):
                    module.check_parity(value, make_plan()["tolerances"])
        with self.assertRaises(ValueError):
            module.check_parity({}, make_plan()["tolerances"])

    def test_native_single_gpu_allocation_identity_and_typed_tres(self):
        text = ("JobId=42 JobState=RUNNING UserId=fixture(1000) NodeList=owl-sr001 NumNodes=1 "
                "AllocTRES=cpu=8,mem=128G,node=1,gres/gpu=1,gres/gpu:b200=1")
        proof = module.validate_gpu_allocation(text, "42", "owl-sr001", 1000)
        self.assertIsInstance(proof, dict)
        for before, after in (("JobState=RUNNING", "JobState=PENDING"), ("JobId=42", "JobId=43"),
                              ("fixture(1000)", "fixture(1001)"), ("NodeList=owl-sr001", "NodeList=owl-sr002"),
                              ("gres/gpu=1", "gres/gpu=2"), ("NumNodes=1", "NumNodes=2")):
            with self.assertRaises(ValueError):
                module.validate_gpu_allocation(text.replace(before, after), "42", "owl-sr001", 1000)

    def test_compute_hostname_owl123_is_valid_but_exact_login_hosts_are_rejected(self):
        def allocation(node):
            return ("JobId=42 JobState=RUNNING UserId=fixture(1000) NodeList=" + node +
                " NumNodes=1 AllocTRES=cpu=8,mem=128G,node=1,gres/gpu=1,gres/gpu:b200=1")
        self.assertEqual(module.validate_gpu_allocation(allocation("owl123"), "42", "owl123", 1000)["node"], "owl123")
        for node in ("owl1", "owl1.arc.vt.edu", "tc2", "tc2.arc.vt.edu", "owl-login02"):
            with self.subTest(node=node), self.assertRaises(ValueError):
                module.validate_gpu_allocation(allocation(node), "42", node, 1000)

    def test_cpu_only_or_wrong_gpu_type_cannot_admit_real_model(self):
        for tres in ("cpu=8,mem=128G,node=1", "cpu=8,node=1,gres/gpu=1,gres/gpu:a100=1"):
            text = "JobId=42 JobState=RUNNING UserId=fixture(1000) NodeList=owl-sr001 NumNodes=1 AllocTRES=" + tres
            with self.assertRaises(ValueError):
                module.validate_gpu_allocation(text, "42", "owl-sr001", 1000)


class NativeAdmissionInputFilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.base = self.root / "fixture-base"
        self.base.mkdir()
        self.originals = {}
        for name in module.ORIGINAL_FILES:
            content = ("synthetic small byte fixture for " + name).encode()
            (self.base / name).write_bytes(content)
            self.originals[name] = hashlib.sha256(content).hexdigest()
        self.tokenizers = {name: self.originals[name] for name in module.TOKENIZER_FILES}
        for name, value in (("ORIGINAL_FILES", self.originals), ("TOKENIZER_FILES", self.tokenizers)):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {"SLURM_JOB_ID": "42"})
        environment.start()
        self.addCleanup(environment.stop)
        self.plan = make_plan()
        self.plan_path = "extension/fixture/plan.json"
        self.prepared_path = "extension/fixture/prepared.json"
        self.qa_path = "extension/fixture/qa.json"
        self.admission_path = "extension/fixture/admission.json"

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(raw)
        return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    def fixture(self, *, overlap=False, seal_complete=True, preparation_complete=True):
        plan = self.write(self.plan_path, self.plan)
        prepared = make_prepared(self.plan, plan["sha256"])
        prepared["base"]["path"] = str(self.base)
        identity = {"path": str(self.base), "model_type": "qwen2",
                    "files": {name: {"sha256": checksum} for name, checksum in self.originals.items()}}
        identity["identity_sha256"] = data.value_sha256(identity)
        prepared["base"]["identity_receipt"] = self.write("extension/fixture/identity.json", identity)
        pool = self.write("extension/fixture/pool.json", [])
        features = self.write("extension/fixture/features.json", {})
        prepared["source_identity"] = {"pool_sha256": pool["sha256"], "features_sha256": features["sha256"]}
        for row in prepared["rows"]["short"] + [prepared["rows"]["long"]]:
            row["provenance"]["source_identity_sha256"] = data.value_sha256(prepared["source_identity"])
        reserved_ids = [prepared["rows"]["short"][0]["id"]] if overlap else []
        reserved = self.write("extension/fixture/reserved.json", reserved_ids)
        blocked = self.write("extension/fixture/blocked.json", [])
        seal = self.write("extension/fixture/seal.json", {
            "schema": "extension-development-seal-v2", "complete": seal_complete,
            "input_stats_unchanged_at_closeout": True,
            "output_sha256": {"reserved_training_pool_ids.json": reserved["sha256"]},
            "reservation": {"reserved_training_pool_ids_sha256": data.value_sha256(reserved_ids)}})
        question_hashes = {}
        for name in ("aime", "aime25", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"):
            question_hashes[name] = self.write("deadline/artifacts/benchmarks/" + name + "/test.jsonl",
                {"question": "Synthetic question " + name})["sha256"]
        benchmark = self.write("extension/fixture/benchmark.json", {
            "schema": "extension-final-benchmark-pool-exclusions-v1", "complete": True,
            "pool_sha256": pool["sha256"], "criterion": module.CRITERION,
            "scope": "all_nine_final_benchmarks", "answers_or_outcomes_used": False,
            "benchmark_question_sha256": question_hashes,
            "excluded_pool_ids_sha256": data.value_sha256([]), "excluded_pool_ids_file_sha256": blocked["sha256"]})
        components = {}
        for purpose, scope, ids, receipt in (
            ("reserved_source_evaluation", "all_reserved_source_splits", reserved_ids, seal),
            ("final_benchmark_questions", "all_nine_final_benchmarks", [], benchmark)):
            components[purpose] = {"scope": scope, "pool_ids": ids, "audit_receipt_sha256": receipt["sha256"],
                "criterion_sha256": data.value_sha256(module.CRITERION), "question_inventory_sha256": "8" * 64}
        exclusion = self.write("extension/fixture/exclusion.json", {"schema": data.EXCLUSION_SCHEMA,
            "id_namespace": data.POOL_ID_NAMESPACE,
            "population_ids_sha256": data.value_sha256(sorted(row["id"] for row in
                prepared["rows"]["short"] + [prepared["rows"]["long"]])), "components": components})
        for row in prepared["rows"]["short"] + [prepared["rows"]["long"]]:
            row["provenance"]["exclusion_receipt_sha256"] = exclusion["sha256"]
        code = {}
        for relative in module.REQUIRED_CODE | {"extension/smoke_preparation.py", "tests/test_smoke_preparation.py"}:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic bounded source fixture")
            code[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        unit = self.write("extension/fixture/unit_qa.json", {
            "schema": "extension-native-smoke-preparation-unit-qa-v1", "success": True,
            "tests_run": 1, "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0,
            "unexpected_successes": 0, "gpu_disabled": True, "native_plan_sha256": plan["sha256"],
            "code_sha256": code, "job_id": "41", "node": "owl-cpu001",
            "compute_proof": {"job_id": "41", "node": "owl-cpu001", "gpu_requested_or_allocated": False}})
        long = prepared["rows"]["long"]
        preparation = self.write("extension/fixture/preparation.json", {
            "schema": "extension-native-smoke-preparation-v1", "complete": preparation_complete,
            "cpu_only": True, "gpu_seconds": 0, "plan_sha256": plan["sha256"],
            "rows_sha256": data.value_sha256(prepared["rows"]), "exclusion_receipt_sha256": exclusion["sha256"],
            "pool_sha256": pool["sha256"], "features_sha256": features["sha256"],
            "source_identity": prepared["source_identity"],
            "native_qa": unit, "code_sha256": code,
            "input_files": {str(self.root / pool["path"]): {"sha256": pool["sha256"],
                "bytes": (self.root / pool["path"]).stat().st_size}},
            "no_model_or_benchmark_outcomes": True, "input_snapshot_unchanged": True,
            "longest": {"id": long["id"], "total_tokens": len(long["input_ids"]),
                "max_eligible_total_tokens": len(long["input_ids"]),
                "after_reserved_exclusion": True, "after_benchmark_exclusion": True}})
        prepared["source_audit"] = dict(preparation=preparation, development_seal=seal,
            reserved_pool_ids=reserved, benchmark_audit=benchmark, benchmark_pool_ids=blocked,
            exclusion_receipt=exclusion, pool=pool, features=features)
        cache = self.write(self.prepared_path, prepared)
        qa = self.write(self.qa_path, {"schema": "extension-native-model-admission-native-qa-v1", "success": True,
            "tests_run": 1, "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0,
            "unexpected_successes": 0, "gpu_disabled": True, "plan_sha256": plan["sha256"],
            "prepared_sha256": cache["sha256"], "code_sha256": code})
        self.write(self.admission_path, {"schema": "extension-native-model-admission-job-v1", "status": "admitted",
            "job_id": "42", "plan_sha256": plan["sha256"], "prepared_sha256": cache["sha256"],
            "qa_sha256": qa["sha256"], "code_sha256": code, "automatic_downstream": False})
        return prepared

    def load(self):
        return module.load_inputs(self.root, self.plan_path, self.prepared_path, self.qa_path, self.admission_path)

    def test_actual_small_file_hash_chain_validates_all_input_roles(self):
        prepared = self.fixture()
        context = self.load()
        self.assertEqual(context["prepared"], prepared)
        self.assertEqual(context["admission"]["job_id"], "42")
        self.assertEqual(context["prepared_sha256"], hashlib.sha256((self.root / self.prepared_path).read_bytes()).hexdigest())

    def test_raw_bytes_tampering_in_model_input_or_seal_is_detected(self):
        self.fixture()
        for path in (self.base / "config.json", self.root / "extension/fixture/pool.json",
                     self.root / "extension/fixture/seal.json"):
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(before)

    def test_semantically_unfinished_or_overlapping_sealed_inputs_are_rejected(self):
        for options in ({"overlap": True}, {"seal_complete": False}, {"preparation_complete": False}):
            self.fixture(**options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load()

    def test_wrong_current_job_cannot_reuse_a_valid_admission_receipt(self):
        self.fixture()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "43"}):
            with self.assertRaises(ValueError):
                self.load()


class FixtureEvidence:
    def __init__(self):
        self.files, self.calls, self.fail_after = {}, 0, None

    def unchanged(self):
        self.calls += 1
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ValueError("fixture input changed")


class NativeAdmissionRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "extension/runs/unique-attempt"
        self.plan = make_plan()
        self.plan_sha = data.value_sha256(self.plan)
        self.prepared = make_prepared(self.plan, self.plan_sha)
        self.evidence = FixtureEvidence()
        self.context = dict(root=self.root, plan=self.plan, prepared=self.prepared,
            plan_path=self.root / "extension/plan.json", prepared_path=self.root / "extension/cache.json",
            qa_path=self.root / "extension/qa.json", admission_path=self.root / "extension/admission.json",
            plan_sha256=self.plan_sha, prepared_sha256=data.value_sha256(self.prepared),
            code_sha256={name: "5" * 64 for name in module.REQUIRED_CODE},
            binding_sha256="6" * 64, qa={"success": True}, admission={"job_id": "42"}, evidence=self.evidence)
        for name, value in (("require_gpu_allocation", lambda: {"job_id": "42", "node": "owl-sr001"}),
                            ("load_inputs", lambda *args, **kwargs: self.context),
                            ("deadline_guard", lambda *args, **kwargs: None)):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        disk = patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 * GIB))
        disk.start()
        self.addCleanup(disk.stop)
        self.calls = []

    def case_receipt(self, case):
        directory = self.output / "cases" / case
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir(parents=True)
        files = {"checkpoint/config.json": b'{"fixture":true}',
                 "checkpoint/model.safetensors": b"synthetic fixture bytes, never a real model"}
        for name, content in files.items():
            (directory / name).write_bytes(content)
        mode, kind = case.rsplit("-", 1)
        memory = {"peak_allocated_bytes": 80 * GIB, "peak_reserved_bytes": 82 * GIB, "total_bytes": 192 * GIB}
        memory["headroom"] = module.headroom_report(82 * GIB, 192 * GIB, 16 * GIB, .1)
        return {"schema": "extension-native-model-admission-case-v1", "complete": True,
            "case": case, "mode": mode, "kind": kind, "binding_sha256": self.context["binding_sha256"],
            "plan_sha256": self.context["plan_sha256"], "prepared_sha256": self.context["prepared_sha256"],
            "code_sha256": self.context["code_sha256"], "job_id": "42",
            "presentation_ids": module.case_schedule(self.prepared)[case],
            "optimizer_updates": 2 if kind == "short" else 1,
            "checks": {name: True for name in module.REQUIRED_CHECKS}, "memory": memory,
            "output_files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}

    def successful_child(self, command, **kwargs):
        self.calls.append((command, kwargs))
        case = command[command.index("--case") + 1]
        receipt = self.case_receipt(case)
        path = self.output / "cases" / case / "complete.json"
        path.write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    def run_admission(self, child=None, output=None):
        return module.run(self.root, "extension/plan.json", "extension/cache.json", output or self.output,
            "extension/qa.json", "extension/admission.json", runner=child or self.successful_child)

    def test_all_eight_cases_run_in_fresh_subprocess_commands_exactly_once(self):
        receipt = self.run_admission()
        self.assertTrue(receipt["complete"])
        self.assertFalse(receipt["automatic_downstream"])
        self.assertFalse(receipt["training_pilot"])
        self.assertFalse(receipt["resume_supported"])
        self.assertEqual(tuple(receipt["case_receipts"]), module.CASES)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual([command[command.index("--case") + 1] for command, _ in self.calls], list(module.CASES))
        self.assertTrue(all(command[1:3] == ["-m", "extension.native_model_admission"] for command, _ in self.calls))
        self.assertTrue(all(0 < kwargs["timeout"] <= 7200 and kwargs["check"] is False for _, kwargs in self.calls))
        self.assertEqual(self.evidence.calls, 9)

    def test_first_failed_child_stops_without_retry_or_complete(self):
        def child(command, **kwargs):
            if self.calls:
                self.calls.append((command, kwargs))
                return SimpleNamespace(returncode=1)
            return self.successful_child(command, **kwargs)
        with self.assertRaises(ValueError):
            self.run_admission(child)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / "complete.json").exists())
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(tuple(failure["completed_cases"]), (module.CASES[0],))
        self.assertFalse(failure["retry"])
        self.assertFalse(failure["automatic_downstream"])

    def test_zero_child_exit_without_actual_complete_receipt_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.run_admission(lambda *args, **kwargs: SimpleNamespace(returncode=0))
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "complete.json").exists())

    def test_child_timeout_is_preserved_without_second_call(self):
        calls = []
        def child(command, **kwargs):
            calls.append(command)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_admission(child)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.output / "complete.json").exists())
        self.assertEqual(json.loads((self.output / "failure.json").read_text())["exception"], "TimeoutExpired")

    def test_input_mutation_cannot_publish_final_complete(self):
        self.evidence.fail_after = 1
        with self.assertRaisesRegex(ValueError, "fixture input changed"):
            self.run_admission()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "complete.json").exists())

    def test_existing_attempt_is_not_reused_or_resumed(self):
        self.run_admission()
        before = (self.output / "complete.json").read_bytes()
        with self.assertRaises(ValueError):
            self.run_admission()
        self.assertEqual(len(self.calls), 8)
        self.assertEqual((self.output / "complete.json").read_bytes(), before)

    def test_outputs_outside_runtime_root_and_insufficient_disk_block_children(self):
        with self.assertRaises(ValueError):
            self.run_admission(output=self.root / "unsafe")
        self.assertFalse((self.root / "unsafe").exists())
        with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaises(ValueError):
                self.run_admission()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.output / "complete.json").exists())

    def test_case_flags_schedule_updates_and_binding_are_not_trusted_blindly(self):
        case = module.CASES[0]
        receipt = self.case_receipt(case)
        path = self.output / "cases" / case
        self.assertEqual(module.validate_case_receipt(receipt, self.context, case, path), receipt)
        changes = [lambda value: value.update(complete=False), lambda value: value.update(optimizer_updates=1),
            lambda value: value.update(binding_sha256="0" * 64),
            lambda value: value.update(presentation_ids=list(reversed(value["presentation_ids"]))),
            lambda value: value["checks"].update(reference_unchanged=False),
            lambda value: value["checks"].pop("checkpoint_isolation")]
        for change in changes:
            value = copy.deepcopy(receipt)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_case_receipt(value, self.context, case, path)

    def test_case_headroom_uses_reserved_peak_and_rehashes_saved_bytes(self):
        case = module.CASES[0]
        receipt = self.case_receipt(case)
        path = self.output / "cases" / case
        value = copy.deepcopy(receipt)
        value["memory"]["headroom"] = module.headroom_report(80 * GIB, 192 * GIB, 16 * GIB, .1)
        with self.assertRaises(ValueError):
            module.validate_case_receipt(value, self.context, case, path)
        (path / "checkpoint/model.safetensors").write_bytes(b"changed after child verification")
        with self.assertRaises(ValueError):
            module.validate_case_receipt(receipt, self.context, case, path)

    def test_case_inventory_cannot_escape_its_directory(self):
        case = module.CASES[0]
        receipt = self.case_receipt(case)
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"outside")
        receipt["output_files"][str(outside)] = hashlib.sha256(outside.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            module.validate_case_receipt(receipt, self.context, case, self.output / "cases" / case)


class NativeAdmissionTinyTensorTests(unittest.TestCase):
    def test_student_update_keeps_reference_outside_optimizer_and_unchanged(self):
        import torch
        from extension.preserving_trainer import check_optimizer, check_reference
        student = torch.nn.Linear(2, 2, dtype=torch.float32)
        with torch.no_grad():
            student.weight.fill_(.25)
            student.bias.zero_()
        reference = copy.deepcopy(student).eval().requires_grad_(False)
        before = module.tensor_inventory(reference)
        optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
        loss = student(torch.tensor([[1., 2.]])).square().mean()
        loss.backward()
        self.assertEqual(module.assert_finite_model(student, gradients=True), 2)
        check_optimizer(student, optimizer, reference)
        optimizer.step()
        self.assertEqual(module.assert_finite_model(student), 2)
        check_optimizer(student, optimizer, reference)
        check_reference(student, reference)
        self.assertEqual(module.tensor_inventory(reference), before)
        self.assertNotEqual(module.tensor_inventory(student), before)
        wrong = torch.optim.AdamW(list(student.parameters()) + list(reference.parameters()))
        with self.assertRaises(ValueError):
            check_optimizer(student, wrong, reference)

    def test_shared_student_reference_storage_and_nonfinite_parameters_fail(self):
        import torch
        from extension.preserving_trainer import disjoint_models
        student = torch.nn.Linear(2, 2, dtype=torch.float32)
        reference = copy.deepcopy(student)
        reference.weight = torch.nn.Parameter(student.weight.detach())
        with self.assertRaises(ValueError):
            disjoint_models(student, reference)
        with torch.no_grad():
            student.weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            module.assert_finite_model(student)
        with self.assertRaises(ValueError):
            module.assert_finite_model(student, gradients=True)

    def test_saved_checkpoint_must_equal_post_update_student_and_exclude_reference(self):
        import torch
        from safetensors.torch import save_file
        student = torch.nn.Linear(2, 2, dtype=torch.float32)
        expected = module.tensor_inventory(student)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            file = directory / "model.safetensors"
            state = {name: value.detach().clone() for name, value in student.state_dict().items()}
            save_file(state, str(file))
            receipt = module._checkpoint_inventory(directory, expected)
            self.assertEqual(receipt["tensors"], 2)
            self.assertTrue(receipt["all_finite_fp32"])
            state["weight"] = state["weight"] + 1
            save_file(state, str(file))
            with self.assertRaises(ValueError):
                module._checkpoint_inventory(directory, expected)
            state = {name: value.detach().clone() for name, value in student.state_dict().items()}
            state["reference.weight"] = state["weight"].clone()
            save_file(state, str(file))
            with self.assertRaises(ValueError):
                module._checkpoint_inventory(directory, expected)

    def test_inventory_and_checkpoint_reject_non_fp32_storage(self):
        import torch
        from safetensors.torch import save_file
        student = torch.nn.Linear(2, 2, dtype=torch.float32)
        expected = module.tensor_inventory(student)
        with self.assertRaises(ValueError):
            module.tensor_inventory(copy.deepcopy(student).to(torch.bfloat16))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_file({name: value.detach().to(torch.bfloat16) for name, value in student.state_dict().items()},
                      str(directory / "model.safetensors"))
            with self.assertRaises(ValueError):
                module._checkpoint_inventory(directory, expected)


if __name__ == "__main__":
    unittest.main()
