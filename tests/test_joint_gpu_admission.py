"""Offline metadata fixtures only; execute later on the admitted CPU allocation.

No model/cache/dataset reads, scheduler calls or network requests. Synthetic
79-file receipts exercise this adapter's graph; the already-tested scientific
inventory builder and prior-nine queue validator are explicit fixture seams.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import joint_gpu_admission as admission


NOW = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path):
    raw = path.read_bytes()
    return {"sha256": admission._sha(raw), "bytes": len(raw)}


def cpu_proof(job="700001", node="owl123"):
    raw = (f"JobId={job} JobState=RUNNING UserId=fixture({os.getuid()}) NumNodes=1 NodeList={node} "
        "Partition=normal_q NumCPUs=8 AllocTRES=cpu=8,mem=64G,node=1,billing=20 "
        "ReqTRES=cpu=8,mem=64G,node=1,billing=20\n")
    return admission.validate_allocation(raw, job, node, os.getuid())


def accounting(job="700001", name="sft-extension-joint-gpu-metadata-qa", elapsed=5):
    rows = []
    for suffix, row_name in (("", name), (".batch", "batch"), (".extern", "extern")):
        rows.append(f"{job}{suffix}|{row_name}|COMPLETED|0:0|{elapsed}|2026-10-21T01:00:00|"
            f"2026-10-21T01:00:{elapsed:02d}|billing=20,cpu=8,mem=64G,node=1")
    return {"schema": "extension-terminal-cpu-accounting-v1", "complete": True, "job_id": job,
        "elapsed_seconds": elapsed, "gpu_seconds": 0, "no_gpu_in_alloc_tres": True,
        "timezone": "UTC", "command": ["sacct", "--clusters=owl", "-j", job, "--parsable2", "--noheader",
            "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,AllocTRES"],
        "stdout": "\n".join(rows) + "\n", "stderr": ""}


def passed(schema, sources, job, tests):
    proof = cpu_proof(job)
    return {"schema": schema, "success": True, "job_id": job, "node": proof["node"], "compute_proof": proof,
        "gpu_disabled": True, "wrapper_syntax_passed": True, "tests_run": tests,
        "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
        "source_inventory": sources, "code_sha256": {name: value["sha256"] for name, value in sources.items()},
        "automatic_downstream": False}


def entry(role, job, sources=None, **extra):
    binding = {"role": role, "automatic_downstream": False, "source_sha256": sources or {}, **extra}
    return {"status": "submitted", "job_id": job, "requested_gpu_seconds": 0, "requested_gpu_hours": 0,
        "source_sha256": sources or {}, "binding": binding, "binding_sha256": admission._sha(admission._canonical(binding))}


class Runner:
    def __init__(self, stdout="", *, response=None, after_queue=None, error=None):
        self.stdout, self.after_queue, self.error = stdout, after_queue, error
        self.response = response or SimpleNamespace(returncode=0, stdout="900001;owl\n", stderr="")
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[0] == "squeue":
            if self.after_queue is not None:
                self.after_queue()
            return SimpleNamespace(returncode=0, stdout=self.stdout, stderr="")
        if command[0] == "sbatch":
            if self.error is not None:
                raise self.error
            return self.response
        raise AssertionError("No scheduler/accounting subprocess other than the injected two is permitted")

    @property
    def submissions(self):
        return [command for command, _ in self.calls if command[0] == "sbatch"]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "extension/operations").mkdir(parents=True)
        (self.root / "extension/logs").mkdir()
        self.qa_run = self.root / "extension/runs" / ("joint-memory-qa-" + admission.CPU_JOB)
        self.parser_run = self.root / "extension/runs/joint-gpu-metadata-qa-700001"
        self.parser_path = self.parser_run / "native_qa.json"
        installed = Path(admission.__file__).resolve().parent
        for name in (admission.PLAN, "extension/campaign.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((installed / Path(name).name).read_bytes())
        expected = {"job-name": admission.NAME, "account": "tml_2026", "partition": "b200_normal_q",
            "qos": "owl_b200_normal_short", "nodes": "1", "ntasks": "1", "gres": "gpu:1",
            "cpus-per-task": "8", "mem": "128G", "time": "02:00:00", "no-requeue": "",
            "output": str(self.root / "extension/logs/joint-%j.out"),
            "error": str(self.root / "extension/logs/joint-%j.err")}
        self.script = self.root / admission.SCRIPT
        self.script.write_text("#!/bin/bash\n" + "\n".join("#SBATCH --" + key + ("=" + value if value else "")
            for key, value in expected.items()) + "\nset -euo pipefail\n")
        science_names = [admission.PLAN, admission.SCRIPT, "extension/campaign.json"]
        for index in range(76):
            name = f"extension/frozen_fixture_{index:02d}.py"
            (self.root / name).write_text(f"# metadata fixture {index}\n")
            science_names.append(name)
        self.science_names = science_names
        self.sources = self.inventory()
        self.parser_names = sorted(admission.PARSER_SOURCES | {
            "extension/joint_gpu_metadata_qa.py", "extension/joint_gpu_metadata_qa.slurm"})
        for name in self.parser_names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# synthetic parser source fixture: " + name + "\n")
        self.parser_sources = {name: record(self.root / name) for name in self.parser_names}
        historical = self.root / "deadline/job_ledger.json"
        historical.parent.mkdir()
        historical.write_text("{\"historical\": true}\n")
        self.history_sha = record(historical)["sha256"]
        eleven = {f"prior_{index:02d}": entry(f"prior_{index:02d}", str(600000 + index)) for index in range(11)}
        cpu = entry(admission.qa.ROLE, admission.CPU_JOB, self.sources, plan_sha256=admission.PLAN_SHA256,
            wall_time_seconds=7200, closeout_reserve_seconds=172800,
            prior_binding_sha256={name: value["binding_sha256"] for name, value in eleven.items()})
        self.twelve = {**eleven, admission.qa.ROLE: cpu}
        unit = passed(admission.qa.SCHEMA, self.sources, admission.CPU_JOB, 410)
        unit.update(plan_sha256=admission.PLAN_SHA256, prior_admissions=eleven,
            admission_binding_sha256=cpu["binding_sha256"],
            test_counts=dict(zip(admission.qa.PATTERNS, admission.qa.COUNTS)), source_grader_executed=False,
            source_grader_admitted=False, full_source_readiness_465_scope_replayed=False)
        save(self.qa_run / "unit_qa.json", unit)
        native = {**unit, "schema": "extension-joint-memory-admission-native-qa-v1",
            "unit_qa_sha256": record(self.qa_run / "unit_qa.json")["sha256"],
            "prepared_sha256": admission._read(self.root / admission.PLAN)["inputs"]["prepared"]["sha256"]}
        save(self.qa_run / "native_qa.json", native)
        save(self.qa_run / "complete.json", {"schema": "extension-joint-memory-qa-complete-v1", "complete": True,
            "cpu_only": True, "job_id": admission.CPU_JOB, "gpu_seconds": 0, "tests_run": 410,
            "plan_sha256": admission.PLAN_SHA256, "native_qa_sha256": record(self.qa_run / "native_qa.json")["sha256"],
            "unit_qa_sha256": record(self.qa_run / "unit_qa.json")["sha256"], "source_inventory": self.sources,
            "input_snapshot_unchanged": True, "automatic_downstream": False})
        save(self.qa_run / "accounting.json", accounting(admission.CPU_JOB, admission.CPU_NAME))
        self.qa_pins = {name: record(self.qa_run / name)["sha256"] for name in admission.QA_PINS}
        metadata_cpu = entry(admission.METADATA_ROLE, "700001", self.parser_sources,
            wall_time_seconds=600, expected_native_tests=51, prior_entries_sha256=admission.value_digest(self.twelve))
        parser_qa = passed(admission.METADATA_QA_SCHEMA, self.parser_sources, "700001", 51)
        parser_qa.update(prior_admissions=self.twelve, admission_binding_sha256=metadata_cpu["binding_sha256"])
        save(self.parser_path, parser_qa)
        save(self.parser_run / "accounting.json", accounting())
        self.ledger_path = self.root / "extension/job_ledger.json"
        self.ledger = {"schema": admission.metadata.SCHEMA, "historical_ledger_unchanged": "deadline/job_ledger.json",
            "gpu_hour_cap": None, "jobs": {**self.twelve, admission.METADATA_ROLE: metadata_cpu}}
        save(self.ledger_path, self.ledger)
        self.addCleanup(patch.stopall)
        patch.object(admission, "QA_PINS", self.qa_pins).start()
        patch.object(admission.qa, "source_inventory", side_effect=lambda _: self.inventory()).start()
        patch.object(admission.qa, "HISTORICAL_LEDGER_SHA256", self.history_sha).start()
        patch.object(admission.previous, "check_queue", return_value=None).start()

    def inventory(self):
        return {name: record(self.root / name) for name in self.science_names}

    def evidence(self):
        return admission.load_evidence(self.root, self.qa_run, self.parser_path,
            record(self.parser_path)["sha256"], admission._read(self.ledger_path))

    def submit(self, runner=None, **kwargs):
        return admission.submit(self.root, self.qa_run, self.parser_path, record(self.parser_path)["sha256"],
            now=kwargs.pop("now", NOW), runner=runner or Runner(), **kwargs)

    def change_json(self, path, change):
        value = admission._read(path)
        change(value)
        save(path, value)

    def replace_prior_with_no_id(self, *, role=admission.RECONCILED_ROLE):
        """Synthetic native-history binding; never modifies a real old receipt."""
        old = admission._read(self.qa_run / "unit_qa.json")
        priors = deepcopy(old["prior_admissions"])
        priors.pop("prior_10")
        rejected = {"status": "uncertain", "binding_sha256": "ced9ee106485ebdcdc23475c682085852f3b428d2e98f7b8ee9dc70af7f95f69",
            "requested_gpu_seconds": 0, "requested_gpu_hours": 0, "scheduler_returncode": 1,
            "scheduler_stdout": "", "scheduler_stderr": "sbatch: fatal: Could not establish a configuration source\n"}
        priors[role] = rejected
        cpu = deepcopy(self.twelve[admission.qa.ROLE])
        cpu["binding"]["prior_binding_sha256"] = {name: item["binding_sha256"] for name, item in priors.items()}
        cpu["binding_sha256"] = admission._sha(admission._canonical(cpu["binding"]))
        self.twelve = {**priors, admission.qa.ROLE: cpu}
        old.update(prior_admissions=priors, admission_binding_sha256=cpu["binding_sha256"])
        save(self.qa_run / "unit_qa.json", old)
        native = admission._read(self.qa_run / "native_qa.json")
        native.update(prior_admissions=priors, admission_binding_sha256=cpu["binding_sha256"],
            unit_qa_sha256=record(self.qa_run / "unit_qa.json")["sha256"])
        save(self.qa_run / "native_qa.json", native)
        self.change_json(self.qa_run / "complete.json", lambda value: value.update(
            unit_qa_sha256=record(self.qa_run / "unit_qa.json")["sha256"],
            native_qa_sha256=record(self.qa_run / "native_qa.json")["sha256"]))
        self.qa_pins.update({name: record(self.qa_run / name)["sha256"] for name in self.qa_pins})
        parser_entry = deepcopy(self.ledger["jobs"][admission.METADATA_ROLE])
        parser_entry["binding"]["prior_entries_sha256"] = admission.value_digest(self.twelve)
        parser_entry["binding_sha256"] = admission._sha(admission._canonical(parser_entry["binding"]))
        self.ledger["jobs"] = {**self.twelve, admission.METADATA_ROLE: parser_entry}
        save(self.ledger_path, self.ledger)
        self.change_json(self.parser_path, lambda value: value.update(prior_admissions=self.twelve,
            admission_binding_sha256=parser_entry["binding_sha256"]))
        return rejected


class ResourceParserTests(unittest.TestCase):
    def test_real_slurm_gpu_forms_and_typed_variants(self):
        for value, count in (("gres/gpu:1", 1), ("gpu:1", 1), ("gpu:b200:1", 1),
                ("gres/gpu:B200:1", 1), ("gpu:a100:2", 2), ("gres/gpu:h100-80gb:3", 3), ("gpu:0", 0)):
            with self.subTest(value=value):
                self.assertEqual(admission.gpu_count(value), count)

    def test_explicit_zero_resource_forms(self):
        for value in ("N/A", "(null)", "null", "None", "none", "0"):
            self.assertEqual(admission.gpu_count(value), 0)

    def test_unknown_gres_is_not_silently_zero(self):
        for value in ("", "gpu", "gpu:unknown", "gpu:-1", "gres/gpu:", "license:1", "mps:20",
                "gpu:1,mps:20", "gpu:b200:1(IDX:0)", "gpu:1,gpu:1", "gpu:1025", None):
            with self.subTest(value=value), self.assertRaises(admission.SubmissionError):
                admission.gpu_count(value)

    def test_queue_allocation_and_pending_count(self):
        rows = admission.parse_queue("CLUSTER: owl\n1|RUNNING|a|gres/gpu:1\n2|PENDING|b|gpu:b200:1\n"
            "3|RUNNING|cpu|N/A\n4|COMPLETED|past|gpu:1\n")
        with self.assertRaises(admission.SubmissionError):
            admission.check_capacity({"jobs": rows})
        self.assertEqual(admission.check_capacity({"jobs": rows}, already_submitted=True), 2)

    def test_one_other_gpu_allows_exactly_second_allocation(self):
        rows = admission.parse_queue("1|RUNNING|other|gres/gpu:4\n2|PENDING|cpu|N/A\n")
        self.assertEqual(admission.check_capacity({"jobs": rows}), 1)

    def test_nonterminal_states_conservatively_count(self):
        for state in admission.NONTERMINAL:
            rows = admission.parse_queue(f"1|{state}|a|gpu:1\n2|RUNNING|b|gpu:1\n")
            with self.subTest(state=state), self.assertRaises(admission.SubmissionError):
                admission.check_capacity({"jobs": rows})

    def test_arrays_duplicate_rows_and_unknown_states_rejected(self):
        for text in ("1_0|RUNNING|a|gpu:1\n", "1_[0-3]|PENDING|a|gpu:1\n",
                "1|RUNNING|a|N/A\n1|PENDING|a|N/A\n", "1|FUTURE_STATE|a|N/A\n",
                "1|RUNNING|a\n", "1|RUNNING|a|gpu:1|extra\n"):
            with self.subTest(text=text), self.assertRaises(admission.SubmissionError):
                admission.parse_queue(text)

    def test_snapshot_uses_four_fields_and_all_user_allocations(self):
        runner = Runner("1|PENDING|unrelated|gres/gpu:1\n")
        result = admission.queue_snapshot(runner)
        self.assertEqual(result["jobs"][0]["gpus"], 1)
        command = runner.calls[0][0]
        self.assertIn("--format=%64i|%64T|%128j|%128b", command)
        self.assertIn("--states=all", command)
        self.assertIn("--array", command)

    def test_snapshot_stderr_and_timeout_fail_closed(self):
        for runner in (lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="partial"),
                lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("squeue", 30))):
            with self.assertRaises(admission.SubmissionError):
                admission.queue_snapshot(runner)


class AccountingTests(unittest.TestCase):
    def test_complete_eight_field_accounting_and_optional_step(self):
        value = accounting()
        value["stdout"] += "700001.0|fixture|COMPLETED|0:0|2|2026-10-21T01:00:01|2026-10-21T01:00:03|cpu=1,mem=1G,node=1\n"
        self.assertEqual(admission.validate_accounting(value, "700001"), "sft-extension-joint-gpu-metadata-qa")

    def test_all_mandatory_cpu_rows_required(self):
        for omitted in ("700001|", "700001.batch|", "700001.extern|"):
            value = accounting()
            value["stdout"] = "\n".join(line for line in value["stdout"].splitlines() if not line.startswith(omitted)) + "\n"
            with self.subTest(omitted=omitted), self.assertRaises(admission.SubmissionError):
                admission.validate_accounting(value, "700001")

    def test_cpu_failed_or_active_steps_rejected(self):
        for state, exit_code in (("RUNNING", "0:0"), ("FAILED", "1:0"), ("COMPLETED", "0:9")):
            value = accounting()
            value["stdout"] = value["stdout"].replace(".batch|batch|COMPLETED|0:0", f".batch|batch|{state}|{exit_code}")
            with self.subTest(state=state), self.assertRaises(admission.SubmissionError):
                admission.validate_accounting(value, "700001")

    def test_gpu_unknown_resources_duplicate_rows_and_arrays_rejected(self):
        for text in ("cpu=8,mem=64G,node=1,gres/gpu=1", "cpu=8,mem=64G,node=1,unknown=1",
                "cpu=8,mem=64G,node=1,cpu=8", "mem=64G,node=1"):
            value = accounting()
            value["stdout"] = value["stdout"].replace("billing=20,cpu=8,mem=64G,node=1", text)
            with self.subTest(text=text), self.assertRaises(admission.SubmissionError):
                admission.validate_accounting(value, "700001")
        for extra in (accounting()["stdout"].splitlines()[0],
                "700001_1|array|COMPLETED|0:0|5|2026-10-21T01:00:00|2026-10-21T01:00:05|cpu=8,node=1"):
            value = accounting()
            value["stdout"] += extra + "\n"
            with self.assertRaises(admission.SubmissionError):
                admission.validate_accounting(value, "700001")

    def test_elapsed_timestamps_command_and_timezone_binding(self):
        variants = [("elapsed_seconds", 0), ("elapsed_seconds", 7201), ("timezone", "local"),
            ("gpu_seconds", False), ("complete", False), ("stderr", "partial"),
            ("command", ["sacct"]), ("stdout", accounting()["stdout"].replace("01:00:05", "Unknown"))]
        for key, value in variants:
            data = accounting()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(admission.SubmissionError):
                admission.validate_accounting(data, "700001")

    def test_cpu_proof_accepts_preserved_stripped_newline_hash(self):
        value = passed(admission.METADATA_QA_SCHEMA, {}, "700001", 1)
        self.assertNotEqual(admission._sha(value["compute_proof"]["scontrol"].encode()), value["compute_proof"]["scontrol_sha256"])
        admission._passed_qa(value, admission.METADATA_QA_SCHEMA)

    def test_compute_proof_rejects_login_owner_gpu_and_hash_changes(self):
        for change in (lambda x: x.update(node="owl1"),
                lambda x: x["compute_proof"].update(user_id=os.getuid() + 1),
                lambda x: x["compute_proof"].update(scontrol_sha256="0" * 64),
                lambda x: x["compute_proof"].update(gpu_requested_or_allocated=True)):
            value = passed(admission.METADATA_QA_SCHEMA, {}, "700001", 1)
            change(value)
            with self.assertRaises(admission.SubmissionError):
                admission._passed_qa(value, admission.METADATA_QA_SCHEMA)


class EvidenceTests(Fixture):
    def test_full_scientific_and_parser_qa_graph(self):
        evidence = self.evidence()
        self.assertEqual(len(evidence["sources"]), 79)
        self.assertEqual(len(evidence["prior"]), 13)
        self.assertEqual(len(evidence["parser_sources"]), 4)
        self.assertEqual(evidence["metadata_name"], "sft-extension-joint-gpu-metadata-qa")
        self.assertFalse((self.root / "extension/runs/native-smoke-preparation-1224497/prepared.json").exists())

    def test_pinned_scientific_receipts_cannot_be_replaced_consistently(self):
        self.change_json(self.qa_run / "native_qa.json", lambda x: x.update(tests_run=409))
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_scientific_source_mutation_rejected(self):
        (self.root / self.science_names[-1]).write_text("# changed source\n")
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_parser_source_and_test_actual_bytes_are_required(self):
        for name in admission.PARSER_SOURCES:
            before = (self.root / name).read_bytes()
            (self.root / name).write_text("# modified\n")
            with self.subTest(name=name), self.assertRaises(admission.SubmissionError):
                self.evidence()
            (self.root / name).write_bytes(before)

    def test_metadata_qa_sha_is_mandatory_and_exact(self):
        for checksum in (None, "", "0" * 64, "bad"):
            with self.subTest(checksum=checksum), self.assertRaises(admission.SubmissionError):
                admission.load_evidence(self.root, self.qa_run, self.parser_path, checksum, self.ledger)

    def test_metadata_native_failure_or_skip_never_qualifies(self):
        original = admission._read(self.parser_path)
        for key, value in (("success", False), ("tests_run", 0), ("skipped", 1), ("errors", 1),
                ("expected_failures", 1), ("unexpected_successes", 1), ("gpu_disabled", False),
                ("wrapper_syntax_passed", False), ("schema", "old-410-scope")):
            save(self.parser_path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(admission.SubmissionError):
                self.evidence()

    def test_parser_cpu_entry_must_bind_actual_qa_sources_and_job(self):
        for key, value in (("status", "uncertain"), ("job_id", "700002"),
                ("requested_gpu_seconds", 1), ("source_sha256", {}), ("binding_sha256", "0" * 64)):
            changed = deepcopy(self.ledger)
            changed["jobs"][admission.METADATA_ROLE][key] = value
            save(self.ledger_path, changed)
            with self.subTest(key=key), self.assertRaises(admission.SubmissionError):
                self.evidence()

    def test_parser_qa_old_scope_and_test_count_cannot_be_reused(self):
        self.change_json(self.parser_path, lambda x: x.update(tests_run=410))
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_parser_accounting_must_be_successful_and_same_job(self):
        original = admission._read(self.parser_run / "accounting.json")
        self.change_json(self.parser_run / "accounting.json", lambda x: x.update(job_id="700002"))
        with self.assertRaises(admission.SubmissionError):
            self.evidence()
        save(self.parser_run / "accounting.json", original)
        self.change_json(self.ledger_path, lambda x: x["jobs"][admission.METADATA_ROLE]["binding"].update(wall_time_seconds=1))
        ledger = admission._read(self.ledger_path)
        item = ledger["jobs"][admission.METADATA_ROLE]
        item["binding_sha256"] = admission._sha(admission._canonical(item["binding"]))
        save(self.ledger_path, ledger)
        self.change_json(self.parser_path, lambda x: x.update(admission_binding_sha256=item["binding_sha256"]))
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_extra_ledger_role_or_changed_prior_entry_blocks(self):
        changed = deepcopy(self.ledger)
        changed["jobs"]["later_role"] = entry("later_role", "700003")
        save(self.ledger_path, changed)
        with self.assertRaises(admission.SubmissionError):
            self.evidence()
        changed = deepcopy(self.ledger)
        changed["jobs"]["prior_00"]["scheduler_stdout"] = "changed"
        save(self.ledger_path, changed)
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_historical_ledger_is_only_read_and_pinned(self):
        history = self.root / "deadline/job_ledger.json"
        before = history.read_bytes()
        self.evidence()
        self.assertEqual(history.read_bytes(), before)
        history.write_text("{}\n")
        with self.assertRaises(admission.SubmissionError):
            self.evidence()

    def test_paths_escape_and_symlinks_are_rejected(self):
        alias = self.root / "extension/runs/qa-alias.json"
        alias.symlink_to(self.parser_path)
        for path in (alias, self.root / "extension/../extension/runs" / self.parser_path.name,
                self.root / "deadline/job_ledger.json", self.parser_path.parent.parent.parent / "campaign.json"):
            with self.subTest(path=str(path)), self.assertRaises(admission.SubmissionError):
                admission.load_evidence(self.root, self.qa_run, path, record(self.parser_path)["sha256"], self.ledger)

    def test_inventory_cannot_follow_private_cache_or_outside_sources(self):
        for name in ("extension/runs/private/prepared.json", "../outside.py", "/tmp/outside.py",
                "extension/artifacts/pool.json", "extension/models/config.json"):
            inventory = {**self.parser_sources, name: {"sha256": "0" * 64, "bytes": 1}}
            with self.subTest(name=name), self.assertRaises(admission.SubmissionError):
                admission._parser_inventory(self.root, inventory)

    def test_wrapper_unknown_options_or_gpu_count_changes_rejected(self):
        original = self.script.read_text()
        for text in (original.replace("--gres=gpu:1", "--gres=gpu:2"),
                original.replace("--time=02:00:00", "--time=03:00:00"),
                original.replace("set -euo", "#SBATCH --array=0-1\nset -euo"),
                original + "#SBATCH --gres=gpu:1\n"):
            self.script.write_text(text)
            with self.assertRaises(admission.SubmissionError):
                admission.validate_wrapper(self.root)


class AdmissionTests(Fixture):
    def test_sealed_reconciled_no_id_prior_allows_submit_and_idempotence(self):
        rejected = self.replace_prior_with_no_id()
        # Simulate the exact byte-pinned prior in this isolated synthetic receipt
        # family. Production uses the actual constant and actual immutable QA.
        with patch.object(admission, "RECONCILED_ENTRY_SHA256", admission._sha(admission._canonical(rejected))):
            runner = Runner()
            self.assertEqual(self.submit(runner), "900001")
            self.assertEqual(len(runner.submissions), 1)
            prior = admission._read(self.ledger_path)["jobs"][admission.RECONCILED_ROLE]
            self.assertEqual(prior, rejected)
            self.assertNotIn("job_id", prior)
            self.assertNotIn("possible_job_id", prior)
            self.assertEqual(self.submit(Runner()), "900001")

    def test_unknown_or_altered_uncertain_no_id_prior_blocks_before_sbatch(self):
        rejected = self.replace_prior_with_no_id(role="unknown_no_id_role")
        with patch.object(admission, "RECONCILED_ENTRY_SHA256", admission._sha(admission._canonical(rejected))):
            runner = Runner()
            with self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertEqual(runner.submissions, [])
        for value in ({**rejected, "job_id": "700099"}, {**rejected, "possible_job_id": "700099"},
                {**rejected, "status": "submitted"}, {**rejected, "scheduler_stdout": "700099;owl\n"},
                {**rejected, "scheduler_stderr": "changed"}):
            with patch.object(admission, "RECONCILED_ENTRY_SHA256", admission._sha(admission._canonical(rejected))):
                with self.subTest(value=value), self.assertRaises(admission.SubmissionError):
                    admission.scheduler_identities({admission.RECONCILED_ROLE: value})

    def test_exact_submit_handoff_and_matching_idempotence(self):
        runner = Runner("800001|RUNNING|other-gpu|gres/gpu:1\n")
        prior = deepcopy(self.ledger["jobs"])
        history = (self.root / "deadline/job_ledger.json").read_bytes()
        self.assertEqual(self.submit(runner), "900001")
        self.assertEqual(len(runner.submissions), 1)
        self.assertEqual(runner.submissions[0][-1], str(self.qa_run.relative_to(self.root)))
        ledger = admission._read(self.ledger_path)
        self.assertEqual({key: ledger["jobs"][key] for key in prior}, prior)
        self.assertEqual((self.root / "deadline/job_ledger.json").read_bytes(), history)
        receipt = admission._read(self.root / "extension/operations/joint-memory-admission-job-900001.json")
        self.assertEqual(receipt["code_sha256"], {key: value["sha256"] for key, value in self.sources.items()})
        self.assertEqual(receipt["schema"], admission.SCHEMA)
        self.assertEqual(receipt["metadata_source_sha256"], self.parser_sources)
        self.assertFalse(receipt["automatic_downstream"])
        again = Runner("900001|PENDING|" + admission.NAME + "|gpu:1\n800001|RUNNING|other|gpu:b200:1\n")
        self.assertEqual(self.submit(again), "900001")
        self.assertEqual(again.submissions, [])

    def test_pending_plus_running_gpu_capacity_blocks_without_ledger_write(self):
        runner = Runner("800001|RUNNING|other|gres/gpu:1\n800002|PENDING|other2|gpu:b200:1\n")
        before = self.ledger_path.read_bytes()
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner)
        self.assertEqual(runner.submissions, [])
        self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_untracked_same_name_rejected(self):
        runner = Runner("800001|PENDING|" + admission.NAME + "|gpu:1\n")
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner)
        self.assertEqual(runner.submissions, [])

    def test_source_or_receipt_changes_after_queue_block_submission(self):
        for target in (self.root / self.parser_names[0], self.qa_run / "complete.json"):
            old = target.read_bytes()
            runner = Runner(after_queue=lambda: target.write_bytes(old + b"\n"))
            with self.subTest(path=str(target)), self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertEqual(runner.submissions, [])
            target.write_bytes(old)

    def test_ledger_change_after_live_snapshot_is_not_overwritten(self):
        mutated = deepcopy(self.ledger)
        mutated["annotation"] = "changed during queue read"
        runner = Runner(after_queue=lambda: save(self.ledger_path, mutated))
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner)
        self.assertEqual(admission._read(self.ledger_path), mutated)
        self.assertEqual(runner.submissions, [])

    def test_timeout_journals_uncertainty_and_never_retries(self):
        runner = Runner(error=subprocess.TimeoutExpired("sbatch", 60))
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner)
        self.assertEqual(admission._read(self.ledger_path)["jobs"][admission.ROLE]["status"], "uncertain")
        second = Runner()
        with self.assertRaises(admission.SubmissionError):
            self.submit(second)
        self.assertEqual(second.submissions, [])

    def test_rejection_malformed_output_or_stderr_requires_reconciliation(self):
        original = self.ledger_path.read_bytes()
        for response in (SimpleNamespace(returncode=1, stdout="", stderr="rejected"),
                SimpleNamespace(returncode=0, stdout="900001;other\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="900001\n", stderr="warning")):
            self.ledger_path.write_bytes(original)
            runner = Runner(response=response)
            with self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertIn(admission._read(self.ledger_path)["jobs"][admission.ROLE]["status"], {"uncertain", "submission_failed"})
            second = Runner()
            with self.assertRaises(admission.SubmissionError):
                self.submit(second)
            self.assertEqual(second.submissions, [])

    def test_missing_or_tampered_current_job_receipt_cannot_appear_complete(self):
        self.submit()
        path = self.root / "extension/operations/joint-memory-admission-job-900001.json"
        original = path.read_bytes()
        for value in (None, {}):
            if value is None:
                path.unlink()
            else:
                save(path, value)
            runner = Runner()
            with self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertEqual(runner.submissions, [])
            path.write_bytes(original)

    def test_receipt_write_failure_keeps_accepted_job_uncertain(self):
        original_write = admission.metadata._write
        def write(path, value):
            if Path(path).name.startswith("joint-memory-admission-job-"):
                raise OSError("receipt filesystem failure")
            original_write(path, value)
        runner = Runner()
        with patch.object(admission.metadata, "_write", side_effect=write), self.assertRaises(OSError):
            self.submit(runner)
        current = admission._read(self.ledger_path)["jobs"][admission.ROLE]
        self.assertEqual((current["status"], current["job_id"]), ("uncertain", "900001"))
        second = Runner()
        with self.assertRaises(admission.SubmissionError):
            self.submit(second)
        self.assertEqual(second.submissions, [])

    def test_changed_submitted_binding_and_renamed_job_rejected(self):
        self.submit()
        runner = Runner("900001|RUNNING|wrong-name|gpu:1\n")
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner)
        self.assertEqual(runner.submissions, [])
        self.change_json(self.ledger_path, lambda x: x["jobs"][admission.ROLE]["binding"].update(wall_time_seconds=9000))
        with self.assertRaises(admission.SubmissionError):
            self.submit()

    def test_earlier_cpu_stage_active_or_renamed_rejected(self):
        for row in (f"{admission.CPU_JOB}|RUNNING|{admission.CPU_NAME}|N/A\n",
                "700001|RUNNING|sft-extension-joint-gpu-metadata-qa|N/A\n",
                "700002|COMPLETED|sft-extension-joint-gpu-metadata-qa|N/A\n",
                "600000|RUNNING|old-cpu|gpu:1\n"):
            runner = Runner(row)
            with self.subTest(row=row), self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertEqual(runner.submissions, [])
        # A known admitted GPU request may not disappear from %b while live.
        prior = deepcopy(self.ledger["jobs"])
        prior["prior_00"]["requested_gpu_seconds"] = 7200
        snapshot = {"jobs": admission.parse_queue("600000|RUNNING|old-gpu|N/A\n")}
        with self.assertRaises(admission.SubmissionError):
            admission._known_queue(snapshot, prior, {"job_id": "700001"}, "sft-extension-joint-gpu-metadata-qa")

    def test_current_gpu_resources_and_duplicate_identity_rejected(self):
        self.submit()
        for row in (f"900001|RUNNING|{admission.NAME}|N/A\n",
                f"900001|RUNNING|{admission.NAME}|gpu:2\n",
                f"900002|PENDING|{admission.NAME}|gpu:1\n"):
            runner = Runner(row)
            with self.subTest(row=row), self.assertRaises(admission.SubmissionError):
                self.submit(runner)
            self.assertEqual(runner.submissions, [])

    def test_deadline_requires_whole_allocation_plus_48_hour_reserve(self):
        runner = Runner()
        from datetime import timedelta
        with self.assertRaises(admission.SubmissionError):
            self.submit(runner, now=admission.metadata.DEADLINE - timedelta(seconds=179999))
        self.assertEqual(runner.calls, [])

    def test_submission_environment_cannot_override_gpu_or_array_directives(self):
        runner = Runner()
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:4", "SBATCH_ARRAY_INX": "0-9", "SLURM_JOB_ID": "parent", "CUDA_VISIBLE_DEVICES": "7"}):
            self.submit(runner)
        env = next(kwargs["env"] for command, kwargs in runner.calls if command[0] == "sbatch")
        self.assertFalse(any(key.startswith("SBATCH_") for key in env))
        self.assertNotIn("SLURM_JOB_ID", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)


if __name__ == "__main__":
    unittest.main()
