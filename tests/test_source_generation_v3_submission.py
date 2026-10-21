"""Unexecuted metadata-only fixtures; no private cache/model/scheduler access."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import source_generation_v3_submission as module
from extension.cpu_feasibility import value_digest


def entry(role="fixture", job="100", seconds=0, sources=None):
    binding = {"role": role, "automatic_downstream": False, "wall_time_seconds": 1200}
    if sources is not None: binding["source_sha256"] = sources
    return {"status": "submitted", "job_id": job, "requested_gpu_seconds": seconds, "requested_gpu_hours": seconds / 3600,
            "binding": binding, "binding_sha256": value_digest(binding), "source_sha256": sources,
            "command": ["sbatch", "fixture.slurm"]}


def seals():
    return {"prior_snapshot": {"path": "extension/report/synthetic-final/job_ledger_snapshot.json", "sha256": "a" * 64},
        "prior_entries": 24, "prior_entries_sha256": "b" * 64,
        "allocator97": {"directory": "extension/report/synthetic97", "job_id": "101", "elapsed_seconds": 32,
            "pins": {name: "c" * 64 for name in ("unit_qa.json", "native_qa.json", "complete.json", "accounting.json", "source_install_receipt.json")}},
        "allocator_v4": {"role": "synthetic_allocator_v4", "job_id": "102",
            "admission": {"path": "extension/operations/synthetic-v4-job-102.json", "sha256": "d" * 64}},
        "executed_bootstraps": {name: "e" * 64 for name in module.ALLOCATOR_BOOTSTRAPS}}


def proof_fixture():
    initial = {"fixture": entry(), module.generation.RETRY_FAILURE["role"]:
        entry(module.generation.RETRY_FAILURE["role"], job="1228729", sources={"old-core.py": {"sha256": "0" * 64, "bytes": 1}})}
    return {"initial": initial, "initial_sha256": value_digest(initial), "prior": deepcopy(initial),
            "sources": {"core.py": {"sha256": "a" * 64, "bytes": 1}},
            "metadata": {"metadata.py": {"sha256": "b" * 64, "bytes": 1}}, "allocator_sources": {},
            "plan_sha256": "c" * 64, "qa_sha256": "d" * 64, "qa_run": "extension/runs/source-generation-v3-qa-101",
            "assets": {module.BOOTSTRAP: {"sha256": "e" * 64, "bytes": 1}}}


def cpu_entry(proof):
    current = entry(module.QA_ROLE, job="101", sources=proof["sources"])
    current["binding"].update(prior_entries_sha256=proof["initial_sha256"], metadata_source_inventory=proof["metadata"],
        plan_sha256=proof["plan_sha256"], admission_code_sha256=proof["assets"][module.BOOTSTRAP]["sha256"],
        initial_evidence={module.BOOTSTRAP: proof["assets"][module.BOOTSTRAP]}, closeout_reserve_seconds=172800,
        expected_native_tests=60)
    current["binding"].update(module.retry_bindings())
    current["binding_sha256"] = value_digest(current["binding"])
    return current


def qa_fixture():
    proof = proof_fixture()
    current = cpu_entry(proof)
    qa = {"tests_run": 60, "test_counts": {"test_source_generation_identity_v3.py": 40, "test_source_generation_v3_submission.py": 20},
        "parent_tests_previously_passed": 478, "parent_tests_replayed": False,
        "identity_parent_tests_previously_passed": 65, "identity_parent_tests_replayed": False,
        "allocator_tests_previously_passed": 97, "allocator_tests_replayed": False, "collector_tests": 40, "metadata_tests": 20,
        "source_inventory": proof["sources"], "code_sha256": {name: value["sha256"] for name, value in proof["sources"].items()},
        "metadata_source_inventory": proof["metadata"], "allocator_source_inventory": proof["allocator_sources"],
        "plan_sha256": proof["plan_sha256"], "cache_sha256": module.generation.CACHE["sha256"],
        "parent_complete_sha256": module.generation.PARENT_PINS["complete.json"],
        "scientific_parent_sha256": module.generation.SCIENTIFIC_PARENT["sha256"],
        "identity_parent_complete_sha256": module.generation.IDENTITY_PINS["complete.json"],
        "identity_helper_sha256": module.generation.IDENTITY_HELPER["sha256"],
        **module.retry_bindings(), "old_failed_cpu60_replayed": False, "old_failed_cpu60_passed": False,
        "old_fixture_failure_preserved": True,
        "actual_vllm_interface_established": False, "wrapper_syntax_passed": True,
        "admission_binding_sha256": current["binding_sha256"], "cpu_bootstrap_sha256": proof["assets"][module.BOOTSTRAP]["sha256"]}
    return qa, proof, {module.QA_ROLE: current}


def wrapper(root):
    return "\n".join(["#!/bin/bash", "#SBATCH --job-name=" + module.NAME, "#SBATCH --account=tml_2026",
        "#SBATCH --partition=b200_normal_q", "#SBATCH --qos=owl_b200_normal_short", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=128G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(root / "extension/logs/fixture-%j.out"),
        "#SBATCH --error=" + str(root / "extension/logs/fixture-%j.err"), "set -euo pipefail"]) + "\n"


class Flow:
    def __init__(self, directory, *, stdout="999;owl\n", stderr="", returncode=0, queue="", transport=None):
        self.root = Path(directory).resolve()
        (self.root / "extension/operations").mkdir(parents=True)
        self.path = self.root / "extension/job_ledger.json"
        self.proof = proof_fixture()
        self.ledger = {"schema": module.m.SCHEMA, "gpu_hour_cap": None, "historical_ledger_unchanged": "deadline/job_ledger.json",
                       "jobs": deepcopy(self.proof["prior"])}
        self.path.write_text(json.dumps(self.ledger))
        self.stdout, self.stderr, self.returncode, self.queue, self.transport = stdout, stderr, returncode, queue, transport
        self.calls, self.at_submit = [], None

    def runner(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if command[0] == "squeue": return SimpleNamespace(returncode=0, stdout=self.queue, stderr="")
        if command[0] != "sbatch": raise AssertionError("Unexpected synthetic command")
        self.at_submit = json.loads(self.path.read_text())
        if self.transport: raise self.transport
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)

    def run(self, changed=None):
        with ExitStack() as stack:
            if changed:
                altered = deepcopy(self.proof)
                altered["plan_sha256"] = "f" * 64
                values = [self.proof, altered] if changed == "before" else [self.proof, self.proof, altered]
                stack.enter_context(patch.object(module, "load_evidence", side_effect=values))
            else: stack.enter_context(patch.object(module, "load_evidence", return_value=self.proof))
            stack.enter_context(patch.object(module, "known_queue"))
            return module.submit(self.root, self.proof["qa_run"], self.proof["qa_sha256"], runner=self.runner)

    def submissions(self): return [value for value in self.calls if value[0][0] == "sbatch"]
    def current(self): return json.loads(self.path.read_text())["jobs"].get(module.ROLE)
    def handoff(self): return self.root / "extension/operations/source-generation-original-v3-job-999.json"


class CollectorV3AdmissionTests(unittest.TestCase):
    def test_deadline_boundary_keeps_full_job_plus_48_hours(self):
        boundary = module.m.DEADLINE - timedelta(seconds=180000)
        self.assertEqual(module.guard(boundary), boundary)
        for value in (boundary + timedelta(microseconds=1), boundary.replace(tzinfo=None), "tomorrow"):
            with self.assertRaises(module.m.SubmissionError): module.guard(value)

    def test_unresolved_final_snapshot_cannot_be_replaced_by_old22_or_implicit_history(self):
        value = seals()
        self.assertIs(module.validate_seals(value), value)
        for field in ("prior_entries", "prior_entries_sha256"):
            changed = deepcopy(value); changed[field] = None
            with self.assertRaises(module.m.SubmissionError): module.validate_seals(changed)
        changed = deepcopy(value); changed["prior_entries"] = 22
        with self.assertRaises(module.m.SubmissionError): module.validate_seals(changed)
        changed = deepcopy(value); changed["prior_snapshot"]["path"] = None
        with self.assertRaises(module.m.SubmissionError): module.validate_seals(changed)

    def test_seals_bind_exact_public_paths_terminal97_v4_and_executed_bootstraps(self):
        value = seals()
        changes = (lambda row: row["prior_snapshot"].update(path="extension/runs/private/cache.json"),
            lambda row: row["prior_snapshot"].update(sha256="pending"), lambda row: row["allocator97"].update(job_id=None),
            lambda row: row["allocator97"].update(elapsed_seconds=0), lambda row: row["allocator97"]["pins"].pop("accounting.json"),
            lambda row: row["allocator_v4"].update(job_id=row["allocator97"]["job_id"]),
            lambda row: row["allocator_v4"]["admission"].update(path="../escape.json"),
            lambda row: row["executed_bootstraps"].clear())
        for change in changes:
            changed = deepcopy(value); change(changed)
            with self.assertRaises(module.m.SubmissionError): module.validate_seals(changed)

    def test_actual_resolved_metadata_source_proof_never_opens_private_cache_or_weights(self):
        root = Path(module.__file__).resolve().parents[1]
        proof = module.initial_evidence(root)
        self.assertEqual((len(proof["sources"]), len(proof["metadata"]), len(proof["allocator_sources"])), (128, 4, 128))
        self.assertEqual(value_digest(proof["initial"]), proof["initial_sha256"])
        self.assertEqual(len(proof["initial"]), 25)
        self.assertEqual(proof["plan"]["admission_seals"]["prior_entries"], 24)
        self.assertEqual(proof["initial"][module.generation.RETRY_FAILURE["role"]]["job_id"], "1228729")
        self.assertEqual(proof["failed_cpu_snapshot_sha256"], module.retry_bindings()["failed_cpu_snapshot_sha256"])
        self.assertEqual(proof["plan"]["retry_failure"], module.generation.RETRY_FAILURE)
        self.assertNotIn(module.generation.CACHE["path"], proof["assets"])
        self.assertFalse(any(name.endswith(".safetensors") for name in proof["assets"]))
        self.assertIn(module.BOOTSTRAP, proof["assets"])

    def test_wrapper_requires_exact_resources_no_requeue_no_late_directives_and_bound_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); path = root / module.SCRIPT; path.parent.mkdir()
            original = wrapper(root); path.write_text(original)
            self.assertEqual(module.validate_wrapper(root)["gres"], "gpu:1")
            for value in (original.replace("gpu:1", "gpu:2"), original.replace("02:00:00", "03:00:00"),
                          original.replace("128G", "256G"), original.replace("--no-requeue", "--requeue"),
                          original + "#SBATCH --gres=gpu:1\n", original.replace("set -euo pipefail", "#SBATCH --exclusive\nset -euo pipefail"),
                          original.replace(str(root / "extension/logs"), "/tmp")):
                path.write_text(value)
                with self.assertRaises(module.m.SubmissionError): module.validate_wrapper(root)

    def test_history_refuses_rewritten_unknown_uncertain_or_wrong_current_cpu(self):
        proof = proof_fixture(); current = cpu_entry(proof)
        ledger = {"schema": module.m.SCHEMA, "gpu_hour_cap": None, "historical_ledger_unchanged": "deadline/job_ledger.json",
                  "jobs": {**deepcopy(proof["initial"]), module.QA_ROLE: current}}
        self.assertEqual(module.validate_history(ledger, proof, "101")[module.QA_ROLE], current)
        variants = []
        for kind in ("old", "failed-parent", "unknown", "uncertain", "gpu"):
            changed = deepcopy(ledger)
            if kind == "old": changed["jobs"]["fixture"]["job_id"] = "300"
            if kind == "failed-parent": changed["jobs"][module.generation.RETRY_FAILURE["role"]]["status"] = "complete"
            if kind == "unknown": changed["jobs"]["unknown"] = entry(job="300")
            if kind == "uncertain": changed["jobs"][module.QA_ROLE]["status"] = "uncertain"
            if kind == "gpu": changed["jobs"][module.QA_ROLE]["requested_gpu_seconds"] = 1
            variants.append(changed)
        for changed in variants:
            with self.assertRaises(module.m.SubmissionError): module.validate_history(changed, proof, "101")

    def test_cpu_bootstrap_execution_hash_and_full_snapshot_binding_are_required(self):
        proof = proof_fixture(); current = cpu_entry(proof)
        ledger = {"schema": module.m.SCHEMA, "gpu_hour_cap": None, "historical_ledger_unchanged": "deadline/job_ledger.json",
                  "jobs": {**deepcopy(proof["initial"]), module.QA_ROLE: current}}
        for field in ("prior_entries_sha256", "admission_code_sha256", "initial_evidence", "plan_sha256", *module.retry_bindings()):
            changed = deepcopy(ledger)
            changed["jobs"][module.QA_ROLE]["binding"][field] = {} if field == "initial_evidence" else "0" * 64
            changed["jobs"][module.QA_ROLE]["binding_sha256"] = value_digest(changed["jobs"][module.QA_ROLE]["binding"])
            with self.subTest(field=field), self.assertRaises(module.m.SubmissionError): module.validate_history(changed, proof, "101")

    def test_known_queue_keeps_uuid_independent_job_names_arrays_and_live_gpu_counts_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root / "fixture.slurm").write_text("#SBATCH --job-name=fixture-name\n")
            prior = {"fixture": entry(seconds=7200)}
            row = {"job_id": "100", "name": "fixture-name", "state": "PENDING", "gpus": 1}
            self.assertEqual(module.known_queue(root, {"jobs": [row]}, prior), {"100": "fixture-name"})
            for change in ({"name": "renamed"}, {"gpus": 2}, {"gpus": 0}, {"job_id": "100_2"}, {"job_id": "200"}):
                with self.assertRaises(module.m.SubmissionError): module.known_queue(root, {"jobs": [{**row, **change}]}, prior)

    def test_native_qa_requires_exact60_and_retains_parent_scope_qualifications(self):
        qa, proof, prior = qa_fixture()
        self.assertIs(module.validate_native_qa(qa, proof, prior), qa)
        for change in ({"tests_run": 59}, {"parent_tests_replayed": True}, {"identity_parent_tests_replayed": True},
                       {"allocator_tests_replayed": True}, {"actual_vllm_interface_established": True},
                       {"old_failed_cpu60_replayed": True}, {"old_failed_cpu60_passed": True}, {"old_fixture_failure_preserved": False},
                       {"test_counts": {"test_source_generation_identity_v3.py": 32, "test_source_generation_v3_submission.py": 20}}):
            with self.assertRaises(module.m.SubmissionError): module.validate_native_qa({**qa, **change}, proof, prior)

    def test_native_qa_binds_all_code_metadata_parent_helper_and_bootstrap_hashes(self):
        qa, proof, prior = qa_fixture()
        for field in ("source_inventory", "code_sha256", "metadata_source_inventory", "allocator_source_inventory",
                      "plan_sha256", "scientific_parent_sha256", "identity_parent_complete_sha256", "identity_helper_sha256",
                      "admission_binding_sha256", "cpu_bootstrap_sha256", *module.retry_bindings()):
            changed = deepcopy(qa)
            changed[field] = {"changed": True} if isinstance(changed[field], dict) else "0" * 64
            with self.subTest(field=field), self.assertRaises(module.m.SubmissionError): module.validate_native_qa(changed, proof, prior)

    def test_success_journals_before_sbatch_and_exact_existing_handoff_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            flow = Flow(temporary)
            self.assertEqual(flow.run(), "999")
            self.assertEqual(flow.at_submit["jobs"][module.ROLE]["status"], "submitting")
            self.assertEqual(flow.current()["status"], "submitted")
            self.assertEqual(json.loads(flow.handoff().read_text())["binding_sha256"], flow.current()["binding_sha256"])
            for name, value in module.retry_bindings().items():
                self.assertEqual(flow.current()["binding"][name], value)
                self.assertEqual(json.loads(flow.handoff().read_text())[name], value)
            self.assertEqual(flow.run(), "999")
            self.assertEqual(len(flow.submissions()), 1)
            self.assertEqual(flow.submissions()[0][0], ["sbatch", "--clusters=owl", "--parsable", str(flow.root / module.SCRIPT), flow.proof["qa_run"] + "/native_qa.json"])

    def test_two_pending_or_running_gpus_block_before_any_ledger_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            flow = Flow(temporary, queue="200|PENDING|other|gres/gpu:1\n201|RUNNING|another|gpu:A100:1\n")
            before = flow.path.read_bytes()
            with self.assertRaisesRegex(module.m.SubmissionError, "At most two"): flow.run()
            self.assertEqual(flow.path.read_bytes(), before)
            self.assertEqual(flow.submissions(), [])

    def test_source_changes_before_submission_or_after_acceptance_never_complete_admission(self):
        for phase in ("before", "after"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                flow = Flow(temporary); before = flow.path.read_bytes()
                with self.assertRaises(module.m.SubmissionError): flow.run(changed=phase)
                if phase == "before":
                    self.assertEqual(flow.path.read_bytes(), before)
                    self.assertEqual(flow.submissions(), [])
                else:
                    self.assertEqual((flow.current()["status"], flow.current()["job_id"]), ("uncertain", "999"))
                    self.assertEqual(len(flow.submissions()), 1)
                    self.assertFalse(flow.handoff().exists())

    def test_rejected_ambiguous_warning_or_wrong_cluster_response_never_retries(self):
        for rc, stdout, stderr in ((1, "", "denied"), (0, "", ""), (0, "999;owl\n", "warning"), (0, "999;other\n", ""), (False, "999;owl\n", "")):
            with tempfile.TemporaryDirectory() as temporary:
                flow = Flow(temporary, stdout=stdout, stderr=stderr, returncode=rc)
                with self.assertRaises(module.m.SubmissionError): flow.run()
                self.assertEqual(flow.current()["status"], "uncertain")
                with self.assertRaises(module.m.SubmissionError): flow.run()
                self.assertEqual(len(flow.submissions()), 1)

    def test_transport_timeout_preserves_uncertainty_and_cannot_be_submitted_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            flow = Flow(temporary, transport=subprocess.TimeoutExpired("sbatch", 60))
            with self.assertRaises(subprocess.TimeoutExpired): flow.run()
            self.assertEqual(flow.current()["status"], "uncertain")
            with self.assertRaises(module.m.SubmissionError): flow.run()
            self.assertEqual(len(flow.submissions()), 1)

    def test_returned_existing_scheduler_identity_is_preserved_without_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            flow = Flow(temporary, stdout="100;owl\n")
            with self.assertRaises(module.m.SubmissionError): flow.run()
            self.assertEqual((flow.current()["status"], flow.current()["job_id"]), ("uncertain", "100"))
            self.assertEqual(len(flow.submissions()), 1)
            self.assertFalse((flow.root / "extension/operations/source-generation-original-v3-job-100.json").exists())

    def test_exclusive_handoff_failure_preserves_accepted_identity_and_has_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            flow = Flow(temporary)
            with patch.object(module, "write_exclusive", side_effect=OSError("synthetic disk failure")), self.assertRaises(OSError): flow.run()
            self.assertEqual((flow.current()["status"], flow.current()["job_id"]), ("uncertain", "999"))
            with self.assertRaises(module.m.SubmissionError): flow.run()
            self.assertEqual(len(flow.submissions()), 1)

    def test_changed_existing_binding_or_missing_handoff_never_resubmits(self):
        for kind in ("binding", "missing"):
            with tempfile.TemporaryDirectory() as temporary:
                flow = Flow(temporary); flow.run()
                if kind == "missing": flow.handoff().unlink()
                else:
                    ledger = json.loads(flow.path.read_text())
                    ledger["jobs"][module.ROLE]["binding"]["model_role"] = "trained"
                    ledger["jobs"][module.ROLE]["binding_sha256"] = value_digest(ledger["jobs"][module.ROLE]["binding"])
                    flow.path.write_text(json.dumps(ledger))
                with self.assertRaises(module.m.SubmissionError): flow.run()
                self.assertEqual(len(flow.submissions()), 1)

    def test_submission_strips_all_scheduler_overrides_but_preserves_slurm_configuration(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"SBATCH_GRES": "gpu:4", "SLURM_JOB_ID": "777",
                "CUDA_VISIBLE_DEVICES": "7", "SLURM_CONF": "/synthetic/slurm.conf", "SLURM_CONF_SERVER": "fixture-controller"}):
            flow = Flow(temporary); flow.run()
            environment = flow.submissions()[0][1]["env"]
            for key in ("SBATCH_GRES", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"): self.assertNotIn(key, environment)
            self.assertEqual(environment["SLURM_CONF"], "/synthetic/slurm.conf")
            self.assertEqual(environment["SLURM_CONF_SERVER"], "fixture-controller")

    def test_exclusive_json_handoff_cannot_overwrite_and_cleans_partial_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "handoff.json"
            value = {"fixture": "metadata only"}
            module.write_exclusive(path, value)
            before = path.read_bytes()
            self.assertEqual(json.loads(before), value)
            with self.assertRaises(FileExistsError): module.write_exclusive(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__": unittest.main()
