"""Metadata-only scheduler oracles; execute later in a native CPU allocation."""
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extension import submission as module


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout,stderr=stderr,returncode=returncode)


class Scheduler:
    def __init__(self, responses=None):
        self.responses=list(responses or [result(),result("12345;owl\n")])
        self.calls=[]

    def __call__(self, command, **kwargs):
        self.calls.append((command,kwargs))
        value=self.responses.pop(0)
        if isinstance(value,Exception):raise value
        if callable(value):return value(command,kwargs)
        return value


class ExtensionSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.extension=self.root/"extension"
        self.extension.mkdir()
        (self.root/"tests").mkdir()
        (self.root/"reasoning").mkdir()
        for name in module.REQUIRED:
            (self.root/name).write_text("small reviewed source\n")
        self.campaign=dict(schema="reasoning-sft-extension-campaign-v1",preserve_completed_experiments=True,
            authorization=dict(new_deadline_utc=module.DEADLINE.isoformat(),
                new_deadline_eastern="2026-10-27T23:59:00-04:00",latest_user_total_gpu_hour_limit=None),
            budget=dict(new_user_global_cap_gpu_hours=None,historical_cap_and_ledger_remain_unchanged=True,
                historical_ledger="deadline/job_ledger.json",new_extension_ledger="extension/job_ledger.json"))
        self.campaign_path=self.extension/"campaign.json"
        self.campaign_path.write_text(json.dumps(self.campaign))
        self.script=self.extension/"cpu_feasibility.slurm"
        self.script.write_text("#!/bin/bash\n"+"\n".join("#SBATCH "+value for value in (
            "--job-name=sft-extension-feasibility","--account=tml_2026","--partition=normal_q",
            "--qos=owl_normal_short","--nodes=1","--ntasks=1","--cpus-per-task=8","--mem=64G",
            "--time=00:30:00","--no-requeue",f"--output={self.extension}/logs/cpu-%j.out",
            f"--error={self.extension}/logs/cpu-%j.err"))+"\nset -euo pipefail\n")
        self.now=module.DEADLINE-timedelta(days=1)
        self.ledger_path=self.extension/"job_ledger.json"
        patcher=patch.object(module.getpass,"getuser",return_value="fixture_user")
        patcher.start()
        self.addCleanup(patcher.stop)

    def submit(self,scheduler=None,**kwargs):
        return module.submit_cpu_feasibility(self.root,now=kwargs.pop("now",self.now),
            runner=scheduler or Scheduler(),**kwargs)

    def ledger(self):return json.loads(self.ledger_path.read_text())

    def test_exact_cpu_command_source_binding_and_live_snapshot(self):
        scheduler=Scheduler([result("CLUSTER: owl\n900_0|RUNNING|other-gpu-work\n"),result("12345;owl\n")])
        self.assertEqual(self.submit(scheduler),"12345")
        self.assertEqual(scheduler.calls[0][0],["squeue","--clusters=owl","--array","--states=all",
            "--user=fixture_user","--noheader","--format=%64i|%64T|%128j"])
        self.assertEqual(scheduler.calls[1][0],["sbatch","--clusters=owl","--parsable",str(self.script)])
        entry=self.ledger()["jobs"][module.ROLE]
        self.assertEqual(entry["status"],"submitted")
        self.assertEqual(entry["requested_gpu_hours"],0)
        self.assertEqual(entry["requested_gpu_seconds"],0)
        self.assertEqual(entry["source_sha256"],module.source_inventory(self.root))
        self.assertEqual(entry["campaign_sha256"],hashlib.sha256(self.campaign_path.read_bytes()).hexdigest())
        self.assertEqual(entry["scheduler_snapshot"]["jobs"],[dict(job_id="900_0",state="RUNNING",name="other-gpu-work")])
        self.assertTrue((self.extension/"logs").is_dir())

    def test_submitting_receipt_is_durable_before_scheduler_call(self):
        def admitted(command,kwargs):
            entry=self.ledger()["jobs"][module.ROLE]
            self.assertEqual(entry["status"],"submitting")
            self.assertEqual(entry["command"],command)
            return result("12345")
        self.submit(Scheduler([result(),admitted]))

    def test_exact_retry_is_idempotent_and_runtime_receipts_are_not_source(self):
        self.submit()
        (self.extension/"runtime_receipt.json").write_text('{"mutable":true}')
        (self.extension/"runs").mkdir()
        (self.extension/"runs/model.json").write_text("not source metadata")
        scheduler=Scheduler([result("12345|PENDING|sft-extension-feasibility\n")])
        self.assertEqual(self.submit(scheduler),"12345")
        self.assertEqual(len(scheduler.calls),1)

    def test_changed_static_source_or_campaign_does_not_resubmit(self):
        self.submit()
        for path in (self.root/"tests/test_extension_submission.py",self.extension/"research.json",self.campaign_path):
            original=path.read_text()
            path.write_text(original+"\n")
            scheduler=Scheduler([result()])
            with self.subTest(path=path),self.assertRaisesRegex(module.SubmissionError,"changed"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls),1)
            path.write_text(original)

    def test_all_ambiguous_failed_or_unknown_record_states_fail_closed(self):
        for status in ("unknown","submitting","uncertain","submission_failed","complete",None):
            ledger=dict(schema=module.SCHEMA,historical_ledger_unchanged="deadline/job_ledger.json",
                        gpu_hour_cap=None,jobs={module.ROLE:dict(status=status)})
            self.ledger_path.write_text(json.dumps(ledger))
            scheduler=Scheduler([result()])
            with self.subTest(status=status),self.assertRaisesRegex(module.SubmissionError,"never resubmit"):
                self.submit(scheduler)
            self.assertEqual(len(scheduler.calls),1)

    def test_inconsistent_record_cannot_hide_a_gpu_request(self):
        self.submit()
        value=self.ledger();value["jobs"][module.ROLE]["requested_gpu_hours"]=1
        self.ledger_path.write_text(json.dumps(value))
        with self.assertRaises(module.SubmissionError):self.submit(Scheduler([result()]))

    def test_timeout_preserves_uncertain_evidence_and_prevents_retry(self):
        scheduler=Scheduler([result(),subprocess.TimeoutExpired("sbatch",60,output=b"12345;owl\n")])
        with self.assertRaisesRegex(module.SubmissionError,"uncertain"):
            self.submit(scheduler)
        entry=self.ledger()["jobs"][module.ROLE]
        self.assertEqual(entry["status"],"uncertain")
        self.assertEqual(entry["scheduler_stdout"],"12345;owl\n")
        retry=Scheduler([result()])
        with self.assertRaises(module.SubmissionError):self.submit(retry)
        self.assertEqual(len(retry.calls),1)

    def test_nonzero_response_retains_possible_job_identity(self):
        with self.assertRaisesRegex(module.SubmissionError,"rejected/failed"):
            self.submit(Scheduler([result(),result("12345;owl\n","network failure",1)]))
        entry=self.ledger()["jobs"][module.ROLE]
        self.assertEqual(entry["status"],"submission_failed")
        self.assertEqual(entry["possible_job_id"],"12345")

    def test_malformed_or_other_cluster_response_is_uncertain(self):
        for response in ("Submitted batch job 12345","12345;other","12345\n12346","0"):
            if self.ledger_path.exists():self.ledger_path.unlink()
            with self.subTest(response=response),self.assertRaisesRegex(module.SubmissionError,"Ambiguous"):
                self.submit(Scheduler([result(),result(response)]))
            self.assertEqual(self.ledger()["jobs"][module.ROLE]["status"],"uncertain")

    def test_snapshot_failure_or_duplicate_rows_prevents_sbatch(self):
        responses=(result("","controller down",1),result("9|RUNNING|a\n9|RUNNING|a\n"),
                   result("9_[0-2]|PENDING|array\n"),result("CLUSTER: other\n"))
        for response in responses:
            scheduler=Scheduler([response])
            with self.subTest(response=response),self.assertRaises(module.SubmissionError):self.submit(scheduler)
            self.assertEqual(len(scheduler.calls),1)
            self.assertFalse(self.ledger_path.exists())

    def test_untracked_matching_live_job_blocks_admission(self):
        scheduler=Scheduler([result("987|SUSPENDED|sft-extension-feasibility\n")])
        with self.assertRaisesRegex(module.SubmissionError,"untracked"):
            self.submit(scheduler)
        self.assertEqual(len(scheduler.calls),1)

    def test_scheduler_environment_cannot_add_gpus_arrays_or_inherited_job_identity(self):
        scheduler=Scheduler()
        with patch.dict(os.environ,{"SBATCH_GRES":"gpu:8","SBATCH_ARRAY_INX":"0-9",
                                   "SLURM_JOB_GPUS":"3","CUDA_VISIBLE_DEVICES":"3"}):
            self.submit(scheduler)
        env=scheduler.calls[1][1]["env"]
        self.assertFalse(any(key.startswith("SBATCH_") for key in env))
        self.assertNotIn("SLURM_JOB_GPUS",env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES",env)

    def test_gpu_directives_unknown_options_and_unbounded_time_refused(self):
        original=self.script.read_text()
        for option in ("--gpus=1","--gres=gpu:1","--tres-per-node=gres/gpu:1","-G 1"):
            self.script.write_text(original.replace("set -euo pipefail","#SBATCH "+option+"\nset -euo pipefail"))
            with self.subTest(option=option),self.assertRaisesRegex(module.SubmissionError,"GPU/unknown"):
                self.submit(Scheduler([]))
        for time in ("UNLIMITED","01:00:00","00:00:00"):
            self.script.write_text(original.replace("00:30:00",time))
            with self.subTest(time=time),self.assertRaisesRegex(module.SubmissionError,"30-minute"):
                self.submit(Scheduler([]))

    def test_runtime_logs_must_stay_below_extension(self):
        self.script.write_text(self.script.read_text().replace(str(self.extension/"logs"),str(self.root/"deadline/logs")))
        with self.assertRaisesRegex(module.SubmissionError,"below extension"):
            self.submit(Scheduler([]))

    def test_inactive_scheduler_header_cannot_hide_the_time_bound(self):
        self.script.write_text(self.script.read_text().replace("#!/bin/bash","#!/bin/bash\necho executable-before-directives"))
        with self.assertRaisesRegex(module.SubmissionError,"would be ignored"):
            self.submit(Scheduler([]))

    def test_changed_source_during_snapshot_prevents_sbatch(self):
        def change(command,kwargs):
            (self.extension/"cpu_feasibility.py").write_text("changed during scheduler query")
            return result()
        scheduler=Scheduler([change])
        with self.assertRaisesRegex(module.SubmissionError,"during admission"):
            self.submit(scheduler)
        self.assertEqual(len(scheduler.calls),1)
        self.assertFalse(self.ledger_path.exists())

    def test_one_hour_reserve_plus_runtime_and_campaign_deadline_are_required(self):
        with self.assertRaisesRegex(module.SubmissionError,"one-hour"):
            self.submit(Scheduler([]),now=module.DEADLINE-timedelta(seconds=5399))
        value=copy.deepcopy(self.campaign)
        value["authorization"]["new_deadline_utc"]="2026-10-21T03:59:00+00:00"
        self.campaign_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(module.SubmissionError,"deadline"):
            self.submit(Scheduler([]))

    def test_historical_ledger_is_never_opened_or_changed(self):
        historical=self.root/"deadline/job_ledger.json"
        historical.parent.mkdir()
        historical.write_text("historical immutable sentinel, intentionally not JSON")
        self.submit()
        self.assertEqual(historical.read_text(),"historical immutable sentinel, intentionally not JSON")

    def test_symlinked_extension_ledger_cannot_overwrite_history(self):
        historical=self.root/"deadline/job_ledger.json"
        historical.parent.mkdir()
        historical.write_text("immutable")
        self.ledger_path.symlink_to(historical)
        with self.assertRaisesRegex(module.SubmissionError,"symlinks"):
            self.submit(Scheduler([]))
        self.assertEqual(historical.read_text(),"immutable")

    def test_unknown_role_in_isolated_ledger_prevents_any_scheduler_action(self):
        value=dict(schema=module.SCHEMA,historical_ledger_unchanged="deadline/job_ledger.json",
                   gpu_hour_cap=None,jobs={"unreviewed_gpu_stage":dict(status="submitted",job_id="999")})
        self.ledger_path.write_text(json.dumps(value))
        scheduler=Scheduler([])
        with self.assertRaisesRegex(module.SubmissionError,"Unknown extension ledger"):
            self.submit(scheduler)
        self.assertEqual(scheduler.calls,[])

    def test_large_source_or_missing_required_test_cannot_enter_metadata_binding(self):
        path=self.extension/"research.json"
        path.write_bytes(b"x"*(module.MAX_SOURCE_BYTES+1))
        scheduler=Scheduler([])
        with self.assertRaisesRegex(module.SubmissionError,"small source"):
            self.submit(scheduler)
        self.assertEqual(scheduler.calls,[])
        path.write_text("small source")
        (self.root/"tests/test_extension_feasibility.py").unlink()
        with self.assertRaisesRegex(module.SubmissionError,"Missing required"):
            self.submit(scheduler)


if __name__ == "__main__":unittest.main()
