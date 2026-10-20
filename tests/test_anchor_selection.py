"""Offline invariants for the frozen anchor screen; run in a CPU allocation."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deadline.anchor_search import screen
from deadline.recovery.common import digest, write_new_json


def metric(correct=400, solved=110, lengths=3, repetition=.05):
    p1, p8 = Fraction(correct, screen.ROWS*screen.N), Fraction(solved, screen.ROWS)
    h = 2*p1*p8/(p1+p8) if p1+p8 else Fraction(0)
    return dict(rows=screen.ROWS, samples=screen.ROWS*screen.N,
                agreeing_samples=correct, solved_questions=solved,
                pass_at_1=float(p1), pass_at_8=float(p8), harmonic_score=float(h),
                length_finishes=lengths, mean_repetition_8gram=repetition)


def identifiers(split):
    return [hashlib.sha256(f"{split}:{i}".encode()).hexdigest() for i in range(screen.ROWS)]


def observations(ids, correct=400, solved=110):
    result, remaining = [], correct-solved
    for i, identifier in enumerate(ids):
        count = 1 if i < solved else 0
        extra = min(screen.N-1, remaining) if count else 0
        count += extra
        remaining -= extra
        result.append(dict(id=identifier, teacher_answer_sha256="a"*64,
            teacher_agreement=[True]*count+[False]*(screen.N-count),
            finish_reason=["stop"]*screen.N, boxed_completion=[True]*screen.N,
            repetition_8gram=[.05]*screen.N, output_tokens=[5]*screen.N,
            output_sha256=["b"*64]*screen.N))
    assert remaining == 0
    return result


class SelectionRuleTests(unittest.TestCase):
    def metrics(self):
        return {"original": metric(), "incumbent": metric(),
                "alpha0125": metric(480,125), "alpha0375": metric(470,123)}

    def test_highest_harmonic_candidate_wins(self):
        result = screen.choose_candidate(self.metrics())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["winner"],"alpha0125")

    def test_candidate_tie_prefers_lower_alpha(self):
        values = self.metrics()
        values["alpha0375"] = copy.deepcopy(values["alpha0125"])
        self.assertEqual(screen.choose_candidate(values)["winner"],"alpha0125")

    def test_exact_incumbent_tie_never_promotes(self):
        values = {name: metric() for name in screen.ROLES}
        result = screen.choose_candidate(values)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["winner"],"incumbent")

    def test_partial_role_set_rejected(self):
        values = self.metrics()
        del values["alpha0375"]
        with self.assertRaisesRegex(ValueError,"four complete"):
            screen.choose_candidate(values)

    def test_negative_candidate_is_not_promoted(self):
        values = self.metrics()
        values["alpha0125"] = metric(300,100)
        values["alpha0375"] = metric(310,100)
        self.assertFalse(screen.choose_candidate(values)["accepted"])

    def test_both_original_and_incumbent_accuracy_floors_apply(self):
        gate = screen.candidate_gate(metric(550,140), metric(600,140), metric(400,100))
        self.assertFalse(gate["checks"]["pass_at_1_floor"])
        gate = screen.candidate_gate(metric(600,130), metric(600,140), metric(400,100))
        self.assertFalse(gate["checks"]["pass_at_8_floor"])

    def test_length_ceiling_uses_all_eight_samples_and_ceil(self):
        gate = screen.candidate_gate(metric(480,125,20), metric(lengths=3),metric(lengths=4))
        self.assertEqual(gate["length_finish_ceiling"],20)
        self.assertTrue(gate["accepted"])
        self.assertFalse(screen.candidate_gate(metric(480,125,21),metric(lengths=3),
                                             metric(lengths=4))["accepted"])

    def test_repetition_boundary(self):
        self.assertTrue(screen.candidate_gate(metric(480,125,repetition=.06),metric(),metric())["accepted"])
        self.assertFalse(screen.candidate_gate(metric(480,125,repetition=.06000001),metric(),metric())["accepted"])

    def test_forged_or_nonfinite_aggregate_refused(self):
        values = self.metrics()
        values["alpha0125"]["pass_at_1"] = .99
        with self.assertRaisesRegex(ValueError,"exact counts"):
            screen.choose_candidate(values)
        values = self.metrics()
        values["alpha0125"]["mean_repetition_8gram"] = float("nan")
        with self.assertRaises(ValueError):
            screen.choose_candidate(values)

    def test_confirmation_cannot_switch_candidates(self):
        selection = dict(accepted=True,winner="alpha0125")
        values = {"original": metric(),"incumbent":metric(),"alpha0375":metric(500,130)}
        with self.assertRaisesRegex(ValueError,"single selected"):
            screen.confirmation_gate(selection,values)

    def test_failed_confirmation_retains_incumbent(self):
        values = {"original":metric(),"incumbent":metric(),"alpha0125":metric(300,100)}
        result = screen.confirmation_gate(dict(accepted=True,winner="alpha0125"),values)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["winner"],"incumbent")
        self.assertEqual(result["proposed_winner"],"alpha0125")


class ObservationTests(unittest.TestCase):
    def test_eight_sample_p1_and_any_sample_p8(self):
        ids = identifiers("search")
        result = screen.summarize_records(observations(ids),ids)
        self.assertEqual(result["agreeing_samples"],400)
        self.assertEqual(result["solved_questions"],110)
        self.assertEqual(result["pass_at_1"],400/1536)
        self.assertEqual(result["pass_at_8"],110/192)

    def test_partial_or_reordered_question_records_rejected(self):
        ids = identifiers("search")
        rows = observations(ids)
        for invalid in (rows[:-1],list(reversed(rows))):
            with self.assertRaisesRegex(ValueError,"complete ordered"):
                screen.summarize_records(invalid,ids)

    def test_partial_samples_and_raw_text_fields_rejected(self):
        ids = identifiers("search")
        rows = observations(ids)
        rows[0]["output_tokens"].pop()
        with self.assertRaisesRegex(ValueError,"Missing sample"):
            screen.summarize_records(rows,ids)
        rows = observations(ids)
        rows[0]["generated_text"] = "must never be exported"
        with self.assertRaisesRegex(ValueError,"Unexpected observation"):
            screen.summarize_records(rows,ids)


class SealedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.prepared = dict(schema="deadline-anchor-search-prepared-v1",plan_sha256=digest(screen.PLAN),
            code_sha256={"screen.py":digest(screen.__file__)},
            models={role:dict(path=f"/models/{role}",files={"model.safetensors":"f"*64}) for role in screen.ROLES},
            source_audit=dict(artifacts="/source",files={"heldout.json":"d"*64}),
            splits={split:[dict(id=identifier,teacher_answer_sha256="a"*64) for identifier in identifiers(split)]
                    for split in ("search","confirmation")})
        write_new_json(self.output/"prepared.json",self.prepared)
        deadline = screen.datetime.fromisoformat(screen.read_json(screen.PLAN)["resources"]["deadline_utc"])
        allocation = dict(job_id="1234",time_limit_seconds=5400,allocated_gpus=1,
            scheduler_record_sha256="c"*64,started_at_utc="2026-10-20T18:00:00+00:00",
            start_reserve_hours=6.5,deadline_utc=deadline.isoformat())
        write_new_json(self.output/"gpu_allocation.json",allocation)
        self.allocation = allocation
        patcher = patch.object(screen,"_load",return_value=self.prepared)
        patcher.start()
        self.addCleanup(patcher.stop)

    def stage(self,split,roles):
        ids = identifiers(split)
        for role in roles:
            path = self.output/"screen"/split/role
            records = observations(ids,480 if role=="alpha0125" else 400,
                                   125 if role=="alpha0125" else 110)
            manifest = dict(schema=screen.SCHEMA,phase="run",role=role,split=split,
                protocol=screen._protocol(),model=self.prepared["models"][role],
                prepared_sha256=digest(self.output/"prepared.json"),plan_sha256=digest(screen.PLAN),
                code_sha256=self.prepared["code_sha256"],source_audit=self.prepared["source_audit"],
                ordered_ids=ids,allocation=self.allocation,
                selection_sha256=digest(self.output/"selection.json") if split=="confirmation" else None,
                versions={"vllm":"0.12.0"},hardware={"name":"test B200"})
            write_new_json(path/"manifest.json",manifest)
            write_new_json(path/"observations.json",records)
            receipt = dict(schema=screen.SCHEMA,complete=True,role=role,split=split,
                metrics=screen.summarize_records(records,ids),
                manifest_sha256=digest(path/"manifest.json"),observations_sha256=digest(path/"observations.json"))
            write_new_json(path/"complete.json",receipt)

    def accepted(self):
        self.stage("search",screen.ROLES)
        write_new_json(self.output/"selection.json",screen._selection_payload(self.output))
        self.stage("confirmation",("original","incumbent","alpha0125"))
        write_new_json(self.output/"decision.json",screen._decision_payload(self.output))

    def test_complete_sealed_decision_validates_without_model_reads(self):
        self.accepted()
        result = screen.validate_decision(self.output)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["winner"],"alpha0125")
        self.assertEqual(result["model_path"],"/models/alpha0125")

    def test_missing_search_role_refuses_selection(self):
        self.stage("search",screen.ROLES[:-1])
        with self.assertRaisesRegex(ValueError,"Missing or unexpected"):
            screen._selection_payload(self.output)

    def test_tampered_selection_and_decision_refused(self):
        self.accepted()
        path = self.output/"decision.json"
        value = json.loads(path.read_text())
        value["model_path"] = "/other/model"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"Stored decision"):
            screen.validate_decision(self.output)

    def test_extra_confirmation_candidate_refused(self):
        self.accepted()
        (self.output/"screen/confirmation/alpha0375").mkdir()
        with self.assertRaisesRegex(ValueError,"Missing or unexpected"):
            screen.validate_decision(self.output)

    def test_modified_observations_fail_even_with_unchanged_aggregates(self):
        self.accepted()
        path = self.output/"screen/confirmation/alpha0125/observations.json"
        value = json.loads(path.read_text())
        value[0]["output_sha256"][0] = "e"*64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"incompatible"):
            screen.validate_decision(self.output)


if __name__ == "__main__":
    unittest.main()
