"""Small scheduler-text fixtures only; no remote calls or ledger mutations."""
import copy
import unittest

from deadline.gpu_budget import (BudgetError, CAP_GPU_SECONDS, conservative_budget,
    make_terminal_receipt, parse_sacct, reconcile_ledger, require_capacity, validate_receipt)


OBSERVED = "2026-10-20T00:45:00+00:00"


def entry(identifier, hours=6):
    return dict(status="submitted", job_id=str(identifier), requested_gpu_hours=hours)


def row(raw, display=None, elapsed=3564, state="COMPLETED", tres="cpu=8,gres/gpu=1,gres/gpu:b200=1,mem=64G"):
    return f"{raw}|{raw if display is None else display}|{state}|{elapsed}|{tres}\n"


def receipt(role, value, text, live=()):
    return make_terminal_receipt(role, value, text, live_job_ids=live, observed_at_utc=OBSERVED)


def archived(status="submission_failed"):
    return {"attempt": {"status": status, "requested_gpu_hours": 12}, "admitted_gpu_hours": 0,
            "proof": {"live_owl_jobs": "", "owl_accounting_today": "",
                      "cancelled_origin_accounting": "8067760|CANCELLED by22825097\n8067760.extern|COMPLETED"},
            "reconciled_at_utc": OBSERVED,
            "reason": "Origin cancelled and successful live/accounting queries confirmed no admitted jobs."}


class BudgetTests(unittest.TestCase):
    def test_completed_exact_allocation_rounds_up_minute_without_typed_double_count(self):
        value = receipt("original_evaluation", entry(1207982), row(1207982))
        self.assertEqual(value["charged_gpu_seconds"], 3600)
        self.assertEqual(value["allocations"][0]["allocated_gpus"], 1)
        self.assertEqual(validate_receipt("original_evaluation", entry(1207982), value, live_job_ids=()), 3600)

    def test_real_two_task_array_uses_each_raw_allocation_once(self):
        text = row(1208041, "1208021_0", 5189) + row(1208021, "1208021_1", 4299)
        value = receipt("probes_and_scores", entry(1208021, 12), text)
        self.assertEqual(value["charged_gpu_seconds"], 5220 + 4320)
        self.assertEqual({r["job_id_raw"] for r in value["allocations"]}, {"1208041", "1208021"})

    def test_cancelled_and_timeout_allocations_still_charge_elapsed(self):
        for state in ("CANCELLED by22825097", "TIMEOUT", "FAILED", "OUT_OF_MEMORY"):
            with self.subTest(state=state):
                value = receipt("role", entry(101), row(101, elapsed=3601, state=state))
                self.assertEqual(value["charged_gpu_seconds"], 3660)

    def test_allocation_zero_elapsed_still_costs_minute(self):
        self.assertEqual(receipt("role", entry(101), row(101, elapsed=0, state="FAILED"))["charged_gpu_seconds"], 60)

    def test_unknown_pending_or_missing_gpu_accounting_never_becomes_zero(self):
        for text in (row(101, elapsed=0, state="PENDING", tres=""),
                     row(101, elapsed=0, state="CANCELLED", tres=""),
                     row(101, tres="gres/gpu:b200=1"), row(101, tres="cpu=8")):
            with self.subTest(text=text), self.assertRaises(BudgetError):
                receipt("role", entry(101), text)
        ledger = {"jobs": {"role": entry(101)}}
        self.assertEqual(conservative_budget(ledger, live_job_ids=())["conservative_committed_gpu_seconds"], 21600)

    def test_explicit_zero_gpu_terminal_record_is_distinct_from_unknown(self):
        value = receipt("role", entry(101), row(101, elapsed=0, state="CANCELLED", tres="gres/gpu=0"))
        self.assertEqual(value["charged_gpu_seconds"], 0)
        with self.assertRaises(BudgetError):
            receipt("role", entry(101), row(101, elapsed=1, tres="gres/gpu=0"))

    def test_partial_wrong_parent_duplicate_or_steps_fail_closed(self):
        value = entry(100, 12)
        invalid = (row(101, "100_0"), row(101, "100_0") * 2,
                   row(101, "100_0") + row(101, "100_1"),
                   row(101, "100_0") + row(102, "999_1"),
                   row(100), row("100.batch"))
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(BudgetError):
                receipt("probes_and_scores", value, text)

    def test_nonarray_raw_id_must_equal_admitted_parent(self):
        with self.assertRaises(BudgetError):
            receipt("role", entry(101), row(999, "101"))

    def test_live_display_parent_or_raw_ids_prohibit_reconciliation(self):
        text = row(101, "100_0") + row(102, "100_1")
        for live in (("100",), ("100_0",), ("101",), ("102",)):
            with self.subTest(live=live), self.assertRaisesRegex(BudgetError, "live"):
                receipt("probes_and_scores", entry(100, 12), text, live)
        with self.assertRaises(BudgetError):
            receipt("role", entry(101), row(101), ("101_[0-1]",))

    def test_invalid_receipt_or_changed_request_cannot_undercharge(self):
        good = receipt("role", entry(101), row(101))
        bad = copy.deepcopy(good)
        bad["charged_gpu_seconds"] = 0
        for value, job in ((bad, entry(101)), (good, entry(102)), (good, entry(101, 12))):
            with self.subTest(job=job), self.assertRaises(BudgetError):
                validate_receipt("role", job, value, live_job_ids=())
        with self.assertRaises(BudgetError):
            validate_receipt("role", entry(101), good, live_job_ids=("101",))

    def test_reconcile_preserves_history_immutability_and_all_evidence(self):
        ledger = {"jobs": {"original_evaluation": entry(1207982), "probes_and_scores": entry(1208021, 12),
            "random_training": entry(1208033), "advanced_training": entry(1208058),
            "random_evaluation": entry(1208059), "advanced_evaluation": entry(1208060)}}
        before = copy.deepcopy(ledger)
        closed = {
            "original_evaluation": receipt("original_evaluation", ledger["jobs"]["original_evaluation"], row(1207982)),
            "probes_and_scores": receipt("probes_and_scores", ledger["jobs"]["probes_and_scores"],
                row(1208041, "1208021_0", 5189) + row(1208021, "1208021_1", 4299)),
            "random_training": receipt("random_training", ledger["jobs"]["random_training"], row(1208033, elapsed=6734)),
        }
        result, audit = reconcile_ledger(ledger, closed, live_job_ids=("1208058", "1208059"))
        self.assertEqual(ledger, before)
        self.assertEqual(audit["after"]["cumulative_admitted_requested_gpu_seconds"], 42 * 3600)
        self.assertEqual(audit["after"]["conservative_committed_gpu_seconds"], 3600 + 9540 + 6780 + 18 * 3600)
        self.assertEqual(audit["evidence"], closed)
        for role, job in result["jobs"].items():
            self.assertEqual(job["requested_gpu_hours"], before["jobs"][role]["requested_gpu_hours"])
        additional = require_capacity(result, live_job_ids=("1208058", "1208059"), additional_requested_gpu_hours=18)
        self.assertLessEqual(additional["projected_gpu_seconds"], CAP_GPU_SECONDS)
        self.assertEqual(additional["cumulative_admitted_requested_gpu_seconds"], 42 * 3600)
        with_long_job = copy.deepcopy(result)
        with_long_job["jobs"]["long_evaluation"] = entry(9999999, 18)
        admitted = conservative_budget(with_long_job, live_job_ids=("9999999",))
        self.assertEqual(admitted["cumulative_admitted_requested_gpu_seconds"], 60 * 3600)
        self.assertEqual(admitted["conservative_committed_gpu_seconds"], additional["projected_gpu_seconds"])
        self.assertTrue(admitted["within_cap"])
        with self.assertRaisesRegex(BudgetError, "exceeds"):
            require_capacity(result, live_job_ids=(), additional_requested_gpu_hours=25)

    def test_explicit_zero_cpu_roles_and_failed_submissions_keep_separate_history(self):
        ledger = {"jobs": {"cpu": entry(100, 0), "unknown_submission": {"status": "uncertain", "requested_gpu_hours": 6}},
            "reconciled_unadmitted_attempts": [archived()]}
        budget = conservative_budget(ledger, live_job_ids=())
        self.assertEqual(budget["conservative_committed_gpu_seconds"], 6 * 3600)
        self.assertEqual(budget["cumulative_all_attempt_requested_gpu_seconds"], 18 * 3600)

    def test_extra_gpu_allocation_can_exceed_original_reservation(self):
        job = entry(101, 6)
        value = receipt("role", job, row(101, elapsed=21601, tres="gres/gpu=2,gres/gpu:b200=2"))
        self.assertEqual(value["charged_gpu_seconds"], 21660 * 2)
        ledger = {"jobs": {"role": {**job, "gpu_accounting_receipt": value}}}
        self.assertGreater(conservative_budget(ledger, live_job_ids=())["conservative_committed_gpu_seconds"], 6 * 3600)

    def test_tres_duplicates_typed_contradictions_and_unknown_elapsed_fail(self):
        for text in (row(101, tres="gres/gpu=1,gres/gpu=1"),
                     row(101, tres="gres/gpu=1,gres/gpu:b200=2"),
                     row(101, elapsed="Unknown"), row(101, elapsed=-1)):
            with self.subTest(text=text), self.assertRaises(BudgetError):
                receipt("role", entry(101), text)

    def test_header_parsing_utc_binding_and_unknown_states(self):
        text = "JobIDRaw|JobID|State|ElapsedRaw|AllocTRES\n" + row(101)
        self.assertEqual(len(parse_sacct(text)), 1)
        with self.assertRaises(BudgetError):
            make_terminal_receipt("role", entry(101), text, live_job_ids=(), observed_at_utc="2026-10-20T00:00:00")
        with self.assertRaises(BudgetError):
            receipt("role", entry(101), row(101, state="UNKNOWN"))

    def test_missing_reservation_or_unknown_historical_admission_is_not_zero(self):
        with self.assertRaisesRegex(BudgetError, "explicit original"):
            conservative_budget({"jobs": {"role": {"status": "submitted", "job_id": "101"}}}, live_job_ids=())
        with self.assertRaisesRegex(BudgetError, "zero-admission"):
            conservative_budget({"jobs": {}, "reconciled_unadmitted_attempts": [
                {"attempt": {"requested_gpu_hours": 6}}]}, live_job_ids=())

    def test_archived_submitted_uncertain_or_known_live_identity_cannot_be_zero(self):
        for status in ("submitted", "uncertain", "unknown"):
            with self.subTest(status=status), self.assertRaises(BudgetError):
                conservative_budget({"jobs": {}, "reconciled_unadmitted_attempts": [archived(status)]}, live_job_ids=())
        for status in ("submission_failed", "submitting", "submitted"):
            for field, value in (("job_id", "999"), ("scheduler_response", "999;owl"),
                                 ("job_ids", ["999"]), ("array_job_id", "999")):
                item = archived(status)
                item["attempt"][field] = value
                with self.subTest(status=status, field=field), self.assertRaises(BudgetError):
                    conservative_budget({"jobs": {}, "reconciled_unadmitted_attempts": [item]}, live_job_ids=("999",))

    def test_archived_interrupted_identity_free_request_keeps_prior_proof(self):
        item = archived("submitting")
        before = copy.deepcopy(item)
        result = conservative_budget({"jobs": {}, "reconciled_unadmitted_attempts": [item]}, live_job_ids=("123",))
        self.assertEqual(result["conservative_committed_gpu_seconds"], 0)
        self.assertEqual(item, before)
        self.assertEqual(result["cumulative_all_attempt_requested_gpu_seconds"], 12 * 3600)

    def test_incomplete_archived_proof_retains_full_request(self):
        for proof in ({}, {"live_owl_jobs": ""}, {"live_owl_jobs": "999 RUNNING", "owl_accounting_today": ""}):
            item = archived()
            item["proof"] = proof
            result = conservative_budget({"jobs": {}, "reconciled_unadmitted_attempts": [item]}, live_job_ids=())
            self.assertEqual(result["conservative_committed_gpu_seconds"], 12 * 3600)


if __name__ == "__main__":
    unittest.main()
