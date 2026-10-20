"""Transparent GPU accounting for the fresh experiment's unchanged 48-hour cap.

Policy: charge verified TERMINAL allocations by their generic gres/gpu count,
rounding each allocation's elapsed time up to a minute. Every open, missing or
unverified allocation retains its entire original requested reservation. A job
that allocated GPUs but reports zero elapsed seconds still costs one minute.
Unknown/empty GPU accounting is never interpreted as zero. Historical requested
hours remain immutable and are reported separately from this conservative bound.

This module performs no scheduler calls, submissions, ledger writes or model
work. Callers obtain sacct -X --parsable2 allocation rows with fields
JobIDRaw,JobID,State,ElapsedRaw,AllocTRES and an uncompressed squeue --array ID
snapshot. They must apply reconcile_ledger's returned copy while holding the
existing ledger lock, retaining its returned audit evidence in the same write.
Use sacct --duplicates (-D) with -X so restart history cannot be omitted; extra
allocation records then fail closed. Use squeue --states=all --array and exclude
only known terminal states, so suspended/requeue transitions remain live.
"""
import copy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import re


CAP_GPU_SECONDS = 48 * 3600
SCHEMA = "deadline-terminal-gpu-accounting-v1"
POLICY = "verified-terminal-allocations-plus-full-open-reservations-v1"
TERMINAL = frozenset({"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
                     "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"})
FIELDS = ("JobIDRaw", "JobID", "State", "ElapsedRaw", "AllocTRES")


class BudgetError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BudgetError(message)


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def seconds(hours):
    require(type(hours) in (int, float, str, Decimal), "Requested GPU hours must be numeric")
    try:
        value = Decimal(str(hours))
        require(value.is_finite() and value >= 0, "Requested GPU hours must be finite and nonnegative")
        return int((value * 3600).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise BudgetError("Invalid requested GPU hours") from error


def utc_timestamp(value):
    require(isinstance(value, str), "Accounting observation requires a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BudgetError("Invalid accounting UTC timestamp") from error
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "Accounting timestamp must explicitly identify UTC")
    return parsed.isoformat()


def live_ids(values):
    require(not isinstance(values, (str, bytes)), "Pass an explicit collection of live job IDs")
    result = set(values)
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value)
                for value in result), "Live IDs must be uncompressed squeue --array identities")
    return result


def parse_sacct(text):
    """Parse allocation-only sacct output; steps/heterogeneous IDs fail closed."""
    require(isinstance(text, str), "sacct evidence must be exact text")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split("|")]
        if len(values) == 6 and values[-1] == "":
            values.pop()
        if tuple(values) == FIELDS:
            continue
        require(len(values) == 5, "sacct evidence must contain exactly the five requested fields")
        row = dict(zip(FIELDS, values))
        require(re.fullmatch(r"[0-9]+", row["JobIDRaw"]) is not None and
                re.fullmatch(r"[0-9]+(?:_[0-9]+)?", row["JobID"]) is not None,
                "sacct -X base allocation IDs are required; steps or unknown IDs are forbidden")
        require(re.fullmatch(r"[0-9]+", row["ElapsedRaw"]) is not None,
                "ElapsedRaw must be a known nonnegative integer")
        rows.append(row)
    return rows


def allocated_gpus(tres):
    values = {}
    for part in tres.split(","):
        if not part.strip():
            continue
        name, separator, value = part.strip().partition("=")
        require(separator and name not in values, "Malformed or duplicate AllocTRES key")
        values[name] = value
    require("gres/gpu" in values and re.fullmatch(r"[0-9]+", values["gres/gpu"]) is not None,
            "Generic gres/gpu allocation count is unknown; retain the full reservation")
    count = int(values["gres/gpu"])
    typed = [value for key, value in values.items() if key.startswith("gres/gpu:")]
    require(all(re.fullmatch(r"[0-9]+", value) for value in typed), "Invalid typed GPU allocation")
    require(sum(map(int, typed)) <= count, "Typed GPU allocations contradict the generic GPU count")
    return count


def _expected(role, entry):
    parent = str(entry.get("job_id", ""))
    require(re.fullmatch(r"[0-9]+", parent) is not None, "A known admitted parent job ID is required")
    require(entry.get("status") == "submitted", "Only admitted submitted jobs can use allocation accounting")
    require("requested_gpu_hours" in entry, "The original requested GPU reservation is missing")
    return parent, ({parent + "_0", parent + "_1"} if role == "probes_and_scores" else {parent})


def _not_live(parent, rows, current):
    raw = {row["JobIDRaw"] for row in rows}
    require(not any(identifier == parent or identifier.startswith(parent + "_") or identifier in raw
                    for identifier in current), "A live job/array allocation cannot reclaim its reservation")


def make_terminal_receipt(role, entry, sacct_text, *, live_job_ids, observed_at_utc, cluster="owl"):
    """Create complete scheduler evidence for exactly one terminal job/array."""
    require(cluster == "owl", "Fresh experiment accounting must identify the Owl cluster")
    observed, current = utc_timestamp(observed_at_utc), live_ids(live_job_ids)
    parent, expected = _expected(role, entry)
    rows = parse_sacct(sacct_text)
    require(len(rows) == len(expected) and {row["JobID"] for row in rows} == expected,
            "Missing, duplicate, partial or wrong-parent allocation accounting")
    require(len({row["JobIDRaw"] for row in rows}) == len(rows), "Duplicate raw allocation job IDs")
    _not_live(parent, rows, current)
    allocations = []
    for row in sorted(rows, key=lambda value: value["JobID"]):
        state = row["State"].split()[0].rstrip("+") if row["State"] else ""
        require(state in TERMINAL, "Only verified terminal states may replace a requested reservation")
        if role != "probes_and_scores":
            require(row["JobIDRaw"] == parent, "Nonarray raw allocation ID differs from its admitted parent")
        elapsed, count = int(row["ElapsedRaw"]), allocated_gpus(row["AllocTRES"])
        require(count > 0 or elapsed == 0, "Zero GPUs with positive elapsed time is contradictory for a GPU job")
        rounded = max(1, (elapsed + 59) // 60) * 60 if count else 0
        allocations.append(dict(job_id=row["JobID"], job_id_raw=row["JobIDRaw"], state=state,
            elapsed_seconds=elapsed, allocated_gpus=count, rounded_elapsed_seconds=rounded,
            charged_gpu_seconds=rounded * count, alloc_tres=row["AllocTRES"]))
    result = dict(schema=SCHEMA, policy=POLICY, role=role, job_id=parent, cluster=cluster,
        requested_gpu_seconds=seconds(entry.get("requested_gpu_hours", 0)),
        charged_gpu_seconds=sum(row["charged_gpu_seconds"] for row in allocations), allocations=allocations,
        source=dict(sacct_text=sacct_text, observed_at_utc=observed, live_job_ids=sorted(current),
            sacct_fields=list(FIELDS), allocation_only=True, live_ids_expanded=True))
    result["evidence_sha256"] = checksum(result)
    return result


def validate_receipt(role, entry, receipt, *, live_job_ids):
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA,
            "Unknown GPU accounting receipt schema")
    source = receipt.get("source", {})
    expected = make_terminal_receipt(role, entry, source.get("sacct_text"),
        live_job_ids=source.get("live_job_ids", ()), observed_at_utc=source.get("observed_at_utc"),
        cluster=receipt.get("cluster"))
    require(expected == receipt, "GPU accounting receipt is stale, tampered, partial or bound to another request")
    _not_live(str(entry["job_id"]), parse_sacct(source["sacct_text"]), live_ids(live_job_ids))
    return receipt["charged_gpu_seconds"]


def _archived_attempt(record):
    """Honor prior zero-admission evidence, never a zero declaration alone.

    The existing archive stores the exact empty Owl live/accounting outputs and
    cancellation of the origin CPU submitter. No new attestation format or
    retrospective rewrite of that request is needed. Unknown proof retains the
    full request; an accepted/uncertain identity is a ledger-integrity error.
    """
    attempt = record.get("attempt", {})
    require(isinstance(attempt, dict) and "requested_gpu_hours" in attempt and
            type(record.get("admitted_gpu_hours")) in (int, float) and record["admitted_gpu_hours"] == 0,
            "Unadmitted historical attempts require an explicit prior zero-admission reconciliation")
    require(attempt.get("status") in ("submission_failed", "submitting"),
            "Submitted, uncertain or unknown archived attempts cannot be treated as unadmitted")
    identity_fields = ("job_id", "job_id_raw", "job_ids", "array_job_id", "scheduler_response")
    require(not any(value.get(key) not in (None, "", [], ()) for value in (attempt, record)
                    for key in identity_fields),
            "An archived attempt with a known/possible scheduler identity cannot be charged zero")
    requested = seconds(attempt["requested_gpu_hours"])
    proof = record.get("proof", {})
    proven = (isinstance(proof, dict) and
              proof.get("live_owl_jobs") == "" and proof.get("owl_accounting_today") == "" and
              isinstance(proof.get("cancelled_origin_accounting"), str) and
              "CANCELLED" in proof["cancelled_origin_accounting"] and
              isinstance(record.get("reason"), str) and bool(record["reason"].strip()) and
              isinstance(record.get("reconciled_at_utc"), str))
    if proven:
        utc_timestamp(record["reconciled_at_utc"])
    return requested, (0 if proven else requested)


def conservative_budget(ledger, *, live_job_ids, additional_requested_gpu_hours=0):
    """Report closed actual-allocation ceilings plus full outstanding bounds.

    This does not authorize a submission. Use require_capacity before admission.
    Requested history can exceed 48 hours after completed allocations consume
    less than reserved; the allocated-plus-open bound must never exceed 48.
    """
    current = live_ids(live_job_ids)
    jobs = ledger.get("jobs", {})
    require(isinstance(jobs, dict), "Ledger jobs must be a role mapping")
    charges, requested = {}, 0
    for role, entry in jobs.items():
        require(isinstance(entry, dict) and "requested_gpu_hours" in entry,
                "Every ledger role must retain an explicit original requested GPU reservation")
        reservation = seconds(entry["requested_gpu_hours"])
        requested += reservation
        receipt = entry.get("gpu_accounting_receipt")
        charge = validate_receipt(role, entry, receipt, live_job_ids=current) if receipt is not None else reservation
        charges[role] = dict(requested_gpu_seconds=reservation, charged_gpu_seconds=charge,
            basis="verified terminal allocation accounting" if receipt is not None else "full requested reservation")
    unadmitted, archived_charges = 0, []
    for attempt in ledger.get("reconciled_unadmitted_attempts", []):
        requested_attempt, charged_attempt = _archived_attempt(attempt)
        unadmitted += requested_attempt
        archived_charges.append(dict(role=attempt.get("role"), requested_gpu_seconds=requested_attempt,
            charged_gpu_seconds=charged_attempt, basis="prior explicit zero-admission scheduler evidence"
            if charged_attempt == 0 else "full request; zero-admission proof incomplete"))
    total = sum(row["charged_gpu_seconds"] for row in charges.values()) + sum(
        row["charged_gpu_seconds"] for row in archived_charges)
    additional = seconds(additional_requested_gpu_hours)
    return dict(policy=POLICY, cap_gpu_seconds=CAP_GPU_SECONDS,
        cumulative_admitted_requested_gpu_seconds=requested,
        cumulative_all_attempt_requested_gpu_seconds=requested + unadmitted,
        conservative_committed_gpu_seconds=total, additional_requested_gpu_seconds=additional,
        projected_gpu_seconds=total + additional, remaining_gpu_seconds=max(0, CAP_GPU_SECONDS - total),
        within_cap=total + additional <= CAP_GPU_SECONDS, roles=charges, historical_attempt_charges=archived_charges)


def require_capacity(ledger, *, live_job_ids, additional_requested_gpu_hours):
    result = conservative_budget(ledger, live_job_ids=live_job_ids,
                                 additional_requested_gpu_hours=additional_requested_gpu_hours)
    require(result["within_cap"], "Verified allocation accounting plus full remaining reservations exceeds 48 GPU hours")
    return result


def reconcile_ledger(ledger, receipts_by_role, *, live_job_ids):
    """Return a new ledger and append-only audit; caller owns the ledger lock."""
    result, current = copy.deepcopy(ledger), live_ids(live_job_ids)
    require(isinstance(receipts_by_role, dict) and receipts_by_role, "Supply explicit terminal receipts by role")
    before = conservative_budget(ledger, live_job_ids=current)
    original_requests = {role: entry.get("requested_gpu_hours", 0) for role, entry in ledger.get("jobs", {}).items()}
    for role, receipt in receipts_by_role.items():
        require(role in result["jobs"], "Accounting role is absent from the original ledger")
        entry = result["jobs"][role]
        validate_receipt(role, entry, receipt, live_job_ids=current)
        previous = entry.get("gpu_accounting_receipt")
        require(previous is None or previous == receipt, "Existing terminal evidence cannot be silently replaced")
        entry["gpu_accounting_receipt"] = copy.deepcopy(receipt)
    require(original_requests == {role: entry.get("requested_gpu_hours", 0) for role, entry in result["jobs"].items()},
            "Original requested history must remain unchanged")
    after = conservative_budget(result, live_job_ids=current)
    audit = dict(schema="deadline-gpu-budget-reconciliation-v1", policy=POLICY,
        reconciled_at_utc=datetime.now(timezone.utc).isoformat(), original_requested_gpu_hours=original_requests,
        evidence=copy.deepcopy(receipts_by_role), live_job_ids=sorted(current), before=before, after=after)
    audit["audit_sha256"] = checksum(audit)
    result.setdefault("gpu_budget_reconciliations", []).append(audit)
    return result, audit
