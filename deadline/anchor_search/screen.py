"""Frozen source-only global-anchor search and single-candidate confirmation.

GPU generation and CPU selection run only in compute allocations. Observations
retain agreement flags and diagnostics, never teacher answers or generated text.
The lightweight admission validator replays sealed selection without model I/O.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from deadline.recovery.common import (digest, json_digest, read_json, require,
    require_compute, reserve_output, write_new_json)
from deadline.recovery.screen import canonical_rows, render_prompt, text_digest
from reasoning.data import boxed
from reasoning.features import repetition

SCHEMA = "deadline-anchor-search-screen-v1"
ROLES = ("original", "incumbent", "alpha0125", "alpha0375")
CANDIDATES = {"alpha0125": Fraction(1, 8), "alpha0375": Fraction(3, 8)}
ROWS, N, SEED = 192, 8, 20261021
MAX_GPU_SECONDS = 5400
PLAN = Path(__file__).with_name("plan.json")
HASH = re.compile(r"[0-9a-f]{64}")


def _load(output, verify_models=False):
    from .core import load_prepared
    prepared = load_prepared(output, verify_models=verify_models)
    require(prepared.get("schema") == "deadline-anchor-search-prepared-v1" and
            prepared.get("plan_sha256") == digest(PLAN), "Anchor preparation/plan changed")
    return prepared


def _protocol():
    protocol = read_json(PLAN)["source_protocol"]
    require(protocol["n"] == N and protocol["seed"] == SEED and
            protocol["max_tokens"] == 4096 and protocol["max_model_len"] == 8192,
            "Unexpected frozen source protocol")
    return protocol


def _identifiers(selected):
    require(isinstance(selected, list) and len(selected) == ROWS and
            all(isinstance(row, dict) and isinstance(row.get("id"), str) and
                HASH.fullmatch(row["id"]) for row in selected), "Invalid fixed split metadata")
    identifiers = [row["id"] for row in selected]
    require(len(set(identifiers)) == ROWS, "Duplicate split identities")
    return identifiers


def summarize_records(records, identifiers):
    """Validate all eight samples and compute official record-weighted proxies."""
    require(len(identifiers) == ROWS and len(set(identifiers)) == ROWS and
            isinstance(records, list) and len(records) == ROWS and
            all(isinstance(row, dict) for row in records) and
            [row.get("id") for row in records] == identifiers,
            "Observations must cover the complete ordered split exactly once")
    fields = {"id", "teacher_answer_sha256", "teacher_agreement", "finish_reason",
              "boxed_completion", "repetition_8gram", "output_tokens", "output_sha256"}
    for row in records:
        require(set(row) == fields and isinstance(row["teacher_answer_sha256"], str) and
                HASH.fullmatch(row["teacher_answer_sha256"]), "Unexpected observation fields/hash")
        require(all(isinstance(row[k], list) and len(row[k]) == N for k in fields
                    if k not in ("id", "teacher_answer_sha256")), "Missing sample observations")
        for i in range(N):
            require(type(row["teacher_agreement"][i]) is bool and
                    type(row["boxed_completion"][i]) is bool and
                    row["finish_reason"][i] in ("stop", "length") and
                    type(row["output_tokens"][i]) is int and 0 <= row["output_tokens"][i] <= 4096 and
                    type(row["repetition_8gram"][i]) in (int, float) and
                    math.isfinite(row["repetition_8gram"][i]) and
                    0 <= row["repetition_8gram"][i] <= 1 and
                    isinstance(row["output_sha256"][i], str) and
                    HASH.fullmatch(row["output_sha256"][i]), "Invalid per-sample diagnostics")
    correct = sum(sum(row["teacher_agreement"]) for row in records)
    solved = sum(any(row["teacher_agreement"]) for row in records)
    p1, p8 = correct / (ROWS * N), solved / ROWS
    return dict(rows=ROWS, samples=ROWS * N, agreeing_samples=correct, solved_questions=solved,
                pass_at_1=p1, pass_at_8=p8, harmonic_score=0. if p1 + p8 == 0 else 2*p1*p8/(p1+p8),
                length_finishes=sum(row["finish_reason"].count("length") for row in records),
                boxed_completions=sum(sum(row["boxed_completion"]) for row in records),
                output_tokens=sum(sum(row["output_tokens"]) for row in records),
                mean_repetition_8gram=math.fsum(x for row in records for x in row["repetition_8gram"])/(ROWS*N))


def _validate_metrics(value):
    require(isinstance(value, dict) and value.get("rows") == ROWS and
            value.get("samples") == ROWS * N, "Incomplete metric population")
    for key, maximum in (("agreeing_samples", ROWS*N), ("solved_questions", ROWS),
                         ("length_finishes", ROWS*N)):
        require(type(value.get(key)) is int and 0 <= value[key] <= maximum, "Invalid metric count")
    correct, solved = value["agreeing_samples"], value["solved_questions"]
    require(solved <= correct <= N*solved, "Inconsistent agreement/solved counts")
    p1, p8 = Fraction(correct, ROWS*N), Fraction(solved, ROWS)
    harmonic = 2*p1*p8/(p1+p8) if p1+p8 else Fraction(0)
    for key, expected in (("pass_at_1", p1), ("pass_at_8", p8), ("harmonic_score", harmonic)):
        require(type(value.get(key)) in (int, float) and math.isfinite(value[key]) and
                math.isclose(value[key], float(expected), rel_tol=0, abs_tol=1e-15),
                "Metrics differ from exact counts")
    require(type(value.get("mean_repetition_8gram")) in (int, float) and
            math.isfinite(value["mean_repetition_8gram"]) and
            0 <= value["mean_repetition_8gram"] <= 1, "Invalid repetition metric")
    return p1, p8, harmonic


def candidate_gate(candidate, original, incumbent):
    cp1, cp8, ch = _validate_metrics(candidate)
    op1, op8, _ = _validate_metrics(original)
    ip1, ip8, ih = _validate_metrics(incumbent)
    ceiling = max(original["length_finishes"], incumbent["length_finishes"]) + math.ceil(.01*ROWS*N)
    rep_ceiling = max(Fraction(str(original["mean_repetition_8gram"])),
                      Fraction(str(incumbent["mean_repetition_8gram"]))) + Fraction(1, 100)
    checks = dict(pass_at_1_floor=cp1 >= max(op1, ip1)-Fraction(1,100),
                  pass_at_8_floor=cp8 >= max(op8, ip8)-Fraction(2,100),
                  length_finish_ceiling=candidate["length_finishes"] <= ceiling,
                  mean_repetition_ceiling=Fraction(str(candidate["mean_repetition_8gram"])) <= rep_ceiling,
                  strictly_greater_harmonic_score=ch > ih)
    return dict(accepted=all(checks.values()), checks=checks,
                length_finish_ceiling=ceiling, mean_repetition_ceiling=float(rep_ceiling))


def choose_candidate(metrics):
    """Pure frozen search rule; exact count fractions decide score/tie order."""
    require(set(metrics) == set(ROLES), "All four complete search roles are required")
    for value in metrics.values():
        _validate_metrics(value)
    gates = {role: candidate_gate(metrics[role], metrics["original"], metrics["incumbent"])
             for role in CANDIDATES}
    passing = [role for role in CANDIDATES if gates[role]["accepted"]]
    winner = max(passing, key=lambda role: (_validate_metrics(metrics[role])[2],
                 _validate_metrics(metrics[role])[0], -CANDIDATES[role])) if passing else "incumbent"
    return dict(accepted=bool(passing), winner=winner, gates=gates)


def confirmation_gate(selection, metrics):
    require(selection.get("accepted") is True and selection.get("winner") in CANDIDATES,
            "No accepted new search candidate to confirm")
    winner = selection["winner"]
    require(set(metrics) == {"original", "incumbent", winner},
            "Confirmation requires only the single selected candidate and two controls")
    gate = candidate_gate(metrics[winner], metrics["original"], metrics["incumbent"])
    return dict(accepted=gate["accepted"], winner=winner if gate["accepted"] else "incumbent",
                proposed_winner=winner, gate=gate)


def _duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    require(match is not None, "Unverifiable scheduler duration")
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    return days*86400 + hours*3600 + minutes*60 + seconds


def scheduler_guard(output):
    compute = require_compute()
    job = os.environ["SLURM_JOB_ID"]
    require(re.fullmatch(r"[0-9]+", job) is not None, "Ordinary Slurm allocation required")
    text = subprocess.run(["scontrol", "--clusters=owl", "show", "job", "--oneliner", job],
                          check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", text))
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING",
            "Cannot verify running development allocation")
    limit = _duration(fields.get("TimeLimit", ""))
    tres = dict(item.split("=",1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    require(0 < limit <= MAX_GPU_SECONDS and tres.get("gres/gpu") == "1",
            "Development requires one GPU and at most 1.5 hours")
    seal = Path(output)/"gpu_allocation.json"
    if seal.exists():
        allocation = read_json(seal)
        require(allocation.get("job_id") == job and allocation.get("time_limit_seconds") == limit,
                "All source roles must share the same bounded GPU allocation")
        return allocation
    plan = read_json(PLAN)
    now = datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(plan["resources"]["deadline_utc"])
    require((deadline-now).total_seconds() >= 3600*plan["resources"]["development_start_reserve_hours"],
            "Insufficient frozen development/benchmark/closeout reserve")
    allocation = dict(compute=compute, job_id=job, time_limit_seconds=limit,
                      allocated_gpus=1,
                      scheduler_record_sha256=text_digest(text), started_at_utc=now.isoformat(),
                      start_reserve_hours=plan["resources"]["development_start_reserve_hours"],
                      deadline_utc=deadline.isoformat())
    write_new_json(seal, allocation)
    return allocation


def run(output, split, role):
    require(split in ("search", "confirmation") and role in ROLES, "Unknown split/model role")
    output = Path(output).resolve()
    prepared = _load(output, verify_models=(role,))
    selection = None
    if split == "confirmation":
        selection = _validate_selection(output)
        require(selection["accepted"] and role in ("original", "incumbent", selection["winner"]),
                "Only the selected search candidate may enter confirmation")
    allocation = scheduler_guard(output)
    selected = prepared["splits"][split]
    identifiers = _identifiers(selected)
    artifacts = Path(prepared["source_audit"]["artifacts"])
    require(digest(artifacts/"heldout.json") == prepared["source_audit"]["files"]["heldout.json"],
            "Actual frozen heldout source changed")
    heldout = canonical_rows(read_json(artifacts/"heldout.json"), 1000)
    role_output = reserve_output(output/"screen"/split/role)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "One BF16-capable assigned GPU required")
    from transformers import AutoTokenizer
    model = prepared["models"][role]
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
    prompts, teachers = [], []
    for item in selected:
        row = heldout[item["id"]]
        text, tokens = render_prompt(tokenizer, row)
        teacher = boxed(row["output"])
        require(len(tokens) == item["prompt_tokens"] <= 2048 and
                text_digest(text) == item["prompt_sha256"] and
                json_digest(tokens) == item["prompt_ids_sha256"] and teacher is not None and
                text_digest(teacher) == item["teacher_answer_sha256"],
                "Native prompt/teacher identity differs from prepared split")
        prompts.append(tokens)
        teachers.append(teacher)
    grader_root = Path(__file__).resolve().parents[2]/"eval/utils"
    require(all(digest(grader_root/name) == expected for name, expected in prepared["grader_sha256"].items()),
            "Frozen rule grader changed")
    sys.path.insert(0, str(grader_root.parent))
    from utils.parser import extract_answer
    from utils.grader import check_is_correct
    from reasoning.grader_preflight import check_grader
    check_grader()
    prop = torch.cuda.get_device_properties(0)
    hardware = dict(name=prop.name, total_memory=prop.total_memory,
                    capability=list(torch.cuda.get_device_capability(0)))
    manifest = dict(schema=SCHEMA, phase="run", role=role, split=split, protocol=_protocol(),
                    prepared_sha256=digest(output/"prepared.json"), plan_sha256=digest(PLAN),
                    code_sha256=prepared["code_sha256"], model=model, source_audit=prepared["source_audit"],
                    ordered_ids=identifiers, allocation=allocation, hardware=hardware,
                    selection_sha256=digest(output/"selection.json") if selection else None,
                    versions={name: importlib.metadata.version(name) for name in
                              ("vllm","transformers","torch","sympy","antlr4-python3-runtime","latex2sympy2")})
    write_new_json(role_output/"manifest.json", manifest)
    from vllm import LLM, SamplingParams
    llm = LLM(model=model["path"], dtype="bfloat16", max_model_len=8192, seed=SEED,
              tensor_parallel_size=1, gpu_memory_utilization=.85, trust_remote_code=False,
              generation_config="vllm", max_num_seqs=128)
    params = [SamplingParams(n=N, temperature=.6, top_p=.95, max_tokens=4096,
              seed=(SEED+int(identifier[:8],16)) % (2**31), top_k=-1, min_p=0., repetition_penalty=1.)
              for identifier in identifiers]
    generated = llm.generate([{"prompt_token_ids": tokens} for tokens in prompts], params)
    require(len(generated) == ROWS, "Incomplete source generation batch")
    records = []
    for item, tokens, teacher, result in zip(selected, prompts, teachers, generated, strict=True):
        require(result.prompt_token_ids == tokens and len(result.outputs) == N,
                "Generated requests differ from fixed eight-sample ordered prompts")
        completions = sorted(result.outputs, key=lambda value: value.index)
        require([c.index for c in completions] == list(range(N)), "Duplicate/missing sample indices")
        records.append(dict(id=item["id"], teacher_answer_sha256=item["teacher_answer_sha256"],
            teacher_agreement=[bool(check_is_correct(extract_answer(c.text),teacher)) for c in completions],
            finish_reason=[c.finish_reason for c in completions],
            boxed_completion=[boxed(c.text) is not None for c in completions],
            repetition_8gram=[repetition(c.text) for c in completions],
            output_tokens=[len(c.token_ids) for c in completions],
            output_sha256=[text_digest(c.text) for c in completions]))
    metrics = summarize_records(records, identifiers)
    write_new_json(role_output/"observations.json", records)
    receipt = dict(schema=SCHEMA, complete=True, role=role, split=split, metrics=metrics,
                   manifest_sha256=digest(role_output/"manifest.json"),
                   observations_sha256=digest(role_output/"observations.json"))
    write_new_json(role_output/"complete.json", receipt)
    return receipt


def _read_stage(output, prepared, split, roles):
    selected = prepared["splits"][split]
    identifiers = _identifiers(selected)
    directory = output/"screen"/split
    require(directory.is_dir() and {p.name for p in directory.iterdir()} == set(roles),
            "Missing or unexpected model-role outputs for fixed stage")
    allocation = read_json(output/"gpu_allocation.json")
    metrics, proofs, environments = {}, {}, []
    for role in roles:
        path = directory/role
        receipt, manifest = read_json(path/"complete.json"), read_json(path/"manifest.json")
        require(receipt.get("schema") == SCHEMA and receipt.get("complete") is True and
                receipt.get("role") == role and receipt.get("split") == split and
                manifest.get("schema") == SCHEMA and manifest.get("phase") == "run" and
                manifest.get("role") == role and manifest.get("split") == split and
                manifest.get("protocol") == _protocol() and manifest.get("model") == prepared["models"][role] and
                manifest.get("prepared_sha256") == digest(output/"prepared.json") and
                manifest.get("plan_sha256") == digest(PLAN) and
                manifest.get("code_sha256") == prepared["code_sha256"] and
                manifest.get("source_audit") == prepared["source_audit"] and
                manifest.get("ordered_ids") == identifiers and manifest.get("allocation") == allocation and
                receipt.get("manifest_sha256") == digest(path/"manifest.json") and
                receipt.get("observations_sha256") == digest(path/"observations.json"),
                "Incomplete, changed or incompatible source model-role evidence")
        require(manifest.get("selection_sha256") ==
                (digest(output/"selection.json") if split == "confirmation" else None),
                "Confirmation is not bound to the frozen search winner")
        records = read_json(path/"observations.json")
        metrics[role] = summarize_records(records, identifiers)
        require([r["teacher_answer_sha256"] for r in records] ==
                [r["teacher_answer_sha256"] for r in selected] and
                json_digest(metrics[role]) == json_digest(receipt.get("metrics")),
                "Teacher identities/aggregate metrics differ from sealed observations")
        proofs[role] = dict(complete_sha256=digest(path/"complete.json"),
                            manifest_sha256=digest(path/"manifest.json"),
                            observations_sha256=digest(path/"observations.json"))
        environments.append((manifest.get("versions"), manifest.get("hardware")))
    require(all(v == environments[0] for v in environments) and
            all(isinstance(v, dict) and v for v in environments[0]),
            "Model roles used different/missing runtime environments")
    require(isinstance(allocation.get("job_id"), str) and re.fullmatch(r"[0-9]+",allocation["job_id"]) and
            type(allocation.get("time_limit_seconds")) is int and
            0 < allocation["time_limit_seconds"] <= MAX_GPU_SECONDS and
            allocation.get("allocated_gpus") == 1 and
            isinstance(allocation.get("scheduler_record_sha256"), str) and
            HASH.fullmatch(allocation["scheduler_record_sha256"]) and
            allocation.get("deadline_utc") == datetime.fromisoformat(
                read_json(PLAN)["resources"]["deadline_utc"]).isoformat() and
            allocation.get("start_reserve_hours") == read_json(PLAN)["resources"]["development_start_reserve_hours"],
            "Missing bounded single-allocation provenance")
    deadline = datetime.fromisoformat(allocation["deadline_utc"])
    started = datetime.fromisoformat(allocation["started_at_utc"])
    require(started.tzinfo is not None and
            (deadline-started).total_seconds() >= 3600*allocation["start_reserve_hours"],
            "Development allocation lacked the fixed deadline reserve")
    return metrics, proofs, dict(versions=environments[0][0], hardware=environments[0][1])


def _selection_payload(output):
    prepared = _load(output)
    metrics, proofs, environment = _read_stage(output, prepared, "search", ROLES)
    choice = choose_candidate(metrics)
    return dict(schema=SCHEMA, phase="selection", complete=True, **choice,
                model_path=prepared["models"][choice["winner"]]["path"], metrics=metrics,
                prepared_sha256=digest(output/"prepared.json"), plan_sha256=digest(PLAN),
                role_proofs=proofs, runtime_environment=environment,
                exploratory=True, teacher_agreement_is_not_truth=True,
                benchmark_based_candidate_selection=False)


def select(output):
    require_compute(cpu_only=True)
    output = Path(output).resolve()
    result = _selection_payload(output)
    write_new_json(output/"selection.json", result)
    return result


def _validate_selection(output):
    expected = _selection_payload(output)
    require(json_digest(read_json(output/"selection.json")) == json_digest(expected),
            "Stored selection differs from sealed complete search evidence")
    return expected


def _decision_payload(output):
    prepared = _load(output)
    selection = _validate_selection(output)
    require(selection["accepted"], "No new search winner; retain incumbent without confirmation")
    metrics, proofs, environment = _read_stage(output, prepared, "confirmation",
                                  ("original","incumbent",selection["winner"]))
    require(environment == selection["runtime_environment"],
            "Search and confirmation runtime environments differ")
    result = confirmation_gate(selection, metrics)
    return dict(schema=SCHEMA, phase="decision", complete=True, **result,
                model_path=prepared["models"][result["winner"]]["path"], metrics=metrics,
                prepared_sha256=digest(output/"prepared.json"), plan_sha256=digest(PLAN),
                selection_sha256=digest(output/"selection.json"), role_proofs=proofs,
                source_audit=prepared["source_audit"], models=prepared["models"],
                runtime_environment=environment,
                exploratory=True, posthoc_after_primary_outcomes=True,
                teacher_agreement_is_not_truth=True, no_retry_other_candidate=True,
                benchmark_based_candidate_selection=False)


def confirm(output):
    require_compute(cpu_only=True)
    output = Path(output).resolve()
    result = _decision_payload(output)
    write_new_json(output/"decision.json", result)
    return result


def validate_decision(output):
    """Lightweight read-only admission; a future evaluator must hash model bytes."""
    output = Path(output).resolve()
    expected = _decision_payload(output)
    require(json_digest(read_json(output/"decision.json")) == json_digest(expected),
            "Stored decision differs from sealed confirmation evidence")
    require(expected["accepted"] and expected["winner"] in CANDIDATES,
            "No accepted new candidate; benchmark admission forbidden")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run","select","confirm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("search","confirmation"))
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args()
    if args.phase == "run":
        require(args.split is not None and args.role is not None, "run requires split and role")
        result = run(args.output,args.split,args.role)
    else:
        require(args.split is None and args.role is None, "Selection takes no role or split override")
        result = (select if args.phase == "select" else confirm)(args.output)
    print(json.dumps({k: result.get(k) for k in ("schema","complete","accepted","winner","model_path")}), flush=True)


if __name__ == "__main__":
    main()
