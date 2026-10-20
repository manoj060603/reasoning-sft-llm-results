"""One frozen, source-heldout recovery screen; never an assignment benchmark.

Prepare in a CPU allocation, then run base/advanced/recovery sequentially in ONE
GPU allocation with a scheduler limit of at most 45 minutes. No source solutions,
teacher answers, decoded answers, or generations are written by this module.
The comparison measures agreement with source teachers, not ground truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from reasoning.data import SYSTEM, boxed, canonical, prompt_id, read_rows
from reasoning.features import repetition
from deadline.recovery.common import (digest, json_digest, read_json,
                                      require, require_compute, reserve_output,
                                      verify_model_compatibility, write_new_json)

ROLES = ("base", "advanced", "recovery")
SEED = 20261020
SIZE = 64
MAX_GPU_SECONDS = 45 * 60
DEADLINE = datetime(2026, 10, 21, 3, 59, tzinfo=timezone.utc)
RESERVE_SECONDS = 3 * 3600
SOURCE_SHA256 = "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8"
SCHEMA = "deadline-source-heldout-recovery-screen-v1"
PROTOCOL = dict(n=1, temperature=.6, top_p=.95, seed=SEED,
                max_tokens=4096, max_model_len=8192, dtype="bfloat16",
                max_num_seqs=64, gpu_memory_utilization=.85,
                tensor_parallel_size=1, generation_config="vllm",
                top_k=-1, min_p=0., repetition_penalty=1., system=SYSTEM,
                prompt_token_limit=2048, rows=SIZE,
                prompt_input="explicit native token IDs; no additional special tokens",
                selection="sha256(seed:source-heldout-screen:id), ascending",
                teacher_metric="source teacher boxed-answer agreement; not truth",
                gate=dict(agreement_ge_advanced=True,
                          agreement_ge_base_minus=2,
                          length_finishes_max_fraction_of_advanced=.5))


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_ids(values, count=None):
    require(isinstance(values, list) and values and
            all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in values),
            "Expected a nonempty list of canonical SHA256 prompt IDs")
    require(len(values) == len(set(values)), "Duplicate prompt IDs")
    require(count is None or len(values) == count, "Unexpected prompt ID count")
    return values


def canonical_rows(rows, count=None):
    require(isinstance(rows, list) and (count is None or len(rows) == count),
            "Unexpected canonical source row count")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and
                all(isinstance(row.get(k), str) for k in
                    ("id", "instruction", "input", "output", "system")),
                "Malformed canonical source row")
        require(row["input"] == "" and row["system"] == SYSTEM and bool(row["instruction"]) and
                row["instruction"] == row["instruction"].strip() and
                row["output"] == row["output"].strip() and bool(row["output"]),
                "Source row is not an unchanged canonical example")
        require(row["id"] == prompt_id(row["instruction"]) and row["id"] not in result,
                "Duplicate or incorrect canonical source ID")
        result[row["id"]] = row
    return result


def check_disjoint(heldout_ids, populations):
    heldout_ids = set(strict_ids(list(heldout_ids)))
    require(isinstance(populations, dict) and bool(populations),
            "Missing training/probe populations")
    for name, identifiers in populations.items():
        identifiers = set(strict_ids(list(identifiers)))
        require(not heldout_ids.intersection(identifiers),
                "Source heldout overlaps training/probe population: " + name)
    return {name: len(ids) for name, ids in populations.items()}


def render_prompt(tokenizer, row):
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["instruction"]}],
        tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    require(isinstance(text, str) and isinstance(tokens, list) and tokens and
            all(type(x) is int and x >= 0 for x in tokens), "Invalid native prompt serialization")
    return text, tokens


def choose_screen(rows, tokenizer):
    """No losses, generations, benchmark scores, or correctness enter selection."""
    by_id = canonical_rows(rows, 1000)
    candidates = []
    for identifier, row in by_id.items():
        answer = boxed(row["output"])
        if answer is None:
            continue
        text, tokens = render_prompt(tokenizer, row)
        if len(tokens) <= PROTOCOL["prompt_token_limit"]:
            candidates.append(dict(id=identifier, prompt_tokens=len(tokens),
                prompt_sha256=text_digest(text), prompt_ids_sha256=json_digest(tokens),
                teacher_answer_sha256=text_digest(answer)))
    require(len(candidates) >= SIZE, "Fewer than 64 eligible source-heldout prompts")
    candidates.sort(key=lambda row: text_digest(
        f"{SEED}:source-heldout-screen:{row['id']}"))
    return candidates[:SIZE], len(candidates)


def _safe_file_map(path, files, verify=True):
    require(isinstance(files, dict) and bool(files), "Missing model file inventory")
    require(all(isinstance(k, str) and Path(k).name == k and
                isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in files.items()), "Invalid model file inventory")
    require({"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(files),
            "Missing model/tokenizer identity files")
    require(any(k.endswith(".safetensors") for k in files), "Missing full-model weights")
    require(not any(k.endswith(".bin") for k in files), "Pickle model weights are not permitted")
    path = Path(path).resolve()
    require(path.is_dir(), "Local model directory is missing")
    # HF Trainer's training_args.bin is never loaded by vLLM or part of model
    # identity. It is the sole permitted unlisted .bin sidecar. Bind licenses
    # and notices whenever the merge receipt inventories them.
    relevant = {p.name for p in path.iterdir() if p.is_file() and p.name != "training_args.bin" and
                (p.suffix in (".json", ".safetensors", ".bin", ".model", ".jinja")
                 or p.name == "merges.txt" or p.name in files)}
    require(relevant == set(files), "Model inventory has missing or unexpected files")
    if verify:
        for name, expected in files.items():
            before = (path / name).stat()
            require(digest(path / name) == expected, "Model bytes differ from merge receipt: " + name)
            after = (path / name).stat()
            require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                    (after.st_size, after.st_mtime_ns, after.st_ino),
                    "Model changed while hashing")
    return dict(path=str(path), files=files)


def model_inputs(merge_receipt, verify_roles=ROLES):
    receipt = read_json(merge_receipt)
    require(receipt.get("schema") == "deadline-base-anchor-merge-v1" and
            receipt.get("complete") is True and receipt.get("alpha_advanced") == .25,
            "A complete fixed-alpha recovery merge receipt is required")
    require(receipt.get("output", {}).get("all_finite") is True and
            receipt["output"].get("finite_fp32_tensors") == 434,
            "Recovery weights lack a complete finite FP32 scan")
    require(receipt.get("primary_plan_sha256") == digest(Path(__file__).resolve().parents[1] / "experiment_plan.json") and
            receipt.get("recovery_plan_sha256") == digest(Path(__file__).with_name("plan.json")),
            "Frozen primary/recovery plan identity changed")
    models = {role: _safe_file_map(receipt["inputs"][role]["path"],
                                  receipt["inputs"][role]["files"], verify=role in verify_roles)
              for role in ("base", "advanced")}
    models["recovery"] = _safe_file_map(receipt["output"]["path"], receipt["output"]["files"],
                                       verify="recovery" in verify_roles)
    require(len({m["path"] for m in models.values()}) == 3, "Models must have distinct local directories")
    advanced = receipt["inputs"]["advanced"]
    run = Path(advanced["path"]).resolve().parent
    for key, name in (("training_manifest_sha256", "manifest.json"),
                      ("training_receipt_sha256", "complete.json")):
        require(advanced.get(key) == digest(run / name), "Advanced training identity changed")
    return receipt, models


def audit_sources(artifacts, models):
    """Validate real primary artifacts, including original source target bytes."""
    artifacts = Path(artifacts).resolve()
    evidence = {}

    def read(relative):
        path = artifacts / relative
        evidence[relative] = digest(path)
        return read_json(path)

    heldout = read("heldout.json")
    hmap = canonical_rows(heldout, 1000)
    pool = canonical_rows(read("pool.json"))
    preparation = read("preparation_complete.json")
    require(preparation.get("source_sha256") == SOURCE_SHA256, "Unexpected original source identity")
    identity = read("model_identity.json")
    frozen_files = identity.get("files", {})
    require(Path(identity.get("path", "")).resolve() == Path(models["base"]["path"]) and
            frozen_files and all(models["base"]["files"].get(name) == info.get("sha256")
                                 for name, info in frozen_files.items()),
            "Screen base is not the actual original model used in source preparation")
    source = artifacts / "source/data/acereason11_100k.json"
    require(digest(source) == SOURCE_SHA256, "Original source bytes differ from the frozen source")
    evidence["source/data/acereason11_100k.json"] = SOURCE_SHA256
    originals = {}
    for raw in read_rows(source):
        try:
            row = canonical(raw)
        except (ValueError, TypeError, KeyError):
            continue
        if row["id"] in hmap and row["id"] not in originals:
            originals[row["id"]] = row
    require(originals == hmap, "Heldout rows are not the original canonical source pairs")
    transfer = read("transfer_meta.json")
    reservations = transfer.get("probe_training_ids", {})
    require(set(reservations) == {"probeA", "probeB"}, "Missing actual probe reservations")
    populations = {"entire_prepared_pool": list(pool)}
    runs = artifacts.parent / "runs"
    for name in ("probeA", "probeB"):
        identifiers = strict_ids(read(name + "_ids.json"), 512)
        require(identifiers == reservations[name], "Probe reservation files disagree")
        receipt_path, manifest_path = runs / name / "probe_receipt.json", runs / name / "run_manifest.json"
        probe, manifest = read_json(receipt_path), read_json(manifest_path)
        evidence[str(receipt_path)] = digest(receipt_path)
        evidence[str(manifest_path)] = digest(manifest_path)
        require(probe.get("complete") is True and probe.get("training_ids") == identifiers and
                manifest.get("training_ids") == identifiers and
                probe.get("base") == identity and manifest.get("base") == identity and
                probe.get("completed_updates") == manifest.get("steps") and
                probe.get("manifest_sha256") == json_digest(manifest),
                "Actual probe receipt does not confirm reserved training IDs")
        populations[name] = identifiers
    require(not set(populations["probeA"]).intersection(populations["probeB"]),
            "Probe reservations overlap")
    selection = read("selected/selection_manifest.json")
    selection_sources = selection.get("build_source_file_sha256", {})
    require(selection_sources.get("artifacts/heldout.json") == evidence["heldout.json"] and
            selection_sources.get("artifacts/pool.json") == evidence["pool.json"],
            "Heldout/source pool differs from the actual pretraining selection seal")
    for role in ("random", "advanced"):
        identifiers = strict_ids(selection["datasets"][role]["selected_ids"], 15000)
        require(read("training/" + role + "/selected_ids.json") == identifiers,
                "Actual training IDs differ from selection")
        selected = read("selected/" + role + ".json")
        training = read("training/" + role + "/dataset.json")
        require(selected == training and len(selected) == 15000, "Original training exports differ")
        require(all(row == {key: pool[identifier][key] for key in
                            ("instruction", "input", "output", "system")}
                    for row, identifier in zip(selected, identifiers)),
                "Training exports differ from original canonical pool")
        meta = read("training/" + role + "/cache_meta.json")
        require(meta.get("selected_ids_sha256") == evidence["training/" + role + "/selected_ids.json"] and
                meta.get("dataset_sha256") == evidence["training/" + role + "/dataset.json"],
                "Training export metadata is stale")
        if role == "advanced":
            require(meta.get("selection_source_sha256") == evidence["selected/selection_manifest.json"],
                    "Advanced training cache does not bind the actual selection manifest")
        manifest_path, receipt_path = runs / role / "manifest.json", runs / role / "complete.json"
        manifest, receipt = read_json(manifest_path), read_json(receipt_path)
        evidence[str(manifest_path)] = digest(manifest_path)
        evidence[str(receipt_path)] = digest(receipt_path)
        require(manifest.get("cache", {}).get("id_order_sha256") == json_digest(identifiers) and
                all(manifest.get("base_identity", {}).get(name) == info.get("sha256")
                    for name, info in frozen_files.items()) and
                manifest.get("cache", {}).get("cache_meta_sha256") ==
                evidence["training/" + role + "/cache_meta.json"] and
                receipt.get("binding_sha256") == json_digest(manifest) and
                receipt.get("unique_examples") == 15000 and receipt.get("example_presentations") == 15000,
                "Actual primary training does not bind these exact IDs")
        if role == "advanced":
            require(Path(receipt["model"]).resolve() == Path(models["advanced"]["path"]),
                    "Screen advanced model is not the actual advanced training output")
        populations[role] = identifiers
    require(read("random_ids.json") == populations["random"], "Frozen random IDs differ")
    counts = check_disjoint(list(hmap), populations)
    require(all(set(ids).issubset(pool) for ids in populations.values()),
            "Reserved/training IDs are outside the original prepared pool")
    return heldout, dict(artifacts=str(artifacts), files=evidence, population_counts=counts,
                        disjointness="exact normalized canonical prompt identities; not semantic equivalence")


def deadline_guard(remaining_seconds=MAX_GPU_SECONDS, now=None):
    now = now or datetime.now(timezone.utc)
    require((DEADLINE - now).total_seconds() >= remaining_seconds + RESERVE_SECONDS,
            "Insufficient time for screen and explicit three-hour closeout reserve")


def scheduler_guard():
    compute = require_compute()
    job = os.environ["SLURM_JOB_ID"]
    require(re.fullmatch(r"[0-9]+", job) is not None, "An ordinary Slurm allocation is required")
    output = subprocess.run(["scontrol", "--clusters=owl", "show", "job", "--oneliner", job],
                            check=True, text=True, capture_output=True, timeout=20).stdout.strip()
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", output))
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING",
            "Cannot verify this running Slurm allocation")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", fields.get("TimeLimit", ""))
    require(match is not None, "Cannot verify allocation time limit")
    days, hours, minutes, seconds = [int(x or 0) for x in match.groups()]
    limit = days * 86400 + hours * 3600 + minutes * 60 + seconds
    require(0 < limit <= MAX_GPU_SECONDS, "Screen allocation must be at most 45 minutes")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    require(tres.get("gres/gpu") == "1", "Screen must use exactly one allocated GPU")
    deadline_guard(limit)
    return dict(compute=compute, job_id=job, time_limit_seconds=limit,
                scheduler_record_sha256=text_digest(output), reserve_seconds=RESERVE_SECONDS,
                deadline_utc=DEADLINE.isoformat())


def prepare(artifacts, merge_receipt, output):
    compute = require_compute(cpu_only=True)
    deadline_guard()
    _, models = model_inputs(merge_receipt)
    heldout, source_audit = audit_sources(artifacts, models)
    compatibility = {role: verify_model_compatibility(models["base"]["path"], models[role]["path"])
                     for role in ("advanced", "recovery")}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models["base"]["path"], local_files_only=True,
                                              trust_remote_code=False)
    selected, eligible = choose_screen(heldout, tokenizer)
    # Seal IDs before any generation, without copying teacher solutions/prompts.
    output = Path(output).resolve()
    reserve_output(output)
    write_new_json(output / "selected_ids.json", [row["id"] for row in selected])
    write_new_json(output / "selected.json", selected)
    grader_root = Path(__file__).resolve().parents[2] / "eval/utils"
    receipt = dict(schema=SCHEMA, phase="prepare", complete=True,
                   exploratory=True, post_primary_observation=True, primary_runs_immutable=True,
                   protocol=PROTOCOL, models=models, compatibility=compatibility, eligible_rows=eligible,
                   source_audit=source_audit, merge_receipt=str(Path(merge_receipt).resolve()),
                   merge_receipt_sha256=digest(merge_receipt), compute=compute,
                   selected_ids_sha256=digest(output / "selected_ids.json"),
                   selected_sha256=digest(output / "selected.json"),
                   code_sha256=digest(__file__),
                   grader_sha256={name: digest(grader_root / name) for name in
                                  ("grader.py", "parser.py", "math_normalization.py")})
    write_new_json(output / "prepared.json", receipt)
    return receipt


def load_prepared(output):
    output = Path(output).resolve()
    prepared = read_json(output / "prepared.json")
    require(prepared.get("schema") == SCHEMA and prepared.get("phase") == "prepare" and
            prepared.get("complete") is True and
            json_digest(prepared.get("protocol")) == json_digest(PROTOCOL) and
            prepared.get("code_sha256") == digest(__file__), "Screen preparation/protocol/code changed")
    require(prepared.get("selected_sha256") == digest(output / "selected.json") and
            prepared.get("selected_ids_sha256") == digest(output / "selected_ids.json"),
            "Frozen screen selection changed")
    selected = read_json(output / "selected.json")
    identifiers = strict_ids(read_json(output / "selected_ids.json"), SIZE)
    require(isinstance(selected, list) and [row.get("id") for row in selected] == identifiers,
            "Frozen screen records differ from selected IDs")
    return prepared, selected


def summarize_records(records, identifiers):
    require(isinstance(records, list) and [r.get("id") for r in records] == identifiers,
            "Screen observations must cover the exact fixed IDs once")
    for row in records:
        require(set(row) == {"id", "teacher_answer_sha256", "teacher_agreement", "finish_reason",
                             "boxed_completion", "repetition_8gram", "output_tokens", "output_sha256"},
                "Unexpected screen fields; raw answers/solutions must not be stored")
        require(type(row["teacher_agreement"]) is bool and type(row["boxed_completion"]) is bool and
                row["finish_reason"] in ("stop", "length") and
                type(row["output_tokens"]) is int and 0 <= row["output_tokens"] <= 4096 and
                type(row["repetition_8gram"]) in (int, float) and
                math.isfinite(row["repetition_8gram"]) and 0 <= row["repetition_8gram"] <= 1,
                "Malformed or incomplete screen diagnostics")
        require(all(isinstance(row[k], str) and re.fullmatch(r"[0-9a-f]{64}", row[k])
                    for k in ("teacher_answer_sha256", "output_sha256")), "Invalid diagnostic hash")
    return dict(rows=len(records), teacher_agreement=sum(r["teacher_agreement"] for r in records),
                length_finishes=sum(r["finish_reason"] == "length" for r in records),
                boxed_completions=sum(r["boxed_completion"] for r in records),
                output_tokens=sum(r["output_tokens"] for r in records),
                mean_repetition_8gram=sum(r["repetition_8gram"] for r in records) / len(records))


def fixed_gate(metrics):
    require(set(metrics) == set(ROLES), "All three fixed model roles are required")
    for value in metrics.values():
        require(type(value.get("rows")) is int and value["rows"] == SIZE and
                all(type(value.get(k)) is int and 0 <= value[k] <= SIZE
                for k in ("teacher_agreement", "length_finishes")), "Invalid gate counts")
    base, advanced, recovery = (metrics[role] for role in ROLES)
    gates = dict(agreement_ge_advanced=recovery["teacher_agreement"] >= advanced["teacher_agreement"],
                 agreement_ge_base_minus_two=recovery["teacher_agreement"] >= base["teacher_agreement"] - 2,
                 length_finishes_at_most_half_advanced=
                     2 * recovery["length_finishes"] <= advanced["length_finishes"])
    return dict(accepted=all(gates.values()), checks=gates)


def run(output, role):
    require(role in ROLES, "Unsupported screen role")
    allocation = scheduler_guard()
    output = Path(output).resolve()
    prepared, selected = load_prepared(output)
    require(digest(prepared["merge_receipt"]) == prepared["merge_receipt_sha256"], "Merge receipt changed")
    _, models = model_inputs(prepared["merge_receipt"], verify_roles=(role,))
    require(models == prepared["models"], "Screen model identity changed")
    # Preparation seals every actual source/reservation/export/training artifact.
    artifacts = Path(prepared["source_audit"]["artifacts"])
    for name, expected in prepared["source_audit"]["files"].items():
        require(digest(artifacts / name) == expected, "Frozen screen source/training evidence changed")
    heldout = canonical_rows(read_json(artifacts / "heldout.json"), 1000)
    bound_allocation = output / "gpu_allocation.json"
    if bound_allocation.exists():
        require(read_json(bound_allocation)["job_id"] == allocation["job_id"],
                "All model roles must share one <=45-minute GPU allocation")
    else:
        write_new_json(bound_allocation, allocation)
    role_output = output / role
    reserve_output(role_output)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.is_bf16_supported(), "One BF16-capable assigned GPU is required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models[role]["path"], local_files_only=True,
                                              trust_remote_code=False)
    prompt_ids, teachers = [], []
    for item in selected:
        row = heldout[item["id"]]
        prompt, tokens = render_prompt(tokenizer, row)
        answer = boxed(row["output"])
        require(len(tokens) == item["prompt_tokens"] <= 2048 and
                text_digest(prompt) == item["prompt_sha256"] and
                json_digest(tokens) == item["prompt_ids_sha256"] and answer is not None and
                text_digest(answer) == item["teacher_answer_sha256"],
                "Screen tokenizer/prompt/teacher identity changed")
        prompt_ids.append(tokens)
        teachers.append(answer)
    grader_root = Path(__file__).resolve().parents[2] / "eval/utils"
    require(all(digest(grader_root / name) == expected for name, expected in
                prepared["grader_sha256"].items()), "Assignment rule grader changed")
    sys.path.insert(0, str(grader_root.parent))
    from utils.parser import extract_answer
    from utils.grader import check_is_correct
    from reasoning.grader_preflight import check_grader
    check_grader()
    from vllm import LLM, SamplingParams
    manifest = dict(schema=SCHEMA, phase="run", role=role, protocol=PROTOCOL,
                    prepared_sha256=digest(output / "prepared.json"), model=models[role],
                    allocation=allocation,
                    versions={name: importlib.metadata.version(name) for name in
                              ("vllm", "transformers", "torch", "sympy", "antlr4-python3-runtime", "latex2sympy2")})
    write_new_json(role_output / "manifest.json", manifest)
    llm = LLM(model=models[role]["path"], dtype="bfloat16", max_model_len=8192,
              seed=SEED, tensor_parallel_size=1, gpu_memory_utilization=.85,
              trust_remote_code=False, generation_config="vllm", max_num_seqs=64)
    params = SamplingParams(n=1, temperature=.6, top_p=.95, max_tokens=4096,
                            seed=SEED, top_k=-1, min_p=0., repetition_penalty=1.)
    generated = llm.generate([{"prompt_token_ids": tokens} for tokens in prompt_ids], params)
    require(len(generated) == SIZE, "Incomplete source-heldout generation batch")
    records = []
    for item, tokens, teacher, result in zip(selected, prompt_ids, teachers, generated, strict=True):
        require(result.prompt_token_ids == tokens and len(result.outputs) == 1,
                "Generation does not match the fixed single-sample input order")
        completion = result.outputs[0]
        text = completion.text
        records.append(dict(id=item["id"], teacher_answer_sha256=item["teacher_answer_sha256"],
            teacher_agreement=bool(check_is_correct(extract_answer(text), teacher)),
            finish_reason=completion.finish_reason, boxed_completion=boxed(text) is not None,
            repetition_8gram=repetition(text), output_tokens=len(completion.token_ids),
            output_sha256=text_digest(text)))
    metrics = summarize_records(records, [r["id"] for r in selected])
    write_new_json(role_output / "observations.json", records)
    receipt = dict(schema=SCHEMA, complete=True, role=role, metrics=metrics,
                   manifest_sha256=digest(role_output / "manifest.json"),
                   observations_sha256=digest(role_output / "observations.json"))
    write_new_json(role_output / "complete.json", receipt)
    return receipt


def _summary_payload(output):
    """Recompute the substantive summary using only small receipt/record files."""
    output = Path(output).resolve()
    prepared, selected = load_prepared(output)
    require(digest(prepared["merge_receipt"]) == prepared["merge_receipt_sha256"],
            "Screen merge receipt changed")
    # Check current plan bindings, declared identities and inventory presence.
    # This is intentionally not another multi-GB weight hash pass. A subsequent
    # evaluator must independently verify its actual loaded model bytes.
    _, models = model_inputs(prepared["merge_receipt"], verify_roles=())
    require(models == prepared["models"], "Screen prepared model identity changed")
    identifiers = [row["id"] for row in selected]
    metrics, receipts, job_ids, versions = {}, {}, set(), []
    for role in ROLES:
        directory = output / role
        receipt, manifest = read_json(directory / "complete.json"), read_json(directory / "manifest.json")
        require(receipt.get("schema") == SCHEMA and receipt.get("complete") is True and
                receipt.get("role") == role and manifest.get("schema") == SCHEMA and
                manifest.get("phase") == "run" and manifest.get("role") == role and
                json_digest(manifest.get("protocol")) == json_digest(PROTOCOL) and
                manifest.get("model") == prepared["models"][role] and
                manifest.get("prepared_sha256") == digest(output / "prepared.json") and
                receipt.get("manifest_sha256") == digest(directory / "manifest.json") and
                receipt.get("observations_sha256") == digest(directory / "observations.json"),
                "Incomplete, stale, or incompatible model-role screen receipt")
        records = read_json(directory / "observations.json")
        require([r["teacher_answer_sha256"] for r in records] ==
                [r["teacher_answer_sha256"] for r in selected], "Teacher answer identity changed")
        metrics[role] = summarize_records(records, identifiers)
        require(json_digest(metrics[role]) == json_digest(receipt["metrics"]),
                "Screen aggregate differs from observations")
        allocation = manifest["allocation"]
        require(isinstance(allocation.get("job_id"), str) and
                re.fullmatch(r"[0-9]+", allocation["job_id"]) and
                type(allocation.get("time_limit_seconds")) is int and
                0 < allocation["time_limit_seconds"] <= MAX_GPU_SECONDS and
                allocation.get("reserve_seconds") == RESERVE_SECONDS and
                allocation.get("deadline_utc") == DEADLINE.isoformat(),
                "Screen role lacks the fixed bounded allocation/deadline receipt")
        job_ids.add(allocation["job_id"])
        versions.append(manifest["versions"])
        receipts[role] = digest(directory / "complete.json")
    require(len(job_ids) == 1 and versions[0] == versions[1] == versions[2],
            "Screen roles used different allocations or software versions")
    require(next(iter(job_ids)) == read_json(output / "gpu_allocation.json")["job_id"],
            "Screen allocation seal differs")
    return dict(schema=SCHEMA, complete=True, exploratory=True,
                  post_primary_observation=True, primary_runs_immutable=True,
                  metric="source teacher boxed-answer agreement; not truth or benchmark accuracy",
                  protocol=PROTOCOL, metrics=metrics, gate=fixed_gate(metrics),
                  prepared_sha256=digest(output / "prepared.json"), role_receipt_sha256=receipts,
                  limitations=["64 source-only prompts, one sample each, one fixed seed",
                               "Teacher solutions may be incorrect; agreement is not ground truth",
                               "Exploratory recovery was proposed after primary benchmark observations",
                               "No primary benchmark gains or complete assignment comparisons are inferred"])


def summarize(output):
    require_compute(cpu_only=True)
    result = _summary_payload(output)
    write_new_json(Path(output).resolve() / "summary.json", result)
    return result


def validate_summary(output):
    """Read-only, lightweight admission check for an already accepted screen.

    Recompute the exact substantive payload, including the fixed gate, from all
    three sealed 64-row observations. Reject rejected/tampered/stale screens.
    This does not admit a GPU job, establish benchmark gains, or verify model
    weight bytes for a future evaluation; the caller retains those duties.
    """
    result = _summary_payload(output)
    require(json_digest(read_json(Path(output).resolve() / "summary.json")) == json_digest(result),
            "Stored screen summary differs from its sealed recomputed evidence")
    require(result["gate"]["accepted"] is True, "The fixed recovery screen gate did not pass")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--artifacts", type=Path, default=Path("deadline/artifacts"))
    prep.add_argument("--merge-receipt", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--role", choices=ROLES, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.artifacts, args.merge_receipt, args.output)
    elif args.phase == "run":
        result = run(args.output, args.role)
    else:
        result = summarize(args.output)
    print(json.dumps(dict(schema=result["schema"], phase=args.phase, complete=True,
                          gate=result.get("gate")), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
