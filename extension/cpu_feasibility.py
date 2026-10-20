"""Audit cached source-data feasibility without models, tokenization or scores.

This diagnostic does not select a new subset. Cached native serialized token
counts describe the finalized feature population, not the untouched 100K.
Only aggregate diagnostics, file identities and allocation evidence are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time

from reasoning.data import SYSTEM, alpaca, boxed, canonical, prompt_id


SCHEMA = "extension-cpu-feasibility-v1"
SOURCE_SHA256 = "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8"
SOURCE_REVISION = "ce815bd726c0c4ffdc0912789754646910188195"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
LIMITS = (8192, 16384, 32768)
BRANCH_WORDS = ("perhaps", "another", "alternatively")
BRANCH_PATTERN = re.compile(r"\b(?:perhaps|another|alternatively)\b", re.IGNORECASE)
CODE_FILES = ("extension/__init__.py", "extension/cpu_feasibility.py",
              "extension/cpu_feasibility.slurm", "tests/test_extension_feasibility.py",
              "reasoning/data.py")
REPOSITORY = Path(__file__).resolve().parents[1]
_UNSET = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def deadline_guard(duration_seconds=5400, now=None):
    require(type(duration_seconds) is int and duration_seconds > 0, "Invalid duration guard")
    now = datetime.now(timezone.utc) if now is None else now
    require(now.tzinfo is not None and (DEADLINE - now).total_seconds() >= duration_seconds,
            "Insufficient time before Oct 27 23:59 Eastern / Oct 28 03:59 UTC")
    return {"deadline_utc": DEADLINE.isoformat(), "guard_seconds": duration_seconds,
            "checked_at_utc": now.isoformat()}


def validate_allocation(text, job_id, node, uid):
    """Fail closed on incomplete, nonrunning, multi-node or GPU allocations."""
    require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id), "Numeric CPU job ID required")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", text))
    require(fields.get("JobId") == job_id and fields.get("JobState") == "RUNNING",
            "CPU allocation is not the requested RUNNING job")
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    require(user is not None and int(user.group(1)) == uid, "CPU allocation belongs to another user")
    require(fields.get("NumNodes") == "1" and fields.get("NodeList") == node and node and
            not any(word in node.lower() for word in ("login", "owl1.arc", "tc2.arc")),
            "Audit requires its actual single compute node")
    require(fields.get("Partition") == "normal_q", "Expected the Owl CPU normal_q partition")
    allocated = fields.get("AllocTRES", "")
    cpus = re.search(r"(?:^|,)cpu=([0-9]+)(?:,|$)", allocated)
    require(cpus is not None and int(cpus.group(1)) > 0 and
            fields.get("NumCPUs", "").isdigit() and int(fields["NumCPUs"]) >= int(cpus.group(1)),
            "Missing allocated CPU TRES proof")
    for key, value in fields.items():
        if "tres" in key.lower() or "gres" in key.lower():
            require("gpu" not in value.lower(), "GPU request/allocation is forbidden for this audit")
    return {"job_id": job_id, "node": node, "user_id": uid, "state": "RUNNING",
            "allocated_cpus": int(cpus.group(1)), "allocated_tres": allocated,
            "requested_tres": fields.get("ReqTRES"), "gpu_requested_or_allocated": False,
            "scontrol_sha256": hashlib.sha256(text.encode()).hexdigest(), "scontrol": text.strip()}


def require_cpu_allocation():
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be explicitly empty")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        require(os.environ.get(key, "") in ("", "0", "(null)", "N/A"), "GPU environment is forbidden: " + key)
    require(node and socket.gethostname().split(".")[0] == node.split(".")[0],
            "Process is not running on the allocation's compute node")
    require(re.fullmatch(r"[0-9]+", job) is not None, "A real Slurm CPU job is required")
    result = subprocess.run(["scontrol", "--clusters=owl", "show", "job", "--oneliner", job],
                            check=True, capture_output=True, text=True, timeout=30)
    proof = validate_allocation(result.stdout, job, node, os.getuid())
    proof.update(step_id=os.environ.get("SLURM_STEP_ID", "batch"), hostname=socket.gethostname(),
                 checked_at_utc=datetime.now(timezone.utc).isoformat(), cuda_visible_devices="")
    return proof


class Evidence:
    """Hash each explicit read-only input once; detect changes during and after reads."""
    def __init__(self):
        self.files = {}
        self.stats = {}

    @staticmethod
    def fingerprint(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def check(self, path, expected=_UNSET):
        path = Path(path).resolve()
        require(expected is _UNSET or (isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)),
                "Malformed expected input SHA")
        require(path.is_file(), "Missing audit input: " + str(path))
        key = str(path)
        if key not in self.files:
            before = self.fingerprint(path)
            checksum = digest(path)
            require(self.fingerprint(path) == before, "Input changed while hashing: " + str(path))
            self.files[key], self.stats[key] = {"sha256": checksum, "bytes": before[2]}, before
        require(self.fingerprint(path) == self.stats[key], "Audit input changed: " + str(path))
        actual = self.files[key]["sha256"]
        require(expected is _UNSET or actual == expected, "Receipt/input SHA mismatch: " + str(path))
        return actual

    def read(self, path, expected=_UNSET):
        self.check(path, expected)
        value = read_json(path)
        self.check(path, expected)
        return value

    def unchanged(self):
        for path in self.files:
            self.check(path)


def canonical_map(rows):
    require(isinstance(rows, list) and bool(rows), "Nonempty canonical population required")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"id", "instruction", "input", "output", "system"},
                "Unexpected canonical source row schema")
        require(all(isinstance(row[key], str) for key in row) and row["input"] == "" and
                row["system"] == SYSTEM and row["instruction"] == row["instruction"].strip() and
                row["output"] == row["output"].strip() and row["instruction"] and row["output"],
                "Malformed canonical source row")
        require(row["id"] == prompt_id(row["instruction"]) and row["id"] not in result,
                "Duplicate or incorrect canonical prompt ID")
        result[row["id"]] = row
    return result


def feature_map(features, rows):
    require(isinstance(features, list) and bool(features), "Nonempty feature population required")
    result = {}
    for value in features:
        require(isinstance(value, dict) and value.get("id") in rows and value["id"] not in result,
                "Duplicate, unknown or malformed feature ID")
        for key in ("total_tokens", "response_tokens", "cluster"):
            require(type(value.get(key)) is int and value[key] >= (-1 if key == "cluster" else 1),
                    "Invalid cached exact token count/cluster")
        require(value["response_tokens"] < value["total_tokens"], "Cached prompt must be nonempty")
        rep = value.get("repetition")
        require(type(rep) in (int, float) and math.isfinite(rep) and 0 <= rep <= 1 and
                type(value.get("complete_box")) is bool and
                value["complete_box"] == (boxed(rows[value["id"]]["output"]) is not None),
                "Invalid/stale cached structural feature")
        result[value["id"]] = value
    require(set(result) == set(rows), "Features must cover every prepared row exactly once")
    return result


def subset_ids(identifiers, exported, rows):
    require(isinstance(identifiers, list) and bool(identifiers) and
            all(isinstance(identifier, str) and identifier in rows for identifier in identifiers) and
            len(set(identifiers)) == len(identifiers), "Subset IDs must uniquely identify prepared rows")
    require(isinstance(exported, list) and exported == alpaca([rows[key] for key in identifiers]),
            "Subset export/order differs from unchanged original prepared pairs")
    return identifiers


def verify_originals(source_rows, populations):
    expected = {key: value_digest(row) for population in populations for key, row in population.items()}
    require(sum(map(len, populations)) == len(expected), "Prepared/heldout source populations overlap")
    require(isinstance(source_rows, list) and bool(source_rows), "Original source must be a nonempty JSON array")
    seen, matched, invalid, duplicates = set(), set(), 0, 0
    for raw in source_rows:
        try:
            row = canonical(raw)
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        identifier = row["id"]
        if identifier in seen:
            duplicates += 1
            continue
        seen.add(identifier)
        if identifier in expected:
            require(value_digest(row) == expected[identifier], "Prepared pair differs from first canonical original source row")
            matched.add(identifier)
    require(matched == set(expected), "Prepared/heldout pair is absent from the original source")
    return {"raw_source_rows": len(source_rows), "valid_unique_canonical_source_prompts": len(seen),
            "invalid_source_rows": invalid, "duplicate_canonical_source_rows": duplicates,
            "prepared_and_heldout_original_pairs_verified": len(matched)}


def quantiles(values):
    ordered = sorted(values)
    require(bool(ordered) and all(type(value) in (int, float) and math.isfinite(value) for value in ordered),
            "Quantiles require nonempty finite numeric data")
    result = {}
    for name, fraction in (("min", 0), ("p01", .01), ("p10", .1), ("p25", .25),
                           ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1)):
        position = (len(ordered) - 1) * fraction
        lo, hi = math.floor(position), math.ceil(position)
        result[name] = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return result


def text_diagnostics(text):
    """Formatting-sensitive source diagnostics; no correctness/quality claim."""
    opens, closes, depth, ordered = 0, 0, 0, True
    for match in re.finditer(r"</?think>", text):
        if match.group() == "<think>":
            opens, depth = opens + 1, depth + 1
        else:
            closes, depth = closes + 1, depth - 1
            ordered = ordered and depth >= 0
    balanced = ordered and depth == 0
    steps = [line for line in text.splitlines() if line.strip()]
    branch_steps = sum(bool(BRANCH_PATTERN.search(step)) for step in steps)
    words = text.split()
    grams = [tuple(words[index:index + 8]) for index in range(max(0, len(words) - 7))]
    repetition = 0. if not grams else 1 - len(set(grams)) / len(grams)
    return {"characters": len(text), "whitespace_words": len(words), "repetition": repetition,
            "think_open_tags": opens, "think_close_tags": closes, "think_balanced_ordered": balanced,
            "no_think_tags": opens == closes == 0, "starts_think_open": text.lstrip().startswith("<think>"),
            "starts_think_close": text.lstrip().startswith("</think>"),
            "steps": len(steps), "branch_steps": branch_steps,
            "branch_step_density": branch_steps / len(steps) if steps else 0.}


def summarize_population(identifiers, features, diagnostics):
    require(isinstance(identifiers, list) and bool(identifiers) and len(set(identifiers)) == len(identifiers),
            "Population requires unique IDs")
    require(all(identifier in features and identifier in diagnostics for identifier in identifiers), "Unknown population ID")
    fs, ds = [features[key] for key in identifiers], [diagnostics[key] for key in identifiers]
    totals, responses = [value["total_tokens"] for value in fs], [value["response_tokens"] for value in fs]
    prompts = [total - response for total, response in zip(totals, responses)]
    fits = {}
    for limit in LIMITS:
        full = sum(total <= limit for total in totals)
        fits[str(limit)] = {"complete_serialized_rows": full, "complete_fraction": full / len(fs),
            "overlength_rows": len(fs) - full, "prompt_at_or_over_limit": sum(prompt >= limit for prompt in prompts),
            "supervised_tokens_if_prefix_limited": sum(min(response, max(0, limit - prompt))
                for response, prompt in zip(responses, prompts)),
            "rows_fitting_with_32_token_reserve": sum(total <= limit - 32 for total in totals)}
    tags = {key: sum(bool(value[key]) for value in ds) for key in
            ("no_think_tags", "starts_think_open", "starts_think_close", "think_balanced_ordered")}
    tags.update(think_open_tags=sum(value["think_open_tags"] for value in ds),
                think_close_tags=sum(value["think_close_tags"] for value in ds),
                unbalanced_or_misordered_rows=sum(not value["think_balanced_ordered"] for value in ds))
    steps, branching = sum(value["steps"] for value in ds), sum(value["branch_steps"] for value in ds)
    return {"rows": len(fs), "id_order_sha256": value_digest(identifiers),
            "total_serialized_tokens": sum(totals), "total_response_span_tokens": sum(responses),
            "total_prompt_tokens": sum(prompts), "exact_cached_token_fit": fits,
            "complete_box_rows": sum(value["complete_box"] for value in fs),
            "cluster_counts": dict(sorted(Counter(str(value["cluster"]) for value in fs).items())),
            "quantiles": {"serialized_tokens": quantiles(totals), "response_span_tokens": quantiles(responses),
                "prompt_tokens": quantiles(prompts), **{key: quantiles([value[key] for value in ds])
                for key in ("characters", "whitespace_words", "repetition", "branch_step_density")}},
            "think_tags": tags, "branch_steps": {"nonempty_line_steps": steps,
                "steps_with_keyword": branching, "pooled_density": branching / steps if steps else 0.}}


def validate_cache_meta(meta, identifiers, features):
    require(meta.get("rows") == meta.get("expected_rows") == len(identifiers) and
            meta.get("unique_rows_dropped") == 0 and meta.get("target_rewriting") is False,
            "Training-cache row/preservation metadata differs")
    cutoff = meta.get("cutoff")
    require(type(cutoff) is int and cutoff > 0, "Training cutoff is missing")
    truncated = [key for key in identifiers if features[key]["total_tokens"] > cutoff]
    require(meta.get("truncated_ids") == truncated and meta.get("truncated_rows") == len(truncated),
            "Cached exact features disagree with actual truncation metadata")
    original = sum(features[key]["total_tokens"] for key in identifiers)
    supervised = sum(min(features[key]["response_tokens"], max(0, cutoff -
        (features[key]["total_tokens"] - features[key]["response_tokens"]))) for key in identifiers)
    require(meta.get("original_serialized_tokens") == original and
            meta.get("supervised_training_tokens") == supervised and
            meta.get("truncation_policy") == ("prefix_truncate" if truncated else "none"),
            "Training-token totals/policy disagree with exact cached features")
    return {"cutoff": cutoff, "truncated_rows": len(truncated), "original_serialized_tokens": original,
            "supervised_training_tokens": supervised, "rows_dropped": 0, "target_rewriting": False,
            "eos_note": "No EOS ID is inferred from text/features; overlength prefixes omit the original serialized tail."}


def code_hashes():
    return {relative: digest(REPOSITORY / relative) for relative in CODE_FILES}


def validate_qa(path, output, compute, evidence):
    path = Path(path).resolve()
    require(path.parent == output and path.name == "native_qa.json", "QA receipt must be the output's native_qa.json")
    qa = evidence.read(path)
    require(qa.get("schema") == "extension-feasibility-native-qa-v1" and qa.get("success") is True and
            type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and
            all(type(qa.get(key)) is int and qa[key] == 0 for key in ("failures", "errors", "skipped")) and
            isinstance(qa.get("code_sha256"), dict) and
            all(qa["code_sha256"].get(name) == checksum for name, checksum in code_hashes().items()) and
            qa.get("job_id") == compute["job_id"] and
            qa.get("node") == compute["node"] and qa.get("gpu_disabled") is True,
            "Scoped native CPU QA is incomplete, failed, stale or belongs to another allocation")
    return {"path": str(path), "sha256": evidence.check(path), "tests_run": qa["tests_run"]}


def _inside(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Unsafe receipt-relative path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and not any(part in ("results", "benchmarks") for part in Path(relative).parts),
            "Receipt attempts to read outside source/training artifacts")
    return path


def audit(root, output, qa_receipt):
    compute, deadline = require_cpu_allocation(), deadline_guard()
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.parent == REPOSITORY / "extension/runs" and
            output.name == "feasibility-20261020-" + compute["job_id"], "Use the fixed unique job-specific feasibility output")
    require(output.is_dir() and {path.name for path in output.iterdir()} == {"native_qa.json"},
            "Output must contain only its successful native QA receipt; completed/partial outputs cannot be reused")
    files = Evidence()
    qa = validate_qa(qa_receipt, output, compute, files)
    code = code_hashes()
    for relative, checksum in code.items():
        files.check(REPOSITORY / relative, checksum)
    # Exclusive reservation remains after failure; use a new allocation/output for retry.
    write_new_json(output / "started.json", {"schema": SCHEMA, "complete": False, "compute": compute,
        "code_sha256": code, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    artifacts = root / "artifacts"
    preparation = files.read(artifacts / "preparation_complete.json")
    require(preparation.get("source_sha256") == SOURCE_SHA256 and preparation.get("source_revision") == SOURCE_REVISION and
            preparation.get("model_revision") == MODEL_REVISION, "Unexpected source/base revision")
    finalized = files.read(artifacts / "finalized.json")
    complete = files.read(artifacts / "selection_complete.json")
    require(complete.get("complete") is True and complete.get("rows_per_arm") == 15000 and complete.get("cutoff") == 8192,
            "Complete primary selection receipt is required")
    sealed_outputs = complete.get("files_sha256", {})
    require(isinstance(sealed_outputs, dict) and {
        "selected/advanced.json", "selected/random.json", "selected/selection_manifest.json",
        "training/advanced/cache_meta.json", "training/advanced/selected_ids.json",
        "training/advanced/dataset.json", "training/advanced/training.jsonl"}.issubset(sealed_outputs),
        "Selection completion output-file seal is incomplete")
    for relative, checksum in sealed_outputs.items():
        files.check(_inside(artifacts, relative), checksum)
    selection = files.read(artifacts / "selected/selection_manifest.json")
    check_selection = dict(selection)
    seal = check_selection.pop("selection_sha256", None)
    require(seal == value_digest(check_selection) == complete.get("selection_sha256"), "Selection content seal differs")
    sources = selection.get("build_source_file_sha256", {})
    require(isinstance(sources, dict) and {"artifacts/pool.json", "artifacts/features.json", "artifacts/features.json.meta.json"}.issubset(sources),
            "Selection source-file seal is incomplete")
    for relative, checksum in sources.items():
        files.check(_inside(root, relative), checksum)
    pool = canonical_map(files.read(artifacts / "pool.json"))
    heldout = canonical_map(files.read(artifacts / "heldout.json"))
    require(not set(pool).intersection(heldout), "Source heldout overlaps prepared feature population")
    features = feature_map(files.read(artifacts / "features.json"), pool)
    feature_meta = files.read(artifacts / "features.json.meta.json")
    require(feature_meta.get("input_sha256") == files.check(artifacts / "pool.json") and
            feature_meta.get("source_sha256") == SOURCE_SHA256 and
            feature_meta.get("resolved_tokenizer_commit") == MODEL_REVISION and
            finalized.get("feature_sha256") == files.check(artifacts / "features.json"),
            "Exact feature population/tokenizer provenance is stale")
    require(finalized.get("source_pool_after_all_exclusions") == len(pool) and
            finalized.get("source_heldout_after_all_exclusions") == len(heldout), "Finalized population counts differ")
    transfer = files.read(artifacts / "transfer_meta.json")
    require(transfer.get("input_sha256") == files.check(artifacts / "pool.json") and
            finalized.get("transfer_meta_sha256") == files.check(artifacts / "transfer_meta.json"), "Transfer source seal differs")
    reservations = transfer.get("probe_training_ids", {})
    require(set(reservations) == {"probeA", "probeB"}, "Both source probe reservations are required")
    probe_ids = set()
    for role in ("probeA", "probeB"):
        ids = files.read(artifacts / (role + "_ids.json"))
        subset_ids(ids, alpaca([pool[key] for key in ids if key in pool]), pool)
        require(ids == reservations[role] and not probe_ids.intersection(ids), "Probe IDs differ/overlap")
        probe_ids.update(ids)
    require(selection.get("prepared_pool_rows") == len(pool) and
            selection.get("random_sampling_pool_rows") == len(pool) - len(probe_ids),
            "Selection population counts differ from audited source/probe populations")
    reservoir = transfer.get("reservoir_ids", [])
    require(isinstance(reservoir, list) and len(reservoir) == len(set(reservoir)) and
            all(identifier in pool and identifier not in probe_ids for identifier in reservoir),
            "Invalid reserved advanced candidate reservoir")
    arms, cache_stats = {}, {}
    for role in ("random", "advanced"):
        identifiers = selection["datasets"][role]["selected_ids"]
        exported = files.read(artifacts / "selected" / (role + ".json"))
        arms[role] = subset_ids(identifiers, exported, pool)
        require(len(identifiers) == 15000 and not set(identifiers).intersection(probe_ids), "Training arm has wrong size or probe overlap")
        training = artifacts / "training" / role
        require(files.read(training / "dataset.json") == exported and
                files.read(training / "selected_ids.json") == identifiers, "Training export/IDs differ from selection")
        meta = files.read(training / "cache_meta.json")
        require(meta.get("dataset_role") == role and meta.get("dataset_sha256") == files.check(training / "dataset.json") and
                meta.get("selected_ids_sha256") == files.check(training / "selected_ids.json") and
                isinstance(meta.get("cache_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", meta["cache_sha256"]),
                "Training export/cache metadata SHA differs")
        files.check(training / "training.jsonl", meta.get("cache_sha256"))
        cache_stats[role] = validate_cache_meta(meta, identifiers, features)
    require(files.read(artifacts / "random_ids.json") == arms["random"], "Pre-score random IDs differ")
    source = files.read(artifacts / "source/data/acereason11_100k.json", SOURCE_SHA256)
    original_counts = verify_originals(source, (pool, heldout))
    require(original_counts["raw_source_rows"] == 100000, "Pinned original source is not the expected 100K population")
    del source
    gc.collect()
    # Additional independent old source seals are checked when their local receipt is available.
    old_checks = []
    candidates = (root / "anchor-search-runs/validation-v2-20261020-attempt2/prepared.json",
                  root / "report/anchor-v2-preparation-20261020/prepared.json",
                  root / "recovery-runs/base-anchor-75-25-20261020/development/prepared.json")
    for path in candidates:
        if not path.is_file():
            continue
        prepared = files.read(path)
        source_audit = prepared.get("source_audit", {})
        require(Path(source_audit.get("artifacts", "")).resolve() == artifacts, "Old prepared source path differs")
        compared = 0
        for name, checksum in source_audit.get("files", {}).items():
            other = Path(name) if Path(name).is_absolute() else artifacts / name
            key = str(other.resolve())
            if key in files.files:
                files.check(other, checksum)
                compared += 1
        require(compared >= 8, "Old prepared receipt lacks shared source/export bindings")
        old_checks.append({"path": str(path), "sha256": files.check(path), "matching_input_bindings": compared})
    print("Verified source, cached features and unchanged training exports; aggregating text diagnostics.", flush=True)
    diagnostics = {}
    for index, (identifier, row) in enumerate(pool.items(), 1):
        result = text_diagnostics(row["output"])
        require(abs(result["repetition"] - features[identifier]["repetition"]) <= 1e-12,
                "Cached repetition differs from original response")
        diagnostics[identifier] = result
        if index % 10000 == 0:
            print(json.dumps({"diagnosed_rows": index, "population_rows": len(pool)}), flush=True)
    populations = {"finalized_feature_population": list(pool), **arms}
    summaries = {name: summarize_population(ids, features, diagnostics) for name, ids in populations.items()}
    versions = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "installed_distributions": sorted({(str(item.metadata.get("Name") or "unknown"), item.version)
                    for item in importlib.metadata.distributions()})}
    require(code_hashes() == code, "Feasibility source changed during the audit")
    files.unchanged()
    ending_compute = require_cpu_allocation()
    require(ending_compute["job_id"] == compute["job_id"] and ending_compute["node"] == compute["node"], "Allocation changed during audit")
    result = {"schema": SCHEMA, "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "output": str(output), "compute_start": compute, "compute_end": ending_compute,
        "gpu_seconds_requested_or_used": 0, "deadline": deadline, "native_qa": qa, "code_sha256": code,
        "versions": versions, "cached_feature_metadata": feature_meta,
        "input_files": files.files, "input_stats_unchanged_at_closeout": True,
        "original_source": {"repository": "redsgnaoh/acereason11_100k", "revision": SOURCE_REVISION,
            "sha256": SOURCE_SHA256, **original_counts},
        "populations": {"raw_source_rows": original_counts["raw_source_rows"], "finalized_prepared_pool_rows": len(pool),
            "cached_feature_rows": len(features), "source_heldout_rows": len(heldout), "reserved_probe_rows": len(probe_ids),
            "advanced_scored_reservoir_rows": len(reservoir),
            "broad_prepared_random_sampling_pool_rows": len(pool) - len(probe_ids),
            "random_rows": len(arms["random"]), "advanced_rows": len(arms["advanced"]),
            "random_advanced_overlap_rows": len(set(arms["random"]) & set(arms["advanced"]))},
        "historical_preparation_counts": preparation.get("counts"), "old_prepared_source_checks": old_checks,
        "population_diagnostics": summaries, "actual_training_cache_metadata_checks": cache_stats,
        "definitions": {"token_counts": "Previously cached exact native Qwen serialized counts; response span includes assistant template tail. No new tokenization was run.",
            "quantiles": "Linear interpolation at (n-1)*p, including min/max.",
            "branch_step_proxy": {"scope": "Entire original output, split into nonempty newline-delimited steps",
                "case_insensitive_whole_words": list(BRANCH_WORDS), "status": "Formatting-sensitive English-keyword diagnostic, biased across languages; not validated reasoning quality or selection policy"},
            "repetition": "One minus unique whitespace 8-gram count divided by total whitespace 8-gram count; zero below eight words.",
            "think_tags": "Literal <think>/</think> counts; ordered balance uses a nonnegative nesting depth ending at zero; no-tag rows are balanced.",
            "stopping": "Source tag structure and existing prefix-truncation metadata only; no generated stopping behavior or teacher correctness was assessed.",
            "population_limit": "Feature-fit counts cover the finalized prepared pool, not all untouched 100K source records.",
            "cluster_sentinels": "Cluster -1 is an unfitted producer sentinel; the frozen finalizer remaps unfitted rows to64, outside its real0–63 topics. Neither sentinel is a learned topic.",
            "actions": "Read-only diagnostics: no selection, model inference, tokenizer loading, downloads, benchmark outcome reads or downstream admission."},
        "elapsed_seconds": time.monotonic() - started}
    write_new_json(output / "complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="deadline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args()
    value = audit(args.root, args.output, args.qa_receipt)
    print(json.dumps({"complete": True, "output": str(Path(args.output).resolve() / "complete.json"),
                      "populations": value["populations"]}, indent=2))


if __name__ == "__main__":
    main()
