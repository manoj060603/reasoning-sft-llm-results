"""CPU-only, separately admitted preparation of four private native witnesses.

No model, model scores, benchmark answers/outcomes, GPU stage, scheduler or upload
is used. Frozen cached lengths choose the witnesses only after both question-only
exclusion audits. Actual serialization failure never causes replacement/refill.
All raw cohorts, matches and original-pair caches remain private on compute.
The parent wrapper owns final native_qa.json and complete.json after this module
writes acyclic preparation.json then prepared.json; neither admits a GPU job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from extension.cpu_feasibility import (Evidence, SOURCE_REVISION, SOURCE_SHA256,
    canonical_map, deadline_guard, digest, read_json, require, require_cpu_allocation,
    value_digest, write_new_json)
from extension.development_seal_v2 import TARGETS, SPLITS
from extension.development_retry_submission import SOURCE_PATHS as DEVELOPMENT_CODE
from extension.labeled_feasibility import (BENCHMARK_COUNTS, CRITERION, ExactIndex,
    inside, validate_benchmarks)
from extension import native_model_admission as native
from extension.preserving_data import (EXCLUSION_SCHEMA, POOL_ID_NAMESPACE,
    prepare_pool, serialize_selected, text_sha256, verify_serialized_examples)
from reasoning.data import prompt_id

SCHEMA = "extension-native-smoke-preparation-v1"
PLAN_SCHEMA = "extension-native-smoke-preparation-plan-v1"
UNIT_SCHEMA = "extension-native-smoke-preparation-unit-qa-v1"
POOL_SHA256 = "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0"
FEATURES_SHA256 = "c6ef548173b6b4e93a62fd5c43783ad49a6e3dc9b9437b3b31941ffef8c2b55f"
FEATURE_META_SHA256 = "d69155338e79e63690f3b9fd615a1026489046dbad239e96dd8cf14b91b51afc"
SEAL_SHA256 = "354859dd44e2fe3aa7800ec991b60649f00a99ce4768d8075af42c007289987d"
IDENTITY_SHA256 = "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2"
SOURCE_AUDIT_SHA256 = "8cee12034e7c9f9f6f3334cdac2d3edf5504658ef8000c664761a16887440505"
LABELED_PLAN_SHA256 = "d44572f18e3de76dea44b243a7427b8cac64cf5d87a294ae005b544412f28d80"
TERMINAL_PROBE = {"system": "Native serialization diagnostic.", "user": "Return the fixed sentinel.",
                  "assistant": "native-smoke-termination-sentinel-20261021"}
REQUIRED_CODE = frozenset({"extension/smoke_preparation.py", "tests/test_smoke_preparation.py",
    "extension/preserving_data.py", "tests/test_preservation_data.py", "extension/native_model_admission.py",
    "tests/test_native_model_admission.py", "extension/cpu_feasibility.py", "extension/labeled_feasibility.py",
    "extension/development_seal_v2.py", "extension/development_retry_submission.py",
    "reasoning/data.py", "reasoning/features.py"})
_INPUT_PINS = {
    "pool": ("deadline/artifacts/pool.json", POOL_SHA256),
    "features": ("deadline/artifacts/features.json", FEATURES_SHA256),
    "feature_meta": ("deadline/artifacts/features.json.meta.json", FEATURE_META_SHA256),
    "development_seal": ("extension/runs/development-seal-v2-20261021-1223943/complete.json", SEAL_SHA256),
    "labeled_plan": ("extension/labeled_plan.json", LABELED_PLAN_SHA256),
    "original_source_audit": ("extension/runs/feasibility-20261020-1222225/complete.json", SOURCE_AUDIT_SHA256)}


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA and
        plan.get("cpu_only") is True and plan.get("choose_smoke_witnesses") is True and
        all(plan.get(name) is False for name in ("training", "model_inference", "automatic_downstream")),
        "Only the prospective CPU witness-preparation scope is permitted")
    require(plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        type(plan.get("wall_time_seconds")) is int and plan["wall_time_seconds"] == 7200 and
        type(plan.get("reserve_seconds")) is int and plan["reserve_seconds"] == 172800 and
        plan.get("criterion") == CRITERION and plan.get("terminal_probe") == TERMINAL_PROBE,
        "Preserve the fixed CPU runtime, reserve, lexical criterion and synthetic terminal probe")
    for name, (path, checksum) in _INPUT_PINS.items():
        pin = plan.get(name)
        require(isinstance(pin, dict) and pin.get("path") == path and pin.get("sha256") == checksum,
            "Unexpected pinned CPU preparation input: " + name)
    require(plan["pool"].get("expected_rows") == 83029, "The full canonical pool must contain 83,029 rows")
    native._file_pin(plan.get("native_plan"))
    require(plan["native_plan"]["path"] == "extension/native_model_admission_plan.json", "Use the separate native plan")
    base, witness = plan.get("base", {}), plan.get("witness_selection", {})
    require(isinstance(base, dict) and isinstance(base.get("path"), str) and Path(base["path"]).is_absolute() and
        base.get("files") == native.ORIGINAL_FILES and base.get("identity_receipt") ==
        {"path": "deadline/artifacts/model_identity.json", "sha256": IDENTITY_SHA256}, "Only the pinned original base is allowed")
    require(isinstance(witness, dict) and witness.get("context_role") == "diagnostic" and
        witness.get("cutoffs") == {"short": 512, "long": 32768} and
        witness.get("cached_counts_must_match_actual_native_serialization") is True,
        "Native witness contexts and exact cached-count matching must remain explicit")
    return plan


def _pin(root, evidence, spec):
    native._file_pin(spec)
    return evidence.read(inside(root, spec["path"]), spec["sha256"])


def _jsonl(path, evidence):
    evidence.check(path)
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            require(bool(line.strip()), "Blank private JSONL record")
            value = json.loads(line, object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            require(isinstance(value, dict), "Private JSONL requires object rows")
            result.append(value)
    evidence.check(path)
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def validate_unit_qa(path, root, output, compute, evidence, plan_path, native_plan_path):
    path = Path(path).resolve()
    require(path == output / "unit_qa.json", "Use this current run's immutable unit_qa.json")
    qa = evidence.read(path)
    hashes, inventory = qa.get("code_sha256"), qa.get("source_inventory")
    require(qa.get("schema") == UNIT_SCHEMA and qa.get("success") is True and
        type(qa.get("tests_run")) is int and qa["tests_run"] > 230 and
        all(type(qa.get(name)) is int and qa[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("gpu_disabled") is True and qa.get("job_id") == compute["job_id"] and qa.get("node") == compute["node"] and
        qa.get("plan_sha256") == evidence.check(plan_path) and
        qa.get("native_plan_sha256") == evidence.check(native_plan_path) and
        isinstance(hashes, dict) and isinstance(inventory, dict) and set(hashes) == set(inventory) and
        REQUIRED_CODE <= set(hashes), "New successful same-job native unit QA must bind both plans and all new code")
    proof = qa.get("compute_proof", {})
    require(proof.get("job_id") == compute["job_id"] and proof.get("node") == compute["node"] and
        proof.get("gpu_requested_or_allocated") is False, "Unit QA requires this CPU allocation proof")
    for name, checksum in hashes.items():
        file = inside(root, name)
        require(file.suffix in (".py", ".json", ".slurm") and
            not any(part in ("runs", "logs", "report") for part in Path(name).parts) and
            0 < file.stat().st_size <= 2 << 20 and inventory[name] ==
            {"sha256": checksum, "bytes": file.stat().st_size}, "Invalid native unit source inventory")
        evidence.check(file, native._sha(checksum))
    return {"path": str(path.relative_to(root)), "sha256": evidence.check(path), "schema": UNIT_SCHEMA,
        "tests_run": qa["tests_run"], "code_sha256": hashes}


def validate_cohorts(cohorts, ids_by_split, groups_by_split, receipt):
    """Check every private component/record identity without choosing by labels."""
    seen_ids, seen_groups, questions = set(), set(), []
    require(set(cohorts) == set(ids_by_split) == set(groups_by_split) == set(SPLITS), "All three private cohorts are required")
    for split in SPLITS:
        records, identifiers, groups = cohorts[split], ids_by_split[split], groups_by_split[split]
        require(isinstance(records, list) and records and isinstance(identifiers, list) and
            all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in records) and
            identifiers == sorted(row["id"] for row in records) and len(set(identifiers)) == len(identifiers) and
            not seen_ids.intersection(identifiers), "Private source splits overlap or their ID seals differ")
        by_id = {row["id"]: row for row in records}
        require(isinstance(groups, list) and groups and all(isinstance(group, dict) for group in groups) and
            [group.get("group_id") for group in groups] == sorted(group.get("group_id") for group in groups),
            "Private groups require their original deterministic order")
        members, group_ids = [], []
        for group in groups:
            require(isinstance(group, dict) and set(group) == {"group_id", "member_ids", "rows", "domain", "stratum"},
                "Unexpected private component schema")
            ids = group["member_ids"]
            require(isinstance(ids, list) and ids and ids == sorted(set(ids)) and
                all(identifier in by_id for identifier in ids) and group["rows"] == len(ids) and
                group["group_id"] == value_digest({"criterion": CRITERION, "member_ids": ids}) and
                group["group_id"] not in seen_groups and group["domain"] in TARGETS[split] and
                all(by_id[identifier].get("group_id") == group["group_id"] and
                    by_id[identifier].get("source_kind") == group["domain"] for identifier in ids),
                "Private whole-group membership/domain/source binding differs")
            group_ids.append(group["group_id"])
            members.extend(ids)
        require(sorted(members) == identifiers and len(group_ids) == len(set(group_ids)), "A reserved group was split or duplicated")
        stats = receipt.get("splits", {}).get(split, {})
        counts = dict(Counter(row.get("source_kind") for row in records))
        require(stats.get("rows") == len(records) and stats.get("group_count") == len(groups) and
            stats.get("domain_rows") == counts == TARGETS[split] and stats.get("ids_sha256") == value_digest(identifiers) and
            stats.get("group_ids_sha256") == value_digest(group_ids), "Private split rows/groups differ from completed seal")
        for row in records:
            native._sha(row["id"])
            variants = row.get("lexical_screen_variants")
            require(isinstance(variants, list) and variants and all(isinstance(text, str) and text.strip() for text in variants),
                "Reserved question variants are missing")
            questions.append({"id": row["id"], "group_id": row["group_id"], "source_kind": row["source_kind"],
                "lexical_screen_variants": list(variants)})
        seen_ids.update(identifiers)
        seen_groups.update(group_ids)
    return {"record_ids_sha256": value_digest(sorted(seen_ids)), "groups": len(seen_groups),
        "records": len(seen_ids), "question_inventory_sha256": value_digest(sorted(questions, key=lambda row: row["id"]))}


def load_development(root, spec, evidence):
    receipt = _pin(root, evidence, spec)
    directory = inside(root, spec["path"]).parent
    require(receipt.get("schema") == "extension-development-seal-v2" and receipt.get("complete") is True and
        receipt.get("input_stats_unchanged_at_closeout") is True and receipt.get("gpu_seconds_requested_or_used") == 0 and
        receipt.get("compute_start", {}).get("job_id") == receipt.get("compute_end", {}).get("job_id") == "1223943" and
        receipt.get("targets") == TARGETS and receipt.get("actions") ==
        {"splits_chosen": True, "generation": False, "training": False, "automatic_downstream": False},
        "The completed repaired CPU source seal is required")
    qa_spec = receipt.get("native_qa", {})
    qa_path = Path(qa_spec.get("path", ""))
    require(qa_path.is_absolute() and qa_path == directory / "native_qa.json", "Repaired seal must bind its private native QA")
    qa = evidence.read(qa_path, native._sha(qa_spec.get("sha256")))
    hashes = qa.get("code_sha256", {})
    require(qa.get("schema") == "extension-development-seal-native-qa-v2" and qa.get("success") is True and
        qa.get("job_id") == "1223943" and qa.get("tests_run") == qa_spec.get("tests_run") == 230 and
        qa.get("gpu_disabled") is True and all(qa.get(key) == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        isinstance(hashes, dict) and hashes == qa_spec.get("code_sha256") and set(hashes) == set(DEVELOPMENT_CODE),
        "Repaired source seal native230/code evidence differs")
    for name, checksum in hashes.items():
        evidence.check(inside(root, name), native._sha(checksum))
    expected = {split + suffix for split in SPLITS for suffix in (".jsonl", "_ids.json", "_groups.json")}
    expected |= {"unused_groups.json", "reserved_training_pool_ids.json", "reserved_training_pool_matches.json"}
    outputs = receipt.get("output_sha256", {})
    require(isinstance(outputs, dict) and set(outputs) == expected, "The entire private source output inventory is required")
    for name, checksum in outputs.items():
        evidence.check(directory / name, native._sha(checksum))
    cohorts = {split: _jsonl(directory / (split + ".jsonl"), evidence) for split in SPLITS}
    ids = {split: evidence.read(directory / (split + "_ids.json")) for split in SPLITS}
    groups = {split: evidence.read(directory / (split + "_groups.json")) for split in SPLITS}
    cohort_binding = validate_cohorts(cohorts, ids, groups, receipt)
    require(cohort_binding["record_ids_sha256"] == receipt.get("selected_record_ids_sha256"), "Selected private record binding differs")
    unused = evidence.read(directory / "unused_groups.json")
    selected_groups = {group["group_id"] for values in groups.values() for group in values}
    require(isinstance(unused, list) and all(isinstance(group, dict) and group.get("group_id") not in selected_groups
        for group in unused), "Unused source groups overlap reserved groups")
    reserved = evidence.read(directory / "reserved_training_pool_ids.json")
    require(isinstance(reserved, list) and reserved == sorted(set(reserved)) and len(reserved) == 67 and
        all(type(identifier) is str and native._SHA.fullmatch(identifier) for identifier in reserved) and
        receipt.get("reservation", {}).get("lexical_criterion") == CRITERION and
        receipt["reservation"].get("training_pool_rows_scanned") == 83029 and
        receipt["reservation"].get("reserved_training_pool_rows") == len(reserved) and
        receipt["reservation"].get("reserved_training_pool_ids_sha256") == value_digest(reserved),
        "Repaired source reservation ID/count/criterion binding differs")
    return receipt, reserved, {"path": str((directory / "reserved_training_pool_ids.json").relative_to(root)),
        "sha256": outputs["reserved_training_pool_ids.json"]}, cohort_binding


def benchmark_matches(questions, variants):
    """Exact question-only lexical scan; returned matches contain no answer text."""
    require(isinstance(questions, dict) and questions and isinstance(variants, list) and variants,
        "Nonempty canonical questions and benchmark question variants required")
    index, inventory, seen = ExactIndex(), {}, set()
    for variant in variants:
        require(isinstance(variant, dict) and set(variant) == {"task", "field", "row_ordinal", "question"} and
            variant["task"] in BENCHMARK_COUNTS and variant["field"] in ("question", "problem", "Question") and
            type(variant["row_ordinal"]) is int and variant["row_ordinal"] >= 0 and
            isinstance(variant["question"], str) and variant["question"].strip(), "Only explicit question variants may be indexed")
        owner = variant["task"] + ":" + variant["field"] + ":" + str(variant["row_ordinal"])
        require(owner not in seen, "Duplicate benchmark question variant identity")
        seen.add(owner)
        index.add(variant["question"], owner)
        inventory.setdefault(variant["task"], []).append({"field": variant["field"], "row_ordinal": variant["row_ordinal"],
            "question_text_sha256": text_sha256(variant["question"]), "question_id": prompt_id(variant["question"])})
    matches = []
    for identifier, question in sorted(questions.items()):
        native._sha(identifier)
        require(identifier == prompt_id(question), "Canonical pool question identity differs")
        found = index.query(question)
        if found:
            matches.append({"id": identifier, "benchmark_question_matches": found,
                "match_kinds": sorted({kind for kinds in found.values() for kind in kinds})})
    ordered_inventory = {task: sorted(values, key=lambda row: (row["field"], row["row_ordinal"]))
                         for task, values in sorted(inventory.items())}
    return {"ids": [row["id"] for row in matches], "matches": matches,
        "question_inventory_sha256": value_digest(ordered_inventory),
        "question_variant_sha256": {task: value_digest(values) for task, values in ordered_inventory.items()},
        "variant_counts": {task: len(values) for task, values in sorted(inventory.items())}}


def load_benchmark_variants(root, labeled_plan, evidence):
    readiness = validate_benchmarks(root, labeled_plan, evidence)
    variants = []
    for spec in labeled_plan["exclusions"]:
        if spec.get("format") != "benchmark_jsonl":
            continue
        path = inside(root, spec["path"])
        evidence.check(path, spec["sha256"])
        rows = _jsonl(path, evidence)
        require(len(rows) == spec["expected_rows"], "Benchmark question count differs from readiness")
        for ordinal, row in enumerate(rows):
            # Deliberately access only the plan's question field. Gold answer,
            # correctness, prediction and response fields are never consulted.
            variants.append({"task": path.parent.name, "field": spec["question_field"],
                "row_ordinal": ordinal, "question": row.get(spec["question_field"])})
    return variants, readiness


def exclusion_union(pool_ids, reserved, blocked, seal_sha256, benchmark_sha256, reserved_questions_sha256, benchmark_questions_sha256):
    require(isinstance(reserved, list) and reserved == sorted(set(reserved)) and
        isinstance(blocked, list) and blocked == sorted(set(blocked)) and set(reserved) | set(blocked) <= set(pool_ids),
        "Both canonical exclusion ID lists must resolve in the full pool")
    components = {}
    for purpose, scope, ids, audit_sha, questions_sha in (
        ("reserved_source_evaluation", "all_reserved_source_splits", reserved, seal_sha256, reserved_questions_sha256),
        ("final_benchmark_questions", "all_nine_final_benchmarks", blocked, benchmark_sha256, benchmark_questions_sha256)):
        components[purpose] = {"scope": scope, "pool_ids": list(ids), "audit_receipt_sha256": native._sha(audit_sha),
            "criterion_sha256": value_digest(CRITERION), "question_inventory_sha256": native._sha(questions_sha)}
    return {"schema": EXCLUSION_SCHEMA, "id_namespace": POOL_ID_NAMESPACE,
        "population_ids_sha256": value_digest(sorted(pool_ids)), "components": components}


def choose_witnesses(eligible_ids, feature_rows, pool_order):
    """Fixed cached-length rule only; no outputs, quality flags or model scores."""
    require(isinstance(feature_rows, list) and feature_rows and all(isinstance(row, dict) for row in feature_rows) and
        isinstance(pool_order, list) and
        [row.get("id") for row in feature_rows] == pool_order and len(set(pool_order)) == len(pool_order),
        "Cached feature and full-pool source order must match exactly")
    features = {}
    for row in feature_rows:
        require(isinstance(row, dict) and type(row.get("total_tokens")) is int and
            type(row.get("response_tokens")) is int and 0 < row["response_tokens"] < row["total_tokens"],
            "Cached complete-pair token counts are invalid")
        features[row["id"]] = row
    require(type(eligible_ids) in (set, frozenset) and eligible_ids and eligible_ids <= set(features), "Eligible pool IDs must resolve")
    ordered = sorted(eligible_ids, key=lambda identifier: (features[identifier]["total_tokens"], identifier))
    short, lengths = [], set()
    for identifier in ordered:
        count = features[identifier]
        if count["total_tokens"] > 512:
            break
        if count["response_tokens"] not in lengths:
            short.append(identifier)
            lengths.add(count["response_tokens"])
        if len(short) == 3:
            break
    require(len(short) == 3, "Fewer than three eligible short rows with distinct active lengths; no refill")
    longest = min(eligible_ids, key=lambda identifier: (-features[identifier]["total_tokens"], identifier))
    require(longest not in short and features[longest]["total_tokens"] > max(features[key]["total_tokens"] for key in short) and
        features[longest]["total_tokens"] <= 32768, "Longest complete eligible context exceeds the declared envelope")
    return {"short": short, "long": longest, "features": features,
        "eligible_rows": len(eligible_ids), "eligible_ids_sha256": value_digest(sorted(eligible_ids)),
        "pool_order_sha256": value_digest(pool_order),
        "feature_order_sha256": value_digest([row["id"] for row in feature_rows])}


def derive_terminal(tokenizer, probe):
    require(probe == TERMINAL_PROBE, "Use the fixed source-independent native terminal probe")
    messages = [{"role": name, "content": probe[name]} for name in ("system", "user")]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": probe["assistant"]}],
        tokenize=False, add_generation_prompt=False)
    require(isinstance(prompt, str) and isinstance(full, str) and full.startswith(prompt + probe["assistant"]),
        "Native template rewrites the fixed sentinel or assistant prefix")
    terminal_text = full[len(prompt + probe["assistant"]):]
    suffix = tokenizer.encode(terminal_text, add_special_tokens=False)
    require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id == 151645 and
        isinstance(suffix, list) and suffix and suffix[0] == tokenizer.eos_token_id and
        suffix.count(tokenizer.eos_token_id) == 1 and all(type(token) is int and 0 <= token < 151936 for token in suffix),
        "Derived native tail must begin with actual pinned EOS and preserve following template tokens")
    return terminal_text, tuple(suffix)


def _load_tokenizer(base):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)


def _validate_base(root, spec, evidence):
    base = Path(spec["path"]).resolve()
    for name, checksum in native.ORIGINAL_FILES.items():
        evidence.check(base / name, checksum)
    identity = _pin(root, evidence, spec["identity_receipt"])
    require(identity.get("path") == str(base) and identity.get("model_type") == "qwen2" and
        {name: entry.get("sha256") for name, entry in identity.get("files", {}).items()} == native.ORIGINAL_FILES and
        identity.get("identity_sha256") == value_digest({key: value for key, value in identity.items() if key != "identity_sha256"}),
        "The actual pinned original file/identity receipt differs")
    config = evidence.read(base / "config.json")
    require(config.get("model_type") == "qwen2" and config.get("vocab_size") == 151936 and
        config.get("eos_token_id") == 151645 and config.get("max_position_embeddings") == 32768,
        "Pinned native tokenizer/model vocabulary/context/EOS differs")
    return base


def assert_native_counts(examples, features):
    for example in examples:
        proof, cached = example["provenance"], features[example["id"]]
        require(proof["total_tokens"] == cached["total_tokens"] and proof["active_tokens"] == cached["response_tokens"],
            "Selected actual native tokens disagree with cached counts; do not replace or refill: " + example["id"])


def prepare(root, plan_path, output, qa_receipt):
    compute, begun = require_cpu_allocation(), time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = inside(root, str(plan_path))
    require(output.parent == root / "extension/runs" and output.name == "native-smoke-preparation-" + compute["job_id"] and
        output.is_dir() and {path.name for path in output.iterdir()} == {"unit_qa.json"},
        "Use the exclusive current-job output containing only immutable unit_qa.json")
    evidence = Evidence()
    plan = validate_plan(evidence.read(plan_path))
    deadline_guard(plan["wall_time_seconds"] + plan["reserve_seconds"])
    native_plan_path = inside(root, plan["native_plan"]["path"])
    native_plan = native.validate_plan(evidence.read(native_plan_path, plan["native_plan"]["sha256"]))
    unit_qa = validate_unit_qa(Path(qa_receipt), root, output, compute, evidence, plan_path, native_plan_path)
    seal, reserved, reserved_pin, cohort_proof = load_development(root, plan["development_seal"], evidence)
    pool_rows = _pin(root, evidence, {key: plan["pool"][key] for key in ("path", "sha256")})
    by_id = canonical_map(pool_rows)
    require(len(by_id) == plan["pool"]["expected_rows"], "Full canonical source pool count differs")
    feature_rows, meta = _pin(root, evidence, plan["features"]), _pin(root, evidence, plan["feature_meta"])
    require(meta.get("input_sha256") == POOL_SHA256 and meta.get("source_sha256") == SOURCE_SHA256 and
        meta.get("resolved_tokenizer_commit") == native.REVISION, "Cached native feature/source identity differs")
    source_audit = _pin(root, evidence, plan["original_source_audit"])
    original = source_audit.get("original_source", {})
    require(source_audit.get("schema") == "extension-cpu-feasibility-v1" and source_audit.get("complete") is True and
        original.get("repository") == "redsgnaoh/acereason11_100k" and original.get("revision") == SOURCE_REVISION and
        original.get("sha256") == SOURCE_SHA256 and original.get("prepared_and_heldout_original_pairs_verified") == 84029,
        "Pinned earlier original-pair/source verification is required")
    labeled_plan = _pin(root, evidence, plan["labeled_plan"])
    variants, benchmark_binding = load_benchmark_variants(root, labeled_plan, evidence)
    scan = benchmark_matches({identifier: row["instruction"] for identifier, row in by_id.items()}, variants)
    write_new_json(output / "benchmark_pool_ids.json", scan["ids"])
    write_new_json(output / "benchmark_pool_matches.json", scan["matches"])
    benchmark_audit = {"schema": "extension-final-benchmark-pool-exclusions-v1", "complete": True,
        "pool_sha256": POOL_SHA256, "pool_rows_scanned": len(by_id), "criterion": CRITERION,
        "scope": "all_nine_final_benchmarks", "benchmark_question_sha256": benchmark_binding["benchmark_hashes"],
        "question_variant_inventory_sha256": scan["question_inventory_sha256"], "question_variant_counts": scan["variant_counts"],
        "question_variant_sha256": scan["question_variant_sha256"],
        "excluded_pool_ids_sha256": value_digest(scan["ids"]),
        "excluded_pool_ids_file_sha256": digest(output / "benchmark_pool_ids.json"),
        "matches_file_sha256": digest(output / "benchmark_pool_matches.json"), "answers_or_outcomes_used": False,
        "semantic_decontamination_claimed": False, "benchmark_readiness": benchmark_binding}
    write_new_json(output / "benchmark_audit.json", benchmark_audit)
    union = exclusion_union(list(by_id), reserved, scan["ids"], plan["development_seal"]["sha256"],
        digest(output / "benchmark_audit.json"), cohort_proof["question_inventory_sha256"], scan["question_inventory_sha256"])
    write_new_json(output / "exclusion_receipt.json", union)
    source_identity = {"pool_sha256": POOL_SHA256, "features_sha256": FEATURES_SHA256,
        "feature_meta_sha256": FEATURE_META_SHA256, "raw_repository": "redsgnaoh/acereason11_100k",
        "raw_revision": SOURCE_REVISION, "raw_source_sha256": SOURCE_SHA256,
        "original_source_audit_sha256": plan["original_source_audit"]["sha256"]}
    excluded = set(reserved) | set(scan["ids"])
    eligible = prepare_pool(pool_rows, excluded_pool_ids=excluded, exclusion_receipt=union, source_identity=source_identity)
    choice = choose_witnesses({row["id"] for row in eligible["rows"]}, feature_rows, [row["id"] for row in pool_rows])
    base = _validate_base(root, plan["base"], evidence)
    tokenizer = _load_tokenizer(base)
    terminal_text, suffix = derive_terminal(tokenizer, plan["terminal_probe"])
    tokenizer_identity = {"repository": native.REPOSITORY, "revision": native.REVISION,
        "files": native.TOKENIZER_FILES, "chat_template_sha256": text_sha256(tokenizer.chat_template), "eos_token_id": tokenizer.eos_token_id}
    common = dict(excluded_pool_ids=excluded, exclusion_receipt=union, source_identity=source_identity,
        tokenizer_identity=tokenizer_identity, context_role="diagnostic", eos_token_id=151645,
        terminal_suffix=suffix, terminal_text=terminal_text, vocab_size=151936)
    short = serialize_selected(pool_rows, choice["short"], tokenizer, cutoff=512, **common)["examples"]
    long = serialize_selected(pool_rows, [choice["long"]], tokenizer, cutoff=32768, **common)["examples"][0]
    assert_native_counts(short + [long], choice["features"])
    replay = {}
    for kind, examples, cutoff in (("short", short, 512), ("long", [long], 32768)):
        replay[kind] = verify_serialized_examples(examples, tokenizer, source_identity=source_identity,
            tokenizer_identity=tokenizer_identity, exclusion_receipt_sha256=value_digest(union), cutoff=cutoff,
            context_role="diagnostic", eos_token_id=151645, terminal_suffix=suffix, terminal_text=terminal_text, vocab_size=151936)
    rows = {"short": short, "long": long}
    evidence.unchanged()
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    preparation = {"schema": SCHEMA, "complete": True, "cpu_only": True, "gpu_seconds": 0,
        "plan_sha256": evidence.check(native_plan_path), "cpu_plan_sha256": evidence.check(plan_path),
        "rows_sha256": value_digest(rows), "exclusion_receipt_sha256": digest(output / "exclusion_receipt.json"),
        "exclusion_receipt_value_sha256": value_digest(union), "pool_sha256": POOL_SHA256, "features_sha256": FEATURES_SHA256,
        "source_identity": source_identity, "native_qa": unit_qa, "code_sha256": unit_qa["code_sha256"],
        "no_model_or_benchmark_outcomes": True, "input_snapshot_unchanged": True, "compute_start": compute, "compute_end": ending,
        "source_cohorts": cohort_proof, "eligible_population": {key: choice[key] for key in
            ("eligible_rows", "eligible_ids_sha256", "pool_order_sha256", "feature_order_sha256")},
        "longest": {"id": long["id"], "total_tokens": len(long["input_ids"]),
            "max_eligible_total_tokens": choice["features"][choice["long"]]["total_tokens"],
            "after_reserved_exclusion": True, "after_benchmark_exclusion": True}, "native_replay": replay,
        "input_files": evidence.files, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - begun, "gpu_stage_admitted": False}
    write_new_json(output / "preparation.json", preparation)
    def pin(path):
        return {"path": str(path.relative_to(root)), "sha256": digest(path)}
    prepared = {"schema": "extension-native-model-smoke-cache-v1", "complete": True,
        "plan_sha256": evidence.check(native_plan_path), "source_identity": source_identity,
        "base": {"repository": native.REPOSITORY, "revision": native.REVISION, "path": str(base),
            "files": native.ORIGINAL_FILES, "identity_receipt": plan["base"]["identity_receipt"]},
        "serialization": {"eos_token_id": 151645, "terminal_suffix": list(suffix), "terminal_text": terminal_text,
            "tokenizer_identity": tokenizer_identity}, "rows": rows,
        "presentation_schedule": {"short": choice["short"], "long": [choice["long"]] * 2},
        "source_audit": {"preparation": pin(output / "preparation.json"), "development_seal": plan["development_seal"],
            "reserved_pool_ids": reserved_pin, "benchmark_audit": pin(output / "benchmark_audit.json"),
            "benchmark_pool_ids": pin(output / "benchmark_pool_ids.json"), "exclusion_receipt": pin(output / "exclusion_receipt.json"),
            "pool": {key: plan["pool"][key] for key in ("path", "sha256")}, "features": plan["features"]}}
    native.validate_prepared(prepared, native_plan, evidence.check(native_plan_path))
    evidence.unchanged()
    write_new_json(output / "prepared.json", prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="extension/smoke_preparation_plan.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args()
    prepared = prepare(args.root, args.plan, args.output, args.qa_receipt)
    print(json.dumps({"private_preparation_complete": True, "witnesses": 4,
        "rows_sha256": value_digest(prepared["rows"]), "gpu_admitted": False}), flush=True)


if __name__ == "__main__":
    main()
