"""Versioned metadata-only repair of the first whole-group development seal.

The unchanged official literal Level ? has its own diagnostic-bound stratum.
No source records, groups, targets, hash ordering or exclusions are rewritten.

This stage performs no tokenization, generation, scoring or training. Exact
subset-sum guards protect each greedy choice; they are necessary feasibility
checks, not a complete solver for the joint multi-split partition problem.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, require_cpu_allocation, value_digest, write_new_json)
from extension.labeled_feasibility import (CRITERION, MATH_COUNTS, SOURCE_PINS,
    ExactIndex, canonical_questions, inside, source_record)

SCHEMA = "extension-development-seal-v2"
PLAN_SCHEMA = "extension-development-seal-plan-v2"
QA_SCHEMA = "extension-development-seal-native-qa-v2"
DOMAINS = ("math", "arc_challenge")
SPLITS = ("calibration", "development", "confirmation")
TARGETS = {"calibration": {"math": 16, "arc_challenge": 16},
    "development": {"math": 512, "arc_challenge": 256},
    "confirmation": {"math": 1024, "arc_challenge": 512}}
HASH_TEMPLATE = "20261021:development-seal-v1:{domain}:{split}:{group_id}"
LABELED_DIRECTORY = "extension/runs/labeled-feasibility-20261020-1222926"
POOL_PATH = "deadline/artifacts/pool.json"
POOL_SHA256 = "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0"
UNKNOWN_LEVELS = {"Level ?": 1}
DIAGNOSTIC = {"job_id": "1223713", "directory": "extension/runs/development-level-diagnostic-20261021-1223713",
    "complete_sha256": "7944215f48b8bd12ba543fe5cf4b543ef1c21cfc674fe35ea4b17c49698cf96c"}
_SHA = re.compile(r"[0-9a-f]{64}")


def sha(value):
    require(isinstance(value, str) and _SHA.fullmatch(value), "Expected a SHA256 string")
    return value


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA and
        plan.get("cpu_only") is True and plan.get("model_inference") is False and
        plan.get("choose_splits") is True and plan.get("automatic_downstream") is False and
        plan.get("training") is False and plan.get("selection_uses_labels_or_model_outcomes") is False and
        plan.get("historical_inputs_read_only") is True,
        "Expected the sealed CPU-only development plan")
    require(plan.get("seed") == 20261021 and plan.get("split_order") == list(SPLITS) and
        plan.get("targets") == TARGETS and plan.get("hash_template") == HASH_TEMPLATE and
        plan.get("near_duplicate") == CRITERION, "Split targets/order/seed/criterion differ")
    require(plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        type(plan.get("wall_time_seconds")) is int and plan["wall_time_seconds"] > 0 and
        type(plan.get("reserve_seconds")) is int and plan["reserve_seconds"] > 0,
        "Missing prospective deadline guards")
    labeled, pool = plan.get("labeled_audit", {}), plan.get("pool", {})
    require(isinstance(labeled, dict) and labeled.get("directory") == LABELED_DIRECTORY and
        isinstance(pool, dict) and pool.get("path") == POOL_PATH and pool.get("sha256") == POOL_SHA256,
        "Unexpected labeled audit or canonical training population")
    sha(labeled.get("complete_sha256"))
    sha(labeled.get("qa_sha256"))
    require(plan.get("level_diagnostic") == DIAGNOSTIC and plan.get("math_unknown_levels") == UNKNOWN_LEVELS,
        "Retry requires the exact native unknown-level diagnostic and observed count")
    return plan


def group_order(domain, split, group_id):
    require(domain in DOMAINS and split in SPLITS, "Unknown domain/split")
    sha(group_id)
    return hashlib.sha256(HASH_TEMPLATE.format(domain=domain, split=split,
        group_id=group_id).encode("utf-8")).hexdigest()


def row_stratum(record):
    """Use source/question metadata only; never inspect labels or solution text."""
    domain = record["source_kind"]
    if domain == "math":
        subject = record["source"]["file"].split("/")[0]
        level = record["original_record"]["level"]
        require(subject in MATH_COUNTS and isinstance(level, str) and
            (re.fullmatch(r"Level [1-5]", level) or level in UNKNOWN_LEVELS),
            "MATH subject/official level is missing")
        stratum_level = "unknown:" + level if level in UNKNOWN_LEVELS else level
        return "math:" + json.dumps([subject, stratum_level], ensure_ascii=False, separators=(",", ":"))
    require(domain == "arc_challenge", "Unknown labeled domain")
    words = len(record["question"].split())
    bucket = "<=15" if words <= 15 else "<=30" if words <= 30 else "<=60" if words <= 60 else ">60"
    count = len(record["original_record"]["choices"]["label"])
    return "arc_challenge:" + json.dumps([bucket, count], separators=(",", ":"))


def validate_population(records, groups):
    """Rebuild every surviving record's native schema and all component seals."""
    require(isinstance(records, list) and records and isinstance(groups, list) and groups,
        "Nonempty records and component inventory required")
    by_id = {}
    for record in records:
        require(isinstance(record, dict) and record.get("source_kind") in DOMAINS and
            isinstance(record.get("source"), dict), "Unsupported surviving record")
        source = record["source"]
        require(set(source) == {"repository", "revision", "file", "row_ordinal"} and
            source["repository"] in SOURCE_PINS, "Malformed native source identity")
        kind, revision, files = SOURCE_PINS[source["repository"]]
        maximum = MATH_COUNTS.get(source["file"].split("/")[0], 0) if kind == "math" else 1119
        require(record["source_kind"] == kind and source["revision"] == revision and
            source["file"] in files and type(source["row_ordinal"]) is int and
            0 <= source["row_ordinal"] < maximum, "Source row lies outside pinned train files")
        rebuilt, quarantine = source_record(kind, record.get("original_record"), source)
        require(quarantine is None and record == dict(rebuilt, group_id=record.get("group_id")),
            "Surviving record differs from its unchanged native question/options/label")
        identifier = record["id"]
        require(identifier not in by_id, "Duplicate surviving source identity")
        by_id[identifier] = record
    seen, group_ids, eligible, excluded_groups = set(), set(), [], 0
    for group in groups:
        require(isinstance(group, dict) and set(group) == {"group_id", "member_ids", "excluded",
            "directly_excluded_member_ids", "group_exclusion_reasons"}, "Unsupported component schema")
        members = group["member_ids"]
        require(isinstance(members, list) and members and members == sorted(set(members)) and
            all(isinstance(identifier, str) and _SHA.fullmatch(identifier) for identifier in members),
            "Component members must be nonempty, unique sorted SHA identities")
        expected = value_digest({"criterion": CRITERION, "member_ids": members})
        require(group["group_id"] == expected and expected not in group_ids and not seen.intersection(members),
            "Duplicate/tampered component membership")
        group_ids.add(expected)
        seen.update(members)
        require(type(group["excluded"]) is bool and isinstance(group["directly_excluded_member_ids"], list) and
            isinstance(group["group_exclusion_reasons"], list), "Malformed component exclusion state")
        if group["excluded"]:
            require(not set(members).intersection(by_id) and bool(group["directly_excluded_member_ids"]) and
                set(group["directly_excluded_member_ids"]) <= set(members) and bool(group["group_exclusion_reasons"]),
                "Excluded component appears in surviving population")
            excluded_groups += 1
            continue
        require(not group["directly_excluded_member_ids"] and not group["group_exclusion_reasons"] and
            all(identifier in by_id and by_id[identifier]["group_id"] == expected for identifier in members),
            "Surviving component is incomplete or has exclusion evidence")
        domains = {by_id[identifier]["source_kind"] for identifier in members}
        strata = {row_stratum(by_id[identifier]) for identifier in members}
        eligible.append({"group_id": expected, "member_ids": members, "rows": len(members),
            "domain": next(iter(domains)) if len(domains) == 1 else "cross_domain",
            "stratum": next(iter(strata)) if len(strata) == 1 else "mixed"})
    require(set(by_id) <= seen and sum(group["rows"] for group in eligible) == len(by_id),
        "Surviving records and whole-group inventory do not reconcile")
    return by_id, sorted(eligible, key=lambda item: item["group_id"]), excluded_groups


def reachable_sums(size_counts, limit):
    """Exact bounded subset-sum bitset, with binary multiplicity decomposition."""
    require(type(limit) is int and limit >= 0, "Nonnegative subset-sum limit required")
    bits, mask = 1, (1 << (limit + 1)) - 1
    for size, count in sorted(size_counts.items()):
        require(type(size) is int and size > 0 and type(count) is int and count >= 0,
            "Invalid component-size multiplicity")
        count, block = min(count, limit // size), 1
        while count:
            take = min(block, count)
            bits = (bits | (bits << (take * size))) & mask
            count -= take
            block *= 2
    return bits


def acceptance_feasible(size_counts, chosen_size, remainder, future_targets):
    """Necessary exact guards after removing one candidate component."""
    require(type(chosen_size) is int and chosen_size > 0 and type(remainder) is int and remainder >= 0 and
        all(type(value) is int and value >= 0 for value in future_targets), "Malformed allocation targets")
    if size_counts.get(chosen_size, 0) < 1:
        return False
    counts = Counter(size_counts)
    counts[chosen_size] -= 1
    total = sum(size * count for size, count in counts.items())
    future = sum(future_targets)
    if total < remainder + future:
        return False
    checks = {remainder, future, remainder + future, *future_targets}
    bits = reachable_sums(counts, max(checks))
    return all((bits >> target) & 1 for target in checks)


def allocate_groups(groups, targets=None):
    """Fixed proportional-deficit allocation with exact row targets or failure.

    Fixed initial row strata determine every split's desired proportions. A
    later failure is a failed deterministic allocation, not proof that no joint
    multi-split partition exists. Cross-domain components are never selected.
    """
    targets = TARGETS if targets is None else targets
    require(set(targets) == set(SPLITS) and all(set(targets[split]) == set(DOMAINS) and
        all(type(value) is int and value > 0 for value in targets[split].values()) for split in SPLITS),
        "All three positive per-domain row targets required")
    assigned = {split: [] for split in SPLITS}
    initial, statistics = {}, {}
    for domain in DOMAINS:
        remaining = {group["group_id"]: group for group in groups if group["domain"] == domain}
        require(len(remaining) == sum(group["domain"] == domain for group in groups), "Duplicate component ID")
        strata = Counter()
        sizes = Counter()
        for group in remaining.values():
            require(type(group["rows"]) is int and group["rows"] > 0 and
                len(group["member_ids"]) == group["rows"], "Malformed component row count")
            strata[group["stratum"]] += group["rows"]
            sizes[group["rows"]] += 1
        population = sum(strata.values())
        require(population >= sum(targets[split][domain] for split in SPLITS),
            "Insufficient eligible whole-group row capacity for " + domain)
        initial[domain] = {"rows": population, "groups": len(remaining), "stratum_rows": dict(sorted(strata.items()))}
        for position, split in enumerate(SPLITS):
            target, selected, achieved = targets[split][domain], [], Counter()
            future = [targets[name][domain] for name in SPLITS[position + 1:]]
            used = 0
            while used < target:
                feasibility = {}
                choices = sorted(remaining.values(), key=lambda group: (
                    -(target * strata[group["stratum"]] - achieved[group["stratum"]] * population),
                    group["stratum"], group_order(domain, split, group["group_id"])))
                chosen = None
                for group in choices:
                    size = group["rows"]
                    if size > target - used:
                        continue
                    if size not in feasibility:
                        feasibility[size] = acceptance_feasible(sizes, size, target - used - size, future)
                    if feasibility[size]:
                        chosen = group
                        break
                require(chosen is not None, "Deterministic allocation failed for " + domain + "/" + split +
                    "; targets were not reduced; this is not a global multi-split impossibility proof")
                selected.append(chosen)
                del remaining[chosen["group_id"]]
                sizes[chosen["rows"]] -= 1
                achieved[chosen["stratum"]] += chosen["rows"]
                used += chosen["rows"]
            assigned[split].extend(selected)
            statistics.setdefault(split, {})[domain] = {"target_rows": target, "achieved_rows": used,
                "selected_groups": len(selected), "stratum_rows": dict(sorted(achieved.items())),
                "desired_stratum_rows": {name: {"numerator": target * rows, "denominator": population}
                    for name, rows in sorted(strata.items())}}
    selected_ids = [group["group_id"] for values in assigned.values() for group in values]
    require(len(selected_ids) == len(set(selected_ids)), "A whole group crossed split boundaries")
    selected_set = set(selected_ids)
    unused = sorted([dict(group, unused_reason="cross_domain_component" if group["domain"] == "cross_domain"
        else "not_selected") for group in groups if group["group_id"] not in selected_set],
        key=lambda group: group["group_id"])
    return assigned, unused, {"initial_eligible_population": initial, "splits": statistics,
        "unused_rows": sum(group["rows"] for group in unused), "unused_groups": len(unused),
        "cross_domain_unused_rows": sum(group["rows"] for group in unused if group["domain"] == "cross_domain")}


def reserve_pool(questions, selected_records):
    """Return every old-pool question that meets the fixed lexical criterion."""
    index = ExactIndex()
    for record in selected_records:
        for text in record["lexical_screen_variants"]:
            index.add(text, record["id"])
    matches = []
    for identifier, question in sorted(questions.items()):
        found = index.query(question)
        if found:
            matches.append({"id": identifier, "reserved_record_matches": found,
                "match_kinds": sorted({kind for kinds in found.values() for kind in kinds})})
    return [row["id"] for row in matches], matches


def validate_native_qa(path, root, output, compute, evidence, plan_path):
    require(path.resolve() == output / "native_qa.json", "Use this run's native_qa.json")
    qa = evidence.read(path)
    hashes, inventory = qa.get("code_sha256"), qa.get("source_inventory")
    require(qa.get("schema") == QA_SCHEMA and qa.get("success") is True and
        type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and
        all(type(qa.get(name)) is int and qa[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("gpu_disabled") is True and qa.get("wrapper_syntax_passed") is True and
        qa.get("job_id") == compute["job_id"] and qa.get("node") == compute["node"] and
        isinstance(hashes, dict) and isinstance(inventory, dict) and set(hashes) == set(inventory),
        "Missing/failed/mismatched native CPU QA")
    proof = qa.get("compute_proof", {})
    require(proof.get("job_id") == compute["job_id"] and proof.get("node") == compute["node"] and
        proof.get("gpu_requested_or_allocated") is False, "QA is not bound to this CPU allocation")
    counts = qa.get("test_counts", {})
    require(isinstance(counts, dict) and set(counts) == {"test_extension_*.py", "test_labeled_*.py",
        "test_preserving_*.py", "test_development_*.py", "test_preservation_development_seal_v2.py",
        "test_preservation_retry_submission.py"} and
        counts.get("test_extension_*.py") == 64 and counts.get("test_labeled_*.py") == 44 and
        counts.get("test_preserving_*.py") == 36 and counts.get("test_development_*.py") == 48 and
        all(type(counts.get(name)) is int and counts[name] >= 1 for name in
            ("test_preservation_development_seal_v2.py", "test_preservation_retry_submission.py")) and
        qa["tests_run"] == sum(counts.values()),
        "Native QA must include all 192 earlier checks and both exact retry modules")
    from extension.development_retry_submission import SOURCE_PATHS
    require(len(SOURCE_PATHS) == 38 and set(hashes) == set(SOURCE_PATHS) and
        str(plan_path.relative_to(root)) == "extension/development_retry_plan.json",
        "Native QA must bind exactly the 38 reviewed retry and immutable earlier sources")
    for name, checksum in hashes.items():
        file = inside(root, name)
        require(file.suffix in (".py", ".json", ".slurm") and 0 < file.stat().st_size <= 2 << 20 and
            not any(part in ("runs", "logs", "report") for part in Path(name).parts) and
            isinstance(inventory[name], dict) and inventory[name].get("sha256") == checksum and
            inventory[name].get("bytes") == file.stat().st_size, "Invalid QA source inventory")
        evidence.check(file, sha(checksum))
    return {"path": str(path), "sha256": evidence.check(path), "tests_run": qa["tests_run"],
        "code_sha256": hashes, "versions": qa.get("versions", {})}


def _project_receipt_path(root, name):
    """Old receipts contain native absolute paths; never remap foreign roots."""
    path = Path(name)
    require(path.is_absolute() and path.is_relative_to(root) and path == path.resolve(),
        "Prior source receipt path escapes this project")
    return path


def load_labeled(root, spec, evidence):
    directory = inside(root, spec["directory"])
    complete_path, qa_path = directory / "complete.json", directory / "native_qa.json"
    complete = evidence.read(complete_path, spec["complete_sha256"])
    qa = evidence.read(qa_path, spec["qa_sha256"])
    require(complete.get("schema") == "extension-labeled-feasibility-v1" and complete.get("complete") is True and
        complete.get("input_stats_unchanged_at_closeout") is True and complete.get("gpu_seconds_requested_or_used") == 0 and
        complete.get("lexical_criterion") == CRITERION and complete.get("actions") ==
            {"splits_chosen": False, "generation": False, "training": False, "automatic_downstream": False},
        "Labeled source receipt is incomplete or has downstream actions")
    previous = complete.get("native_qa", {})
    start, end = complete.get("compute_start", {}), complete.get("compute_end", {})
    require(previous.get("sha256") == spec["qa_sha256"] and previous.get("tests_run") == qa.get("tests_run") and
        previous.get("code_sha256") == qa.get("code_sha256") and qa.get("schema") == "extension-labeled-feasibility-native-qa-v1" and
        qa.get("success") is True and type(qa.get("tests_run")) is int and qa["tests_run"] >= 108 and
        all(type(qa.get(name)) is int and qa[name] == 0 for name in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")) and
        qa.get("gpu_disabled") is True and qa.get("wrapper_syntax_passed") is True and
        start.get("job_id") == end.get("job_id") == qa.get("job_id") == "1222926" and
        start.get("node") == end.get("node") == qa.get("node") and
        start.get("gpu_requested_or_allocated") is False and end.get("gpu_requested_or_allocated") is False,
        "Labeled native QA/allocation evidence differs")
    require(isinstance(complete.get("input_files"), dict) and complete["input_files"], "Missing labeled input snapshot")
    for name, entry in complete["input_files"].items():
        file = _project_receipt_path(root, name)
        require(isinstance(entry, dict) and type(entry.get("bytes")) is int and file.stat().st_size == entry["bytes"],
            "Prior input byte count changed")
        evidence.check(file, sha(entry.get("sha256")))
    outputs = complete.get("output_sha256", {})
    require(set(outputs) == {"surviving.jsonl", "surviving_ids.json", "groups.json", "excluded.jsonl", "quarantined.jsonl"},
        "Labeled output inventory differs")
    for name, checksum in outputs.items():
        evidence.check(directory / name, sha(checksum))
    records = []
    with (directory / "surviving.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            require(line.strip(), "Blank surviving JSONL row")
            record = json.loads(line, object_pairs_hook=_strict_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            records.append(record)
    groups = evidence.read(directory / "groups.json")
    by_id, eligible, excluded_groups = validate_population(records, groups)
    require(evidence.read(directory / "surviving_ids.json") == sorted(by_id), "Labeled ID order/seal differs")
    counts = Counter(row["source_kind"] for row in records)
    require(all(counts[domain] == complete.get("counts", {}).get(domain + ":surviving") for domain in DOMAINS) and
        len(records) == complete.get("groups", {}).get("surviving_records") and
        len(eligible) == complete.get("groups", {}).get("surviving_components") and
        len(groups) == complete.get("groups", {}).get("all_label_valid_components"),
        "Labeled records/components/count receipt disagree")
    evidence.unchanged()
    return by_id, eligible, {"directory": str(directory), "complete_sha256": spec["complete_sha256"],
        "qa_sha256": spec["qa_sha256"], "native_tests_run": qa["tests_run"], "output_sha256": outputs,
        "counts": complete["counts"], "groups": complete["groups"], "excluded_components": excluded_groups,
        "benchmark_readiness": complete.get("benchmark_readiness"), "source_metadata": complete.get("source_metadata"),
        "exclusions": complete.get("exclusions")}


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _write_jsonl(path, records):
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def assert_no_prior_overlap(reserved_ids, categories):
    """A same-criterion prior exclusion must already have removed these rows."""
    reserved, report = set(reserved_ids), {}
    for reason, identifiers in sorted(categories.items()):
        require(isinstance(identifiers, list) and len(set(identifiers)) == len(identifiers),
            "Prior pool category contains duplicate IDs")
        overlap = reserved.intersection(identifiers)
        require(not overlap, "Reservation unexpectedly overlaps prior excluded pool category: " + reason)
        report[reason] = {"prior_ids": len(identifiers), "reserved_overlap": 0}
    return report


def prior_pool_categories(root, source_audit, pool_path, evidence):
    categories = {}
    require(isinstance(source_audit.get("exclusions"), list), "Missing prior exclusion provenance")
    for entry in source_audit["exclusions"]:
        if entry.get("format") != "id_list" or inside(root, entry["lookup"]) != pool_path:
            continue
        path = _project_receipt_path(root, entry["path"])
        value = evidence.read(path, entry["sha256"])
        pointer = entry.get("json_pointer", "")
        require(pointer in ("", "/reservoir_ids"), "Unexpected prior-pool ID pointer")
        identifiers = value["reservoir_ids"] if pointer else value
        require(isinstance(identifiers, list) and identifiers and
            all(isinstance(identifier, str) and _SHA.fullmatch(identifier) for identifier in identifiers) and
            len(identifiers) == entry["question_records"] and entry["reason"] not in categories,
            "Prior pool exclusion category differs from source receipt")
        categories[entry["reason"]] = identifiers
    require(categories, "No prior training/probe/reservoir categories were bound")
    return categories


def validate_diagnostic_receipt(value):
    """The pinned diagnostic inspected metadata only, before any split choice."""
    require(isinstance(value, dict) and value.get("schema") == "extension-development-level-diagnostic-v1" and
        value.get("complete") is True and value.get("input_sha256") ==
            "f1304725d60087666fa2c3995fe7b4b344067ad28d1f2af35e8dbc5dab4e0551" and
        value.get("rows") == 7132 and value.get("math_rows") == 6013 and
        value.get("non_1_to_5_levels") == UNKNOWN_LEVELS and value.get("gpu_seconds") == 0 and
        value.get("no_labels_or_solutions_inspected") is True and value.get("no_split_assignment_or_model_outcomes") is True,
        "Unexpected or incomplete metadata-only level diagnostic")
    compute = value.get("compute", {})
    require(isinstance(compute, dict) and compute.get("job_id") == DIAGNOSTIC["job_id"] and
        compute.get("gpu_requested_or_allocated") is False and compute.get("state") == "RUNNING" and
        isinstance(compute.get("node"), str) and bool(compute["node"]),
        "Diagnostic lacks its original CPU allocation proof")
    counts = value.get("subject_level_counts")
    require(isinstance(counts, list) and counts and all(isinstance(row, dict) and
        set(row) == {"subject", "official_level", "rows"} and row["subject"] in MATH_COUNTS and
        isinstance(row["official_level"], str) and
        (re.fullmatch(r"Level [1-5]", row["official_level"]) or row["official_level"] in UNKNOWN_LEVELS) and
        type(row["rows"]) is int and row["rows"] > 0 for row in counts), "Invalid diagnostic subject/level counts")
    keys = [(row["subject"], row["official_level"]) for row in counts]
    require(len(set(keys)) == len(keys) and sum(row["rows"] for row in counts) == value["math_rows"] and
        dict(Counter({level: sum(row["rows"] for row in counts if row["official_level"] == level)
            for level in UNKNOWN_LEVELS})) == UNKNOWN_LEVELS, "Diagnostic counts do not reconcile")
    return value


def validate_level_counts(records, diagnostic, plan):
    """Recompute the exact raw metadata counts; do not alter or remove any row."""
    math_rows = [row for row in records if row["source_kind"] == "math"]
    actual = Counter((row["source"]["file"].split("/")[0], row["original_record"]["level"]) for row in math_rows)
    invalid = Counter(row["original_record"]["level"] for row in math_rows
        if re.fullmatch(r"Level [1-5]", row["original_record"]["level"]) is None)
    expected = {(row["subject"], row["official_level"]): row["rows"] for row in diagnostic["subject_level_counts"]}
    require(dict(invalid) == diagnostic["non_1_to_5_levels"] == plan["math_unknown_levels"] == UNKNOWN_LEVELS and
        dict(actual) == expected and len(math_rows) == diagnostic["math_rows"] and len(records) == diagnostic["rows"],
        "Actual native levels/subject counts differ from diagnostic and sealed plan")
    return {"math_rows": len(math_rows), "all_surviving_rows": len(records),
        "unknown_levels": dict(sorted(invalid.items())),
        "subject_level_counts": [{"subject": subject, "official_level": level, "rows": count}
            for (subject, level), count in sorted(actual.items())],
        "unknown_stratum_label": "unknown:Level ?", "raw_records_unchanged": True}


def seal(root, plan_path, output, qa_receipt):
    compute, begun = require_cpu_allocation(), time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = inside(root, str(plan_path))
    require(output.parent == root / "extension/runs" and output.name.endswith("-" + compute["job_id"]) and
        output.is_dir() and {path.name for path in output.iterdir()} == {"native_qa.json"},
        "Use the exclusive current-job directory containing only native_qa.json")
    evidence = Evidence()
    plan = validate_plan(evidence.read(plan_path))
    guard = deadline_guard(plan["wall_time_seconds"] + plan["reserve_seconds"])
    qa = validate_native_qa(Path(qa_receipt).resolve(), root, output, compute, evidence, plan_path)
    write_new_json(output / "started.json", {"schema": SCHEMA, "complete": False, "compute": compute,
        "plan_sha256": evidence.check(plan_path), "native_qa": qa, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    diagnostic_path = inside(root, plan["level_diagnostic"]["directory"]) / "complete.json"
    diagnostic = validate_diagnostic_receipt(evidence.read(diagnostic_path, plan["level_diagnostic"]["complete_sha256"]))
    by_id, groups, source_audit = load_labeled(root, plan["labeled_audit"], evidence)
    require(diagnostic["input_sha256"] == source_audit["output_sha256"]["surviving.jsonl"],
        "Diagnostic and labeled surviving population differ")
    level_audit = validate_level_counts(list(by_id.values()), diagnostic, plan)
    assigned, unused, statistics = allocate_groups(groups)
    selected_records, outputs, split_receipts = [], [], {}
    for split in SPLITS:
        chosen = sorted(assigned[split], key=lambda group: group["group_id"])
        identifiers = sorted(identifier for group in chosen for identifier in group["member_ids"])
        records = [by_id[identifier] for identifier in identifiers]
        _write_jsonl(output / (split + ".jsonl"), records)
        write_new_json(output / (split + "_ids.json"), identifiers)
        write_new_json(output / (split + "_groups.json"), chosen)
        outputs.extend([split + ".jsonl", split + "_ids.json", split + "_groups.json"])
        split_receipts[split] = {"rows": len(records), "group_count": len(chosen),
            "domain_rows": dict(sorted(Counter(row["source_kind"] for row in records).items())),
            "ids_sha256": value_digest(identifiers), "group_ids_sha256": value_digest([group["group_id"] for group in chosen])}
        require(split_receipts[split]["domain_rows"] == TARGETS[split], "Exact per-domain split targets were not achieved")
        selected_records.extend(records)
    require(len({row["id"] for row in selected_records}) == len(selected_records), "Selected records overlap")
    write_new_json(output / "unused_groups.json", unused)
    outputs.append("unused_groups.json")
    pool_path = inside(root, plan["pool"]["path"])
    evidence.check(pool_path, plan["pool"]["sha256"])
    questions = canonical_questions(pool_path, evidence)
    require(len(questions) == plan["pool"].get("expected_rows", 83029), "Canonical training-pool count differs")
    reserved_ids, matches = reserve_pool(questions, selected_records)
    prior_overlap = assert_no_prior_overlap(reserved_ids,
        prior_pool_categories(root, source_audit, pool_path, evidence))
    write_new_json(output / "reserved_training_pool_ids.json", reserved_ids)
    write_new_json(output / "reserved_training_pool_matches.json", matches)
    outputs.extend(["reserved_training_pool_ids.json", "reserved_training_pool_matches.json"])
    evidence.unchanged()
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    output_hashes = {name: digest(output / name) for name in outputs}
    evidence.unchanged()
    receipt = {"schema": SCHEMA, "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": evidence.check(plan_path), "native_qa": qa, "compute_start": compute, "compute_end": ending,
        "deadline_guard": guard, "versions": {**qa["versions"], "python": sys.version}, "gpu_seconds_requested_or_used": 0,
        "source_audit": source_audit, "input_files": evidence.files, "input_stats_unchanged_at_closeout": True,
        "level_diagnostic": {**DIAGNOSTIC, "path": str(diagnostic_path), "level_audit": level_audit},
        "seed": 20261021, "split_order": list(SPLITS), "targets": TARGETS, "hash_template": HASH_TEMPLATE,
        "splits": split_receipts, "allocation": statistics, "output_sha256": output_hashes,
        "selected_record_ids_sha256": value_digest(sorted(row["id"] for row in selected_records)),
        "reservation": {"training_pool_rows_scanned": len(questions), "reserved_training_pool_rows": len(reserved_ids),
            "reserved_training_pool_ids_sha256": value_digest(reserved_ids), "lexical_criterion": CRITERION,
            "exact_match_rows": sum("exact" in row["match_kinds"] for row in matches),
            "near_match_rows": sum("near" in row["match_kinds"] for row in matches),
            "prior_pool_overlap_check": prior_overlap,
            "scope": "All selected component members: MATH question; ARC stem and ordered stem-plus-options. Correct answer keys and solution text are never indexed.",
            "required_future_policy": "Every future training/probe selector must verify this complete seal and exclude every reserved_training_pool_id; all selected labeled groups remain held out."},
        "limitations": ["Source-only metadata stratification; no answer-based selection.",
            "Exact for the declared lexical criterion, not a semantic leakage guarantee.",
            "Greedy exact-reachability checks are not a globally complete multi-split partition solver.",
            "This seal records reservations; future selectors must explicitly enforce them.",
            "No tokenization, model inference, correctness scoring, generation or training occurred."],
        "actions": {"splits_chosen": True, "generation": False, "training": False, "automatic_downstream": False},
        "elapsed_seconds": time.monotonic() - begun}
    write_new_json(output / "complete.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="extension/development_retry_plan.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args()
    receipt = seal(args.root, args.plan, args.output, args.qa_receipt)
    print(json.dumps({"complete": True, "splits": receipt["splits"], "reservation": receipt["reservation"]}, indent=2))


if __name__ == "__main__":
    main()
