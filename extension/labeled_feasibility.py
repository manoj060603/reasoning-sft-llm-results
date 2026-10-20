"""CPU-only labeled training-source inventory; no split choice or generation.

The lexical near-duplicate criterion is exact, not semantic decontamination.
An inverted index plus a mathematically complete query prefix finds every
qualifying pair; full set intersections determine the integer4/5 threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from extension.cpu_feasibility import (Evidence, deadline_guard, digest, read_json,
    require, require_cpu_allocation, value_digest, write_new_json)
from reasoning.data import boxed, normalize, prompt_id

SCHEMA = "extension-labeled-feasibility-v1"
MATH_REVISION = "fe39eaeb3783a7dd5d7e305da9a8db9b0c4e6b93"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
MATH_COUNTS = {"algebra": 1744, "counting_and_probability": 771, "geometry": 870,
    "intermediate_algebra": 1295, "number_theory": 869, "prealgebra": 1205, "precalculus": 746}
SOURCE_PINS = {
    "EleutherAI/hendrycks_math": ("math", MATH_REVISION,
        tuple(name + "/train/0000.parquet" for name in sorted(MATH_COUNTS))),
    "allenai/ai2_arc": ("arc_challenge", ARC_REVISION,
        ("ARC-Challenge/train-00000-of-00001.parquet",))}
CRITERION = {"normalization": "NFKC casefold whitespace; preserve numbers/operators",
    "tokenization": "word-or-single-nonwhitespace-character", "shingle_tokens": 3,
    "jaccard_numerator": 4, "jaccard_denominator": 5, "search": "exact inverted index",
    "short_question_policy": "exact normalized equality when fewer than3tokens"}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
BENCHMARK_COUNTS = {"aime": 30, "aime25": 30, "amc": 40, "cn_math_2024": 30,
    "gpqa": 198, "kaoyan": 199, "math": 500, "minerva": 272, "olympiadbench": 675}
LICENSE_CARDS = {"EleutherAI/hendrycks_math": {
    "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3", "filename": "README.md", "declared_license": "mit"},
    "allenai/ai2_arc": {"revision": ARC_REVISION, "filename": "README.md", "declared_license": "cc-by-sa-4.0"}}


def lexical(text):
    require(isinstance(text, str) and bool(text.strip()), "Question must be a nonempty string")
    normalized = normalize(text)
    tokens = tuple(TOKEN_PATTERN.findall(normalized))
    grams = frozenset(tuple(tokens[index:index + 3]) for index in range(max(0, len(tokens) - 2)))
    return normalized, tokens, grams


def near_duplicate(first, second):
    a, at, ag = lexical(first)
    b, bt, bg = lexical(second)
    if a == b:
        return True
    if len(at) < 3 or len(bt) < 3:
        return False
    intersection = len(ag & bg)
    return 5 * intersection >= 4 * (len(ag) + len(bg) - intersection)


class ExactIndex:
    """All shingles are indexed; query only a complete, rare-first prefix.

    A Jaccard>=4/5 match needs at least ceil(4*|S|/5) members of S. Every
    subset of S of size |S|-ceil(4*|S|/5)+1 therefore intersects that match.
    Rare-first ordering reduces candidates without weakening completeness.
    """
    def __init__(self):
        self.entries, self.exact, self.postings = [], {}, defaultdict(list)

    def add(self, text, owner):
        require(isinstance(owner, str) and owner, "Index owner must be nonempty")
        normalized, tokens, grams = lexical(text)
        if normalized in self.exact:
            self.entries[self.exact[normalized]]["owners"].add(owner)
            return
        index = len(self.entries)
        self.exact[normalized] = index
        self.entries.append({"tokens": len(tokens), "grams": grams, "owners": {owner}})
        for gram in sorted(grams):
            self.postings[gram].append(index)

    def query(self, text):
        normalized, tokens, grams = lexical(text)
        result = defaultdict(set)
        exact = self.exact.get(normalized)
        if exact is not None:
            for owner in self.entries[exact]["owners"]:
                result[owner].add("exact")
        if len(tokens) >= 3:
            minimum_overlap = (4 * len(grams) + 4) // 5
            prefix_size = len(grams) - minimum_overlap + 1
            prefix = sorted(grams, key=lambda gram: (len(self.postings.get(gram, ())), gram))[:prefix_size]
            candidates = {index for gram in prefix for index in self.postings.get(gram, ())}
            for index in sorted(candidates):
                entry = self.entries[index]
                if index == exact or entry["tokens"] < 3:
                    continue
                other = entry["grams"]
                if 5 * min(len(grams), len(other)) < 4 * max(len(grams), len(other)):
                    continue
                intersection = len(grams & other)
                if 5 * intersection >= 4 * (len(grams) + len(other) - intersection):
                    for owner in entry["owners"]:
                        result[owner].add("near")
        return {owner: sorted(kinds) for owner, kinds in sorted(result.items())}


def _nonempty(value, name):
    require(isinstance(value, str) and value.strip(), "Nonempty string required: " + name)
    return value


def source_record(kind, raw, source):
    """Strict schema; unavailable official labels are explicitly quarantined."""
    require(isinstance(raw, dict), "Source row must be an object")
    if kind == "math":
        require(set(raw) == {"problem", "level", "type", "solution"} and
                all(isinstance(value, str) for value in raw.values()), "Unsupported MATH row schema")
        question = _nonempty(raw["problem"], "MATH problem")
        prompt, variants = question, [question]
        answer = boxed(raw["solution"])
        label = {"answer": answer, "extraction": "reasoning.data.boxed; final balanced literal boxed group"}
        quarantine = None if answer is not None else "unextractable_final_boxed_answer"
    elif kind == "arc_challenge":
        require(set(raw) == {"id", "question", "choices", "answerKey"} and
                isinstance(raw["id"], str) and isinstance(raw["question"], str) and
                (isinstance(raw["answerKey"], str) or raw["answerKey"] is None), "Unsupported ARC row schema")
        _nonempty(raw["id"], "ARC id")
        question = _nonempty(raw["question"], "ARC question")
        choices = raw["choices"]
        require(isinstance(choices, dict) and set(choices) == {"label", "text"} and
                isinstance(choices["label"], list) and isinstance(choices["text"], list) and
                len(choices["label"]) == len(choices["text"]) >= 2 and
                all(isinstance(value, str) and value.strip() for value in choices["label"] + choices["text"]) and
                len(set(choices["label"])) == len(choices["label"]), "Unsupported ARC choice schema")
        prompt = question + "\n\n" + "\n".join(label + ". " + text for label, text in zip(choices["label"], choices["text"]))
        variants = [question, prompt]
        matches = [index for index, key in enumerate(choices["label"]) if key == raw["answerKey"]]
        label = {"answer_key": raw["answerKey"], "choice_index": matches[0] if len(matches) == 1 else None,
                 "extraction": "exact original answerKey-to-choice-label match; no label remapping"}
        quarantine = None if len(matches) == 1 else "missing_or_unmatched_answer_key"
    else:
        raise ValueError("Unsupported source kind")
    identity = {"source": source, "question_sha256": prompt_id(question)}
    record = {"id": value_digest(identity), "source": source, "source_kind": kind,
        "question": question, "question_sha256": prompt_id(question), "prompt": prompt,
        "prompt_sha256": prompt_id(prompt), "lexical_screen_variants": variants,
        "label": label, "label_id": value_digest(label), "label_sha256": value_digest(label), "original_record": raw}
    return record, quarantine


def group_and_filter(records, exclusion_index):
    """Connected components first; propagate any direct exclusion to its group."""
    require(isinstance(records, list), "Expected validated labeled records")
    parents, by_id, seen_questions = {}, {}, ExactIndex()

    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    def union(first, second):
        a, b = find(first), find(second)
        parents[max(a, b)] = min(a, b)

    direct = {}
    for record in records:
        identifier = record["id"]
        require(identifier not in by_id, "Duplicate labeled source-record identity")
        by_id[identifier], parents[identifier] = record, identifier
        reasons = defaultdict(set)
        for text in record["lexical_screen_variants"]:
            for other in seen_questions.query(text):
                union(identifier, other)
            for reason, kinds in exclusion_index.query(text).items():
                reasons[reason].update(kinds)
        for text in record["lexical_screen_variants"]:
            seen_questions.add(text, identifier)
        if reasons:
            direct[identifier] = {reason: sorted(kinds) for reason, kinds in sorted(reasons.items())}
    components = defaultdict(list)
    for identifier in sorted(by_id):
        components[find(identifier)].append(identifier)
    surviving, excluded, groups = [], [], []
    for members in sorted(components.values(), key=lambda items: items[0]):
        group_id = value_digest({"criterion": CRITERION, "member_ids": members})
        direct_members = [identifier for identifier in members if identifier in direct]
        reasons = sorted({reason for identifier in direct_members for reason in direct[identifier]})
        groups.append({"group_id": group_id, "member_ids": members, "excluded": bool(direct_members),
                       "directly_excluded_member_ids": direct_members, "group_exclusion_reasons": reasons})
        for identifier in members:
            if direct_members:
                excluded.append({"id": identifier, "source": by_id[identifier]["source"],
                    "question_sha256": by_id[identifier]["question_sha256"], "group_id": group_id,
                    "direct_matches": direct.get(identifier, {}), "group_exclusion_reasons": reasons,
                    "propagated_group_exclusion": identifier not in direct})
            else:
                surviving.append(dict(by_id[identifier], group_id=group_id))
    return sorted(surviving, key=lambda row: row["id"]), sorted(excluded, key=lambda row: row["id"]), groups


def inside(root, value):
    require(isinstance(value, str) and value, "Missing project-relative input path")
    path = (root / value).resolve()
    require(not Path(value).is_absolute() and path.is_relative_to(root) and not (root / value).is_symlink(),
            "Input path must remain inside the project")
    return path


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == "extension-labeled-feasibility-plan-v1" and
        plan.get("cpu_only") is True and plan.get("model_inference") is False and
        plan.get("choose_splits") is False and plan.get("automatic_downstream") is False,
        "Expected a CPU-only labeled feasibility plan")
    require(plan.get("deadline_utc") == "2026-10-28T03:59:00+00:00" and
        plan.get("wall_time_seconds") == 7200 and plan.get("reserve_seconds") == 172800,
        "Unsealed deadline/lexical criterion")
    near, grouping = plan.get("near_duplicate", {}), plan.get("grouping_policy", {})
    require(near == CRITERION and isinstance(grouping, dict) and
        grouping.get("within_labeled") == "connected components across all label-valid records" and
        grouping.get("exclusion_propagation") == "Exclude entire labeled component if any member directly matches a declared prior or benchmark exclusion",
        "Unsealed lexical grouping/propagation policy")
    sources = plan.get("sources")
    require(isinstance(sources, list) and len(sources) == 2 and
            {value.get("repo_id") for value in sources if isinstance(value, dict)} == set(SOURCE_PINS), "Only the two pinned train sources are permitted")
    for value in sources:
        kind, revision, filenames = SOURCE_PINS[value["repo_id"]]
        require(value.get("revision") == revision and isinstance(value.get("files"), list) and
                len(value["files"]) == len(set(value["files"])) and set(value["files"]) == set(filenames),
                "Unexpected source revision/files; test or ARC-Easy data is forbidden")
        require(value.get("kind") == kind and value.get("expected_rows") == (7500 if kind == "math" else 1119) and
                value.get("license_card") == LICENSE_CARDS[value["repo_id"]], "Source count/license-card pin differs")
    require(isinstance(plan.get("exclusions"), list) and bool(plan["exclusions"]), "Explicit exclusion inventory required")
    return plan


def _rows(path, evidence, jsonl=False):
    evidence.check(path)
    if jsonl:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                require(bool(line.strip()), "Blank exclusion JSONL line")
                value = json.loads(line)
                require(isinstance(value, dict), "Exclusion JSONL requires object rows")
                yield value
    else:
        values = evidence.read(path)
        require(isinstance(values, list), "Exclusion JSON must be a row array")
        for value in values:
            require(isinstance(value, dict), "Exclusion requires object rows")
            yield value
    evidence.check(path)


def canonical_questions(path, evidence):
    result = {}
    for value in _rows(path, evidence):
        require(set(value) == {"id", "instruction", "input", "output", "system"} and
                isinstance(value["input"], str) and value["input"] == "", "Unsupported canonical lookup schema")
        text = _nonempty(value.get("instruction"), "lookup instruction")
        identifier = value.get("id")
        require(identifier == prompt_id(text) and identifier not in result, "Duplicate/incorrect lookup identity")
        result[identifier] = text
    require(bool(result), "Empty canonical question lookup")
    return result


def _pointer(value, pointer):
    require(isinstance(pointer, str) and (not pointer or pointer.startswith("/")), "Invalid JSON pointer")
    for key in pointer.split("/")[1:] if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        require(isinstance(value, dict) and key in value, "Exclusion JSON pointer is absent")
        value = value[key]
    return value


def load_exclusions(root, entries, evidence):
    index, receipt, lookups, viewed = ExactIndex(), [], {}, set()
    for spec in entries:
        require(isinstance(spec, dict), "Exclusion specification must be an object")
        path, reason = inside(root, spec.get("path")), _nonempty(spec.get("reason"), "exclusion reason")
        format_name, texts = spec.get("format"), []
        if format_name == "canonical_json":
            texts = list(canonical_questions(path, evidence).values())
        elif format_name == "id_list":
            lookup = inside(root, spec.get("lookup"))
            if "lookup_sha256" in spec:
                evidence.check(lookup, spec["lookup_sha256"])
            if lookup not in lookups:
                lookups[lookup] = canonical_questions(lookup, evidence)
            identifiers = _pointer(evidence.read(path), spec.get("json_pointer", ""))
            require(isinstance(identifiers, list) and bool(identifiers) and
                    all(isinstance(value, str) and value in lookups[lookup] for value in identifiers) and
                    len(set(identifiers)) == len(identifiers), "Exclusion IDs must uniquely resolve in their declared lookup")
            texts = [lookups[lookup][identifier] for identifier in identifiers]
            if reason.startswith("viewed_"):
                require(not viewed.intersection(identifiers), "Previously viewed development categories overlap")
                viewed.update(identifiers)
        elif format_name == "benchmark_jsonl":
            field = _nonempty(spec.get("question_field"), "benchmark question_field")
            require(field in ("question", "problem", "Question"), "Unsupported benchmark question field")
            for row in _rows(path, evidence, jsonl=True):
                # Only the explicit question field enters this question-only index.
                texts.append(_nonempty(row.get(field), "benchmark question"))
        else:
            raise ValueError("Unsupported exclusion schema: " + str(format_name))
        require(bool(texts), "Empty exclusion inventory")
        if "expected_rows" in spec:
            require(type(spec["expected_rows"]) is int and len(texts) == spec["expected_rows"], "Exclusion row count differs")
        if "sha256" in spec:
            evidence.check(path, spec["sha256"])
        for text in texts:
            index.add(text, reason)
        receipt.append({"path": str(path), "sha256": evidence.check(path), "format": format_name,
                        "reason": reason, "question_records": len(texts),
                        "lookup": spec.get("lookup"), "lookup_sha256": spec.get("lookup_sha256"),
                        "question_field": spec.get("question_field"), "json_pointer": spec.get("json_pointer", "")})
    return index, receipt


def validate_benchmarks(root, plan, evidence):
    spec = plan.get("benchmark_readiness", {})
    path = inside(root, spec.get("path"))
    readiness = evidence.read(path, spec.get("sha256"))
    binding = {key: value for key, value in readiness.items() if key != "binding_sha256"}
    checksum = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(readiness.get("schema") == "deadline-benchmark-readiness-v1" and readiness.get("complete") is True and
            readiness.get("binding_sha256") == checksum, "Benchmark readiness seal is invalid")
    original = readiness.get("evidence", {})
    entries = [item for item in plan["exclusions"] if item.get("format") == "benchmark_jsonl"]
    names = [Path(item["path"]).parent.name for item in entries]
    expected_occurrences = {name: (2 if name == "amc" else 1) for name in BENCHMARK_COUNTS}
    require(len(entries) == 10 and set(names) == set(BENCHMARK_COUNTS) and
            Counter(names) == Counter(expected_occurrences) and
            {item.get("question_field") for item in entries if Path(item["path"]).parent.name == "amc"} == {"question", "problem"} and
            original.get("benchmark_counts") == BENCHMARK_COUNTS and
            set(original.get("benchmark_hashes", {})) == set(BENCHMARK_COUNTS), "All nine benchmark question sources are required")
    for item in entries:
        name = Path(item["path"]).parent.name
        require(item.get("expected_rows") == BENCHMARK_COUNTS[name] and
                item.get("sha256") == original["benchmark_hashes"][name] and
                inside(root, item["path"]) == root / "deadline/artifacts/benchmarks" / name / "test.jsonl",
                "Benchmark plan path/hash/count differs from readiness")
    require(original.get("supplied_benchmark_hashes") == {name: original["benchmark_hashes"][name]
            for name in BENCHMARK_COUNTS if name != "aime25"}, "Supplied-eight source identities differ")
    return {"path": str(path), "sha256": evidence.check(path), "benchmark_hashes": original["benchmark_hashes"],
            "benchmark_counts": BENCHMARK_COUNTS}


def card_license(text, expected):
    """Verify conventional scalar/list card fields when present; preserve bytes."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {"field_present": False, "declared_license": expected}
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        raise ValueError("Unterminated source-card metadata")
    header = lines[1:end]
    for index, line in enumerate(header):
        if not re.match(r"^license\s*:", line):
            continue
        value = line.split(":", 1)[1].strip().strip("\"'")
        if value:
            values = [value]
        else:
            values = []
            for child in header[index + 1:]:
                if re.match(r"^\s*-\s+", child):
                    values.append(re.sub(r"^\s*-\s+", "", child).strip().strip("\"'"))
                else:
                    break
        require(expected in values, "Source-card license field differs from frozen declaration")
        return {"field_present": True, "declared_license": expected, "recorded_values": values}
    return {"field_present": False, "declared_license": expected}


def default_download(spec, filename, destination):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
        filename=filename, local_dir=str(destination)))


def parquet_rows(path):
    import pyarrow.parquet as parquet
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def validate_qa(path, root, output, compute, evidence, plan_path, metadata_path):
    require(path.resolve() == output / "native_qa.json", "Use the reserved output's native_qa.json")
    qa = evidence.read(path)
    inventory, hashes = qa.get("source_inventory"), qa.get("code_sha256")
    require(qa.get("schema") == "extension-labeled-feasibility-native-qa-v1" and qa.get("success") is True and
            type(qa.get("tests_run")) is int and qa["tests_run"] > 0 and
            all(type(qa.get(key)) is int and qa[key] == 0 for key in ("failures", "errors", "skipped")) and
            all(type(qa.get(key)) is int and qa[key] == 0 for key in ("expected_failures", "unexpected_successes")) and
            qa.get("gpu_disabled") is True and qa.get("wrapper_syntax_passed") is True and
            qa.get("job_id") == compute["job_id"] and qa.get("node") == compute["node"] and
            isinstance(inventory, dict) and isinstance(hashes, dict) and set(inventory) == set(hashes),
            "Native CPU QA is missing, failed or belongs to another allocation")
    proof = qa.get("compute_proof", {})
    require(isinstance(proof, dict) and proof.get("job_id") == compute["job_id"] and
            proof.get("node") == compute["node"] and proof.get("gpu_requested_or_allocated") is False,
            "Native QA lacks its matching CPU allocation proof")
    required = {"extension/labeled_feasibility.py", "tests/test_labeled_feasibility.py",
        "extension/cpu_feasibility.py", "reasoning/data.py", "extension/labeled_feasibility.slurm",
        str(plan_path.relative_to(root)), str(metadata_path.relative_to(root))}
    require(required <= set(hashes), "Native QA omits required labeled sources/helpers/plan")
    for name, checksum in hashes.items():
        file = inside(root, name)
        require(file.suffix in (".py", ".json", ".slurm") and 0 < file.stat().st_size <= 2 << 20 and
                not any(part in ("runs", "logs", "report") for part in Path(name).parts) and
                isinstance(inventory[name], dict) and inventory[name].get("sha256") == checksum and
                inventory[name].get("bytes") == file.stat().st_size, "Invalid native source inventory")
        evidence.check(file, checksum)
    return {"path": str(path), "sha256": evidence.check(path), "tests_run": qa["tests_run"], "code_sha256": hashes}


def audit(root, plan_path, output, qa_receipt, *, downloader=None, reader=None):
    compute = require_cpu_allocation()
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = inside(root, str(plan_path))
    require(output.parent == root / "extension/runs" and output.name.endswith("-" + compute["job_id"]) and
            output.is_dir() and {path.name for path in output.iterdir()} == {"native_qa.json"},
            "Use the exclusive current-job run directory containing only native_qa.json")
    evidence, begun = Evidence(), time.monotonic()
    plan = validate_plan(evidence.read(plan_path))
    deadline_guard(plan["wall_time_seconds"] + plan["reserve_seconds"])
    metadata_path = inside(root, plan.get("source_metadata"))
    metadata = evidence.read(metadata_path)
    qa = validate_qa(Path(qa_receipt).resolve(), root, output, compute, evidence, plan_path, metadata_path)
    write_new_json(output / "started.json", {"schema": SCHEMA, "complete": False, "compute": compute,
        "plan_sha256": evidence.check(plan_path), "native_qa": qa, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    require(metadata.get("schema") == "reasoning-sft-fresh-development-source-pins-v1", "Unsupported source pin metadata")
    require(isinstance(metadata.get("sources"), list) and len(metadata["sources"]) == 2 and
            all(isinstance(value, dict) and value.get("repo_id") in SOURCE_PINS for value in metadata["sources"]),
            "Unexpected source metadata inventory")
    pins = {value["repo_id"]: value for value in metadata["sources"]}
    require(set(pins) == set(SOURCE_PINS), "Duplicate/missing source metadata identity")
    benchmark_binding = validate_benchmarks(root, plan, evidence)
    exclusions, exclusion_receipts = load_exclusions(root, plan["exclusions"], evidence)
    download, read = downloader or default_download, reader or parquet_rows
    records, quarantined, downloaded, counts = [], [], [], Counter({kind + ":" + status: 0
        for kind in ("math", "arc_challenge") for status in
        ("raw", "label_valid", "quarantined", "surviving", "direct_excluded", "propagated_excluded")})
    source_native_ids = defaultdict(set)
    for spec in sorted(plan["sources"], key=lambda value: value["repo_id"]):
        kind, revision, filenames = SOURCE_PINS[spec["repo_id"]]
        require(spec["repo_id"] in pins and pins[spec["repo_id"]].get("resolved_revision") == revision and
                set(filenames) <= set(pins[spec["repo_id"]].get("files", [])), "Source metadata pin differs from plan")
        card = spec["license_card"]
        card_directory = output / "downloads" / kind / "license-card"
        card_path = Path(download(dict(spec, revision=card["revision"]), card["filename"], card_directory)).resolve()
        require(card_path.is_relative_to(card_directory.resolve()), "Source license card escaped its isolated directory")
        card_sha = evidence.check(card_path)
        card_metadata = card_license(card_path.read_text(encoding="utf-8"), card["declared_license"])
        evidence.check(card_path, card_sha)
        downloaded.append({"repository": spec["repo_id"], "revision": card["revision"], "file": card["filename"],
            "path": str(card_path), "sha256": card_sha, "role": "license_card", **card_metadata})
        for filename in filenames:
            directory = output / "downloads" / kind
            file = Path(download(spec, filename, directory)).resolve()
            require(file.is_relative_to(directory.resolve()) and file.is_file(), "Downloaded file is outside its isolated source directory")
            checksum = evidence.check(file)
            file_rows = 0
            for ordinal, raw in enumerate(read(file)):
                source = {"repository": spec["repo_id"], "revision": revision, "file": filename, "row_ordinal": ordinal}
                record, unavailable = source_record(kind, raw, source)
                if kind == "arc_challenge":
                    require(raw["id"] not in source_native_ids[kind], "Duplicate official ARC source ID")
                    source_native_ids[kind].add(raw["id"])
                file_rows += 1
                counts[kind + ":raw"] += 1
                if unavailable:
                    quarantined.append({"id": record["id"], "source": source,
                        "question_sha256": record["question_sha256"], "reason": unavailable})
                    counts[kind + ":quarantined"] += 1
                else:
                    records.append(record)
                    counts[kind + ":label_valid"] += 1
            expected = MATH_COUNTS[filename.split("/")[0]] if kind == "math" else 1119
            require(file_rows == expected, "Official source file row count differs: " + filename)
            evidence.check(file, checksum)
            downloaded.append({"repository": spec["repo_id"], "revision": revision, "file": filename,
                               "path": str(file), "sha256": checksum, "rows": file_rows})
        require(counts[kind + ":raw"] == spec["expected_rows"], "Total source rows differ from plan")
    surviving, excluded, groups = group_and_filter(records, exclusions)
    for record in surviving:
        counts[record["source_kind"] + ":surviving"] += 1
    for record in excluded:
        kind = SOURCE_PINS[record["source"]["repository"]][0]
        counts[kind + (":propagated_excluded" if record["propagated_group_exclusion"] else ":direct_excluded")] += 1
    for kind in ("math", "arc_challenge"):
        require(counts[kind + ":raw"] == counts[kind + ":quarantined"] + counts[kind + ":surviving"] +
                counts[kind + ":direct_excluded"] + counts[kind + ":propagated_excluded"], "Source accounting does not reconcile")
    for filename, values in (("surviving.jsonl", surviving), ("excluded.jsonl", excluded), ("quarantined.jsonl", quarantined)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            for record in values:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    write_new_json(output / "surviving_ids.json", [row["id"] for row in surviving])
    write_new_json(output / "groups.json", groups)
    outputs = {name: digest(output / name) for name in
        ("surviving.jsonl", "surviving_ids.json", "excluded.jsonl", "quarantined.jsonl", "groups.json")}
    evidence.unchanged()
    ending = require_cpu_allocation()
    require(ending["job_id"] == compute["job_id"] and ending["node"] == compute["node"], "CPU allocation changed")
    receipt = {"schema": SCHEMA, "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": evidence.check(plan_path), "native_qa": qa, "compute_start": compute, "compute_end": ending,
        "gpu_seconds_requested_or_used": 0, "versions": {"python": sys.version,
            **{name: importlib.metadata.version(name) for name in ("pyarrow", "huggingface_hub")}},
        "input_files": evidence.files, "input_stats_unchanged_at_closeout": True,
        "downloaded_sources": downloaded, "source_metadata": metadata, "benchmark_readiness": benchmark_binding,
        "exclusions": exclusion_receipts,
        "counts": dict(sorted(counts.items())), "groups": {"all_label_valid_components": len(groups),
            "surviving_components": sum(not group["excluded"] for group in groups),
            "surviving_records": len(surviving), "quarantined_records": len(quarantined),
            "direct_excluded_records": sum(not row["propagated_group_exclusion"] for row in excluded),
            "propagated_excluded_records": sum(row["propagated_group_exclusion"] for row in excluded)},
        "direct_reason_counts": dict(sorted(Counter(reason for row in excluded for reason in row["direct_matches"]).items())),
        "output_sha256": outputs, "lexical_criterion": CRITERION,
        "screening_scope": "MATH problem; ARC question stem and separately rendered stem plus all original ordered choice labels/text. Main ARC question identity uses stem only. Correct answer keys and solution text never enter question identities or exclusion text.",
        "propagation": "Connected components across all label-valid new records; any direct prior/benchmark match excludes the entire component. No old-exclusion-to-old-exclusion graph.",
        "limitations": "Exact for the declared lexical criterion, not a semantic decontamination guarantee. Official boxed answers are extracted, not independently verified. License metadata is preserved without inventing a license for the parquet conversion.",
        "actions": {"splits_chosen": False, "generation": False, "training": False, "automatic_downstream": False},
        "elapsed_seconds": time.monotonic() - begun}
    write_new_json(output / "complete.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="extension/labeled_plan.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-receipt", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.plan, args.output, args.qa_receipt)
    print(json.dumps({"complete": True, "counts": result["counts"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
