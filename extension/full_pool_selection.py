"""Unadmitted pure-stdlib implementation of the prospectively approved selector.

Functions consume metadata maps/receipts only. They do not load files, fit topics,
score a model, inspect text/answers, change targets, execute a CLI or authorize
their own use. Real-data calls require a separately admitted CPU allocation and
caller verification of actual source/cache/exclusion/runtime FILE identities.
Hash checks here bind supplied metadata; they cannot verify unseen file bytes.

R is a separate once-only stage with no score/topic argument. S consumes that
sealed R receipt and never resamples it. External one-use execution/journaling
remains the caller's obligation: a stateless function cannot prevent its caller
from invoking it again. There is no redraw/refill/fallback API. Small explicit
count fixtures are reported NONNATIVE and establish no actual population draw.
No function in this source was called while authoring it.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from types import MappingProxyType


PROTOCOL_SHA256 = "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29"
APPROVAL_SHA256 = "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6"
POOL_FILE_SHA256 = "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0"
EXCLUSION_FILE_SHA256 = "0066e69a996cf963173018d68196d7ff69e658420c40b250941c2fe8bb6da406"
DEVELOPMENT_COMPLETE_SHA256 = "354859dd44e2fe3aa7800ec991b60649f00a99ce4768d8075af42c007289987d"
BENCHMARK_AUDIT_SHA256 = "2671a58d0abf9f52de5445649aecf9a389363f0fe54d8f05dbf5bca243c6fa6f"
BENCHMARK_IDS_FILE_SHA256 = "5a687438e7b5c31dbc16f2671582b448ff970739219328dbf9c5836474a89984"
BASE_IDENTITY_SHA256 = "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2"
SEED, TOPICS = 20261021, 64
LENGTH_BINS = ((1, 2048), (2049, 4096), (4097, 8192), (8193, 16384), (16385, 32768))
CELLS = tuple((topic, length) for topic in range(TOPICS) for length in range(len(LENGTH_BINS)))
PURPOSES = frozenset({"S15000.coverage", "S15000.order", "S2048.low_tag", "S2048.coverage_tag", "S2048.order"})
FRACTIONS = ((0, 1), (1, 10), (1, 4), (1, 2), (3, 4), (9, 10), (19, 20), (99, 100), (1, 1))
ROW_KEYS = frozenset({"total_tokens", "active_tokens", "original_pair_sha256", "cache_record_sha256"})
SCORE_KEYS = frozenset({"active_tokens", "nll_sum", "nll_sum_hex", "mean_ce", "mean_ce_hex",
    "original_pair_sha256", "cache_record_sha256", "score_definition_sha256",
    "loaded_base_identity_sha256", "runtime_precision_identity_sha256", "components"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, low, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _plain(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def canonical_value_bytes(value):
    """VALUE encoding, never a hash of an implied physical file/newline."""
    _require(_plain(value), "Only finite plain JSON values may be canonically bound")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_sha256(value):
    return hashlib.sha256(canonical_value_bytes(value)).hexdigest()


def _copy(value):
    return json.loads(canonical_value_bytes(value))


def _ids(values, count, label):
    _require(type(values) in (list, tuple) and len(values) == count and all(_sha(value) for value in values) and
        len(set(values)) == count, "Exact unique canonical IDs required for " + label)
    return tuple(sorted(values))


@dataclass(frozen=True)
class SelectionCounts:
    pool: int = 83029
    reserved: int = 67
    benchmark: int = 1
    eligible: int = 82961
    full: int = 15000
    pilot: int = 2048
    fixture_only: bool = False

    def __post_init__(self):
        _require(type(self.fixture_only) is bool and all(_int(getattr(self, key), 0) for key in
            ("pool", "reserved", "benchmark", "eligible", "full", "pilot")) and
            self.pool - self.reserved - self.benchmark == self.eligible and 0 < self.pilot <= self.full <= self.eligible,
            "Invalid explicit population/arm counts")
        _require(self.fixture_only or (self.pool, self.reserved, self.benchmark, self.eligible, self.full, self.pilot) ==
            (83029, 67, 1, 82961, 15000, 2048), "Small count seams must be explicitly NONNATIVE fixtures")

    def value(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _counts_match(value, counts):
    expected = counts.value()
    return (type(value) is dict and set(value) == set(expected) and
        type(value["fixture_only"]) is bool and value["fixture_only"] is counts.fixture_only and
        all(type(value[name]) is int and value[name] == expected[name]
            for name in expected if name != "fixture_only"))


@dataclass(frozen=True)
class CacheRow:
    total_tokens: int
    active_tokens: int
    original_pair_sha256: str
    cache_record_sha256: str

    def value(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Population:
    counts: SelectionCounts
    pool_ids: tuple
    reserved_ids: tuple
    benchmark_ids: tuple
    eligible_ids: tuple
    rows: object
    identity_json: bytes
    identity_sha256: str

    @property
    def identity_receipt(self):
        return json.loads(self.identity_json)


@dataclass(frozen=True)
class TopicPopulation:
    population: Population
    assignments: object
    cells: object
    assignments_sha256: str
    receipt_json: bytes

    @property
    def identity_receipt(self):
        return json.loads(self.receipt_json)


def length_bin(total_tokens):
    _require(_int(total_tokens, 1, 32768), "Length-bin input must be a complete integer total length")
    return next(index for index, (low, high) in enumerate(LENGTH_BINS) if low <= total_tokens <= high)


def _row_values(rows, ordered):
    return [{"id": identity, **rows[identity].value()} for identity in ordered]


def validate_population(pool_ids, reserved_ids, benchmark_ids, rows, identity_receipt, *, counts=SelectionCounts()):
    """Validate supplied complete-cache/exclusion metadata, without reading it.

    rows maps eligible ID to exactly ROW_KEYS (no question/answer/old-score text).
    identity_receipt uses extension-full-pool-selection-inputs-v1 and the exact
    keys below. The caller MUST independently verify every physical-file pin and
    actual Python3.10.19/random source/runtime before a real invocation.
    """
    _require(type(counts) is SelectionCounts, "Explicit validated count contract required")
    pool = _ids(pool_ids, counts.pool, "source pool")
    reserved = _ids(reserved_ids, counts.reserved, "reserved evaluation")
    benchmark = _ids(benchmark_ids, counts.benchmark, "final benchmark questions")
    _require(not set(reserved) & set(benchmark) and set(reserved) | set(benchmark) <= set(pool),
        "Exclusion components must be disjoint members of the canonical pool")
    excluded = tuple(sorted(set(reserved) | set(benchmark)))
    excluded_set = set(excluded)
    eligible = tuple(identity for identity in pool if identity not in excluded_set)
    _require(len(excluded) == counts.reserved + counts.benchmark and len(eligible) == counts.eligible and
        type(rows) is dict and set(rows) == set(eligible), "Every eligible complete-cache record must occur exactly once")
    checked = {}
    for identity in eligible:
        row = rows[identity]
        _require(type(row) is dict and set(row) == ROW_KEYS and _int(row["total_tokens"], 2, 32736) and
            _int(row["active_tokens"], 1, row["total_tokens"] - 1) and
            _sha(row["original_pair_sha256"]) and _sha(row["cache_record_sha256"]),
            "A complete untruncated native pair with32 unused context tokens is required")
        checked[identity] = CacheRow(**row)
    computed = {"pool_sorted_ids_sha256": value_sha256(list(pool)), "reserved_sorted_ids_sha256": value_sha256(list(reserved)),
        "benchmark_sorted_ids_sha256": value_sha256(list(benchmark)), "excluded_sorted_ids_sha256": value_sha256(list(excluded)),
        "eligible_sorted_ids_sha256": value_sha256(list(eligible)), "cache_rows_value_sha256": value_sha256(_row_values(checked, eligible))}
    fixed = {"pool_file_sha256": POOL_FILE_SHA256, "combined_exclusion_receipt_file_sha256": EXCLUSION_FILE_SHA256,
        "development_seal_complete_file_sha256": DEVELOPMENT_COMPLETE_SHA256,
        "benchmark_audit_file_sha256": BENCHMARK_AUDIT_SHA256, "benchmark_ids_file_sha256": BENCHMARK_IDS_FILE_SHA256,
        "base_identity_file_sha256": BASE_IDENTITY_SHA256}
    other_hashes = {"cache_complete_file_sha256", "source_identity_sha256", "tokenizer_identity_sha256",
        "python_runtime_identity_sha256", "python_random_source_sha256"}
    keys = set(computed) | set(fixed) | other_hashes | {"schema", "complete", "counts", "fixture_only",
        "protocol_file_sha256", "root_approval_file_sha256", "python_version", "caller_verified_files_and_runtime"}
    receipt = _copy(identity_receipt)
    _require(type(receipt) is dict and set(receipt) == keys and receipt["schema"] == "extension-full-pool-selection-inputs-v1" and
        receipt["complete"] is True and _counts_match(receipt["counts"], counts) and receipt["fixture_only"] is counts.fixture_only and
        receipt["protocol_file_sha256"] == PROTOCOL_SHA256 and receipt["root_approval_file_sha256"] == APPROVAL_SHA256 and
        receipt["python_version"] == "3.10.19" and receipt["caller_verified_files_and_runtime"] is True and
        all(receipt[name] == value for name, value in computed.items()) and
        all(_sha(receipt[name]) for name in set(fixed) | other_hashes) and
        (counts.fixture_only or all(receipt[name] == value for name, value in fixed.items())),
        "Caller cache/source/exclusion/runtime identities do not bind the exact supplied population")
    payload = canonical_value_bytes(receipt)
    return Population(counts, pool, reserved, benchmark, eligible, MappingProxyType(checked), payload, hashlib.sha256(payload).hexdigest())


def _population(value):
    _require(type(value) is Population and type(value.counts) is SelectionCounts and
        type(value.identity_json) is bytes and _sha(value.identity_sha256) and
        hashlib.sha256(value.identity_json).hexdigest() == value.identity_sha256 and
        all(type(getattr(value, name)) is tuple for name in ("pool_ids", "reserved_ids", "benchmark_ids", "eligible_ids")) and
        type(value.rows) is MappingProxyType and all(type(row) is CacheRow for row in value.rows.values()),
        "Use the unchanged validated metadata population")
    # Public dataclass constructors are not evidence. Revalidate their complete
    # immutable value once per public operation, never once per cell RNG call.
    checked = validate_population(value.pool_ids, value.reserved_ids, value.benchmark_ids,
        {identity: row.value() for identity, row in value.rows.items()}, value.identity_receipt, counts=value.counts)
    _require(value == checked, "Validated population canonical order, cache or identity changed")
    return value


def draw_random_order(population):
    """ONE broad R draw per caller-admitted stage; no score/topic input exists."""
    population = _population(population)
    order = random.Random(SEED).sample(list(population.eligible_ids), population.counts.full)
    return {"schema": "extension-full-pool-random-order-v1", "complete": True,
        "fixture_only": population.counts.fixture_only, "population_identity_sha256": population.identity_sha256,
        "eligible_sorted_ids_sha256": population.identity_receipt["eligible_sorted_ids_sha256"],
        "python_runtime_identity_sha256": population.identity_receipt["python_runtime_identity_sha256"],
        "python_random_source_sha256": population.identity_receipt["python_random_source_sha256"],
        "seed": SEED, "algorithm": "random.Random(20261021).sample(sorted_eligible_ids,full_count)",
        "rng_sample_calls": 1, "ordered_ids": order, "ordered_ids_sha256": value_sha256(order),
        "sorted_ids_sha256": value_sha256(sorted(order)), "counts": population.counts.value(),
        "physical_file_or_one_use_execution_verified_here": False, "execution_admitted_here": False}


def validate_random_order(population, receipt):
    population = _population(population)
    return _random_order(population, receipt)


def _random_order(population, receipt):
    _require(type(receipt) is dict, "The original once-frozen R receipt is required")
    expected = {"schema", "complete", "fixture_only", "population_identity_sha256", "eligible_sorted_ids_sha256",
        "python_runtime_identity_sha256", "python_random_source_sha256", "seed", "algorithm", "rng_sample_calls",
        "ordered_ids", "ordered_ids_sha256", "sorted_ids_sha256", "counts",
        "physical_file_or_one_use_execution_verified_here", "execution_admitted_here"}
    _require(set(receipt) == expected and receipt["schema"] == "extension-full-pool-random-order-v1" and receipt["complete"] is True and
        receipt["fixture_only"] is population.counts.fixture_only and receipt["population_identity_sha256"] == population.identity_sha256 and
        _counts_match(receipt["counts"], population.counts) and type(receipt["seed"]) is int and receipt["seed"] == SEED and
        type(receipt["rng_sample_calls"]) is int and receipt["rng_sample_calls"] == 1 and
        receipt["algorithm"] == "random.Random(20261021).sample(sorted_eligible_ids,full_count)" and
        receipt["physical_file_or_one_use_execution_verified_here"] is False and receipt["execution_admitted_here"] is False,
        "Original R receipt configuration/count/lineage differs")
    _require(type(receipt["ordered_ids"]) is list, "R order must retain its exact plain JSON array")
    ordered = tuple(receipt["ordered_ids"])
    _ids(ordered, population.counts.full, "R")
    _require(set(ordered) <= set(population.eligible_ids) and receipt["ordered_ids_sha256"] == value_sha256(list(ordered)) and
        receipt["sorted_ids_sha256"] == value_sha256(sorted(ordered)) and
        all(receipt[name] == population.identity_receipt[name] for name in
            ("eligible_sorted_ids_sha256", "python_runtime_identity_sha256", "python_random_source_sha256")),
        "R membership/order/runtime identity changed; no redraw is permitted")
    return ordered


def bind_topics(population, assignments, receipt):
    population = _population(population)
    return _topics(population, assignments, receipt)


def _topics(population, assignments, receipt):
    _require(type(assignments) is dict and set(assignments) == set(population.eligible_ids) and
        all(_int(value, 0, TOPICS - 1) for value in assignments.values()), "Every eligible ID needs one actual topic0..63")
    records = [{"id": identity, "topic": assignments[identity]} for identity in population.eligible_ids]
    checksum = value_sha256(records)
    receipt = _copy(receipt)
    _require(type(receipt) is dict and set(receipt) == {"schema", "complete", "fixture_only", "population_identity_sha256",
        "eligible_sorted_ids_sha256", "topic_assignments_sha256", "topic_fit_receipt_file_sha256", "topic_runtime_identity_sha256"} and
        receipt["schema"] == "extension-full-pool-topic-bindings-v1" and receipt["complete"] is True and
        receipt["fixture_only"] is population.counts.fixture_only and receipt["population_identity_sha256"] == population.identity_sha256 and
        receipt["eligible_sorted_ids_sha256"] == population.identity_receipt["eligible_sorted_ids_sha256"] and
        receipt["topic_assignments_sha256"] == checksum and _sha(receipt["topic_fit_receipt_file_sha256"]) and
        _sha(receipt["topic_runtime_identity_sha256"]), "Caller topic-fit identity does not bind all actual assignments")
    cells = {cell: [] for cell in CELLS}
    for identity in population.eligible_ids:
        cells[(assignments[identity], length_bin(population.rows[identity].total_tokens))].append(identity)
    return TopicPopulation(population, MappingProxyType(dict(assignments)),
        MappingProxyType({cell: tuple(values) for cell, values in cells.items()}), checksum, canonical_value_bytes(receipt))


def cell_namespace(population, topic_assignments_sha256, random_ordered_ids_sha256, purpose, cell):
    population = _population(population)
    return _namespace(population, topic_assignments_sha256, random_ordered_ids_sha256, purpose, cell)


def _namespace(population, topic_assignments_sha256, random_ordered_ids_sha256, purpose, cell):
    _require(_sha(topic_assignments_sha256) and _sha(random_ordered_ids_sha256) and type(purpose) is str and purpose in PURPOSES and
        type(cell) is tuple and len(cell) == 2 and _int(cell[0], 0, 63) and _int(cell[1], 0, 4), "Exact purpose/cell namespace required")
    return {"protocol": "extension-full-pool-selector-v1", "seed": SEED, "purpose": purpose,
        "eligible_sorted_ids_sha256": population.identity_receipt["eligible_sorted_ids_sha256"],
        "topic_assignments_sha256": topic_assignments_sha256, "random_ordered_ids_sha256": random_ordered_ids_sha256,
        "cell": {"topic": cell[0], "length_bin": cell[1]}}


def _sample(population, topics_sha256, random_sha256, purpose, cell, available, count):
    _require(type(count) is int and 0 <= count <= len(available) and len(set(available)) == len(available), "Insufficient exact tag/cell capacity; never refill")
    namespace = _namespace(population, topics_sha256, random_sha256, purpose, cell)
    checksum = value_sha256(namespace)
    if count == 0:
        return [], {"purpose": purpose, "namespace_value_sha256": checksum, "sample_calls": 0, "count": 0}
    seed = int.from_bytes(bytes.fromhex(checksum), byteorder="big", signed=False)
    sampled = random.Random(seed).sample(sorted(available), count)
    return sampled, {"purpose": purpose, "namespace_value_sha256": checksum, "sample_calls": 1, "count": count}


def _float_hex(number, text, label):
    _require(type(number) is float and math.isfinite(number) and number >= 0 and type(text) is str,
        "Every " + label + " must be a finite nonnegative binary64 float and canonical hex")
    try:
        decoded = float.fromhex(text)
    except (ValueError, OverflowError) as error:
        raise ValueError("Malformed binary64 hexadecimal score") from error
    _require(math.isfinite(decoded) and decoded.hex() == number.hex() == text, "Binary64 score/hex identity differs")
    return number


def _scores(population, random_receipt, random_receipt_file_sha256, scores, receipt):
    _require(_sha(random_receipt_file_sha256) and type(scores) is dict and set(scores) == set(population.eligible_ids),
        "One complete score for every eligible ID and the exact frozen R FILE pin required")
    receipt = _copy(receipt)
    keys = {"schema", "complete", "fixture_only", "population_identity_sha256", "eligible_sorted_ids_sha256",
        "cache_complete_file_sha256", "random_order_receipt_file_sha256", "random_ordered_ids_sha256", "score_rows_value_sha256",
        "score_definition_sha256", "loaded_base_identity_sha256", "runtime_precision_identity_sha256", "score_complete_file_sha256"}
    _require(type(receipt) is dict and set(receipt) == keys and receipt["schema"] == "extension-full-pool-complete-score-bindings-v1" and
        receipt["complete"] is True and receipt["fixture_only"] is population.counts.fixture_only and
        receipt["population_identity_sha256"] == population.identity_sha256 and
        receipt["eligible_sorted_ids_sha256"] == population.identity_receipt["eligible_sorted_ids_sha256"] and
        receipt["cache_complete_file_sha256"] == population.identity_receipt["cache_complete_file_sha256"] and
        receipt["random_order_receipt_file_sha256"] == random_receipt_file_sha256 and
        receipt["random_ordered_ids_sha256"] == random_receipt["ordered_ids_sha256"] and
        all(_sha(receipt[name]) for name in ("score_definition_sha256", "loaded_base_identity_sha256", "runtime_precision_identity_sha256", "score_complete_file_sha256")) and
        receipt["loaded_base_identity_sha256"] == population.identity_receipt["base_identity_file_sha256"],
        "Complete score/model/cache/runtime/R receipt bindings differ")
    means, records = {}, []
    for identity in population.eligible_ids:
        row, original = scores[identity], population.rows[identity]
        _require(type(row) is dict and set(row) == SCORE_KEYS and type(row["active_tokens"]) is int and row["active_tokens"] == original.active_tokens and
            row["original_pair_sha256"] == original.original_pair_sha256 and row["cache_record_sha256"] == original.cache_record_sha256 and
            all(row[name] == receipt[name] for name in ("score_definition_sha256", "loaded_base_identity_sha256", "runtime_precision_identity_sha256")),
            "Original score row/cache/model identity differs")
        total = _float_hex(row["nll_sum"], row["nll_sum_hex"], "full-target sum")
        mean = _float_hex(row["mean_ce"], row["mean_ce_hex"], "full-target mean")
        _require((total / original.active_tokens).hex() == mean.hex(), "Mean must divide the exact full active-token sum once")
        parts = row["components"]
        _require(type(parts) is dict and set(parts) == {"eos", "non_eos", "tail"}, "Complete finite EOS/non-EOS/tail diagnostics required")
        for name, part in parts.items():
            _require(type(part) is dict and set(part) == {"count", "nll_sum", "nll_sum_hex"} and
                _int(part["count"], 1 if name in ("eos", "tail") else 0, original.active_tokens), "Invalid complete score component count")
            subtotal = _float_hex(part["nll_sum"], part["nll_sum_hex"], name + " sum")
            _require(part["count"] > 0 or subtotal == 0., "Empty diagnostic subset must have a zero sum")
        _require(parts["eos"]["count"] + parts["non_eos"]["count"] == original.active_tokens,
            "EOS/non-EOS counts must partition every active target")
        means[identity] = mean
        records.append({"id": identity, **row})
    _require(receipt["score_rows_value_sha256"] == value_sha256(records), "Exact complete score VALUE hash differs")
    return means, receipt


def _paired_order(reference_order, lookup, queues):
    pending = {cell: deque(values) for cell, values in queues.items()}
    result = []
    for identity in reference_order:
        cell = lookup[identity]
        _require(pending[cell], "Paired cell-order queue exhausted early; never replace")
        result.append(pending[cell].popleft())
    _require(all(not values for values in pending.values()), "Paired cell-order queues did not exhaust exactly")
    _require(len(result) == len(set(result)) == len(reference_order), "Duplicate or incomplete paired selected order")
    return result


def select_full_and_pilot(topics, random_receipt, scores, score_receipt, *, random_receipt_file_sha256):
    """Consume complete fixed inputs; never draw R, fit topics or use outcomes."""
    _require(type(topics) is TopicPopulation, "Bound complete actual topic assignments required")
    population = _population(topics.population)
    random_order = _random_order(population, random_receipt)
    _require(type(topics.assignments) is MappingProxyType and type(topics.cells) is MappingProxyType and
        type(topics.receipt_json) is bytes and
        topics == _topics(population, dict(topics.assignments), topics.identity_receipt),
        "Topic assignments changed after binding")
    means, score_binding = _scores(population, random_receipt, random_receipt_file_sha256, scores, score_receipt)
    lookup = {identity: (topics.assignments[identity], length_bin(population.rows[identity].total_tokens)) for identity in population.eligible_ids}
    # _topics above rebuilds and compares every cell in one ascending-ID pass;
    # no repeated full-population scan is needed for the320 cell tables.
    quotas = Counter(lookup[identity] for identity in random_order)
    pilot_random = list(random_order[:population.counts.pilot])
    pilot_quotas = Counter(lookup[identity] for identity in pilot_random)
    full_queues, pilot_queues, tags, cells = {}, {}, {}, []
    random_sha = random_receipt["ordered_ids_sha256"]
    for cell in CELLS:
        members, quota, pilot_quota = topics.cells[cell], quotas[cell], pilot_quotas[cell]
        _require(0 <= pilot_quota <= quota <= len(members), "Cell quotas/capacity differ")
        low = sorted(members, key=lambda identity: (means[identity], identity))[:quota // 2]
        low_set = set(low)
        coverage, full_cover_rng = _sample(population, topics.assignments_sha256, random_sha, "S15000.coverage", cell,
            [identity for identity in members if identity not in low_set], quota - len(low))
        selected = sorted(low + coverage)
        _require(len(selected) == len(set(selected)) == quota, "Full half-mixture must be unique and exact")
        full_queues[cell], full_order_rng = _sample(population, topics.assignments_sha256, random_sha, "S15000.order", cell, selected, quota)
        for identity in low:
            tags[identity] = "low_ce"
        for identity in coverage:
            tags[identity] = "seeded_coverage"
        pilot_low, pilot_low_rng = _sample(population, topics.assignments_sha256, random_sha, "S2048.low_tag", cell, low, pilot_quota // 2)
        pilot_coverage, pilot_cover_rng = _sample(population, topics.assignments_sha256, random_sha, "S2048.coverage_tag", cell,
            coverage, pilot_quota - len(pilot_low))
        pilot_members = sorted(pilot_low + pilot_coverage)
        _require(len(pilot_members) == len(set(pilot_members)) == pilot_quota and set(pilot_members) <= set(selected),
            "Nested pilot tags must be exact subsets of the already selected full tags")
        pilot_queues[cell], pilot_order_rng = _sample(population, topics.assignments_sha256, random_sha, "S2048.order", cell, pilot_members, pilot_quota)
        cells.append({"cell": {"topic": cell[0], "length_bin": cell[1]}, "population_count": len(members),
            "full_quota": quota, "low_ce_count": len(low), "coverage_count": len(coverage), "pilot_quota": pilot_quota,
            "pilot_low_ce_count": len(pilot_low), "pilot_coverage_count": len(pilot_coverage),
            "rng": [full_cover_rng, full_order_rng, pilot_low_rng, pilot_cover_rng, pilot_order_rng]})
    full = _paired_order(random_order, lookup, full_queues)
    pilot = _paired_order(pilot_random, lookup, pilot_queues)
    _require(len(full) == population.counts.full and len(pilot) == population.counts.pilot and set(pilot) <= set(full) and
        set(full) <= set(population.eligible_ids) and len(tags) == population.counts.full and
        [lookup[identity] for identity in full] == [lookup[identity] for identity in random_order] and
        [lookup[identity] for identity in pilot] == [lookup[identity] for identity in pilot_random], "Final paired count/order/subset invariants differ")
    return {"schema": "extension-full-pool-selected-orders-v1", "complete": True, "fixture_only": population.counts.fixture_only,
        "native_population_or_method_improvement_established": False, "execution_admitted_here": False,
        "counts": population.counts.value(), "population_identity_sha256": population.identity_sha256,
        "topic_assignments_sha256": topics.assignments_sha256, "score_binding_value_sha256": value_sha256(score_binding),
        "random_receipt_file_sha256": random_receipt_file_sha256, "random_ordered_ids_sha256": random_sha,
        "selected_ordered_ids": full, "selected_ordered_ids_sha256": value_sha256(full), "selected_sorted_ids_sha256": value_sha256(sorted(full)),
        "random_pilot_ordered_ids": pilot_random, "random_pilot_ordered_ids_sha256": value_sha256(pilot_random),
        "selected_pilot_ordered_ids": pilot, "selected_pilot_ordered_ids_sha256": value_sha256(pilot),
        "selected_pilot_sorted_ids_sha256": value_sha256(sorted(pilot)),
        "selection_tags": [{"id": identity, "tag": tags[identity]} for identity in sorted(tags)],
        "full_random_selected_overlap_count": len(set(full) & set(random_order)),
        "pilot_random_selected_overlap_count": len(set(pilot) & set(pilot_random)), "cells": cells,
        "zero_quota_cells_recorded": True, "no_redraw_refill_fallback": True,
        "source_targets_modified": False, "gold_or_development_outcomes_consumed": False}


def length_probe_ids(population):
    """Fixed integer ranks of complete lengths only, before any model score."""
    population = _population(population)
    ordered = sorted(population.eligible_ids, key=lambda identity: (population.rows[identity].total_tokens, identity))
    selected, ranks = [], []
    for numerator, denominator in FRACTIONS:
        rank = numerator * (len(ordered) - 1) // denominator
        identity = ordered[rank]
        if identity not in selected:
            selected.append(identity)
            ranks.append(rank)
    return {"schema": "extension-full-pool-length-probe-ids-v1", "complete": True,
        "fixture_only": population.counts.fixture_only, "population_identity_sha256": population.identity_sha256,
        "ordered_ids": selected, "ordered_ids_sha256": value_sha256(selected), "ranks": ranks,
        "length_only": True, "score_or_topic_inputs_used": False, "execution_admitted_here": False}
