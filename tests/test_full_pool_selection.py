"""Prospective stdlib selector fixtures; no real fit, score, data draw or admission.

All IDs, cache records and scores below are synthetic.  Every small population
sets fixture_only=True.  Independent JSON/SHA, cell, rank and stdlib-RNG oracles
describe the frozen mathematical policy, not evidence that physical source
files, Python runtime, scorer reductions or a one-use invocation were verified.
This file is authored for a future admitted CPU QA; authoring is not execution.
"""
from collections import Counter, defaultdict
import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import inspect
import json
import math
import random
from types import MappingProxyType
import unittest
from unittest import mock

from extension import full_pool_selection as selector


REAL_RANDOM = random.Random
PROTOCOL = "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29"
APPROVAL = "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6"
SEED = 20261021
BIN_ENDS = (2048, 4096, 8192, 16384, 32768)
PURPOSES = ("S15000.coverage", "S15000.order", "S2048.low_tag",
            "S2048.coverage_tag", "S2048.order")
DEFAULT_R_INDICES = (0, 12, 21, 1, 13, 22, 2, 14, 23, 3, 15, 24, 4, 16, 5)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def pin(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def identity(number):
    return format(number, "064x")


def own_bin(total):
    return next(index for index, upper in enumerate(BIN_ENDS) if total <= upper)


def own_namespace(case, purpose, cell):
    return {"protocol": "extension-full-pool-selector-v1", "seed": SEED,
            "purpose": purpose,
            "eligible_sorted_ids_sha256": digest(list(case.eligible)),
            "topic_assignments_sha256": digest(
                [{"id": item, "topic": case.assignments[item]} for item in case.eligible]),
            "random_ordered_ids_sha256": digest(case.random_ids),
            "cell": {"topic": cell[0], "length_bin": cell[1]}}


def own_sample(case, purpose, cell, available, count):
    namespace_hash = digest(own_namespace(case, purpose, cell))
    sampled = [] if count == 0 else REAL_RANDOM(int(namespace_hash, 16)).sample(sorted(available), count)
    return sampled, {"purpose": purpose, "namespace_value_sha256": namespace_hash,
                     "sample_calls": int(count > 0), "count": count}


def independent_selection(case):
    """Reference prescription using independent values, queues and RNG objects."""
    lookup = {item: (case.assignments[item], own_bin(case.rows[item]["total_tokens"]))
              for item in case.eligible}
    members = defaultdict(list)
    for item in case.eligible:
        members[lookup[item]].append(item)
    full_quota = Counter(lookup[item] for item in case.random_ids)
    random_pilot = case.random_ids[:case.counts.pilot]
    pilot_quota = Counter(lookup[item] for item in random_pilot)
    full_queues, pilot_queues, tags, cells = {}, {}, {}, []
    for topic in range(64):
        for length in range(5):
            cell = (topic, length)
            q, p = full_quota[cell], pilot_quota[cell]
            low = sorted(members[cell], key=lambda item: (case.scores[item]["mean_ce"], item))[:q // 2]
            coverage, cover_rng = own_sample(case, PURPOSES[0], cell,
                [item for item in members[cell] if item not in set(low)], q - len(low))
            full_queues[cell], full_rng = own_sample(case, PURPOSES[1], cell, low + coverage, q)
            tags.update({item: "low_ce" for item in low})
            tags.update({item: "seeded_coverage" for item in coverage})
            low_pilot, low_rng = own_sample(case, PURPOSES[2], cell, low, p // 2)
            cover_pilot, pc_rng = own_sample(case, PURPOSES[3], cell, coverage, p - p // 2)
            pilot_queues[cell], pilot_rng = own_sample(case, PURPOSES[4], cell,
                                                      low_pilot + cover_pilot, p)
            cells.append({"cell": {"topic": topic, "length_bin": length},
                "population_count": len(members[cell]), "full_quota": q,
                "low_ce_count": len(low), "coverage_count": len(coverage),
                "pilot_quota": p, "pilot_low_ce_count": len(low_pilot),
                "pilot_coverage_count": len(cover_pilot),
                "rng": [cover_rng, full_rng, low_rng, pc_rng, pilot_rng]})

    def pair(reference, queues):
        used = Counter()
        result = []
        for item in reference:
            cell = lookup[item]
            result.append(queues[cell][used[cell]])
            used[cell] += 1
        if any(used[cell] != len(queue) for cell, queue in queues.items()):
            raise AssertionError("Reference oracle did not consume each cell exactly")
        return result

    return {"full": pair(case.random_ids, full_queues),
            "pilot": pair(random_pilot, pilot_queues), "tags": tags,
            "cells": cells, "lookup": lookup}


class SyntheticCase:
    """Caller-sealed metadata fixture, with a separately explicit synthetic R."""

    def __init__(self, eligible=28, full=15, pilot=9, *, layout="mixed", means=None, active_tokens=None):
        self.counts = selector.SelectionCounts(pool=eligible + 3, reserved=2,
            benchmark=1, eligible=eligible, full=full, pilot=pilot, fixture_only=True)
        self.pool = [identity(index) for index in range(1, eligible + 4)]
        self.reserved, self.benchmark = self.pool[:2], self.pool[-1:]
        self.eligible = tuple(self.pool[2:-1])
        self.rows, self.assignments = {}, {}
        for index, item in enumerate(self.eligible):
            if layout == "one":
                topic, total = 7, 1024
            elif index < eligible * 3 // 7:
                topic, total = 0, (1024, 2048)[index % 2]
            elif index < eligible * 3 // 4:
                topic, total = 1, (4097, 8192)[index % 2]
            else:
                topic, total = 63, (16385, 32736)[index % 2]
            self.assignments[item] = topic
            self.rows[item] = {"total_tokens": total,
                "active_tokens": 3 + index % 4 if active_tokens is None else active_tokens,
                "original_pair_sha256": pin("unchanged-original:" + item),
                "cache_record_sha256": pin("native-complete-cache:" + item)}
        self.inputs = self.input_receipt()
        self.population = self.make_population()
        self.topic_receipt = self.make_topic_receipt()
        self.topics = selector.bind_topics(self.population, self.assignments, self.topic_receipt)
        if eligible == 28 and full == 15 and layout == "mixed":
            self.random_ids = [self.eligible[index] for index in DEFAULT_R_INDICES]
        else:
            self.random_ids = list(self.eligible[:full])
        self.random_receipt = self.make_random_receipt()
        self.random_file_pin = pin("physical-frozen-R-file-in-this-fixture")
        self.definition = pin("full-original-active-target-mean-ce")
        self.precision = pin("caller-verified-inference-precision")
        self.scores = {}
        for index, item in enumerate(self.eligible):
            mean = float((index % 7 + 1) / 8) if means is None else float(means[index])
            self.scores[item] = self.score_row(item, mean)
        self.score_receipt = self.make_score_receipt(self.scores)

    def input_receipt(self):
        excluded = sorted(self.reserved + self.benchmark)
        return {"schema": "extension-full-pool-selection-inputs-v1", "complete": True,
            "counts": self.counts.value(), "fixture_only": True,
            "protocol_file_sha256": PROTOCOL, "root_approval_file_sha256": APPROVAL,
            "python_version": "3.10.19", "caller_verified_files_and_runtime": True,
            "pool_sorted_ids_sha256": digest(sorted(self.pool)),
            "reserved_sorted_ids_sha256": digest(sorted(self.reserved)),
            "benchmark_sorted_ids_sha256": digest(sorted(self.benchmark)),
            "excluded_sorted_ids_sha256": digest(excluded),
            "eligible_sorted_ids_sha256": digest(list(self.eligible)),
            "cache_rows_value_sha256": digest([{"id": item, **self.rows[item]} for item in self.eligible]),
            "pool_file_sha256": pin("fixture-pool-physical-file"),
            "combined_exclusion_receipt_file_sha256": pin("fixture-combined-exclusion-file"),
            "development_seal_complete_file_sha256": pin("fixture-development-complete-file"),
            "benchmark_audit_file_sha256": pin("fixture-final-benchmark-audit-file"),
            "benchmark_ids_file_sha256": pin("fixture-final-benchmark-IDs-file"),
            "base_identity_file_sha256": pin("fixture-original-base-identity-file"),
            "cache_complete_file_sha256": pin("fixture-complete-native-cache-file"),
            "source_identity_sha256": pin("fixture-source-identity"),
            "tokenizer_identity_sha256": pin("fixture-native-tokenizer-identity"),
            "python_runtime_identity_sha256": pin("fixture-verified-python-runtime"),
            "python_random_source_sha256": pin("fixture-verified-python-random-source")}

    def make_population(self, *, pool=None, reserved=None, benchmark=None, rows=None, receipt=None):
        return selector.validate_population(self.pool if pool is None else pool,
            self.reserved if reserved is None else reserved, self.benchmark if benchmark is None else benchmark,
            self.rows if rows is None else rows, self.inputs if receipt is None else receipt, counts=self.counts)

    def make_topic_receipt(self):
        return {"schema": "extension-full-pool-topic-bindings-v1", "complete": True,
            "fixture_only": True, "population_identity_sha256": self.population.identity_sha256,
            "eligible_sorted_ids_sha256": digest(list(self.eligible)),
            "topic_assignments_sha256": digest(
                [{"id": item, "topic": self.assignments[item]} for item in self.eligible]),
            "topic_fit_receipt_file_sha256": pin("fixture-one-full-question-topic-fit-file"),
            "topic_runtime_identity_sha256": pin("fixture-topic-runtime")}

    def make_random_receipt(self):
        return {"schema": "extension-full-pool-random-order-v1", "complete": True,
            "fixture_only": True, "population_identity_sha256": self.population.identity_sha256,
            "eligible_sorted_ids_sha256": digest(list(self.eligible)),
            "python_runtime_identity_sha256": self.inputs["python_runtime_identity_sha256"],
            "python_random_source_sha256": self.inputs["python_random_source_sha256"],
            "seed": SEED, "algorithm": "random.Random(20261021).sample(sorted_eligible_ids,full_count)",
            "rng_sample_calls": 1, "ordered_ids": list(self.random_ids),
            "ordered_ids_sha256": digest(self.random_ids), "sorted_ids_sha256": digest(sorted(self.random_ids)),
            "counts": self.counts.value(), "physical_file_or_one_use_execution_verified_here": False,
            "execution_admitted_here": False}

    def score_row(self, item, mean):
        source = self.rows[item]
        active = source["active_tokens"]
        total = float(mean * active)

        def part(count):
            subtotal = float(mean * count)
            return {"count": count, "nll_sum": subtotal, "nll_sum_hex": subtotal.hex()}

        return {"active_tokens": active, "nll_sum": total, "nll_sum_hex": total.hex(),
            "mean_ce": mean, "mean_ce_hex": mean.hex(),
            "original_pair_sha256": source["original_pair_sha256"],
            "cache_record_sha256": source["cache_record_sha256"],
            "score_definition_sha256": self.definition,
            "loaded_base_identity_sha256": self.inputs["base_identity_file_sha256"],
            "runtime_precision_identity_sha256": self.precision,
            "components": {"eos": part(1), "non_eos": part(active - 1), "tail": part(1)}}

    def make_score_receipt(self, scores):
        return {"schema": "extension-full-pool-complete-score-bindings-v1", "complete": True,
            "fixture_only": True, "population_identity_sha256": self.population.identity_sha256,
            "eligible_sorted_ids_sha256": digest(list(self.eligible)),
            "cache_complete_file_sha256": self.inputs["cache_complete_file_sha256"],
            "random_order_receipt_file_sha256": self.random_file_pin,
            "random_ordered_ids_sha256": digest(self.random_ids),
            "score_rows_value_sha256": digest([{"id": item, **scores[item]} for item in self.eligible]),
            "score_definition_sha256": self.definition,
            "loaded_base_identity_sha256": self.inputs["base_identity_file_sha256"],
            "runtime_precision_identity_sha256": self.precision,
            "score_complete_file_sha256": pin("fixture-complete-full-score-file")}

    def select(self, *, topics=None, random_receipt=None, scores=None, score_receipt=None, file_pin=None):
        return selector.select_full_and_pilot(self.topics if topics is None else topics,
            self.random_receipt if random_receipt is None else random_receipt,
            self.scores if scores is None else scores, self.score_receipt if score_receipt is None else score_receipt,
            random_receipt_file_sha256=self.random_file_pin if file_pin is None else file_pin)


class FullPoolSelectionTests(unittest.TestCase):
    def test_fixed_native_protocol_counts_namespaces_and_stage_separation_are_explicit(self):
        self.assertEqual(selector.PROTOCOL_SHA256, PROTOCOL)
        self.assertEqual(selector.APPROVAL_SHA256, APPROVAL)
        self.assertEqual(selector.SelectionCounts().value(), {"pool": 83029, "reserved": 67,
            "benchmark": 1, "eligible": 82961, "full": 15000, "pilot": 2048, "fixture_only": False})
        self.assertEqual(selector.SEED, SEED)
        self.assertEqual({"pool": selector.POOL_FILE_SHA256, "exclusions": selector.EXCLUSION_FILE_SHA256,
            "development": selector.DEVELOPMENT_COMPLETE_SHA256, "benchmark": selector.BENCHMARK_AUDIT_SHA256,
            "benchmark_ids": selector.BENCHMARK_IDS_FILE_SHA256, "base": selector.BASE_IDENTITY_SHA256},
            {"pool": "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0",
             "exclusions": "0066e69a996cf963173018d68196d7ff69e658420c40b250941c2fe8bb6da406",
             "development": "354859dd44e2fe3aa7800ec991b60649f00a99ce4768d8075af42c007289987d",
             "benchmark": "2671a58d0abf9f52de5445649aecf9a389363f0fe54d8f05dbf5bca243c6fa6f",
             "benchmark_ids": "5a687438e7b5c31dbc16f2671582b448ff970739219328dbf9c5836474a89984",
             "base": "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2"})
        self.assertEqual(tuple(selector.CELLS), tuple((topic, length) for topic in range(64) for length in range(5)))
        self.assertEqual(set(selector.PURPOSES), set(PURPOSES))
        self.assertEqual(tuple(inspect.signature(selector.draw_random_order).parameters), ("population",))
        self.assertNotIn("scores", inspect.signature(selector.bind_topics).parameters)
        self.assertNotIn("gold", inspect.signature(selector.select_full_and_pilot).parameters)

    def test_small_counts_require_explicit_nonnative_and_plain_integer_conservation(self):
        case = SyntheticCase()
        self.assertTrue(case.population.counts.fixture_only)
        for key in ("pool", "reserved", "benchmark", "eligible", "full", "pilot"):
            for bad in (True, float(getattr(case.counts, key)), -1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    replace(case.counts, **{key: bad})
        for changes in ({"fixture_only": 1}, {"fixture_only": False}, {"full": 29},
                        {"pilot": 0}, {"pilot": 16}, {"eligible": 27}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(case.counts, **changes)

    def test_value_serialization_matches_independent_utf8_json_and_is_not_a_file_hash(self):
        value = {"z": [0, True, None, 0.25], "a": "NFKC question π ≤ 2"}
        self.assertEqual(selector.canonical_value_bytes(value), encoded(value))
        self.assertEqual(selector.value_sha256(value), digest(value))
        self.assertNotEqual(selector.value_sha256(value), hashlib.sha256(encoded(value) + b"\n").hexdigest())
        self.assertEqual(selector.value_sha256({"a": 1, "b": 2}), selector.value_sha256({"b": 2, "a": 1}))
        for bad in (float("nan"), float("inf"), {1: "not-string-key"}, {"tuple": (1, 2)}, b"not-json"):
            with self.subTest(bad_type=type(bad).__name__), self.assertRaises(ValueError):
                selector.canonical_value_bytes(bad)

    def test_exclusions_are_the_exact_disjoint_union_with_no_eligible_refill(self):
        case = SyntheticCase()
        self.assertEqual(case.population.eligible_ids,
            tuple(sorted(set(case.pool) - set(case.reserved) - set(case.benchmark))))
        self.assertFalse(set(case.population.eligible_ids) & set(case.reserved + case.benchmark))
        for changes in ({"benchmark": [case.reserved[0]]},
                        {"reserved": [case.reserved[0], identity(99999)]},
                        {"benchmark": []}, {"reserved": case.reserved + case.benchmark}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                case.make_population(**changes)

    def test_canonical_unique_ids_and_exact_eligible_row_namespace_are_required(self):
        case = SyntheticCase()
        bad_pools = [case.pool[:-1] + [case.pool[0]], case.pool[:-1],
                     case.pool[:-1] + ["A" * 64], case.pool[:-1] + ["0" * 63], set(case.pool)]
        for bad in bad_pools:
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError):
                case.make_population(pool=bad)
        for bad in ({key: value for key, value in case.rows.items() if key != case.eligible[-1]},
                    {**case.rows, case.reserved[0]: copy.deepcopy(case.rows[case.eligible[0]])},
                    {**case.rows, identity(99999): copy.deepcopy(case.rows[case.eligible[0]])}):
            with self.subTest(keys=len(bad)), self.assertRaises(ValueError):
                case.make_population(rows=bad)

    def test_complete_original_cache_rows_enforce_context_reserve_and_active_denominator(self):
        case = SyntheticCase()
        for changes in ({"total_tokens": 32737}, {"total_tokens": 32768}, {"total_tokens": 1},
                        {"total_tokens": 32736.0}, {"total_tokens": True}, {"active_tokens": 0},
                        {"active_tokens": case.rows[case.eligible[0]]["total_tokens"]},
                        {"active_tokens": True}, {"original_pair_sha256": "bad"},
                        {"cache_record_sha256": "F" * 64}, {"output": "must not inspect or replace targets"}):
            rows = copy.deepcopy(case.rows)
            rows[case.eligible[0]].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                case.make_population(rows=rows)
        self.assertIn(32736, [row.total_tokens for row in case.population.rows.values()])

    def test_population_receipts_bind_values_files_runtime_approval_and_strict_counts(self):
        case = SyntheticCase()
        for key in ("pool_sorted_ids_sha256", "reserved_sorted_ids_sha256", "benchmark_sorted_ids_sha256",
                    "excluded_sorted_ids_sha256", "eligible_sorted_ids_sha256", "cache_rows_value_sha256",
                    "protocol_file_sha256", "root_approval_file_sha256"):
            changed = copy.deepcopy(case.inputs)
            changed[key] = pin("wrong:" + key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                case.make_population(receipt=changed)
        for changes in ({"python_version": "3.10.20"}, {"caller_verified_files_and_runtime": False},
                        {"fixture_only": 1}, {"complete": 1}, {"extra_answer_field": "forbidden"},
                        {"cache_complete_file_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                case.make_population(receipt={**case.inputs, **changes})
        for key, bad in (("pool", float(case.counts.pool)), ("benchmark", True), ("fixture_only", 1)):
            changed = copy.deepcopy(case.inputs)
            changed["counts"][key] = bad
            with self.subTest(count=key), self.assertRaises(ValueError):
                case.make_population(receipt=changed)

    def test_validated_population_copies_inputs_and_exposes_only_immutable_bound_values(self):
        case = SyntheticCase()
        original = case.population.identity_receipt
        row = case.population.rows[case.eligible[0]]
        case.rows[case.eligible[0]]["total_tokens"] = 2000
        case.inputs["python_version"] = "mutated-external-dict"
        case.pool.reverse()
        leaked = case.population.identity_receipt
        leaked["counts"]["pool"] = 0
        self.assertEqual(case.population.identity_receipt, original)
        self.assertEqual(row.total_tokens, 1024)
        self.assertEqual(case.population.pool_ids, tuple(sorted(case.pool)))
        with self.assertRaises(TypeError):
            case.population.rows[case.eligible[0]] = row
        with self.assertRaises(FrozenInstanceError):
            row.active_tokens = 0
        with self.assertRaises(FrozenInstanceError):
            case.population.identity_sha256 = pin("changed")

    def test_counterfeit_population_dataclasses_and_reordered_snapshots_fail_revalidation(self):
        case = SyntheticCase()
        rows = dict(case.population.rows)
        rows[case.eligible[0]] = replace(rows[case.eligible[0]], total_tokens=2000)
        forged = [replace(case.population, eligible_ids=tuple(reversed(case.eligible))),
                  replace(case.population, rows=MappingProxyType(rows)),
                  replace(case.population, rows=dict(case.population.rows)),
                  replace(case.population, identity_json=case.population.identity_json + b"\n"),
                  replace(case.population, identity_sha256=pin("other-identity"))]
        for value in forged:
            with self.subTest(field=value.identity_sha256), self.assertRaises(ValueError):
                selector.draw_random_order(value)

    def test_all_length_boundaries_use_complete_total_tokens_and_plain_integer_inputs(self):
        for value, expected in ((1, 0), (2048, 0), (2049, 1), (4096, 1), (4097, 2),
                                (8192, 2), (8193, 3), (16384, 3), (16385, 4), (32736, 4), (32768, 4)):
            with self.subTest(value=value):
                self.assertEqual(selector.length_bin(value), expected)
        for bad in (0, 32769, True, 2048.0, "2048"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selector.length_bin(bad)

    def test_broad_random_matches_one_independent_stdlib_sample_of_all_sorted_eligible_ids(self):
        case = SyntheticCase()
        calls = []

        class CountingRandom:
            def __init__(self, seed):
                self.rng = REAL_RANDOM(seed)
                calls.append({"seed": seed, "samples": []})
                self.call = calls[-1]

            def sample(self, values, count):
                self.call["samples"].append((list(values), count))
                return self.rng.sample(values, count)

        expected = REAL_RANDOM(SEED).sample(sorted(case.eligible), case.counts.full)
        with mock.patch.object(selector.random, "Random", CountingRandom):
            result = selector.draw_random_order(case.population)
        self.assertEqual(result["ordered_ids"], expected)
        self.assertEqual(calls, [{"seed": SEED, "samples": [(list(case.eligible), case.counts.full)]}])
        self.assertEqual(result["rng_sample_calls"], 1)
        self.assertEqual(set(calls[0]["samples"][0][0]), set(case.eligible))
        self.assertIn(32736, [case.population.rows[item].total_tokens for item in calls[0]["samples"][0][0]])

    def test_random_receipt_preserves_order_hashes_original_population_and_nonnative_limits(self):
        case = SyntheticCase()
        result = selector.draw_random_order(case.population)
        self.assertEqual(result["ordered_ids_sha256"], digest(result["ordered_ids"]))
        self.assertEqual(result["sorted_ids_sha256"], digest(sorted(result["ordered_ids"])))
        self.assertEqual(result["eligible_sorted_ids_sha256"], digest(list(case.eligible)))
        self.assertTrue(result["fixture_only"])
        self.assertFalse(result["physical_file_or_one_use_execution_verified_here"])
        self.assertFalse(result["execution_admitted_here"])
        result["ordered_ids"][0] = case.reserved[0]
        self.assertEqual(case.population.eligible_ids, case.eligible)

    def test_random_validation_never_redraws_and_rejects_partial_duplicate_or_stale_seals(self):
        case = SyntheticCase()
        with mock.patch.object(selector.random, "Random", side_effect=AssertionError("validation must not draw")):
            self.assertEqual(selector.validate_random_order(case.population, case.random_receipt), tuple(case.random_ids))
        mutations = []
        for field, value in (("seed", True), ("rng_sample_calls", True), ("fixture_only", 1),
                             ("ordered_ids_sha256", pin("stale")), ("population_identity_sha256", pin("other-pop")),
                             ("python_random_source_sha256", pin("other-random")),
                             ("execution_admitted_here", True)):
            item = copy.deepcopy(case.random_receipt)
            item[field] = value
            mutations.append(item)
        for ids in (case.random_ids[:-1], case.random_ids[:-1] + [case.random_ids[0]],
                    case.random_ids[:-1] + [case.reserved[0]], tuple(case.random_ids)):
            item = copy.deepcopy(case.random_receipt)
            item["ordered_ids"] = ids
            mutations.append(item)
        for key, value in (("pool", float(case.counts.pool)), ("benchmark", True), ("fixture_only", 1)):
            item = copy.deepcopy(case.random_receipt)
            item["counts"][key] = value
            mutations.append(item)
        for item in mutations:
            with self.subTest(keys=list(item)), self.assertRaises(ValueError):
                selector.validate_random_order(case.population, item)

    def test_topics_require_complete_eligible_coverage_plain_labels_and_exact_receipt(self):
        case = SyntheticCase()
        for changed in ({key: value for key, value in case.assignments.items() if key != case.eligible[-1]},
                        {**case.assignments, case.reserved[0]: 0},
                        {**case.assignments, case.eligible[0]: -1},
                        {**case.assignments, case.eligible[0]: 64},
                        {**case.assignments, case.eligible[0]: True},
                        {**case.assignments, case.eligible[0]: 0.0}):
            with self.subTest(count=len(changed)), self.assertRaises(ValueError):
                selector.bind_topics(case.population, changed, case.topic_receipt)
        for field in ("topic_assignments_sha256", "population_identity_sha256", "eligible_sorted_ids_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                selector.bind_topics(case.population, case.assignments,
                    {**case.topic_receipt, field: pin("different:" + field)})

    def test_topic_cells_are_immutable_sorted_320_cell_partition_including_empty_topics(self):
        case = SyntheticCase()
        self.assertEqual(list(case.topics.cells), [(topic, length) for topic in range(64) for length in range(5)])
        self.assertEqual(sorted(item for members in case.topics.cells.values() for item in members), list(case.eligible))
        for cell, members in case.topics.cells.items():
            self.assertEqual(members, tuple(sorted(members)))
            self.assertTrue(all((case.assignments[item], own_bin(case.rows[item]["total_tokens"])) == cell for item in members))
        self.assertEqual(case.topics.cells[(62, 3)], ())
        with self.assertRaises(TypeError):
            case.topics.cells[(62, 3)] = (case.eligible[0],)
        with self.assertRaises(TypeError):
            case.topics.assignments[case.eligible[0]] = 62
        case.assignments[case.eligible[0]] = 62
        leaked = case.topics.identity_receipt
        leaked["complete"] = False
        self.assertEqual(case.topics.assignments[case.eligible[0]], 0)
        self.assertTrue(case.topics.identity_receipt["complete"])

    def test_namespace_matches_independent_exact_json_and_separates_all_five_rng_purposes(self):
        case = SyntheticCase()
        hashes = set()
        for purpose in PURPOSES:
            actual = selector.cell_namespace(case.population, case.topics.assignments_sha256,
                                             case.random_receipt["ordered_ids_sha256"], purpose, (63, 4))
            expected = own_namespace(case, purpose, (63, 4))
            self.assertEqual(actual, expected)
            self.assertEqual(selector.canonical_value_bytes(actual), encoded(expected))
            self.assertEqual(selector.value_sha256(actual), digest(expected))
            self.assertEqual(int.from_bytes(bytes.fromhex(digest(expected)), "big"), int(digest(expected), 16))
            hashes.add(digest(actual))
            self.assertFalse(set(actual) & {"score_sha256", "mean_ce", "job_id", "selected_ids_sha256"})
        self.assertEqual(len(hashes), 5)
        for purpose, cell in (("S15000.low", (0, 0)), (PURPOSES[0], [0, 0]),
                              (PURPOSES[0], (True, 0)), (PURPOSES[0], (64, 0)), (PURPOSES[0], (0, 5))):
            with self.subTest(purpose=purpose, cell=cell), self.assertRaises(ValueError):
                selector.cell_namespace(case.population, case.topics.assignments_sha256,
                    case.random_receipt["ordered_ids_sha256"], purpose, cell)

    def test_full_and_nested_selection_match_independent_stdlib_mathematical_oracle(self):
        case = SyntheticCase()
        expected = independent_selection(case)
        actual = case.select()
        self.assertEqual(actual["selected_ordered_ids"], expected["full"])
        self.assertEqual(actual["selected_pilot_ordered_ids"], expected["pilot"])
        self.assertEqual(actual["cells"], expected["cells"])
        self.assertEqual(actual["selection_tags"], [{"id": item, "tag": expected["tags"][item]} for item in sorted(expected["tags"])])
        self.assertEqual(actual["selected_ordered_ids_sha256"], digest(expected["full"]))
        self.assertEqual(actual["selected_sorted_ids_sha256"], digest(sorted(expected["full"])))
        self.assertEqual(actual["score_binding_value_sha256"], digest(case.score_receipt))

    def test_odd_quotas_floor_low_ce_and_singleton_cells_use_only_seeded_coverage(self):
        for eligible, full, pilot in ((8, 3, 1), (3, 1, 1)):
            case = SyntheticCase(eligible, full, pilot, layout="one", means=[0.25] * eligible)
            result = case.select()
            nonzero = [cell for cell in result["cells"] if cell["full_quota"]]
            self.assertEqual(len(nonzero), 1)
            cell = nonzero[0]
            self.assertEqual((cell["low_ce_count"], cell["coverage_count"]), (full // 2, full - full // 2))
            self.assertEqual((cell["pilot_low_ce_count"], cell["pilot_coverage_count"]), (pilot // 2, pilot - pilot // 2))
            low = [row["id"] for row in result["selection_tags"] if row["tag"] == "low_ce"]
            self.assertEqual(low, list(case.eligible[:full // 2]))
            self.assertEqual(len(result["selected_ordered_ids"]), full)
            self.assertEqual(len(set(result["selected_ordered_ids"])), full)

    def test_low_ce_ranking_is_numeric_binary64_then_canonical_id_not_hex_lexicographic(self):
        means = [1.0, 0.5, 2.0, 0.25, 4.0, 0.125, 0.25, 0.25]
        case = SyntheticCase(8, 4, 2, layout="one", means=means)
        actual = case.select()
        low = [row["id"] for row in actual["selection_tags"] if row["tag"] == "low_ce"]
        self.assertEqual(low, sorted((case.eligible[5], case.eligible[3])))
        self.assertNotEqual(sorted(case.eligible, key=lambda item: case.scores[item]["mean_ce_hex"])[:2],
                            sorted(case.eligible, key=lambda item: (case.scores[item]["mean_ce"], item))[:2])
        self.assertIn(case.eligible[5], low)  # A low-CE member outside the broad synthetic R is eligible.

    def test_all_zero_quota_cells_are_recorded_without_creating_rng_or_sample_calls(self):
        case = SyntheticCase()
        calls = []

        def tracked(seed):
            calls.append(seed)
            return REAL_RANDOM(seed)

        with mock.patch.object(selector.random, "Random", side_effect=tracked):
            result = case.select()
        zero = [cell for cell in result["cells"] if cell["full_quota"] == 0]
        self.assertEqual(len(result["cells"]), 320)
        self.assertGreater(len(zero), 300)
        self.assertTrue(all(item["sample_calls"] == item["count"] == 0 for cell in zero for item in cell["rng"]))
        self.assertEqual(len(calls), sum(item["sample_calls"] for cell in result["cells"] for item in cell["rng"]))
        self.assertTrue(result["zero_quota_cells_recorded"])

    def test_selected_arm_never_redraws_r_and_preserves_all_caller_input_values(self):
        case = SyntheticCase()
        originals = copy.deepcopy((case.rows, case.inputs, case.assignments, case.topic_receipt,
                                    case.random_receipt, case.scores, case.score_receipt))
        seeds = []

        def tracked(seed):
            seeds.append(seed)
            return REAL_RANDOM(seed)

        with mock.patch.object(selector, "draw_random_order", side_effect=AssertionError("No second broad R draw")), \
                mock.patch.object(selector.random, "Random", side_effect=tracked):
            result = case.select()
        self.assertNotIn(SEED, seeds)
        self.assertEqual((case.rows, case.inputs, case.assignments, case.topic_receipt,
                          case.random_receipt, case.scores, case.score_receipt), originals)
        self.assertEqual(result["random_ordered_ids_sha256"], case.random_receipt["ordered_ids_sha256"])
        self.assertEqual(result["random_pilot_ordered_ids"], case.random_ids[:case.counts.pilot])

    def test_iteration_order_of_pool_rows_topics_and_scores_cannot_change_outputs(self):
        case = SyntheticCase()
        expected = case.select()
        population = case.make_population(pool=list(reversed(case.pool)), reserved=list(reversed(case.reserved)),
                                         rows=dict(reversed(list(case.rows.items()))))
        topics = selector.bind_topics(population, dict(reversed(list(case.assignments.items()))), case.topic_receipt)
        actual = selector.select_full_and_pilot(topics, copy.deepcopy(case.random_receipt),
            dict(reversed(list(case.scores.items()))), copy.deepcopy(case.score_receipt),
            random_receipt_file_sha256=case.random_file_pin)
        self.assertEqual(actual, expected)

    def test_score_coverage_is_the_entire_eligible_population_not_only_selected_or_reservoir(self):
        case = SyntheticCase()
        unselected_by_r = next(item for item in case.eligible if item not in case.random_ids)
        for changed in ({item: row for item, row in case.scores.items() if item != unselected_by_r},
                        {**case.scores, case.reserved[0]: copy.deepcopy(case.scores[case.eligible[0]])},
                        {**case.scores, identity(999999): copy.deepcopy(case.scores[case.eligible[0]])}):
            with self.subTest(rows=len(changed)), self.assertRaises(ValueError):
                case.select(scores=changed)
        malformed = copy.deepcopy(case.scores)
        malformed[unselected_by_r]["teacher_answer"] = "not a permitted selector field"
        with self.assertRaises(ValueError):
            case.select(scores=malformed)
        self.assertEqual(len(case.scores), case.counts.eligible)

    def test_scores_require_finite_plain_binary64_canonical_hex_and_exact_one_division(self):
        case = SyntheticCase()
        first = case.eligible[0]
        for field, value in (("mean_ce", 1), ("mean_ce", True), ("mean_ce", float("nan")),
                             ("mean_ce", float("inf")), ("nll_sum", -0.125),
                             ("mean_ce_hex", "0X1.0000000000000P-3"),
                             ("nll_sum_hex", "not-a-hex-float")):
            changed = copy.deepcopy(case.scores)
            changed[first][field] = value
            with self.subTest(field=field, value=str(value)), self.assertRaises(ValueError):
                case.select(scores=changed)
        changed = copy.deepcopy(case.scores)
        mean = math.nextafter(changed[first]["mean_ce"], math.inf)
        changed[first].update(mean_ce=mean, mean_ce_hex=mean.hex())
        with self.assertRaisesRegex(ValueError, "divide"):
            case.select(scores=changed)
        changed = copy.deepcopy(case.scores)
        changed[first]["nll_sum_hex"] = changed[first]["nll_sum_hex"] + "0"
        with self.assertRaises(ValueError):
            case.select(scores=changed)

    def test_complete_score_diagnostics_keep_eos_non_eos_tail_counts_and_original_target_identity(self):
        case = SyntheticCase()
        only_eos = SyntheticCase(4, 2, 1, layout="one", means=[0.0] * 4, active_tokens=1)
        self.assertTrue(only_eos.select()["complete"])
        self.assertTrue(all(row["components"]["non_eos"] == {"count": 0, "nll_sum": 0.0,
            "nll_sum_hex": 0.0.hex()} for row in only_eos.scores.values()))
        first = case.eligible[0]
        for field, value in (("active_tokens", True), ("active_tokens", case.rows[first]["active_tokens"] + 1),
                             ("original_pair_sha256", pin("rewritten-target")),
                             ("cache_record_sha256", pin("truncated-cache")),
                             ("loaded_base_identity_sha256", pin("other-base"))):
            changed = copy.deepcopy(case.scores)
            changed[first][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                case.select(scores=changed)
        for name, field, value in (("eos", "count", 0), ("eos", "count", True),
                                   ("tail", "count", 0), ("non_eos", "count", 0),
                                   ("tail", "nll_sum", float("nan"))):
            changed = copy.deepcopy(case.scores)
            changed[first]["components"][name][field] = value
            with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                case.select(scores=changed)
        changed = copy.deepcopy(case.scores)
        changed[first]["components"].pop("tail")
        with self.assertRaises(ValueError):
            case.select(scores=changed)

    def test_score_receipts_separate_exact_r_file_pin_from_value_hash_and_bind_all_lineage(self):
        case = SyntheticCase()
        self.assertNotEqual(case.random_file_pin, digest(case.random_receipt))
        for field in ("population_identity_sha256", "eligible_sorted_ids_sha256", "cache_complete_file_sha256",
                      "random_order_receipt_file_sha256", "random_ordered_ids_sha256", "score_rows_value_sha256",
                      "score_definition_sha256", "loaded_base_identity_sha256", "runtime_precision_identity_sha256"):
            changed = copy.deepcopy(case.score_receipt)
            changed[field] = digest(case.random_receipt) if field == "random_order_receipt_file_sha256" else pin("wrong:" + field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                case.select(score_receipt=changed)
        for bad in (None, "bad", digest(case.random_receipt)):
            with self.subTest(pin=bad), self.assertRaises(ValueError):
                selector.select_full_and_pilot(case.topics, case.random_receipt, case.scores,
                    case.score_receipt, random_receipt_file_sha256=bad)
        self.assertEqual(case.select()["random_receipt_file_sha256"], case.random_file_pin)

    def test_nested_pilot_samples_within_frozen_full_tags_instead_of_taking_new_lowest_prefix(self):
        case = SyntheticCase(20, 12, 7, layout="one", means=[float(index) for index in range(20)])

        class SuffixFixtureRandom:
            def __init__(self, seed):
                if type(seed) is not int:
                    raise AssertionError("An integer namespace seed is required")

            def sample(self, values, count):
                return list(reversed(values))[:count]

        with mock.patch.object(selector.random, "Random", SuffixFixtureRandom):
            result = case.select()
        full_tags = {item["id"]: item["tag"] for item in result["selection_tags"]}
        pilot = set(result["selected_pilot_ordered_ids"])
        pilot_low = sorted(item for item in pilot if full_tags[item] == "low_ce")
        pilot_coverage = sorted(item for item in pilot if full_tags[item] == "seeded_coverage")
        self.assertEqual(pilot_low, list(case.eligible[3:6]))
        self.assertNotEqual(pilot_low, list(case.eligible[:3]))
        self.assertEqual(pilot_coverage, list(case.eligible[-4:]))
        self.assertTrue(pilot <= set(result["selected_ordered_ids"]))
        self.assertEqual(len(pilot), 7)
        self.assertTrue(result["fixture_only"])

    def test_quota_or_tag_capacity_corruption_fails_without_fallback_or_rng_work(self):
        case = SyntheticCase()
        cells = dict(case.topics.cells)
        cells[(0, 0)] = cells[(0, 0)][1:]
        forged = replace(case.topics, cells=MappingProxyType(cells))
        with mock.patch.object(selector.random, "Random", side_effect=AssertionError("Do not refill malformed inputs")):
            with self.assertRaises(ValueError):
                case.select(topics=forged)
            with self.assertRaises(ValueError):
                selector._sample(case.population, case.topics.assignments_sha256,
                    case.random_receipt["ordered_ids_sha256"], PURPOSES[2], (0, 0), [case.eligible[0]], 2)
        self.assertTrue(case.select()["no_redraw_refill_fallback"])

    def test_full_and_pilot_orders_match_reference_cell_sequences_and_overlap_is_reported(self):
        case = SyntheticCase()
        expected = independent_selection(case)
        result = case.select()
        lookup = expected["lookup"]
        self.assertEqual([lookup[item] for item in result["selected_ordered_ids"]],
                         [lookup[item] for item in case.random_ids])
        self.assertEqual([lookup[item] for item in result["selected_pilot_ordered_ids"]],
                         [lookup[item] for item in case.random_ids[:case.counts.pilot]])
        self.assertEqual(Counter(lookup[item] for item in result["selected_ordered_ids"]),
                         Counter(lookup[item] for item in case.random_ids))
        self.assertEqual(result["full_random_selected_overlap_count"],
                         len(set(case.random_ids) & set(result["selected_ordered_ids"])))
        self.assertEqual(result["pilot_random_selected_overlap_count"],
                         len(set(case.random_ids[:case.counts.pilot]) & set(result["selected_pilot_ordered_ids"])))
        self.assertEqual(len(set(result["selected_ordered_ids"])), case.counts.full)
        self.assertEqual(len(set(result["selected_pilot_ordered_ids"])), case.counts.pilot)

    def test_length_only_probe_uses_independent_fraction_ranks_id_ties_and_first_occurrence_dedup(self):
        for case in (SyntheticCase(), SyntheticCase(2, 2, 1, layout="one"), SyntheticCase(1, 1, 1, layout="one")):
            ordered = sorted(case.eligible, key=lambda item: (case.rows[item]["total_tokens"], item))
            chosen, ranks = [], []
            fractions = (Fraction(0), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2),
                         Fraction(3, 4), Fraction(9, 10), Fraction(19, 20), Fraction(99, 100), Fraction(1))
            for fraction in fractions:
                rank = int(fraction * (len(ordered) - 1))
                if ordered[rank] not in chosen:
                    chosen.append(ordered[rank])
                    ranks.append(rank)
            with mock.patch.object(selector.random, "Random", side_effect=AssertionError("Length-only probe must not draw")):
                result = selector.length_probe_ids(case.population)
            self.assertEqual(result["ordered_ids"], chosen)
            self.assertEqual(result["ranks"], ranks)
            self.assertEqual(result["ordered_ids_sha256"], digest(chosen))
            self.assertTrue(result["length_only"])
            self.assertFalse(result["score_or_topic_inputs_used"])
            self.assertEqual(chosen[0], ordered[0])
            self.assertEqual(chosen[-1], ordered[-1])

    def test_outputs_are_finite_plain_json_metadata_and_do_not_claim_targets_or_native_admission(self):
        case = SyntheticCase()
        result = case.select()
        self.assertEqual(json.loads(encoded(result)), result)
        self.assertTrue(result["fixture_only"])
        self.assertTrue(result["complete"])
        for key in ("native_population_or_method_improvement_established", "execution_admitted_here",
                    "source_targets_modified", "gold_or_development_outcomes_consumed"):
            self.assertIs(result[key], False)
        for key, values in (("selected_ordered_ids_sha256", result["selected_ordered_ids"]),
                            ("selected_pilot_ordered_ids_sha256", result["selected_pilot_ordered_ids"]),
                            ("selected_pilot_sorted_ids_sha256", sorted(result["selected_pilot_ordered_ids"])),
                            ("random_pilot_ordered_ids_sha256", result["random_pilot_ordered_ids"])):
            self.assertEqual(result[key], digest(values))
        self.assertFalse(set(result) & {"input", "output", "system", "question", "gold", "target", "labels", "input_ids"})
        self.assertEqual(result["counts"], case.counts.value())

    def test_stale_cross_stage_topic_random_score_and_population_snapshots_cannot_be_promoted(self):
        case = SyntheticCase()
        changed_topics = dict(case.topics.assignments)
        changed_topics[case.eligible[0]] = 2
        invalid_topics = [replace(case.topics, assignments=MappingProxyType(changed_topics)),
                          replace(case.topics, assignments_sha256=pin("different-topics")),
                          replace(case.topics, receipt_json=encoded({**case.topics.identity_receipt, "complete": False}))]
        for forged in invalid_topics:
            with self.subTest(topic_hash=forged.assignments_sha256), self.assertRaises(ValueError):
                case.select(topics=forged)
        random_changed = copy.deepcopy(case.random_receipt)
        random_changed["ordered_ids"].reverse()
        with self.assertRaises(ValueError):
            case.select(random_receipt=random_changed)
        score_changed = copy.deepcopy(case.scores)
        score_changed[case.eligible[-1]]["mean_ce"] = 99.0
        with self.assertRaises(ValueError):
            case.select(scores=score_changed)
        forged_pop = replace(case.population, identity_sha256=pin("stale-population"))
        with self.assertRaises(ValueError):
            case.select(topics=replace(case.topics, population=forged_pop))


if __name__ == "__main__":
    unittest.main()
