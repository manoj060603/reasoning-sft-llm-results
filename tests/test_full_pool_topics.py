"""UNEXECUTED prospective32 source-only topic fixtures, never a native alias.

The parent imports only stdlib and the stdlib-only topic helper. One fresh
child fits64 synthetic question-only rows with the actual pinned libraries;
its closed, bounded aggregate receipt is shared by several independent
assertions. Four other fresh children exercise pre-fit failures. There is no
corpus/tokenizer/model input, latch reset, thread setter or package fallback.

A future admitted CPU2/8GiB/30min wrapper must bind the exact runtime receipt
and sources before discovery. Children are sequential, at most five, each
limited to60s. These fixtures do not establish native82961-row fitting,
memory/RSS, durable no-retry journals or original source/cache provenance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unicodedata
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extension import full_pool_topics as topics


RECEIPT_RELATIVE = "extension/operations/full-pool-runtime-source-metadata-20261021.json"
RECEIPT_SHA256 = "5788c73cff4f26607c9af604a11634334cdec15a0b7b687c48968fac06b73436"
MAX_CHILDREN, CHILD_TIMEOUT = 5, 60
_CHILD_COUNT = 0
_POSITIVE = None
_POSITIVE_ATTEMPTED = False


def independent_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def independent_sha(value):
    return hashlib.sha256(independent_bytes(value)).hexdigest()


def synthetic_rows(count=64):
    rows = []
    for index in range(count):
        instruction = ("Question %03d: A route contains %d red stones and %d blue stones. "
            "Determine the sum and explain the arithmetic operation." % (index, index + 11, index * 3 + 7))
        normalized = " ".join(unicodedata.normalize("NFKC", instruction).split())
        identifier = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
        rows.append({"id": identifier, "instruction": instruction})
    return sorted(rows, key=lambda row: row["id"])


def identity_for(rows):
    normalized = [" ".join(unicodedata.normalize("NFKC", row["instruction"]).split()) for row in rows]
    identity = {name: "a" * 64 for name in topics.SOURCE_KEYS}
    identity.update(topics.SOURCE_FIXED)
    identity.update(eligible_sorted_ids_value_sha256=independent_sha([row["id"] for row in rows]),
        question_rows_value_sha256=independent_sha(rows), normalized_questions_value_sha256=independent_sha(normalized))
    return identity


def required_receipt():
    expected = ROOT / RECEIPT_RELATIVE
    configured = os.environ.get("TOPIC_QA_RUNTIME_RECEIPT")
    if configured != str(expected) or not expected.is_absolute() or expected.is_symlink() or expected.resolve() != expected:
        raise ValueError("Exact admitted public runtime receipt path required; no fallback")
    fd = os.open(expected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 << 20:
            raise ValueError("Bounded public runtime metadata required")
        blocks, remaining = [], before.st_size
        while remaining:
            block = os.read(fd, min(remaining, 1 << 20))
            if not block:
                raise ValueError("Truncated runtime metadata")
            blocks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
        fingerprint = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        payload = b"".join(blocks)
        if os.read(fd, 1) or fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(expected.stat()) or \
                hashlib.sha256(payload).hexdigest() != RECEIPT_SHA256:
            raise ValueError("Runtime metadata changed or differs from its fixed FILE pin")
        return payload
    finally:
        os.close(fd)


def child_result(scenario):
    global _CHILD_COUNT
    if scenario not in {"positive", "preloaded", "bad_thread", "bad_bytecode", "bad_receipt"} or _CHILD_COUNT >= MAX_CHILDREN:
        raise ValueError("Finite declared child scenarios only")
    required_receipt()
    environment = dict(os.environ)
    if scenario == "bad_thread":
        environment["OMP_NUM_THREADS"] = "2"
    elif scenario == "bad_bytecode":
        environment["PYTHONDONTWRITEBYTECODE"] = "0"
    _CHILD_COUNT += 1
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--topic-fixture-worker", scenario],
        cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=CHILD_TIMEOUT, check=False)
    if len(result.stdout) > 128 << 10 or len(result.stderr) > 128 << 10 or result.returncode != 0:
        raise AssertionError("Declared tiny topic child failed; preserve admitted private QA logs")
    value = json.loads(result.stdout)
    if type(value) is not dict or len(independent_bytes(value)) > 128 << 10 or value.get("fixture_only") is not True or \
            value.get("native_proof") is not False or value.get("scenario") != scenario:
        raise AssertionError("Malformed/nonfixture child result")
    return value


def positive_result():
    global _POSITIVE, _POSITIVE_ATTEMPTED
    if not _POSITIVE_ATTEMPTED:
        _POSITIVE_ATTEMPTED = True
        _POSITIVE = child_result("positive")
    if _POSITIVE is None:
        raise AssertionError("The single positive fixture attempt failed; no child refit or retry is allowed")
    return _POSITIVE


def _worker(scenario):
    """Actual-library tiny fixture entry; no execution during authoring."""
    receipt = required_receipt()
    rows = synthetic_rows()
    ids = [row["id"] for row in rows]
    source = identity_for(rows)
    result = {"scenario": scenario, "fixture_only": True, "native_proof": False,
        "runtime_receipt_file_sha256": hashlib.sha256(receipt).hexdigest()}
    if scenario == "preloaded":
        __import__("numpy")  # Deliberate negative fixture; no fit may follow.
    supplied = receipt + b" " if scenario == "bad_receipt" else receipt
    if scenario != "positive":
        try:
            topics.fit_topics(rows, expected_ids=ids, source_identity=source,
                runtime_receipt_json_bytes=supplied, fixture_only=True)
        except topics.TopicFitError as error:
            result.update(error_stage=error.stage, failure_metadata=error.metadata,
                scientific_packages_loaded=[prefix for prefix in topics.SCIENTIFIC_MODULES if prefix in sys.modules])
        else:
            raise AssertionError("Negative child unexpectedly fitted")
        try:
            topics.fit_topics(rows, expected_ids=ids, source_identity=source,
                runtime_receipt_json_bytes=receipt, fixture_only=True)
        except ValueError as error:
            result["reuse_refused"] = type(error) is ValueError and "reuse" in str(error)
        else:
            raise AssertionError("Failed attempt was silently retried")
        return result
    fitted = topics.fit_topics(rows, expected_ids=ids, source_identity=source,
        runtime_receipt_json_bytes=receipt, fixture_only=True)
    # Numerical imports occur only after the real helper has completed guards.
    np = sys.modules["numpy"]
    artifacts, metadata = fitted["artifacts"], fitted["metadata"]
    arrays = {}
    for name in metadata["arrays"]:
        array = artifacts[name]
        arrays[name] = {"shape": list(array.shape), "dtype": array.dtype.str,
            "bytes": int(array.nbytes), "c_order_content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "physical_file_hash": False}
    matrix = artifacts["tfidf_matrix"]
    sparse_arrays = {name: {"shape": list(getattr(matrix, name).shape), "dtype": getattr(matrix, name).dtype.str,
        "bytes": int(getattr(matrix, name).nbytes),
        "c_order_content_sha256": hashlib.sha256(getattr(matrix, name).tobytes(order="C")).hexdigest(),
        "physical_file_hash": False} for name in ("data", "indices", "indptr")}
    independent_assignments = [{"id": identifier, "topic": int(label)} for identifier, label in zip(ids, artifacts["labels"])]
    independent_counts = [sum(int(value) == label for value in artifacts["labels"]) for label in range(64)]
    normalized = artifacts["normalized_representation"]
    singular_count = min(128, 63, matrix.shape[1] - 1)
    result.update(metadata=metadata, metadata_sha256=fitted["metadata_value_sha256"],
        independent_metadata_sha256=independent_sha(metadata), independent_arrays=arrays,
        independent_sparse_arrays=sparse_arrays, independent_assignments_sha256=independent_sha(independent_assignments),
        independent_vocabulary_sha256=independent_sha(artifacts["vocabulary"]), independent_counts=independent_counts,
        exact_order=tuple(ids) == artifacts["ordered_ids"], actual_labels_unrenamed=np.array_equal(artifacts["labels"], artifacts["kmeans"].labels_),
        representation_shape=[len(ids), singular_count], csr_float32=matrix.format == "csr" and matrix.dtype == np.dtype("float32"),
        finite_positive_representation=bool(np.isfinite(normalized).all()) and bool((np.linalg.norm(normalized, axis=1) > 0).all()),
        unit_l2_within_1e6=bool(np.allclose(np.linalg.norm(normalized, axis=1), np.ones(64), rtol=0, atol=1e-6)),
        estimator_effective_init_size=int(artifacts["kmeans"]._init_size), fitted_once=metadata["fits_started"] == {"tfidf": 1, "svd": 1, "kmeans": 1},
        constructor_identity_unchanged={"tfidf": topics.canonical_parameters(artifacts["vectorizer"].get_params(deep=False), np),
            "svd": topics.canonical_parameters(artifacts["svd"].get_params(deep=False), np),
            "kmeans": topics.canonical_parameters(artifacts["kmeans"].get_params(deep=False), np)},
        original_questions_unchanged=rows == synthetic_rows(), artifacts_private_not_returned=True)
    try:
        topics.fit_topics(rows, expected_ids=ids, source_identity=source,
            runtime_receipt_json_bytes=receipt, fixture_only=True)
    except ValueError as error:
        result["reuse_refused"] = type(error) is ValueError and "reuse" in str(error)
    else:
        raise AssertionError("Actual successful fit was retried")
    return result


class FullPoolTopicsTests(unittest.TestCase):
    def rows_and_identity(self, count=64):
        rows = synthetic_rows(count)
        return rows, [row["id"] for row in rows], identity_for(rows)

    def validate(self, rows, ids, identity, **kwargs):
        return topics.validate_question_rows(rows, expected_ids=ids, source_identity=identity, fixture_only=kwargs.pop("fixture_only", True), **kwargs)

    def test_01_exact_fixed_native_policy_and_complete_constructors(self):
        self.assertEqual((topics.NATIVE_ROWS, topics.TOPICS, topics.SEED), (82961, 64, 20261021))
        self.assertEqual(topics.PROTOCOL_SHA256, "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29")
        self.assertEqual(topics.APPROVAL_SHA256, "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6")
        self.assertEqual(topics.TFIDF_PARAMETERS, {"input": "content", "encoding": "utf-8", "decode_error": "strict",
            "strip_accents": None, "lowercase": True, "preprocessor": None, "tokenizer": None, "analyzer": "char",
            "stop_words": None, "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [3, 5], "max_df": 1.0, "min_df": 1,
            "max_features": 50000, "vocabulary": None, "binary": False, "dtype": "numpy.float32", "norm": "l2",
            "use_idf": True, "smooth_idf": True, "sublinear_tf": True})
        self.assertEqual(topics.SVD_PARAMETERS, {"algorithm": "randomized", "n_iter": 5, "n_oversamples": 10,
            "power_iteration_normalizer": "auto", "random_state": 20261021, "tol": 0.0})
        self.assertEqual(topics.NORMALIZE_PARAMETERS, {"norm": "l2", "axis": 1, "copy": True})
        self.assertEqual(topics.KMEANS_PARAMETERS, {"n_clusters": 64, "init": "k-means++", "max_iter": 100,
            "batch_size": 1024, "verbose": 0, "compute_labels": True, "random_state": 20261021, "tol": 0.0,
            "max_no_improvement": 10, "init_size": None, "n_init": 3, "reassignment_ratio": .01})

    def test_02_unicode_normalization_retains_full_mathematical_content(self):
        self.assertEqual(topics.normalize_question("  Ａ\u2003+\n２ = 3\t % "), "A + 2 = 3 %")
        self.assertNotEqual(topics.normalize_question("Compute 10 + 2"), topics.normalize_question("Compute 1 + 2"))
        for value in ("", " \n\t", None, 3, "\ud800"):
            with self.subTest(value=repr(value)), self.assertRaises((ValueError, UnicodeError)):
                topics.normalize_question(value)

    def test_03_complete_sorted_population_and_independent_value_seals(self):
        rows, ids, source = self.rows_and_identity()
        got_ids, texts, got_source = self.validate(iter(rows), ids, source)
        self.assertEqual(got_ids, tuple(ids))
        self.assertEqual(texts, tuple(row["instruction"] for row in rows))
        self.assertEqual(source["eligible_sorted_ids_value_sha256"], independent_sha(ids))
        self.assertEqual(source["question_rows_value_sha256"], independent_sha(rows))
        self.assertEqual(got_source, source)

    def test_04_input_snapshot_and_returned_identity_are_not_mutated(self):
        rows, ids, source = self.rows_and_identity()
        before = copy.deepcopy((rows, ids, source))
        _, _, returned = self.validate(rows, ids, source)
        returned["source_identity_sha256"] = "f" * 64
        self.assertEqual((rows, ids, source), before)

    def test_05_gold_answers_system_losses_and_extra_fields_never_enter(self):
        rows, ids, source = self.rows_and_identity()
        for name in ("output", "answer", "gold", "system", "loss", "topic", "input", "labels"):
            changed = copy.deepcopy(rows)
            changed[0][name] = "secret fixture answer"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Only ordered"):
                self.validate(changed, ids, source)

    def test_06_canonical_full_question_id_and_plain_row_guards(self):
        rows, ids, source = self.rows_and_identity()
        changed = copy.deepcopy(rows)
        changed[0]["instruction"] += " changed numerical question"
        with self.assertRaisesRegex(ValueError, "Canonical question"):
            self.validate(changed, ids, source)
        for bad in (rows[0]["id"].upper(), "g" * 64, "0" * 63, True):
            changed = copy.deepcopy(rows)
            changed[0]["id"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.validate(changed, ids, source)

    def test_07_duplicate_descending_and_noncanonical_expected_ids_fail(self):
        rows, ids, source = self.rows_and_identity()
        for changed in (list(reversed(ids)), [ids[0]] + ids[:-1], ["A" * 64] + ids[1:], tuple(ids[:-1]) + (False,)):
            with self.subTest(ids=repr(changed[:2])), self.assertRaises(ValueError):
                self.validate(rows, changed, source)

    def test_08_explicit_nonnative_counts_and_plain_fixture_flag(self):
        rows, ids, source = self.rows_and_identity()
        with self.assertRaisesRegex(ValueError, "native82961"):
            self.validate(rows, ids, source, fixture_only=False)
        for flag in (1, None, "true"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.validate(rows, ids, source, fixture_only=flag)
        too_short = synthetic_rows(63)
        with self.assertRaises(ValueError):
            self.validate(too_short, [row["id"] for row in too_short], identity_for(too_short))

    def test_09_missing_extra_and_reordered_questions_do_not_drop_or_refill(self):
        rows, ids, source = self.rows_and_identity()
        for changed in (rows[:-1], rows + [rows[-1]], [rows[1], rows[0], *rows[2:]]):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.validate(iter(changed), ids, source)

    def test_10_fixed_source_file_pins_and_exact_identity_keyset(self):
        rows, ids, source = self.rows_and_identity()
        for key in topics.SOURCE_FIXED:
            changed = dict(source, **{key: "0" * 64})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Exact supplied"):
                self.validate(rows, ids, changed)
        for changed in ({**source, "extra": "a" * 64}, {key: value for key, value in source.items() if key != "cache_manifest_file_sha256"},
                {**source, "source_identity_sha256": "not-a-sha"}):
            with self.assertRaises(ValueError):
                self.validate(rows, ids, changed)

    def test_11_each_population_raw_and_normalized_seal_is_required(self):
        rows, ids, source = self.rows_and_identity()
        for key in ("eligible_sorted_ids_value_sha256", "question_rows_value_sha256", "normalized_questions_value_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "VALUE seals"):
                self.validate(rows, ids, {**source, key: "0" * 64})
        changed = copy.deepcopy(rows)
        changed[0]["instruction"] = "  " + changed[0]["instruction"] + "\t"
        with self.assertRaisesRegex(ValueError, "VALUE seals"):
            self.validate(changed, ids, source)
        self.validate(changed, ids, identity_for(changed))

    def test_12_canonical_json_and_streamed_sequence_hash_have_independent_oracle(self):
        values = [{"β": 2, "a": [True, None]}, "x", 0]
        self.assertEqual(topics.canonical_value_bytes(values), independent_bytes(values))
        self.assertEqual(topics._sequence_sha256(iter(values)), independent_sha(values))
        for bad in (float("nan"), float("inf"), {1: "x"}, (1, 2), object()):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                topics.canonical_value_bytes(bad)

    def test_13_effective_parameters_canonicalize_dtype_and_tuples_exactly(self):
        fake_np = types.SimpleNamespace(float32=object())
        raw = {"dtype": fake_np.float32, "ngram_range": (3, 5), "plain": [None, 1., False]}
        self.assertEqual(topics.canonical_parameters(raw, fake_np),
            {"dtype": "numpy.float32", "ngram_range": [3, 5], "plain": [None, 1., False]})

    def test_14_nonfinite_and_unsupported_effective_parameters_fail(self):
        fake_np = types.SimpleNamespace(float32=object())
        for bad in (float("nan"), float("inf"), object(), {1: "x"}, {"x": {1, 2}}):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                topics.canonical_parameters(bad, fake_np)

    def test_15_original_get_params_and_complete_constructor_map_are_required(self):
        class OriginalBase:
            def get_params(self, deep=True):
                self.deep = deep
                return {"fixed": 3}
        class Child(OriginalBase):
            pass
        class Rebound(OriginalBase):
            def get_params(self, deep=True):
                return {"fixed": 3}
        runtime = {"np": types.SimpleNamespace(float32=object()), "BaseEstimator": OriginalBase}
        instance = Child()
        self.assertEqual(topics._parameters(instance, {"fixed": 3}, runtime), {"fixed": 3})
        self.assertIs(instance.deep, False)
        with self.assertRaisesRegex(ValueError, "Original installed"):
            topics._parameters(Rebound(), {"fixed": 3}, runtime)
        for expected in ({"fixed": 4}, {}, {"fixed": 3, "extra": None}):
            with self.assertRaisesRegex(ValueError, "Complete installed"):
                topics._parameters(instance, expected, runtime)

    def test_16_thread_environment_is_observed_and_never_set(self):
        with mock.patch.dict(os.environ, topics.THREAD_ENVIRONMENT, clear=True):
            before = dict(os.environ)
            self.assertEqual(topics._thread_environment(), topics.THREAD_ENVIRONMENT)
            self.assertEqual(dict(os.environ), before)
            for name in topics.THREAD_ENVIRONMENT:
                with mock.patch.dict(os.environ, {name: "2"}):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        topics._thread_environment()

    def test_17_bad_bytecode_child_fails_without_scientific_import_or_refit(self):
        result = child_result("bad_bytecode")
        self.assertEqual(result["error_stage"], "input_guard")
        self.assertEqual(result["failure_metadata"]["fits_started"], {"tfidf": 0, "svd": 0, "kmeans": 0})
        self.assertEqual(result["scientific_packages_loaded"], [])
        self.assertTrue(result["reuse_refused"])

    def test_18_actual_runtime_receipt_is_exact_source_metadata_without_native_alias(self):
        payload = required_receipt()
        value = topics.validate_runtime_receipt(payload)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), RECEIPT_SHA256)
        self.assertEqual(value["actual_small_source_file_inventory"], topics.RUNTIME_SOURCES)
        self.assertEqual(value["versions"], {"numpy": "1.26.4", "scipy": "1.15.3", "scikit-learn": "1.7.2",
            "threadpoolctl": "3.6.0", "joblib": "1.5.2"})
        self.assertIs(value["native_runtime_qa_established"], False)
        self.assertEqual(len(topics.RUNTIME_SOURCES), 8)

    def test_19_runtime_receipt_mutation_type_and_size_have_no_fallback(self):
        payload = required_receipt()
        for changed in (payload + b" ", payload[:-1], b"{}", bytearray(payload), b"x" * ((2 << 20) + 1)):
            with self.subTest(kind=type(changed).__name__, length=len(changed)), self.assertRaises(ValueError):
                topics.validate_runtime_receipt(changed)

    def test_20_installed_source_reader_verifies_exact_bounded_ordinary_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "source.py"
            payload = b"# synthetic installed-source reader fixture\n"
            path.write_bytes(payload)
            record = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            with mock.patch.object(topics, "RUNTIME_SOURCES", {str(path): record}):
                result = topics._source_snapshot()
            self.assertEqual(result[str(path)]["sha256"], record["sha256"])
            self.assertEqual(result[str(path)]["bytes"], len(payload))
            self.assertEqual(len(result[str(path)]["stat"]), 5)

    def test_21_size_mismatch_and_oversize_fail_before_source_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "source.py"
            path.write_bytes(b"abc")
            for expected in (4, 0, (2 << 20) + 1):
                with self.subTest(expected=expected), mock.patch.object(topics, "RUNTIME_SOURCES", {str(path): {"bytes": expected, "sha256": "0" * 64}}), \
                        mock.patch.object(topics.os, "read", wraps=os.read) as read:
                    with self.assertRaisesRegex(ValueError, "size/type"):
                        topics._source_snapshot()
                    read.assert_not_called()

    def test_22_installed_source_symlink_and_wrong_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "source.py"
            path.write_bytes(b"abc")
            alias = path.with_name("alias.py")
            alias.symlink_to(path)
            for candidate, sha in ((alias, hashlib.sha256(b"abc").hexdigest()), (path, "0" * 64)):
                with self.subTest(path=candidate.name), mock.patch.object(topics, "RUNTIME_SOURCES", {str(candidate): {"bytes": 3, "sha256": sha}}):
                    with self.assertRaises(ValueError):
                        topics._source_snapshot()

    def test_23_threadpool_metadata_is_sorted_complete_and_detached(self):
        values = [{"user_api": "openmp", "internal_api": "openmp", "num_threads": 1, "prefix": "libomp", "filepath": "/fixture/libomp.so", "version": None},
            {"user_api": "blas", "internal_api": "openblas", "num_threads": 1, "prefix": "libblas", "filepath": "/fixture/libblas.so", "version": "fixture"}]
        before = copy.deepcopy(values)
        result = topics.validate_threadpools(values)
        self.assertEqual(result, list(reversed(values)))
        result[0]["num_threads"] = 99
        self.assertEqual(values, before)

    def test_24_threadpool_false_zero_missing_blas_duplicate_and_unbounded_fail(self):
        good = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1, "prefix": "blas", "filepath": "/fixture/blas.so", "version": None}
        bad_values = [[], [good, good], [{**good, "num_threads": True}], [{**good, "num_threads": 0}],
            [{**good, "num_threads": 2}], [{**good, "filepath": "relative.so"}], [{**good, "version": float("nan")}],
            [{**good, "user_api": "openmp"}], [{**good, "extra": "x" * (64 << 10)}]]
        for value in bad_values:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                topics.validate_threadpools(value)

    def test_25_real_tiny_fit_uses_once_only_complete_frozen_pipeline(self):
        result = positive_result()
        metadata = result["metadata"]
        self.assertTrue(metadata["complete"] and metadata["complete_supplied_question_fit"] and result["fitted_once"])
        self.assertEqual(metadata["rows"], 64)
        self.assertEqual(metadata["fits_started"], {"tfidf": 1, "svd": 1, "kmeans": 1})
        self.assertEqual(metadata["constructors"], {"tfidf": topics.TFIDF_PARAMETERS,
            "svd": {"n_components": 63, **topics.SVD_PARAMETERS}, "normalize": topics.NORMALIZE_PARAMETERS, "kmeans": topics.KMEANS_PARAMETERS})
        self.assertEqual(result["constructor_identity_unchanged"], {name: metadata["constructors"][name] for name in ("tfidf", "svd", "kmeans")})
        self.assertTrue(result["csr_float32"] and result["exact_order"] and result["original_questions_unchanged"])

    def test_26_real_artifact_array_sparse_vocabulary_and_assignment_hash_oracles(self):
        result = positive_result()
        metadata = result["metadata"]
        self.assertEqual(metadata["arrays"], result["independent_arrays"])
        self.assertEqual({name: metadata["sparse_matrix"][name] for name in ("data", "indices", "indptr")}, result["independent_sparse_arrays"])
        self.assertEqual(metadata["topic_assignments_sha256"], result["independent_assignments_sha256"])
        self.assertEqual(metadata["vocabulary_value_sha256"], result["independent_vocabulary_sha256"])
        self.assertEqual(result["metadata_sha256"], result["independent_metadata_sha256"])
        self.assertTrue(all(value["physical_file_hash"] is False for value in metadata["arrays"].values()))

    def test_27_real_normalization_labels_counts_and_effective_init_are_observed(self):
        result = positive_result()
        metadata = result["metadata"]
        self.assertEqual(result["representation_shape"], [64, 63])
        self.assertEqual(metadata["arrays"]["normalized_representation"]["shape"], [64, 63])
        self.assertTrue(result["finite_positive_representation"] and result["unit_l2_within_1e6"] and result["actual_labels_unrenamed"])
        self.assertEqual(metadata["effective_init_size"], 64)
        self.assertEqual(result["estimator_effective_init_size"], 64)
        self.assertEqual(metadata["cluster_counts"], result["independent_counts"])
        self.assertEqual(len(metadata["cluster_counts"]), 64)
        self.assertEqual(sum(metadata["cluster_counts"]), 64)
        self.assertEqual(metadata["empty_clusters"], [i for i, count in enumerate(metadata["cluster_counts"]) if count == 0])
        self.assertEqual(metadata["representation_norms"]["svd_after_l2"]["rows"], 64)

    def test_28_real_runtime_source_thread_bytecode_and_proof_limits(self):
        result = positive_result()
        metadata = result["metadata"]
        before, after = metadata["runtime_before"], metadata["runtime_after"]
        self.assertEqual(before, after)
        self.assertTrue(metadata["runtime_unchanged"])
        self.assertEqual(before["versions"], topics.VERSIONS)
        self.assertEqual(before["thread_environment"], topics.THREAD_ENVIRONMENT)
        self.assertEqual(before["bytecode_policy"], {"dont_write_bytecode": True, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(set(before["sources"]), set(topics.RUNTIME_SOURCES))
        for path, expected in topics.RUNTIME_SOURCES.items():
            self.assertEqual({name: before["sources"][path][name] for name in ("sha256", "bytes")}, expected)
        self.assertTrue(before["threadpools"] and all(pool["num_threads"] == 1 for pool in before["threadpools"]))
        for name in ("actual_source_or_compute_admission_claim", "persistence_or_file_hash_claim", "bitwise_cross_platform_determinism_claim",
                "semantic_topic_or_decontamination_claim", "model_or_accuracy_input_used", "automatic_retry", "automatic_downstream"):
            self.assertIs(metadata[name], False)
        self.assertTrue(result["fixture_only"] and result["artifacts_private_not_returned"])
        self.assertIs(result["native_proof"], False)
        self.assertFalse(any(prefix in sys.modules for prefix in topics.SCIENTIFIC_MODULES))

    def test_29_real_success_latch_blocks_reuse_without_second_fit(self):
        result = positive_result()
        self.assertTrue(result["reuse_refused"])
        self.assertEqual(result["metadata"]["fits_started"], {"tfidf": 1, "svd": 1, "kmeans": 1})
        self.assertLessEqual(len(independent_bytes(result)), 128 << 10)
        encoded = independent_bytes(result)
        for row in synthetic_rows():
            self.assertNotIn(row["id"].encode(), encoded)
            self.assertNotIn(row["instruction"].encode(), encoded)
        self.assertNotIn("artifacts", result)

    def test_30_preimported_real_numpy_child_never_fits_or_retries(self):
        result = child_result("preloaded")
        self.assertEqual(result["error_stage"], "input_guard")
        self.assertEqual(result["failure_metadata"]["fits_started"], {"tfidf": 0, "svd": 0, "kmeans": 0})
        self.assertIn("numpy", result["scientific_packages_loaded"])
        self.assertTrue(result["reuse_refused"])

    def test_31_bad_thread_child_stops_before_numerical_imports_and_refit(self):
        result = child_result("bad_thread")
        self.assertEqual(result["error_stage"], "input_guard")
        self.assertEqual(result["failure_metadata"]["fits_started"], {"tfidf": 0, "svd": 0, "kmeans": 0})
        self.assertEqual(result["scientific_packages_loaded"], [])
        self.assertTrue(result["reuse_refused"])

    def test_32_changed_receipt_child_fails_before_import_and_poisoned_attempt_is_final(self):
        result = child_result("bad_receipt")
        self.assertEqual(result["error_stage"], "input_guard")
        self.assertEqual(result["failure_metadata"]["fits_started"], {"tfidf": 0, "svd": 0, "kmeans": 0})
        self.assertEqual(result["scientific_packages_loaded"], [])
        self.assertTrue(result["reuse_refused"])
        self.assertIs(result["failure_metadata"]["automatic_retry"], False)
        self.assertIs(result["failure_metadata"]["automatic_downstream"], False)
        self.assertLessEqual(_CHILD_COUNT, MAX_CHILDREN)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--topic-fixture-worker":
        value = _worker(sys.argv[2])
        payload = independent_bytes(value)
        if len(payload) > 128 << 10:
            raise RuntimeError("Bounded fixture metadata exceeded declared size")
        sys.stdout.buffer.write(payload + b"\n")
    else:
        unittest.main()
