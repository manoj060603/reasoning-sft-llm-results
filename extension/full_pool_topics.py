"""Unadmitted S2 question-only topic fit; no corpus/artifact I/O or admission.

Importing this module uses only stdlib. A future separately admitted caller
owns original-question extraction, source/exclusion/cache/plan/current-job
proofs, fresh-child lifecycle and exclusive private artifact persistence.
The only file reads here hash eight prospectively pinned installed runtime
SOURCE files, bounded to2MiB each. No model, tokenizer, answer, system text,
loss, benchmark outcome, CLI, file writer, thread setter or fallback exists.

fit_topics accepts one attempt per fresh process, with numerical libraries
unimported and the four thread variables already1. It performs one complete
TF-IDF fit_transform, one fixed-formula SVD fit_transform, the prescribed L2
normalization and one MiniBatchKMeans fit_predict. n_init=3 belongs to that
single fixed estimator fit. Caller-side durable journals must additionally
prevent new-process reexecution; a module latch cannot establish that proof.

All returned vocabulary/arrays/estimators/IDs are PRIVATE in-memory artifacts.
Only bounded aggregate/hash metadata is suitable for an outer public receipt.
Array hashes identify canonical in-memory C-order bytes, never unsaved FILEs.
Synthetic fixtures require explicit fixture_only=True and at least64 rows;
they cannot establish original82961-question/native-compute provenance.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import unicodedata


SCHEMA = "extension-full-pool-topic-fit-v1"
PROTOCOL_SHA256 = "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29"
APPROVAL_SHA256 = "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6"
RUNTIME_RECEIPT_SHA256 = "5788c73cff4f26607c9af604a11634334cdec15a0b7b687c48968fac06b73436"
NATIVE_ROWS, TOPICS, SEED = 82961, 64, 20261021
MAX_SOURCE_BYTES, MAX_IO_BYTES, MAX_METADATA_BYTES = 2 << 20, 1 << 20, 64 << 10
THREAD_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
VERSIONS = {"numpy": "1.26.4", "scipy": "1.15.3", "scikit-learn": "1.7.2", "threadpoolctl": "3.6.0", "joblib": "1.5.2"}
SCIENTIFIC_MODULES = ("numpy", "scipy", "sklearn", "threadpoolctl", "joblib")
RUNTIME_SOURCES = {
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/random.py": {"bytes": 33221, "sha256": "ce80a2471965e64ae93caea14490f16850432e65dce87f15edfe25b6f562f8d1"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/numpy/version.py": {"bytes": 216, "sha256": "3932e74a1d0d19f5b22fc56b9c88f435db7f29939397567e903f427fe8d13266"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/scipy/version.py": {"bytes": 318, "sha256": "a271fdebc4f15ae566cd3c8ca9686bcf52e123dd5d2254b1618acbd9d4292ebc"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/base.py": {"bytes": 47777, "sha256": "a2ad39c5ff6491d04c35738e942860def0e80d6800aeb1690cfd68bf46be8d91"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py": {"bytes": 81743, "sha256": "2e0da803f41cc870df45c59a24cd2da93c46d064785fef3e8d5c2e4912d9c803"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/decomposition/_truncated_svd.py": {"bytes": 11708, "sha256": "9fe23f1ebc98fc0bf27050d95adf30296119f2719433c05962e6011ed6faaa3d"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/feature_extraction/text.py": {"bytes": 77416, "sha256": "876f0d334735f77174822fabff287fab5602ccff8973a5430ebc70548916da62"},
    "/home/manoj663/.conda/envs/reasoning/lib/python3.10/site-packages/sklearn/preprocessing/_data.py": {"bytes": 127903, "sha256": "1255040a0eb9c99f58775d55cf34f675cf61539f7c0a0dd0d6a36cb1f0da1e15"}}
SOURCE_FIXED = {"protocol_file_sha256": PROTOCOL_SHA256, "root_approval_file_sha256": APPROVAL_SHA256,
    "pool_file_sha256": "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0",
    "raw_source_file_sha256": "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8",
    "exclusion_receipt_file_sha256": "0066e69a996cf963173018d68196d7ff69e658420c40b250941c2fe8bb6da406",
    "runtime_receipt_file_sha256": RUNTIME_RECEIPT_SHA256}
SOURCE_KEYS = frozenset(SOURCE_FIXED) | {"preparation_plan_file_sha256", "source_identity_sha256",
    "exclusion_receipt_value_sha256", "eligible_sorted_ids_value_sha256", "question_rows_value_sha256",
    "normalized_questions_value_sha256", "cache_manifest_file_sha256"}
TFIDF_PARAMETERS = {"input": "content", "encoding": "utf-8", "decode_error": "strict", "strip_accents": None,
    "lowercase": True, "preprocessor": None, "tokenizer": None, "analyzer": "char", "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [3, 5], "max_df": 1.0, "min_df": 1,
    "max_features": 50000, "vocabulary": None, "binary": False, "dtype": "numpy.float32", "norm": "l2",
    "use_idf": True, "smooth_idf": True, "sublinear_tf": True}
SVD_PARAMETERS = {"algorithm": "randomized", "n_iter": 5, "n_oversamples": 10,
    "power_iteration_normalizer": "auto", "random_state": SEED, "tol": 0.0}
NORMALIZE_PARAMETERS = {"norm": "l2", "axis": 1, "copy": True}
KMEANS_PARAMETERS = {"n_clusters": TOPICS, "init": "k-means++", "max_iter": 100, "batch_size": 1024,
    "verbose": 0, "compute_labels": True, "random_state": SEED, "tol": 0.0, "max_no_improvement": 10,
    "init_size": None, "n_init": 3, "reassignment_ratio": 0.01}
_FIT_ATTEMPTED = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _plain(value):
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(item) for item in value)
    return type(value) is dict and all(type(key) is str and _plain(item) for key, item in value.items())


def canonical_value_bytes(value):
    _require(_plain(value), "Only finite plain JSON values may be bound")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_sha256(value):
    return hashlib.sha256(canonical_value_bytes(value)).hexdigest()


def _sequence_sha256(values):
    digest = hashlib.sha256(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(canonical_value_bytes(value))
    digest.update(b"]")
    return digest.hexdigest()


def normalize_question(instruction):
    """Topic-only full-text normalization; the source/cache text is not altered."""
    _require(type(instruction) is str, "Question-only instruction must be a plain string")
    text = " ".join(unicodedata.normalize("NFKC", instruction).split())
    _require(bool(text), "Empty normalized question is not a topic input")
    text.encode("utf-8", errors="strict")
    return text


def validate_question_rows(rows, *, expected_ids, source_identity, fixture_only=False):
    """Freeze exact supplied question-only order; no unseen source-byte proof."""
    _require(type(fixture_only) is bool and type(expected_ids) in (list, tuple), "Explicit expected IDs/fixture policy required")
    ids = tuple(expected_ids)
    _require(TOPICS <= len(ids) <= NATIVE_ROWS and (fixture_only or len(ids) == NATIVE_ROWS) and
        all(_sha(identifier) for identifier in ids) and all(a < b for a, b in zip(ids, ids[1:])),
        "Exact ascending native82961 IDs, or explicit NONNATIVE fixtures with at least64 IDs, required")
    _require(type(source_identity) is dict and set(source_identity) == SOURCE_KEYS and
        all(_sha(value) for value in source_identity.values()) and
        all(source_identity[key] == value for key, value in SOURCE_FIXED.items()), "Exact supplied source/cache/exclusion/runtime identity required")
    iterator, texts, raw_digests = iter(rows), [], hashlib.sha256(b"[")
    for index, identifier in enumerate(ids):
        try:
            row = next(iterator)
        except StopIteration as error:
            raise ValueError("Missing topic question; no subset fit/drop/refill") from error
        _require(type(row) is dict and set(row) == {"id", "instruction"} and _sha(row["id"]) and row["id"] == identifier,
            "Only ordered id/instruction fields may enter the topic fit; no answer/system/gold/loss fields")
        instruction = row["instruction"]
        text = normalize_question(instruction)
        _require(hashlib.sha256(text.casefold().encode("utf-8")).hexdigest() == identifier,
            "Canonical question identity differs from supplied full original instruction")
        if index:
            raw_digests.update(b",")
        raw_digests.update(canonical_value_bytes({"id": identifier, "instruction": instruction}))
        _require(row == {"id": identifier, "instruction": instruction}, "Supplied question changed during its snapshot")
        texts.append(text)
    sentinel = object()
    _require(next(iterator, sentinel) is sentinel, "Extra/duplicate topic question; no population substitution")
    raw_digests.update(b"]")
    _require(source_identity["eligible_sorted_ids_value_sha256"] == _sequence_sha256(ids) and
        source_identity["question_rows_value_sha256"] == raw_digests.hexdigest() and
        source_identity["normalized_questions_value_sha256"] == _sequence_sha256(texts),
        "Supplied exact question population/normalization VALUE seals differ")
    return ids, tuple(texts), json.loads(canonical_value_bytes(source_identity))


def validate_runtime_receipt(payload):
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_SOURCE_BYTES and
        hashlib.sha256(payload).hexdigest() == RUNTIME_RECEIPT_SHA256, "Exact root-sealed runtime metadata FILE bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate runtime JSON key")
            result[key] = value
        return result
    value = json.loads(payload, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite runtime JSON")))
    _require(type(value) is dict and _plain(value) and value.get("schema") == "extension-source-only-selection-runtime-metadata-v1" and
        value.get("actual_small_source_file_inventory") == RUNTIME_SOURCES and value.get("versions") == VERSIONS and
        value.get("native_runtime_qa_established") is False and type(value.get("python_version")) is str and
        value["python_version"].startswith("3.10.19 "), "Root runtime metadata is not the exact source-only observation")
    return value


def _fingerprint(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _source_snapshot():
    result = {}
    for name, expected in RUNTIME_SOURCES.items():
        path = Path(name)
        _require(path.is_absolute() and not path.is_symlink() and path.resolve() == path, "Installed runtime source path alias refused")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size == expected["bytes"] <= MAX_SOURCE_BYTES,
                "Installed runtime source size/type differs before reading")
            digest, remaining = hashlib.sha256(), before.st_size
            while remaining:
                block = os.read(fd, min(remaining, MAX_IO_BYTES))
                _require(bool(block), "Truncated installed source file")
                digest.update(block)
                remaining -= len(block)
            _require(not os.read(fd, 1) and not path.is_symlink() and path.resolve() == path and
                _fingerprint(before) == _fingerprint(os.fstat(fd)) == _fingerprint(path.stat()) and
                digest.hexdigest() == expected["sha256"], "Installed source bytes/origin changed")
            result[name] = {**expected, "stat": _fingerprint(before)}
        finally:
            os.close(fd)
    return result


def _thread_environment():
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    _require(observed == THREAD_ENVIRONMENT, "Set the root-sealed thread1 environment BEFORE this fresh process imports numerical libraries")
    return observed


def _bytecode_policy():
    _require(sys.dont_write_bytecode is True and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "Start the fresh child with bytecode writes disabled; do not mutate flags after imports")
    return {"dont_write_bytecode": True, "PYTHONDONTWRITEBYTECODE": "1"}


def validate_threadpools(values):
    _require(type(values) is list and 0 < len(values) <= 32 and _plain(values), "Actual bounded numerical thread pools required")
    result, identities = [], set()
    for pool in values:
        _require(type(pool) is dict and pool.get("user_api") in {"blas", "openmp"} and
            type(pool.get("num_threads")) is int and pool["num_threads"] == 1 and
            all(type(pool.get(key)) is str and pool[key] for key in ("internal_api", "prefix", "filepath")) and
            Path(pool["filepath"]).is_absolute() and (pool.get("version") is None or type(pool["version"]) is str),
            "Every actual BLAS/OpenMP pool must report exactly one effective thread and a real vendor/source path")
        identity = (pool["user_api"], pool["internal_api"], pool["filepath"])
        _require(identity not in identities, "Duplicate numerical thread-pool identity")
        identities.add(identity)
        result.append(json.loads(canonical_value_bytes(pool)))
    _require(any(pool["user_api"] == "blas" for pool in result), "Missing actual BLAS pool is not thread-policy proof")
    result.sort(key=lambda pool: (pool["user_api"], pool["internal_api"], pool["filepath"]))
    _require(len(canonical_value_bytes(result)) <= MAX_METADATA_BYTES // 2, "Thread-pool metadata exceeds its bound")
    return result


def _load_runtime(before_sources):
    _require(sys.implementation.name == "cpython" and tuple(sys.version_info[:3]) == (3, 10, 19), "Exact CPython3.10.19 required")
    installed = {name: importlib.metadata.version(name) for name in VERSIONS}
    _require(installed == VERSIONS, "Root-observed five installed package versions differ")
    np, scipy, sklearn = (importlib.import_module(name) for name in ("numpy", "scipy", "sklearn"))
    threadpoolctl, joblib = (importlib.import_module(name) for name in ("threadpoolctl", "joblib"))
    random = importlib.import_module("random")
    base = importlib.import_module("sklearn.base")
    vectorizer = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer
    svd = importlib.import_module("sklearn.decomposition").TruncatedSVD
    cluster = importlib.import_module("sklearn.cluster").MiniBatchKMeans
    normalize = importlib.import_module("sklearn.preprocessing").normalize
    normalize_signature = inspect.signature(normalize)
    _require(normalize_signature.parameters["return_norm"].default is False, "Pinned normalize return contract changed")
    runtime = {"np": np, "scipy": scipy, "sklearn": sklearn, "threadpoolctl": threadpoolctl, "joblib": joblib,
        "random": random, "TfidfVectorizer": vectorizer, "TruncatedSVD": svd,
        "MiniBatchKMeans": cluster, "normalize": normalize, "threadpool_info": threadpoolctl.threadpool_info,
        "BaseEstimator": base.BaseEstimator}
    observed = _runtime_observation(runtime)
    _require(observed["sources"] == before_sources, "Installed sources changed during numerical imports")
    return runtime, observed


def _runtime_observation(runtime):
    _require(sys.implementation.name == "cpython" and tuple(sys.version_info[:3]) == (3, 10, 19), "Python runtime changed")
    actual_versions = {"numpy": runtime["np"].__version__, "scipy": runtime["scipy"].__version__,
        "scikit-learn": runtime["sklearn"].__version__, "threadpoolctl": runtime["threadpoolctl"].__version__,
        "joblib": runtime["joblib"].__version__}
    _require(actual_versions == VERSIONS and {name: importlib.metadata.version(name) for name in VERSIONS} == VERSIONS,
        "Actual imported/installed package versions differ")
    origins = {"random": inspect.getsourcefile(runtime["random"].Random),
        "numpy.version": inspect.getsourcefile(importlib.import_module("numpy.version")),
        "scipy.version": inspect.getsourcefile(importlib.import_module("scipy.version")),
        "sklearn.base": inspect.getsourcefile(runtime["BaseEstimator"]),
        "TfidfVectorizer": inspect.getsourcefile(runtime["TfidfVectorizer"]),
        "TruncatedSVD": inspect.getsourcefile(runtime["TruncatedSVD"]),
        "MiniBatchKMeans": inspect.getsourcefile(runtime["MiniBatchKMeans"]),
        "normalize": inspect.getsourcefile(inspect.unwrap(runtime["normalize"]))}
    expected_origins = {"random": "random.py", "numpy.version": "site-packages/numpy/version.py",
        "scipy.version": "site-packages/scipy/version.py", "sklearn.base": "site-packages/sklearn/base.py",
        "TfidfVectorizer": "site-packages/sklearn/feature_extraction/text.py",
        "TruncatedSVD": "site-packages/sklearn/decomposition/_truncated_svd.py",
        "MiniBatchKMeans": "site-packages/sklearn/cluster/_kmeans.py", "normalize": "site-packages/sklearn/preprocessing/_data.py"}
    _require(origins == {key: "/home/manoj663/.conda/envs/reasoning/lib/python3.10/" + name for key, name in expected_origins.items()},
        "Original loaded source origins differ from root's exact eight-file scope")
    return {"python": sys.version, "implementation": sys.implementation.name,
        "versions": actual_versions, "source_origins": origins, "sources": _source_snapshot(),
        "normalize_decorator_source_origin_unpinned": inspect.getsourcefile(runtime["normalize"]),
        "thread_environment": _thread_environment(), "bytecode_policy": _bytecode_policy(),
        "threadpools": validate_threadpools(runtime["threadpool_info"]()),
        "source_identity_scope": "Eight installed source files only; not all transitive packages, compiled binaries or bitwise cross-platform determinism"}


def canonical_parameters(values, np):
    if values is np.float32:
        return "numpy.float32"
    if type(values) in (tuple, list):
        return [canonical_parameters(value, np) for value in values]
    if type(values) is dict:
        _require(all(type(key) is str for key in values), "Estimator parameter names must be plain strings")
        return {key: canonical_parameters(value, np) for key, value in values.items()}
    _require(values is None or type(values) in (str, bool, int, float), "Unsupported effective estimator parameter type")
    _require(type(values) is not float or math.isfinite(values), "Nonfinite effective estimator parameter")
    return values


def _parameters(estimator, expected, runtime):
    np = runtime["np"]
    _require(type(estimator).get_params is runtime["BaseEstimator"].get_params, "Original installed get_params implementation required")
    value = canonical_parameters(estimator.get_params(deep=False), np)
    _require(canonical_value_bytes(value) == canonical_value_bytes(expected), "Complete installed get_params differs from frozen S2 constructor policy")
    return value


def _array_identity(value, np):
    _require(type(value) is np.ndarray and value.dtype.kind in "fiu" and bool(np.isfinite(value).all()),
        "Finite ordinary numeric array artifact required")
    contiguous = np.ascontiguousarray(value)
    view = memoryview(contiguous).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), MAX_IO_BYTES):
        digest.update(view[start:start + MAX_IO_BYTES])
    return {"shape": [int(size) for size in value.shape], "dtype": value.dtype.str,
        "bytes": len(view), "c_order_content_sha256": digest.hexdigest(), "physical_file_hash": False}


def _norm_summary(value, np):
    _require(type(value) is np.ndarray and value.ndim == 1 and len(value) > 0 and
        bool(np.isfinite(value).all()) and bool((value > 0).all()), "Every complete question needs a positive finite representation norm")
    return {"rows": len(value), "minimum": float(value.min()), "maximum": float(value.max())}


class TopicFitError(ValueError):
    """Outer admitted caller may persist bounded metadata privately, then stop."""
    def __init__(self, stage, metadata):
        super().__init__("Question-only topic fit failed without retry: " + stage)
        self.stage = stage
        self.metadata = json.loads(canonical_value_bytes(metadata))


def fit_topics(rows, *, expected_ids, source_identity, runtime_receipt_json_bytes, fixture_only=False):
    """One attempt in a fresh admitted child; return PRIVATE in-memory artifacts.

    The caller must independently verify original source/cache/exclusion bytes
    and compute/current-job identity. Supplied hashes are bindings, not a claim
    that this helper opened those files. Native/synthetic both use the exact
    same sealed runtime and constructor route; fixtures never substitute pins.
    """
    global _FIT_ATTEMPTED
    _require(not _FIT_ATTEMPTED, "No process reuse/refit/retry; start only the separately admitted fresh child")
    _FIT_ATTEMPTED = True
    stage, started = "input_guard", time.monotonic()
    metadata = {"schema": SCHEMA, "fixture_only": fixture_only if type(fixture_only) is bool else None, "complete": False,
        "fits_started": {"tfidf": 0, "svd": 0, "kmeans": 0}, "actual_source_or_compute_admission_claim": False}
    try:
        _require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in SCIENTIFIC_MODULES),
            "Numerical packages were already imported; pre-import thread policy is unproven")
        _thread_environment()
        _bytecode_policy()
        validate_runtime_receipt(runtime_receipt_json_bytes)
        ids, texts, source = validate_question_rows(rows, expected_ids=expected_ids, source_identity=source_identity, fixture_only=fixture_only)
        metadata.update(rows=len(ids), source_identity=source, source_identity_value_sha256=value_sha256(source),
            runtime_receipt_file_sha256=RUNTIME_RECEIPT_SHA256, protocol_file_sha256=PROTOCOL_SHA256,
            root_approval_file_sha256=APPROVAL_SHA256, original_question_order_value_sha256=_sequence_sha256(ids))
        stage = "runtime_guard"
        before_sources = _source_snapshot()
        runtime, runtime_before = _load_runtime(before_sources)
        np, timings = runtime["np"], {}
        metadata["runtime_before"] = runtime_before
        stage, begin = "tfidf", time.monotonic()
        tfidf_kwargs = {**TFIDF_PARAMETERS, "dtype": np.float32, "ngram_range": (3, 5)}
        vectorizer = runtime["TfidfVectorizer"](**tfidf_kwargs)
        tfidf_parameters = _parameters(vectorizer, TFIDF_PARAMETERS, runtime)
        _require(_runtime_observation(runtime) == runtime_before, "Runtime changed before TF-IDF fit")
        metadata["fits_started"]["tfidf"] = 1
        matrix = vectorizer.fit_transform(texts)
        _require(runtime["scipy"].sparse.isspmatrix_csr(matrix) and matrix.shape[0] == len(ids) and
            2 <= matrix.shape[1] <= 50000 and matrix.dtype == np.dtype("float32") and matrix.nnz > 0 and
            bool(np.isfinite(matrix.data).all()) and bool((matrix.data >= 0).all()), "Complete float32 TF-IDF matrix is invalid")
        tfidf_norms = _norm_summary(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1), np)
        timings[stage] = time.monotonic() - begin
        stage, begin = "svd", time.monotonic()
        k = min(128, matrix.shape[0] - 1, matrix.shape[1] - 1)
        _require(type(k) is int and k >= 1, "Frozen SVD dimension formula is infeasible; no bypass or alternate rank")
        svd = runtime["TruncatedSVD"](n_components=k, **SVD_PARAMETERS)
        svd_parameters = _parameters(svd, {"n_components": k, **SVD_PARAMETERS}, runtime)
        _require(_runtime_observation(runtime) == runtime_before, "Runtime changed before SVD fit")
        metadata["fits_started"]["svd"] = 1
        representation = svd.fit_transform(matrix)
        _require(type(representation) is np.ndarray and representation.shape == (len(ids), k) and
            representation.dtype == np.dtype("float32"), "SVD representation shape/dtype differs")
        raw_norms = _norm_summary(np.linalg.norm(representation, axis=1), np)
        normalized = runtime["normalize"](representation, **NORMALIZE_PARAMETERS)
        _require(type(normalized) is np.ndarray and normalized.shape == (len(ids), k) and
            normalized.dtype == np.dtype("float32"), "Normalized SVD shape/dtype differs")
        normalized_norms = _norm_summary(np.linalg.norm(normalized, axis=1), np)
        timings[stage] = time.monotonic() - begin
        stage, begin = "kmeans", time.monotonic()
        kmeans = runtime["MiniBatchKMeans"](**KMEANS_PARAMETERS)
        kmeans_parameters = _parameters(kmeans, KMEANS_PARAMETERS, runtime)
        _require(_runtime_observation(runtime) == runtime_before, "Runtime changed before clustering fit")
        metadata["fits_started"]["kmeans"] = 1
        labels = kmeans.fit_predict(normalized)
        _require(type(labels) is np.ndarray and labels.shape == (len(ids),) and labels.dtype.kind in "iu" and
            bool(((labels >= 0) & (labels < TOPICS)).all()) and np.array_equal(labels, kmeans.labels_),
            "Every question requires its actual unrenamed cluster label0..63; no sentinel/invalid-row fallback")
        effective_init_size = getattr(kmeans, "_init_size", None)
        _require(type(effective_init_size) is int and effective_init_size == min(len(ids), 3 * min(1024, len(ids))),
            "Pinned actual sklearn1.7.2 effective init_size differs")
        timings[stage] = time.monotonic() - begin
        stage = "artifact_and_closeout_guard"
        _require(type(vectorizer.vocabulary_) is dict and all(type(key) is str and
            type(value) in (int, np.int32, np.int64) for key, value in vectorizer.vocabulary_.items()),
            "Fitted vocabulary needs plain string features and actual integer indices")
        vocabulary = {key: int(value) for key, value in vectorizer.vocabulary_.items()}
        _require(len(vocabulary) == matrix.shape[1] and
            sorted(vocabulary.values()) == list(range(matrix.shape[1])), "Fitted vocabulary does not cover every actual feature exactly once")
        _require(vectorizer.idf_.shape == (matrix.shape[1],) and svd.components_.shape == (k, matrix.shape[1]) and
            svd.singular_values_.shape == (k,) and kmeans.cluster_centers_.shape == (TOPICS, k), "Fitted artifact dimensions differ")
        arrays = {"idf": vectorizer.idf_, "svd_components": svd.components_, "svd_singular_values": svd.singular_values_,
            "svd_explained_variance": svd.explained_variance_, "svd_explained_variance_ratio": svd.explained_variance_ratio_,
            "unnormalized_representation": representation, "normalized_representation": normalized,
            "centers": kmeans.cluster_centers_, "labels": labels}
        array_identities = {name: _array_identity(value, np) for name, value in arrays.items()}
        sparse_identity = {"format": "csr", "shape": [int(size) for size in matrix.shape], "nnz": int(matrix.nnz), "dtype": matrix.dtype.str,
            "data": _array_identity(matrix.data, np), "indices": _array_identity(matrix.indices, np),
            "indptr": _array_identity(matrix.indptr, np), "index_order": "Actual estimator output; no sorting or row replacement"}
        assignments = tuple({"id": identifier, "topic": int(label)} for identifier, label in zip(ids, labels))
        counts = [int(count) for count in np.bincount(labels, minlength=TOPICS)]
        _require(len(counts) == TOPICS and sum(counts) == len(ids), "All64 cluster counts, including empty clusters, required")
        runtime_after = _runtime_observation(runtime)
        _require(runtime_after == runtime_before and
            _parameters(vectorizer, TFIDF_PARAMETERS, runtime) == tfidf_parameters and
            _parameters(svd, {"n_components": k, **SVD_PARAMETERS}, runtime) == svd_parameters and
            _parameters(kmeans, KMEANS_PARAMETERS, runtime) == kmeans_parameters, "Runtime/thread/vendor/source/constructor policy changed during fit")
        _require(type(kmeans.n_iter_) in (int, np.int32, np.int64) and 1 <= int(kmeans.n_iter_) <= KMEANS_PARAMETERS["max_iter"] and
            type(kmeans.n_steps_) in (int, np.int32, np.int64) and int(kmeans.n_steps_) >= 1 and
            math.isfinite(float(kmeans.inertia_)) and float(kmeans.inertia_) >= 0,
            "Finite valid fitted minibatch iterations/steps/inertia required")
        metadata.update(complete=True, complete_supplied_question_fit=True, constructors={"tfidf": tfidf_parameters,
            "svd": svd_parameters, "normalize": NORMALIZE_PARAMETERS, "kmeans": kmeans_parameters},
            runtime_after=runtime_after, runtime_unchanged=True, topic_runtime_identity_sha256=value_sha256(runtime_before),
            topic_assignments_sha256=_sequence_sha256(assignments), cluster_counts=counts, empty_clusters=[i for i, count in enumerate(counts) if count == 0],
            vocabulary_value_sha256=value_sha256(vocabulary), sparse_matrix=sparse_identity, arrays=array_identities,
            representation_norms={"tfidf_squared_l2": tfidf_norms, "svd_before_l2": raw_norms, "svd_after_l2": normalized_norms},
            effective_init_size=effective_init_size, effective_init_size_scope="Pinned sklearn1.7.2 _init_size observed, not an extra fit choice",
            fitted_iterations=int(kmeans.n_iter_), fitted_minibatch_steps=int(kmeans.n_steps_), inertia=float(kmeans.inertia_),
            stage_elapsed_seconds=timings, elapsed_seconds=time.monotonic() - started,
            persistence_or_file_hash_claim=False, bitwise_cross_platform_determinism_claim=False,
            source_scope="Supplied full questions only; caller verifies actual source/cache/exclusion bytes and native process admission",
            semantic_topic_or_decontamination_claim=False, model_or_accuracy_input_used=False, automatic_retry=False, automatic_downstream=False)
        _require(_plain(metadata) and len(canonical_value_bytes(metadata)) <= MAX_METADATA_BYTES, "Bounded finite aggregate/hash topic metadata required")
        return {"artifacts": {"ordered_ids": ids, "assignments": assignments, "vocabulary": vocabulary, "tfidf_matrix": matrix,
            **arrays, "vectorizer": vectorizer, "svd": svd, "kmeans": kmeans}, "metadata": metadata,
            "metadata_value_sha256": value_sha256(metadata)}
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        metadata.update(complete=False, failed_stage=stage, error_type=type(error).__name__, elapsed_seconds=time.monotonic() - started,
            automatic_retry=False, automatic_downstream=False)
        raise TopicFitError(stage, metadata) from error
