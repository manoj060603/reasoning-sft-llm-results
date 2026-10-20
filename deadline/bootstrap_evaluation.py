"""Prepare immutable benchmark evidence and admit the original-model evaluation.

Run in a CPU compute allocation. This never reads training preparation, model
scores, or source-selection outcomes. ensure_benchmarks(artifacts_root) is also
used by the training finalizer: a shared lock plus byte verification prevents
either producer from rewriting files while the original evaluation uses them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from deadline.audit_results import EXPECTED, OFFICIAL
from reasoning.benchmark import record_ids
from reasoning.data import digest, prompt_id, question, read_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIME25_REVISION = "563bb8404243c5f09de6ec262f2db674fe5bce9b"
SCHEMA = "deadline-benchmark-readiness-v1"
READINESS_NAME = "benchmark_readiness.json"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _write_new_json(path, value):
    """Publish a complete new file without replacing a concurrent existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_benchmarks(benchmark_root, original_root):
    benchmark_root, original_root = Path(benchmark_root), Path(original_root)
    originals = {path.parent.name: path for path in original_root.glob("*/test.jsonl")}
    _require(set(originals) == set(OFFICIAL), "exactly the original eight benchmark files are required")
    published = {path.parent.name: path for path in benchmark_root.glob("*/test.jsonl")}
    _require(set(published) == set(EXPECTED), "exactly the complete nine benchmark files are required")
    hashes, counts, ids, original_hashes = {}, {}, {}, {}
    for name, count in EXPECTED.items():
        path = published[name]
        rows = read_rows(path)
        _require(len(rows) == count, f"{name}: unexpected frozen benchmark record count")
        _require(all("answer" in row and row["answer"] is not None for row in rows),
                 f"{name}: missing reference answer")
        # Stable record identities retain Kaoyan's four repeated-question pairs.
        benchmark_ids = record_ids(rows)
        _require(len(set(benchmark_ids)) == count, f"{name}: duplicate record identities")
        hashes[name], counts[name], ids[name] = digest(path), count, benchmark_ids
        if name in originals:
            original_hashes[name] = digest(originals[name])
            _require(hashes[name] == original_hashes[name], f"{name}: supplied benchmark bytes differ")
    supplement_path = benchmark_root / "supplement.json"
    supplement = json.loads(supplement_path.read_text())
    _require(supplement.get("repo") == "math-ai/aime25" and
             supplement.get("revision") == AIME25_REVISION,
             "AIME25 supplement must identify the exact pinned public source")
    _require(supplement.get("problems") == 30 and
             supplement.get("canonical_sha256") == hashes["aime25"],
             "AIME25 supplement count or canonical hash differs")
    _require(supplement.get("supplied_benchmark_hashes") == original_hashes,
             "AIME25 supplement has different original-eight source hashes")
    _require(isinstance(supplement.get("source_sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", supplement["source_sha256"]),
             "AIME25 supplement must identify the downloaded source bytes")
    aime25 = read_rows(published["aime25"])
    _require(len({prompt_id(question(row)) for row in aime25}) == 30,
             "AIME25 must contain 30 distinct normalized questions")
    _require(all(str(row["answer"]).isdigit() and 0 <= int(row["answer"]) <= 999 for row in aime25),
             "AIME25 references must be integers from 0 through 999")
    return {"benchmark_hashes": hashes, "benchmark_counts": counts,
            "problem_ids": ids, "supplied_benchmark_hashes": original_hashes,
            "aime25_revision": AIME25_REVISION,
            "aime25_source_sha256": supplement["source_sha256"],
            "supplement_sha256": digest(supplement_path)}


def ensure_benchmarks(artifacts_root, *, base_benchmarks=None):
    """Return verified readiness under artifacts_root, preparing at most once.

    Existing complete files may be adopted without rewriting. Existing partial
    files, inconsistent receipts, or changed hashes fail closed. Preparation is
    staged beside the destination and published only after all nine files pass.
    """
    artifacts_root = Path(artifacts_root).resolve()
    original_root = (PROJECT_ROOT / "eval/data" if base_benchmarks is None
                     else Path(base_benchmarks).resolve())
    artifacts_root.mkdir(parents=True, exist_ok=True)
    destination = artifacts_root / "benchmarks"
    readiness_path = artifacts_root / READINESS_NAME
    with (artifacts_root / "benchmark_readiness.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if readiness_path.exists():
            receipt = json.loads(readiness_path.read_text())
            _require(receipt.get("schema") == SCHEMA and receipt.get("complete") is True,
                     "invalid benchmark readiness receipt")
            binding = {key: value for key, value in receipt.items() if key != "binding_sha256"}
            _require(receipt.get("binding_sha256") == _content_hash(binding),
                     "benchmark readiness receipt checksum differs")
            _require(receipt.get("benchmark_root") == str(destination) and
                     receipt.get("original_root") == str(original_root.resolve()),
                     "benchmark readiness paths differ")
            evidence = _audit_benchmarks(destination, original_root)
            _require(receipt.get("evidence") == evidence, "frozen benchmark readiness bytes changed")
            return receipt

        if destination.exists():
            # Do not invoke the old downloader on a published directory: its
            # copyfile/replace operations would rewrite even identical data.
            evidence = _audit_benchmarks(destination, original_root)
        else:
            staging = Path(tempfile.mkdtemp(prefix=".benchmarks-preparing-", dir=artifacts_root))
            subprocess.run([sys.executable, "-m", "reasoning.download_aime25",
                            "--revision", AIME25_REVISION,
                            "--base-benchmarks", str(original_root), "--output", str(staging)],
                           cwd=PROJECT_ROOT, check=True)
            evidence = _audit_benchmarks(staging, original_root)
            # Claim an absent destination atomically. Replacing our own empty
            # reservation publishes the entire verified directory in one rename.
            destination.mkdir(exist_ok=False)
            try:
                staging.rename(destination)
            except BaseException:
                destination.rmdir()
                raise
        receipt = {"schema": SCHEMA, "complete": True,
                   "benchmark_root": str(destination), "original_root": str(original_root.resolve()),
                   "evidence": evidence, "producer_sha256": digest(Path(__file__)),
                   "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                   "historical_aime25_equivalence_verified": False}
        receipt["binding_sha256"] = _content_hash(receipt)
        _write_new_json(readiness_path, receipt)
        return receipt


def run_native_preflight():
    """Run all current deadline CPU tests and reject skipped native checks."""
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"),
                                               pattern="test_deadline_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun < 113:
        raise RuntimeError("native deadline test discovery found fewer than the required 113 checks")
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("native deadline CPU checks must pass without dependency skips before evaluation")
    from reasoning.grader_preflight import check_grader
    grader = check_grader()
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "grader": grader}


def bootstrap(artifacts_root=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("evaluation bootstrap must run in a CPU compute allocation")
    artifacts_root = (PROJECT_ROOT / "deadline/artifacts" if artifacts_root is None
                      else Path(artifacts_root).resolve())
    os.chdir(PROJECT_ROOT)
    checks = run_native_preflight()
    readiness = ensure_benchmarks(artifacts_root)
    preflight = {"schema": "deadline-original-evaluation-bootstrap-v1", "complete": True,
                 "slurm_job_id": os.environ["SLURM_JOB_ID"], "native_preflight": checks,
                 "benchmark_readiness_binding_sha256": readiness["binding_sha256"],
                 "benchmark_readiness_sha256": digest(artifacts_root / READINESS_NAME),
                 "producer_sha256": digest(Path(__file__))}
    preflight_path = artifacts_root / f"evaluation_bootstrap_preflight-{os.environ['SLURM_JOB_ID']}.json"
    if preflight_path.exists():
        _require(json.loads(preflight_path.read_text()) == preflight,
                 "existing evaluation bootstrap preflight differs")
    else:
        _write_new_json(preflight_path, preflight)
    from deadline.pipeline import ROOT, submit
    identifier = submit("original_evaluation", ROOT / "gpu_eval.slurm",
                        arguments=["original"], gpu_hours=6)
    return {"job_id": identifier, "native_preflight": checks,
            "benchmark_readiness_binding_sha256": readiness["binding_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.artifacts_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
