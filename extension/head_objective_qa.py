"""Unadmitted CPU-only synthetic head-objective fixture stage.

Imports above the guarded run are standard-library only. This controller never
opens a corpus, private cache, model checkpoint, tokenizer or benchmark result.
The exact 36 tests exercise synthetic FP32 mathematics, not Qwen/CUDA memory or
BF16/model integration. A separate metadata bootstrap must admit this role.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import unittest


ROLE = "head_objective_qa_20261021_v1"
NAME = "sft-extension-head-objective-qa"
PLAN = "extension/head_objective_qa_plan.json"
SCRIPT = "extension/head_objective_qa.slurm"
BOOTSTRAP = "extension/operations/submit-head-objective-qa-20261021.py"
SCHEMA = "extension-head-objective-native-qa-v1"
LEDGER_SCHEMA = "reasoning-sft-extension-job-ledger-v1"
PATTERN, EXPECTED_TESTS = "test_checkpointed_head_objectives.py", 36
WALL_SECONDS, RESERVE_SECONDS = 1200, 172800
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
HELPER = "extension/checkpointed_head_objectives.py"
TEST = "tests/" + PATTERN
PINS = {HELPER: "a0c02ac3730f616d7ccd0fcb2b768af60387cea15e9cf82c1050b9deb60d7a34",
        TEST: "c442f7d873c31778676f6ade1b322cdd239a19ec4d6c4086b3dbaac846c3a3c0"}
SOURCES = frozenset({HELPER, TEST, PLAN, SCRIPT, "extension/head_objective_qa.py"})
VERSIONS = {"torch": "2.9.0", "transformers": "4.57.3", "accelerate": "1.12.0", "safetensors": "0.7.0"}
CLAIMS = {"synthetic_cpu_fixture_scope_only": True, "older_tests_replayed": False,
    "actual_qwen_model_loaded": False, "private_cache_or_source_data_opened": False,
    "tokenizer_loaded": False, "gpu_or_bf16_cuda_admission_established": False,
    "long_context_memory_admission_established": False, "pilot_or_training_admitted": False,
    "automatic_retry": False, "automatic_downstream": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def inside(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Expected an explicit project-relative metadata path")
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Metadata path escapes the project")
    return path


def record(root, name):
    path = inside(root, name)
    before = path.stat()
    require(path.is_file() and 0 < before.st_size <= 2 << 20, "Expected bounded source/receipt metadata")
    payload = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Input changed while reading")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def read_json(root, name, expected=None):
    before = record(root, name)
    require(expected is None or before["sha256"] == expected, "Pinned metadata SHA differs")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON metadata key")
            result[key] = value
        return result
    value = json.loads(inside(root, name).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON metadata")))
    require(record(root, name) == before, "Metadata changed during parsing")
    return value


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_inventory(root):
    result = {name: record(root, name) for name in sorted(SOURCES)}
    require(all(result[name]["sha256"] == sha for name, sha in PINS.items()), "Reviewed helper/test bytes changed")
    return result


def validate_plan(value):
    require(type(value) is dict and value.get("schema") == "extension-head-objective-qa-plan-v1" and
        value.get("role") == ROLE and value.get("test_pattern") == PATTERN and
        value.get("expected_tests") == EXPECTED_TESTS and value.get("source_files") == sorted(SOURCES) and
        value.get("reviewed_input_sha256") == PINS and value.get("runtime_versions") == VERSIONS and
        value.get("resources") == {"cpus": 2, "ram_gib": 8, "gpus": 0, "wall_time_seconds": WALL_SECONDS,
                                  "reserve_seconds": RESERVE_SECONDS} and
        value.get("deadline_utc") == DEADLINE.isoformat() and value.get("claims") == CLAIMS and
        value.get("native_qa_schema") == SCHEMA and
        value.get("cpu_bootstrap") == {"path": BOOTSTRAP, "outside_source_inventory": True} and
        value.get("tolerances") == {"loss_atol": 1e-6, "metrics_atol": 1e-6,
                                    "gradient_and_update_rtol": 1e-5, "gradient_and_update_atol": 1e-6},
        "Only the reviewed CPU36/source5 scope and fixed prospective tolerances are permitted")
    prior, historical = value.get("prior_ledger", {}), value.get("historical_ledger", {})
    require(type(prior) is dict and set(prior) == {"path", "sha256", "entries", "entries_sha256"} and
        type(prior["path"]) is str and prior["path"].startswith("extension/report/") and
        type(prior["entries"]) is int and prior["entries"] > 0 and
        all(type(prior[key]) is str and re.fullmatch(r"[0-9a-f]{64}", prior[key]) for key in ("sha256", "entries_sha256")) and
        historical.get("path") == "deadline/job_ledger.json" and type(historical.get("sha256")) is str and
        re.fullmatch(r"[0-9a-f]{64}", historical["sha256"]),
        "Root must seal the exact future prior history and historical ledger before admission")
    require(not Path(prior["path"]).is_absolute() and ".." not in Path(prior["path"]).parts,
        "Prior snapshot must be explicit public project metadata")
    return value


def deadline_guard():
    now = datetime.now(timezone.utc)
    require((DEADLINE - now).total_seconds() >= WALL_SECONDS + RESERVE_SECONDS,
        "Keep the complete CPU20-minute stage and48-hour closeout reserve")
    return {"checked_at_utc": now.isoformat(), "guard_seconds": WALL_SECONDS + RESERVE_SECONDS,
            "deadline_utc": DEADLINE.isoformat()}


def require_cpu_allocation():
    """Verify this running zero-GPU allocation before torch/test imports."""
    job, node = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURMD_NODENAME", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) is not None and node and
        socket.gethostname().split(".")[0] == node.split(".")[0] and
        node.split(".")[0].lower() not in {"owl1", "tc2"} and "login" not in node.lower(),
        "A real assigned CPU compute process is required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key, "") in ("", "0", "(null)", "N/A")
        for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")), "GPU visibility/request forbidden")
    command = ["scontrol", "--clusters=owl", "show", "job", "--oneliner", job]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    require(not result.stderr.strip() and len(result.stdout.encode()) <= 128 << 10, "Incomplete allocation proof")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result.stdout))
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    tres = fields.get("AllocTRES", "")
    memory = re.search(r"(?:^|,)mem=([0-9]+)([KMGT])(?:,|$)", tres)
    mem_kib = None if memory is None else int(memory.group(1)) * {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}[memory.group(2)]
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING" and user is not None and
        int(user.group(1)) == os.getuid() and fields.get("NumNodes") == "1" and fields.get("NodeList") == node and
        fields.get("NumCPUs") == "2" and re.search(r"(?:^|,)cpu=2(?:,|$)", tres) is not None and mem_kib == 8 * 1024**2 and
        fields.get("Partition") == "normal_q" and fields.get("Account") == "tml_2026" and
        fields.get("QOS") == "owl_normal_short" and fields.get("TimeLimit") == "00:20:00" and
        all("gpu" not in value.lower() for key, value in fields.items() if "tres" in key.lower() or "gres" in key.lower()),
        "Exact CPU2/8GiB/20-minute single-node zero-GPU allocation required")
    return {"job_id": job, "node": node, "hostname": socket.gethostname(), "user_id": os.getuid(), "state": "RUNNING",
        "allocated_cpus": 2, "allocated_ram_gib": 8, "allocated_tres": tres, "gpu_requested_or_allocated": False,
        "step_id": os.environ.get("SLURM_STEP_ID", "batch"), "cuda_visible_devices": "", "command": command,
        "scontrol": result.stdout.strip(), "scontrol_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def validate_wrapper(root):
    expected = {"job-name": NAME, "account": "tml_2026", "partition": "normal_q", "qos": "owl_normal_short",
        "nodes": "1", "ntasks": "1", "cpus-per-task": "2", "mem": "8G", "time": "00:20:00", "no-requeue": ""}
    found, header = {}, True
    for line in inside(root, SCRIPT).read_text().splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"): header = False
            continue
        require(header, "Late scheduler directive")
        parts = shlex.split(line.lstrip()[7:].strip())
        require(bool(parts), "Empty scheduler directive")
        option, separator, arg = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in set(expected) | {"output", "error"} and option[2:] not in found and
            (len(parts) == 1 if separator else len(parts) in (1, 2)), "Unknown/ambiguous/duplicate scheduler directive")
        found[option[2:]] = arg if separator else parts[1] if len(parts) == 2 else ""
    require(set(found) == set(expected) | {"output", "error"} and all(found[name] == value for name, value in expected.items()),
        "Only the exact finite CPU wrapper is permitted")
    for name in ("output", "error"):
        path = Path(found[name])
        require(path.is_absolute() and ".." not in path.parts and path.parent == root / "extension/logs", "Bounded extension logs required")
    return found


def load_context(root, plan_path, allocation):
    sources = source_inventory(root)
    require(plan_path == PLAN, "The exact reviewed plan path is required")
    plan = validate_plan(read_json(root, plan_path))
    pin = plan["prior_ledger"]
    prior = read_json(root, pin["path"], pin["sha256"])
    require(prior.get("schema") == LEDGER_SCHEMA and prior.get("gpu_hour_cap", "missing") is None and
        prior.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(prior.get("jobs")) is dict and
        len(prior["jobs"]) == pin["entries"] and value_digest(prior["jobs"]) == pin["entries_sha256"] and ROLE not in prior["jobs"],
        "Exact sealed prior history required")
    require(all(type(item) is dict for item in prior["jobs"].values()), "Malformed sealed prior admission")
    historical = record(root, plan["historical_ledger"]["path"])
    require(historical["sha256"] == plan["historical_ledger"]["sha256"], "Historical completed-experiment ledger changed")
    bootstrap = record(root, BOOTSTRAP)
    ledger = read_json(root, "extension/job_ledger.json")
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("gpu_hour_cap", "missing") is None and
        ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and type(ledger.get("jobs")) is dict and
        set(ledger["jobs"]) == set(prior["jobs"]) | {ROLE} and
        all(ledger["jobs"].get(name) == value for name, value in prior["jobs"].items()), "Only this new CPU role may extend sealed history")
    entry = ledger["jobs"][ROLE]
    require(type(entry) is dict and type(entry.get("binding")) is dict, "Current CPU admission binding missing")
    binding = entry["binding"]
    command = ["sbatch", "--clusters=owl", "--parsable", str(root / SCRIPT)]
    require(entry.get("status") == "submitted" and entry.get("job_id") == allocation["job_id"] and
        entry.get("requested_gpu_seconds") == entry.get("requested_gpu_hours") == 0 and entry.get("source_sha256") == sources and
        entry.get("command") == command == binding.get("command") and entry.get("binding_sha256") == value_digest(binding) and
        binding.get("role") == ROLE and binding.get("source_sha256") == sources and binding.get("plan_sha256") == sources[PLAN]["sha256"] and
        binding.get("prior_entries_sha256") == pin["entries_sha256"] and binding.get("expected_native_tests") == EXPECTED_TESTS and
        binding.get("wall_time_seconds") == WALL_SECONDS and binding.get("closeout_reserve_seconds") == RESERVE_SECONDS and
        binding.get("requested_gpu_seconds") == 0 and binding.get("admission_code_sha256") == bootstrap["sha256"] and
        binding.get("initial_evidence", {}).get(BOOTSTRAP) == bootstrap and
        binding.get("automatic_retry") is False and binding.get("automatic_downstream") is False,
        "Exact current-job CPU36/source5/plan/history/bootstrap admission required")
    require(not any(allocation["job_id"] == item.get(key) for item in prior["jobs"].values()
        for key in ("job_id", "possible_job_id")), "Current CPU scheduler identity collides with prior history")
    validate_wrapper(root)
    return {"sources": sources, "plan": plan, "ledger": ledger, "admission": entry, "bootstrap": bootstrap,
        "prior_snapshot": record(root, pin["path"]), "historical_ledger": historical}


def run(root, plan_path, output):
    root = Path(root).resolve()
    require(root == Path(__file__).resolve().parents[1], "Run only the installed source project")
    allocation = require_cpu_allocation()
    guard = deadline_guard()
    context = load_context(root, plan_path, allocation)
    output = Path(output)
    require(not output.is_symlink(), "No output alias")
    output = output.resolve()
    require(output.parent == root / "extension/runs" and output.name == "head-objective-qa-" + allocation["job_id"] and
        not output.exists(), "Exclusive current-job output required; no resume")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "Pinned native fixture versions differ")
    subprocess.run(["bash", "-n", str(root / SCRIPT)], check=True)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    base = {"schema": SCHEMA, "job_id": allocation["job_id"], "node": allocation["node"], "compute_proof": allocation,
        "started_at_utc": started, "source_inventory": context["sources"],
        "code_sha256": {name: item["sha256"] for name, item in context["sources"].items()},
        "plan_sha256": context["sources"][PLAN]["sha256"], "prior_entries_sha256": context["plan"]["prior_ledger"]["entries_sha256"],
        "prior_snapshot": context["prior_snapshot"], "historical_ledger": context["historical_ledger"],
        "admission_binding_sha256": context["admission"]["binding_sha256"], "cpu_bootstrap_sha256": context["bootstrap"]["sha256"],
        "versions": {"python": sys.version, **versions}, "gpu_disabled": True, "wrapper_syntax_passed": True,
        "test_pattern": PATTERN, "expected_tests": EXPECTED_TESTS, "deadline_guard": guard, **CLAIMS}
    write_new_json(output / "started.json", base)
    metrics = {"tests_run": 0, "test_counts": {PATTERN: 0}, "failures": 0, "errors": 0,
               "skipped": 0, "expected_failures": 0, "unexpected_successes": 0}
    begun = time.monotonic()
    try:
        # No torch, head helper or test module is imported before all guards above.
        import torch
        require(str(torch.__version__).split("+")[0] == VERSIONS["torch"] and
            torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == "cpu" and
            not torch.cuda.is_initialized(), "Expected untouched FP32 CPU-only fixture runtime")
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=PATTERN, top_level_dir=str(root / "tests"))
        discovered = suite.countTestCases()
        require(discovered == EXPECTED_TESTS, "Exact36 authored fixtures required; no older scope replay")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        metrics = {"tests_run": result.testsRun, "test_counts": {PATTERN: discovered}, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses)}
        require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and all(metrics[key] == 0 for key in
            ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")), "Synthetic CPU36 fixtures did not all pass")
        require(torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == "cpu" and
            not torch.cuda.is_initialized(), "Fixture runtime mutated CPU/default-precision policy")
        require(load_context(root, plan_path, allocation) == context, "Sources, prior/current/historical ledger or bootstrap changed during QA")
        ending = require_cpu_allocation()
        require(ending["job_id"] == allocation["job_id"] and ending["node"] == allocation["node"], "CPU allocation changed")
        receipt = {**base, **metrics, "success": True, "compute_end": ending,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_elapsed_seconds": time.monotonic() - begun,
            "input_snapshot_unchanged": True, "started_sha256": record(root, str((output / "started.json").relative_to(root)))["sha256"]}
        write_new_json(output / "native_qa.json", receipt)
        write_new_json(output / "complete.json", {"schema": "extension-head-objective-qa-complete-v1", "complete": True,
            "job_id": allocation["job_id"], "tests_run": EXPECTED_TESTS, "gpu_seconds": 0, "cpu_only": True,
            "source_inventory": context["sources"], "plan_sha256": base["plan_sha256"],
            "native_qa_sha256": record(root, str((output / "native_qa.json").relative_to(root)))["sha256"],
            "input_snapshot_unchanged": True, "created_at_utc": datetime.now(timezone.utc).isoformat(), **CLAIMS})
        return receipt
    except BaseException as error:
        write_new_json(output / "failure.json", {**base, **metrics, "success": False,
            "error_type": type(error).__name__, "message": "CPU fixture stage failed; inspect its native log without retry",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begun,
            "input_snapshot_unchanged": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.plan, args.output)
    print(json.dumps({"head_objective_cpu_fixture_qa_complete": True, "tests_run": EXPECTED_TESTS,
                      "gpu_seconds": 0, "actual_model_or_memory_admission": False}), flush=True)


if __name__ == "__main__":
    main()
