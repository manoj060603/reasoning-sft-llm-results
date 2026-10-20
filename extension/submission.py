"""Metadata-only admission of the single extension CPU feasibility stage.

This adapter neither imports model/data code nor admits GPU work. The completed
experiment's deadline/ledger are never opened or modified. An ambiguous or
failed submission requires explicit scheduler reconciliation, never a retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

ROLE = "cpu_feasibility_20261020_v1"
SCHEMA = "reasoning-sft-extension-job-ledger-v1"
DEADLINE = datetime(2026, 10, 28, 3, 59, tzinfo=timezone.utc)
WALL_SECONDS, RESERVE_SECONDS = 1800, 3600
MAX_SOURCE_BYTES, MAX_TOTAL_SOURCE_BYTES = 1 << 20, 8 << 20
REQUIRED = frozenset({"extension/__init__.py", "extension/submission.py",
    "extension/cpu_feasibility.py", "extension/cpu_feasibility.slurm", "extension/campaign.json",
    "extension/research.json", "extension/development_source_metadata.json",
    "tests/test_extension_submission.py", "tests/test_extension_feasibility.py", "reasoning/data.py"})


class SubmissionError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SubmissionError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate metadata key")
            result[key] = value
        return result
    require(Path(path).is_file() and Path(path).stat().st_size <= 2 << 20,
            "Expected bounded metadata JSON")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(SubmissionError("Nonfinite metadata")))


def _small_bytes(path, root):
    path, root = Path(path), Path(root)
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Source path escapes the project")
    before = path.stat()
    require(path.is_file() and 0 < before.st_size <= MAX_SOURCE_BYTES, "Expected small source metadata")
    value = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "Source changed during binding")
    return value


def source_inventory(root):
    """Only small source/config/test files; never models, caches or runtime data."""
    root = Path(root).resolve()
    extension = root / "extension"
    paths = {path for suffix in ("*.py", "*.slurm") for path in extension.glob(suffix)}
    paths.update(extension/name for name in
                 ("campaign.json","research.json","development_source_metadata.json"))
    paths.update((root / "tests").glob("test_extension_*.py"))
    paths.add(root / "reasoning/data.py")
    names = {str(path.relative_to(root)) for path in paths}
    require(REQUIRED <= names, "Missing required reviewed extension source/test files")
    inventory, total = {}, 0
    for path in sorted(paths):
        value = _small_bytes(path, root)
        total += len(value)
        require(total <= MAX_TOTAL_SOURCE_BYTES, "Extension source inventory exceeds its metadata-only bound")
        inventory[str(path.relative_to(root))] = dict(sha256=_sha(value), bytes=len(value))
    return inventory


def validate_campaign(value):
    require(isinstance(value, dict) and value.get("schema") == "reasoning-sft-extension-campaign-v1" and
            value.get("preserve_completed_experiments") is True,
            "Expected the separate extension campaign preserving completed experiments")
    auth, budget = value.get("authorization", {}), value.get("budget", {})
    require(auth.get("new_deadline_utc") == DEADLINE.isoformat() and
            auth.get("new_deadline_eastern") == "2026-10-27T23:59:00-04:00",
            "Campaign does not match the newly authorized deadline")
    require("latest_user_total_gpu_hour_limit" in auth and auth["latest_user_total_gpu_hour_limit"] is None and
            "new_user_global_cap_gpu_hours" in budget and budget["new_user_global_cap_gpu_hours"] is None and
            budget.get("historical_cap_and_ledger_remain_unchanged") is True and
            budget.get("historical_ledger") == "deadline/job_ledger.json" and
            budget.get("new_extension_ledger") == "extension/job_ledger.json",
            "Extension authorization/history/ledger scope differs")


def validate_wrapper(path, root):
    text = _small_bytes(path, root).decode("utf-8")
    allowed = {"job-name", "account", "partition", "qos", "nodes", "ntasks", "cpus-per-task",
               "mem", "time", "no-requeue", "output", "error"}
    directives, header = {}, True
    for line in text.splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            if line.strip() and not line.lstrip().startswith("#"):
                header = False
            continue
        require(header, "SBATCH directives after executable code would be ignored")
        parts = shlex.split(line.lstrip()[7:].strip(), comments=True)
        require(bool(parts), "Empty SBATCH directive")
        option, separator, argument = parts[0].partition("=")
        require(option.startswith("--") and option[2:] in allowed, "GPU/unknown SBATCH option is forbidden")
        key = option[2:]
        require(key not in directives, "Duplicate SBATCH option")
        if separator:
            require(len(parts) == 1, "Ambiguous SBATCH directive")
        else:
            require(len(parts) in (1, 2), "Ambiguous SBATCH directive")
            argument = parts[1] if len(parts) == 2 else ""
        directives[key] = argument
    require(set(directives) == allowed, "The fixed CPU wrapper must declare every bounded scheduler option")
    require(directives["time"] == "00:30:00" and directives["nodes"] == "1" and
            directives["ntasks"] == "1" and directives["no-requeue"] == "" and
            directives["account"] == "tml_2026" and directives["partition"] == "normal_q" and
            directives["qos"] == "owl_normal_short" and
            directives["job-name"] == "sft-extension-feasibility",
            "Only the reviewed single-node, zero-GPU, 30-minute CPU stage is permitted")
    require(re.fullmatch(r"[1-9][0-9]*", directives["cpus-per-task"]) and
            re.fullmatch(r"[1-9][0-9]*[MG]", directives["mem"]), "Finite CPU/memory bounds are required")
    for key in ("output", "error"):
        target = Path(directives[key])
        require(target.is_absolute() and target.resolve().is_relative_to(Path(root)/"extension") and
                ".." not in target.parts, "Runtime logs must stay below extension")
    return directives


def _write(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Refusing a symlinked extension ledger")
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _queue_snapshot(runner):
    user = getpass.getuser()
    require(isinstance(user, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", user), "Unknown scheduler user")
    command = ["squeue", "--clusters=owl", "--array", "--states=all", "--user="+user,
               "--noheader", "--format=%64i|%64T|%128j"]
    try:
        result = runner(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise SubmissionError("Cannot establish current scheduler snapshot") from error
    require(result.returncode == 0 and not result.stderr.strip(), "Scheduler snapshot failed or was incomplete")
    require(len(result.stdout.encode()) <= 256 << 10, "Scheduler snapshot exceeds its metadata bound")
    jobs, seen = [], set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line == "CLUSTER: owl":
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        require(len(parts) == 3 and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", parts[0]) and
                re.fullmatch(r"[A-Z_]+", parts[1]) and parts[2] and parts[0] not in seen,
                "Ambiguous/duplicate/nonexpanded scheduler row")
        seen.add(parts[0])
        jobs.append(dict(job_id=parts[0], state=parts[1], name=parts[2]))
    return dict(command=command, user=user, stdout=result.stdout, stderr=result.stderr,
                stdout_sha256=_sha(result.stdout.encode()), jobs=jobs,
                observed_at_utc=datetime.now(timezone.utc).isoformat())


def _submission_environment():
    # SBATCH_* can override script directives, including adding GPUs or arrays.
    # Do not export an enclosing allocation's GPU/job identity into the CPU job.
    removed = {key for key in os.environ if key.startswith("SBATCH_") or
               (key.startswith("SLURM_") and key not in ("SLURM_CONF", "SLURM_CONF_SERVER")) or
               key == "CUDA_VISIBLE_DEVICES"}
    return {key: value for key, value in os.environ.items() if key not in removed}, sorted(removed)


def submit_cpu_feasibility(root=Path("."), *, now=None, runner=None):
    """Return its one scheduler ID; an exact submitted binding is idempotent.

    ``root`` is the project directory (including unit-fixture projects). No role,
    script, scheduler options, arguments or GPU request can be overridden.
    """
    root = Path(root).resolve()
    extension = root/"extension"
    require(extension.is_dir() and not extension.is_symlink() and extension.resolve() == extension,
            "A real isolated extension directory is required")
    runner = subprocess.run if runner is None else runner
    clock = (lambda: datetime.now(timezone.utc)) if now is None else (lambda: now)
    def guard():
        checked = clock()
        require(checked.tzinfo is not None and (DEADLINE-checked).total_seconds() >= WALL_SECONDS+RESERVE_SECONDS,
                "Insufficient time for the CPU stage and one-hour deadline reserve")
        return checked
    now = guard()
    _small_bytes(extension/"campaign.json", root)
    campaign = _read(extension/"campaign.json")
    validate_campaign(campaign)
    script = extension/"cpu_feasibility.slurm"
    directives = validate_wrapper(script, root)
    sources = source_inventory(root)
    command = ["sbatch", "--clusters=owl", "--parsable", str(script)]
    binding = dict(role=ROLE, command=command, campaign_sha256=sources["extension/campaign.json"]["sha256"],
                   source_sha256=sources, requested_gpu_hours=0, requested_gpu_seconds=0,
                   wall_time_seconds=WALL_SECONDS, deadline_utc=DEADLINE.isoformat(),
                   closeout_reserve_seconds=RESERVE_SECONDS)
    binding_sha256 = _sha(_canonical(binding))
    ledger_path, lock_path = extension/"job_ledger.json", extension/"job_ledger.lock"
    require(not ledger_path.is_symlink() and not lock_path.is_symlink(), "Extension ledger/lock cannot be symlinks")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(ledger_path) if ledger_path.exists() else dict(schema=SCHEMA,
            historical_ledger_unchanged="deadline/job_ledger.json", gpu_hour_cap=None, jobs={})
        require(isinstance(ledger, dict) and ledger.get("schema") == SCHEMA and
                ledger.get("historical_ledger_unchanged") == "deadline/job_ledger.json" and
                ledger.get("gpu_hour_cap", "missing") is None and isinstance(ledger.get("jobs"), dict) and
                set(ledger["jobs"]) <= {ROLE}, "Unknown extension ledger or admitted role; reconcile manually")
        snapshot = _queue_snapshot(runner)
        require(source_inventory(root) == sources and validate_wrapper(script, root) == directives,
                "Source changed during admission")
        existing = ledger["jobs"].get(ROLE)
        same_name = {row["job_id"] for row in snapshot["jobs"] if row["name"] == directives["job-name"]}
        if existing is not None:
            require(isinstance(existing, dict) and existing.get("status") == "submitted" and
                    existing.get("binding_sha256") == binding_sha256 and existing.get("binding") == binding and
                    existing.get("command") == command and existing.get("source_sha256") == sources and
                    existing.get("campaign_sha256") == binding["campaign_sha256"] and
                    existing.get("requested_gpu_hours") == 0 and existing.get("requested_gpu_seconds") == 0 and
                    isinstance(existing.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*",existing["job_id"]),
                    "Existing submission is changed, uncertain or failed; reconcile manually, never resubmit")
            require(not (same_name-{existing["job_id"]}), "An untracked duplicate CPU job requires reconciliation")
            return existing["job_id"]
        require(not same_name, "An untracked live CPU feasibility job requires reconciliation")
        now = guard()
        for key in ("output","error"):
            Path(directives[key]).parent.mkdir(parents=True,exist_ok=True)
        environment, removed = _submission_environment()
        entry = dict(status="submitting", binding=binding, binding_sha256=binding_sha256,
                     command=command, source_sha256=sources, campaign_sha256=binding["campaign_sha256"],
                     requested_gpu_hours=0, requested_gpu_seconds=0, requested_at_utc=now.isoformat(),
                     scheduler_snapshot=snapshot, removed_scheduler_environment_keys=removed)
        ledger["jobs"][ROLE] = entry
        _write(ledger_path, ledger)
        try:
            completed = runner(command, capture_output=True, text=True, timeout=60, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            entry.update(status="uncertain", error_type=type(error).__name__)
            for name in ("stdout", "stderr"):
                value = getattr(error,name,None)
                if value:
                    entry["scheduler_"+name] = (value.decode(errors="replace") if isinstance(value,bytes) else str(value))[-2000:]
            _write(ledger_path,ledger)
            raise SubmissionError("Scheduler submission is uncertain; reconcile manually before any retry") from error
        entry.update(scheduler_stdout=completed.stdout[-2000:], scheduler_stderr=completed.stderr[-2000:],
                     scheduler_returncode=completed.returncode)
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;owl)?",completed.stdout.strip())
        if completed.returncode != 0:
            entry["status"] = "submission_failed"
            if parsed:
                entry["possible_job_id"] = parsed.group(1)
            _write(ledger_path,ledger)
            raise SubmissionError("Scheduler rejected/failed submission; preserve evidence and reconcile, never resubmit")
        if parsed is None:
            entry["status"] = "uncertain"
            _write(ledger_path,ledger)
            raise SubmissionError("Ambiguous scheduler response; preserve evidence and do not submit a duplicate")
        entry.update(status="submitted",job_id=parsed.group(1),
                     submitted_at_utc=datetime.now(timezone.utc).isoformat())
        _write(ledger_path,ledger)
        return entry["job_id"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    args=parser.parse_args()
    identifier=submit_cpu_feasibility(args.root)
    print(json.dumps(dict(role=ROLE,job_id=identifier,requested_gpu_hours=0)),flush=True)


if __name__ == "__main__":
    main()
