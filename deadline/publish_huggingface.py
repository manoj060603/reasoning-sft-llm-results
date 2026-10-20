"""Publish the verified fresh experiment, preserving exact serialized artifacts.

CPU allocation only. No model loading, conversion, selection or evaluation occurs.
An immutable publication manifest claims each fixed destination before content is
added. Existing different content is never overwritten. Repeating the same call
checks remote commits and uploads only missing files; ambiguous repo creation
without a matching manifest requires manual reconciliation. Hub0.36 create_commit
uses parent_commit to reject concurrent changes. Only a verified PUBLIC remote
tree, not an upload response, permits a completion receipt.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal

from deadline.audit_results import OFFICIAL
from deadline.package_deliverables import Files, MODEL_SIDECARS, verify_evaluations
from deadline.train_full import weight_files


OWNER = "adversarialbandit"
BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LICENSE_SHA256 = "ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9"
LICENSE_URL = f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/{BASE_REVISION}/LICENSE"
NOTICE = "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.\n"
MANIFEST = "publication_manifest.json"
DESTINATIONS = {
    "random_model": ("model", "random", "qwen25-3b-reasoning-random15k-20261019"),
    "advanced_model": ("model", "advanced", "qwen25-3b-reasoning-transfer-coverage15k-20261019"),
    "random_dataset": ("dataset", "random", "reasoning-random15k-20261019"),
    "advanced_dataset": ("dataset", "advanced", "reasoning-transfer-coverage15k-20261019"),
}


class PublicationError(ValueError):
    """An actionable publisher-authored error containing no SDK credentials."""


def require(condition, message):
    if not condition:
        raise PublicationError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def freeze(path, data):
    """Retain generated attribution/identity bytes unchanged across retries."""
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, "Previously frozen generated publication artifact changed")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def compute_only():
    require(bool(os.environ.get("SLURM_JOB_ID")) and
            not os.environ.get("SLURM_JOB_PARTITION", "").lower().startswith("login"),
            "Publication and hashing require a CPU Slurm allocation")
    for key in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        require(os.environ.get(key, "") in ("", "0", "(null)", "N/A"),
                "Publication requires a CPU allocation without GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def fingerprint(path, files):
    path = Path(path)
    result = {"sha256": files.hash(path), "bytes": path.stat().st_size}
    # Small regular Git objects use SHA1(blob-header + unmodified bytes).
    # Large weights/data must expose their content SHA256 through LFS metadata.
    if result["bytes"] <= 10 * 1024 * 1024:
        result["git_blob_sha1"] = hashlib.sha1(
            b"blob " + str(result["bytes"]).encode() + b"\0" + path.read_bytes()).hexdigest()
    return result


def prepare(root, package, state):
    """Validate the sealed package and freeze exact allowlisted upload inputs."""
    root, package, state = Path(root).resolve(), Path(package).resolve(), Path(state).resolve()
    metadata, summary = read(package / "package_manifest.json"), read(package / "evaluation_summary.json")
    require(metadata.get("schema") == "deadline-publication-metadata-v1" and
            metadata.get("status") == "staged_metadata_not_uploaded" and metadata.get("uploaded") is False,
            "A complete staged publication package is required")
    require(metadata.get("rows_per_dataset") == 15000 and metadata.get("cutoff") == 8192,
            "Publication requires the frozen15K/8K package")
    require(summary.get("matched_official_eight") is True and
            set(summary.get("models", {})) == {"original", "random", "advanced"},
            "All three complete matched evaluation roles are required")
    files = Files()
    inventory = metadata.get("verified_input_files", {})
    require(isinstance(inventory, dict) and inventory, "Package contains no sealed input inventory")
    for name, record in inventory.items():
        files.hash(name, record["sha256"])
        require(Path(name).stat().st_size == record["bytes"], "Sealed input size changed")
    refs = metadata.get("model_and_dataset_references", {})
    require(set(refs) == {"random", "advanced"}, "Both final trained model references are required")
    for role, reference in refs.items():
        require(Path(reference["folder"]).resolve() == root / "runs" / role / "model" and
                Path(reference["dataset"]).resolve() == root / "artifacts/training" / role / "dataset.json",
                "Package model/dataset references do not match this experiment")
        complete = read(root / "runs" / role / "complete.json")
        require(complete.get("binding_sha256") == reference.get("binding_sha256") and
                complete.get("steps") == 1875 and complete.get("unique_examples") == 15000 and
                complete.get("example_presentations") == 15000 and complete.get("full_parameter") is True and
                complete.get("final_model_verification", {}).get("parameter_dtype") == "float32",
                "Full finite FP32 training completion is missing or changed")
    # Existing audit code rechecks actual matched benchmark coverage/model hashes;
    # no package summary can substitute for actual terminal evaluation evidence.
    require(verify_evaluations(root, refs, files) == summary, "Staged evaluation summary changed")
    plan = read(root / "experiment_plan.json")
    require(plan.get("base_revision") == BASE_REVISION, "Unexpected original Qwen revision")
    license_path = Path(__file__).parent / "licenses/Qwen2.5-3B-Instruct.LICENSE"
    files.hash(license_path, LICENSE_SHA256)
    package_sha = files.hash(package / "package_manifest.json")
    summary_sha = files.hash(package / "evaluation_summary.json")
    generated = state / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    notice_path = generated / "NOTICE"
    freeze(notice_path, NOTICE.encode())
    payloads = {}
    for key, (kind, role, name) in DESTINATIONS.items():
        paths = {"README.md": package / ("models" if kind == "model" else "datasets") / role / "README.md",
                 "evaluation_summary.json": package / "evaluation_summary.json"}
        common = {
            "package_manifest.json": package / "package_manifest.json",
            "experiment_plan.json": root / "experiment_plan.json",
            "selection_complete.json": root / "artifacts/selection_complete.json",
            "selection_manifest.json": root / "artifacts/selected/selection_manifest.json",
            "selected_ids.json": root / f"artifacts/training/{role}/selected_ids.json",
            "cache_meta.json": root / f"artifacts/training/{role}/cache_meta.json",
            "training_manifest.json": root / f"runs/{role}/manifest.json",
            "training_complete.json": root / f"runs/{role}/complete.json",
            "cache_audit.json": root / f"runs/{role}/cache_audit.json",
            "consumption_final.json": root / f"runs/{role}/consumption_final.json",
            "evaluation_manifest.json": root / f"results/{role}/manifest.json",
            "evaluation_metrics.json": root / f"results/{role}/metrics.json",
        }
        for name_in_repo, path in common.items():
            if name_in_repo != "package_manifest.json":
                require(str(path.resolve()) in inventory, "Required provenance is not sealed by the package: " + name_in_repo)
            paths["provenance/" + name_in_repo] = path
        if kind == "model":
            require("Built with Qwen" in paths["README.md"].read_text(encoding="utf-8"),
                    "Model card must prominently attribute Built with Qwen")
            model = Path(refs[role]["folder"])
            weights = weight_files(model)
            require({p.name for p in model.glob("*.safetensors")} == {p.name for p in weights},
                    "Unexpected unsealed weight file in final model")
            for path in weights + [model / name for name in sorted(MODEL_SIDECARS) if (model / name).is_file()]:
                require(str(path.resolve()) in inventory, "Model file is not bound by the package: " + path.name)
                paths[path.name] = path
            paths.update(LICENSE=license_path, NOTICE=notice_path)
            paths["provenance/LICENSE.source.json"] = license_path.with_name(license_path.name + ".source.json")
            modified = generated / f"{role}-MODIFICATIONS.md"
            freeze(modified, ("Built with Qwen.\n\nAll model*.safetensors weight files in this repository were modified "
                "by full-parameter supervised fine-tuning of Qwen2.5-3B-Instruct, revision " + BASE_REVISION +
                ". The original weights were not converted or substituted during publication; serialized trained "
                "FP32 bytes are preserved. Configuration/tokenizer provenance and all uploaded-file hashes are in "
                "publication_manifest.json.\n").encode())
            paths["MODIFICATIONS.md"] = modified
        else:
            paths["data/train.json"] = Path(refs[role]["dataset"])
        descriptors = {name_in_repo: fingerprint(path, files) for name_in_repo, path in sorted(paths.items())}
        for name_in_repo, descriptor in descriptors.items():
            descriptor["require_lfs"] = name_in_repo.endswith(".safetensors") or name_in_repo == "data/train.json"
        payloads[key] = dict(repo_id=OWNER + "/" + name, repo_type=kind, role=role,
                             files=descriptors, local_paths={k: str(v.resolve()) for k, v in paths.items()})
    identity = dict(schema="deadline-hf-publication-v1", owner=OWNER,
        package_manifest_sha256=package_sha, evaluation_summary_sha256=summary_sha,
        selection_sha256=metadata["selection_sha256"],
        destinations=copy.deepcopy({key: {k: v for k, v in value.items() if k != "local_paths"}
                                    for key, value in payloads.items()}),
        attribution=dict(base_model="Qwen/Qwen2.5-3B-Instruct", base_revision=BASE_REVISION,
            license_url=LICENSE_URL, license_sha256=LICENSE_SHA256,
            source="https://huggingface.co/datasets/redsgnaoh/acereason11_100k",
            source_revision=plan["source_revision"], upstream_data="https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT",
            upstream_data_license="https://creativecommons.org/licenses/by/4.0/"))
    identity["publication_sha256"] = sha(canonical(identity))
    if (state / "publication_plan.json").exists():
        require(read(state / "publication_plan.json") == identity,
                "Publication inputs changed; refusing to reuse destinations/state")
    for key, payload in payloads.items():
        manifest = dict(schema=identity["schema"], publication_sha256=identity["publication_sha256"],
            owner=OWNER, repo_id=payload["repo_id"], repo_type=payload["repo_type"], role=payload["role"],
            package_manifest_sha256=package_sha, evaluation_summary_sha256=summary_sha,
            selection_sha256=metadata["selection_sha256"], attribution=identity["attribution"], files=payload["files"])
        manifest_path = generated / (key + "-manifest.json")
        freeze(manifest_path, canonical(manifest))
        payload["local_paths"][MANIFEST] = str(manifest_path)
        payload["files"][MANIFEST] = dict(fingerprint(manifest_path, files), require_lfs=False)
    files.unchanged()
    return identity, payloads, files


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def verify_file(remote, expected):
    require(field(remote, "size") == expected["bytes"], "Remote file size differs")
    lfs = field(remote, "lfs")
    if lfs is not None:
        require(field(lfs, "sha256") == expected["sha256"] and field(lfs, "size") == expected["bytes"],
                "Remote LFS SHA256/size differs")
        return {"method": "LFS SHA256 and size", "sha256": expected["sha256"], "bytes": expected["bytes"]}
    require(not expected.get("require_lfs") and expected.get("git_blob_sha1") is not None and
            field(remote, "blob_id") == expected["git_blob_sha1"], "Remote Git blob is unverified or LFS metadata is missing")
    return {"method": "Git blob SHA1 and size", "blob_sha1": expected["git_blob_sha1"], "bytes": expected["bytes"]}


class Hub:
    """Thin0.36-compatible SDK adapter; never print authentication or HTTP errors."""
    def __init__(self):
        require(importlib.metadata.version("huggingface_hub") == "0.36.0",
                "Publisher SDK adapter requires the verified huggingface_hub0.36.0 runtime")
        os.environ["HF_DEBUG"] = "0"
        os.environ["HF_HUB_VERBOSITY"] = "error"
        import requests
        from huggingface_hub import HfApi, configure_http_backend
        from huggingface_hub.utils import RepositoryNotFoundError

        class BoundedSession(requests.Session):
            def request(self, *args, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = (15, 120)
                return super().request(*args, **kwargs)

        configure_http_backend(backend_factory=BoundedSession)
        self.api, self.not_found = HfApi(endpoint="https://huggingface.co"), RepositoryNotFoundError

    def whoami(self):
        return self.api.whoami()["name"]

    def snapshot(self, repo_id, repo_type, *, public=False):
        token = False if public else None
        try:
            info = self.api.repo_info(repo_id, repo_type=repo_type, token=token)
        except self.not_found:
            return None
        require(field(info, "id") == repo_id,
                "Destination resolved to a different canonical namespace/repository")
        require(field(info, "private") is False and field(info, "gated", False) in (False, None),
                "Publication destination must be public and ungated")
        commit = info.sha
        tree = {entry.path: {"size": entry.size, "blob_id": entry.blob_id, "lfs": entry.lfs}
                for entry in self.api.list_repo_tree(repo_id, repo_type=repo_type, revision=commit,
                                                     recursive=True, token=token) if hasattr(entry, "blob_id")}
        return {"commit": commit, "files": tree}

    def create(self, repo_id, repo_type):
        self.api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=False)

    def commit(self, repo_id, repo_type, paths, parent):
        from huggingface_hub import CommitOperationAdd
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in paths.items()]
        result = self.api.create_commit(repo_id, repo_type=repo_type, operations=operations,
            commit_message="Publish verified fresh reasoning experiment artifacts", parent_commit=parent, num_threads=2)
        return result.oid


def inspect(hub, payload, journal, *, public=False):
    snapshot = hub.snapshot(payload["repo_id"], payload["repo_type"], public=public)
    if snapshot is None:
        return None
    tree, expected = snapshot["files"], payload["files"]
    require(not (set(tree) - set(expected) - {".gitattributes"}), "Destination contains unexpected files; refusing to overwrite")
    if MANIFEST not in tree:
        require(journal.get("created") is True and set(tree) <= {".gitattributes"},
                "Existing destination has no matching publication identity; reconcile manually")
    else:
        verify_file(tree[MANIFEST], expected[MANIFEST])
    for name in set(tree) & set(expected):
        verify_file(tree[name], expected[name])
    return snapshot


def publish(root, package, state, *, expected_owner, hub=None):
    compute_only()
    require(expected_owner == OWNER, "Expected owner must explicitly equal adversarialbandit")
    state = Path(state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    with (state / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hub = Hub() if hub is None else hub
        require(hub.whoami() == OWNER, "Authenticated Hugging Face account differs from expected owner")
        identity, payloads, files = prepare(root, package, state)
        plan_path, journal_path = state / "publication_plan.json", state / "journal.json"
        if plan_path.exists():
            require(read(plan_path) == identity, "Publication inputs changed; refusing to reuse destinations/state")
        else:
            write(plan_path, identity)
        journal = read(journal_path) if journal_path.exists() else {"publication_sha256": identity["publication_sha256"], "repos": {}}
        require(journal["publication_sha256"] == identity["publication_sha256"], "Publication journal identity differs")
        # Detect ALL destination conflicts before creating or committing ANY repo.
        for key, payload in payloads.items():
            inspect(hub, payload, journal["repos"].get(key, {}))
        receipts = {}
        for key, payload in payloads.items():
            record = journal["repos"].setdefault(key, {})
            journal["active_destination"] = key
            write(journal_path, journal)
            snapshot = inspect(hub, payload, record)
            if snapshot is None:
                hub.create(payload["repo_id"], payload["repo_type"])
                record["created"] = True
                write(journal_path, journal)
                snapshot = inspect(hub, payload, record)
                require(snapshot is not None, "Created destination is not observable; rerun after reconciliation")
            if MANIFEST not in snapshot["files"]:
                files.unchanged()
                committed = hub.commit(payload["repo_id"], payload["repo_type"],
                    {MANIFEST: payload["local_paths"][MANIFEST]}, snapshot["commit"])
                snapshot = inspect(hub, payload, record)
                require(snapshot is not None and snapshot["commit"] == committed and MANIFEST in snapshot["files"],
                        "Remote identity commit was not independently verified")
            missing = [name for name in payload["files"] if name not in snapshot["files"]]
            # License/NOTICE/cards precede exact large weights. Each large file
            # is committed separately so a later interruption preserves progress.
            small = [name for name in missing if not payload["files"][name]["require_lfs"]]
            groups = ([small] if small else []) + [[name] for name in missing if name not in small]
            for names in groups:
                files.unchanged()
                journal["active_files"] = names
                write(journal_path, journal)
                committed = hub.commit(payload["repo_id"], payload["repo_type"],
                    {name: payload["local_paths"][name] for name in names}, snapshot["commit"])
                snapshot = inspect(hub, payload, record)
                require(snapshot is not None and snapshot["commit"] == committed and
                        all(name in snapshot["files"] for name in names), "Remote commit was not independently verified")
                record["last_verified_commit"] = committed
                write(journal_path, journal)
            public = inspect(hub, payload, record, public=True)
            require(public is not None and set(payload["files"]).issubset(public["files"]),
                    "Public remote tree is incomplete")
            proof = {name: verify_file(public["files"][name], expected) for name, expected in payload["files"].items()}
            prefix = "datasets/" if payload["repo_type"] == "dataset" else ""
            url = "https://huggingface.co/" + prefix + payload["repo_id"]
            receipts[key] = dict(repo_id=payload["repo_id"], repo_type=payload["repo_type"], url=url,
                commit=public["commit"], tree_url=url + "/tree/" + public["commit"], public_access_verified=True,
                files=proof, complete=True)
            record["receipt"] = receipts[key]
            write(journal_path, journal)
        # Recheck all four public heads, including those uploaded earlier.
        for key, payload in payloads.items():
            current = inspect(hub, payload, journal["repos"][key], public=True)
            require(current is not None and current["commit"] == receipts[key]["commit"],
                    "Remote publication changed before final completion")
        files.unchanged()
        receipt = dict(schema="deadline-hf-publication-complete-v1", complete=True,
            owner=OWNER, publication_sha256=identity["publication_sha256"],
            package_manifest_sha256=identity["package_manifest_sha256"],
            evaluated_and_published_weight_bytes_identical=True, repositories=receipts,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(state / "publication_complete.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("deadline"))
    parser.add_argument("--package", type=Path, default=Path("deadline/publication-metadata"))
    parser.add_argument("--state", type=Path, default=Path("deadline/hf/publication"))
    parser.add_argument("--expected-owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    args = parser.parse_args()
    require(60 <= args.timeout_seconds <= 7200, "Publication timeout must be 60..7200 seconds")
    def expired(_signum, _frame):
        raise TimeoutError("Publication time budget exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_seconds)
    try:
        receipt = publish(args.root, args.package, args.state, expected_owner=args.expected_owner)
        print(json.dumps({"complete": True, "repositories": {key: value["url"]
            for key, value in receipt["repositories"].items()}}, indent=2), flush=True)
    except Exception as error:
        # Never echo SDK exception text, headers, token objects or presigned URLs.
        safe_reason = str(error) if isinstance(error, PublicationError) else None
        write(args.state / "failure.json", {"complete": False, "error_type": type(error).__name__,
            "reason": safe_reason, "at_utc": datetime.now(timezone.utc).isoformat(),
            "instruction": "Retain state and rerun the same invocation; inspect journal stage. Conflicting or ambiguous destinations require manual reconciliation."})
        raise SystemExit("Publication incomplete (" + type(error).__name__ + "). " +
                         ((safe_reason + ". ") if safe_reason else "") + "Retained resumable state; no success inferred.") from None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
