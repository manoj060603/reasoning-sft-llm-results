"""UNADMITTED bounded hooks for the separate native numerical diagnostic.

No model/data loading, forward/backward, CLI, job, admission or save/reload is
provided. Already live original Trainers/reference and source-bound native
primitives are mandatory. Their real source/current-job/origin/resource truth
still requires future worker review. Synthetic CPU seams are always NONNATIVE.

Full FQN comparisons are persisted in exclusive owner-only bounded reports;
controllers receive only aggregate statistics and physical FILE pins. Tensor
maps reference existing storage and are never cloned/offloaded. The one common
initial shadow is created only after exact post-LR0 original-byte identity.
The supplied ScratchBudget is cumulative supervisor-owned accounting. This
adapter cannot establish a cross-process budget/snapshot protocol, discover
external writes or reserve physical blocks. No per-case reset is supplied.

Authoring is not execution. This new draft has not been imported or tested.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import types

import torch

from extension import native_tensor_comparison as comparison
from extension import paired_training_step_controller as paired
from extension.standalone_training_step_controller import StandaloneStepHooks


PROTOCOL_SHA256 = "27cedace388a17f3b97a0b1466e119c59a4d63913085263c689c8cfbbb003cb3"
BASE_IDENTITY_SHA256 = "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2"
SCHEMA = "extension-checkpointed-head-native-pair-hooks-v1"
MAX_REPORT_BYTES, MAX_RECEIPT_BYTES = 2 << 20, 128 << 10
PRIMITIVES = ("verify_native_context", "verify_original_numerics", "verify_phase_accounting")
CASE_HASHES = frozenset({"base_identity_file_sha256", "input_bundle_value_sha256", "terminal_suffix_value_sha256",
    "admission_binding_sha256", "context_receipt_file_sha256", "stage_budget_snapshot_value_sha256"})
CASE_KEYS = CASE_HASHES | {"schema", "objective", "subcase", "expected_parameters", "fixture_only", "device",
    "source_root", "source_inventory", "source_inventory_value_sha256", "numerical_protocol_file_sha256"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _bytes(value):
    _require(paired._plain(value), "Only finite plain JSON metadata may be persisted")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)[:-1]).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _path(path, *, directory=False):
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts and all(not item.is_symlink() for item in (path, *path.parents)),
        "Explicit absolute no-alias path required")
    if directory:
        value = path.stat()
        _require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid() and stat.S_IMODE(value.st_mode) == 0o700,
            "An owner0700 private directory is required")
    return path


def _file(path, *, maximum, private=False):
    path = _path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, "Bounded ordinary file required")
        if private:
            _require(before.st_uid == os.getuid() and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1,
                "Closed report must remain owner0600/single-link")
        payload, left = [], before.st_size
        while left:
            chunk = os.read(fd, min(left, 1 << 20))
            _require(bool(chunk), "Truncated bound file")
            payload.append(chunk)
            left -= len(chunk)
        after = os.fstat(fd)
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
            value.st_uid, stat.S_IMODE(value.st_mode), value.st_nlink)
        _require(not os.read(fd, 1) and fields(before) == fields(after) == fields(path.stat()), "Bound file changed while reading")
        data = b"".join(payload)
        return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    finally:
        os.close(fd)


def tensor_tolerances(kind):
    """Literal translation of frozen27ced; absent gates remain explicitNone."""
    table = {
        "identity": (0., 0., None, None, None, None, False),
        "gradients": (1e-6, .05, 1e-7, .05, .01, .9999, False),
        "adam_first_moments": (1e-7, .05, None, None, .01, None, False),
        "adam_second_moments": (1e-12, .1, None, None, .03, None, False),
        "updated_weights": (1e-7, 1e-6, None, None, None, None, False),
        "actual_update2_deltas": (2e-8, .1, None, None, .05, None, True)}
    _require(kind in table, "No tolerance override or unknown comparison kind")
    keys = ("elementwise_atol", "elementwise_rtol", "rms_atol", "rms_rtol", "group_relative_l2_maximum",
        "group_cosine_minimum", "require_positive_group_norms")
    return comparison.validate_tolerances(dict(zip(keys, table[kind])))


@dataclass(frozen=True)
class NativeHookPrimitives:
    verify_native_context: object
    verify_original_numerics: object
    verify_phase_accounting: object

    def __post_init__(self):
        _require(all(callable(getattr(self, name)) for name in PRIMITIVES), "Every native primitive is mandatory; no success defaults")


@dataclass(frozen=True)
class _BoundShadow:
    reader: object
    case_binding_sha256: str
    phase_binding_sha256: str
    manifest_file_sha256: str


class PairHookFailure(RuntimeError):
    def __init__(self, result):
        self.partial_result = paired._json_copy(result, MAX_REPORT_BYTES)
        super().__init__("Native pair hook failed; preserve its private partial reports and abort")


class NativePairHookAdapter:
    """One objective/subcase adapter; no stage reuse or implicit native proof."""
    def __init__(self, *, private_directory, scratch_budget, protocol_path, case_binding, primitives,
                 primitive_sources, fixture_only=False, expected_parameters=434):
        _require(type(fixture_only) is bool and type(expected_parameters) is int and 2 <= expected_parameters <= 434 and
            (fixture_only or expected_parameters == 434), "Production requires434; explicit small fixtures are NONNATIVE")
        binding = paired._json_copy(case_binding, MAX_REPORT_BYTES)
        _require(type(binding) is dict and set(binding) == CASE_KEYS and
            binding["schema"] == "extension-native-pair-hook-case-binding-v1" and
            binding["objective"] in ("sft", "ce_kl", "asft", "prior_anchored") and binding["subcase"] in ("standalone", "ga9") and
            type(binding["expected_parameters"]) is int and binding["expected_parameters"] == expected_parameters and
            binding["fixture_only"] is fixture_only and binding["device"] == ("cpu" if fixture_only else "cuda:0") and
            all(_sha(binding[name]) for name in CASE_HASHES) and
            (fixture_only or binding["base_identity_file_sha256"] == BASE_IDENTITY_SHA256) and
            binding["numerical_protocol_file_sha256"] == PROTOCOL_SHA256,
            "Exact source-bound native case identity is required")
        self.root = _path(binding["source_root"])
        inventory = binding["source_inventory"]
        _require(type(inventory) is dict and inventory and _digest(inventory) == binding["source_inventory_value_sha256"] and
            all(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts and
                type(row) is dict and set(row) == {"sha256", "bytes"} and _sha(row["sha256"]) and
                type(row["bytes"]) is int and 0 < row["bytes"] <= 2 << 20 for name, row in inventory.items()),
            "Case must bind the reviewed source inventory")
        protocol_file = _path(protocol_path)
        _require(protocol_file == self.root / "extension/planning/checkpointed_head_native_numerical_protocol.json",
            "Use the exact root-frozen numerical protocol path")
        payload, protocol_record = _file(protocol_file, maximum=2 << 20)
        _require(protocol_record["sha256"] == PROTOCOL_SHA256, "Frozen numerical protocol FILE SHA differs")
        self.protocol = json.loads(payload)
        _require(self.protocol["schema"] == "extension-checkpointed-head-native-numerical-protocol-v1",
            "Wrong numerical protocol schema")
        _require(type(primitives) is NativeHookPrimitives and type(primitive_sources) is dict and set(primitive_sources) == set(PRIMITIVES),
            "Actual reviewed native primitive sources are mandatory")
        self.case, self.case_sha = binding, _digest(binding)
        self.primitives, self.primitive_sources = primitives, paired._json_copy(primitive_sources)
        self.fixture_only, self.expected_parameters = fixture_only, expected_parameters
        self._verify_primitive_sources()
        _require(type(scratch_budget) is comparison.ScratchBudget, "Pass the supervisor-owned actual ScratchBudget; no new/reset per-case budget")
        self.budget = scratch_budget
        self.directory = _path(private_directory)
        _require(self.directory.parent == self.budget.directory and not self.directory.exists(), "Fresh direct-child private report directory required")
        _path(self.directory.parent, directory=True)
        self.directory.mkdir(mode=0o700, exist_ok=False)
        _path(self.directory, directory=True)
        self.reports, self._models, self._shadow, self._last_identity = [], None, None, None
        self._pending_report, self._shadow_directory = None, None
        self._failed, self._entered = False, set()

    def _verify_primitive_sources(self):
        for name in PRIMITIVES:
            supplied = self.primitive_sources[name]
            fn = getattr(self.primitives, name)
            raw = fn.__func__ if isinstance(fn, types.MethodType) else fn
            _require(isinstance(raw, types.FunctionType) and type(supplied) is dict and
                set(supplied) == {"path", "sha256", "qualname"} and supplied["path"] in self.case["source_inventory"] and
                self.case["source_inventory"][supplied["path"]]["sha256"] == supplied["sha256"] and
                raw.__qualname__ == supplied["qualname"], "Primitive callable/source/qualname is not bound")
            if not self.fixture_only:
                _require(raw.__name__ != "<lambda>" and not raw.__closure__ and "<locals>" not in raw.__qualname__,
                    "Production primitives forbid lambda/closure/dynamic/partial bypasses")
                resolved = sys.modules.get(raw.__module__)
                _require(resolved is not None, "Primitive defining module missing")
                _require(raw.__globals__ is vars(resolved),
                    "Primitive globals must be its actual defining-module namespace")
                arguments = ("stage", "dense_trainer", "candidate_trainer", "reference_model", "context")
                positional = int(isinstance(fn, types.MethodType))
                code = raw.__code__
                _require(raw.__defaults__ is None and raw.__kwdefaults__ is None and
                    code.co_argcount == positional and code.co_posonlyargcount == 0 and
                    code.co_kwonlyargcount == len(arguments) and
                    code.co_varnames[positional:positional + len(arguments)] == arguments and
                    not code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS),
                    "Production primitives require exactly five mandatory keyword-only arguments and no defaults")
                for part in raw.__qualname__.split("."):
                    resolved = getattr(resolved, part, None)
                resolved = resolved.__func__ if isinstance(resolved, (staticmethod, classmethod, types.MethodType)) else resolved
                _require(resolved is raw, "Primitive must be its actual source-module function/method")
            source = _path(inspect.getsourcefile(raw) or "")
            _require(source == self.root / supplied["path"] and Path(raw.__code__.co_filename).resolve() == source.resolve(),
                "Primitive runtime code file differs from the reviewed inventory")
            payload, record = _file(source, maximum=2 << 20)
            _require(record == self.case["source_inventory"][supplied["path"]], "Primitive source bytes changed")
            if not self.fixture_only:
                _require(raw.__code__.co_name == raw.__name__ == raw.__qualname__.split(".")[-1],
                    "Production primitive function metadata cannot disguise another code object")
                # Compile the pinned source into code metadata only; NEVER exec
                # it or reimport the worker. This binds runtime bytecode to the
                # original source rather than trusting a forged co_filename.
                compiled = compile(payload, raw.__code__.co_filename, "exec", dont_inherit=True, optimize=sys.flags.optimize)
                pending, matches = [compiled], []
                while pending:
                    item = pending.pop()
                    if item.co_name == raw.__code__.co_name and item.co_firstlineno == raw.__code__.co_firstlineno:
                        matches.append(item)
                    pending.extend(value for value in item.co_consts if isinstance(value, types.CodeType))
                # Pinned CPython3.10.19 code_richcompare checks bytecode,
                # signature/name/flags, nested typed constants and variable
                # bindings. Also bind filename/stack/line metadata explicitly;
                # avoid marshal identity/interning representation artifacts.
                # https://raw.githubusercontent.com/python/cpython/v3.10.19/Objects/codeobject.c
                _require(len(matches) == 1 and matches[0] == raw.__code__ and
                    all(getattr(matches[0], key) == getattr(raw.__code__, key)
                        for key in ("co_filename", "co_stacksize", "co_linetable")),
                    "Runtime primitive code differs from its pinned actual source; dynamic code is forbidden")

    def _primitive(self, name, stage, dense, candidate, reference, context, *, next_write_bytes=0):
        self._verify_primitive_sources()
        supplied = {**context, "case_binding_sha256": self.case_sha, "fixture_only": self.fixture_only,
            "budget": self.budget.evidence(), "next_write_bytes": next_write_bytes}
        receipt = paired._json_copy(getattr(self.primitives, name)(stage=stage, dense_trainer=dense,
            candidate_trainer=candidate, reference_model=reference, context=supplied))
        self._verify_primitive_sources()
        _require(type(receipt) is dict and receipt.get("complete") is True and receipt.get("passed") is True and
            receipt.get("stage") == stage and receipt.get("case_binding_sha256") == self.case_sha and
            receipt.get("fixture_only") is self.fixture_only and receipt.get("primitive_source_sha256") == self.primitive_sources[name]["sha256"],
            "Native primitive did not prove its exact source-bound stage/context")
        if name == "verify_phase_accounting":
            remaining = receipt.get("planned_remaining_bytes")
            _require(type(remaining) is int and remaining >= next_write_bytes and
                receipt.get("budget_snapshot_value_sha256") == _digest(self.budget.evidence()) and
                receipt.get("crossprocess_accounting_owned_by_supervisor") is True,
                "Supervisor-bound cumulative phase accounting required; no child reset")
            self.budget.check_planned_write(remaining)
        return receipt

    def _models_and_groups(self, dense, candidate, reference):
        models = (dense.model, candidate.model, reference)
        _require(self._models is None or all(a is b for a, b in zip(self._models, models)), "Adapter models/reference cannot be replaced")
        self._models = models
        maps = [paired._parameters(model, self.expected_parameters, trainable=index < 2) for index, model in enumerate(models)]
        _require(not reference.training and all(value.grad is None for value in maps[2].values()) and
            list(maps[0]) == list(maps[1]) == list(maps[2]) and
            all(str(value.device) == self.case["device"] for values in maps for value in values.values()) and
            dense.objective == candidate.objective == self.case["objective"] and dense.beta == candidate.beta == .05 and
            dense.reference_model is candidate.reference_model is reference and
            type(dense.terminal_suffix) is tuple and dense.terminal_suffix == candidate.terminal_suffix and
            _digest(list(dense.terminal_suffix)) == self.case["terminal_suffix_value_sha256"] and
            type(dense.eos_token_id) is int and dense.eos_token_id == candidate.eos_token_id and
            dense.eos_token_id in dense.terminal_suffix and (self.fixture_only or dense.eos_token_id == 151645),
            "Original FP32/reference/device/objective binding differs")
        tied_names = []
        for model, values in zip(models, maps):
            tied = model.get_input_embeddings().weight
            _require(model.get_output_embeddings().weight is tied, "Original input/output embedding Parameter must be tied")
            found = [name for name, value in values.items() if value is tied]
            _require(len(found) == 1, "Exactly one unique tied Parameter FQN is required")
            tied_names.append(found[0])
        _require(len(set(tied_names)) == 1 and self.expected_parameters > 1 and
            all(maps[0][name].shape == maps[1][name].shape == maps[2][name].shape for name in maps[0]),
            "Every original FQN/shape and nonempty tied/backbone partition must agree")
        stores = [{(str(value.device), value.untyped_storage().data_ptr()) for value in values.values()} for values in maps]
        _require(not any(stores[i] & stores[j] for i in range(3) for j in range(i)), "Student/reference storages must be disjoint")
        return maps, {"tied": (tied_names[0],), "backbone": tuple(name for name in maps[0] if name != tied_names[0])}

    def _write_report(self, stage, value, *, native_args):
        payload = _bytes({"schema": SCHEMA, "stage": stage, "case_binding_sha256": self.case_sha,
            "fixture_only": self.fixture_only, "native_admission_established": False, "details": value})
        _require(len(payload) <= MAX_REPORT_BYTES, "Detailed private report exceeds2MiB; do not truncate FQN coverage")
        self._primitive("verify_phase_accounting", stage + "_report_write", *native_args, next_write_bytes=len(payload))
        name = "report-" + str(len(self.reports)).zfill(4) + ".json"
        path = self.directory / name
        self._pending_report = {"path": name, "bytes_intended": len(payload), "bytes_charged": 0, "complete": False}
        _path(self.directory, directory=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                "Private owner0600 exclusive report required")
            offset = 0
            while offset < len(payload):
                self.budget.check_planned_write(len(payload) - offset)
                count = os.write(fd, payload[offset:offset + (1 << 20)])
                _require(type(count) is int and count > 0, "Private report write made no progress")
                self.budget.charge_written(count)
                offset += count
                self._pending_report["bytes_charged"] = offset
            os.fsync(fd)
        finally:
            os.close(fd)
        _, record = _file(path, maximum=MAX_REPORT_BYTES, private=True)
        _require(record == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}, "Closed private report bytes differ")
        self.reports.append({"path": name, **record})
        self._pending_report = None
        return self.reports[-1]

    def _aggregate(self, stage, results, files, *, tensor_count=0, **extra):
        _require(all(value.get("complete") is True and value.get("passed") is True for value in results),
            "A frozen numerical gate failed; private detailed reports are retained")
        return paired._json_copy({"complete": True, "passed": True, "stage": stage,
            "compared_tensor_count": tensor_count, "case_binding_sha256": self.case_sha,
            "fixture_only": self.fixture_only, "native_provenance_established": False,
            "crossprocess_budget_proven_here": False, "private_reports": files,
            "comparisons": [{"groups": value["groups"], "streaming": value["streaming"],
                "tensor_count": value["tensor_count"], "tolerances": value["tolerances"]} for value in results], **extra}, MAX_RECEIPT_BYTES)

    def _comparison(self, stage, candidate, dense, groups, kind, *, native_args):
        value = comparison.compare_named_tensors(candidate, dense, groups=groups, tolerances=tensor_tolerances(kind),
            expected_count=self.expected_parameters, chunk_elements=1048576)
        if kind == "identity":
            exact = all(row["candidate_sha256"] == row["dense_sha256"] for row in value["tensors"].values())
            value["exact_stored_fp32_bytes_equal"] = exact
            value["passed"] = value["passed"] and exact
        return value, self._write_report(stage, value, native_args=native_args)

    def _phase(self, dense, candidate, context):
        return {"case_binding_sha256": self.case_sha, "context": context,
            "branches": {branch: {"global_step": trainer.state.global_step, "lrs": paired._lrs(trainer),
                "parameters": {name: {"version": value._version, "shape": list(value.shape), "object_id": id(value),
                    "storage": value.untyped_storage().data_ptr()} for name, value in trainer.model.named_parameters()}}
                for branch, trainer in (("dense", dense), ("candidate", candidate))}}

    def _scalar(self, stage, left, right, objective):
        _require(type(left) is dict and type(right) is dict and paired._plain(left) and paired._plain(right), "Detached scalar packets required")
        gates = self.protocol["dense_vs_checkpointed_numeric_gates"]
        standalone = self.case["subcase"] == "standalone"
        clip = re.fullmatch(r"standalone_clip_norm_([012])" if standalone else r"clip_norm_([12])", stage)
        presentation = re.fullmatch(r"standalone_presentation_([012])" if standalone else r"presentation_([0-8])", stage)
        _require(clip is not None or presentation is not None, "Unknown scalar comparison stage")
        if clip is not None:
            _require(set(left) == set(right) == {"clip_norm"}, "Clip stage requires the actual norm-only packet")
            a, b = left["clip_norm"], right["clip_norm"]
            _require(_number(a) and _number(b) and a >= 0 and b >= 0, "Actual finite clip norms required")
            gate = gates["clip_norm"]
            checks = {"clip_norm": abs(a - b) <= gate["atol"] + gate["rtol"] * abs(a)}
        else:
            key = "group1_loss" if self.case["subcase"] == "standalone" else "scaled_loss"
            _require(set(left) == set(right) == {key, "group_size", "metrics"} and
                type(left["group_size"]) is int and type(right["group_size"]) is int and
                left["group_size"] == right["group_size"] == (1 if standalone or int(presentation.group(1)) == 8 else 8),
                "Exact actual accumulation packet required")
            checks, means = {}, {}
            for branch, packet in (("dense", left), ("candidate", right)):
                m = packet["metrics"]
                required = {"loss", "ce", "weighted_ce", "kl", "effective_weight", "active_tokens", "eos_tokens",
                    "non_eos_tokens", "supervised_tokens", "examples"} | {prefix + term for prefix in ("eos_", "non_eos_")
                        for term in ("ce", "weighted_ce", "kl", "effective_weight")}
                _require(type(m) is dict and set(m) == required and all(_number(value) for value in m.values()) and _number(packet[key]),
                    "Every finite objective/EOS/count metric is mandatory")
                _require(m["examples"] == 1 and m["active_tokens"] == m["supervised_tokens"] > 0 and m["eos_tokens"] > 0 and
                    m["non_eos_tokens"] >= 0 and m["eos_tokens"] + m["non_eos_tokens"] == m["active_tokens"] and
                    all(float(m[name]).is_integer() for name in ("examples", "active_tokens", "eos_tokens", "non_eos_tokens", "supervised_tokens")),
                    "Exact active/EOS/nonEOS/example counts required")
                _require(abs(m["loss"] - (m["weighted_ce"] + .05 * m["kl"])) <= 1e-6 and
                    abs(packet[key] * packet["group_size"] - m["loss"]) <= 1e-6, "Loss formula or once-only accumulation scaling differs")
                if objective in ("sft", "ce_kl"):
                    _require(m["effective_weight"] == m["eos_effective_weight"] == 1. and
                        m["non_eos_effective_weight"] == (1. if m["non_eos_tokens"] else 0.), "Unweighted objective weights must be exactly1")
                if objective == "sft":
                    _require(m["kl"] == m["eos_kl"] == m["non_eos_kl"] == 0., "SFT KL must be exactly zero")
                means[branch] = m
            for name in ("examples", "active_tokens", "eos_tokens", "non_eos_tokens", "supervised_tokens"):
                checks[name] = means["dense"][name] == means["candidate"][name]
            for name in ("total_loss", "ordinary_ce_means", "weighted_ce_means", "forward_kl_means", "effective_weight_means"):
                gate = gates[name]
                for field in gate.get("metrics", ["loss"]):
                    a, b = means["dense"][field], means["candidate"][field]
                    checks[field] = abs(a - b) <= gate["atol"] + gate["rtol"] * abs(a)
        return {"complete": True, "passed": all(checks.values()), "checks": checks, "dense": left, "candidate": right}

    def _dispatch(self, kind, *, stage, dense_trainer, candidate_trainer, reference_model, context, **extra):
        try:
            _require(not self._failed and type(stage) is str and (kind, stage) not in self._entered, "Failed/reused hook stage cannot continue")
            standalone = self.case["subcase"] == "standalone"
            known = {
                "invariants": ({"standalone_prepared", "standalone_before_clip_0", "standalone_before_clip_1", "standalone_before_clip_2", "standalone_final"}
                    if standalone else {"prepared", "before_update_1", "before_update_2", "final"}),
                "identity": ({"standalone_initial_original", "standalone_unchanged_original_0", "standalone_unchanged_original_1", "standalone_unchanged_original_2"}
                    if standalone else {"initial_original", "post_lr0_original"}),
                "shadow": set() if standalone else {"post_lr0_pre_update2"},
                "deltas": set() if standalone else {"actual_update2_deltas"}}
            if kind in known:
                _require(stage in known[kind], "Unknown fixed hook stage")
            elif kind == "scalars":
                _require(re.fullmatch(r"standalone_(presentation|clip_norm)_([012])" if standalone else
                    r"(presentation_[0-8]|clip_norm_[12])", stage) is not None, "Unknown scalar stage")
            elif kind == "tensors":
                _require(re.fullmatch(r"standalone_(raw_gradients|clipped_gradients)_([012])" if standalone else
                    r"(raw_gradients|clipped_gradients|adam_first_moments|adam_second_moments|updated_weights)_([12])", stage) is not None,
                    "Unknown tensor stage")
            else:
                raise ValueError("Unknown hook operation")
            self._entered.add((kind, stage))
            _require(type(context) is dict and context.get("expected_parameters") == self.expected_parameters and
                context.get("objective") == self.case["objective"], "Controller/native case context differs")
            maps, groups = self._models_and_groups(dense_trainer, candidate_trainer, reference_model)
            args = (dense_trainer, candidate_trainer, reference_model, context)
            native = self._primitive("verify_native_context", stage, *args)
            accounting = self._primitive("verify_phase_accounting", stage, *args, next_write_bytes=MAX_REPORT_BYTES)
            results, files = [], []
            if kind == "invariants":
                allowed = {"prepared", "before_update_1", "before_update_2", "final"} if self.case["subcase"] == "ga9" else {
                    "standalone_prepared", "standalone_before_clip_0", "standalone_before_clip_1", "standalone_before_clip_2", "standalone_final"}
                _require(stage in allowed, "Unknown invariant phase")
                files.append(self._write_report(stage, {"native_context": native, "phase_accounting": accounting}, native_args=args))
                receipt = self._aggregate(stage, [], files, verified_parameter_count=self.expected_parameters)
            elif kind == "identity":
                allowed = {"initial_original", "post_lr0_original"} if self.case["subcase"] == "ga9" else {
                    "standalone_initial_original", "standalone_unchanged_original_0", "standalone_unchanged_original_1", "standalone_unchanged_original_2"}
                _require(stage in allowed, "Unknown original-identity phase")
                for branch, values in zip(("dense", "candidate"), maps[:2]):
                    value, pin = self._comparison(stage + "_" + branch, values, maps[2], groups, "identity", native_args=args)
                    results.append(value); files.append(pin)
                if stage in {"initial_original", "post_lr0_original", "standalone_initial_original"}:
                    actual = self._primitive("verify_original_numerics", stage, *args)
                    limits = self.protocol["unchanged_existing_gates"]["fresh_original_student_vs_original_reference_zero_update"]
                    _require(type(actual.get("measurements")) is dict and set(actual["measurements"]) == set(limits) and
                        actual.get("student_branches") == ["dense", "candidate"] and type(actual.get("witness_count")) is int and
                        actual["witness_count"] == 3 and actual.get("numeric_protocol_file_sha256") == PROTOCOL_SHA256 and
                        all(_number(actual["measurements"][name]) and 0 <= actual["measurements"][name] <= limit for name, limit in limits.items()),
                        "Actual original-zero logit/CE/KL controls did not meet unchanged bounds")
                    files.append(self._write_report(stage + "_original_numerics", actual, native_args=args))
                receipt = self._aggregate(stage, results, files, tensor_count=self.expected_parameters)
                self._last_identity = {"stage": stage, "phase": self._phase(dense_trainer, candidate_trainer, context), "reports": files}
            elif kind == "scalars":
                value = self._scalar(stage, extra["dense"], extra["candidate"], self.case["objective"])
                files.append(self._write_report(stage, value, native_args=args))
                _require(value["passed"], "Frozen scalar comparison failed")
                receipt = self._aggregate(stage, [], files, scalar_checks=value["checks"])
            elif kind == "tensors":
                match = re.fullmatch(r"(raw_gradients|clipped_gradients|adam_first_moments|adam_second_moments|updated_weights)_([12])", stage)
                standalone = re.fullmatch(r"standalone_(raw_gradients|clipped_gradients)_([012])", stage)
                _require((match is not None and self.case["subcase"] == "ga9") or (standalone is not None and self.case["subcase"] == "standalone"),
                    "Unknown tensor comparison stage")
                label = (match or standalone).group(1)
                if label in ("raw_gradients", "clipped_gradients"):
                    values = [paired._gradients(trainer, parameters) for trainer, parameters in zip((dense_trainer, candidate_trainer), maps[:2])]
                    tolerance = "gradients"
                elif label.startswith("adam_"):
                    step = int(match.group(2))
                    for trainer, parameters in zip((dense_trainer, candidate_trainer), maps[:2]): paired._state(trainer, parameters, step)
                    key = "exp_avg" if label == "adam_first_moments" else "exp_avg_sq"
                    values = [{name: paired._optimizer(trainer.optimizer).state[parameter][key] for name, parameter in parameters.items()}
                        for trainer, parameters in zip((dense_trainer, candidate_trainer), maps[:2])]
                    tolerance = label
                else:
                    values, tolerance = maps[:2], "updated_weights"
                value, pin = self._comparison(stage, values[1], values[0], groups, tolerance, native_args=args)
                receipt = self._aggregate(stage, [value], [pin], tensor_count=self.expected_parameters)
            elif kind == "shadow":
                _require(stage == "post_lr0_pre_update2" and self.case["subcase"] == "ga9" and self._shadow is None and
                    self._last_identity is not None and self._last_identity["stage"] == "post_lr0_original" and
                    self._last_identity["phase"] == self._phase(dense_trainer, candidate_trainer, context) and
                    context.get("optimizer_updates") == {"dense": 1, "candidate": 1}, "Common shadow requires the exact independently verified post-LR0 phase")
                for trainer, parameters in zip((dense_trainer, candidate_trainer), maps[:2]):
                    paired._state(trainer, parameters, 1); paired._expect_lr(trainer, 0)
                phase = self._last_identity["phase"]
                # Ordinary version guards alone cannot detect every .data
                # mutation; recompare both complete stored byte inventories
                # immediately before writing the common initial shadow.
                identity_results = []
                for branch, values in zip(("dense", "candidate"), maps[:2]):
                    value, pin = self._comparison(stage + "_recheck_" + branch, values, maps[2], groups, "identity", native_args=args)
                    identity_results.append(value); files.append(pin)
                _require(all(value["passed"] for value in identity_results), "Pre-shadow original bytes changed")
                self._primitive("verify_phase_accounting", stage, *args,
                    next_write_bytes=sum(value.numel() * 4 for value in maps[0].values()) + comparison.MAX_MANIFEST_BYTES + MAX_REPORT_BYTES)
                self._shadow_directory = self.budget.directory / (self.directory.name + "-initial-shadow")
                result = comparison.write_initial_shadow(maps[0], self._shadow_directory,
                    budget=self.budget, expected_count=self.expected_parameters, chunk_elements=1048576)
                _require(phase == self._phase(dense_trainer, candidate_trainer, context), "Initial shadow phase changed during write")
                reader = comparison.open_initial_shadow(result["directory"], expected_manifest_sha256=result["manifest_sha256"],
                    expected_count=self.expected_parameters, chunk_elements=1048576)
                self._shadow = _BoundShadow(reader, self.case_sha, _digest(phase), result["manifest_sha256"])
                files.append(self._write_report(stage, {"phase": phase, "prior_identity_reports": self._last_identity["reports"],
                    "shadow_manifest_file_sha256": result["manifest_sha256"], "shadow_manifest_bytes": result["manifest_bytes"],
                    "one_common_shadow": True, "no_adam_formula": True}, native_args=args))
                receipt = paired.ShadowArtifact(self._shadow, self._aggregate(stage, [], files, tensor_count=self.expected_parameters,
                    shadow_manifest_file_sha256=result["manifest_sha256"], phase_binding_sha256=self._shadow.phase_binding_sha256))
            elif kind == "deltas":
                _require(stage == "actual_update2_deltas" and self.case["subcase"] == "ga9" and self._shadow is not None and
                    extra.get("initial_shadow") is self._shadow and self._shadow.case_binding_sha256 == self.case_sha and
                    context.get("optimizer_updates") == {"dense": 2, "candidate": 2}, "Use this exact same case's post-LR0 common shadow object")
                for trainer, parameters in zip((dense_trainer, candidate_trainer), maps[:2]): paired._state(trainer, parameters, 2)
                value = comparison.compare_stored_deltas(maps[1], maps[0], self._shadow.reader, groups=groups,
                    tolerances=tensor_tolerances("actual_update2_deltas"), expected_count=self.expected_parameters, chunk_elements=1048576)
                files.append(self._write_report(stage, value, native_args=args))
                positive = all(value["groups"][name]["positive_norms"] for name in ("tied", "backbone"))
                _require(positive, "Both actual stored delta groups must be positive in both branches")
                receipt = self._aggregate(stage, [value], files, tensor_count=self.expected_parameters,
                    positive_update_groups={branch: {"tied": True, "backbone": True} for branch in ("dense", "candidate")})
            else:
                raise ValueError("Unknown hook operation")
            self._primitive("verify_native_context", stage + "_after", *args)
            self._primitive("verify_phase_accounting", stage + "_after", *args)
            return receipt
        except BaseException as error:
            self._failed = True
            raise PairHookFailure({"schema": SCHEMA, "complete": False, "case_binding_sha256": self.case_sha,
                "fixture_only": self.fixture_only, "stage": stage, "operation": kind, "error_type": type(error).__name__,
                "private_reports": list(self.reports), "budget": self.budget.evidence(),
                "partial_private_report": self._pending_report,
                "private_shadow_directory_name": None if self._shadow_directory is None else self._shadow_directory.name,
                "native_admission_established": False, "caller_must_preserve_partial_reports_and_abort": True}) from error

    def verify_invariants(self, **kwargs): return self._dispatch("invariants", **kwargs)
    def verify_identity(self, **kwargs): return self._dispatch("identity", **kwargs)
    def compare_scalars(self, **kwargs): return self._dispatch("scalars", **kwargs)
    def compare_tensors(self, **kwargs): return self._dispatch("tensors", **kwargs)
    def create_shadow(self, **kwargs): return self._dispatch("shadow", **kwargs)
    def compare_deltas(self, **kwargs): return self._dispatch("deltas", **kwargs)

    def paired_hooks(self):
        _require(self.case["subcase"] == "ga9", "Wrong controller/subcase")
        return paired.PairedStepHooks(self.verify_invariants, self.verify_identity, self.compare_scalars,
            self.compare_tensors, self.create_shadow, self.compare_deltas)

    def standalone_hooks(self):
        _require(self.case["subcase"] == "standalone", "Wrong controller/subcase")
        return StandaloneStepHooks(self.verify_invariants, self.verify_identity, self.compare_scalars, self.compare_tensors)
