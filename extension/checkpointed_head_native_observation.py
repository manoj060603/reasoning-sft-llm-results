"""Unadmitted bounded hooks for a future original-Qwen native worker.

No Torch import, model/data loading, tensor computation, filesystem, CLI, job or
admission occurs at module load. The caller supplies the guarded actual Torch
module and the frozen original-parts validator. That validator must have the
``_qwen_parts(model, *, name, trainable)`` contract and return the ORIGINAL
backbone, bias-free tied head and embedding Parameter. Model/checkpoint bytes,
434 provenance, input/label identity, allocator, numerical finiteness, memory,
optimizer/update checks and actual completed backward remain caller obligations.

Install an observer around the entire forward AND backward lifetime. A NEW
training_step subclass may enclose super().training_step in phase('backward'),
while a NEW compute_loss subclass encloses super().compute_loss in a nested
phase('forward'). Returning from compute_loss restores the outer backward phase
before checkpoint recomputation. No model forward, global, library or existing
Trainer method is replaced by this helper. Inactive observers ignore calls, so
two installed branch observers may share a frozen reference without attributing
the other branch's reference calls to themselves. The caller must not activate
both branch phases for the same reference invocation.

Pre/post hooks attach to original child backbones, every original decoder layer
and original heads. Only counters, bounded shape/dtype/context sets and integer
presentation indices survive a hook. No output, detached output, tensor value,
storage address or graph is retained. Non-reentrant checkpoint early-stop may
interrupt a module before a normal output exists: entry/completed/interrupted
counts are distinct and backward post-hook count equality is NOT required.
Actual backward success is proved by the caller, never inferred from an entry.
Fakes can test this API but cannot establish a native observation or admission.
"""
from __future__ import annotations

from contextlib import contextmanager


SCHEMA = "extension-checkpointed-head-native-observation-v1"
MAX_CHUNK_TOKENS = 128
MAX_DENSE_TOKENS = 512
MAX_SEQUENCE_TOKENS = 32768
MAX_LAYERS = 64
MAX_DISTINCT_SHAPES = 128
MAX_PRESENTATIONS = 64
MAX_NESTING = 8
PHASES = frozenset({"forward", "backward"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(tensor):
    shape = tuple(int(size) for size in tensor.shape)
    _require(shape and all(size > 0 for size in shape), "Observed tensor shape must be nonempty and positive")
    return shape


class _Bucket:
    def __init__(self):
        self.entries = self.completed = self.interrupted = 0
        self.input_shapes, self.output_shapes = {}, {}
        self.input_dtypes, self.output_dtypes = set(), set()
        self.grad_enabled, self.autocast = set(), set()
        self.presentations = set()

    def shape(self, which, value):
        shapes = getattr(self, which + "_shapes")
        _require(value in shapes or len(shapes) < MAX_DISTINCT_SHAPES,
                 "Observation exceeded the fixed distinct-shape bound")
        shapes[value] = shapes.get(value, 0) + 1

    def snapshot(self):
        return {"entries": self.entries, "completed_outputs": self.completed,
            "interrupted_without_output": self.interrupted,
            "input_shapes": [{"shape": list(shape), "calls": count} for shape, count in sorted(self.input_shapes.items())],
            "output_shapes": [{"shape": list(shape), "calls": count} for shape, count in sorted(self.output_shapes.items())],
            "input_dtypes": sorted(self.input_dtypes), "output_dtypes": sorted(self.output_dtypes),
            "grad_enabled_states": sorted(self.grad_enabled),
            "cuda_autocast": [{"enabled": enabled, "dtype": dtype} for enabled, dtype in sorted(self.autocast)],
            "presentation_indices": sorted(self.presentations)}


class NativeObservation:
    """Observe original modules only during caller-declared branch phases.

    ``parts_validator`` is required, not guessed/imported or monkeypatched.
    Pass the unchanged checkpointed_head_trainer._qwen_parts in the future
    native worker AFTER its allocation/source/runtime guards. Injection is a
    fixture seam and is not proof of original Qwen or real CUDA provenance.

    ``branch`` is 'dense' or 'candidate'. Dense head outputs are bounded by512
    sequence positions; candidate heads must project2-D active chunks<=128.
    Models remain strongly referenced so their hook bindings can be checked;
    no activation/output tensor is kept. Enter exactly once and remove all
    handles on exit, including exceptional exit. Validation never runs implicitly
    on an exceptional exit and therefore does not mask the original failure.
    """
    def __init__(self, torch_module, *, student, reference=None, branch, parts_validator):
        _require(branch in ("dense", "candidate") and callable(parts_validator),
                 "Explicit branch and frozen original-parts validator required")
        self.torch, self.branch, self.parts_validator = torch_module, branch, parts_validator
        self._bindings, self._modules, self._buckets, self._pending = {}, {}, {}, {}
        self._handles, self._phases = [], []
        self._entered = self._active = self._closed = self._validated = False
        self._ignored_calls = 0
        self._add_model("student", student, True)
        if reference is not None:
            _require(reference is not student, "Student and reference must be separate original objects")
            self._add_model("reference", reference, False)
            _require(self._bindings["student"]["device"] == self._bindings["reference"]["device"],
                     "Student and reference must use the same original assigned CUDA device")
            student_storages = self._storages(student)
            _require(not student_storages & self._storages(reference),
                     "Student and reference Parameter storage must be disjoint")

    def _storages(self, model):
        return {(str(parameter.device), parameter.untyped_storage().data_ptr()) for parameter in model.parameters()}

    def _add_model(self, role, model, trainable):
        parts = self.parts_validator(model, name=role.title(), trainable=trainable)
        _require(type(parts) in (tuple, list) and len(parts) == 3, "Original-parts validator must return exactly three original objects")
        backbone, head, embedding = parts
        layers = tuple(backbone.layers)
        count = getattr(model.config, "num_hidden_layers", None)
        _require(type(count) is int and 1 <= count <= MAX_LAYERS and len(layers) == count and
                 len({id(layer) for layer in layers}) == count,
                 "All original decoder layers must have one bounded exact identity")
        _require(backbone is model.model and head is model.lm_head and
                 head is model.get_output_embeddings() and embedding is model.get_input_embeddings().weight and
                 head.weight is embedding and embedding.dtype == self.torch.float32 and
                 embedding.requires_grad is trainable and embedding.device.type == "cuda",
                 "Only the original same-device FP32 tied embedding/head boundary is supported")
        parameters = tuple(model.parameters())
        _require(parameters and all(parameter.dtype == self.torch.float32 and
                 parameter.requires_grad is trainable and parameter.device == embedding.device for parameter in parameters),
                 "All observed original Parameters must retain FP32 storage and the declared gradient policy")
        if not trainable:
            _require(not model.training and all(parameter.grad is None for parameter in parameters),
                     "Reference must be frozen eval mode with no gradients")
        vocabulary, hidden = getattr(head, "out_features", None), getattr(head, "in_features", None)
        _require(type(vocabulary) is int and vocabulary > 0 and type(hidden) is int and hidden > 0,
                 "Original head dimensions must be explicit positive integers")
        self._bindings[role] = {"model": model, "backbone": backbone, "head": head,
            "embedding": embedding, "layers": layers, "trainable": trainable,
            "vocabulary": vocabulary, "hidden": hidden, "device": embedding.device}
        for component, module in [("backbone", backbone), ("head", head)] + [
                ("decoder_" + str(index).zfill(2), layer) for index, layer in enumerate(layers)]:
            key = (role, component)
            _require(all(module is not old for old in self._modules.values()),
                     "Original observed child modules must not be shared or duplicated")
            self._modules[key], self._pending[key] = module, []

    def _identity(self, role):
        binding = self._bindings[role]
        model = binding["model"]
        _require(model.model is binding["backbone"] and model.lm_head is binding["head"] and
                 model.get_output_embeddings() is binding["head"] and
                 model.get_input_embeddings().weight is binding["embedding"] and
                 binding["head"].weight is binding["embedding"] and
                 tuple(model.model.layers) == binding["layers"],
                 "An original backbone/layer/tied-head identity changed while hooks were installed")
        _require(binding["embedding"].dtype == self.torch.float32 and
                 binding["embedding"].device == binding["device"] and
                 binding["embedding"].requires_grad is binding["trainable"],
                 "Original tied Parameter storage dtype/device/gradient policy changed")
        if role == "reference":
            _require(not model.training and all(not parameter.requires_grad and parameter.grad is None
                     for parameter in model.parameters()), "Reference gradient/eval policy changed")

    def _context(self, role):
        self._identity(role)
        _require(not self.torch.is_inference_mode_enabled(),
                 "Native training observations require normal tensors/no_grad, never inference_mode")
        enabled = bool(self.torch.is_autocast_enabled("cuda"))
        dtype = self.torch.get_autocast_dtype("cuda")
        _require(enabled and dtype == self.torch.bfloat16, "Every active original-module call requires actual CUDA BF16 autocast")
        grad = bool(self.torch.is_grad_enabled())
        if role == "reference":
            _require(not grad, "Reference backbone/layer/head must execute under no_grad")
        elif self._phases[-1][0] == "backward":
            _require(grad, "Candidate/dense student recomputation must execute with grad enabled")
        return grad, (enabled, str(dtype))

    def _input(self, role, component, args, kwargs):
        torch = self.torch
        if component == "backbone":
            tensor = kwargs.get("input_ids", args[0] if args else None)
            _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.dtype == torch.int64,
                     "Original backbone observation requires native input_ids, not an inferred prompt")
            shape = _shape(tensor)
            _require(shape[0] == 1 and shape[1] <= MAX_SEQUENCE_TOKENS, "Native observation supports complete microbatch1 contexts<=32768")
        else:
            tensor = kwargs.get("hidden_states", args[0] if args else None)
            _require(isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.float32, torch.bfloat16),
                     "Original layer/head must receive actual FP32/BF16 hidden states")
            shape = _shape(tensor)
            expected_rank = 2 if component == "head" and self.branch == "candidate" else 3
            _require(tensor.ndim == expected_rank and shape[-1] == self._bindings[role]["hidden"],
                     "Original layer/head hidden shape differs from the declared branch")
            if component == "head":
                _require((self.branch == "candidate" and 0 < shape[0] <= MAX_CHUNK_TOKENS) or
                         (self.branch == "dense" and shape[0] == 1 and shape[1] <= MAX_DENSE_TOKENS),
                         "Candidate projection must stay<=128 rows; dense oracle is short-only<=512")
            else:
                _require(shape[0] == 1 and shape[1] <= MAX_SEQUENCE_TOKENS, "Original decoder shape must remain microbatch1<=32768")
        _require(tensor.device == self._bindings[role]["device"] and tensor.device.type == "cuda",
                 "Observed input must be on the original assigned CUDA device")
        return tensor, shape

    def _pre(self, key):
        def hook(module, args, kwargs):
            if not self._phases:
                self._ignored_calls += 1
                return None
            role, component = key
            _require(module is self._modules[key], "Original observed module identity changed")
            phase, presentation, _ = self._phases[-1]
            grad, autocast = self._context(role)
            tensor, shape = self._input(role, component, args, kwargs)
            bucket = self._buckets.setdefault((phase, role, component), _Bucket())
            bucket.entries += 1
            bucket.shape("input", shape)
            bucket.input_dtypes.add(str(tensor.dtype))
            bucket.grad_enabled.add(grad)
            bucket.autocast.add(autocast)
            if presentation is not None:
                _require(presentation in bucket.presentations or len(bucket.presentations) < MAX_PRESENTATIONS,
                         "Observed presentation set exceeded its fixed bound")
                bucket.presentations.add(presentation)
            _require(len(self._pending[key]) < MAX_NESTING, "Observed module recursion exceeded its fixed bound")
            self._pending[key].append((phase, presentation, shape))
            return None
        return hook

    def _post(self, key):
        def hook(module, args, kwargs, output):
            if not self._phases:
                return None
            role, component = key
            _require(module is self._modules[key] and self._pending[key], "Observed module output has no matching original entry")
            phase, presentation, input_shape = self._pending[key].pop()
            _require((phase, presentation) == self._phases[-1][:2], "Observation phase changed inside one module invocation")
            bucket = self._buckets[(phase, role, component)]
            self._context(role)
            if output is None:
                bucket.interrupted += 1
                return None
            if component == "backbone":
                value = output.get("last_hidden_state") if isinstance(output, dict) else getattr(output, "last_hidden_state", None)
                if value is None and isinstance(output, (tuple, list)) and output:
                    value = output[0]
            else:
                value = output[0] if isinstance(output, (tuple, list)) and output else output
            torch = self.torch
            _require(isinstance(value, torch.Tensor) and value.device == self._bindings[role]["device"],
                     "Original observed output must remain an actual same-device tensor")
            shape = _shape(value)
            if component == "head":
                expected = input_shape[:-1] + (self._bindings[role]["vocabulary"],)
                _require(shape == expected and value.dtype == torch.bfloat16,
                         "Every completed original head projection must have actual BF16 output and bounded matching shape")
            else:
                expected = input_shape + (self._bindings[role]["hidden"],) if component == "backbone" else input_shape
                _require(shape == expected and value.dtype in (torch.float32, torch.bfloat16),
                         "Original backbone/decoder output shape or FP32/BF16 dtype changed")
            if role == "reference":
                _require(not value.requires_grad and value.grad_fn is None,
                         "Reference output must actually remain detached under no_grad")
            bucket.completed += 1
            bucket.shape("output", shape)
            bucket.output_dtypes.add(str(value.dtype))
            return None
        return hook

    def __enter__(self):
        _require(not self._entered and not self._closed, "Observation lifecycle cannot be reentered or reused")
        for role, binding in self._bindings.items():
            self.parts_validator(binding["model"], name=role.title(), trainable=binding["trainable"])
            self._identity(role)
        self._entered = self._active = True
        try:
            for key, module in self._modules.items():
                self._handles.append(module.register_forward_pre_hook(self._pre(key), with_kwargs=True))
                self._handles.append(module.register_forward_hook(self._post(key), with_kwargs=True, always_call=True))
        except BaseException:
            for handle in reversed(self._handles):
                try:
                    handle.remove()
                except BaseException:
                    pass
            self._handles.clear()
            self._active, self._closed = False, True
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cleanup_errors = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
        self._handles.clear()
        self._active, self._closed = False, True
        if exc_type is None:
            _require(not cleanup_errors, "An original-module observation hook could not be removed")
            _require(not self._phases and not any(self._pending.values()), "Observation exited with an unfinished phase/module call")
            for role, binding in self._bindings.items():
                self.parts_validator(binding["model"], name=role.title(), trainable=binding["trainable"])
                self._identity(role)
        return False

    @contextmanager
    def phase(self, name, *, presentation_index=None):
        _require(self._active and name in PHASES and len(self._phases) < MAX_NESTING,
                 "Hooks must remain installed during an explicit bounded forward/backward phase")
        _require(presentation_index is None or type(presentation_index) is int and 0 <= presentation_index < MAX_PRESENTATIONS,
                 "Observation presentation index must be an explicit bounded plain integer")
        marker = (name, presentation_index, object())
        for role, binding in self._bindings.items():
            self.parts_validator(binding["model"], name=role.title(), trainable=binding["trainable"])
            self._identity(role)
        self._phases.append(marker)
        self._validated = False
        try:
            yield self
        finally:
            _require(self._phases and self._phases[-1] is marker, "Observation phase nesting must unwind in order")
            self._phases.pop()

    def snapshot(self):
        return {"schema": SCHEMA, "branch": self.branch, "hooks_active": self._active,
            "lifecycle_closed": self._closed, "requirements_validated": self._validated,
            "expected_decoder_layers": {role: len(binding["layers"]) for role, binding in self._bindings.items()},
            "observations": {phase: {role: {component: bucket.snapshot() for (p, r, component), bucket in
                sorted(self._buckets.items()) if p == phase and r == role} for role in sorted(self._bindings)}
                for phase in sorted(PHASES)},
            "inactive_calls_ignored": self._ignored_calls,
            "pending_module_calls": sum(len(values) for values in self._pending.values()),
            "tensor_outputs_values_graphs_or_storage_addresses_retained": False,
            "numeric_finiteness_or_original_checkpoint_provenance_established": False,
            "actual_backward_completion_or_native_admission_established": False,
            "qualification": "Explicit-phase original-module hook metadata only; backward entry is not completed backward or numerical/memory proof. Non-reentrant early-stop may omit normal backward outputs. Inactive shared-reference calls are not attributed."}

    def validate(self, *, require_backward_recompute=True, require_all_layers=True, require_reference=None):
        _require(self._entered and not self._phases and not any(self._pending.values()),
                 "Validate only after complete caller phases/backward have unwound")
        _require(type(require_backward_recompute) is bool and type(require_all_layers) is bool and
                 (require_reference is None or type(require_reference) is bool), "Explicit observation validation policy required")
        reference_needed = "reference" in self._bindings if require_reference is None else require_reference
        _require(not reference_needed or "reference" in self._bindings, "Required reference observation has no original reference")
        roles = ["student"] + (["reference"] if reference_needed else [])
        for role in roles:
            components = ["backbone", "head"]
            layers = ["decoder_" + str(index).zfill(2) for index in range(len(self._bindings[role]["layers"]))]
            if require_all_layers:
                components += layers
            for component in components:
                bucket = self._buckets.get(("forward", role, component))
                _require(bucket is not None and bucket.entries > 0 and bucket.completed > 0,
                         "Missing completed explicit-forward observation for original " + role + "/" + component)
        if self.branch == "candidate" and require_backward_recompute:
            head = self._buckets.get(("backward", "student", "head"))
            layer_entries = sum(bucket.entries for (phase, role, component), bucket in self._buckets.items()
                if phase == "backward" and role == "student" and component.startswith("decoder_"))
            _require(head is not None and head.entries > 0 and layer_entries > 0,
                     "Actual explicit-backward candidate head AND original decoder-layer entries are required")
        self._validated = True
        result = self.snapshot()
        result["validation_policy"] = {"require_backward_recompute": require_backward_recompute,
            "require_all_forward_layers": require_all_layers, "require_reference": reference_needed,
            "all_backward_layers_or_equal_recompute_counts_required": False,
            "backward_completion_must_be_verified_by_caller": True}
        return result
