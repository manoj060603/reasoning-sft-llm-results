"""24 prospective, UNEXECUTED fake-only observation fixtures.

This file imports no Torch, Transformers, tokenizer, model or private inputs.
Fake CUDA devices, AMP flags, tensors, module calls, storage pointers and
checkpoint interruption are explicit metadata simulations. They test hook/API
guards only and cannot prove original434 identity, CUDA/BF16 execution,
recomputation, numerical equivalence, memory or any native admission. A future
approved CPU allocation owns execution; no test has been run by its author.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
from types import SimpleNamespace
import unittest

from extension import checkpointed_head_native_observation as module


@dataclass(frozen=True)
class FixtureDevice:
    type: str = "cuda"
    index: int = 0

    def __str__(self):
        return self.type + (":" + str(self.index) if self.type == "cuda" else "")


class FixtureTensor:
    """Shape/context metadata only: there are no numeric tensor elements."""
    def __init__(self, shape, *, dtype="torch.float32", device=None, requires_grad=False):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = FixtureDevice() if device is None else device
        self.requires_grad = requires_grad
        self.grad_fn = "explicit_fixture_graph" if requires_grad else None

    @property
    def ndim(self):
        return len(self.shape)


class FixtureParameter(FixtureTensor):
    _pointers = itertools.count(1000)

    def __init__(self, shape, *, trainable, device=None, pointer=None):
        super().__init__(shape, device=device, requires_grad=trainable)
        self.grad = None
        self.pointer = next(self._pointers) if pointer is None else pointer

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.pointer)


class FixtureTorch:
    """Explicit state flags; no CUDA library/context is imported or initialized."""
    Tensor = FixtureTensor
    float32, bfloat16, int64 = "torch.float32", "torch.bfloat16", "torch.int64"

    def __init__(self):
        self.amp, self.amp_dtype, self.grad, self.inference = True, self.bfloat16, True, False

    def is_autocast_enabled(self, device):
        if device != "cuda":
            raise AssertionError("Fixture expects the explicit CUDA autocast API")
        return self.amp

    def get_autocast_dtype(self, device):
        if device != "cuda":
            raise AssertionError("Fixture expects the explicit CUDA autocast API")
        return self.amp_dtype

    def is_grad_enabled(self):
        return self.grad

    def is_inference_mode_enabled(self):
        return self.inference

    @contextmanager
    def state(self, **changes):
        before = {key: getattr(self, key) for key in changes}
        for key, value in changes.items():
            setattr(self, key, value)
        try:
            yield self
        finally:
            for key, value in before.items():
                setattr(self, key, value)


class FixtureHandle:
    def __init__(self, owner, key, collection):
        self.owner, self.key, self.collection = owner, key, collection
        self.remove_calls = 0

    def remove(self):
        self.remove_calls += 1
        if self.owner.fail_remove:
            raise RuntimeError("fixture removal failure")
        self.collection.pop(self.key, None)


class FixtureModule:
    """Public-hook call seam, including always_call on an interrupted call."""
    def __init__(self):
        self.pre, self.post, self.handles = {}, {}, []
        self.ids = itertools.count()
        self.fail_register_pre = self.fail_register_post = self.fail_remove = False
        self.post_errors_during_original_exception = []
        self.failure = None

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        if self.fail_register_pre:
            raise RuntimeError("fixture registration failure")
        if with_kwargs is not True:
            raise AssertionError("Original keyword inputs must remain observable")
        key = next(self.ids)
        self.pre[key] = hook
        handle = FixtureHandle(self, key, self.pre)
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, hook, *, with_kwargs, always_call):
        if self.fail_register_post:
            raise RuntimeError("fixture registration failure")
        if with_kwargs is not True or always_call is not True:
            raise AssertionError("Interrupted original calls must retain public always_call hooks")
        key = next(self.ids)
        self.post[key] = (hook, always_call)
        handle = FixtureHandle(self, key, self.post)
        self.handles.append(handle)
        return handle

    def __call__(self, *args, **kwargs):
        try:
            for hook in list(self.pre.values()):
                if hook(self, args, kwargs) is not None:
                    raise AssertionError("Observation pre-hooks must not replace original inputs")
            if self.failure is not None:
                raise self.failure
            output = self.forward(*args, **kwargs)
        except BaseException:
            # Match the important public-hook property: hook-cleanup errors do
            # not replace the already raised original call exception.
            for hook, always in list(self.post.values()):
                if always:
                    try:
                        hook(self, args, kwargs, None)
                    except BaseException as error:
                        self.post_errors_during_original_exception.append(type(error).__name__)
            raise
        for hook, _ in list(self.post.values()):
            if hook(self, args, kwargs, output) is not None:
                raise AssertionError("Observation post-hooks must not replace original outputs")
        return output


class FixtureLayer(FixtureModule):
    def __init__(self, torch, trainable, device):
        super().__init__()
        self.torch, self.trainable = torch, trainable
        self.parameter = FixtureParameter((1,), trainable=trainable, device=device)

    def forward(self, hidden_states):
        return FixtureTensor(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device,
            requires_grad=self.trainable and self.torch.grad)


class FixtureHead(FixtureModule):
    def __init__(self, torch, weight, hidden, vocabulary):
        super().__init__()
        self.torch, self.weight = torch, weight
        self.in_features, self.out_features = hidden, vocabulary
        self.output_dtype_override = self.output_shape_override = None

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1] + (self.out_features,)
        dtype = self.torch.bfloat16 if self.torch.amp else self.torch.float32
        return FixtureTensor(shape if self.output_shape_override is None else self.output_shape_override,
            dtype=dtype if self.output_dtype_override is None else self.output_dtype_override,
            device=hidden_states.device, requires_grad=self.weight.requires_grad and self.torch.grad)


class FixtureBackbone(FixtureModule):
    def __init__(self, torch, hidden, layers, trainable, device):
        super().__init__()
        self.torch, self.hidden, self.trainable = torch, hidden, trainable
        self.layers = [FixtureLayer(torch, trainable, device) for _ in range(layers)]
        self.skip_layers = set()

    def forward(self, input_ids):
        hidden = FixtureTensor(input_ids.shape + (self.hidden,), device=input_ids.device,
            requires_grad=self.trainable and self.torch.grad)
        for index, layer in enumerate(self.layers):
            if index not in self.skip_layers:
                hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class FixtureModel:
    def __init__(self, torch, *, trainable=True, layers=2, device=None):
        device = FixtureDevice() if device is None else device
        self.fixture_only = True
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.training = trainable
        self.embedding = SimpleNamespace(weight=FixtureParameter((11, 4), trainable=trainable, device=device))
        self.model = FixtureBackbone(torch, 4, layers, trainable, device)
        self.lm_head = FixtureHead(torch, self.embedding.weight, 4, 11)

    def parameters(self):
        yield self.embedding.weight
        yield from (layer.parameter for layer in self.model.layers)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids=input_ids).last_hidden_state)


def fixture_parts(model, *, name, trainable):
    """Explicit injection, never the original-Qwen/434 provenance validator."""
    if not isinstance(model, FixtureModel) or model.fixture_only is not True:
        raise ValueError("This validator accepts only explicit synthetic fixtures")
    return model.model, model.lm_head, model.embedding.weight


def make(*, branch="candidate", reference=False, layers=2, torch=None):
    torch = FixtureTorch() if torch is None else torch
    student = FixtureModel(torch, layers=layers)
    ref = FixtureModel(torch, trainable=False, layers=layers) if reference else None
    observation = module.NativeObservation(torch, student=student, reference=ref,
        branch=branch, parts_validator=fixture_parts)
    return torch, student, ref, observation


def input_ids(torch, tokens=7, *, device=None):
    return FixtureTensor((1, tokens), dtype=torch.int64, device=device)


def hidden(model, *, branch="candidate", tokens=7, active=5):
    shape = (active, 4) if branch == "candidate" else (1, tokens, 4)
    return FixtureTensor(shape, device=model.embedding.weight.device, requires_grad=model.embedding.weight.requires_grad)


def forward(torch, student, ref, observation, *, index=0, tokens=7, active=5, use_reference=True):
    with observation.phase("forward", presentation_index=index):
        if ref is not None and use_reference:
            with torch.state(grad=False):
                ref.model(input_ids=input_ids(torch, tokens))
                ref.lm_head(hidden(ref, branch=observation.branch, tokens=tokens, active=active))
        student.model(input_ids=input_ids(torch, tokens))
        student.lm_head(hidden(student, branch=observation.branch, tokens=tokens, active=active))


def backward(torch, student, observation, *, index=0, tokens=7, active=5, layer=True, head=True):
    with observation.phase("backward", presentation_index=index):
        if layer:
            student.model.layers[0](FixtureTensor((1, tokens, 4), requires_grad=True))
        if head:
            student.lm_head(hidden(student, branch=observation.branch, tokens=tokens, active=active))


def children(model):
    return [model.model, model.lm_head] + list(model.model.layers)


def plain_json_tree(value):
    if type(value) is dict:
        return all(type(key) is str and plain_json_tree(item) for key, item in value.items())
    if type(value) is list:
        return all(plain_json_tree(item) for item in value)
    return value is None or type(value) in (str, bool, int, float)


class ObservationFixtureTests(unittest.TestCase):
    def test_explicit_branch_and_validator_are_required_without_altering_fake_context(self):
        torch, student = FixtureTorch(), FixtureModel(FixtureTorch())
        before = vars(torch).copy()
        for branch, validator in (("unknown", fixture_parts), (True, fixture_parts), ("candidate", None)):
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, branch=branch, parts_validator=validator)
        self.assertEqual(vars(torch), before)

    def test_validator_result_and_bounded_distinct_original_decoder_layers_are_required(self):
        for replacement in ((), (None, None), (None, None, None, None)):
            torch, student, _, _ = make()
            with self.subTest(parts=replacement), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, branch="candidate",
                    parts_validator=lambda *args, **kwargs: replacement)
        for count in (0, 65):
            torch, student = FixtureTorch(), FixtureModel(FixtureTorch(), layers=count)
            with self.subTest(count=count), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, branch="candidate", parts_validator=fixture_parts)
        torch, student, _, _ = make()
        student.model.layers[1] = student.model.layers[0]
        with self.assertRaises(ValueError):
            module.NativeObservation(torch, student=student, branch="candidate", parts_validator=fixture_parts)

    def test_every_student_parameter_storage_and_gradient_policy_is_checked(self):
        for change in (lambda m: setattr(m.model.layers[0].parameter, "dtype", "torch.bfloat16"),
                       lambda m: setattr(m.model.layers[0].parameter, "requires_grad", False)):
            torch, student, _, _ = make()
            change(student)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, branch="candidate", parts_validator=fixture_parts)

    def test_exact_original_tied_head_and_positive_declared_dimensions_are_required(self):
        for change in (lambda m: setattr(m.lm_head, "weight", FixtureParameter((11, 4), trainable=True)),
                       lambda m: setattr(m.lm_head, "out_features", 0),
                       lambda m: setattr(m.lm_head, "in_features", True)):
            torch, student, _, _ = make()
            change(student)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, branch="candidate", parts_validator=fixture_parts)

    def test_reference_object_and_every_parameter_storage_are_disjoint(self):
        torch, student, ref, _ = make(reference=True)
        with self.assertRaises(ValueError):
            module.NativeObservation(torch, student=student, reference=student, branch="candidate", parts_validator=fixture_parts)
        ref.model.layers[0].parameter.pointer = student.model.layers[0].parameter.pointer
        with self.assertRaises(ValueError):
            module.NativeObservation(torch, student=student, reference=ref, branch="candidate", parts_validator=fixture_parts)

    def test_reference_must_be_frozen_eval_and_gradient_free(self):
        for change in (lambda r: setattr(r, "training", True),
                       lambda r: setattr(r.model.layers[0].parameter, "requires_grad", True),
                       lambda r: setattr(r.model.layers[0].parameter, "grad", "fixture_nonempty_grad")):
            torch, student, ref, _ = make(reference=True)
            change(ref)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, reference=ref, branch="candidate", parts_validator=fixture_parts)

    def test_cuda_only_and_original_captured_device_cannot_drift(self):
        for device in (FixtureDevice("cpu"), FixtureDevice("cuda", 1)):
            torch = FixtureTorch()
            student = FixtureModel(torch)
            ref = FixtureModel(torch, trainable=False, device=device)
            with self.subTest(device=device), self.assertRaises(ValueError):
                module.NativeObservation(torch, student=student, reference=ref, branch="candidate", parts_validator=fixture_parts)
        torch, student, _, observation = make()
        with observation:
            student.embedding.weight.device = FixtureDevice("cuda", 1)
            with self.assertRaises(ValueError):
                with observation.phase("forward"):
                    pass
            student.embedding.weight.device = FixtureDevice()

    def test_hooks_install_on_original_modules_without_forward_rebinding_and_remove_on_exit(self):
        torch, student, ref, observation = make(reference=True)
        original_forward = type(student).forward
        original_children = children(student) + children(ref)
        with observation:
            for child in original_children:
                self.assertEqual(len(child.pre), 1)
                self.assertEqual(len(child.post), 1)
            self.assertIs(type(student).forward, original_forward)
            forward(torch, student, ref, observation)
        for child in original_children:
            self.assertFalse(child.pre)
            self.assertFalse(child.post)
            self.assertTrue(all(handle.remove_calls == 1 for handle in child.handles))

    def test_complete_forward_metadata_is_plain_bounded_json_without_native_or_tensor_claims(self):
        torch, student, ref, observation = make(reference=True)
        with observation:
            forward(torch, student, ref, observation)
            result = observation.validate(require_backward_recompute=False)
        self.assertTrue(plain_json_tree(result))
        self.assertLess(len(json.dumps(result)), 32768)
        self.assertFalse(result["tensor_outputs_values_graphs_or_storage_addresses_retained"])
        self.assertFalse(result["numeric_finiteness_or_original_checkpoint_provenance_established"])
        self.assertFalse(result["actual_backward_completion_or_native_admission_established"])
        self.assertEqual(result["observations"]["forward"]["student"]["backbone"]["output_shapes"],
            [{"shape": [1, 7, 4], "calls": 1}])

    def test_candidate_head_accepts_one_to128_rows_but_no_3d_or129_row_projection(self):
        for shape, valid in (((1, 4), True), ((128, 4), True), ((129, 4), False), ((1, 5, 4), False)):
            torch, student, _, observation = make()
            with observation:
                with observation.phase("forward"):
                    tensor = FixtureTensor(shape, requires_grad=True)
                    if valid:
                        student.lm_head(tensor)
                    else:
                        with self.assertRaises(ValueError):
                            student.lm_head(tensor)

    def test_dense_oracle_accepts_original_3d_short512_only(self):
        for shape, valid in (((1, 1, 4), True), ((1, 512, 4), True), ((1, 513, 4), False),
                             ((3, 4), False), ((2, 5, 4), False)):
            torch, student, _, observation = make(branch="dense")
            with observation:
                with observation.phase("forward"):
                    if valid:
                        student.lm_head(FixtureTensor(shape))
                    else:
                        with self.assertRaises(ValueError):
                            student.lm_head(FixtureTensor(shape))

    def test_completed_head_output_requires_exact_shape_and_actual_bf16_dtype(self):
        for shape, dtype in (((5, 12), None), (None, "torch.float32")):
            torch, student, _, observation = make()
            student.lm_head.output_shape_override, student.lm_head.output_dtype_override = shape, dtype
            with observation:
                with observation.phase("forward"):
                    with self.assertRaises(ValueError):
                        student.lm_head(hidden(student))

    def test_cuda_autocast_disabled_or_wrong_dtype_is_rejected_without_context_mutation(self):
        for changes in ({"amp": False}, {"amp_dtype": "torch.float16"}):
            torch, student, _, observation = make()
            with observation, torch.state(**changes):
                before = vars(torch).copy()
                with observation.phase("forward"):
                    with self.assertRaises(ValueError):
                        student.model(input_ids=input_ids(torch))
                self.assertEqual(vars(torch), before)

    def test_inference_mode_and_reference_grad_enabled_are_rejected(self):
        torch, student, ref, observation = make(reference=True)
        with observation:
            with torch.state(inference=True), observation.phase("forward"):
                with self.assertRaises(ValueError):
                    student.model(input_ids=input_ids(torch))
            with observation.phase("forward"):
                with self.assertRaises(ValueError):
                    ref.model(input_ids=input_ids(torch))

    def test_native_backbone_requires_same_cuda_int64_microbatch1_and_complete_context_bound(self):
        cases = [FixtureTensor((1, 7)), FixtureTensor((2, 7), dtype="torch.int64"),
                 FixtureTensor((1, 32769), dtype="torch.int64"), FixtureTensor((1, 0), dtype="torch.int64"),
                 FixtureTensor((1, 7), dtype="torch.int64", device=FixtureDevice("cpu"))]
        for tensor in cases:
            torch, student, _, observation = make()
            with observation, observation.phase("forward"):
                with self.subTest(shape=tensor.shape), self.assertRaises(ValueError):
                    student.model(input_ids=tensor)

    def test_outer_backward_nested_forward_restores_recompute_phase_without_second_consumption(self):
        torch, student, _, observation = make()
        with observation:
            with observation.phase("backward", presentation_index=3):
                forward(torch, student, None, observation, index=3)
                student.model.layers[0](FixtureTensor((1, 7, 4), requires_grad=True))
                student.lm_head(hidden(student))
            result = observation.validate()
        self.assertEqual(result["observations"]["forward"]["student"]["head"]["entries"], 1)
        self.assertEqual(result["observations"]["backward"]["student"]["head"]["entries"], 1)
        self.assertEqual(result["observations"]["backward"]["student"]["head"]["presentation_indices"], [3])
        self.assertNotIn("consumed_examples", result)

    def test_candidate_validation_requires_both_head_and_original_decoder_backward_entries(self):
        for layer, head in ((False, False), (True, False), (False, True)):
            torch, student, _, observation = make()
            with observation:
                forward(torch, student, None, observation)
                backward(torch, student, observation, layer=layer, head=head)
                with self.subTest(layer=layer, head=head), self.assertRaises(ValueError):
                    observation.validate()

    def test_two_installed_observers_do_not_double_attribute_shared_reference_calls(self):
        torch = FixtureTorch()
        dense, candidate = FixtureModel(torch), FixtureModel(torch)
        reference = FixtureModel(torch, trainable=False)
        d = module.NativeObservation(torch, student=dense, reference=reference, branch="dense", parts_validator=fixture_parts)
        c = module.NativeObservation(torch, student=candidate, reference=reference, branch="candidate", parts_validator=fixture_parts)
        with d, c:
            forward(torch, dense, reference, d)
            self.assertFalse(c.snapshot()["observations"]["forward"]["reference"])
            forward(torch, candidate, reference, c)
            backward(torch, candidate, c)
            ds, cs = d.validate(require_backward_recompute=False), c.validate()
        for result in (ds, cs):
            self.assertEqual(result["observations"]["forward"]["reference"]["head"]["entries"], 1)
            self.assertGreater(result["inactive_calls_ignored"], 0)

    def test_all_forward_layer_coverage_and_optional_unused_reference_policy_are_explicit(self):
        torch, student, ref, observation = make(reference=True)
        student.model.skip_layers = {1}
        with observation:
            forward(torch, student, ref, observation, use_reference=False)
            with self.assertRaises(ValueError):
                observation.validate(require_backward_recompute=False, require_reference=False)
            with self.assertRaises(ValueError):
                observation.validate(require_backward_recompute=False, require_all_layers=False)
            result = observation.validate(require_backward_recompute=False, require_all_layers=False, require_reference=False)
            self.assertFalse(result["validation_policy"]["require_reference"])

    def test_always_call_early_stop_records_interruption_without_claiming_backward_completion(self):
        torch, student, _, observation = make()
        with observation:
            forward(torch, student, None, observation)
            with observation.phase("backward", presentation_index=0):
                student.model.layers[0](FixtureTensor((1, 7, 4), requires_grad=True))
                student.lm_head.failure = RuntimeError("fixture checkpoint early-stop")
                with self.assertRaisesRegex(RuntimeError, "fixture checkpoint early-stop"):
                    student.lm_head(hidden(student))
                student.lm_head.failure = None
            result = observation.validate()
        head = result["observations"]["backward"]["student"]["head"]
        self.assertEqual((head["entries"], head["completed_outputs"], head["interrupted_without_output"]), (1, 0, 1))
        self.assertEqual(result["pending_module_calls"], 0)
        self.assertFalse(result["actual_backward_completion_or_native_admission_established"])

    def test_registration_and_body_errors_survive_best_effort_cleanup_and_clean_exit_reports_failure(self):
        torch, student, _, observation = make()
        student.model.fail_remove = True
        student.lm_head.fail_register_pre = True
        with self.assertRaisesRegex(RuntimeError, "fixture registration failure"):
            observation.__enter__()
        self.assertFalse(observation.snapshot()["hooks_active"])
        torch, student, _, observation = make()
        student.model.fail_remove = True
        with self.assertRaisesRegex(RuntimeError, "original fixture failure"):
            with observation:
                raise RuntimeError("original fixture failure")
        self.assertFalse(observation.snapshot()["hooks_active"])
        torch, student, _, observation = make()
        with self.assertRaisesRegex(ValueError, "could not be removed"):
            with observation:
                student.model.fail_remove = True

    def test_phase_lifecycle_indices_nesting_and_inflight_validation_fail_closed(self):
        torch, student, _, observation = make()
        with self.assertRaises(ValueError):
            with observation.phase("forward"):
                pass
        with observation:
            for name, index in (("unknown", 0), ("forward", True), ("forward", -1), ("forward", 64)):
                with self.subTest(name=name, index=index), self.assertRaises(ValueError):
                    with observation.phase(name, presentation_index=index):
                        pass
            with observation.phase("forward"):
                with self.assertRaises(ValueError):
                    observation.validate(require_backward_recompute=False)
            contexts = []
            try:
                for _ in range(module.MAX_NESTING):
                    context = observation.phase("backward")
                    context.__enter__()
                    contexts.append(context)
                with self.assertRaises(ValueError):
                    with observation.phase("forward"):
                        pass
            finally:
                for context in reversed(contexts):
                    context.__exit__(None, None, None)
        with self.assertRaises(ValueError):
            observation.__enter__()

    def test_original_identity_tie_dtype_and_layer_replacement_during_active_hooks_are_rejected(self):
        changes = [lambda m: setattr(m.lm_head, "weight", FixtureParameter((11, 4), trainable=True)),
                   lambda m: setattr(m.embedding.weight, "dtype", "torch.bfloat16"),
                   lambda m: m.model.layers.__setitem__(0, FixtureLayer(FixtureTorch(), True, FixtureDevice()))]
        for change in changes:
            torch, student, _, observation = make()
            with self.subTest(change=change), self.assertRaises(ValueError):
                with observation:
                    with observation.phase("forward"):
                        change(student)
                        student.lm_head(hidden(student))
            self.assertFalse(observation.snapshot()["hooks_active"])

    def test_production36_layer9_presentation_shape_metadata_stays_bounded_and_not_native434_proof(self):
        torch = FixtureTorch()
        student, ref = FixtureModel(torch, layers=36), FixtureModel(torch, trainable=False, layers=36)
        for model in (student, ref):
            # Metadata digit widths only: FixtureTensor allocates no elements.
            model.model.hidden = model.lm_head.in_features = 2048
            model.lm_head.out_features = 151936
            model.embedding.weight.shape = (151936, 2048)
        observation = module.NativeObservation(torch, student=student, reference=ref,
            branch="candidate", parts_validator=fixture_parts)
        with observation:
            for index in range(9):
                tokens, active = ((7, 5), (11, 7), (17, 9))[index % 3]
                with observation.phase("forward", presentation_index=index):
                    with torch.state(grad=False):
                        ref.model(input_ids=input_ids(torch, tokens))
                        ref.lm_head(FixtureTensor((active, 2048)))
                    student.model(input_ids=input_ids(torch, tokens))
                    student.lm_head(FixtureTensor((active, 2048), requires_grad=True))
                with observation.phase("backward", presentation_index=index):
                    for layer in student.model.layers:
                        layer(FixtureTensor((1, tokens, 2048), requires_grad=True))
                    student.lm_head(FixtureTensor((active, 2048), requires_grad=True))
            result = observation.validate()
        encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        self.assertLess(len(encoded), 128 << 10)
        self.assertEqual(result["expected_decoder_layers"], {"student": 36, "reference": 36})
        forward_student = result["observations"]["forward"]["student"]
        self.assertEqual(sum(name.startswith("decoder_") for name in forward_student), 36)
        self.assertTrue(all(forward_student[f"decoder_{index:02d}"]["entries"] == 9 for index in range(36)))
        backward_student = result["observations"]["backward"]["student"]
        self.assertEqual(sum(name.startswith("decoder_") for name in backward_student), 36)
        self.assertTrue(all(backward_student[f"decoder_{index:02d}"]["entries"] == 9 for index in range(36)))
        self.assertEqual(forward_student["backbone"]["output_shapes"][0]["shape"][-1], 2048)
        self.assertTrue(all(row["shape"][-1] == 151936 for row in backward_student["head"]["output_shapes"]))
        self.assertEqual(result["observations"]["backward"]["student"]["head"]["presentation_indices"], list(range(9)))
        self.assertEqual(len(tuple(student.parameters())), 37)
        self.assertTrue(plain_json_tree(result))
        self.assertFalse(result["actual_backward_completion_or_native_admission_established"])


if __name__ == "__main__":
    unittest.main()
