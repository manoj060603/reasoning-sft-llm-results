"""Unadmitted three-witness diagnostic using inherited Trainer.training_step.

Already constructed original dense/candidate Trainers and one separate frozen
reference are required. Preparation is the unchanged source-bound paired helper
contract. This helper runs three distinct short witnesses at actual group1,
compares raw and clipped gradients, clears them between witnesses and NEVER
calls an optimizer/scheduler step. It cannot be followed by the frozen GA9
controller on these consumed Trainers. The later GA9 subcase must be fresh.

No model/data loading, projection oracle, file I/O, CLI, job, original-file proof,
save/reload or native admission is provided. Required injected hooks own bounded
actual identity/numerical checks, not silent success defaults. The caller owns
the actual source/runtime/current-job/resource/native gates and must preserve a
typed failure's private partial_result. There is no retry/reset or rollback.
Authoring is not execution; native CPU fixtures and root review remain pending.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import importlib.metadata

import torch

from extension import paired_training_step_controller as paired


SCHEMA = "extension-standalone-training-step-controller-v1"
PAIRED_SOURCE_SHA256 = "9e446719e34f2370a00967066813b28acb6bf0a072cfade473eda3ef5e59e726"
BRANCHES = ("dense", "candidate")
ROLES = ("short0", "short1", "short2")


@dataclass(frozen=True)
class StandaloneStepHooks:
    """Every hook is mandatory and receives stage/trainers/reference/context.

    compare_scalars also receives finite detached dense/candidate scalar maps.
    Receipts must be bounded plain JSON, complete/passed and exactly stage-bound.
    Identity and tensor gates declare exact compared_tensor_count. Hook-source
    identity and actual comparison math remain caller/native-worker obligations.
    """
    verify_invariants: object
    verify_identity: object
    compare_scalars: object
    compare_tensors: object

    def __post_init__(self):
        paired._require(all(callable(getattr(self, name)) for name in self.__dataclass_fields__),
            "All standalone invariant/identity/scalar/tensor hooks are mandatory")


class StandaloneControllerFailure(RuntimeError):
    def __init__(self, result):
        self.partial_result = paired._json_copy(result, paired.MAX_JOURNAL_BYTES)
        super().__init__("Standalone training_step diagnostic failed; preserve its private partial result")


def _unchanged_optimization(trainer, scheduler_start, optimizer_groups_start):
    raw = paired._optimizer(trainer.optimizer)
    paired._require(not raw.state and trainer.state.global_step == 0 and
        paired._json_copy(trainer.lr_scheduler.state_dict()) == scheduler_start and
        paired._optimizer_groups(trainer, dict(trainer.model.named_parameters())) == optimizer_groups_start,
        "Standalone phase must keep empty Adam state, zero global step and unchanged scheduler/groups")
    paired._expect_lr(trainer, 0)
    paired._require(trainer.accelerator.gradient_accumulation_steps == 1 and
        trainer.accelerator.optimizer_step_was_skipped is False and trainer.optimizer.step_was_skipped is False,
        "Standalone prepared optimizer/Accelerator policy changed")


def run_standalone_training_steps(dense_trainer, candidate_trainer, batches, *, reference_model,
                                 hooks, observers, expected_parameters=434):
    """Compare three standalone raw/clipped gradients with no state updates.

    Actual inherited training_step owns the single group1 loss normalization and
    backward. Each source-reviewed compute_loss override must wrap only its
    super call in the supplied observer's nested forward phase; this controller
    keeps the outer backward phase and registered observers alive through it.
    A source-reviewed native worker must invoke this in a separate fresh child
    before the separate GA9 child, never reset counters or substitute this for
    full Trainer.train, long-memory or17-unique-row proof.
    """
    journal = paired._Journal()
    completed = {branch: 0 for branch in BRANCHES}
    finite_clip_boundaries = {branch: 0 for branch in BRANCHES}
    started = {branch: 0 for branch in BRANCHES}
    trainers = {"dense": dense_trainer, "candidate": candidate_trainer}
    result = {"schema": SCHEMA, "complete": False, "execution_kind": "standalone_actual_trainer_training_step_diagnostic",
        "trainer_train_called": False, "full_trainer_train_loop_established": False,
        "unique_row_driver_established": False, "long_context_memory_established": False,
        "native_provenance_established": False, "training_or_pilot_admitted": False,
        "paired_prepare_source_sha256": PAIRED_SOURCE_SHA256,
        "source_file_verification_owned_by_caller": True,
        "controller_optimizer_step_calls": 0, "controller_scheduler_step_calls": 0,
        "seed_contract": {"configured_python_numpy_torch_seed": paired.SEED,
            "initialization_and_actual_rng_seeding_owned_by_caller": True,
            "seed_configuration_alone_proves_rng_state": False, "controller_reseeds_generators": False}}
    try:
        paired._require(importlib.metadata.version("transformers") == "4.57.3" and
            importlib.metadata.version("accelerate") == "1.12.0" and
            str(torch.__version__).split("+")[0] == "2.9.0" and paired.np.__version__ == "1.26.4",
            "Pinned standalone diagnostic runtime required")
        paired._require(type(hooks) is StandaloneStepHooks and type(observers) is dict and set(observers) == set(BRANCHES),
            "All standalone comparison hooks and two explicit observers are mandatory")
        paired._require(type(batches) in (tuple, list) and len(batches) == 3 and dense_trainer is not candidate_trainer,
            "Exactly three unchanged distinct short batches and two separate Trainers required")
        batch_records = [paired._batch_record(value) for value in batches]
        semantic = [{name: value for name, value in row.items() if name != "row_index"} for row in batch_records]
        paired._require(len({paired._digest(row) for row in semantic}) == 3,
            "Three distinct source witnesses required; row-index changes alone are insufficient")
        reference = paired._parameters(reference_model, expected_parameters, trainable=False)
        paired._require(not reference_model.training and all(value.grad is None for value in reference.values()),
            "Every objective retains one separate frozen FP32 reference")
        reference_start = paired._reference_versions(reference_model)
        parameter_sets, devices, initial_versions = {}, {}, {}
        scheduler_start, optimizer_groups = {}, {}
        for branch in BRANCHES:
            trainer, observer = trainers[branch], observers[branch]
            paired._require(not getattr(trainer, "_standalone_controller_started", False),
                "Standalone Trainer is already consumed; no re-entry/reset")
            paired._require(getattr(trainer, "paired_observer", None) is observer and
                all(callable(getattr(observer, name, None)) for name in ("phase", "snapshot", "validate")) and
                hasattr(observer, "__enter__") and hasattr(observer, "__exit__"),
                "Source-reviewed nested-forward observer binding required")
            trainer._standalone_controller_started = True
            # This exact frozen preparation also sets the paired consumed flag;
            # a subsequent GA9 call on these objects is intentionally refused.
            parameter_sets[branch], devices[branch] = paired._prepare(trainer, reference_model, expected_parameters, branch)
            initial_versions[branch] = paired._reference_versions(trainer.model)
            scheduler_start[branch] = paired._json_copy(trainer.lr_scheduler.state_dict())
            optimizer_groups[branch] = paired._optimizer_groups(trainer, parameter_sets[branch])
            journal.add("standalone_branch_prepared", branch=branch, parameters=expected_parameters,
                actual_group_size=1, accelerator_accumulation_steps=1, optimizer_or_scheduler_step_called=False)
        device = devices["dense"]
        paired._require(devices["candidate"] == device and next(iter(reference.values())).device == device and
            list(parameter_sets["dense"]) == list(parameter_sets["candidate"]) == list(reference) and
            all(parameter_sets["dense"][name].shape == parameter_sets["candidate"][name].shape == reference[name].shape
                for name in reference), "Original names/shapes/devices must agree in both branches/reference")
        storages = [{(str(value.device), value.untyped_storage().data_ptr()) for value in values.values()}
            for values in (parameter_sets["dense"], parameter_sets["candidate"], reference)]
        paired._require(not any(storages[i] & storages[j] for i in range(3) for j in range(i)),
            "No student/reference Parameter storage may overlap")
        paired._require(dense_trainer.objective == candidate_trainer.objective and
            dense_trainer.terminal_suffix == candidate_trainer.terminal_suffix and
            dense_trainer.eos_token_id == candidate_trainer.eos_token_id and
            optimizer_groups["dense"] == optimizer_groups["candidate"],
            "Standalone objectives/native termination/Adam defaults and group order differ")

        def context(**values):
            return {"expected_parameters": expected_parameters, "objective": dense_trainer.objective,
                "completed_backwards": dict(completed), "finite_clip_boundaries": dict(finite_clip_boundaries),
                "actual_group_size": 1, "optimizer_updates": {branch: 0 for branch in BRANCHES}, **values}

        def invoke(name, stage, *, tensor_count=False, **extra):
            receipt = paired._json_copy(getattr(hooks, name)(stage=stage, dense_trainer=dense_trainer,
                candidate_trainer=candidate_trainer, reference_model=reference_model, context=context(), **extra))
            paired._require(type(receipt) is dict and receipt.get("complete") is True and receipt.get("passed") is True and
                receipt.get("stage") == stage, "Required standalone hook did not pass its exact stage")
            if tensor_count:
                paired._require(type(receipt.get("compared_tensor_count")) is int and
                    receipt["compared_tensor_count"] == expected_parameters, "Standalone hook must compare every Parameter")
            journal.add("standalone_hook_complete", hook=name, stage=stage, receipt=receipt)
            return receipt

        def invariant_versions():
            paired._require(paired._reference_versions(reference_model) == reference_start and not reference_model.training and
                all(value.grad is None for value in reference.values()), "Original frozen reference changed")
            for branch in BRANCHES:
                paired._require(paired._reference_versions(trainers[branch].model) == initial_versions[branch],
                    "Standalone student Parameter/buffer identity or ordinary mutation version changed")
                _unchanged_optimization(trainers[branch], scheduler_start[branch], optimizer_groups[branch])
            paired._require([paired._batch_record(value) for value in batches] == batch_records,
                "Sealed standalone batches changed")

        invoke("verify_invariants", "standalone_prepared")
        invoke("verify_identity", "standalone_initial_original", tensor_count=True)
        invariant_versions()
        journal.add("standalone_logical_rng_entry", configured_seed=paired.SEED,
            actual_state_sha256=paired._rng_hash(paired._rng_capture(device)), actual_seeding_owned_by_caller=True)
        with ExitStack() as stack:
            for branch in BRANCHES:
                stack.enter_context(observers[branch])
                paired._callback(trainers[branch], "on_train_begin")
                paired._callback(trainers[branch], "on_epoch_begin")
            for index in range(3):
                invariant_versions()
                for branch in BRANCHES:
                    trainer = trainers[branch]
                    paired._require(trainer.training_forward_examples == index and
                        all(value.grad is None for value in parameter_sets[branch].values()),
                        "Each standalone witness starts with exact prior count and cleared gradients")
                    trainer.current_gradient_accumulation_steps = 1
                    trainer.accelerator.sync_gradients = True
                    paired._callback(trainer, "on_step_begin")
                before = paired._rng_capture(device)
                after_dense, returned, metrics = None, {}, {}
                for branch in BRANCHES:
                    if branch == "candidate":
                        paired._rng_restore(before, device)
                    trainer, observer = trainers[branch], observers[branch]
                    paired._require(trainer.accelerator.gradient_accumulation_steps == 1 and
                        trainer.current_gradient_accumulation_steps == 1 and trainer.accelerator.sync_gradients is True,
                        "Only actual group1/Accelerator divisor1 standalone scaling is allowed")
                    trainer.paired_presentation_index = index
                    journal.add("standalone_training_step_begin", branch=branch, presentation_index=index,
                        source_role=ROLES[index], group_size=1, rng_before_sha256=paired._rng_hash(before))
                    started[branch] += 1
                    with observer.phase("backward", presentation_index=index):
                        loss = trainer.training_step(trainer.model, batches[index], num_items_in_batch=None)
                    paired._synchronize(device)
                    paired._require(isinstance(loss, torch.Tensor) and loss.numel() == 1 and not loss.requires_grad and
                        bool(torch.isfinite(loss)) and trainer.training_forward_examples == index + 1,
                        "Inherited standalone training_step must return finite detached group1 loss and consume one example")
                    returned[branch], metrics[branch] = float(loss.item()), paired._json_copy(trainer.last_metrics)
                    completed[branch] += 1
                    after = paired._rng_capture(device)
                    if branch == "dense":
                        after_dense = after
                    else:
                        paired._require(paired._rng_hash(after) == paired._rng_hash(after_dense),
                            "Standalone paired logical RNG progression differs")
                    journal.add("standalone_training_step_complete", branch=branch, presentation_index=index,
                        group1_loss=returned[branch], rng_after_sha256=paired._rng_hash(after))
                paired._rng_restore(after_dense, device)
                invoke("compare_scalars", "standalone_presentation_" + str(index),
                    dense={"group1_loss": returned["dense"], "group_size": 1, "metrics": metrics["dense"]},
                    candidate={"group1_loss": returned["candidate"], "group_size": 1, "metrics": metrics["candidate"]})
                for branch in BRANCHES:
                    paired._gradients(trainers[branch], parameter_sets[branch])
                invoke("verify_invariants", "standalone_before_clip_" + str(index))
                invoke("compare_tensors", "standalone_raw_gradients_" + str(index), tensor_count=True)
                for branch in BRANCHES:
                    paired._callback(trainers[branch], "on_pre_optimizer_step")
                    finite_clip_boundaries[branch] += 1
                    norm = trainers[branch].last_gradient_norm
                    paired._require(type(norm) is float and paired.math.isfinite(norm),
                        "Inherited standalone finite/clip callback must return a finite norm")
                invoke("compare_tensors", "standalone_clipped_gradients_" + str(index), tensor_count=True)
                invoke("compare_scalars", "standalone_clip_norm_" + str(index),
                    dense={"clip_norm": dense_trainer.last_gradient_norm}, candidate={"clip_norm": candidate_trainer.last_gradient_norm})
                invariant_versions()
                invoke("verify_identity", "standalone_unchanged_original_" + str(index), tensor_count=True)
                for branch in BRANCHES:
                    trainers[branch].model.zero_grad()
                    paired._require(all(value.grad is None for value in parameter_sets[branch].values()),
                        "Standalone original model.zero_grad must clear every gradient to None")
                journal.add("standalone_boundary_finalized", presentation_index=index, optimizer_steps=0, scheduler_steps=0)
            for branch in BRANCHES:
                paired._callback(trainers[branch], "on_epoch_end")
                paired._callback(trainers[branch], "on_train_end")
                observed = observers[branch].validate(require_backward_recompute=branch == "candidate",
                    require_all_layers=True, require_reference=dense_trainer.objective != "sft")
                journal.add("standalone_observer_validated", branch=branch, metadata=paired._json_copy(observed))
        invariant_versions()
        paired._require(completed == started == finite_clip_boundaries == {branch: 3 for branch in BRANCHES} and
            all(trainers[branch].training_forward_examples == 3 and
                all(value.grad is None for value in parameter_sets[branch].values()) for branch in BRANCHES),
            "Exactly three standalone backwards/clip boundaries and cleared final gradients required")
        invoke("verify_invariants", "standalone_final")
        return paired._json_copy({**result, "complete": True, "expected_parameters": expected_parameters,
            "fixture_count_is_not_native_proof": expected_parameters != 434, "completed_backwards": completed,
            "training_step_attempts": started, "finite_clip_boundaries": finite_clip_boundaries,
            "actual_group_sizes": [1, 1, 1], "events": journal.events}, paired.MAX_JOURNAL_BYTES)
    except BaseException as error:
        failure = {**result, "completed_backwards": completed, "training_step_attempts": started,
            "finite_clip_boundaries": finite_clip_boundaries,
            "partial_gradients_may_be_present": bool(any(started.values())),
            "error_type": type(error).__name__, "caller_must_preserve_private_partial_result_and_abort": True,
            "events": journal.events}
        raise StandaloneControllerFailure(failure) from error
