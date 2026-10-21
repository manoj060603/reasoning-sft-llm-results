"""Unadmitted CPU oracles for the disposable joint-memory diagnostic.

No original model, tokenizer, corpus or allocation is loaded here. Synthetic
metadata fixtures are schema oracles, never measured B200 memory evidence.
Execute this file only in a separately admitted native CPU QA stage.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from extension import joint_memory_admission as module
from extension import pilot_driver as pilot
from extension.cpu_feasibility import value_digest
from reasoning.data import prompt_id


EOS, SUFFIX = 6, (6, 7)
ROLES = ("short0", "short1", "short2", "short0", "short1", "short2", "short0", "short1",
    "long", "long", "short0", "short1", "short2", "short0", "short1", "short2", "short0")


def make_plan():
    return {"schema": module.SCHEMA, "diagnostic_only": True, "automatic_downstream": False,
        "deadline_utc": "2026-10-28T03:59:00+00:00", "case_order": list(module.MODES),
        "presentation_roles": list(ROLES), "diagnostic_repeat_ids": True, "optimizer_updates": 3,
        "settings": dict(module.SETTINGS), "inputs": {name: {"path": "extension/fixture/" + name + ".json",
            "sha256": "4" * 64} for name in ("native_plan", "prepared")},
        "resources": {"gpu_type": "B200", "gpus": 1, "wall_time_seconds": 7200, "host_cpus": 8,
            "host_ram_gib": 128, "scratch_free_gib": 160, "reserve_seconds": 172800},
        "tolerances": {"logit_max_abs": .001, "ce_abs": 1e-5, "kl_abs": 1e-5,
            "reload_logit_max_abs": .001, "headroom_min_bytes": 16 << 30, "headroom_fraction": .1}}


def make_prepared(*, eos=EOS, suffix=SUFFIX):
    rows = []
    for index, length in enumerate((4, 5, 6, 513)):
        question = "Synthetic joint-memory question " + str(index)
        identifier = prompt_id(question)
        tokens = [1, 2] + [3 + index % 3] * (length - 4) + list(suffix)
        labels = [-100, -100] + tokens[2:]
        original = {"id": identifier, "instruction": question, "input": "", "output": "Synthetic unchanged response " + str(index),
            "system": "Synthetic CPU diagnostic"}
        rows.append({"id": identifier, "input_ids": tokens, "labels": labels, "attention_mask": [1] * length,
            "original_row": original, "provenance": {"row_id": identifier, "row_sha256": value_digest(original),
                "prompt_tokens": 2, "total_tokens": length, "active_tokens": length - 2,
                "full_token_ids_sha256": value_digest(tokens), "labels_sha256": value_digest(labels),
                "eos_token_id": eos, "native_terminal_suffix": list(suffix), "complete_original_target": True,
                "truncated": False, "target_rewritten": False, "causal_shift_applied": False}})
    return {"rows": {"short": rows[:3], "long": rows[3]},
        "serialization": {"eos_token_id": eos, "terminal_suffix": list(suffix)}}


def consume(ledger, rows, *, rates=None):
    for update, group in enumerate(pilot.update_groups(len(rows), 8), 1):
        for index in group:
            ledger.begin(index)
            ledger.backward_complete(index)
        values = [.02] if rates is None else rates[update - 1]
        ledger.before_update(update, values)
        ledger.after_update(update, values)
    return ledger.finalize()


def observations(prepared, *, parameters=434):
    return [{"row_index": index, "source_role": role, "tokens": len(row["input_ids"]),
        "optimizer_updates_before_forward": index // 8, "successful_backwards_in_current_update": index % 8,
        "student_parameters_with_adam_moments": 0 if index < 8 else parameters,
        "resident_gradient_parameters_before_forward": 0 if index % 8 == 0 else parameters,
        "forward_complete": True, "backward_complete": True, "optimizer_update_after_backward_complete": True}
        for index, (role, row) in enumerate(zip(ROLES, module.presentation_rows(prepared)))]


class JointPlanAndScheduleTests(unittest.TestCase):
    def test_fixed_seventeen_presentations_put_two_longs_in_second_update(self):
        self.assertEqual(module.PRESENTATION_ROLES, ROLES)
        prepared = make_prepared()
        before = deepcopy(prepared)
        rows = module.presentation_rows(prepared)
        self.assertEqual(len(rows), 17)
        self.assertEqual([row["id"] for row in rows[8:10]], [prepared["rows"]["long"]["id"]] * 2)
        self.assertEqual(pilot.update_groups(17, 8), [list(range(8)), list(range(8, 16)), [16]])
        lookup = {"short" + str(index): row for index, row in enumerate(prepared["rows"]["short"])}
        lookup["long"] = prepared["rows"]["long"]
        self.assertEqual(rows, [lookup[role] for role in ROLES])
        self.assertEqual(prepared, before)

    def test_presentation_envelope_requires_four_distinct_complete_source_witnesses(self):
        for change in (lambda x: x["rows"]["short"].pop(),
            lambda x: x["rows"].update(long=deepcopy(x["rows"]["short"][0])),
            lambda x: x["rows"]["short"][0].update(input_ids=[1] * 513),
            lambda x: x["rows"]["long"].update(input_ids=[1] * 512)):
            value = make_prepared()
            change(value)
            with self.assertRaises(ValueError):
                module.presentation_rows(value)

    def test_exact_diagnostic_plan_is_valid_but_never_a_training_pilot(self):
        plan = make_plan()
        self.assertEqual(module.validate_plan(plan), plan)
        for change in (lambda x: x.update(diagnostic_only=False), lambda x: x.update(automatic_downstream=True),
            lambda x: x.update(diagnostic_repeat_ids=False), lambda x: x.update(optimizer_updates=2),
            lambda x: x["presentation_roles"].__setitem__(9, "short0"), lambda x: x["case_order"].reverse()):
            value = deepcopy(plan)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_precision_accumulation_headroom_and_finite_time_cannot_be_weakened(self):
        for change in (lambda x: x["settings"].update(accumulation=2), lambda x: x["settings"].update(reference_dtype="bfloat16"),
            lambda x: x["settings"].update(resume=True), lambda x: x["resources"].update(wall_time_seconds=7201),
            lambda x: x["resources"].update(reserve_seconds=0), lambda x: x["resources"].update(gpus=2),
            lambda x: x["tolerances"].update(headroom_fraction=.09),
            lambda x: x["tolerances"].update(headroom_min_bytes=(16 << 30) - 1),
            lambda x: x["tolerances"].update(ce_abs=float("nan"))):
            value = make_plan()
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)


class DiagnosticConsumptionTests(unittest.TestCase):
    def test_repeat_source_ids_are_explicit_and_all_seventeen_positions_are_counted(self):
        prepared = make_prepared()
        rows = module.presentation_rows(prepared)
        receipt = consume(module.DiagnosticLedger(rows), rows)
        self.assertTrue(receipt["diagnostic_repeated_ids"])
        self.assertEqual(receipt["unique_source_examples"], 4)
        self.assertEqual(receipt["examples"], 17)
        self.assertEqual(receipt["source_role_sequence"], list(ROLES))
        self.assertEqual(receipt["presentation_indices"], list(range(17)))
        self.assertEqual(receipt["presentation_ids"], [row["id"] for row in rows])
        self.assertEqual([entry["microbatches"] for entry in receipt["updates"]], [8, 8, 1])
        self.assertEqual(receipt["total_tokens"], sum(len(row["input_ids"]) for row in rows))
        self.assertEqual(receipt["active_tokens"], sum(len(row["input_ids"]) - 2 for row in rows))
        self.assertEqual(receipt["eos_tokens"], 17)

    def test_main_pilot_unique_id_control_is_unchanged_and_rejects_diagnostic_repeats(self):
        rows = module.presentation_rows(make_prepared())
        with self.assertRaises(ValueError):
            consume(pilot.ConsumptionLedger(rows, 8), rows)
        self.assertEqual(consume(module.DiagnosticLedger(rows), rows)["unique_source_examples"], 4)

    def test_only_the_exact_seventeen_unchanged_diagnostic_repeats_are_allowed(self):
        rows = module.presentation_rows(make_prepared())
        for changed in (rows[:-1], rows + [rows[0]], list(reversed(rows))):
            with self.assertRaises(ValueError):
                module.DiagnosticLedger(changed)
        changed = deepcopy(rows)
        changed[9]["original_row"]["output"] += " rewritten"
        with self.assertRaises(ValueError):
            module.DiagnosticLedger(changed)
        with self.assertRaises(ValueError):
            module.DiagnosticLedger(rows, accumulation=2)

    def test_failed_backward_missing_tail_or_missing_final_update_never_completes(self):
        rows = module.presentation_rows(make_prepared())
        ledger = module.DiagnosticLedger(rows)
        ledger.begin(0)
        with self.assertRaises(ValueError):
            ledger.finalize()
        ledger = module.DiagnosticLedger(rows)
        for update, group in enumerate(pilot.update_groups(16, 8), 1):
            for index in group:
                ledger.begin(index)
                ledger.backward_complete(index)
            ledger.before_update(update, [.02])
            ledger.after_update(update, [.02])
        with self.assertRaises(ValueError):
            ledger.finalize()
        ledger.begin(16)
        ledger.backward_complete(16)
        ledger.before_update(3, [.02])
        with self.assertRaises(ValueError):
            ledger.finalize()


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, use_cache=False):
        if labels is not None or use_cache:
            raise ValueError("Exactly one shift outside the model and no cache")
        return {"logits": self.head(self.embedding(input_ids).cumsum(dim=1).tanh())}


def collate(batch):
    if len(batch) != 1:
        raise ValueError("Microbatch one only")
    return {name: torch.tensor([batch[0][name]], dtype=torch.int64)
        for name in ("input_ids", "labels", "attention_mask", "row_index")}


def row_batch(row):
    return collate([{**row, "row_index": 0}])


def manual_loss(model, row, reference=None, beta=.05):
    batch = row_batch(row)
    logp = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1].log_softmax(-1)
    target = batch["labels"][:, 1:]
    mask = target != -100
    ce = -logp.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)
    if reference is None:
        return ce[mask].mean()
    with torch.no_grad():
        logq = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1].log_softmax(-1)
    kl = (logq.exp() * (logq - logp)).sum(-1)
    return (ce + beta * kl)[mask].mean()


class PreForwardTensorStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.student = TinyCausal()
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=.001)
        self.parameters = len(list(self.student.parameters()))
        self.prepared = make_prepared()

    def inspect(self, *, completed=0, backwards=0, index=9):
        return module.inspect_pre_forward(self.student, self.optimizer, completed_updates=completed,
            backward_count=backwards, tokens=len(self.prepared["rows"]["long"]["input_ids"]),
            row_index=index, source_role="long", expected_parameters=self.parameters)

    def initialize_adam(self):
        manual_loss(self.student, self.prepared["rows"]["short"][0]).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def test_real_tensor_state_distinguishes_no_adam_adam_only_and_joint_residency(self):
        fresh = self.inspect()
        self.assertEqual(fresh["student_parameters_with_adam_moments"], 0)
        self.assertEqual(fresh["resident_gradient_parameters_before_forward"], 0)
        self.initialize_adam()
        first_long = self.inspect(completed=1, backwards=0, index=8)
        self.assertEqual(first_long["student_parameters_with_adam_moments"], self.parameters)
        self.assertEqual(first_long["resident_gradient_parameters_before_forward"], 0)
        manual_loss(self.student, self.prepared["rows"]["long"]).backward()
        second_long = self.inspect(completed=1, backwards=1)
        self.assertEqual(second_long["student_parameters_with_adam_moments"], self.parameters)
        self.assertEqual(second_long["resident_gradient_parameters_before_forward"], self.parameters)
        self.assertEqual(second_long["successful_backwards_in_current_update"], 1)
        self.assertFalse(second_long["forward_complete"])
        self.assertFalse(second_long["backward_complete"])

    def test_old_two_long_then_first_adam_sequence_has_no_adam_during_either_forward(self):
        first = self.inspect(index=8)
        manual_loss(self.student, self.prepared["rows"]["long"]).backward()
        second = self.inspect(backwards=1)
        self.assertEqual(first["student_parameters_with_adam_moments"], 0)
        self.assertEqual(second["student_parameters_with_adam_moments"], 0)
        self.assertEqual(second["resident_gradient_parameters_before_forward"], self.parameters)

    def test_nonfinite_or_wrong_precision_resident_gradients_and_adam_moments_are_rejected(self):
        self.initialize_adam()
        parameter = next(self.student.parameters())
        self.optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            self.inspect(completed=1)
        self.optimizer.state[parameter]["exp_avg"].zero_()
        self.optimizer.state[parameter]["exp_avg_sq"] = self.optimizer.state[parameter]["exp_avg_sq"].to(torch.bfloat16)
        with self.assertRaises(ValueError):
            self.inspect(completed=1)
        self.optimizer.state[parameter]["exp_avg_sq"] = self.optimizer.state[parameter]["exp_avg_sq"].float()
        parameter.grad = torch.full_like(parameter, float("inf"))
        with self.assertRaises(ValueError):
            self.inspect(completed=1, backwards=1)

    def test_real_student_count_and_integral_update_counters_cannot_be_faked(self):
        with self.assertRaises(ValueError):
            module.inspect_pre_forward(self.student, self.optimizer, completed_updates=0, backward_count=0,
                tokens=513, row_index=9, source_role="long")
        for value in (-1, .5, True):
            with self.assertRaises(ValueError):
                module.inspect_pre_forward(self.student, self.optimizer, completed_updates=value, backward_count=0,
                    tokens=513, row_index=9, source_role="long", expected_parameters=self.parameters)


class ObservationGateTests(unittest.TestCase):
    def test_only_fully_completed_exact_state_sequence_proves_the_joint_envelope(self):
        prepared = make_prepared()
        values = observations(prepared)
        shape = module.validate_observations(values, prepared)
        self.assertTrue(shape["long_context_joint_peak_proven"])
        self.assertEqual(shape["maximum_complete_tokens"], 513)
        self.assertEqual(shape["scope"], "long_after_adam_accumulation")

    def test_missing_reordered_or_short_only_observations_cannot_replace_second_long(self):
        prepared = make_prepared()
        values = observations(prepared)
        for bad in (values[:-1], values + [values[0]], list(reversed(values))):
            with self.assertRaises(ValueError):
                module.validate_observations(bad, prepared)
        for key, wrong in (("tokens", 512), ("source_role", "short0"),
            ("student_parameters_with_adam_moments", 0), ("resident_gradient_parameters_before_forward", 0),
            ("optimizer_updates_before_forward", 0), ("successful_backwards_in_current_update", 0)):
            bad = deepcopy(values)
            bad[9][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_observations(bad, prepared)

    def test_before_forward_flags_never_imply_successful_backward_or_later_optimizer_update(self):
        prepared = make_prepared()
        for index in (0, 8, 9, 16):
            for flag in ("forward_complete", "backward_complete", "optimizer_update_after_backward_complete"):
                values = observations(prepared)
                values[index][flag] = False
                with self.subTest(index=index, flag=flag), self.assertRaises(ValueError):
                    module.validate_observations(values, prepared)


class DiagnosticDataset(Dataset):
    def __init__(self, rows):
        self.rows, self.reads = rows, []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads.append(index)
        return {**deepcopy(self.rows[index]), "row_index": index}


def training_args(path):
    return TrainingArguments(output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=3, num_train_epochs=1,
        max_grad_norm=0., save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True,
        remove_unused_columns=False, dataloader_pin_memory=False, dataloader_num_workers=0,
        dataloader_drop_last=False, label_smoothing_factor=0.)


class NativeRepeatedTrainerTests(unittest.TestCase):
    def test_native_ga8_repeated_schedule_matches_three_manual_equal_example_updates(self):
        torch.manual_seed(22)
        initial = TinyCausal()
        prepared = make_prepared()
        rows = module.presentation_rows(prepared)
        for objective in ("sft", "ce_kl"):
            actual, expected = deepcopy(initial), deepcopy(initial)
            reference = None if objective == "sft" else deepcopy(initial)
            before = None if reference is None else {name: value.clone() for name, value in reference.state_dict().items()}
            oracle_optimizer = torch.optim.SGD(expected.parameters(), lr=.02)
            for group in (range(8), range(8, 16), range(16, 17)):
                oracle_optimizer.zero_grad()
                for index in group:
                    (manual_loss(expected, rows[index], reference) / len(group)).backward()
                oracle_optimizer.step()
            ledger = module.DiagnosticLedger(rows)
            dataset = DiagnosticDataset(rows)
            dataset[0]  # A prefetched source row must not count as a forward.
            with tempfile.TemporaryDirectory() as temporary:
                optimizer = torch.optim.SGD(actual.parameters(), lr=.02)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                trainer = module.make_trainer_class()(model=actual, args=training_args(temporary),
                    objective=objective, reference_model=reference, beta=.05, eos_token_id=EOS,
                    terminal_suffix=SUFFIX, chunk_tokens=128, checkpoint_chunks=True, consumption=ledger,
                    train_dataset=dataset, data_collator=collate, optimizers=(optimizer, scheduler))
                trainer.train()
                self.assertEqual(trainer.state.global_step, 3)
                self.assertEqual(trainer.training_forward_examples, 17)
                receipt = ledger.finalize()
                self.assertEqual([entry["microbatches"] for entry in receipt["updates"]], [8, 8, 1])
                self.assertEqual(receipt["presentation_indices"], list(range(17)))
                self.assertEqual(receipt["eos_tokens"], 17)
                self.assertTrue(receipt["diagnostic_repeated_ids"])
                for first, second in zip(actual.parameters(), expected.parameters()):
                    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
                if reference is not None:
                    for name, value in reference.state_dict().items():
                        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                    self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters()))

    def test_actual_tiny_adam_and_gradient_residency_are_observed_before_second_long_forward(self):
        torch.manual_seed(23)
        prepared = make_prepared()
        rows = module.presentation_rows(prepared)
        student, reference = TinyCausal(), TinyCausal()
        count = len(list(student.parameters()))
        values = []
        ledger = module.DiagnosticLedger(rows)
        base = module.make_trainer_class()

        class ObservedTrainer(base):
            def training_step(self, model, inputs, num_items_in_batch=None):
                index = int(inputs["row_index"].item())
                value = module.inspect_pre_forward(model, self.optimizer, completed_updates=len(ledger.updates),
                    backward_count=len(ledger.presentations) - 8 * len(ledger.updates), tokens=inputs["input_ids"].shape[1],
                    row_index=index, source_role=ROLES[index], expected_parameters=count)
                values.append(value)
                result = super().training_step(model, inputs, num_items_in_batch)
                value["forward_complete"] = value["backward_complete"] = True
                return result

        class CompletedUpdates(TrainerCallback):
            def on_optimizer_step(self, args, state, control, **kwargs):
                for index in ledger.groups[state.global_step]:
                    values[index]["optimizer_update_after_backward_complete"] = True

        with tempfile.TemporaryDirectory() as temporary:
            optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            trainer = ObservedTrainer(model=student, args=training_args(temporary), objective="ce_kl",
                reference_model=reference, beta=.05, eos_token_id=EOS, terminal_suffix=SUFFIX,
                chunk_tokens=128, checkpoint_chunks=True, consumption=ledger,
                train_dataset=DiagnosticDataset(rows), data_collator=collate, optimizers=(optimizer, scheduler))
            trainer.add_callback(CompletedUpdates())
            trainer.train()
        # The tensor count is deliberately tiny; this is no B200 admission.
        shape = module.validate_observations(values, prepared, expected_parameters=count)
        self.assertEqual(values[8]["resident_gradient_parameters_before_forward"], 0)
        self.assertEqual(values[9]["student_parameters_with_adam_moments"], count)
        self.assertEqual(values[9]["resident_gradient_parameters_before_forward"], count)
        self.assertTrue(shape["long_context_joint_peak_proven"])

    def test_changed_batch_or_missing_native_eos_cannot_be_reported_as_original_presentation(self):
        rows = module.presentation_rows(make_prepared())
        def tampered(batch):
            value = collate(batch)
            value["input_ids"][0, -1] = 5
            value["labels"][0, -1] = 5
            return value
        ledger = module.DiagnosticLedger(rows)
        with tempfile.TemporaryDirectory() as temporary:
            trainer = module.make_trainer_class()(model=TinyCausal(), args=training_args(temporary), objective="sft",
                eos_token_id=EOS, terminal_suffix=SUFFIX, consumption=ledger,
                train_dataset=DiagnosticDataset(rows), data_collator=tampered)
            with self.assertRaises(ValueError):
                trainer.train()
            with self.assertRaises(ValueError):
                ledger.finalize()


class FixtureEvidence:
    def __init__(self):
        self.files, self.calls, self.fail = {}, 0, False

    def unchanged(self):
        self.calls += 1
        if self.fail:
            raise ValueError("synthetic input snapshot changed")


class JointCaseAndRunGuardsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "extension/runs/joint-fixture-attempt"
        self.plan, self.prepared = make_plan(), make_prepared(eos=151645, suffix=(151645, 198))
        self.evidence = FixtureEvidence()
        self.context = {"root": self.root, "plan": self.plan, "prepared": self.prepared,
            "admission": {"job_id": "42"}, "evidence": self.evidence,
            "plan_sha256": "1" * 64, "prepared_sha256": "2" * 64, "binding_sha256": "3" * 64,
            "code_sha256": {name: "4" * 64 for name in module.REQUIRED_CODE},
            "paths": {name: self.root / ("extension/fixture/" + name + ".json") for name in ("plan", "qa", "admission")}}
        self.calls = []
        for owner, name, value in ((module.native, "require_gpu_allocation", lambda: {"job_id": "42", "node": "owl-sr001"}),
            (module, "load_inputs", lambda *args, **kwargs: self.context),
            (module, "deadline_guard", lambda *args, **kwargs: None)):
            hook = patch.object(owner, name, value)
            hook.start()
            self.addCleanup(hook.stop)
        disk = patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30))
        disk.start()
        self.addCleanup(disk.stop)

    def case_receipt(self, mode, directory):
        directory.mkdir(parents=True)
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir()
        rows = module.presentation_rows(self.prepared)
        before = [[2e-6 * pilot.cosine_factor(index)] for index in range(3)]
        after = [[2e-6 * pilot.cosine_factor(index + 1)] for index in range(3)]
        consumption = consume(module.DiagnosticLedger(rows), rows, rates=before)
        shape = module.validate_observations(observations(self.prepared), self.prepared)
        memory = {"peak_allocated_bytes": 80 << 30, "peak_reserved_bytes": 82 << 30, "total_bytes": 192 << 30}
        memory["headroom"] = module.native.headroom_report(82 << 30, 192 << 30, 16 << 30, .1)
        files = {"checkpoint/config.json": b'{"synthetic":true}',
            "checkpoint/model.safetensors": b"small fixture bytes, never actual model weights",
            "consumption.json": json.dumps(consumption, sort_keys=True).encode(),
            "shape_observations.json": json.dumps(shape, sort_keys=True).encode(),
            "presentations.jsonl": b"".join(json.dumps(value, sort_keys=True).encode() + b"\n"
                for value in consumption["presentations"])}
        for name, content in files.items():
            (directory / name).write_bytes(content)
        return {"schema": module.CASE_SCHEMA, "complete": True, "mode": mode,
            "binding_sha256": self.context["binding_sha256"], "plan_sha256": self.context["plan_sha256"],
            "prepared_sha256": self.context["prepared_sha256"], "code_sha256": self.context["code_sha256"],
            "base": {"repo_id": module.native.REPOSITORY, "revision": module.native.REVISION,
                "files": dict(module.native.ORIGINAL_FILES)}, "versions": dict(module.native.VERSIONS),
            "gpu_identity": {"name": "NVIDIA B200"},
            "settings": {"objective": mode, "beta": 0. if mode == "sft" else .05, "chunk_tokens": 128,
                "checkpoint_chunks": True, "parameter_dtype": "float32", "reference_dtype": "float32",
                "compute_dtype": "bfloat16", "attention": "sdpa"},
            "accumulation": 8, "completed_optimizer_updates": 3, "diagnostic_repeated_ids": True,
            "presentation_ids": [row["id"] for row in rows], "shape_evidence": shape, "consumption": consumption,
            "source_native_serialization_verified": True, "global_loss_unchanged": True,
            "finite_weights": True, "finite_moments": True, "reference_unchanged": True, "memory": memory,
            "checks": {name: True for name in module.native.REQUIRED_CHECKS}, "lr_used": before,
            "lr_after_scheduler": after, "output_files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}

    def successful_child(self, command, **kwargs):
        self.calls.append((command, kwargs))
        mode = command[command.index("--mode") + 1]
        directory = self.output / "cases" / mode
        receipt = self.case_receipt(mode, directory)
        (directory / "complete.json").write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    def run_diagnostic(self, runner=None, output=None):
        return module.run(self.root, "extension/fixture/plan.json", output or self.output,
            "extension/fixture/qa.json", "extension/fixture/admission.json", runner=runner or self.successful_child)

    def test_four_fresh_ordered_children_complete_only_after_all_case_proofs(self):
        result = self.run_diagnostic()
        self.assertTrue(result["complete"])
        self.assertFalse(result["training_pilot"])
        self.assertTrue(result["diagnostic_repeat_ids"])
        self.assertFalse(result["automatic_downstream"])
        self.assertEqual(tuple(result["case_receipts"]), module.MODES)
        self.assertEqual([command[command.index("--mode") + 1] for command, _ in self.calls], list(module.MODES))
        self.assertTrue(all(command[1:3] == ["-m", "extension.joint_memory_admission"] for command, _ in self.calls))
        self.assertTrue(all(kwargs["check"] is False and 0 < kwargs["timeout"] <= 7200 for _, kwargs in self.calls))
        self.assertGreaterEqual(self.evidence.calls, 5)

    def test_first_failed_child_preserves_completed_cases_and_never_retries(self):
        def child(command, **kwargs):
            if self.calls:
                self.calls.append((command, kwargs))
                return SimpleNamespace(returncode=1)
            return self.successful_child(command, **kwargs)
        with self.assertRaises(ValueError):
            self.run_diagnostic(child)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / "complete.json").exists())
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(tuple(failure["completed_cases"]), (module.MODES[0],))
        self.assertFalse(failure["retry"])
        self.assertFalse(failure["resume_supported"])

    def test_child_timeout_or_missing_receipt_leaves_no_complete_result(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_diagnostic(timeout)
        self.assertFalse((self.output / "complete.json").exists())
        with tempfile.TemporaryDirectory() as temporary:
            other = self.root / "extension/runs/second-incomplete-attempt"
            with self.assertRaises(FileNotFoundError):
                self.run_diagnostic(lambda *args, **kwargs: SimpleNamespace(returncode=0), output=other)
            self.assertTrue((other / "failure.json").exists())

    def test_wrong_allocation_reused_output_or_foreign_output_never_starts_more_work(self):
        self.context["admission"]["job_id"] = "43"
        with self.assertRaises(ValueError):
            self.run_diagnostic()
        self.assertEqual(self.calls, [])
        self.context["admission"]["job_id"] = "42"
        self.run_diagnostic()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_diagnostic()
        self.assertEqual(len(self.calls), 4)
        with self.assertRaises(ValueError):
            self.run_diagnostic(output=self.root / "outside")

    def test_input_change_or_insufficient_scratch_blocks_complete_without_automatic_retry(self):
        self.evidence.fail = True
        with self.assertRaisesRegex(ValueError, "input snapshot changed"):
            self.run_diagnostic()
        self.assertFalse((self.output / "complete.json").exists())
        with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=159 << 30)):
            with self.assertRaises(ValueError):
                self.run_diagnostic(output=self.root / "extension/runs/no-scratch")

    def test_case_binding_missing_joint_state_or_missing_finite_check_is_rejected(self):
        directory = self.root / "extension/runs/direct-case"
        receipt = self.case_receipt("ce_kl", directory)
        self.assertEqual(module.validate_case_receipt(receipt, self.context, "ce_kl", directory), receipt)
        for change in (lambda x: x.update(code_sha256={}), lambda x: x.update(prepared_sha256="0" * 64),
            lambda x: x.update(diagnostic_repeated_ids=False),
            lambda x: x["shape_evidence"]["observations"][9].update(resident_gradient_parameters_before_forward=0),
            lambda x: x["checks"].pop("finite_moments"),
            lambda x: x["consumption"].update(eos_tokens=0),
            lambda x: x["lr_used"].__setitem__(0, [2e-6])):
            value = deepcopy(receipt)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_case_receipt(value, self.context, "ce_kl", directory)

    def test_case_saved_bytes_and_exact_journal_inventory_are_verified(self):
        directory = self.root / "extension/runs/direct-case"
        receipt = self.case_receipt("sft", directory)
        path = directory / "checkpoint/model.safetensors"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaises(ValueError):
            module.validate_case_receipt(receipt, self.context, "sft", directory)
        path.write_bytes(original)
        (directory / "checkpoint/unbound.bin").write_bytes(b"unbound")
        with self.assertRaises(ValueError):
            module.validate_case_receipt(receipt, self.context, "sft", directory)


if __name__ == "__main__":
    unittest.main()
