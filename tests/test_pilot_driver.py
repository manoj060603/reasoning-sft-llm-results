"""Prospective pilot-driver oracles; run only in admitted native CPU QA.

Every tensor/model below is a tiny synthetic CPU fixture, with no download,
canonical source processing, pretrained model or experiment execution.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import TrainingArguments

from extension import pilot_driver as module
from extension import preserving_data as data
from reasoning.data import prompt_id


EOS, SUFFIX = 6, (6, 7)


def make_plan(*, count=9, stage="integration_diagnostic"):
    updates = math.ceil(count / 8)
    return {"schema": module.SCHEMA, "stage": stage, "objective": "sft", "beta": 0.,
        "seed": 20261021, "learning_rate": 2e-6, "weight_decay": .01, "clip_norm": 1.,
        "chunk_tokens": 128, "checkpoint_chunks": True, "accumulation": 8, "microbatch": 1,
        "parameter_dtype": "float32", "reference_dtype": "float32", "compute_dtype": "bfloat16",
        "attention": "sdpa", "gradient_checkpointing": True, "use_reentrant": False,
        "epochs": 1, "resume": False, "automatic_downstream": False,
        "cutoff": 32768 if stage == "pilot" else 512,
        "schedule": {"kind": "cosine", "total_steps": 1875, "warmup_steps": 57, "num_cycles": .5},
        "exposure": {"examples": count, "updates": updates},
        "checkpoints": [64, updates] if stage == "pilot" else [updates],
        "resources": {"wall_time_seconds": 7200, "reserve_seconds": 172800, "host_cpus": 8,
            "host_ram_gib": 128, "scratch_free_gib": 160, "gpu_type": "B200", "gpus": 1},
        "tolerances": {"logit_max_abs": .001, "ce_abs": 1e-5, "kl_abs": 1e-5,
            "reload_logit_max_abs": .001, "headroom_min_bytes": 16 << 30, "headroom_fraction": .1}}


class ExactFixtureTokenizer:
    eos_token_id = 151645
    chat_template = "synthetic pilot serialization fixture"
    tail = "<FIXTURE_EOS>\n"

    def encode(self, text, add_special_tokens=False):
        if text.endswith(self.tail):
            return [1 + ord(char) % 100 for char in text[:-len(self.tail)]] + [151645, 198]
        return [1 + ord(char) % 100 for char in text]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, add_generation_prompt=False):
        text = "SYS:" + messages[0]["content"] + "\nUSER:" + messages[1]["content"] + "\nASSISTANT:"
        if len(messages) == 3:
            text += messages[2]["content"] + self.tail
        return self.encode(text) if tokenize else text


def make_cache(plan, plan_sha256):
    rows = [{"id": prompt_id("Synthetic preserved question " + str(index)),
        "instruction": "Synthetic preserved question " + str(index), "input": "",
        "output": "original answer " + "x" * (index % 3 + 1), "system": "Fixture system"}
        for index in range(plan["exposure"]["examples"])]
    components = {purpose: {"scope": scope, "pool_ids": [], "audit_receipt_sha256": "1" * 64,
        "criterion_sha256": "2" * 64, "question_inventory_sha256": "3" * 64}
        for purpose, scope in (("reserved_source_evaluation", "all_reserved_source_splits"),
            ("final_benchmark_questions", "all_nine_final_benchmarks"))}
    exclusions = {"schema": data.EXCLUSION_SCHEMA, "id_namespace": data.POOL_ID_NAMESPACE,
        "population_ids_sha256": data.value_sha256(sorted(row["id"] for row in rows)), "components": components}
    source_identity = {"pool_sha256": "4" * 64, "features_sha256": "5" * 64}
    tokenizer = ExactFixtureTokenizer()
    tokenizer_identity = {"files": dict(module.TOKENIZER_FILES), "chat_template_sha256": data.text_sha256(tokenizer.chat_template),
        "eos_token_id": 151645}
    role = "pilot" if plan["stage"] == "pilot" else "diagnostic"
    examples = data.serialize_selected(rows, [row["id"] for row in rows], tokenizer,
        excluded_pool_ids=set(), exclusion_receipt=exclusions, source_identity=source_identity,
        tokenizer_identity=tokenizer_identity, context_role=role, cutoff=plan["cutoff"],
        eos_token_id=151645, terminal_suffix=(151645, 198), terminal_text=tokenizer.tail,
        vocab_size=151936)["examples"]
    pins = set(module.SOURCE_PINS) | ({"accumulation8_admission", "long_after_adam_admission"}
        if plan["stage"] == "pilot" else set())
    return {"schema": module.CACHE_SCHEMA, "plan_sha256": plan_sha256,
        "base": {"repo_id": module.REPOSITORY, "revision": module.REVISION, "path": "/fixture/original-base",
            "files": dict(module.ORIGINAL_FILES)},
        "serialization": {"cutoff": plan["cutoff"], "context_role": role, "eos_token_id": 151645,
            "terminal_suffix": [151645, 198], "terminal_text": tokenizer.tail, "vocab_size": 151936},
        "source_identity": source_identity, "tokenizer_identity": tokenizer_identity, "exclusion_receipt": exclusions,
        "examples": examples, "ordered_ids_sha256": data.value_sha256([row["id"] for row in examples]),
        "examples_sha256": data.value_sha256(examples), "parity_row_id": examples[0]["id"],
        "source_audit": {name: {"path": "extension/fixture/" + name + ".json", "sha256": "1" * 64} for name in pins}}


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, use_cache=False):
        if labels is not None or use_cache:
            raise ValueError("The preserving hook must omit labels and disable cache")
        self.calls += 1
        return {"logits": self.head(self.embedding(input_ids).cumsum(dim=1).tanh())}


def synthetic_rows(count):
    result = []
    for index in range(count):
        response = [3 + index % 3] * (index % 3) + list(SUFFIX)
        tokens = [1 + index % 2, 2] + response
        identifier = hashlib.sha256(("synthetic pilot question " + str(index)).encode()).hexdigest()
        labels = [-100, -100] + response
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append({"id": identifier,
            "input_ids": tokens, "labels": [-100, -100] + response,
            "attention_mask": [1] * len(tokens), "row_index": index,
            "provenance": {"row_id": identifier, "prompt_tokens": 2, "total_tokens": len(tokens),
                "active_tokens": len(response), "eos_token_id": EOS, "native_terminal_suffix": list(SUFFIX),
                "full_token_ids_sha256": digest(tokens), "labels_sha256": digest(labels),
                "complete_original_target": True, "truncated": False, "target_rewritten": False,
                "causal_shift_applied": False}})
    return result


class ObservedDataset(Dataset):
    def __init__(self, rows):
        self.rows, self.reads = rows, []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads.append(index)
        return deepcopy(self.rows[index])


def collate(batch):
    if len(batch) != 1:
        raise ValueError("The driver requires microbatch one")
    row = batch[0]
    return {name: torch.tensor([row[name]], dtype=torch.int64)
        for name in ("input_ids", "labels", "attention_mask", "row_index")}


def training_args(path, count, **changes):
    value = dict(output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=math.ceil(count / 8),
        num_train_epochs=1, max_grad_norm=0., save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, remove_unused_columns=False, dataloader_pin_memory=False,
        dataloader_num_workers=0, label_smoothing_factor=0.)
    value.update(changes)
    return TrainingArguments(**value)


def manual_loss(model, row, *, mode="sft", reference=None, beta=.05):
    batch = collate([row])
    logp = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1].float().log_softmax(-1)
    target = batch["labels"][:, 1:]
    active = target != -100
    gold = logp.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)
    weight, kl = torch.ones_like(gold), torch.zeros_like(gold)
    if mode != "sft":
        with torch.no_grad():
            logq = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)["logits"][:, :-1].float().log_softmax(-1)
        if mode == "asft":
            weight = gold.detach().exp()
        elif mode == "prior_anchored":
            weight = logq.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1).exp()
        kl = (logq.exp() * (logq - logp)).sum(-1)
    return ((-weight * gold + beta * kl) * active).sum() / active.sum()


class PilotScheduleTests(unittest.TestCase):
    def test_full_and_pilot_exposures_are_exact_eight_example_groups(self):
        for rows, updates in ((15000, 1875), (2048, 256)):
            groups = module.update_groups(rows, 8)
            self.assertEqual(groups, [list(range(start, start + 8)) for start in range(0, rows, 8)])
            self.assertEqual(len(groups), updates)
            self.assertEqual([index for group in groups for index in group], list(range(rows)))

    def test_nine_and_seventeen_example_diagnostics_keep_single_example_tail(self):
        self.assertEqual(module.update_groups(9, 8), [list(range(8)), [8]])
        self.assertEqual(module.update_groups(17, 8), [list(range(8)), list(range(8, 16)), [16]])

    def test_prefix256_uses_full1875_cosine_not_a_shortened_pilot_schedule(self):
        def oracle(step):
            return step / 57 if step < 57 else .5 * (1 + math.cos(math.pi * (step - 57) / (1875 - 57)))
        for step in range(257):
            self.assertAlmostEqual(module.cosine_factor(step), oracle(step), places=14)
        self.assertEqual(module.cosine_factor(0), 0.)
        self.assertEqual(module.cosine_factor(57), 1.)
        self.assertEqual(module.cosine_factor(1875), 0.)
        self.assertGreater(module.cosine_factor(256), .9)
        shortened = .5 * (1 + math.cos(math.pi * (256 - 57) / (256 - 57)))
        self.assertEqual(shortened, 0.)
        self.assertNotEqual(module.cosine_factor(256), shortened)

    def test_actual_lambda_scheduler_lr_used_before_every_update_matches_independent_formula(self):
        parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
        optimizer = torch.optim.SGD([parameter], lr=2e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, module.cosine_factor)
        used = []
        for update in range(1, 257):
            used.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            index = update - 1
            expected = 2e-6 * (index / 57 if index < 57 else
                .5 * (1 + math.cos(math.pi * (index - 57) / (1875 - 57))))
            self.assertAlmostEqual(used[-1], expected, places=18)
        self.assertEqual(used[0], 0.)
        self.assertAlmostEqual(used[57], 2e-6, places=18)
        self.assertAlmostEqual(used[-1], 2e-6 * module.cosine_factor(255), places=18)

    def test_checkpoint_policy_is_diagnostic64_and_final_only(self):
        self.assertEqual(module.checkpoint_steps(256), [64, 256])
        self.assertEqual(module.checkpoint_steps(1875), [64, 1875])
        self.assertEqual(module.checkpoint_steps(2), [2])
        self.assertEqual(module.checkpoint_steps(64), [64])
        self.assertNotIn(128, module.checkpoint_steps(256))
        self.assertNotIn(192, module.checkpoint_steps(256))

    def test_invalid_group_and_schedule_configuration_is_not_silently_coerced(self):
        for count, accumulation in ((0, 8), (-1, 8), (9, 0), (9, -1), (9.5, 8), (9, True)):
            with self.subTest(count=count, accumulation=accumulation), self.assertRaises(ValueError):
                module.update_groups(count, accumulation)
        for step in (-1, .5, True):
            with self.assertRaises(ValueError):
                module.cosine_factor(step)


def complete_ledger(ledger, rows, *, lr=.05):
    for update, group in enumerate(module.update_groups(len(rows), 8), 1):
        for index in group:
            ledger.begin(index)
            ledger.backward_complete(index)
        ledger.before_update(update, [lr])
        ledger.after_update(update, [lr])
    return ledger.finalize()


class PilotConsumptionTests(unittest.TestCase):
    def test_receipt_counts_actual_complete_pairs_and_eos_with_single_example_tail(self):
        rows = synthetic_rows(9)
        receipt = complete_ledger(module.ConsumptionLedger(rows, 8), rows)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["examples"], 9)
        self.assertEqual(receipt["optimizer_updates"], 2)
        self.assertEqual(receipt["presentation_indices"], list(range(9)))
        self.assertEqual(receipt["presentation_ids"], [row["id"] for row in rows])
        self.assertEqual(receipt["total_tokens"], sum(len(row["input_ids"]) for row in rows))
        self.assertEqual(receipt["active_tokens"], sum(sum(target != -100 for target in row["labels"]) for row in rows))
        self.assertEqual(receipt["eos_tokens"], 9)
        self.assertEqual([item["microbatches"] for item in receipt["updates"]], [8, 1])
        self.assertEqual([item["indices"] for item in receipt["updates"]], [list(range(8)), [8]])
        self.assertTrue(all(item["backward_complete"] for item in receipt["presentations"]))
        self.assertEqual([item["full_token_ids_sha256"] for item in receipt["presentations"]],
            [row["provenance"]["full_token_ids_sha256"] for row in rows])

    def test_dataset_reads_and_prefetch_do_not_count_as_training_consumption(self):
        rows = synthetic_rows(9)
        dataset = ObservedDataset(rows)
        ledger = module.ConsumptionLedger(rows, 8)
        dataset[0]
        dataset[1]
        dataset[2]
        with self.assertRaises(ValueError):
            ledger.finalize()
        receipt = complete_ledger(ledger, rows)
        self.assertEqual(receipt["examples"], 9)
        self.assertEqual(receipt["presentation_ids"], [row["id"] for row in rows])

    def test_duplicate_out_of_order_or_uncompleted_backward_cannot_advance_update(self):
        rows = synthetic_rows(9)
        ledger = module.ConsumptionLedger(rows, 8)
        with self.assertRaises(ValueError):
            ledger.begin(1)
        ledger = module.ConsumptionLedger(rows, 8)
        ledger.begin(0)
        with self.assertRaises(ValueError):
            ledger.begin(0)
        with self.assertRaises(ValueError):
            ledger.before_update(1, [.05])
        with self.assertRaises(ValueError):
            ledger.backward_complete(1)

    def test_update_number_or_missing_optimizer_completion_cannot_publish_complete(self):
        rows = synthetic_rows(9)
        ledger = module.ConsumptionLedger(rows, 8)
        for index in range(8):
            ledger.begin(index)
            ledger.backward_complete(index)
        with self.assertRaises(ValueError):
            ledger.before_update(2, [.05])
        ledger.before_update(1, [.05])
        with self.assertRaises(ValueError):
            ledger.after_update(2, [.05])
        with self.assertRaises(ValueError):
            ledger.finalize()

    def test_nonfinite_observed_lr_is_rejected_before_update(self):
        for rate in (float("nan"), float("inf"), -.01):
            rows = synthetic_rows(9)
            ledger = module.ConsumptionLedger(rows, 8)
            for index in range(8):
                ledger.begin(index)
                ledger.backward_complete(index)
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                ledger.before_update(1, [rate])

    def test_learning_rates_before_and_after_update_are_actual_observations(self):
        rows = synthetic_rows(9)
        receipt = complete_ledger(module.ConsumptionLedger(rows, 8), rows, lr=.003)
        self.assertEqual([event["update"] for event in receipt["updates"]], [1, 2])
        self.assertEqual([event["learning_rates_before"] for event in receipt["updates"]], [[.003], [.003]])
        self.assertEqual([event["learning_rates_after"] for event in receipt["updates"]], [[.003], [.003]])


class PilotOutputTests(unittest.TestCase):
    def test_each_attempt_requires_an_exclusive_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "extension/runs/one-attempt"
            self.assertEqual(module.reserve_output(output), output.resolve())
            self.assertTrue(output.is_dir())
            with self.assertRaises((ValueError, FileExistsError)):
                module.reserve_output(output)

    def test_symlinked_attempts_cannot_reuse_an_existing_checkpoint_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "old-attempt"
            existing.mkdir()
            alias = Path(temporary) / "new-attempt"
            alias.symlink_to(existing, target_is_directory=True)
            with self.assertRaises((ValueError, FileExistsError)):
                module.reserve_output(alias)


class PilotPlanAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.plan_sha = data.value_sha256(self.plan)
        self.cache = make_cache(self.plan, self.plan_sha)

    def test_valid_fresh_diagnostic_and_full_horizon_pilot_envelopes(self):
        self.assertEqual(module.validate_plan(self.plan), self.plan)
        self.assertEqual(module.validate_cache(self.cache, self.plan, self.plan_sha), self.cache)
        pilot = make_plan(count=2048, stage="pilot")
        self.assertEqual(module.validate_plan(pilot), pilot)
        self.assertEqual(pilot["checkpoints"], [64, 256])

    def test_resume_wrong_precision_scope_and_shortened_horizon_are_rejected(self):
        changes = [lambda x: x.update(resume=True), lambda x: x.update(automatic_downstream=True),
            lambda x: x.update(parameter_dtype="bfloat16"), lambda x: x.update(reference_dtype="bfloat16"),
            lambda x: x.update(accumulation=2), lambda x: x.update(microbatch=2),
            lambda x: x["schedule"].update(total_steps=256), lambda x: x["schedule"].update(warmup_steps=8)]
        for change in changes:
            value = deepcopy(self.plan)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_exact_exposure_seed_and_nonfinite_hyperparameters_are_checked(self):
        for change in (lambda x: x["exposure"].update(examples=8), lambda x: x["exposure"].update(updates=3),
            lambda x: x.update(seed=True), lambda x: x.update(learning_rate=float("nan")),
            lambda x: x.update(beta=float("inf")), lambda x: x.update(objective="unbounded-new-objective")):
            value = deepcopy(self.plan)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_plan(value)

    def test_cache_binds_unchanged_original_text_complete_native_eos_and_mask(self):
        changes = [lambda x: x["examples"][0]["original_row"].update(output="rewritten answer"),
            lambda x: x["examples"][0]["input_ids"].__setitem__(-1, 199),
            lambda x: x["examples"][0]["labels"].__setitem__(0, 1),
            lambda x: x["examples"][0]["provenance"].update(truncated=True),
            lambda x: x["examples"][0]["provenance"].update(causal_shift_applied=True),
            lambda x: x["examples"][0]["provenance"].update(full_token_ids_sha256="0" * 64)]
        for change in changes:
            value = deepcopy(self.cache)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_cache(value, self.plan, self.plan_sha)

    def test_duplicate_reordered_or_stale_plan_bound_cache_is_rejected(self):
        changes = [lambda x: x.update(plan_sha256="0" * 64),
            lambda x: x["examples"].__setitem__(1, deepcopy(x["examples"][0])),
            lambda x: x["examples"].reverse(), lambda x: x.update(ordered_ids_sha256="0" * 64),
            lambda x: x.update(examples_sha256="0" * 64)]
        for change in changes:
            value = deepcopy(self.cache)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_cache(value, self.plan, self.plan_sha)

    def test_original_model_tokenizer_and_source_admission_pins_cannot_change(self):
        changes = [lambda x: x["base"].update(revision="main"),
            lambda x: x["base"]["files"].update({next(iter(module.ORIGINAL_FILES)): "0" * 64}),
            lambda x: x["tokenizer_identity"]["files"].update({next(iter(module.TOKENIZER_FILES)): "0" * 64}),
            lambda x: x["source_audit"].pop("native_model_admission"),
            lambda x: x["source_audit"]["benchmark_question_audit"].update(sha256="invalid")]
        for change in changes:
            value = deepcopy(self.cache)
            change(value)
            with self.assertRaises(ValueError):
                module.validate_cache(value, self.plan, self.plan_sha)

    def test_parity_question_is_prebound_on_cpu_instead_of_chosen_during_gpu_work(self):
        for identifier in (None, "0" * 64):
            value = deepcopy(self.cache)
            value["parity_row_id"] = identifier
            with self.assertRaises(ValueError):
                module.validate_cache(value, self.plan, self.plan_sha)

    def test_short_ga8_plus_long_before_first_adam_cannot_admit_joint_long_context_peak(self):
        plan = make_plan(count=2048, stage="pilot")
        codes = {name: "5" * 64 for name in module.REQUIRED_CODE}
        maximum = 31776
        old_evidence = {"schema": "extension-pilot-driver-complete-v1", "complete": True,
            "stage": "integration_diagnostic", "accumulation": 8, "code_sha256": codes,
            "shape_evidence": {"scope": "short_accumulation_tail_only", "maximum_complete_tokens": 512,
                "observations": []}, "long_context_joint_peak_proven": False,
            "old_long_case": {"tokens": maximum, "optimizer_updates_before_forward": 0,
                "student_parameters_with_adam_moments": 0, "resident_gradient_parameters_before_forward": 434}}
        with self.assertRaises(ValueError):
            module.validate_joint_shape_admission(old_evidence, plan=plan, code_sha256=codes, maximum_tokens=maximum)
        observation = {"tokens": maximum, "optimizer_updates_before_forward": 1,
            "student_parameters_with_adam_moments": 434, "resident_gradient_parameters_before_forward": 434,
            "successful_backwards_in_current_update": 1, "forward_complete": True, "backward_complete": True,
            "optimizer_update_after_backward_complete": True}
        # Synthetic positive schema oracle, not evidence of an actual allocation.
        valid = {"schema": "extension-pilot-long-after-adam-admission-v1", "complete": True,
            "accumulation": 8, "completed_optimizer_updates": 2, "code_sha256": codes,
            "base": {"repo_id": module.REPOSITORY, "revision": module.REVISION, "files": dict(module.ORIGINAL_FILES)},
            "versions": dict(module.VERSIONS), "gpu_identity": {"name": "NVIDIA B200"},
            "settings": {name: plan[name] for name in ("objective", "beta", "chunk_tokens", "checkpoint_chunks",
                "parameter_dtype", "reference_dtype", "compute_dtype", "attention")},
            "source_native_serialization_verified": True, "global_loss_unchanged": True,
            "finite_weights": True, "finite_moments": True, "reference_unchanged": True,
            "memory": {"headroom": {"passed": True}},
            "shape_evidence": {"scope": "long_after_adam_accumulation", "maximum_complete_tokens": maximum,
                "long_context_joint_peak_proven": True, "observations": [observation]}}
        module.validate_joint_shape_admission(valid, plan=plan, code_sha256=codes, maximum_tokens=maximum)
        for key, wrong in (("tokens", 512), ("optimizer_updates_before_forward", 0),
            ("student_parameters_with_adam_moments", 0), ("resident_gradient_parameters_before_forward", 0),
            ("successful_backwards_in_current_update", 0), ("successful_backwards_in_current_update", 8),
            ("forward_complete", False), ("backward_complete", False),
            ("optimizer_update_after_backward_complete", False)):
            value = deepcopy(valid)
            value["shape_evidence"]["observations"][0][key] = wrong
            if key == "tokens":
                value["shape_evidence"]["maximum_complete_tokens"] = wrong
            with self.subTest(missing=key), self.assertRaises(ValueError):
                module.validate_joint_shape_admission(value, plan=plan, code_sha256=codes, maximum_tokens=maximum)


class PilotEarlyBindingTests(unittest.TestCase):
    """Exercise actual small-file gates before any model path can be opened."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = make_plan()
        self.plan_pin = self.write("extension/fixture/plan.json", self.plan)
        self.cache = make_cache(self.plan, self.plan_pin["sha256"])
        self.cache_pin = self.write("extension/fixture/cache.json", self.cache)
        self.codes = {}
        for name in module.REQUIRED_CODE:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"small synthetic source fixture\n")
            self.codes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.qa = {"schema": "extension-pilot-driver-native-qa-v1", "success": True,
            "plan_sha256": self.plan_pin["sha256"], "cache_sha256": self.cache_pin["sha256"],
            "tests_run": 500, "failures": 0, "errors": 0, "skipped": 0, "gpu_disabled": True,
            "expected_failures": 0, "unexpected_successes": 0,
            "code_sha256": self.codes}
        self.qa_pin = self.write("extension/fixture/qa.json", self.qa)
        self.admission = {"schema": "extension-pilot-driver-admission-v1", "admitted": True, "job_id": "42",
            "plan_sha256": self.plan_pin["sha256"], "cache_sha256": self.cache_pin["sha256"],
            "qa_sha256": self.qa_pin["sha256"], "code_sha256": self.codes}
        self.write("extension/fixture/admission.json", self.admission)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def load(self):
        return module.load_inputs(self.root, "extension/fixture/plan.json", "extension/fixture/cache.json",
            "extension/fixture/qa.json", "extension/fixture/admission.json")

    def test_changed_source_code_fails_its_actual_byte_hash_before_model_access(self):
        path = self.root / "extension/pilot_driver.py"
        path.write_bytes(path.read_bytes() + b"changed\n")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.load()

    def test_seed_change_cannot_reuse_a_plan_cache_or_native_qa_binding(self):
        changed = deepcopy(self.plan)
        changed["seed"] += 1
        self.write("extension/fixture/plan.json", changed)
        with self.assertRaisesRegex(ValueError, "different plan"):
            self.load()

    def test_structurally_valid_new_cache_bytes_still_require_new_native_qa(self):
        changed = deepcopy(self.cache)
        changed["unbound_metadata"] = "mutated after CPU QA"
        self.write("extension/fixture/cache.json", changed)
        with self.assertRaisesRegex(ValueError, "CPU QA"):
            self.load()

    def test_reordered_cache_with_recomputed_value_hashes_cannot_reuse_cpu_qa(self):
        changed = deepcopy(self.cache)
        changed["examples"] = list(reversed(changed["examples"]))
        changed["ordered_ids_sha256"] = data.value_sha256([row["id"] for row in changed["examples"]])
        changed["examples_sha256"] = data.value_sha256(changed["examples"])
        self.write("extension/fixture/cache.json", changed)
        with self.assertRaisesRegex(ValueError, "CPU QA"):
            self.load()

    def test_old_341_qa_or_another_current_job_receipt_cannot_admit_driver(self):
        changed = deepcopy(self.qa)
        changed["schema"] = "extension-native-model-admission-native-qa-v1"
        self.write("extension/fixture/qa.json", changed)
        with self.assertRaisesRegex(ValueError, "CPU QA"):
            self.load()
        self.write("extension/fixture/qa.json", self.qa)
        changed = deepcopy(self.admission)
        changed["code_sha256"] = {}
        self.write("extension/fixture/admission.json", changed)
        with self.assertRaisesRegex(ValueError, "current-job"):
            self.load()


class FixtureEvidence:
    def __init__(self):
        self.files, self.fail, self.calls = {}, False, 0

    def unchanged(self):
        self.calls += 1
        if self.fail:
            raise ValueError("fixture input snapshot changed")


class PilotRunGuardsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "extension/runs/new-pilot-attempt"
        self.plan = make_plan()
        self.cache = make_cache(self.plan, data.value_sha256(self.plan))
        self.evidence = FixtureEvidence()
        self.context = {"root": self.root, "plan": self.plan, "cache": self.cache,
            "admission": {"job_id": "42"}, "qa": {"success": True}, "evidence": self.evidence,
            "plan_sha256": data.value_sha256(self.plan), "cache_sha256": data.value_sha256(self.cache),
            "binding_sha256": "6" * 64, "code_sha256": {name: "5" * 64 for name in module.REQUIRED_CODE}}
        self.calls = 0
        for name, value in (("require_gpu_allocation", lambda: {"job_id": "42", "node": "owl-sr001"}),
            ("load_inputs", lambda *args, **kwargs: self.context), ("deadline_guard", lambda *args, **kwargs: None)):
            hook = patch.object(module, name, value)
            hook.start()
            self.addCleanup(hook.stop)
        disk = patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=200 << 30))
        disk.start()
        self.addCleanup(disk.stop)

    def execute(self, context, output, allocation):
        self.calls += 1
        directory = output / "checkpoints/step-2"
        directory.mkdir(parents=True)
        files = {"checkpoints/step-2/config.json": b'{"synthetic":true}',
            "checkpoints/step-2/model.safetensors": b"fixture bytes; no actual model"}
        for name, content in files.items():
            (output / name).write_bytes(content)
        native_rows = context["cache"]["examples"]
        ledger = module.ConsumptionLedger(native_rows, 8)
        for update, group in enumerate(module.update_groups(len(native_rows), 8), 1):
            for index in group:
                ledger.begin(index)
                ledger.backward_complete(index)
            rates = [context["plan"]["learning_rate"] * module.cosine_factor(update - 1)]
            ledger.before_update(update, rates)
            ledger.after_update(update, rates)
        consumption = ledger.finalize()
        files["consumption.json"] = json.dumps(consumption, sort_keys=True).encode()
        files["presentations.jsonl"] = b"".join(json.dumps(event, sort_keys=True).encode() + b"\n"
            for event in consumption["presentations"])
        for name in ("consumption.json", "presentations.jsonl"):
            (output / name).write_bytes(files[name])
        hashes = {name: "7" * 64 for name in ("python", "numpy", "torch_cpu", "torch_cuda")}
        return {"schema": "extension-pilot-driver-complete-v1", "complete": True,
            "stage": context["plan"]["stage"], "binding_sha256": context["binding_sha256"],
            "plan_sha256": context["plan_sha256"], "cache_sha256": context["cache_sha256"],
            "accumulation": 8, "code_sha256": context["code_sha256"],
            "checks": {name: True for name in module.REQUIRED_CHECKS}, "consumption": consumption,
            "checkpoints": {"2": {"directory": str(directory)}},
            "rng": {"seed": context["plan"]["seed"], "resume_supported": False,
                "before_model_initialization": dict(hashes), "before_train": dict(hashes), "after_train": dict(hashes)},
            "lr_used": [[context["plan"]["learning_rate"] * module.cosine_factor(index)] for index in range(2)],
            "lr_after_scheduler": [[context["plan"]["learning_rate"] * module.cosine_factor(index + 1)] for index in range(2)],
            "output_files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}

    def run_driver(self, execute=None, output=None):
        return module.run(self.root, "extension/plan.json", "extension/cache.json", output or self.output,
            "extension/qa.json", "extension/admission.json", execute=execute or self.execute)

    def test_one_fresh_attempt_validates_receipt_before_publishing_complete(self):
        result = self.run_driver()
        self.assertTrue(result["complete"])
        self.assertEqual(self.calls, 1)
        self.assertTrue((self.output / "complete.json").is_file())
        self.assertEqual(result["allocation_end"]["job_id"], "42")
        self.assertGreater(self.evidence.calls, 0)

    def test_reusing_output_never_invokes_training_again(self):
        self.run_driver()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_driver()
        self.assertEqual(self.calls, 1)

    def test_failed_training_preserves_failure_and_never_retries_or_publishes_complete(self):
        def failed(*args):
            self.calls += 1
            raise RuntimeError("synthetic optimizer failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic optimizer failure"):
            self.run_driver(failed)
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.output / "complete.json").exists())
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertFalse(failure["retry"])
        self.assertFalse(failure["resume"])
        self.assertFalse(failure["automatic_downstream"])

    def test_cpu_allocation_or_another_current_job_is_rejected_before_training(self):
        with patch.object(module, "require_gpu_allocation", side_effect=ValueError("no assigned GPU")):
            with self.assertRaises(ValueError):
                self.run_driver()
        self.context["admission"]["job_id"] = "43"
        with self.assertRaises(ValueError):
            self.run_driver()
        self.assertEqual(self.calls, 0)

    def test_insufficient_scratch_and_foreign_output_do_not_invoke_execution(self):
        with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=159 << 30)):
            with self.assertRaises(ValueError):
                self.run_driver()
        with self.assertRaises(ValueError):
            self.run_driver(output=self.root / "outside-stage")
        self.assertEqual(self.calls, 0)

    def test_input_snapshot_change_after_training_blocks_complete(self):
        self.evidence.fail = True
        with self.assertRaisesRegex(ValueError, "input snapshot changed"):
            self.run_driver()
        self.assertFalse((self.output / "complete.json").exists())
        self.assertTrue((self.output / "failure.json").is_file())

    def test_code_cache_plan_order_and_rng_identity_tampering_is_rejected(self):
        changes = [lambda x: x.update(code_sha256={}), lambda x: x.update(cache_sha256="0" * 64),
            lambda x: x.update(plan_sha256="0" * 64),
            lambda x: x["consumption"].update(ordered_ids_sha256="0" * 64),
            lambda x: x["rng"].update(seed=self.plan["seed"] + 1),
            lambda x: x["rng"]["before_model_initialization"].pop("torch_cuda"),
            lambda x: x["rng"]["after_train"].update(torch_cpu="invalid")]
        self.output.mkdir(parents=True)
        receipt = self.execute(self.context, self.output, {"job_id": "42"})
        for change in changes:
            value = deepcopy(receipt)
            change(value)
            with self.assertRaises(ValueError):
                module._validate_complete(value, self.context, self.output)

    def test_wrong_lr_vector_checkpoint_boundary_or_missing_check_blocks_complete(self):
        self.output.mkdir(parents=True)
        receipt = self.execute(self.context, self.output, {"job_id": "42"})
        for change in (lambda x: x["lr_used"].__setitem__(0, [self.plan["learning_rate"]]),
            lambda x: x["lr_after_scheduler"].pop(), lambda x: x["checkpoints"].update({"64": {}}),
            lambda x: x["checks"].pop(next(iter(module.REQUIRED_CHECKS)))):
            value = deepcopy(receipt)
            change(value)
            with self.assertRaises(ValueError):
                module._validate_complete(value, self.context, self.output)

    def test_saved_file_hash_or_escaping_path_cannot_be_accepted(self):
        self.output.mkdir(parents=True)
        receipt = self.execute(self.context, self.output, {"job_id": "42"})
        path = self.output / "checkpoints/step-2/model.safetensors"
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaises(ValueError):
            module._validate_complete(receipt, self.context, self.output)
        receipt["output_files"] = {"../outside.safetensors": "a" * 64}
        with self.assertRaises(ValueError):
            module._validate_complete(receipt, self.context, self.output)


class PilotNativeTrainerTests(unittest.TestCase):
    def train_pair(self, count, *, mode="sft", learning_rate=.05):
        torch.manual_seed(20261021)
        initial = TinyCausal()
        actual, oracle = deepcopy(initial), deepcopy(initial)
        reference = None if mode == "sft" else deepcopy(initial)
        reference_before = None if reference is None else {name: value.detach().clone() for name, value in reference.state_dict().items()}
        rows = synthetic_rows(count)
        oracle_optimizer = torch.optim.SGD(oracle.parameters(), lr=learning_rate)
        for group in module.update_groups(count, 8):
            oracle_optimizer.zero_grad()
            for index in group:
                (manual_loss(oracle, rows[index], mode=mode, reference=reference) / len(group)).backward()
            oracle_optimizer.step()
        ledger = module.ConsumptionLedger(rows, 8)
        dataset = ObservedDataset(rows)
        trainer_type = module.make_pilot_trainer_class()
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = torch.optim.SGD(actual.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            trainer = trainer_type(model=actual, args=training_args(temporary, count),
                objective=mode, reference_model=reference, beta=.05, eos_token_id=EOS,
                terminal_suffix=SUFFIX, chunk_tokens=2, checkpoint_chunks=True,
                consumption=ledger, train_dataset=dataset, data_collator=collate,
                optimizers=(optimizer, scheduler))
            trainer.train()
            self.assertEqual(trainer.state.global_step, math.ceil(count / 8))
            self.assertEqual(trainer.training_forward_examples, count)
            self.assertEqual(actual.calls, count)
            receipt = ledger.finalize()
            self.assertEqual(receipt["presentation_indices"], list(range(count)))
            self.assertEqual(receipt["examples"], count)
            self.assertGreaterEqual(len(dataset.reads), count)
            for (name, parameter), (_, expected) in zip(actual.named_parameters(), oracle.named_parameters()):
                with self.subTest(parameter=name, mode=mode, count=count):
                    torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-6)
            if reference is not None:
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(value, reference_before[name], atol=0, rtol=0)
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters()))
            self.assertTrue(all(parameter.dtype == torch.float32 and bool(torch.isfinite(parameter).all()) for parameter in actual.parameters()))
        return receipt

    def test_actual_hf_ga8_nine_examples_has_exactly_two_equal_example_updates(self):
        receipt = self.train_pair(9)
        self.assertEqual([item["microbatches"] for item in receipt["updates"]], [8, 1])

    def test_actual_hf_ga8_seventeen_examples_has_three_updates_with_correct_final_scale(self):
        receipt = self.train_pair(17)
        self.assertEqual([item["microbatches"] for item in receipt["updates"]], [8, 8, 1])

    def test_all_reference_objectives_preserve_tail_scale_and_frozen_original(self):
        for mode in ("ce_kl", "asft", "prior_anchored"):
            with self.subTest(mode=mode):
                receipt = self.train_pair(9, mode=mode)
                self.assertEqual(receipt["active_tokens"], sum(len(row["labels"]) - 2 for row in synthetic_rows(9)))

    def test_actual_batch_mutation_cannot_be_reported_as_expected_cache_consumption(self):
        rows = synthetic_rows(9)
        ledger = module.ConsumptionLedger(rows, 8)
        def changed_collate(batch):
            value = collate(batch)
            if int(value["row_index"].item()) == 0:
                value["input_ids"][0, 0] = 4
            return value
        with tempfile.TemporaryDirectory() as temporary:
            trainer = module.make_pilot_trainer_class()(model=TinyCausal(), args=training_args(temporary, 9),
                objective="sft", eos_token_id=EOS, terminal_suffix=SUFFIX, consumption=ledger,
                train_dataset=ObservedDataset(rows), data_collator=changed_collate)
            with self.assertRaises(ValueError):
                trainer.train()
            with self.assertRaises(ValueError):
                ledger.finalize()

    def test_resume_and_repeated_train_calls_cannot_repeat_the_same_exposure(self):
        rows = synthetic_rows(9)
        with tempfile.TemporaryDirectory() as temporary:
            trainer = module.make_pilot_trainer_class()(model=TinyCausal(), args=training_args(temporary, 9),
                objective="sft", eos_token_id=EOS, terminal_suffix=SUFFIX,
                consumption=module.ConsumptionLedger(rows, 8), train_dataset=ObservedDataset(rows), data_collator=collate)
            with self.assertRaises(ValueError):
                trainer.train(resume_from_checkpoint="checkpoint-64")
            trainer.train()
            with self.assertRaises(ValueError):
                trainer.train()

    def test_native_adam_moments_and_saved_checkpoint_contain_only_finite_fp32_student(self):
        from extension.preserving_trainer import check_optimizer
        from safetensors import safe_open
        torch.manual_seed(7)
        student = TinyCausal()
        reference = deepcopy(student)
        before = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
        rows = synthetic_rows(9)
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            trainer = module.make_pilot_trainer_class()(model=student, args=training_args(temporary, 9),
                objective="ce_kl", reference_model=reference, beta=.05, eos_token_id=EOS,
                terminal_suffix=SUFFIX, consumption=module.ConsumptionLedger(rows, 8),
                train_dataset=ObservedDataset(rows), data_collator=collate, optimizers=(optimizer, scheduler))
            trainer.train()
            check_optimizer(student, trainer.optimizer, reference)
            self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
                {id(parameter) for parameter in student.parameters()})
            self.assertFalse({id(parameter) for group in optimizer.param_groups for parameter in group["params"]} &
                {id(parameter) for parameter in reference.parameters()})
            for state in optimizer.state.values():
                for name in ("exp_avg", "exp_avg_sq"):
                    self.assertEqual(state[name].dtype, torch.float32)
                    self.assertTrue(bool(torch.isfinite(state[name]).all()))
            checkpoint = Path(temporary) / "saved-student"
            trainer.save_model(str(checkpoint))
            with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
                self.assertEqual(set(handle.keys()), set(student.state_dict()))
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(bool(torch.isfinite(tensor).all()))
                    torch.testing.assert_close(tensor, student.state_dict()[name], atol=0, rtol=0)
            for name, tensor in reference.state_dict().items():
                torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)

    def test_nonfinite_backward_never_becomes_a_completed_optimizer_update(self):
        class CountingAdam(torch.optim.AdamW):
            def __init__(self, parameters):
                super().__init__(parameters, lr=.001)
                self.calls = 0

            def step(self, closure=None):
                self.calls += 1
                return super().step(closure)

        rows = synthetic_rows(9)
        student = TinyCausal()
        before = {name: value.clone() for name, value in student.state_dict().items()}
        ledger = module.ConsumptionLedger(rows, 8)
        optimizer = CountingAdam(student.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        handle = next(student.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
        try:
            with tempfile.TemporaryDirectory() as temporary:
                trainer = module.make_pilot_trainer_class()(model=student, args=training_args(temporary, 9),
                    objective="sft", eos_token_id=EOS, terminal_suffix=SUFFIX, consumption=ledger,
                    train_dataset=ObservedDataset(rows), data_collator=collate, optimizers=(optimizer, scheduler))
                with self.assertRaises((ValueError, RuntimeError)):
                    trainer.train()
                self.assertEqual(optimizer.calls, 0)
                with self.assertRaises(ValueError):
                    ledger.finalize()
        finally:
            handle.remove()
        for name, value in student.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_reference_parameters_cannot_enter_the_optimizer(self):
        student, reference = TinyCausal(), TinyCausal()
        rows = synthetic_rows(9)
        with tempfile.TemporaryDirectory() as temporary:
            wrong_optimizer = torch.optim.AdamW(list(student.parameters()) + list(reference.parameters()), lr=.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(wrong_optimizer, lambda _: 1.)
            with self.assertRaises(ValueError):
                module.make_pilot_trainer_class()(model=student, args=training_args(temporary, 9),
                    objective="ce_kl", reference_model=reference, eos_token_id=EOS, terminal_suffix=SUFFIX,
                    consumption=module.ConsumptionLedger(rows, 8), train_dataset=ObservedDataset(rows),
                    data_collator=collate, optimizers=(wrong_optimizer, scheduler))




