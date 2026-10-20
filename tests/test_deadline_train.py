import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from deadline.train_full import Consumption, check_fp32_state, digest, fixed_order, resume_receipt, trainer_class, update_schedule, validate_cache


class FullTrainingGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [dict(id=f'id-{i}', question_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                         fold=i % 2, input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4]) for i in range(4)]

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, rows=None, **changes):
        cache, meta = self.root / 'cache.jsonl', self.root / 'cache-meta.json'
        cache.write_text(''.join(json.dumps(r) + '\n' for r in (self.rows if rows is None else rows)))
        values = dict(cache_sha256=digest(cache), rows=4, cutoff=4,
                      dataset_role='advanced', truncation_policy='none', truncated_rows=0)
        values.update(changes)
        meta.write_text(json.dumps(values))
        return cache, meta

    def validate(self, rows=None, **changes):
        return validate_cache(*self.fixture(rows, **changes), expected_rows=4, cutoff=4)

    def test_complete_cache_and_masks(self):
        rows, meta, receipt = self.validate()
        self.assertEqual(len(rows), 4)
        self.assertEqual(receipt['supervised_tokens'], 8)
        self.assertEqual(receipt['truncated_rows'], 0)

    def test_no_silent_row_drop_or_truncation(self):
        with self.assertRaisesRegex(ValueError, 'expected 4 rows'):
            self.validate(self.rows[:3])
        rows = copy.deepcopy(self.rows)
        rows[0]['input_ids'].append(5)
        rows[0]['labels'].append(5)
        with self.assertRaisesRegex(ValueError, 'no implicit truncation'):
            self.validate(rows)

    def test_duplicate_question_or_id_rejected(self):
        for field in ('id', 'question_sha256'):
            rows = copy.deepcopy(self.rows)
            rows[1][field] = rows[0][field]
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_incorrect_prompt_mask_or_target_token_rejected(self):
        for labels in ([1, -100, 3, 4], [-100, 2, -100, 4], [-100, -100, 9, 4], [-100] * 4):
            rows = copy.deepcopy(self.rows)
            rows[0]['labels'] = labels
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_cache_hash_and_declared_count_required(self):
        with self.assertRaisesRegex(ValueError, 'cache hash'):
            self.validate(cache_sha256='different')
        cache, meta = self.fixture()
        values = json.loads(meta.read_text())
        values['rows'] = 3
        meta.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, 'row count'):
            validate_cache(cache, meta, expected_rows=4, cutoff=4)

    def test_advanced_truncation_forbidden_random_declared(self):
        with self.assertRaisesRegex(ValueError, 'advanced cache'):
            self.validate(truncation_policy='prefix_truncate', truncated_rows=1)
        rows, _, receipt = self.validate(dataset_role='random', truncation_policy='prefix_truncate', truncated_rows=1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(receipt['truncated_rows'], 1)

    def test_frozen_8k_default_accepts_complete_targets_above_4k(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['input_ids'] = list(range(5000))
        rows[0]['labels'] = [-100, -100] + rows[0]['input_ids'][2:]
        cache, meta = self.fixture(rows, cutoff=8192)
        audited, _, receipt = validate_cache(cache, meta, expected_rows=4)
        self.assertEqual(len(audited), 4)
        self.assertEqual(receipt['max_length'], 5000)
        self.assertEqual(receipt['truncated_rows'], 0)
        values = json.loads(meta.read_text())
        values['cutoff'] = 4096
        meta.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, 'cutoff differs'):
            validate_cache(cache, meta, expected_rows=4)

    def test_exact_one_pass_schedule_and_explicit_no_padding(self):
        schedule = update_schedule(15000, 8, 1875)
        self.assertEqual(schedule['padding_presentations'], 0)
        self.assertEqual(schedule['final_update_examples'], 8)
        self.assertEqual(update_schedule(5, 2, 3)['final_update_examples'], 1)
        with self.assertRaisesRegex(ValueError, 'one pass'):
            update_schedule(15000, 8, 2000)

    def test_actual_consumption_is_exact_and_rejects_repeat(self):
        order = fixed_order(4)
        consumed = Consumption(order)
        for index in order:
            consumed.consume(index)
        consumed.verify_complete()
        self.assertEqual(consumed.receipt(2, 2)['unique_indices'], 4)
        with self.assertRaisesRegex(ValueError, 'extra presentation'):
            consumed.consume(order[0])

    def test_incomplete_consumption_and_bad_resume_order_rejected(self):
        order = fixed_order(4)
        consumed = Consumption(order)
        consumed.consume(order[0])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            consumed.verify_complete()
        with self.assertRaisesRegex(ValueError, 'optimizer updates'):
            consumed.receipt(1, 2)
        with self.assertRaisesRegex(ValueError, 'resume consumption'):
            Consumption(order, [order[-1]])

    def test_resume_requires_complete_bound_checkpoint(self):
        checkpoint = self.root / 'checkpoint-1'
        checkpoint.mkdir()
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            (checkpoint / name).write_bytes(b'fixture')
        (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=1)))
        order = fixed_order(4)
        consumed = Consumption(order, order[:2])
        receipt = dict(consumed.receipt(1, 2), binding_sha256='binding')
        (checkpoint / 'consumption.json').write_text(json.dumps(receipt))
        restored, step = resume_receipt(checkpoint, 'binding', 4, 2, order)
        self.assertEqual(step, 1)
        self.assertEqual(restored.indices, order[:2])
        with self.assertRaisesRegex(ValueError, 'another data/model/recipe'):
            resume_receipt(checkpoint, 'changed', 4, 2, order)
        (checkpoint / 'optimizer.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'optimizer.pt'):
            resume_receipt(checkpoint, 'binding', 4, 2, order)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is unavailable locally')
class ChunkedLossReferenceTest(unittest.TestCase):
    def test_memory_ce_matches_shifted_masked_cross_entropy_and_gradient(self):
        import torch
        import torch.nn.functional as functional
        from reasoning.memory_ce import causal_ce
        torch.manual_seed(4)
        for labels in (torch.tensor([[-100, -100, 2, 3, 4]]), torch.tensor([[-100, 1, 2, 3, 4]])):
            actual = torch.randn(1, 5, 7, requires_grad=True)
            expected = actual.detach().clone().requires_grad_(True)
            loss = causal_ce(actual, labels, 7)
            reference = functional.cross_entropy(expected[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            reference.backward()
            torch.testing.assert_close(loss, reference)
            torch.testing.assert_close(actual.grad, expected.grad)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is unavailable locally')
class FP32MasterPrecisionTest(unittest.TestCase):
    def test_sub_bf16_ulp_adam_update_survives_fp32_storage(self):
        import torch
        fp32 = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float32))
        bf16 = torch.nn.Parameter(torch.tensor(.1, dtype=torch.bfloat16))
        before_fp32, before_bf16 = fp32.item(), bf16.item()
        for parameter in (fp32, bf16):
            optimizer = torch.optim.AdamW([parameter], lr=2e-6, weight_decay=.01)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        update = before_fp32 - fp32.item()
        self.assertGreater(update, 1e-6)
        self.assertLess(update, 5e-6)
        self.assertEqual(before_bf16, bf16.item())

    def test_runtime_precision_guard_rejects_bf16_parameters_and_moments(self):
        import torch
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
        check_fp32_state(model, optimizer)
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        receipt = check_fp32_state(model, optimizer, require_moments=True)
        self.assertGreater(receipt['fp32_moment_tensors'], 0)
        first = next(iter(optimizer.state.values()))
        first['exp_avg'] = first['exp_avg'].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Adam exp_avg'):
            check_fp32_state(model, optimizer)
        model.to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'trainable parameter'):
            check_fp32_state(model)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'native PyTorch/Transformers are unavailable locally')
class NativeTrainerAccumulationTest(unittest.TestCase):
    def test_native_accumulation_averages_examples_not_target_tokens(self):
        import torch
        from transformers import Trainer, TrainingArguments
        from types import SimpleNamespace

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, input_ids, labels, attention_mask=None, use_cache=False):
                # Unequal target counts: gradients are 1 and 3, whose example
                # mean is 2. A token-weighted denominator would give 2.5.
                return SimpleNamespace(loss=self.weight * labels[labels != -100].float().mean())

        rows = [dict(input_ids=[1, 1], labels=[-100, 1], row_index=0),
                dict(input_ids=[1, 1, 1, 1], labels=[-100, 3, 3, 3], row_index=1)]

        def collate(batch):
            row = batch[0]
            return {key: torch.tensor([value]) for key, value in row.items()}

        consumption = Consumption([0, 1])
        model = Toy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        with tempfile.TemporaryDirectory() as output:
            args = TrainingArguments(output_dir=output, use_cpu=True, per_device_train_batch_size=1,
                gradient_accumulation_steps=2, max_steps=1, max_grad_norm=0.,
                save_strategy='no', logging_strategy='no', report_to=[], disable_tqdm=True,
                remove_unused_columns=False, dataloader_pin_memory=False)
            native = trainer_class(Trainer, torch, consumption, [0, 1])(
                model=model, args=args, train_dataset=rows, data_collator=collate,
                optimizers=(optimizer, scheduler))
            native.model_accepts_loss_kwargs = False
            native.train()
        consumption.verify_complete()
        torch.testing.assert_close(model.weight.detach(), torch.tensor(.8))


if __name__ == '__main__':
    unittest.main()
