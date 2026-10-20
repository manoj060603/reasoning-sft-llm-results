"""Cross-fit leakage, native-cache masks, resume integrity and CE oracle checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

from deadline.transfer_probe import (
    SCHEMA, ScoreJournal, TokenCache, batch_sequence_ce, check_disjoint,
    digest, model_identity, read_ids, transfer_gain, _finite_adapter,
    _probe_parameter_precision, _adam_moment_precision,
)


def question(name):
    return hashlib.sha256(name.encode()).hexdigest()


def cache_row(name, fold):
    return {"id": name, "fold": fold, "question_sha256": question(name),
            "input_ids": [10, 11, 12, 13], "labels": [-100, -100, 12, 13]}


class CacheFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.cache_path = self.root / "cache.jsonl"
        self.meta_path = self.root / "cache_meta.json"
        self.rows = [cache_row("a", 0), cache_row("b", 1), cache_row("c", 1)]
        self.write_cache()

    def tearDown(self):
        self.temporary.cleanup()

    def write_cache(self):
        self.cache_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.meta_path.write_text(json.dumps({"cache_sha256": digest(self.cache_path),
                                            "model_path": str(self.base), "cutoff": 4096,
                                            "complete_original_targets": True,
                                            "prefix_truncated_count": 0,
                                            "sourcehash": "sealed-test-source"}))

    def cache(self):
        return TokenCache(self.cache_path, self.meta_path, self.base, cutoff=4096)

    def receipt(self):
        return {"schema": SCHEMA, "complete": True, "training_fold": 0,
                "training_ids": ["a"], "training_question_sha256": [question("a")]}

    def test_opposite_fold_allows_unseen_originals(self):
        cache = self.cache()
        check_disjoint(cache, ["b", "c"], 1, self.receipt())
        self.assertEqual(cache.get("b", {13}, 20)["labels"], [-100, -100, 12, 13])

    def test_same_fold_and_false_candidate_fold_fail(self):
        with self.assertRaisesRegex(ValueError, "same-fold"):
            check_disjoint(self.cache(), ["a"], 0, self.receipt())
        with self.assertRaisesRegex(ValueError, "declared source fold"):
            check_disjoint(self.cache(), ["a"], 1, self.receipt())

    def test_forged_training_fold_or_question_manifest_fails(self):
        receipt = self.receipt()
        receipt["training_fold"] = 1
        with self.assertRaises(ValueError):
            check_disjoint(self.cache(), ["a"], 0, receipt)
        receipt = self.receipt()
        receipt["training_question_sha256"] = [question("b")]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            check_disjoint(self.cache(), ["b"], 1, receipt)

    def test_incomplete_probe_cannot_score(self):
        receipt = self.receipt()
        receipt["complete"] = False
        with self.assertRaisesRegex(ValueError, "completed training"):
            check_disjoint(self.cache(), ["b"], 1, receipt)

    def test_duplicate_question_across_folds_fails(self):
        self.rows[1]["question_sha256"] = self.rows[0]["question_sha256"]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "duplicate question"):
            self.cache()

    def test_wrong_eos_or_gold_alignment_fails(self):
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.cache().get("a", {99}, 100)
        self.rows[0]["labels"][-1] = 14
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "preserve tokens"):
            self.cache().get("a")

    def test_native_qwen_eos_then_newline_is_preserved(self):
        self.rows[0]["input_ids"] = [10, 11, 12, 151645, 198]
        self.rows[0]["labels"] = [-100, -100, 12, 151645, 198]
        self.write_cache()
        original = self.cache().get("a", {151645}, 151936, (198,))
        self.assertEqual(original["labels"][-2:], [151645, 198])
        self.assertEqual(len(original["input_ids"]), 5)
        self.rows[0]["input_ids"][-1] = self.rows[0]["labels"][-1] = 199
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "EOS/template trailer"):
            self.cache().get("a", {151645}, 151936, (198,))

    def test_response_mask_hole_and_unmasked_prompt_fail(self):
        self.rows[0]["labels"] = [-100, 11, -100, 13]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "masked hole"):
            self.cache().get("a")
        self.rows[0]["labels"] = [10, 11, 12, 13]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "masked prompt"):
            self.cache().get("a")

    def test_source_or_cache_mutation_does_not_reuse_receipts(self):
        cache = self.cache()
        with self.cache_path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            cache.assert_unchanged()
        self.write_cache()
        self.rows[1]["input_ids"][0] = 5
        self.cache_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.cache()

    def test_overlength_cannot_be_silently_truncated(self):
        self.rows[0]["input_ids"] = [10] * 4097
        self.rows[0]["labels"] = [-100] + [10] * 4096
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "overlength"):
            self.cache().get("a")

    def test_deadline_8k_cache_preserves_long_complete_examples(self):
        self.rows[0]["input_ids"] = [10] * 8191 + [13]
        self.rows[0]["labels"] = [-100] + [10] * 8190 + [13]
        self.write_cache()
        metadata = json.loads(self.meta_path.read_text())
        metadata["cutoff"] = 8192
        self.meta_path.write_text(json.dumps(metadata))
        cache = TokenCache(self.cache_path, self.meta_path, self.base)
        self.assertEqual(cache.provenance()["cutoff"], 8192)
        self.assertEqual(len(cache.get("a", {13}, 20)["input_ids"]), 8192)
        self.rows[0]["input_ids"].insert(1, 10)
        self.rows[0]["labels"].insert(1, 10)
        self.write_cache()
        metadata = json.loads(self.meta_path.read_text())
        metadata["cutoff"] = 8192
        self.meta_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "overlength"):
            TokenCache(self.cache_path, self.meta_path, self.base).get("a")

    def test_old_4k_cache_requires_explicit_cutoff(self):
        with self.assertRaisesRegex(ValueError, "8192-token cache"):
            TokenCache(self.cache_path, self.meta_path, self.base)
        self.assertEqual(self.cache().cutoff, 4096)

    def test_empty_or_duplicate_id_manifest_fails(self):
        path = self.root / "ids.json"
        for ids in ([], ["a", "a"], [1]):
            path.write_text(json.dumps(ids))
            with self.assertRaises(ValueError):
                read_ids(path)


class GainTests(unittest.TestCase):
    def test_keep_negative_acquisition_observations(self):
        self.assertEqual(transfer_gain(2.0, 1.0), 0.5)
        self.assertEqual(transfer_gain(2.0, 3.0), -0.5)
        self.assertEqual(transfer_gain(0.0, 0.0), 0.0)
        self.assertAlmostEqual(transfer_gain(0.0, 1e-8), -1.0)

    def test_nonfinite_or_negative_loss_is_fatal(self):
        for base, probe in ((math.inf, 0), (0, math.nan), (-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                transfer_gain(base, probe)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {"schema": SCHEMA, "scoring_fold": 1, "probe_id": "A"}
        self.batches = [["b"], ["c"]]

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, key):
        return {"id": key, "fold": 1, "probe_id": "A", "base_ce": 2.0,
                "probe_ce": 3.0, "gain": -0.5, "supervised_tokens": 2}

    def journal(self):
        return ScoreJournal(self.root / "scores", self.manifest, self.batches)

    def test_interruption_resumes_only_validated_completed_batches(self):
        journal = self.journal()
        journal.commit(0, [self.record("b")])
        self.assertFalse((journal.output / "score_receipt.json").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            journal.finish({})
        self.assertFalse((journal.output / "scores.jsonl").exists())
        resumed = self.journal()
        self.assertEqual(resumed.read(0), [self.record("b")])
        self.assertIsNone(resumed.read(1))
        resumed.commit(1, [self.record("c")])
        receipt = resumed.finish({"first_invocation": True})
        self.assertEqual(receipt["records"], 2)
        self.assertEqual(len((resumed.output / "scores.jsonl").read_text().splitlines()), 2)
        self.assertEqual(resumed.finish({"overwrite": "forbidden"}), receipt)

    def test_changed_configuration_or_batch_values_do_not_overwrite(self):
        journal = self.journal()
        journal.commit(0, [self.record("b")])
        changed = {**self.manifest, "probe_id": "B"}
        with self.assertRaisesRegex(ValueError, "incompatible provenance"):
            ScoreJournal(journal.output, changed, self.batches)
        wrong = self.record("b")
        wrong["probe_ce"], wrong["gain"] = 1.0, 0.5
        with self.assertRaisesRegex(ValueError, "overwrite"):
            journal.commit(0, [wrong])
        self.assertEqual(journal.read(0), [self.record("b")])

    def test_corrupted_partial_output_is_rejected(self):
        journal = self.journal()
        journal.commit(0, [self.record("b")])
        part = json.loads(journal.part_path(0).read_text())
        part["records"][0]["probe_ce"] = 0.0
        journal.part_path(0).write_text(json.dumps(part))
        with self.assertRaisesRegex(ValueError, "checksum"):
            journal.read(0)

    def test_completed_output_cannot_be_relabelled_or_modified(self):
        journal = self.journal()
        for index, key in enumerate(("b", "c")):
            journal.commit(index, [self.record(key)])
        journal.finish({})
        with (journal.output / "scores.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "output changed"):
            journal.completed()

    def test_wrong_fold_count_or_gain_cannot_be_committed(self):
        journal = self.journal()
        for field, value in (("fold", 0), ("supervised_tokens", 0), ("gain", 0.0)):
            record = self.record("b")
            record[field] = value
            with self.assertRaises(ValueError):
                journal.commit(0, [record])
        self.assertIsNone(journal.read(0))


class ModelIdentityTests(unittest.TestCase):
    def test_local_weight_changes_change_exact_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
            (root / "model.safetensors").write_bytes(b"fake-weights-for-hash-only")
            first = model_identity(root)
            (root / "model.safetensors").write_bytes(b"different-weights")
            self.assertNotEqual(first["identity_sha256"], model_identity(root)["identity_sha256"])
            (root / "config.json").write_text(json.dumps({"model_type": "other"}))
            with self.assertRaisesRegex(ValueError, "Qwen2"):
                model_identity(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "local torch is unavailable")
class TinyTorchCEOracle(unittest.TestCase):
    def test_probe_storage_and_adam_moments_are_explicit_fp32(self):
        import torch

        class StorageFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
                self.adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

        model = StorageFixture()
        policy = _probe_parameter_precision(model)
        self.assertEqual(policy["adapter_parameter_storage"], "torch.float32")
        optimizer = torch.optim.AdamW([model.adapter], lr=5e-5)
        model.adapter.square().sum().backward()
        optimizer.step()
        moments = _adam_moment_precision(optimizer, [model.adapter])
        self.assertEqual(moments["optimizer_moment_tensors"], 2)
        optimizer.state[model.adapter]["exp_avg"] = optimizer.state[model.adapter]["exp_avg"].bfloat16()
        with self.assertRaisesRegex(ValueError, "FP32"):
            _adam_moment_precision(optimizer, [model.adapter])
        model.adapter.data = model.adapter.data.bfloat16()
        with self.assertRaisesRegex(ValueError, "FP32 storage"):
            _probe_parameter_precision(model)

    def test_shift_mask_per_sequence_mean_and_bounded_head(self):
        import torch

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(
                    [[1., 0.], [0., 1.], [1., 1.]]))

            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        class Head(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 3, bias=False)
                self.weight.data.copy_(torch.tensor([[1., 0.], [0., 2.], [-1., 1.]]))
                self.largest_projection = 0

            def forward(self, states):
                self.largest_projection = max(self.largest_projection, len(states))
                return super().forward(states)

        model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen2"),
                                      model=Backbone(), lm_head=Head())
        inputs = torch.tensor([[0, 1, 2, 0], [2, 0, 0, 0]])
        labels = torch.tensor([[-100, -100, 2, 0], [-100, 0, -100, -100]])
        attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        def ce(logits, target):
            return math.log(sum(math.exp(value) for value in logits)) - logits[target]

        # Targets at positions 2/3 use hidden states from positions 1/2.
        expected = ((ce([0, 2, 1], 2) + ce([1, 2, 0], 0)) / 2,
                    ce([1, 2, 0], 0))
        with torch.inference_mode():
            values = batch_sequence_ce(model, inputs, labels, attention, chunk_tokens=1)
        self.assertEqual([count for _, count in values], [2, 1])
        for (value, _), oracle in zip(values, expected):
            self.assertAlmostEqual(value, oracle, places=6)
        self.assertEqual(model.lm_head.largest_projection, 1)


@unittest.skipUnless(all(importlib.util.find_spec(package) for package in
                         ("torch", "transformers", "peft", "safetensors")),
                     "local model/adapter libraries are unavailable")
class TinyQwenPeftIntegration(unittest.TestCase):
    def test_adapter_disabled_control_chunked_native_ce_and_finite_export(self):
        """Real tiny Qwen2/PEFT, no pretrained downloads or GPU required."""
        import copy
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=32,
                             eos_token_id=31, pad_token_id=0, tie_word_embeddings=False,
                             attention_dropout=0.0, use_cache=False)
        native = Qwen2ForCausalLM(config).eval()
        unchanged = copy.deepcopy(native)
        model = get_peft_model(native, LoraConfig(r=2, lora_alpha=4,
            lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM))
        inputs = torch.tensor([[2, 3, 4, 31]])
        labels = torch.tensor([[-100, -100, 4, 31]])
        attention = torch.ones_like(inputs)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in parameters))
        optimizer = torch.optim.AdamW(parameters, lr=0.1)
        model.train()
        loss = model(input_ids=inputs, labels=labels, attention_mask=attention).loss
        loss.backward()
        optimizer.step()
        self.assertTrue(_adam_moment_precision(optimizer, parameters)["optimizer_moments_verified"])
        model.eval()
        with torch.inference_mode():
            original_ce = float(unchanged(input_ids=inputs, labels=labels,
                                          attention_mask=attention).loss)
            with model.disable_adapter():
                disabled, count = batch_sequence_ce(model, inputs, labels, attention, 1)[0]
            adapted, adapted_count = batch_sequence_ce(model, inputs, labels, attention, 1)[0]
            native_adapted_ce = float(model(input_ids=inputs, labels=labels,
                                            attention_mask=attention).loss)
        self.assertEqual((count, adapted_count), (2, 2))
        self.assertAlmostEqual(disabled, original_ce, places=6)
        self.assertAlmostEqual(adapted, native_adapted_ce, places=6)
        self.assertNotEqual(adapted, disabled)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            exported = _finite_adapter(Path(directory))
            self.assertGreater(exported["tensor_count"], 0)
            self.assertIn("adapter_model.safetensors", exported["files"])


if __name__ == "__main__":
    unittest.main()
