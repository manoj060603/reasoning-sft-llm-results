"""Two-source-row export checks without downloading a tokenizer or model."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from deadline.export_training import _ordered_chunks, export
from deadline.train_full import validate_cache
from reasoning.data import alpaca, prompt_id, read_rows
from reasoning.features import token_ids


class FakeTokenizer:
    """A prefix-consistent chat template with EOS plus a trailing newline."""
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, return_dict=False,
                            add_generation_prompt=False):
        assert tokenize and not return_dict
        assert messages[0]['role'] == 'system' and messages[1]['role'] == 'user'
        encode = lambda text: [200 + ord(character) for character in text]
        prompt = [11] + encode(messages[0]['content']) + [12] + encode(messages[1]['content']) + [13]
        if len(messages) == 2:
            assert add_generation_prompt
            return prompt
        assert not add_generation_prompt and messages[2]['role'] == 'assistant'
        return prompt + encode(messages[2]['content']) + [self.eos_token_id, 10]


def originals(long_second=False):
    rows = []
    for index in (1, 2):
        instruction = f'Q{index}'
        output = ('R' * 80 if long_second and index == 2 else 'Reason ') + f'\\boxed{{{index}}}'
        rows.append(dict(id=prompt_id(instruction), instruction=instruction, input='',
                         output=output, system='sys'))
    return rows


class DeadlineExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = FakeTokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def test_two_complete_originals_system_masks_eos_and_schema(self):
        rows = originals()
        before = copy.deepcopy(rows)
        identifiers = [row['id'] for row in reversed(rows)]
        destination = self.root / 'advanced'
        meta = export(rows, identifiers, self.tokenizer, destination, 'advanced', cutoff=64, expected_rows=2)
        cached, parsed_meta, receipt = validate_cache(destination / 'training.jsonl',
            destination / 'cache_meta.json', expected_rows=2, cutoff=64)
        lookup = {row['id']: row for row in rows}
        self.assertEqual(read_rows(destination / 'dataset.json'), alpaca([lookup[i] for i in identifiers]))
        self.assertEqual(json.loads((destination / 'selected_ids.json').read_text()), identifiers)
        self.assertEqual([row['id'] for row in cached], identifiers)
        self.assertEqual(receipt['truncated_rows'], 0)
        self.assertEqual(meta['dataset_role'], 'advanced')
        self.assertEqual(meta['truncation_policy'], 'none')
        for cached_row in cached:
            prompt, full = token_ids(self.tokenizer, lookup[cached_row['id']])
            self.assertEqual(cached_row['input_ids'], full)
            self.assertEqual(cached_row['labels'], [-100] * len(prompt) + full[len(prompt):])
            self.assertEqual(cached_row['labels'][-2:], [99, 10])
            self.assertEqual(cached_row['fold'], -1)
            self.assertEqual(cached_row['question_sha256'], cached_row['id'])
        self.assertEqual(rows, before)

    def test_random_keeps_both_originals_and_declares_exact_prefix_truncation(self):
        rows = originals(long_second=True)
        destination = self.root / 'random'
        meta = export(rows, [row['id'] for row in rows], self.tokenizer, destination,
                      'random', cutoff=32, expected_rows=2)
        cached, _, receipt = validate_cache(destination / 'training.jsonl',
            destination / 'cache_meta.json', expected_rows=2, cutoff=32)
        self.assertEqual(len(cached), 2)
        self.assertEqual(meta['unique_rows_dropped'], 0)
        self.assertEqual(meta['truncated_rows'], 1)
        self.assertEqual(meta['truncated_ids'], [rows[1]['id']])
        self.assertEqual(receipt['truncation_policy'], 'prefix_truncate')
        self.assertEqual(read_rows(destination / 'dataset.json'), alpaca(rows))
        for source, cache in zip(rows, cached):
            prompt, full = token_ids(self.tokenizer, source)
            expected = full[:32]
            self.assertEqual(cache['input_ids'], expected)
            self.assertEqual(cache['labels'], [-100] * len(prompt) + expected[len(prompt):])
        self.assertFalse(cached[0]['prefix_truncated'])
        self.assertTrue(cached[1]['prefix_truncated'])

    def test_default_8k_export_preserves_complete_original_above_4k(self):
        rows = originals()
        rows[1]['output'] = 'R' * 5000 + '\\boxed{2}'
        destination = self.root / 'advanced-8k'
        meta = export(rows, [row['id'] for row in rows], self.tokenizer, destination,
                      'advanced', expected_rows=2)
        cached, _, receipt = validate_cache(destination / 'training.jsonl',
            destination / 'cache_meta.json', expected_rows=2)
        self.assertEqual(meta['cutoff'], 8192)
        self.assertGreater(len(cached[1]['input_ids']), 4096)
        self.assertLessEqual(len(cached[1]['input_ids']), 8192)
        self.assertEqual(meta['truncated_rows'], 0)
        self.assertEqual(cached[1]['labels'][-2:], [99, 10])
        self.assertEqual(read_rows(destination / 'dataset.json'), alpaca(rows))

    def test_overlength_advanced_fails_without_complete_export_receipt(self):
        rows = originals(long_second=True)
        destination = self.root / 'advanced'
        with self.assertRaisesRegex(ValueError, 'preserve complete original targets'):
            export(rows, [row['id'] for row in rows], self.tokenizer, destination,
                   'advanced', cutoff=32, expected_rows=2)
        self.assertFalse((destination / 'cache_meta.json').exists())
        self.assertFalse((destination / 'training.jsonl').exists())

    def test_prompt_capacity_is_not_silently_dropped(self):
        rows = originals()
        rows[1]['instruction'] = 'Q' * 40
        rows[1]['id'] = prompt_id(rows[1]['instruction'])
        with self.assertRaisesRegex(ValueError, 'Prompt exceeds'):
            export(rows, [row['id'] for row in rows], self.tokenizer, self.root / 'random',
                   'random', cutoff=32, expected_rows=2)

    def test_existing_export_duplicate_or_missing_ids_fail(self):
        rows = originals()
        ids = [row['id'] for row in rows]
        destination = self.root / 'advanced'
        export(rows, ids, self.tokenizer, destination, 'advanced', cutoff=64, expected_rows=2)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            export(rows, ids, self.tokenizer, destination, 'advanced', cutoff=64, expected_rows=2)
        with self.assertRaisesRegex(ValueError, 'unique prompt IDs'):
            export(rows, [ids[0], ids[0]], self.tokenizer, self.root / 'duplicate',
                   'advanced', cutoff=64, expected_rows=2)
        with self.assertRaisesRegex(ValueError, 'unchanged source records'):
            export(rows, [ids[0], 'missing'], self.tokenizer, self.root / 'missing',
                   'advanced', cutoff=64, expected_rows=2)

    def test_parallel_arguments_reject_missing_local_tokenizer_and_invalid_workers(self):
        rows = originals()
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'positive integer'):
                export(rows, [row['id'] for row in rows], self.tokenizer, self.root / 'bad',
                       'random', expected_rows=2, workers=workers)
        for path in (None, self.root / 'missing-tokenizer'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'local tokenizer_path'):
                export(rows, [row['id'] for row in rows], self.tokenizer, self.root / 'bad',
                       'random', expected_rows=2, workers=2, tokenizer_path=path)
        self.assertFalse((self.root / 'bad').exists())

    def test_chunk_queue_is_bounded_ordered_and_only_receives_selected_rows(self):
        class Executor:
            active, peak = 0, 0
            chunks = []

            def submit(self, function, chunk):
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.chunks.append(chunk)
                owner = self

                class Future:
                    def result(self):
                        owner.active -= 1
                        return [(row, [row + 1]) for row in chunk]
                return Future()

        executor = Executor()
        selected = list(range(137))
        output = list(_ordered_chunks(executor, selected, workers=2))
        self.assertEqual(output, [(row, [row + 1]) for row in selected])
        self.assertLessEqual(executor.peak, 4)
        self.assertEqual(executor.active, 0)
        self.assertTrue(all(1 <= len(chunk) <= 16 for chunk in executor.chunks))
        self.assertEqual([row for chunk in executor.chunks for row in chunk], selected)

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'),
                         'requires the native Transformers/tokenizers runtime; no download is used')
    def test_spawn_parallel_matches_serial_bytes_with_real_local_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        native = Tokenizer(models.WordLevel({'[UNK]': 0, '<s>': 1, '</s>': 2,
                                             'system': 3, 'user': 4, 'assistant': 5,
                                             'sys': 6, 'Reason': 7}, unk_token='[UNK]'))
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=native, unk_token='[UNK]',
                                            eos_token='</s>', additional_special_tokens=['<s>'])
        tokenizer.chat_template = ("{% for message in messages %}{{ '<s>' + message['role'] + '\\n' + "
                                   "message['content'] + '</s>\\n' }}{% endfor %}"
                                   "{% if add_generation_prompt %}{{ '<s>assistant\\n' }}{% endif %}")
        local = self.root / 'tiny-local-tokenizer'
        tokenizer.save_pretrained(local)
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        rows = []
        for index in range(73):
            instruction = f'Question {index}'
            rows.append(dict(id=prompt_id(instruction), instruction=instruction, input='',
                             output='Reason ' * (80 if index == 7 else 2) + f'\\boxed{{{index}}}', system='sys'))
        identifiers = [row['id'] for row in reversed(rows)]
        before = copy.deepcopy(rows)
        for role, cutoff in (('random', 32), ('advanced', 256)):
            with self.subTest(role=role):
                serial, parallel = self.root / (role + '-serial'), self.root / (role + '-parallel')
                expected = export(rows, identifiers, tokenizer, serial, role, cutoff=cutoff, expected_rows=len(rows))
                actual = export(rows, identifiers, tokenizer, parallel, role, cutoff=cutoff,
                                expected_rows=len(rows), workers=2, tokenizer_path=local)
                self.assertEqual(actual, expected)
                for name in ('training.jsonl', 'dataset.json', 'selected_ids.json', 'cache_meta.json'):
                    self.assertEqual((parallel / name).read_bytes(), (serial / name).read_bytes(), name)
                self.assertEqual([row['id'] for row in read_rows(parallel / 'training.jsonl')], identifiers)
                self.assertEqual(actual['truncated_rows'], 1 if role == 'random' else 0)
        self.assertEqual(rows, before)


if __name__ == '__main__':
    unittest.main()
