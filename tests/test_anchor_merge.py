"""Lightweight fixtures; run only in the native CPU allocation, never login."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from deadline.anchor_search import core
from reasoning.data import SYSTEM, prompt_id


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]['content'] + '\n' + messages[1]['content'] + '\nassistant:'

    def encode(self, text, add_special_tokens=False):
        return list(range(2049)) if 'LONGQUESTION' in text else [ord(char) for char in text]


def rows():
    return [dict(id=prompt_id(f'Question {number}'), instruction=f'Question {number}',
                 input='', output=r'Teacher reasoning. \boxed{42}', system=SYSTEM)
            for number in range(1000)]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class SelectionTests(unittest.TestCase):
    def test_fresh_splits_are_deterministic_disjoint_and_answer_free(self):
        source = rows()
        old = [row['id'] for row in source[:64]]
        splits, counts = core.choose_splits(source, FakeTokenizer(), old)
        reversed_splits, _ = core.choose_splits(list(reversed(source)), FakeTokenizer(), old)
        self.assertEqual(splits, reversed_splits)
        self.assertEqual(counts['eligible_unused_rows'], 936)
        self.assertEqual(counts['heldout_rows'], 1000)
        self.assertEqual([len(splits[name]) for name in core.SPLITS], [192, 192])
        identifiers = [item['id'] for name in core.SPLITS for item in splits[name]]
        self.assertEqual(len(set(identifiers)), 384)
        self.assertFalse(set(identifiers).intersection(old))
        expected = sorted((row['id'] for row in source[64:]),
                          key=lambda identifier: core.text_digest('20261021:anchor-v2:' + identifier))[:384]
        self.assertEqual(identifiers, expected)
        for name in core.SPLITS:
            for item in splits[name]:
                self.assertEqual(set(item), {'id', 'prompt_tokens', 'prompt_sha256',
                    'prompt_ids_sha256', 'teacher_answer_sha256'})
                self.assertNotIn('42', item.values())

    def test_boxed_teacher_and_native_prompt_capacity_both_apply(self):
        source = rows()
        for item in source[:100]:
            item['output'] = 'No final boxed teacher answer'
        for index in range(100, 150):
            source[index]['instruction'] = f'LONGQUESTION {index}'
            source[index]['id'] = prompt_id(source[index]['instruction'])
        old = [row['id'] for row in source[-64:]]
        splits, counts = core.choose_splits(source, FakeTokenizer(), old)
        self.assertEqual(counts['eligible_before_exclusion'], 850)
        self.assertEqual(counts['eligible_unused_rows'], 786)
        excluded = {row['id'] for row in source[:150]} | set(old)
        self.assertFalse({item['id'] for value in splits.values() for item in value} & excluded)

    def test_insufficient_eligible_unused_source_fails(self):
        source = rows()
        for item in source[400:]:
            item['output'] = 'No boxed answer'
        with self.assertRaisesRegex(ValueError, 'Fewer than 384'):
            core.choose_splits(source, FakeTokenizer(), [item['id'] for item in source[:64]])

    def test_duplicate_or_nonheldout_old_ids_fail(self):
        source = rows()
        old = [item['id'] for item in source[:64]]
        for invalid in ([old[0]] * 64, old[:-1] + ['f' * 64]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                core.choose_splits(source, FakeTokenizer(), invalid)

    def test_source_question_hash_cannot_be_reassigned(self):
        source = rows()
        source[90]['instruction'] = 'Substituted question'
        with self.assertRaises(ValueError):
            core.choose_splits(source, FakeTokenizer(), [item['id'] for item in source[:64]])

    def test_fixed_grid_split_count_and_protocol_are_required(self):
        original = core.read_json(core.PLAN)
        for mutate in (
            lambda value: value['alpha_advanced_candidates'].update(alpha0125=.1),
            lambda value: value['source_selection'].update(search_rows=191),
            lambda value: value['source_selection'].update(seed=42),
            lambda value: value['source_protocol'].update(n=1),
            lambda value: value['source_protocol'].update(max_tokens=8192),
        ):
            value = copy.deepcopy(original)
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.validate_plan(value)
        self.assertEqual(core.validate_plan(original), original)


class MergeTests(unittest.TestCase):
    def test_two_fixed_candidates_and_endpoints_use_fp32_arithmetic(self):
        import torch
        first = torch.tensor([1., -4.], dtype=torch.bfloat16)
        second = torch.tensor([3., 12.], dtype=torch.float32)
        for alpha in (0., .125, .375, 1.):
            result = core.blend_fp32(first, second, alpha)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, (1-alpha) * first.float() + alpha * second)
            self.assertEqual(result.device.type, 'cpu')

    def test_nonfinite_shape_and_coefficient_errors_fail(self):
        import torch
        for first, second, alpha in (
            (torch.tensor([float('nan')]), torch.tensor([1.]), .125),
            (torch.tensor([1.]), torch.tensor([float('inf')]), .375),
            (torch.tensor([1.]), torch.tensor([1., 2.]), .125),
            (torch.tensor([1.]), torch.tensor([2.]), float('nan')),
            (torch.tensor([1.]), torch.tensor([2.]), -0.1),
        ):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                core.blend_fp32(first, second, alpha)

    def test_keys_shapes_and_fp32_advanced_specs_are_guarded(self):
        base = {'a': {'shape': [2, 2], 'dtype': 'BF16'}}
        valid = {'a': {'shape': [2, 2], 'dtype': 'F32'}}
        core.matching_specs(base, valid)
        for invalid in ({'b': valid['a']}, {'a': {'shape': [4], 'dtype': 'F32'}},
                        {'a': {'shape': [2, 2], 'dtype': 'BF16'}}):
            with self.assertRaises(ValueError):
                core.matching_specs(base, invalid)

    def test_output_refusal_precedes_any_model_or_source_work(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(core, 'require_compute', return_value={}), \
                patch.object(core, 'checked_inputs') as checked:
            before = list(Path(directory).iterdir())
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                core.prepare(directory)
            self.assertFalse(checked.called)
            self.assertEqual(list(Path(directory).iterdir()), before)


class PreparedSealTests(unittest.TestCase):
    def fixture(self, directory):
        directory = Path(directory)
        source = rows()
        old = [item['id'] for item in source[:64]]
        splits, _ = core.choose_splits(source, FakeTokenizer(), old)
        old_path = directory / 'old_ids.json'
        write(old_path, old)
        old_sha = core.digest(old_path)
        value = dict(schema=core.SCHEMA, phase='prepare', complete=True, output=str(directory.resolve()),
            plan_sha256=core.digest(core.PLAN), code_sha256=core.code_hashes(), splits=splits,
            old_screen_ids={'path': str(old_path), 'sha256': old_sha}, old_screen_ids_sha256=old_sha)
        shutil.copyfile(core.PLAN, directory / 'plan.json')
        self.seal(directory, value)
        return value, old_sha

    def seal(self, directory, value):
        ids = {split: [item['id'] for item in value['splits'][split]] for split in core.SPLITS}
        value['selected_ids_sha256'], value['split_metadata_sha256'] = {}, {}
        for split in core.SPLITS:
            write(directory / (split + '.json'), value['splits'][split])
            write(directory / (split + '_ids.json'), ids[split])
            value['selected_ids_sha256'][split] = core.digest(directory / (split + '_ids.json'))
            value['split_metadata_sha256'][split] = core.digest(directory / (split + '.json'))
        value['selected_ids_seal'] = core.json_digest(ids)
        write(directory / 'prepared.json', value)

    def test_changed_split_file_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            value, old_sha = self.fixture(directory)
            write(directory / 'search.json', [])
            with patch.object(core, 'OLD_IDS_SHA256', old_sha), \
                    patch.object(core, 'validate_plan', side_effect=lambda plan: plan), \
                    patch.object(core, 'model_inputs') as models:
                with self.assertRaisesRegex(ValueError, 'Source split files changed'):
                    core.load_prepared(directory)
                self.assertFalse(models.called)

    def test_overlap_rejected_even_with_resealed_split_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            value, old_sha = self.fixture(directory)
            value['splits']['confirmation'][0] = copy.deepcopy(value['splits']['search'][0])
            self.seal(directory, value)
            with patch.object(core, 'OLD_IDS_SHA256', old_sha), \
                    patch.object(core, 'validate_plan', side_effect=lambda plan: plan):
                with self.assertRaisesRegex(ValueError, 'Source splits overlap'):
                    core.load_prepared(directory)

    def test_old_screen_overlap_rejected_even_with_resealed_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            value, old_sha = self.fixture(directory)
            value['splits']['search'][0]['id'] = core.read_json(directory / 'old_ids.json')[0]
            self.seal(directory, value)
            with patch.object(core, 'OLD_IDS_SHA256', old_sha), \
                    patch.object(core, 'validate_plan', side_effect=lambda plan: plan):
                with self.assertRaisesRegex(ValueError, 'overlap old screen'):
                    core.load_prepared(directory)


if __name__ == '__main__':
    unittest.main()
