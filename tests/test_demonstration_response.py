import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from reasoning.demonstration_response import (
    candidate_order, categorize, continuation_ids, curriculum, make_prefix,
    prefix_leak_reason, proportional, read_ledger, request_key,
    select_categories, sign_test, stable_seed, text_hash, run_requests,
)


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(c) + 10 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(c - 10) for c in ids)

    def apply_chat_template(self, messages, **kwargs):
        normal = self.encode(messages[0]['content'] + messages[1]['content'] + '<assistant>')
        if messages[-1]['role'] == 'assistant':
            return normal + self.encode(messages[-1]['content']) + (
                [] if kwargs.get('continue_final_message') else [self.eos_token_id])
        return normal


def label(row_id):
    return dict(id=row_id, domain='algebra', skills=['equations', 'factoring'],
                composition_depth=2, representation='symbolic', language='english',
                answer_form='integer')


def records(row_id, normal, assisted, **overrides):
    result = {}
    for condition, scores in [('normal', normal), ('assisted', assisted)]:
        for i, correct in enumerate(scores):
            result[request_key(row_id, condition, i)] = dict(
                finish_reason='stop', supported=True, correct=correct, **overrides)
    return result


class DemonstrationTests(unittest.TestCase):
    def test_proportional_capacity_and_remainders(self):
        self.assertEqual(proportional({'a': 3, 'b': 1}, 3), {'a': 2, 'b': 1})
        self.assertEqual(proportional({'a': 0, 'b': 0}, 0), {'a': 0, 'b': 0})
        with self.assertRaises(ValueError): proportional({'a': 1}, 2)

    def test_reserve_order_is_input_order_independent(self):
        rows = [dict(id=str(i)) for i in range(20)]
        fs = [dict(id=str(i), cluster=i % 3, total_tokens=1000, response_tokens=200+i,
                   complete_box=True, repetition=0.) for i in range(20)]
        a = candidate_order(rows, fs, 12, 42)
        self.assertEqual(a, candidate_order(rows[::-1], fs[::-1], 12, 42))
        self.assertEqual(sum(x['count'] for x in a), 12)

    def test_prefix_and_numeric_answer_withholding(self):
        tok = Tokenizer()
        text = 'We use modular arithmetic.\n\n' + 'Work continues. ' * 80 + r'\boxed{97}'
        prefix, error = make_prefix(tok, text)
        self.assertIsNone(error)
        self.assertEqual(prefix, 'We use modular arithmetic.')
        self.assertEqual(prefix_leak_reason('Let the value be 1 and proceed.', '1'),
                         'answer_surface_overlap')
        self.assertIsNone(prefix_leak_reason('Try 19 cases.', '1'))
        self.assertEqual(prefix_leak_reason(r'We obtain \boxed{3}.', '97'), 'explicit_answer')
        self.assertEqual(make_prefix(tok, 'No complete paragraph.')[1], 'missing_box')

    def test_continuation_has_no_eos_and_preserves_user_prompt(self):
        tok = Tokenizer(); row = {'instruction': 'Original question'}
        normal = continuation_ids(tok, row, '')
        assisted = continuation_ids(tok, row, 'Partial reasoning')
        self.assertEqual(assisted[:len(normal)], normal)
        self.assertNotIn(tok.eos_token_id, assisted)

    def test_censoring_and_categories(self):
        item = dict(id='x', prefix='Useful reasoning')
        self.assertEqual(categorize(item, records('x', [True,True], [True,True])), ('retention',0.))
        self.assertEqual(categorize(item, records('x', [True,False], [True,True])), ('manageable',.5))
        self.assertEqual(categorize(item, records('x', [False,False], [True,False])), ('missed',.5))
        r = records('x', [False,False], [True,True])
        r[request_key('x','normal',0)]['finish_reason'] = 'length'
        self.assertEqual(categorize(item, r), (None,0.))
        r[request_key('x','normal',0)].update(finish_reason='stop', supported=False)
        self.assertEqual(categorize(item, r), (None,0.))

    def test_distinct_selection_preserves_categories_and_curriculum(self):
        candidates, ledger = [], {}
        for group, scores in [('r',([True,True],[True,True])),
                              ('m',([True,False],[True,True])),
                              ('h',([False,False],[True,True]))]:
            for i in range(6):
                row_id = f'{group}{i}'
                candidates.append(dict(id=row_id, cluster=i%2, prefix='Reasoning'))
                ledger.update(records(row_id, *scores))
        labels = [label(x['id']) for x in candidates]
        counts = {'retention':4,'manageable':4,'missed':4}
        chosen, receipt = select_categories(candidates, ledger, labels, counts)
        ordered = curriculum(chosen)
        self.assertEqual(len({x['id'] for x in ordered}), 12)
        self.assertEqual([x['category'] for x in ordered[:6]].count('manageable'),4)
        self.assertEqual([x['category'] for x in ordered[6:]].count('missed'),4)
        self.assertEqual(ordered, curriculum(chosen))
        with self.assertRaises(ValueError):
            select_categories(candidates, ledger, labels, {**counts,'missed':7})

    def test_resume_binds_seed_prefix_and_text(self):
        request = dict(key='x:normal:0', seed=10, prefix='')
        row = dict(key=request['key'], seed=10, prefix_sha256=text_hash(''), text='Saved',
                   text_sha256=text_hash('Saved'))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'ledger.jsonl'; path.write_text(json.dumps(row)+'\n')
            self.assertEqual(len(read_ledger(path,[request])),1)
            path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError): read_ledger(path,[request])
            path.write_text('{"torn":')
            with self.assertRaises(ValueError): read_ledger(path,[request])

    def test_fresh_seed_and_directional_control_test(self):
        self.assertEqual(stable_seed(42,'a','normal',0),stable_seed(42,'a','normal',0))
        self.assertNotEqual(stable_seed(42,'a','assisted',0),stable_seed(42,'a','audit_assisted',0))
        self.assertTrue(sign_test([1]*10+[0]*10)['passed'])
        self.assertFalse(sign_test([-1]*10)['passed'])
        self.assertFalse(sign_test([0]*100)['passed'])

    def test_generation_export_resume_uses_saved_requests_once(self):
        calls = []
        class Engine:
            def __init__(self, **kwargs): pass
            def get_tokenizer(self): return Tokenizer()
            def generate(self, prompts, params, **kwargs):
                calls.append(len(prompts))
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{97}',
                        finish_reason='stop',token_ids=[1,2,3])]) for _ in prompts]
        fake = SimpleNamespace(LLM=Engine,SamplingParams=lambda **kwargs:kwargs)
        requests = [dict(key=f'x:normal:{i}',id='x',condition='normal',sample=i,
                         seed=stable_seed(42,'x','normal',i),prefix='') for i in range(2)]
        plan = {'model':'fake'}; rows = {'x':{'instruction':'Question','output':r'\boxed{97}'}}
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', {'vllm':fake}), \
             patch('reasoning.demonstration_response.grade_output',return_value=(True,True)):
            ledger = Path(d)/'saved.jsonl'; args = SimpleNamespace(batch_size=1)
            first = run_requests(args,plan,requests,rows,ledger)
            second = run_requests(args,plan,requests,rows,ledger)
            self.assertEqual(first,second)
            self.assertEqual(calls,[1,1])
            self.assertTrue(all(r['correct'] for r in first.values()))


if __name__ == '__main__': unittest.main()
