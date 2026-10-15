import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from reasoning.data import boxed, canonical, digest, prompt_id, read_rows, write_json
from reasoning.prepare import ExclusionIndex, prepare
from reasoning.select import advanced_select, allocate, eligible
from reasoning.metrics import pass_at_k, paired_bootstrap, summarize
from reasoning.evaluate import valid_result
from reasoning.compare import load_run, check_protocol

ROOT = Path(__file__).resolve().parents[1]


def example(i):
    # Long unique pseudorandom prompts avoid artificial near-duplicate leakage.
    rng = random.Random(i)
    q = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz ', k=150))
    return canonical(dict(instruction=q, output='Reasoning. '+r'\boxed{42}', input=''))


class PipelineTests(unittest.TestCase):
    def test_unicode_separator_is_not_record_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'x.jsonl'
            p.write_text(json.dumps({'question':'a\u2028b'},ensure_ascii=False)+'\n')
            self.assertEqual(read_rows(p), [{'question':'a\u2028b'}])
            p.write_text('{"question":"a\nb"}\n')
            with self.assertRaises(ValueError): read_rows(p)

    def test_archive_counts(self):
        from reasoning.audit import EXPECTED
        for name,n in EXPECTED.items():
            self.assertEqual(len(read_rows(ROOT/'eval/data'/name/'test.jsonl')),n)

    def test_nested_box_and_incomplete(self):
        self.assertEqual(boxed(r'x \boxed{\frac{1}{2}}'),r'\frac{1}{2}')
        self.assertIsNone(boxed(r'\boxed{\frac{1}{2}'))
        self.assertEqual(boxed(r'\boxed{1} then \boxed{2}'),'2')

    def test_schema(self):
        x=canonical({'messages':[{'role':'user','content':'2+2?'},{'role':'assistant','content':'4'}]})
        self.assertEqual(x['instruction'],'2+2?')
        self.assertNotEqual(prompt_id('2+3'),prompt_id('2-3'))

    def test_contamination_and_split(self):
        rows=[example(i) for i in range(50)]
        index=ExclusionIndex([rows[0]['instruction']])
        self.assertTrue(index.matches('Solve: '+rows[0]['instruction']))
        train,val,stats=prepare(rows+[rows[1]],index,5,7)
        self.assertEqual(stats['benchmark_overlap'],1)
        self.assertEqual(stats['duplicate_prompt'],1)
        self.assertFalse({x['id'] for x in train}&{x['id'] for x in val})
        self.assertEqual((train,val),prepare(rows+[rows[1]],index,5,7)[:2])

    def test_capacity(self):
        q=allocate({0:1,1:50,2:20},30)
        self.assertEqual(sum(q.values()),30)
        self.assertEqual(q[0],1)
        with self.assertRaises(ValueError): allocate({0:2},3)

    def test_selection_and_bad_features(self):
        rows=[example(i) for i in range(100)]
        fs=[dict(id=r['id'],cluster=i%4,total_tokens=1000,response_tokens=800,
                 complete_box=True,repetition=.1,nll=i/100) for i,r in enumerate(rows)]
        fs[0]['total_tokens']=20000
        fs[1]['complete_box']=False
        selected,pool,q=advanced_select(rows,fs,20,42,16384,True)
        self.assertEqual(len(selected),20)
        self.assertEqual(len({r['id'] for r in selected}),20)
        self.assertEqual(len(pool),98)
        self.assertEqual(selected,advanced_select(rows,fs,20,42,16384,True)[0])
        self.assertNotIn(rows[0],selected)
        fs[2].pop('nll')
        with self.assertRaises(ValueError): advanced_select(rows,fs,20,42,16384,True)
        self.assertEqual(len(advanced_select(rows,fs,20,42,16384,False)[0]),20)
        with self.assertRaises(ValueError): advanced_select(rows,fs[:-1],20,42,16384,False)

    def test_pass_metrics(self):
        self.assertEqual(pass_at_k(1,8,1),.125)
        self.assertEqual(pass_at_k(1,8,8),1)
        self.assertAlmostEqual(pass_at_k(2,8,2),1-math.comb(6,2)/math.comb(8,2))
        s=summarize([{'answers_correctness':[True]+[False]*7}])
        self.assertEqual(s['pass_at_1'],.125)
        self.assertEqual(s['pass_at_8'],1)
        with self.assertRaises(ValueError): summarize([{'answers_correctness':[True]}])
        with self.assertRaises(ValueError): pass_at_k(1,8,9)

    def test_paired_bootstrap(self):
        result=paired_bootstrap({'a':0.,'b':0.},{'a':1.,'b':1.},100)
        self.assertEqual(result,{'delta':1.,'ci95':[1.,1.]})
        with self.assertRaises(ValueError): paired_bootstrap({'a':0},{'b':1})

    def test_incomplete_cached_result(self):
        self.assertFalse(valid_result({'problem_id':'a'},'a'))

    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            write_json(Path(t)/'manifest.json',dict(data={'amc':'hash'},problem_counts={'amc':40},problem_ids={'amc':['a']}))
            write_json(Path(t)/'amc/batch_000000.json',[dict(problem_id='a',answers_correctness=[True]*8)])
            with self.assertRaises(ValueError):load_run(t)

    def test_protocol_must_match_in_both_directions(self):
        a = {'model': 'base', 'identity': 'a', 'temperature': .6}
        b = {'model': 'candidate', 'identity': 'b', 'lora_adapter': 'adapter',
             'lora_identity': {'adapter_model.safetensors': 'hash'}, 'temperature': .6}
        check_protocol(a, b)
        b['extra_setting'] = True
        with self.assertRaises(ValueError): check_protocol(a, b)
        with self.assertRaises(ValueError): check_protocol(b, a)

    def test_repeated_benchmark_rows_keep_distinct_stable_ids(self):
        from reasoning.benchmark import record_ids, duplicate_audit
        rows = [dict(question='same question', answer='1'),
                dict(question='another question', answer='2'),
                dict(question='same question', answer='1')]
        ids = record_ids(rows)
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids, record_ids(rows))
        self.assertEqual(ids[1], prompt_id('another question'))
        self.assertEqual(duplicate_audit(rows)['unique_questions'], 2)
        self.assertFalse(duplicate_audit(rows)['conflicting_duplicate_answers'])

    def test_duplicate_question_bootstrap_preserves_cluster(self):
        from reasoning.metrics import paired_cluster_bootstrap
        base = {'a': 0., 'b': 0., 'c': 0.}
        candidate = {'a': 1., 'b': 1., 'c': 0.}
        actual = paired_cluster_bootstrap(base, candidate, {'a':'same','b':'same','c':'other'}, 100)
        self.assertAlmostEqual(actual['delta'], 2/3)
        self.assertEqual(actual['ci95'], [0., 1.])
        self.assertEqual(paired_cluster_bootstrap(base,candidate,{k:k for k in base},100),
                         paired_bootstrap(base,candidate,100))

    def test_cache_migration_copies_only_unchanged_complete_predictions(self):
        from reasoning.migrate_benchmark_ids import migrate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';data=root/'data'
            datasets={'amc':[{'question':'unique','answer':'1'}],
                      'repeated':[{'question':'same','answer':'2'}]*2}
            for name,rows in datasets.items():write_json(data/name/'test.jsonl',rows)
            manifest=dict(data={b:digest(data/b/'test.jsonl') for b in datasets},
                problem_counts={b:len(rows) for b,rows in datasets.items()},
                problem_ids={b:[prompt_id(r['question']) for r in rows] for b,rows in datasets.items()})
            write_json(source/'manifest.json',manifest)
            result=dict(problem_id=prompt_id('unique'),question='unique',gold_answer='1',
                generated_responses=['1']*8,answers_correctness=[True]*8,finish_reasons=['stop']*8)
            write_json(source/'amc/batch_000000.json',[result])
            receipt=migrate(source,root/'migrated',data)
            self.assertEqual(receipt['copied_complete_benchmarks'],['amc'])
            new=json.loads((root/'migrated/manifest.json').read_text())
            self.assertEqual(len(set(new['problem_ids']['repeated'])),2)
            self.assertEqual(digest(source/'amc/batch_000000.json'),
                             digest(root/'migrated/amc/batch_000000.json'))
            self.assertEqual(json.loads((source/'manifest.json').read_text()),manifest)
            write_json(source/'repeated/batch_000000.json',[result])
            with self.assertRaises(ValueError):migrate(source,root/'rejected',data)
            self.assertFalse((root/'rejected').exists())

    def test_cli_select_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            rows=[example(i) for i in range(60)]
            val=[example(i) for i in range(60,70)]
            for name,rs in [('pool',rows),('validation',val)]:
                write_json(d/f'{name}.json',rs)
                fs=[dict(id=r['id'],cluster=i%4,total_tokens=500,response_tokens=450,
                    complete_box=True,repetition=.01,nll=i/100) for i,r in enumerate(rs)]
                write_json(d/f'{name}_f.json',fs)
                write_json(d/f'{name}_f.json.meta.json',{'input_sha256':digest(d/f'{name}.json')})
            subprocess.run([sys.executable,'-m','reasoning.select','--pool',str(d/'pool.json'),
                '--validation',str(d/'validation.json'),'--features',str(d/'pool_f.json'),
                '--validation-features',str(d/'validation_f.json'),'--size','15',
                '--output',str(d/'out')],cwd=ROOT,check=True,capture_output=True)
            self.assertEqual(len(read_rows(d/'out/advanced.json')),15)
            self.assertEqual(set(read_rows(d/'out/advanced.json')[0]),{'instruction','input','output','system'})
            manifest=json.loads((d/'out/selection_manifest.json').read_text())
            self.assertEqual(manifest['response_tokens']['advanced'],6750)


if __name__=='__main__':unittest.main()
