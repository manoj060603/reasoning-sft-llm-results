"""Exercise real tokenization, curation, and config generation on SYNTHETIC data.

Usage: python scripts/smoke_cpu.py --tokenizer Qwen/Qwen2.5-3B-Instruct
No model weights or GPU are required. This does not measure model accuracy.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reasoning.data import write_json, read_rows

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tokenizer', default='Qwen/Qwen2.5-3B-Instruct')
a = p.parse_args()
os.chdir(ROOT)
d = ROOT/'artifacts/smoke'
rows = []
for i in range(48):
    rng = random.Random(i)
    q = ' '.join(''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=8)) for _ in range(20))
    answer = ' '.join(str(j)+': '+''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=12)) for j in range(80))+r' \boxed{42}'
    rows.append(dict(instruction=q,output=answer,input=''))
write_json(d/'source.json', rows)

def run(module, *args):
    subprocess.run([sys.executable,'-m','reasoning.'+module,*map(str,args)],check=True)

run('audit')
run('prepare','--input',d/'source.json','--eval-dir','artifacts/benchmarks','--val-size',8,'--output',d/'prepared')
for name,out in [('pool','features'),('validation','val_features')]:
    run('features','--input',d/f'prepared/{name}.json','--output',d/f'{out}.json',
        '--model',a.tokenizer,'--clusters',4)
run('select','--pool',d/'prepared/pool.json','--validation',d/'prepared/validation.json',
    '--features',d/'features.json','--validation-features',d/'val_features.json',
    '--strategy','diversity','--size',15,'--output',d/'datasets')
for profile in ['full','qlora']:
    run('train_config','--dataset','advanced','--dataset-dir',d/'datasets',
        '--profile',profile,'--output',d/f'smoke-{profile}')
    src = ROOT/'configs'/f'smoke-{profile}.yaml'
    src.replace(d/src.name)
    export = ROOT/'configs'/f'smoke-{profile}-export.yaml'
    if export.exists(): export.replace(d/export.name)
run('devset','--input',d/'prepared/validation.json','--output',d/'dev/acereason_dev/test.jsonl','--size',4)
assert len(read_rows(d/'datasets/advanced.json')) == 15
write_json(d/'result.json',dict(status='passed',synthetic=True,model_accuracy_measured=False,
    source_records=48,train_pool=40,validation=8,selected_per_variant=15,dev=4))
print('CPU smoke PASSED. Synthetic data only; no model accuracy measured.')
