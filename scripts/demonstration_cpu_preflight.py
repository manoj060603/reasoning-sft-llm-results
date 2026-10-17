"""Production tokenizer/LMF export verification using synthetic CPU-only records."""
import argparse
import tempfile
from pathlib import Path

from reasoning.data import SYSTEM, digest, prompt_id, write_json
from reasoning.demonstration_preflight import verify
from reasoning.demonstration_response import CATEGORIES, curriculum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    args = p.parse_args()
    selected = {}; rows = {}
    for category,count in CATEGORIES.items():
        selected[category] = []
        for i in range(count):
            instruction = f'Synthetic CPU serialization case {category} {i}: compute 2+3.'
            row_id = prompt_id(instruction)
            rows[row_id] = dict(instruction=instruction,input='',system=SYSTEM,
                                output='Adding two and three gives five.\n\n'+r'\boxed{5}')
            selected[category].append(dict(id=row_id,cluster=i%8,category=category))
    ordered = curriculum(selected)
    with tempfile.TemporaryDirectory(prefix='demonstration-cpu-') as d:
        path = Path(d)
        write_json(path/'advanced.json',[rows[x['id']] for x in ordered])
        write_json(path/'selection_manifest.json',{'selected_ids':[x['id'] for x in ordered],
                   'selected':ordered,'data_sha256':digest(path/'advanced.json')})
        verify(args.model,path,require_audit=False)
    print('Synthetic 15K CPU tokenizer, actual training labels/EOS, and actual sampler checks passed.')


if __name__=='__main__': main()
