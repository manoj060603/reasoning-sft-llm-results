"""Reuse unchanged CPU features only when source bytes, code and model match."""
import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path
from . import features
from .data import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--from-feature',required=True)
    p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    a=p.parse_args()
    source=Path(a.from_feature);meta_path=Path(str(source)+'.meta.json')
    if not source.exists() or not meta_path.exists():return 1
    meta=json.loads(meta_path.read_text());provenance=meta['provenance'];arguments=meta['arguments']
    expected=dict(model=a.model,revision='main',clusters=128,embedding_model=None,
                  score_nll=False,nll_tokens=512,seed=42)
    valid=(meta['input_sha256']==digest(a.input)
        and provenance['code_sha256']==digest(features.__file__)
        and all(arguments.get(k)==v for k,v in expected.items())
        and all(importlib.metadata.version(k)==v for k,v in provenance['versions'].items())
        and provenance['local_tokenizer_files']=={p.name:digest(p) for p in Path(a.model).glob('*.json')})
    if not valid:return 1
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,temporary);temporary.replace(target)
    meta['reuse']=dict(from_feature=str(source),feature_sha256=digest(source),
        target_input=a.input,reason='byte-identical input and matching feature code/runtime/tokenizer')
    write_json(str(target)+'.meta.json',meta)
    print(f'Reused verified features: {source} -> {target}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
