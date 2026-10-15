"""Supplement the supplied benchmark suite with a pinned public AIME25 source."""
import argparse
import json
import shutil
from pathlib import Path
from .data import digest,prompt_id,question,read_rows,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='563bb8404243c5f09de6ec262f2db674fe5bce9b')
    p.add_argument('--base-benchmarks',default='artifacts/benchmarks')
    p.add_argument('--output',default='artifacts/benchmarks_extended')
    a=p.parse_args()
    from huggingface_hub import HfApi,hf_hub_download
    repo='math-ai/aime25'
    revision=HfApi().dataset_info(repo,revision=a.revision).sha
    raw=hf_hub_download(repo,'test.jsonl',repo_type='dataset',revision=revision)
    rows=[dict(id=str(r['id']),question=question(r),answer=str(r['answer'])) for r in read_rows(raw)]
    if len(rows)!=30 or len({prompt_id(r['question']) for r in rows})!=30:
        raise ValueError('expected exactly 30 distinct AIME25 questions')
    if any(not r['answer'].isdigit() or not 0<=int(r['answer'])<=999 for r in rows):
        raise ValueError('unexpected AIME25 answer schema')
    root=Path(a.output);sources={}
    for source in sorted(Path(a.base_benchmarks).glob('*/test.jsonl')):
        target=root/source.parent.name/'test.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=digest(source):
            raise ValueError(f'refusing to replace a different benchmark: {target}')
        shutil.copyfile(source,target);sources[source.parent.name]=digest(source)
    if len(sources)!=8:
        raise ValueError('expected the eight supplied benchmark files')
    target=root/'aime25'/'test.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    if target.exists() and target.read_text()!=content:
        raise ValueError('use a new directory for a changed AIME25 revision')
    temporary=target.with_suffix('.jsonl.tmp');temporary.write_text(content);temporary.replace(target)
    receipt=dict(repo=repo,revision=revision,source_url=f'https://huggingface.co/datasets/{repo}',
        source_sha256=digest(raw),canonical_sha256=digest(target),problems=len(rows),
        license='Apache-2.0 (dataset card)',supplied_benchmark_hashes=sources,
        limitation='Separately sourced AIME25; the historical leaderboard data/version is not verified.')
    write_json(root/'supplement.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
