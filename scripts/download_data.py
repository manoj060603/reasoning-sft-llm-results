import argparse
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import json

p = argparse.ArgumentParser()
p.add_argument('--output', default='artifacts/raw')
p.add_argument('--revision', default='main')
a = p.parse_args()
repo = 'redsgnaoh/acereason11_100k'
commit = HfApi().dataset_info(repo, revision=a.revision).sha
path = hf_hub_download(repo_id=repo, repo_type='dataset', revision=commit,
                      filename='data/acereason11_100k.json', local_dir=a.output)
Path(a.output, 'source.json').write_text(json.dumps(dict(repo=repo, commit=commit, file=path), indent=2)+'\n')
print(path)
