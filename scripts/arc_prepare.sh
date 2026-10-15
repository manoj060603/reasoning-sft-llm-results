#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export LOKY_MAX_CPU_COUNT=8
BASE=/home/rohits25/.conda/envs/sata-vllm/bin/python
if [[ ! -x .venv-arc/bin/python ]]; then "$BASE" -m venv --system-site-packages .venv-arc; fi
PYTHON="$PWD/.venv-arc/bin/python"
"$PYTHON" -m pip install scikit-learn word2number Pebble timeout-decorator antlr4-python3-runtime==4.7.2
"$PYTHON" -m pip install --no-deps latex2sympy2==1.9.1
"$PYTHON" -m pip freeze > artifacts/arc-environment.txt
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download,HfApi
from reasoning.data import write_json
api=HfApi();repo='Qwen/Qwen2.5-3B-Instruct';rev=api.model_info(repo).sha
path=snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
write_json('artifacts/model.json',dict(repo=repo,revision=rev,path=path))
print('MODEL_READY',path,flush=True)
PY
"$PYTHON" scripts/download_data.py
"$PYTHON" -m reasoning.audit
"$PYTHON" -m reasoning.prepare --input artifacts/raw/data/acereason11_100k.json --eval-dir artifacts/benchmarks
"$PYTHON" -m reasoning.features --model "$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')"
"$PYTHON" -m reasoning.features --input artifacts/prepared/validation.json --output artifacts/validation_features.json --model "$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')"
"$PYTHON" -m reasoning.select --strategy route --output artifacts/datasets-route
"$PYTHON" -m reasoning.devset --size 100
"$PYTHON" -m unittest discover -s tests -v
