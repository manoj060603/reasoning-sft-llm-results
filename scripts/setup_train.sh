#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
python -m pip install 'torch==2.6.0' 'deepspeed==0.16.8' 'bitsandbytes==0.45.5'
python -m pip install -e './LLaMA-Factory[metrics]' -r requirements-selection.txt
python -m pip install 'flash-attn==2.7.4.post1' --no-build-isolation
python -m pip check
python -m pip freeze > artifacts-train-environment.txt
