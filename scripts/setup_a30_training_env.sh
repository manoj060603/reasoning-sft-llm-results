#!/usr/bin/env bash
# CPU-only dependency recovery; leaves .venv-lmf, its base Conda env and evaluation untouched.
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
.venv-lmf/bin/python -m venv --system-site-packages .venv-lmf-a30-ds0169
DS_BUILD_OPS=0 .venv-lmf-a30-ds0169/bin/python -m pip install --no-deps --no-build-isolation -r configs/requirements-a30-execution.txt
.venv-lmf-a30-ds0169/bin/python - <<'PY'
from pathlib import Path
import sysconfig
Path(sysconfig.get_path('purelib'),'parent_lmf_overlay.pth').write_text(
 '/projects/tealab/rohit/reasoning-sft-20261011/.venv-lmf/lib/python3.10/site-packages\n')
PY
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_HOME=/apps/common/software/CUDA/12.8.0 TORCH_EXTENSIONS_DIR="$PWD/artifacts/torch_extensions_ds0169"
export PATH="$CUDA_HOME/bin:$PATH"
.venv-lmf-a30-ds0169/bin/python scripts/check_a30_dependencies.py
