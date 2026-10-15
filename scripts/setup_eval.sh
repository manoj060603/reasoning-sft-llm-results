#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
python -m pip install -r requirements-eval.txt
python -m pip install 'antlr4-python3-runtime==4.7.2'
# latex2sympy2's generated parser requires ANTLR 4.7.2. SymPy's separate
# ANTLR LaTeX parser may be unavailable; the supplied grader catches this and
# falls back to latex2sympy2. Validate representative equivalences below.
python -m pip install --no-deps 'latex2sympy2==1.9.1'
python -m reasoning.grader_preflight
python -m pip freeze > artifacts-eval-environment.txt
