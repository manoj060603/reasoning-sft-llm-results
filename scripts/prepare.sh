#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
STRATEGY=${STRATEGY:-hybrid}
if [[ "$STRATEGY" != hybrid && "$STRATEGY" != diversity ]]; then
  echo 'STRATEGY must be hybrid or diversity' >&2
  exit 2
fi
"$PYTHON" -m reasoning.audit
"$PYTHON" scripts/download_data.py
"$PYTHON" -m reasoning.prepare --input artifacts/raw/data/acereason11_100k.json --eval-dir artifacts/benchmarks
NLL_ARGS=()
if [[ "$STRATEGY" == hybrid ]]; then NLL_ARGS=(--score-nll); fi
"$PYTHON" -m reasoning.features "${NLL_ARGS[@]}"
"$PYTHON" -m reasoning.features --input artifacts/prepared/validation.json --output artifacts/validation_features.json
"$PYTHON" -m reasoning.select --strategy "$STRATEGY"
