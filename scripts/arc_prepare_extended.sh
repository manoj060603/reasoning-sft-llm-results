#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export LOKY_MAX_CPU_COUNT=8
PYTHON="$PWD/.venv-arc/bin/python"
MODEL=$("$PYTHON" -c 'import json;print(json.load(open("artifacts/model.json"))["path"])')
"$PYTHON" -m reasoning.download_aime25
"$PYTHON" - <<'PY'
from reasoning.data import read_rows,question,write_json
from reasoning.prepare import ExclusionIndex
index=ExclusionIndex([question(r) for r in read_rows('artifacts/benchmarks_extended/aime25/test.jsonl')])
overlaps=[r['id'] for r in read_rows('artifacts/dev-pilot/acereason_dev/test.jsonl') if index.matches(question(r))]
write_json('artifacts/benchmarks_extended/pilot_overlap_audit.json',dict(pilot_development_overlap_ids=overlaps))
print('Pilot/AIME25 overlap count:',len(overlaps),flush=True)
if overlaps:
    raise ValueError('Revise the pilot development set before using it to select a full-suite model.')
PY
"$PYTHON" -m reasoning.extend_exclusions --extra-exclusion artifacts/benchmarks_extended/aime25/test.jsonl
if ! "$PYTHON" -m reasoning.reuse_features --input artifacts/prepared_extended/pool.json --from-feature artifacts/features.json --output artifacts/features_extended.json --model "$MODEL"; then
  "$PYTHON" -m reasoning.features --input artifacts/prepared_extended/pool.json --output artifacts/features_extended.json --model "$MODEL"
fi
if ! "$PYTHON" -m reasoning.reuse_features --input artifacts/prepared_extended/validation.json --from-feature artifacts/validation_features.json --output artifacts/validation_features_extended.json --model "$MODEL"; then
  "$PYTHON" -m reasoning.features --input artifacts/prepared_extended/validation.json --output artifacts/validation_features_extended.json --model "$MODEL"
fi
"$PYTHON" -m reasoning.select --pool artifacts/prepared_extended/pool.json --validation artifacts/prepared_extended/validation.json --features artifacts/features_extended.json --validation-features artifacts/validation_features_extended.json --strategy route --cutoff 8192 --size 15000 --output artifacts/datasets-final-8k
"$PYTHON" -m unittest discover -s tests -v
