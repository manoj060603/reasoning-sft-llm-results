#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
PYTHON="$PWD/.venv-arc/bin/python"
"$PYTHON" -m reasoning.select \
  --pool artifacts/prepared_extended/pool.json \
  --validation artifacts/prepared_extended/validation.json \
  --features artifacts/features_extended.json \
  --validation-features artifacts/validation_features_extended.json \
  --strategy capacity_mix --short-cutoff 3072 --short-fraction 0.6666666666666666 \
  --cutoff 4096 --size 15000 --output artifacts/datasets-rescue-4k
"$PYTHON" - <<'PY'
import json
from reasoning.data import read_rows,write_json
manifest=json.load(open('artifacts/datasets-rescue-4k/selection_manifest.json'))
features={r['id']:r for r in read_rows('artifacts/features_extended.json')}
ids=manifest['selected_ids']['advanced']
short=sum(features[i]['total_tokens']<=3072-32 for i in ids)
response_tokens=manifest['response_tokens']['advanced']
assert len(ids)==len(set(ids))==15000
assert short>=10000
assert response_tokens<51_000_000
write_json('artifacts/datasets-rescue-4k/capacity_audit.json',{
    'examples':len(ids),'complete_sequences_at_most_3040_tokens':short,
    'response_tokens':response_tokens,'mean_response_tokens':response_tokens/len(ids),
    'design':'10000 route-diverse short traces plus 5000 route-diverse medium traces'})
PY
"$PYTHON" -m reasoning.devset --input artifacts/prepared_extended/validation.json \
  --output artifacts/dev-rescue/acereason_dev/test.jsonl --size 200 --seed 314159
"$PYTHON" -m unittest discover -s tests -v
