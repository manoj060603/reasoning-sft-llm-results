#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
PYTHON="$PWD/.venv-arc/bin/python"

"$PYTHON" -m reasoning.concise \
  --pool artifacts/prepared_extended/pool.json \
  --features artifacts/features_extended.json \
  --output artifacts/datasets-concise-2k \
  --size 2000 --seed 271828 --cutoff 4096 \
  --max-response-tokens 1536 --max-repetition .10 --max-box-tail-chars 16

"$PYTHON" -m reasoning.devset \
  --input artifacts/prepared_extended/validation.json \
  --exclude artifacts/dev-rescue/acereason_dev/test.jsonl \
  --output artifacts/dev-concise-confirm/acereason_dev/test.jsonl \
  --size 200 --seed 271828

"$PYTHON" -m unittest discover -s tests -v
