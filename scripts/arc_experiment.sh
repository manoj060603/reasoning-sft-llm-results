#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
bash scripts/arc_baseline.sh
if [[ ! -f artifacts/datasets-route/selection_manifest.json ]]; then
  echo 'BASELINE_COMPLETE; CPU preparation is not finished. Pilot must be submitted after preparation.'
  exit 3
fi
bash scripts/arc_pilot.sh
