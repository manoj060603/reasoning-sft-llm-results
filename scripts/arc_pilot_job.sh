#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
# Preparation is on a different SLURM cluster, so use its atomic completion
# manifest rather than a cross-cluster job dependency. Bound the wait.
for attempt in $(seq 1 60); do
  if [[ -f artifacts/datasets-route/selection_manifest.json ]]; then
    exec bash scripts/arc_pilot.sh
  fi
  sleep 15
done
echo 'Preparation did not finish within 15 minutes; no training was started.' >&2
exit 3
