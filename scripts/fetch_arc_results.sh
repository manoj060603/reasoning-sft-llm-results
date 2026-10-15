#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEST="${1:-$ROOT/results/arc-20261011}"
mkdir -p "$DEST/evaluations"
# Never delete local files. Use a fresh destination after remote run renames.
rsync -az --include='*/' --include='*.json' --exclude='*' \
  tc:/projects/tealab/rohit/reasoning-sft-20261011/results/ "$DEST/evaluations/"
scp -q tc:/projects/tealab/rohit/reasoning-sft-20261011/artifacts/prepared/manifest.json "$DEST/preparation.json"
scp -q tc:/projects/tealab/rohit/reasoning-sft-20261011/artifacts/arc-environment.txt "$DEST/environment.txt"
mkdir -p "$DEST/training" "$DEST/selections"
# Receipts and learning curves only; do not copy multi-gigabyte model weights.
if ssh -o BatchMode=yes tc 'test -d /projects/tealab/rohit/reasoning-sft-20261011/runs'; then
  rsync -azm --include='*/' --include='manifest.json' --include='complete.json' \
    --include='hardware.json' --include='data_stats.json' --include='train.jsonl' --exclude='*' \
    tc:/projects/tealab/rohit/reasoning-sft-20261011/runs/ "$DEST/training/"
fi
rsync -azm --include='/datasets-route/' --include='/datasets-pilot/' --include='/datasets-final-8k/' \
  --include='selection_manifest.json' --exclude='*' \
  tc:/projects/tealab/rohit/reasoning-sft-20261011/artifacts/ "$DEST/selections/"
mkdir -p "$DEST/extended-benchmarks"
rsync -azm --include='/benchmarks_extended/' --include='/prepared_extended/' \
  --include='supplement.json' --include='pilot_overlap_audit.json' --include='manifest.json' --exclude='*' \
  tc:/projects/tealab/rohit/reasoning-sft-20261011/artifacts/ "$DEST/extended-benchmarks/"
