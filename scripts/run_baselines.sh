#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"

adseq() { uv run --no-sync adseq "$@"; }

echo "$(date +%H:%M) evaluating persistence ridge sj_mlp"
adseq evaluate --models persistence ridge sj_mlp "${FLAGS[@]}" >>"$LOGS/baselines.log" 2>&1 ||
  { echo "$(date +%H:%M) failed: see $LOGS/baselines.log"; exit 1; }

echo "$(date +%H:%M) done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
