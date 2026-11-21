#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"

adseq() { uv run --no-sync adseq "$@"; }

echo "$(date +%H:%M) tuning lgbm_res lgbm"
adseq tune --models lgbm_res lgbm "${FLAGS[@]}" >>"$LOGS/lgbm.log" 2>&1 ||
  { echo "$(date +%H:%M) tuning failed: see $LOGS/lgbm.log; re-run this script to resume"; exit 1; }

echo "$(date +%H:%M) evaluating lgbm_res lgbm"
adseq evaluate --models lgbm_res lgbm "${FLAGS[@]}" >>"$LOGS/lgbm.log" 2>&1 ||
  { echo "$(date +%H:%M) evaluation failed: see $LOGS/lgbm.log"; exit 1; }

echo "$(date +%H:%M) done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
