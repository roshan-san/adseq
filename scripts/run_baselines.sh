#!/usr/bin/env bash
# Evaluate the baselines (persistence, ridge: nothing to tune), then write the report.
#
#   scripts/run_baselines.sh [adseq flags, e.g. --name main]
#
# A few minutes on a CPU. Log: runs/logs/baselines.log.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"
adseq() { uv run --no-sync adseq "$@"; }

echo "$(date +%H:%M) evaluating persistence ridge"
adseq evaluate --models persistence ridge "${FLAGS[@]}" >>"$LOGS/baselines.log" 2>&1 ||
  { echo "$(date +%H:%M) failed: see $LOGS/baselines.log"; exit 1; }

echo "$(date +%H:%M) done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
