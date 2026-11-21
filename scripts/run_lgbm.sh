#!/usr/bin/env bash
# Tune + evaluate LightGBM (lgbm_res, lgbm), evaluate the untuned baselines (persistence, ridge,
# Schroer & Just's MLP), then write the report.
#
#   scripts/run_lgbm.sh [adseq flags, e.g. --name main --n-trials 20]
#
# Run it inside tmux (`tmux new -s adseq`, detach with Ctrl-b d) so it survives closing the
# terminal. The recurrent models: scripts/run_recurrent.sh. Best run after it, alone: LightGBM's
# evaluation trains on every minute, and its flattened windows (up to 72 h x 36 features = 2,592
# columns) take several GB. LightGBM uses every core.
#
# Log: runs/logs/lgbm.log. Re-running resumes: Optuna studies continue from runs/<name>/optuna/,
# finished trials are not repeated.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"
adseq() { uv run --no-sync adseq "$@"; }

echo "$(date +%H:%M) tuning lgbm_res lgbm"
adseq tune --models lgbm_res lgbm "${FLAGS[@]}" >>"$LOGS/lgbm.log" 2>&1 ||
  { echo "$(date +%H:%M) tuning failed: see $LOGS/lgbm.log; re-run this script to resume"; exit 1; }

echo "$(date +%H:%M) evaluating lgbm_res lgbm persistence ridge"
adseq evaluate --models lgbm_res lgbm persistence ridge "${FLAGS[@]}" >>"$LOGS/lgbm.log" 2>&1 ||
  { echo "$(date +%H:%M) evaluation failed: see $LOGS/lgbm.log"; exit 1; }

echo "$(date +%H:%M) done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
