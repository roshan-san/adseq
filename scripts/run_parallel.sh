#!/usr/bin/env bash
# Tune + evaluate every model in parallel on one CPU machine, then write the report.
#
#   scripts/run_parallel.sh [adseq flags, e.g. --name main --n-trials 20]
#
# Run it inside tmux (`tmux new -s adseq`, detach with Ctrl-b d) so it survives closing the
# terminal.
#
# Tuning uses TUNE_FLAGS: every 10th minute for train / val (neighbouring minutes are
# near-duplicates), with the epoch limits scaled to match. Evaluation, the reported numbers,
# trains on every minute with the default limits. Your flags come last, so they override both.
#
# Each recurrent model gets its own process with ONE thread: these small RNNs train faster on
# one thread than on eight (measured 4.5 vs 8.2 s/epoch), so six single-thread processes get far
# more done than one process using every core. LightGBM and the untrained baselines share a
# seventh process with two threads. Starts are staggered because loading the CSV peaks at ~0.9 GB
# per process (~0.45 GB once loaded). Logs: runs/logs/<model>.log. Re-running resumes: Optuna
# studies continue from runs/<name>/optuna/, finished trials are not repeated.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
TUNE_FLAGS=(--stride 10 --max-epochs 200 --patience 20)
LOGS=runs/logs
mkdir -p "$LOGS"
adseq() { uv run --no-sync adseq "$@"; }  # no concurrent re-syncs of the environment

uv run --no-sync python -c "import torch" ||
  { echo "install a PyTorch build first: uv sync --extra cpu (or --extra cu132)"; exit 1; }

pids=()
for m in lstm_res lstm gru_res gru rnn_res rnn; do  # slowest first
  (export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   adseq tune --models "$m" "${TUNE_FLAGS[@]}" "${FLAGS[@]}" &&
     adseq evaluate --models "$m" "${FLAGS[@]}"
  ) >"$LOGS/$m.log" 2>&1 &
  pids+=($!)
  echo "$(date +%H:%M) started $m (pid $!)"
  sleep 60
done
(export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
 adseq tune --models lgbm_res lgbm "${TUNE_FLAGS[@]}" "${FLAGS[@]}" &&
   adseq evaluate --models lgbm_res lgbm persistence ridge "${FLAGS[@]}"
) >"$LOGS/trees_and_baselines.log" 2>&1 &
pids+=($!)
echo "$(date +%H:%M) started lgbm_res lgbm persistence ridge (pid $!)"

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [ "$failed" = 1 ]; then
  echo "$(date +%H:%M) a model failed: see $LOGS/*.log; re-run this script to resume"
  exit 1
fi
echo "$(date +%H:%M) all models done; writing the report"
adseq report "${FLAGS[@]}"
