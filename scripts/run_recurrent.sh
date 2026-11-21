#!/usr/bin/env bash
# Tune + evaluate the six recurrent models in parallel on one CPU machine, then write the report.
#
#   scripts/run_recurrent.sh [adseq flags, e.g. --name main --n-trials 20]
#
# Run it inside tmux (`tmux new -s adseq`, detach with Ctrl-b d) so it survives closing the
# terminal. LightGBM and the baselines: scripts/run_lgbm.sh, best run once this one is done, as
# LightGBM's evaluation needs several GB next to these six processes.
#
# Each model gets its own process with ONE thread: these small RNNs train faster on one thread
# than on eight (measured 4.5 vs 8.2 s/epoch), so six single-thread processes get far more done
# than one process using every core. Starts are staggered because loading the CSV peaks at
# ~0.9 GB per process (~0.45 GB once loaded).
#
# Logs: runs/logs/<model>.log. Re-running resumes: Optuna studies continue from
# runs/<name>/optuna/, finished trials are not repeated.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"
adseq() { uv run --no-sync adseq "$@"; }  # no concurrent re-syncs of the environment

uv run --no-sync python -c "import torch" ||
  { echo "install a PyTorch build first: uv sync --extra cpu (or --extra cu132)"; exit 1; }

pids=()
for m in lstm_res lstm gru_res gru rnn_res rnn; do  # slowest first
  (export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   adseq run --models "$m" "${FLAGS[@]}"
  ) >"$LOGS/$m.log" 2>&1 &
  pids+=($!)
  echo "$(date +%H:%M) started $m (pid $!)"
  sleep 60
done

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [ "$failed" = 1 ]; then
  echo "$(date +%H:%M) a model failed: see $LOGS/*.log; re-run this script to resume"
  exit 1
fi

echo "$(date +%H:%M) recurrent models done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
