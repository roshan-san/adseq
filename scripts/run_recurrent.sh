#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS=("$@")
LOGS=runs/logs
mkdir -p "$LOGS"

adseq() { uv run --no-sync adseq "$@"; } 


pids=()
for m in lstm_res lstm gru_res gru rnn_res rnn; do 
  (export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   adseq tune --models "$m" "${FLAGS[@]}" &&
     adseq evaluate --models "$m" "${FLAGS[@]}"
  ) >"$LOGS/$m.log" 2>&1 &
  pids+=($!)
  echo "$(date +%H:%M) started $m (pid $!)"
  sleep 60
done

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [ "$failed" = 1 ]; then
  echo "$(date +%H:%M) a model failed: see $LOGS/*.log"
  exit 1
fi

echo "$(date +%H:%M) recurrent models done; writing the report (models not run yet are left out)"
adseq report "${FLAGS[@]}"
