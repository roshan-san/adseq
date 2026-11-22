#!/usr/bin/env bash
# Every result in the paper, from a fresh clone: scripts/setup_and_run.sh
# To skip something, comment out its line: a model in the lists below, or a step at the bottom.
# Ctrl-C stops everything. Re-run to resume: finished trials, data and environment are kept.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- settings ---------------------------------------------------------------------------------
NAME=main             # results: runs/$NAME/ and the MLflow experiment "$NAME"
TRIALS=50             # Optuna trials per tuned model (evaluate needs this many finished)
SEEDS=(1 2 3 4 5)     # evaluation seeds
AT_ONCE=3             # recurrent models trained at the same time (6 ran out of memory on a laptop GPU)

BASELINES=(           # nothing to tune: evaluated only
  persistence
  ridge
  sj_mlp
)
RECURRENT=(           # tuned then evaluated, AT_ONCE at a time, sharing the CPU cores
  rnn_res
  gru_res
  lstm_res
  rnn
  gru
  lstm
)
LGBM=(                # tuned then evaluated after the recurrent models, alone: needs several GB
  lgbm_res
  lgbm
)

# ---- steps ------------------------------------------------------------------------------------
LOGS=runs/logs
mkdir -p "$LOGS"
trap 'trap - INT TERM; echo; echo "stopping everything"; kill 0' INT TERM  # background jobs too

say() { echo "$(date +%H:%M) $*"; }
adseq() { uv run --no-sync adseq "$@"; }

setup_environment() {
  local extra
  if nvidia-smi >/dev/null 2>&1; then extra=cu132; else extra=cpu; fi
  say "[setup] installing the environment (PyTorch: $extra)"
  uv sync --extra "$extra" -q
  if ! uv run --no-sync python -c "import torch; print('PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name() if torch.cuda.is_available() else 'none')
assert torch.cuda.is_available() or '$extra' == 'cpu'"; then
    say "an NVIDIA GPU is present but PyTorch cannot use it: update the NVIDIA driver"
    say "(nvidia-smi must show CUDA Version >= 13.2), then re-run"
    exit 1
  fi
}

prepare_data() {
  say "[data] building data/processed/SCADA-FINAL.csv (skipped once built)"
  uv run --script prepare_data.py
}

run_baselines() {
  [ ${#BASELINES[@]} -gt 0 ] || return 0
  say "[baselines] evaluating ${BASELINES[*]}, seeds ${SEEDS[*]} (log: $LOGS/baselines.log)"
  adseq evaluate --name "$NAME" --models "${BASELINES[@]}" --seeds "${SEEDS[@]}" \
    >>"$LOGS/baselines.log" 2>&1 || { say "failed: see $LOGS/baselines.log"; exit 1; }
}

run_recurrent() {
  [ ${#RECURRENT[@]} -gt 0 ] || return 0
  local at_once=$(( AT_ONCE < ${#RECURRENT[@]} ? AT_ONCE : ${#RECURRENT[@]} )) m failed
  local threads=$(( $(nproc) / at_once ))
  [ "$threads" -ge 1 ] || threads=1
  say "[recurrent] ${RECURRENT[*]}: $TRIALS trials, then seeds ${SEEDS[*]}"
  say "[recurrent] $at_once at a time, $threads CPU threads each"
  say "[recurrent] progress: scripts/mlflow_ui.sh in another window, then http://127.0.0.1:5000"
  rm -f "$LOGS"/*.failed
  for m in "${RECURRENT[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$at_once" ]; do wait -n || true; done  # a slot frees up
    (export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
     if ! { adseq tune --name "$NAME" --models "$m" --n-trials "$TRIALS" &&
            adseq evaluate --name "$NAME" --models "$m" --n-trials "$TRIALS" --seeds "${SEEDS[@]}"; }
     then touch "$LOGS/$m.failed"; fi
    ) >>"$LOGS/$m.log" 2>&1 &
    say "[recurrent] started $m (log: $LOGS/$m.log)"
    sleep 15  # each start loads the data (~0.8 GB peak): not all at once
  done
  wait
  failed=""
  for m in "${RECURRENT[@]}"; do [ ! -e "$LOGS/$m.failed" ] || failed+="$m "; done
  [ -z "$failed" ] || { say "failed: $failed(see $LOGS/); re-run to resume"; exit 1; }
  say "[recurrent] all finished"
}

run_lgbm() {
  [ ${#LGBM[@]} -gt 0 ] || return 0
  say "[lgbm] ${LGBM[*]}: $TRIALS trials, then seeds ${SEEDS[*]}, every core (log: $LOGS/lgbm.log)"
  { adseq tune --name "$NAME" --models "${LGBM[@]}" --n-trials "$TRIALS" &&
      adseq evaluate --name "$NAME" --models "${LGBM[@]}" --n-trials "$TRIALS" --seeds "${SEEDS[@]}"
  } >>"$LOGS/lgbm.log" 2>&1 || { say "lgbm failed: see $LOGS/lgbm.log; re-run to resume"; exit 1; }
}

write_report() {
  say "[report] table and figure from MLflow (models not evaluated yet are left out)"
  adseq report --name "$NAME"
  say "done: runs/$NAME/summary.csv and runs/$NAME/figures/"
}

# ---- run: comment out a step to skip it -------------------------------------------------------
say "experiment '$NAME': $TRIALS trials, seeds ${SEEDS[*]}"
setup_environment
prepare_data
run_baselines
run_recurrent
run_lgbm
write_report
