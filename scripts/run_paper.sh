#!/usr/bin/env bash
# Everything for the paper's results from a fresh clone: environment (CUDA PyTorch when an NVIDIA
# GPU is visible), data, baselines, MODELS tuned + evaluated in parallel, report. Re-run to resume.
# Settings as env vars, e.g.  TRIALS=10 MODELS="gru_res gru lstm_res" scripts/run_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-main}
TRIALS=${TRIALS:-20}
SEEDS=${SEEDS:-"1 2 3"}
MODELS=${MODELS:-"gru_res gru"}
LOGS=runs/logs
mkdir -p "$LOGS"

say() { echo "$(date +%H:%M) $*"; }
adseq() { uv run --no-sync adseq "$@"; }

# 1. environment
if nvidia-smi >/dev/null 2>&1; then extra=cu132; else extra=cpu; fi
say "installing the environment (PyTorch: $extra)"
uv sync --extra "$extra" -q
if ! uv run --no-sync python -c "import torch; print('PyTorch', torch.__version__, '| GPU:',
    torch.cuda.get_device_name() if torch.cuda.is_available() else 'none')
assert torch.cuda.is_available() or '$extra' == 'cpu'"; then
  say "an NVIDIA GPU is present but PyTorch cannot use it: update the NVIDIA driver"
  say "(nvidia-smi must show CUDA Version >= 13.2), then re-run"
  exit 1
fi

# 2. data (skipped once built)
uv run --script prepare_data.py

# 3. baselines
say "evaluating the baselines (persistence, ridge, sj_mlp)"
# shellcheck disable=SC2086  # SEEDS splits into one argument per seed
adseq evaluate --name "$NAME" --models persistence ridge sj_mlp --seeds $SEEDS \
  >>"$LOGS/baselines.log" 2>&1 || { say "failed: see $LOGS/baselines.log"; exit 1; }

# 4. models: tune then evaluate, one process each, all at once
pids=()
for m in $MODELS; do
  # shellcheck disable=SC2086
  (adseq tune --name "$NAME" --models "$m" --n-trials "$TRIALS" &&
    adseq evaluate --name "$NAME" --models "$m" --n-trials "$TRIALS" --seeds $SEEDS
  ) >>"$LOGS/$m.log" 2>&1 &
  pids+=($!)
  say "started $m: $TRIALS trials, then seeds $SEEDS (log: $LOGS/$m.log)"
done
say "progress: scripts/mlflow_ui.sh in another window, then http://127.0.0.1:5000"
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [ "$failed" = 1 ]; then
  say "a model failed: see $LOGS/; re-run this script to resume"
  exit 1
fi

# 5. report
adseq report --name "$NAME"
say "done: runs/$NAME/summary.csv and runs/$NAME/figures/"
