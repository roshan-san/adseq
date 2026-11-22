#!/usr/bin/env bash
# MLflow UI over every experiment: http://127.0.0.1:5000 (extra flags pass through, e.g. --port 5001)
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-sync mlflow ui --backend-store-uri sqlite:///runs/mlflow.db "$@"
