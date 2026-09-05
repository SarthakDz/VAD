#!/usr/bin/env bash
# M0 smoke test: harness round-trips without any model in the loop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
ROOT="${1:-../Train and Test}"
GT="$ROOT/test/ground_truth.csv"

echo "== labels =="        && $PY -m src.labels | tail -2
echo "== dataset =="       && $PY -m src.io_dataset "$ROOT" | tail -4
echo "== baselines =="
$PY -m src.submit --root "$ROOT" --class-name normal           --out outputs/baseline_normal.csv
$PY -m src.submit --root "$ROOT" --class-name traffic_accident --out outputs/baseline_accident.csv
echo "== score =="         && $PY -m src.score --pred outputs/baseline_accident.csv --gt "$GT"
echo "SMOKE OK"
