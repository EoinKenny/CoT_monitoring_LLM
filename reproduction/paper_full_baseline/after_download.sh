#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/experiments/paper-full-baseline-20260930"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
PID="$(cat "$BASE/download.pid")"

while kill -0 "$PID" 2>/dev/null; do sleep 30; done
test -f "$BASE/datasets/hf-release/DOWNLOAD_MANIFEST.json"
"$PY" "$BASE/scripts/inventory_release.py" \
  --repo "$REPO" --hf-data "$BASE/datasets/hf-release" \
  --output "$BASE/release_inventory.json"
"$PY" "$BASE/scripts/build_dataset_catalog.py" \
  --repo "$REPO" --hf-data "$BASE/datasets/hf-release" \
  --output "$BASE/dataset_catalog.json"
touch "$BASE/DATA_READY"
