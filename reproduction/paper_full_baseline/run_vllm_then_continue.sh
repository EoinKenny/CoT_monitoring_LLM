#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/experiments/paper-full-baseline-20260930"
OLD="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/released-18probe-qwen"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
VLLM_PY="$BASE/vllm-venv/bin/python"
PROBE_PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO"

"$VLLM_PY" "$BASE/scripts/run_mixed_qwen_vllm.py" \
  --repo "$REPO" --artifacts "$OLD/artifacts" \
  --batch-size 8 --cpu-offload-gb 28 --gpu-memory-utilization 0.88
"$PROBE_PY" "$OLD/scripts/make_mixed_report.py" \
  --artifacts "$OLD/artifacts" --experiment "$OLD"
touch "$BASE/QWEN_VLLM_COMPLETED"

bash "$BASE/scripts/after_primary_run.sh"
