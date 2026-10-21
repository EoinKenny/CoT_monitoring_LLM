#!/usr/bin/env bash
set -uo pipefail

BASE="/home/ubuntu/experiments/paper-full-baseline-20260930"
OLD="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/released-18probe-qwen"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
VLLM_PY="$BASE/vllm-venv/bin/python"

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO"
export VLLM_USE_FLASHINFER_SAMPLER=0

status=0
"$VLLM_PY" "$BASE/scripts/run_mixed_qwen_binary_vllm.py" \
  --repo "$REPO" --artifacts "$OLD/artifacts" \
  --batch-size 8 --cpu-offload-gb 28 --gpu-memory-utilization 0.88 \
  || status=$?

if [[ "$status" -eq 0 ]]; then
  touch "$BASE/QWEN_BINARY_COMPLETED"
fi

# The all-dataset GLM pass is resumable (skip_existing=True). Always resume it,
# even if Qwen fails, unless an instance is already active.
if ! pgrep -f "run_probe_family.py.*--model-family glm" >/dev/null; then
  nohup bash "$BASE/scripts/run_full_glm_after_setup.sh" \
    > "$BASE/logs/full-glm-resumed-after-binary.log" 2>&1 < /dev/null &
  echo "$!" > "$BASE/full-glm-resumed-after-binary.pid"
fi

exit "$status"
