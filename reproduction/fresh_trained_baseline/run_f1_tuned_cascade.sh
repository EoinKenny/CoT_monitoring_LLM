#!/usr/bin/env bash
set -euo pipefail

EXP="/home/ubuntu/experiments/fresh-8probe-qwen-20260930"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
SCRIPTS="$EXP/scripts"
SOURCE="$EXP/artifacts"
TUNED="$EXP/artifacts/f1_tuned"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
VLLM_PY="/home/ubuntu/experiments/paper-full-baseline-20260930/vllm-venv/bin/python"

mkdir -p "$TUNED" "$EXP/logs"
exec > >(tee -a "$EXP/logs/f1-tuned-cascade.log") 2>&1

# Do not contend with the one-escalation strict run if it is still shutting down.
while pgrep -f "$SCRIPTS/run_fresh_8probe_pipeline.sh" >/dev/null; do sleep 15; done

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"
export VLLM_USE_FLASHINFER_SAMPLER=0

"$PY" "$SCRIPTS/tune_f1_threshold.py" \
  --source-artifacts "$SOURCE" --output-artifacts "$TUNED"
touch "$EXP/F1_THRESHOLD_TUNED"

"$VLLM_PY" "$SCRIPTS/run_mixed_qwen_binary_vllm.py" \
  --repo "$REPO" --artifacts "$TUNED" --batch-size 8 \
  --cpu-offload-gb 28 --gpu-memory-utilization 0.88
"$VLLM_PY" "$SCRIPTS/finalize_mixed_qwen_binary.py" --artifacts "$TUNED"
"$PY" "$SCRIPTS/make_tuned_report.py" \
  --artifacts "$TUNED" --output "$EXP/F1_TUNED_RESULTS.md"

date -Is > "$EXP/f1-tuned-completed.txt"
touch "$EXP/F1_TUNED_COMPLETED"
