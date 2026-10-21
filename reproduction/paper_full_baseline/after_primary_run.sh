#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/experiments/paper-full-baseline-20260930"
OLD="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/released-18probe-qwen"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
PID="$(cat "$OLD/pid")"

while kill -0 "$PID" 2>/dev/null; do sleep 30; done

DEV=(
  instructed-covert-code-sabotage_glm_4_7_flash
  instructed-covert-code-sabotage_benign_glm_4_7_flash
  instructed-strategic-sandbagging_glm_4_7_flash
  instructed-strategic-sandbagging_benign_glm_4_7_flash
  sycophancy sycophancy_benign_glm_4_7_flash
  self-preservation_glm_4_7_flash self-preservation_benign_glm_4_7_flash
  strategic-deception_glm_4_7_flash strategic-deception_benign_glm_4_7_flash
)

DEV_ARGS=()
for d in "${DEV[@]}"; do DEV_ARGS+=(--rollout-dir "$REPO/bloom/bloom-results/$d"); done
export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PROBE_GPU_MEMORY="38GiB" PROBE_CPU_MEMORY="105GiB"
export PROBE_OFFLOAD_DIR="$BASE/offload/glm"
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"
mkdir -p "$PROBE_OFFLOAD_DIR"
"$PY" "$BASE/scripts/evaluate_dataset_list.py" \
  --repo "$REPO" \
  --probe-root "$REPO/probe/probes_paper_main/v4_v2_6_combined_v2_span" \
  --layer 29 "${DEV_ARGS[@]}"

COVERAGE_ARGS=()
for d in "${DEV[@]}"; do COVERAGE_ARGS+=(--dataset "$d"); done
"$PY" "$BASE/scripts/summarize_coverage.py" \
  --safety-dir "$REPO/probe_eval/results/sequence_length_safety" \
  --output "$BASE/dev_token_coverage.json" "${COVERAGE_ARGS[@]}"

bash "$BASE/scripts/reconstruct_thresholds.sh" "$REPO" \
  v4_v2_6_combined_v2_span "$BASE/hardware_safe_thresholds_paper_17.json" paper
bash "$BASE/scripts/reconstruct_thresholds.sh" "$REPO" \
  v4_v2_6_combined_v2_span "$BASE/hardware_safe_thresholds_all_18.json" all
touch "$BASE/THRESHOLDS_READY"
