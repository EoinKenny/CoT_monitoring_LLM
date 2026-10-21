#!/usr/bin/env bash
set -euo pipefail

EXP="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z"
RUN="$EXP/released-18probe-qwen"
REPO="$EXP/repo"
ART="$RUN/artifacts"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
PROBES="$REPO/probe/probes_paper_main/v4_v2_6_combined_v2_span"
HARD="$REPO/ood_misalignment_eval/hard-negatives/rollouts/glm-4.7-flash/hard_negatives"
POS="$REPO/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
THRESHOLD_MANIFEST="$RUN/reconstructed_tuned_thresholds.json"
RESULTS_SUBDIR="v4_v2_6_combined_v2_span"

DEV_POSITIVE=(
  "$REPO/bloom/bloom-results/instructed-covert-code-sabotage_glm_4_7_flash"
  "$REPO/bloom/bloom-results/instructed-strategic-sandbagging_glm_4_7_flash"
  "$REPO/bloom/bloom-results/sycophancy"
  "$REPO/bloom/bloom-results/self-preservation_glm_4_7_flash"
  "$REPO/bloom/bloom-results/strategic-deception_glm_4_7_flash"
)
DEV_BENIGN=(
  "$REPO/bloom/bloom-results/instructed-covert-code-sabotage_benign_glm_4_7_flash"
  "$REPO/bloom/bloom-results/instructed-strategic-sandbagging_benign_glm_4_7_flash"
  "$REPO/bloom/bloom-results/sycophancy_benign_glm_4_7_flash"
  "$REPO/bloom/bloom-results/self-preservation_benign_glm_4_7_flash"
  "$REPO/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
)

mkdir -p "$ART" "$RUN/logs" "$RUN/offload/glm" "$RUN/offload/qwen"
exec > >(tee -a "$RUN/logs/run.log") 2>&1
export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PROBE_GPU_MEMORY="38GiB" PROBE_CPU_MEMORY="105GiB" PROBE_OFFLOAD_DIR="$RUN/offload/glm"
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"

date -Is > "$RUN/started.txt"
ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$RUN/keepalives-before.txt"
nvidia-smi > "$RUN/gpu-before.txt"

echo "[$(date -Is)] scoring the paper's five positive development folders"
for d in "${DEV_POSITIVE[@]}"; do
  "$PY" "$RUN/scripts/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$d"
done
echo "[$(date -Is)] scoring the paper's five benign development folders"
for d in "${DEV_BENIGN[@]}"; do
  "$PY" "$RUN/scripts/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$d" --all-negative
done

echo "[$(date -Is)] reconstructing all ten paper threshold operating points on dev"
for fpr in 0.002 0.005 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3; do
  "$PY" -m probe_eval.misalignment_ground_truth \
    --results-subdir "$RESULTS_SUBDIR" \
    --include-all-negative \
    --exclude-behaviors "test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback,mask,hard_negatives*" \
    --output-suffix "_dev_fpr_${fpr}" \
    --tune-thresholds --fpr-target "$fpr" \
    --tuned-thresholds-save "$THRESHOLD_MANIFEST" \
    --short-sentence-mode discard
done

echo "[$(date -Is)] scoring hard negatives with 18 released probes"
"$PY" "$RUN/scripts/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$HARD" --all-negative
echo "[$(date -Is)] scoring DeceptionBench positives with 18 released probes"
"$PY" "$RUN/scripts/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$POS"
echo "[$(date -Is)] assembling the frozen fpr_0.02 test results"
"$PY" "$RUN/scripts/analyze_mixed.py" --repo "$REPO" --probe-root "$PROBES" --hard-negative-dir "$HARD" --positive-dir "$POS" --artifacts "$ART" --threshold-manifest "$THRESHOLD_MANIFEST" --threshold-version fpr_0.02
echo "[$(date -Is)] judging every probe-positive turn with local Qwen"
"$PY" "$RUN/scripts/run_mixed_qwen_judge.py" --repo "$REPO" --artifacts "$ART" --offload-dir "$RUN/offload/qwen"
"$PY" "$RUN/scripts/make_mixed_report.py" --artifacts "$ART" --experiment "$RUN"

ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$RUN/keepalives-after.txt"
nvidia-smi > "$RUN/gpu-after.txt"
date -Is > "$RUN/completed.txt"
touch "$RUN/COMPLETED"
echo "[$(date -Is)] released-probe baseline complete"
