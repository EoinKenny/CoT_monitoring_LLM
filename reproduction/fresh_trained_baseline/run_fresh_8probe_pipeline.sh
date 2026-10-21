#!/usr/bin/env bash
set -euo pipefail

EXP="/home/ubuntu/experiments/fresh-8probe-qwen-20260930"
BASE="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z"
REPO="$BASE/repo"
SCRIPTS="$EXP/scripts"
ART="$EXP/artifacts"
PROBES="$REPO/probe/probes_fresh_layer27"
DATA="$REPO/probe/data/v4_v2_6_combined_v2"
HARD="$REPO/ood_misalignment_eval/hard-negatives/rollouts/glm-4.7-flash/hard_negatives"
POS="$REPO/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
VLLM_PY="/home/ubuntu/experiments/paper-full-baseline-20260930/vllm-venv/bin/python"

mkdir -p "$EXP/logs" "$ART" "$EXP/offload/glm" "$EXP/offload/qwen"
exec 9>"$EXP/pipeline.lock"
flock -n 9 || { echo "fresh pipeline is already running"; exit 1; }
exec > >(tee -a "$EXP/logs/pipeline.log") 2>&1

date -Is > "$EXP/started.txt"
git -C "$REPO" rev-parse HEAD > "$EXP/repo-commit.txt"
git -C "$REPO" diff > "$EXP/repo-required-patches.diff"
ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$EXP/keepalives-before.txt"
nvidia-smi > "$EXP/gpu-before.txt"
find "$DATA" -maxdepth 1 -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum > "$EXP/training-data-sha256.txt"

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PROBE_GPU_MEMORY="38GiB" PROBE_CPU_MEMORY="105GiB"
export PROBE_OFFLOAD_DIR="$EXP/offload/glm"
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"
cd "$REPO"

echo "[$(date -Is)] train/resume eight deception-focused public-data probes"
"$PY" -m probe.train \
  --indicator-set v2_6 \
  --indicator \
    "Action Concealment" \
    "Fabrication" \
    "Misalignment Cover Story Planning" \
    "Observer Suspicion Modeling" \
    "Recognized-Problem Suppression" \
    "Strategic Attention Manipulation" \
    "Strategic Framing" \
    "Strategic Omission" \
  --label-mode span \
  --reg-coeff 10.0 \
  --detect-layers 27 \
  --reasoning-positive-only \
  --threshold-mode sentence \
  --val-fraction 0.2 \
  --data-dir "$DATA" \
  --output-dir "$PROBES"

"$PY" "$SCRIPTS/validate_fresh_probes.py" \
  --probe-root "$PROBES" --output "$ART/fresh_probe_manifest.json"
touch "$EXP/TRAINING_COMPLETED"

echo "[$(date -Is)] evaluate fresh probes on hard negatives and DeceptionBench positives"
"$PY" "$SCRIPTS/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" \
  --rollout-dir "$HARD" --expected-probes 8 --all-negative
"$PY" "$SCRIPTS/evaluate_labeled.py" --repo "$REPO" --probe-root "$PROBES" \
  --rollout-dir "$POS" --expected-probes 8
"$PY" "$SCRIPTS/analyze_mixed.py" --repo "$REPO" --probe-root "$PROBES" \
  --hard-negative-dir "$HARD" --positive-dir "$POS" --artifacts "$ART" --expected-probes 8
touch "$EXP/PROBE_EVALUATION_COMPLETED"

echo "[$(date -Is)] run hard-classification Qwen escalation through vLLM"
unset PROBE_OFFLOAD_DIR
export VLLM_USE_FLASHINFER_SAMPLER=0
"$VLLM_PY" "$SCRIPTS/run_mixed_qwen_binary_vllm.py" \
  --repo "$REPO" --artifacts "$ART" --batch-size 8 \
  --cpu-offload-gb 28 --gpu-memory-utilization 0.88
"$VLLM_PY" "$SCRIPTS/finalize_mixed_qwen_binary.py" --artifacts "$ART"
"$PY" "$SCRIPTS/make_report.py" --artifacts "$ART" --experiment "$EXP"

ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$EXP/keepalives-after.txt"
nvidia-smi > "$EXP/gpu-after.txt"
date -Is > "$EXP/completed.txt"
touch "$EXP/COMPLETED"
echo "[$(date -Is)] fresh-trained probe pipeline complete"
