#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/experiments/paper-full-baseline-20260930"
REPO="/home/ubuntu/experiments/probe-qwen-retrain-20260928T200000Z/repo"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"

until [[ -f "$BASE/DATA_READY" && -f "$BASE/THRESHOLDS_READY" ]]; do sleep 30; done

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PROBE_GPU_MEMORY="38GiB" PROBE_CPU_MEMORY="105GiB"
export PROBE_OFFLOAD_DIR="$BASE/offload/glm-all-data"
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"
mkdir -p "$PROBE_OFFLOAD_DIR" "$BASE/staged/glm"

ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$BASE/keepalives-before-full-glm.txt"
nvidia-smi > "$BASE/gpu-before-full-glm.txt"
"$PY" "$BASE/scripts/run_probe_family.py" \
  --repo "$REPO" \
  --probe-root "$REPO/probe/probes_paper_main/v4_v2_6_combined_v2_span" \
  --catalog "$BASE/dataset_catalog.json" \
  --model-family glm \
  --staging "$BASE/staged/glm" \
  --layers 27 29 \
  --max-sequence-tokens 8192
"$PY" "$BASE/scripts/summarize_coverage.py" \
  --safety-dir "$REPO/probe_eval/results/sequence_length_safety" \
  --output "$BASE/all_glm_token_coverage.json"
ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$BASE/keepalives-after-full-glm.txt"
nvidia-smi > "$BASE/gpu-after-full-glm.txt"
touch "$BASE/ALL_GLM_INFERENCE_READY"
