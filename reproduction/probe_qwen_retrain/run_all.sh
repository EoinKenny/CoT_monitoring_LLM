#!/usr/bin/env bash
set -euo pipefail

EXP="$1"
SOURCE_REPO="/home/ubuntu/misalignment-indicators-repro"
COMMIT="f1298406b4fbe2834ad1ea1306a6f4195304a123"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"
SCRIPTS="$EXP/scripts"
REPO="$EXP/repo"
ART="$EXP/artifacts"
ROLLOUT="$REPO/ood_misalignment_eval/hard-negatives/rollouts/glm-4.7-flash/hard_negatives"
PROBES="$REPO/probe/probes_fresh_layer27"

mkdir -p "$EXP/logs" "$ART" "$EXP/offload/glm" "$EXP/offload/qwen" "$EXP/hf-cache"
exec > >(tee -a "$EXP/logs/pipeline.log") 2>&1

echo "[$(date -Is)] starting"
if [[ ! -d "$REPO/.git" ]]; then
  git clone "$SOURCE_REPO" "$REPO"
  git -C "$REPO" checkout --detach "$COMMIT"
  git -C "$REPO" apply "$SCRIPTS/model_memory.patch"
fi

git -C "$REPO" rev-parse HEAD > "$EXP/repo-commit.txt"
git -C "$REPO" diff -- ood_misalignment_eval/deception-detection/deception_detection/models.py > "$EXP/bounded-memory-change.diff"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader > "$EXP/gpu-before.txt"
ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$EXP/keepalives-before.txt"
"$PY" - <<'PY' > "$EXP/pip-freeze.txt"
from importlib.metadata import distributions
for dist in sorted(distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
    name = dist.metadata.get("Name")
    if name:
        print(f"{name}=={dist.version}")
PY

export HF_HOME="/home/ubuntu/experiments/hard-negative-repro-20260928T082816Z/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROBE_GPU_MEMORY="38GiB"
export PROBE_CPU_MEMORY="105GiB"
export PROBE_OFFLOAD_DIR="$EXP/offload/glm"
export PYTHONPATH="$REPO:$REPO/ood_misalignment_eval/deception-detection"
cd "$REPO"

echo "[$(date -Is)] training 18 layer-27 probes"
"$PY" -m probe.train \
  --indicator-set v2_6 \
  --indicator \
    "Action Concealment" \
    "Adversarial User Framing" \
    "Concerns on Self-Existence" \
    "Distinct Self-Goal Representation" \
    "Error Calibration" \
    "Fabrication" \
    "Malicious Consequential Action Planning" \
    "Misalignment Cover Story Planning" \
    "Observer Suspicion Modeling" \
    "Recognized-Problem Suppression" \
    "Self-Preservation Action Planning" \
    "Self-Preservation Rationalization" \
    "Self-Preserving Advocacy" \
    "Social-Pressure Compliance" \
    "Strategic Attention Manipulation" \
    "Strategic Framing" \
    "Strategic Omission" \
    "Strategic Underperformance Reasoning" \
  --label-mode span \
  --reg-coeff 10.0 \
  --detect-layers 27 \
  --reasoning-positive-only \
  --threshold-mode sentence \
  --val-fraction 0.2 \
  --data-dir "$REPO/probe/data/v4_v2_6_combined_v2" \
  --output-dir "$PROBES"

echo "[$(date -Is)] evaluating locked hard-negative set"
"$PY" "$SCRIPTS/evaluate_retrained.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$ROLLOUT"

echo "[$(date -Is)] applying held-out thresholds"
"$PY" "$SCRIPTS/analyze_probes.py" --repo "$REPO" --probe-root "$PROBES" --rollout-dir "$ROLLOUT" --artifacts "$ART"

echo "[$(date -Is)] running local Qwen judge only on escalations"
"$PY" "$SCRIPTS/run_qwen_judge.py" --repo "$REPO" --rollout-dir "$ROLLOUT" --artifacts "$ART" --offload-dir "$EXP/offload/qwen"

"$PY" "$SCRIPTS/make_report.py" --artifacts "$ART" --experiment "$EXP"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader > "$EXP/gpu-after.txt"
ps -o pid,ppid,stat,etime,cmd -p 199446,199453,199461 > "$EXP/keepalives-after.txt"
touch "$EXP/COMPLETED"
echo "[$(date -Is)] complete"
