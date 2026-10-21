#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 REPO RESULTS_SUBDIR OUTPUT_JSON MODE[paper|all]" >&2
  exit 2
fi

REPO="$1"
RESULTS_SUBDIR="$2"
OUTPUT="$3"
MODE="$4"
PY="/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python"

EXTRA=()
case "$MODE" in
  paper) ;;
  all) EXTRA+=(--include-preconditions) ;;
  *) echo "MODE must be paper or all" >&2; exit 2 ;;
esac

cd "$REPO"
for fpr in 0.002 0.005 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3; do
  "$PY" -m probe_eval.misalignment_ground_truth \
    --results-subdir "$RESULTS_SUBDIR" \
    --include-all-negative \
    --exclude-behaviors "test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback,mask,hard_negatives*" \
    --output-suffix "_hardware_safe_${MODE}_fpr_${fpr}" \
    --tune-thresholds --fpr-target "$fpr" \
    --tuned-thresholds-save "$OUTPUT" \
    --short-sentence-mode discard \
    "${EXTRA[@]}"
done

"$PY" - "$OUTPUT" "$MODE" <<'PY'
import hashlib, json, sys
from pathlib import Path

p = Path(sys.argv[1])
mode = sys.argv[2]
d = json.loads(p.read_text())
d["hardware_safe_metadata"] = {
    "max_sequence_tokens": 8192,
    "development_only": True,
    "test_labels_used_for_tuning": False,
    "mode": mode,
    "paper_compatible_indicator_count": 17,
    "all_released_glm_probe_count": 18,
    "warning": "Thresholds are reconstructed after excluding overlength development dialogues and are not the authors' missing original manifest.",
}
p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
print(hashlib.sha256(p.read_bytes()).hexdigest(), p)
PY
