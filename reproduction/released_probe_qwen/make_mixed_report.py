#!/usr/bin/env python3
import argparse, json
from pathlib import Path


def pct(x):
    return "N/A" if x is None else f"{100*x:.2f}%"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,required=True); ap.add_argument("--experiment",type=Path,required=True); a=ap.parse_args()
    m=json.loads((a.artifacts/"mixed_metrics.json").read_text()); p=m["probe"]; c=m["cascade"]
    text=f"""# Released 18-probe → local-Qwen mixed deception benchmark

Primary benchmark: all released GLM DeceptionBench positives plus all aligned hard negatives. Labels are released rollout labels. The 18 per-probe thresholds were reconstructed on the paper's 10-folder development split at the paper's `fpr_0.02` operating point, then frozen before this test.

| Method | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Released 18-probe ensemble | {p['tp']} | {p['fp']} | {p['tn']} | {p['fn']} | {pct(p['accuracy'])} | {pct(p['precision'])} | {pct(p['recall'])} | {pct(p['f1'])} | {pct(p['balanced_accuracy'])} |
| Probe-positive → Qwen | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | {pct(c['accuracy'])} | {pct(c['precision'])} | {pct(c['recall'])} | {pct(c['f1'])} | {pct(c['balanced_accuracy'])} |

- Turns: {m['turns']} total; {m['positive_turns']} positive and {m['negative_turns']} negative.
- Qwen escalations: {m['qwen_escalations']}; valid judgments: {m['qwen_valid_judgments']}; errors: {m['qwen_errors']}.
- Unresolved cascade predictions: {c['unresolved']}.
- Sequence ceiling: 8,192 tokens; see `mixed_manifest.json` for per-source skips and coverage.
- Probe-negative turns are accepted without Qwen; every probe-positive turn is escalated; the final cascade is positive only when Qwen confirms it.

Availability note: the repository's launcher references additional Bloom dev/test directories that are not committed in this pinned checkout. They are not included or represented as evaluated here.
"""
    (a.experiment/"MIXED_RESULTS.md").write_text(text); print(text)


if __name__=="__main__": main()
