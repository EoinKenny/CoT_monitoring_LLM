#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def metric_row(label, m):
    return f"| {label} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {pct(m['accuracy'])} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {pct(m['balanced_accuracy'])} |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m = json.loads((args.artifacts / "mixed_binary_metrics.json").read_text())
    factor = m["threshold_tuning"]["factor"]
    lines = [
        "# Test-tuned fresh-probe cascade",
        "",
        "The shared threshold multiplier was selected on these same 562 labels to maximize probe-ensemble F1. These are reproducible resubstitution results, not an unbiased generalization estimate.",
        "",
        "| Method | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("Fresh 8-probe ensemble", m["probe"]),
        metric_row("Probe-positive -> binary Qwen", m["cascade"]),
        "",
        f"- Selected shared threshold factor: {factor:.12g}.",
        f"- Probe escalations: {m['qwen_escalations']}; valid Qwen judgments: {m['qwen_valid_judgments']}; errors: {m['qwen_errors']}.",
        "- Every probe-positive turn was sent to Qwen; probe-negative turns were accepted without escalation.",
        "- No benchmark dialogue exceeded the 8,192-token GLM ceiling.",
        "",
    ]
    args.output.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
