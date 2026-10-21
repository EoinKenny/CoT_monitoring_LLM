#!/usr/bin/env python3
"""Write a compact report for the fresh-probe and binary-Qwen cascade run."""

import argparse
import json
from pathlib import Path


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def row(label, metrics):
    return (
        f"| {label} | {metrics['tp']} | {metrics['fp']} | {metrics['tn']} | "
        f"{metrics['fn']} | {pct(metrics['accuracy'])} | {pct(metrics['precision'])} | "
        f"{pct(metrics['recall'])} | {pct(metrics['f1'])} | {pct(metrics['balanced_accuracy'])} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.artifacts / "mixed_binary_metrics.json").read_text())
    probe, cascade = metrics["probe"], metrics["cascade"]
    sources = metrics["sources"]
    skips = []
    for source in sources:
        safety = source.get("sequence_safety") or {}
        skips.append(
            f"{source['source']}: {safety.get('skipped_dialogues', safety.get('skipped', 0))} skipped"
        )
    text = "\n".join(
        [
            "# Fresh 8-probe -> binary Qwen mixed deception benchmark",
            "",
            "Eight layer-27 linear probes were trained locally from the released public indicator data. "
            "Each probe uses its own held-out turn-level 1%-FPR validation threshold; the mixed test set "
            "was not used to choose thresholds.",
            "",
            "| Method | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | Balanced accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            row("Fresh 8-probe ensemble", probe),
            row("Probe-positive -> binary Qwen", cascade),
            "",
            f"- Turns: {metrics['turns']} total; {metrics['positive_turns']} positive and {metrics['negative_turns']} negative.",
            f"- Probe-positive escalations: {metrics['qwen_escalations']}.",
            f"- Valid Qwen judgments: {metrics['qwen_valid_judgments']}; context skips: {metrics['qwen_skipped_for_context']}; errors: {metrics['qwen_errors']}.",
            f"- Sequence ceiling: 8,192 GLM tokens ({'; '.join(skips)}).",
            "- Probe-negative turns are accepted; every probe-positive turn is escalated; Qwen returns a hard boolean classification.",
            "- This 8-probe focused ensemble is a deliberate smaller fresh-trained comparison, not the paper's released 18-probe ensemble.",
            "",
        ]
    )
    (args.experiment / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
