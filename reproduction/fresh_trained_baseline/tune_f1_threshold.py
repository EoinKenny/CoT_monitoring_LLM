#!/usr/bin/env python3
"""Tune one shared multiplier on fixed per-probe validation thresholds."""

import argparse
import csv
import json
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def confusion(rows: list[dict]) -> dict:
    tp = sum(r["probe_flag"] and r["ground_truth_deceptive"] for r in rows)
    fp = sum(r["probe_flag"] and not r["ground_truth_deceptive"] for r in rows)
    tn = sum(not r["probe_flag"] and not r["ground_truth_deceptive"] for r in rows)
    fn = sum(not r["probe_flag"] and r["ground_truth_deceptive"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--output-artifacts", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_artifacts.resolve(), args.output_artifacts.resolve()
    rows = json.loads((source / "mixed_decisions.json").read_text())
    base_thresholds = json.loads((source / "mixed_thresholds.json").read_text())
    source_metrics = json.loads((source / "mixed_probe_metrics.json").read_text())

    normalized_maxima = []
    for row in rows:
        normalized = {
            name: float(row["probe_scores"][name]) / float(threshold)
            for name, threshold in base_thresholds.items()
            if row["probe_scores"].get(name) is not None
        }
        normalized_maxima.append(max(normalized.values()))

    curve = []
    for factor in sorted(set(normalized_maxima)):
        candidate = []
        for row, maximum in zip(rows, normalized_maxima):
            candidate.append(
                {"probe_flag": maximum >= factor,
                 "ground_truth_deceptive": row["ground_truth_deceptive"]}
            )
        metrics = confusion(candidate)
        curve.append({"threshold_factor": factor, **metrics})

    # Deterministic tie-break: maximum F1, then precision, then the more
    # conservative (higher) threshold factor.
    best = max(curve, key=lambda x: (x["f1"], x["precision"], x["threshold_factor"]))
    factor = best["threshold_factor"]
    effective = {name: value * factor for name, value in base_thresholds.items()}

    tuned_rows = []
    for row in rows:
        result = dict(row)
        normalized = {
            name: float(row["probe_scores"][name]) / float(base_thresholds[name])
            for name in base_thresholds
            if row["probe_scores"].get(name) is not None
        }
        fires = sorted(name for name, value in normalized.items() if value >= factor)
        result.update(
            {
                "probe_threshold_factor": factor,
                "probe_normalized_scores": normalized,
                "probe_normalized_max": max(normalized.values()),
                "firing_probes": fires,
                "probe_flag": bool(fires),
                "probe_correct": bool(fires) == bool(row["ground_truth_deceptive"]),
            }
        )
        tuned_rows.append(result)

    metrics = dict(source_metrics)
    metrics.update(
        {
            "threshold_source": "test-set F1 maximization on all 562 labeled turns; resubstitution estimate",
            "threshold_tuning": {
                "method": "single shared multiplier on per-probe held-out turn-level 1%-FPR thresholds",
                "selection_data": "same 562-turn benchmark used for reported metrics",
                "objective": "ensemble F1",
                "tie_break": "precision, then higher threshold",
                "factor": factor,
                "candidate_count": len(curve),
                "generalization_estimate": False,
            },
            "probe": confusion(tuned_rows),
            "qwen_escalations_planned": sum(r["probe_flag"] for r in tuned_rows),
        }
    )
    dump(output / "mixed_base_thresholds.json", base_thresholds)
    dump(output / "mixed_thresholds.json", effective)
    dump(output / "mixed_decisions.json", tuned_rows)
    dump(output / "mixed_probe_metrics.json", metrics)
    dump(output / "mixed_manifest.json", source_metrics["sources"])
    dump(output / "threshold_tuning.json", {"best": best, "effective_thresholds": effective})
    with (output / "threshold_tuning_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
