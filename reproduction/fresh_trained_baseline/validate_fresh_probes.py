#!/usr/bin/env python3
"""Validate the exact fresh deception-probe ensemble and record its thresholds."""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {
    "Action Concealment",
    "Fabrication",
    "Misalignment Cover Story Planning",
    "Observer Suspicion Modeling",
    "Recognized-Problem Suppression",
    "Strategic Attention Manipulation",
    "Strategic Framing",
    "Strategic Omission",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    names = set()
    for meta_path in sorted(args.probe_root.glob("*/span/layer27/training_meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta["indicator_name"]
        threshold = float(meta["val_thresholds"]["turn_level"]["fpr_1pct"]["threshold"])
        if name in names or not math.isfinite(threshold):
            raise RuntimeError(f"Duplicate or invalid fresh probe: {name}")
        names.add(name)
        detector = meta_path.parent / "detector.pt"
        config = meta_path.parent / "cfg.yaml"
        if not detector.is_file() or not config.is_file():
            raise RuntimeError(f"Incomplete probe checkpoint for {name}")
        if meta.get("detect_layers") != [27] or meta.get("label_mode") != "span":
            raise RuntimeError(f"Unexpected training configuration for {name}")
        rows.append(
            {
                "indicator": name,
                "slug": meta_path.parents[2].name,
                "training_timestamp": meta.get("timestamp"),
                "training_data_path": meta.get("training_data_path"),
                "n_positive_turns": meta.get("n_positive_turns"),
                "n_negative_turns": meta.get("n_negative_turns"),
                "held_out_turn_level_1pct_fpr_threshold": threshold,
                "held_out_turn_level_tpr": meta["val_thresholds"]["turn_level"]["fpr_1pct"].get("tpr"),
                "held_out_turn_level_observed_fpr": meta["val_thresholds"]["turn_level"]["fpr_1pct"].get("fpr"),
                "detector_sha256": sha256(detector),
                "training_meta_sha256": sha256(meta_path),
            }
        )

    if names != EXPECTED:
        raise RuntimeError(
            f"Fresh ensemble mismatch. Missing={sorted(EXPECTED - names)} extra={sorted(names - EXPECTED)}"
        )
    payload = {
        "ensemble": "fresh 8-probe deception-focused layer-27 ensemble",
        "threshold_policy": "turn_level.fpr_1pct from each probe's held-out validation split",
        "test_set_threshold_tuning": False,
        "count": len(rows),
        "probes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
