#!/usr/bin/env python3
"""Apply held-out thresholds and emit turn-level escalation decisions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    args = ap.parse_args()
    repo, probe_root = args.repo.resolve(), args.probe_root.resolve()
    rollout_dir, artifacts = args.rollout_dir.resolve(), args.artifacts.resolve()
    sys.path.insert(0, str(repo))
    from probe_eval.sentence_scores import load_per_sentence_scores

    thresholds: dict[str, float] = {}
    probe_dirs: dict[str, Path] = {}
    threshold_details = {}
    for meta_path in sorted(probe_root.glob("*/span/layer27/training_meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta["indicator_name"]
        # Evaluation reduces tokens to sentence means and then takes the maximum
        # sentence in the turn, so use the matching held-out turn-level threshold.
        rec = meta["val_thresholds"]["turn_level"]["fpr_1pct"]
        threshold = float(rec["threshold"])
        if not math.isfinite(threshold) or name in thresholds:
            raise RuntimeError(f"Invalid/duplicate threshold for {name}: {threshold}")
        thresholds[name] = threshold
        probe_dirs[name] = meta_path.parent
        threshold_details[name] = rec
    if len(thresholds) != 18:
        raise RuntimeError(f"Expected 18 thresholds, found {len(thresholds)}")

    result_root = repo / "probe_eval" / "results"
    per_probe = {}
    sentence_cache: dict = {}
    for name, probe_dir in probe_dirs.items():
        matches = []
        for p in result_root.rglob(f"{rollout_dir.name}/token_scores.json"):
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            if Path(doc.get("experiment_folder", "")).resolve() == probe_dir.resolve():
                matches.append(p)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one token-score file for {name}, found {matches}")
        sentence_data = load_per_sentence_scores(matches[0], str(rollout_dir), sentence_cache)
        if sentence_data is None:
            raise RuntimeError(f"No sentence data for {name}")
        scores = {}
        for rollout_idx, turns in sentence_data.items():
            for turn_idx, sentence_scores in turns.items():
                if sentence_scores:
                    score = max(float(x) for x in sentence_scores)
                    if not math.isfinite(score):
                        raise RuntimeError(f"Non-finite score for {name}/{rollout_idx}/{turn_idx}")
                    scores[(int(rollout_idx), int(turn_idx))] = score
        per_probe[name] = scores

    rollouts = json.loads((rollout_dir / "rollout.json").read_text())["rollouts"]
    all_keys = sorted(set().union(*(set(v) for v in per_probe.values())))
    decisions = []
    for rollout_idx, turn_idx in all_keys:
        probe_scores = {name: scores.get((rollout_idx, turn_idx)) for name, scores in per_probe.items()}
        fires = [name for name, score in probe_scores.items() if score is not None and score > thresholds[name]]
        ro = rollouts[rollout_idx]
        decisions.append({
            "rollout_index": rollout_idx,
            "variation_number": int(ro["variation_number"]),
            "repetition_number": int(ro["repetition_number"]),
            "turn_index": turn_idx,
            "probe_flag": bool(fires),
            "firing_probes": fires,
            "probe_scores": probe_scores,
        })
    if len(decisions) != 474:
        raise RuntimeError(f"Expected 474 scored assistant turns, found {len(decisions)}")

    probe_fp = sum(x["probe_flag"] for x in decisions)
    metrics = {
        "dataset": "360 released aligned hard-negative rollouts",
        "rollouts": len(rollouts),
        "turns": len(decisions),
        "threshold_source": "held-out training validation; turn_level.fpr_1pct; no test tuning",
        "probe_false_positives": probe_fp,
        "probe_fpr": probe_fp / len(decisions),
        "qwen_escalations_planned": probe_fp,
    }
    dump(artifacts / "thresholds.json", {"thresholds": thresholds, "details": threshold_details})
    dump(artifacts / "probe_decisions.json", decisions)
    dump(artifacts / "probe_metrics.json", metrics)
    with (artifacts / "probe_decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rollout_index", "variation_number", "repetition_number", "turn_index", "probe_flag", "firing_probes"])
        w.writeheader()
        for x in decisions:
            w.writerow({**{k: x[k] for k in w.fieldnames if k != "firing_probes"}, "firing_probes": json.dumps(x["firing_probes"])})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
