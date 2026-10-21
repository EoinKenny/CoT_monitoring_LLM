#!/usr/bin/env python3
"""Inventory released probe families and rollout datasets without changing them."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hf-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probe_roots = {
        "glm_paper_main": args.repo / "probe/probes_paper_main/v4_v2_6_combined_v2_span",
        "qwen_cross_model": args.repo / "probe/probes_qwen36/v4_v2_6_combined_v2_span",
    }
    probes = {}
    for family, root in probe_roots.items():
        rows = []
        for detector in sorted(root.glob("*/span/layer*/detector.pt")):
            layer_dir = detector.parent
            rows.append({
                "indicator": detector.parents[2].name,
                "layer": int(layer_dir.name.removeprefix("layer")),
                "detector": str(detector),
                "sha256": sha256(detector),
                "has_config": (layer_dir / "cfg.yaml").exists(),
                "has_training_meta": (layer_dir / "training_meta.json").exists(),
            })
        probes[family] = rows

    rollout_files = sorted(set(args.repo.rglob("rollout.json")) | set(args.hf_data.rglob("rollout.json")))
    datasets = []
    for path in rollout_files:
        try:
            payload = json.loads(path.read_text())
            count = len(payload.get("rollouts", []))
        except Exception as exc:
            count = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        datasets.append({
            "name": path.parent.name,
            "path": str(path.parent),
            "rollouts": count,
            "sha256": sha256(path),
            "error": error,
        })

    result = {
        "probe_families": probes,
        "probe_counts": {k: len(v) for k, v in probes.items()},
        "datasets": datasets,
        "dataset_count": len(datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "probe_counts": result["probe_counts"],
        "dataset_count": result["dataset_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
