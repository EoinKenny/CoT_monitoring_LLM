#!/usr/bin/env python3
"""Run one released probe family over its compatible rollout catalog."""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model-family", choices=["glm", "qwen"], required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--include-auxiliary", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "ood_misalignment_eval/deception-detection")]
    import probe_eval.evaluate as evaluate

    layer_filter = set(args.layers or [])
    probe_dirs = []
    for cfg in sorted(args.probe_root.glob("*/span/layer*/cfg.yaml")):
        layer = int(cfg.parent.name.removeprefix("layer"))
        if not layer_filter or layer in layer_filter:
            probe_dirs.append(str(cfg.parent.resolve()))
    if not probe_dirs:
        raise RuntimeError("No compatible probes found")

    catalog = json.loads(args.catalog.read_text())["datasets"]
    selected = [
        r for r in catalog
        if r["model_family"] == args.model_family
        and (args.include_auxiliary or r["split"] != "auxiliary")
    ]
    args.staging.mkdir(parents=True, exist_ok=True)
    staged = []
    for row in selected:
        link = args.staging / row["id"]
        target = Path(row["path"]).resolve()
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f"Staging collision: {link}")
        elif link.exists():
            raise RuntimeError(f"Non-symlink staging path exists: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
        staged.append(str(link))

    config = evaluate.BatchEvalConfig(
        experiment_folders=probe_dirs,
        rollout_dirs=staged,
        # Labels are applied later from the released per-turn GT. Treating all
        # rows as negative here affects only the evaluator's incidental AUROC,
        # not activation extraction or token/sentence scores.
        all_negative=True,
        skip_existing=True,
        batch_size=1,
        max_sequence_tokens=args.max_sequence_tokens,
    )
    os.chdir(repo)
    evaluate.run_batch_evaluation(config)


if __name__ == "__main__":
    main()
