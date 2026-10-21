#!/usr/bin/env python3
"""Score a fixed dataset list with selected layers in one model load."""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "ood_misalignment_eval/deception-detection")]
    import probe_eval.evaluate as evaluate

    layers = set(args.layer)
    probes = [
        str(p.parent.resolve()) for p in sorted(args.probe_root.glob("*/span/layer*/cfg.yaml"))
        if int(p.parent.name.removeprefix("layer")) in layers
    ]
    expected = 18 * len(layers)
    if len(probes) != expected:
        raise RuntimeError(f"Expected {expected} probes, found {len(probes)}")
    os.chdir(repo)
    evaluate.run_batch_evaluation(evaluate.BatchEvalConfig(
        experiment_folders=probes,
        rollout_dirs=[str(p.resolve()) for p in args.rollout_dir],
        all_negative=True,
        skip_existing=True,
        batch_size=1,
        max_sequence_tokens=args.max_sequence_tokens,
    ))


if __name__ == "__main__":
    main()
