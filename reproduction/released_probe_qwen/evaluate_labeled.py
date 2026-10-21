#!/usr/bin/env python3
"""Evaluate the released layer-27 probes on a rollout directory."""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--expected-probes", type=int, default=18)
    ap.add_argument("--all-negative", action="store_true")
    args = ap.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "ood_misalignment_eval" / "deception-detection")]

    import probe_eval.evaluate as evaluate

    probe_dirs = sorted(str(p.parent) for p in args.probe_root.glob("*/span/layer27/cfg.yaml"))
    if len(probe_dirs) != args.expected_probes:
        raise RuntimeError(
            f"Expected {args.expected_probes} saved layer-27 probes, found {len(probe_dirs)}"
        )
    config = evaluate.BatchEvalConfig(
        experiment_folders=probe_dirs,
        rollout_dirs=[str(args.rollout_dir.resolve())],
        behavior_threshold=5,
        exclude_ambiguous=False,
        scoring_method="mean",
        reasoning_only=False,
        all_negative=args.all_negative,
        skip_existing=True,
        batch_size=1,
        max_sequence_tokens=8192,
    )
    os.chdir(repo)
    evaluate.run_batch_evaluation(config)


if __name__ == "__main__":
    main()
