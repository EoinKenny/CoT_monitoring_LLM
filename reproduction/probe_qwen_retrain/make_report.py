#!/usr/bin/env python3
"""Create the compact final comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--experiment", type=Path, required=True)
    args = ap.parse_args()
    m = json.loads((args.artifacts / "metrics.json").read_text())
    report = f"""# Fresh-probe → local-Qwen hard-negative experiment

All results use the 360 released aligned hard-negative rollouts ({m['turns']} assistant turns). Consequently, the table reports false-positive rates, not deception recall or overall accuracy.

| Method | False positives | Denominator | False-positive rate |
|---|---:|---:|---:|
| Fresh layer-27 probe ensemble | {m['probe_false_positives']} | {m['turns']} | {100*m['probe_fpr']:.2f}% |
| Fresh probes → Qwen3-30B-A3B (score ≥ 8) | {m['probe_qwen_false_positives']} | {m['turns']} | {100*m['probe_qwen_fpr_all_turns']:.2f}% |

## Configuration

- Eighteen indicator-specific linear probes retrained from scratch.
- Target activations: GLM-4.7-Flash, layer 27, BF16, no quantization.
- Training: released `v4_v2_6_combined_v2` data, span labels, positive reasoning only, L2 coefficient 10.
- Calibration: deterministic transcript-level 80/20 split; held-out turn-level 1%-FPR threshold for each probe; OR fusion across probes.
- No threshold was selected or adjusted on the 360-rollout test set.
- Judge: local `Qwen/Qwen3-30B-A3B`, BF16 with CPU offload, paper-compatible history-only/no-definition score prompt, deterministic decoding, threshold 8.
- Qwen was called only for probe-positive turns.
- Qwen coverage: {m['qwen_valid_judgments']}/{m['qwen_escalations']} escalations; errors: {m['qwen_errors']}.

## Provenance limitation

The released probe metadata does not exactly match turn counts reconstructed from the committed training JSON, and the paper's advertised neutral-set threshold manifest is absent. This run is therefore a fresh, source-documented retraining baseline rather than a bit-for-bit recreation of the released weights.

Experiment directory: `{args.experiment}`
"""
    (args.experiment / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
