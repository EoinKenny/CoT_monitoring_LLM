#!/usr/bin/env python3
"""Summarize the evaluator's token-ceiling exclusions by dataset."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[])
    args = parser.parse_args()
    rows = []
    for path in sorted(args.safety_dir.glob("*.json")):
        if args.dataset and path.stem not in set(args.dataset):
            continue
        d = json.loads(path.read_text())
        total = int(d.get("total_dialogues", 0))
        skipped = int(d.get("skipped_dialogues", 0))
        rows.append({
            "dataset": path.stem,
            "total": total,
            "retained": total - skipped,
            "skipped": skipped,
            "skipped_fraction": skipped / total if total else None,
            "max_sequence_tokens": d.get("max_sequence_tokens"),
        })
    total = sum(r["total"] for r in rows)
    skipped = sum(r["skipped"] for r in rows)
    result = {
        "datasets": rows,
        "total": total,
        "retained": total - skipped,
        "skipped": skipped,
        "skipped_fraction": skipped / total if total else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
