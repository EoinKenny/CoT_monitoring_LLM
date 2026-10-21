#!/usr/bin/env python3
"""Build a collision-safe catalog of released rollout datasets."""

import argparse
import json
import re
from pathlib import Path


DEV_NAMES = {
    "instructed-covert-code-sabotage_glm_4_7_flash",
    "instructed-covert-code-sabotage_benign_glm_4_7_flash",
    "instructed-strategic-sandbagging_glm_4_7_flash",
    "instructed-strategic-sandbagging_benign_glm_4_7_flash",
    "sycophancy", "sycophancy_benign_glm_4_7_flash",
    "self-preservation_glm_4_7_flash",
    "self-preservation_benign_glm_4_7_flash",
    "strategic-deception_glm_4_7_flash",
    "strategic-deception_benign_glm_4_7_flash",
}


def classify(path: Path) -> dict:
    s = str(path)
    name = path.name
    language = "vi" if "-vi" in s or name.endswith("_vi") else "cn" if "-cn" in s or name.endswith("_cn") else "en"
    model = "qwen" if "qwen" in name.lower() else "glm"
    if name in DEV_NAMES and "bloom-results" in s and "test" not in s:
        split = "dev"
    elif name.startswith("test_") or "bloom-results-test" in s:
        split = "test"
    elif "hard-negative" in s or "hard_negatives" in name:
        split = "hard_negative"
    elif "ood" in s or "deception-bench" in s or "/mask/" in s or "sycophancy-eval" in s:
        split = "ood"
    else:
        split = "auxiliary"
    return {"language": language, "model_family": model, "split": split}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hf-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = [args.repo / "bloom", args.repo / "ood_misalignment_eval", args.hf_data]
    seen_hashable = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for rollout in sorted(root.rglob("rollout.json")):
            parent = rollout.parent.resolve()
            key = str(parent)
            if key in seen_hashable:
                continue
            seen_hashable.add(key)
            meta = classify(parent)
            payload = json.loads(rollout.read_text())
            source = "hf" if args.hf_data.resolve() in parent.parents else "repo"
            rel = parent.relative_to(args.hf_data.resolve() if source == "hf" else args.repo.resolve())
            stable_id = re.sub(r"[^a-zA-Z0-9_.-]+", "__", f"{source}__{rel}")
            rows.append({
                "id": stable_id,
                "path": str(parent),
                "source": source,
                "rollouts": len(payload.get("rollouts", [])),
                **meta,
            })
    result = {"datasets": rows, "count": len(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "count": len(rows),
        "by_split": {s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
        "by_model": {m: sum(r["model_family"] == m for r in rows) for m in sorted({r["model_family"] for r in rows})},
    }, indent=2))


if __name__ == "__main__":
    main()
