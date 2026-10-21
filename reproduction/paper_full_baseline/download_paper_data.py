#!/usr/bin/env python3
"""Download the immutable public data snapshot used by the paper."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket

from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "kzhou35/misalignment-indicators-bloom-rollouts"


# This EC2 host advertises IPv6 routes that never complete the TLS handshake.
# Keep the workaround local to this downloader rather than changing host
# networking or touching any other process.
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    return [
        item for item in _original_getaddrinfo(*args, **kwargs)
        if item[0] == socket.AF_INET
    ]


socket.getaddrinfo = _ipv4_getaddrinfo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()

    info = HfApi().dataset_info(DATASET_ID, revision=args.revision)
    revision = info.sha
    args.output.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=revision)
    required_names = {
        "rollout.json", "judgment.json", "transform_filter_report.json",
    }
    selected = [
        f for f in files
        if Path(f).name in required_names
        or Path(f).name.startswith("rollout_misalignment_turns")
        or Path(f).name.startswith("ambiguous_rollouts")
    ]

    def download_one(filename: str) -> str:
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=args.output,
        )
        print(filename, flush=True)
        return filename

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download_one, selected))
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "selection": "inference-complete: rollouts, labels, GT, ambiguity files, and released turn-judge scores",
        "selected_file_count": len(selected),
        "file_count": sum(1 for p in args.output.rglob("*") if p.is_file()),
    }
    (args.output / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
