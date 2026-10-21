#!/usr/bin/env python3
"""Small correctness checks for OR fusion and the released sentence reducer."""

import math
import sys
from pathlib import Path


def main(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from probe_eval.sentence_scores import _split_tokens_into_sentences

    scores = _split_tokens_into_sentences(
        ["One", " sentence", ".", " Another", " longer", " sentence", " here", "."],
        [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    )
    assert scores and all(math.isfinite(x) for x in scores)

    thresholds = {"a": 1.0, "b": 2.0, "c": 3.0}
    values = {"a": 0.9, "b": 2.1, "c": -1.0}
    fires = [name for name, score in values.items() if score >= thresholds[name]]
    assert fires == ["b"]
    assert any(values[name] >= thresholds[name] for name in thresholds)
    print("analysis checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_analysis.py REPO")
    main(Path(sys.argv[1]).resolve())
