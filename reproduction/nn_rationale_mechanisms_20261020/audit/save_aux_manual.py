"""Serialize explicit manual classifications only after complete packet reading."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def save(corpus, batch, negative=(), unclear=(), mixed=(), incomplete=(),
         relations=None, axes=None, notes=None, default_axes=("FACT",), reading_view=None):
    root = ROOT/corpus
    source = root/"packets"/(batch+".jsonl")
    rows = list(map(json.loads, source.read_text().splitlines()))
    dest = root/"codes"/(batch+".jsonl")
    assert not dest.exists()
    codes = []
    for i, row in enumerate(rows):
        relation, bridge = (relations or {}).get(i, "N/T").split("/")
        code = dict(
            id=row["id"], relation=relation, bridge=bridge,
            axes=(axes or {}).get(i, list(default_axes)),
            conclusion="N" if i in negative else "U" if i in unclear else "M" if i in mixed else "P",
            quality="I" if i in incomplete else "A",
            evidence=list(dict.fromkeys([row["text"][:120], row["text"][-180:]])))
        if notes and i in notes:
            code["note"] = notes[i]
        codes.append(code)
    dest.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in codes))
    receipt = dict(
        n=len(rows), reader="/root", read_complete=True,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_version="expressed-v1",
        note="Every full text was manually read. Row-level choices are explicit manual classifications; excerpt serialization does not infer labels."
    )
    if reading_view is not None:
        assert reading_view == "lossless-adjacent-repeat-v1"
        receipt["reading_view"] = reading_view
        receipt["note"] += (
            " Exact contiguous repetition may be displayed once with its exact repeat count; "
            "every original string is reconstructed and checked before display. No distinct text is omitted."
        )
    dest.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(corpus, batch, len(rows))
