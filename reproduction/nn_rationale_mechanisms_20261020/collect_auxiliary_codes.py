"""Validate full-read receipts for the two separately frozen auxiliary corpora."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"reasoning_first"))
from collect_codes import check


def main(name):
    root = ROOT/name
    records = list(map(json.loads, gzip.open(root/"corpus/rows.jsonl.gz","rt")))
    texts = {x["text_id"]:x["text"] for x in records}
    batches = list(csv.DictReader((root/"BATCHES.csv").open()))
    codes = {}
    complete_batches = 0
    for batch in batches:
        codefile = root/"codes"/(batch["batch_id"]+".jsonl")
        receiptfile = codefile.with_suffix(".receipt.json")
        if not codefile.exists() and not receiptfile.exists():
            continue
        assert codefile.exists() and receiptfile.exists(), "Uncommitted partial packet"
        receipt = json.loads(receiptfile.read_text())
        source = Path(batch["path"])
        packet = list(map(json.loads,source.read_text().splitlines()))
        data = list(map(json.loads,codefile.read_text().splitlines()))
        assert receipt["read_complete"] and receipt["n"] == int(batch["n"]) == len(data)
        assert receipt["input_sha256"] == batch["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert {x["id"] for x in data} == {x["id"] for x in packet}
        assert len({x["id"] for x in data}) == len(data)
        for code in data:
            check(code,texts[code["id"]])
            assert code["id"] not in codes
            codes[code["id"]] = dict(code, dataset=batch["dataset"], reader=receipt["reader"], reused=False)
        complete_batches += 1
    reused = json.loads((root/"REUSED_EXPRESSED_CODES.json").read_text())
    for item in reused:
        code = item["code"]
        check(code,texts[code["id"]])
        assert hashlib.sha256((item["dataset"]+"\n"+texts[code["id"]]).encode()).hexdigest() == code["id"]
        assert code["id"] not in codes
        codes[code["id"]] = dict(code,reused=True,prior_code_source=item["source"])
    status = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(), corpus=name, selected_records=len(records),
        available_records=sum(bool(x["text"].strip()) for x in records),
        records_read=sum(x["text_id"] in codes for x in records),
        records_unread=sum(bool(x["text"].strip()) and x["text_id"] not in codes for x in records),
        new_unique_read=len(codes)-len(reused), exact_prior_text_codes=len(reused),
        complete_batches=complete_batches, total_batches=len(batches),
        all_available_read=all(not x["text"].strip() or x["text_id"] in codes for x in records),
        contextual_validity_not_inherited=True)
    (root/"results/READING_STATUS.json").write_text(json.dumps(status,indent=2)+"\n")
    (root/"results/validated_codes.jsonl").write_text(
        "".join(json.dumps(c,ensure_ascii=False)+"\n" for c in codes.values()))
    print(json.dumps(status))


if __name__ == "__main__":
    for name in ["native_contrast","broad_contrast"]:
        main(name)
