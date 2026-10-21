"""Join frozen context assessments to already-saved original and new answers."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"broad_contrast/context_safety_followup"
freeze=json.loads((O/"CODING_FREEZE.json").read_text())
assert hashlib.sha256((O/"manual_codes.jsonl").read_bytes()).hexdigest()==freeze["codes_sha256"]
codes={x["id"]:x for x in map(json.loads,(O/"manual_codes.jsonl").read_text().splitlines())}
packet={x["id"]:x for x in map(json.loads,(O/"masked_packet.jsonl").read_text().splitlines())}
private=list(map(json.loads,gzip.open(O/"private_mapping.jsonl.gz","rt")))
keys={(x["judge"],x["backbone"],x["sample_id"]) for p in private for x in p["matched_records"]}
original={}
source=R.parents[1]/"reproduction/nn_contrast_investigation_20261020/results/cases.csv.gz"
for x in csv.DictReader(gzip.open(source,"rt")):
    key=(x["model"],x["backbone"],x["sample_id"])
    if key in keys:
        item={k:x[k] for k in ["control_score","nearest_score","true_label"]}
        assert key not in original or original[key]==item
        original[key]=item
assert set(original)==keys
out=[]
for p in private:
    for x in p["matched_records"]:
        old=original[(x["judge"],x["backbone"],x["sample_id"])]
        raw=x["original_source"]["source_record"]
        prob=raw["positive_probability"]
        assert int(old["true_label"])==x["true_label"]
        control,nearest=float(old["control_score"]),float(old["nearest_score"])
        original_flip=(control>=50)!=(nearest>=50)
        assert original_flip==x["original_source"]["original_fixed50_flip_selected"]
        out.append(dict(
            {**packet[p["id"]],**codes[p["id"]]},
            judge=x["judge"],backbone=x["backbone"],sample_id=x["sample_id"],
            true_label=x["true_label"],control_probability=control,nn_probability=nearest,
            probability_first_answer=prob,answer_correct=(prob>=50)==x["true_label"],
            original_flip=original_flip,
            answer_changed_from_original_nn=(prob>=50)!=(nearest>=50),
            explanation_complete=raw["reasoning_complete"],
            finish_reason=raw["finish_reason"],
            source_file=x["original_source"]["source_file"],
            source_file_sha256=x["original_source"]["source_file_sha256"]))
dest=O/"unmasked_examples.json"
assert not dest.exists()
dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
(O/"UNMASKING_RECEIPT.json").write_text(json.dumps(dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),n=len(out),
    original_scores_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    coding_freeze_sha256=hashlib.sha256((O/"CODING_FREEZE.json").read_bytes()).hexdigest(),
    output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    scope="Purposive extra saved test records, all originally control/NN-unchanged; separate from flip-selected primary study."
),indent=2)+"\n")
for x in out:
    print(x["id"],x["judge"],x["dataset"],"gold",x["true_label"],
          "control/NN",x["control_probability"],x["nn_probability"],
          "probability-first-answer",x["probability_first_answer"],
          "new-correct",x["answer_correct"],"changed",x["answer_changed_from_original_nn"])
