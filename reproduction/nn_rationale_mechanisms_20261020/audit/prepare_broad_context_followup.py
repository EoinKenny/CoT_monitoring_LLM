"""Freeze a purposive, outcome-masked source check; no outcome-based sampling."""
from pathlib import Path
import gzip
import hashlib
import json

R = Path(__file__).resolve().parents[1]
out = R/"broad_contrast"/"context_followup"
out.mkdir(exist_ok=True)
selections = [
    ("contrast_halueval_000", 10, "since-1991 temporal inference"),
    ("contrast_halueval_000", 37, "elliptical Serie A answer"),
    ("contrast_toxicchat_000", 14, "drug price versus stock price"),
    ("contrast_toxicchat_001", 14, "opposite expressed handling of drug price"),
    ("contrast_ragtruth_001", 4, "nationality versus geographical origin"),
    ("contrast_ragtruth_001", 46, "supported warning and source synthesis"),
]
rows = list(map(json.loads, gzip.open(R/"broad_contrast/corpus/rows.jsonl.gz","rt")))
rubrics = json.loads((R/"TASK_RUBRICS.json").read_text())
packet, private = [], []
for batch, index, reason in selections:
    text = json.loads((R/"broad_contrast/packets"/(batch+".jsonl")).read_text().splitlines()[index])
    matches = [x for x in rows if x["text_id"] == text["id"]]
    contexts = {json.dumps(x["context"],sort_keys=True):x["context"] for x in matches}
    for key, context in contexts.items():
        audit_id = "aux_"+hashlib.sha256((text["id"]+key).encode()).hexdigest()[:20]
        packet.append(dict(
            id=audit_id, dataset=matches[0]["dataset"], reason_for_selection=reason,
            task=rubrics[matches[0]["dataset"]]["task"],
            **context, text=text["text"]))
        private.append(dict(id=audit_id,text_id=text["id"],batch=batch,row=index,
                            matched_records=[x for x in matches if x["context"]==context]))
target = out/"masked_packet.jsonl"
assert not target.exists()
target.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in packet))
with gzip.open(out/"private_mapping.jsonl.gz","wt") as f:
    f.writelines(json.dumps(x,ensure_ascii=False)+"\n" for x in private)
(out/"DESIGN.json").write_text(json.dumps(dict(
    scope="Purposive follow-up, not prevalence or independent verification.",
    n_texts=len(selections),n_context_pairs=len(packet),
    masking="Target gold, original control/NN outcomes and judge/backbone identity hidden until manual coding is frozen. Prose and visible neighbour annotations retained.",
    packet_sha256=hashlib.sha256(target.read_bytes()).hexdigest()
),indent=2)+"\n")
print(len(packet),target)
