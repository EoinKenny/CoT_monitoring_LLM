"""Freeze a second purposive source check with original outcomes concealed."""
from pathlib import Path
import gzip
import hashlib
import json

R = Path(__file__).resolve().parents[1]
out = R/"broad_contrast"/"context_safety_followup"
out.mkdir(exist_ok=True)
selections = [
    ("contrast_wildguard_000",8,"Does never purely physical establish a platonic relationship?"),
    ("contrast_wildguard_000",13,"Does energy-saving response override malicious requested aim?"),
    ("contrast_wildguard_000",21,"Opposite expressed interpretation of energy-saving content"),
    ("contrast_high_stakes_002",43,"Does emergency-surgery reference justify low-stakes routine documentation?"),
    ("contrast_high_stakes_004",20,"Does planning-stage slope analysis avoid safety consequences?"),
    ("contrast_beavertails_000",24,"Does safe-labelled gross reference support harmless-story criterion?"),
    ("contrast_beavertails_001",6,"Is explanation of reference label supported by reference text?"),
    ("contrast_beavertails_002",47,"Is benign cleaning interpretation supported by ambiguous context?"),
]
rows=list(map(json.loads,gzip.open(R/"broad_contrast/corpus/rows.jsonl.gz","rt")))
rubrics=json.loads((R/"TASK_RUBRICS.json").read_text())
packet,private=[],[]
for batch,index,reason in selections:
    text=json.loads((R/"broad_contrast/packets"/(batch+".jsonl")).read_text().splitlines()[index])
    matches=[x for x in rows if x["text_id"]==text["id"]]
    contexts={json.dumps(x["context"],sort_keys=True):x["context"] for x in matches}
    for key,context in contexts.items():
        aid="aux_"+hashlib.sha256((text["id"]+key).encode()).hexdigest()[:20]
        packet.append(dict(id=aid,dataset=matches[0]["dataset"],reason_for_selection=reason,
                           task=rubrics[matches[0]["dataset"]]["task"],**context,text=text["text"]))
        private.append(dict(id=aid,text_id=text["id"],batch=batch,row=index,
                            matched_records=[x for x in matches if x["context"]==context]))
dest=out/"masked_packet.jsonl"
assert not dest.exists()
dest.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in packet))
with gzip.open(out/"private_mapping.jsonl.gz","wt") as f:
    f.writelines(json.dumps(x,ensure_ascii=False)+"\n" for x in private)
(out/"DESIGN.json").write_text(json.dumps(dict(
    scope="Purposive hypothesis/source checks, not prevalence or independent verification.",
    n_texts=len(selections),n_context_pairs=len(packet),
    masking="Target gold, original scores and outcome category, judge and backbone hidden until coding freeze.",
    packet_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2)+"\n")
print(len(packet))
