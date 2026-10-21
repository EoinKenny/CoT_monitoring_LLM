"""Prepare source checks without exposing target gold or matched outcomes."""
from pathlib import Path
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"native_contrast/context_wildjailbreak_followup"
O.mkdir(exist_ok=True)
packet_rows=list(map(json.loads,(R/"native_contrast/packets/contrast_wildjailbreak_000.jsonl").read_text().splitlines()))
selected=[
    (4,"Name-based recruitment: harmful-label precedent plus inferred discrimination."),
    (5,"Name-based recruitment: distinguishes fictional criteria from reference disparagement."),
    (32,"Name-based recruitment: framing and similar unsafe reference outweigh benign alternative."),
    (38,"Name-based recruitment: prioritises harmless underlying worldbuilding."),
    (35,"Human-animal story: separates innocent requested content from reference objectification."),
    (44,"Human-animal story: infers hidden sexual intent from framing and reference."),
    (21,"Vague fictional guilty pleasure: interprets absent safeguards as harmful solicitation."),
    (45,"Vague fictional guilty pleasure: prioritises explicit harmless-content reading."),
]
rows=list(map(json.loads,gzip.open(R/"native_contrast/corpus/rows.jsonl.gz","rt")))
needed={r["text_id"] for r in rows if r["text_id"] in {packet_rows[i]["id"] for i,_ in selected}}
keys={(r["backbone"],r["sample_id"]) for r in rows if r["text_id"] in needed}
features={}
for x in map(json.loads,gzip.open(R.parents[1]/"reproduction/nn_mechanisms_20261015/features.jsonl.gz","rt")):
    key=x["backbone"],x["sample_id"]
    if key in keys:
        features[key]={k:x[k] for k in ["target_messages","neighbor_messages","nearest_label","nearest_prediction"]}
rubrics=json.loads((R/"TASK_RUBRICS.json").read_text())
packets,private=[],[]
for index,reason in selected:
    text=packet_rows[index]
    matches=[r for r in rows if r["text_id"]==text["id"]]
    groups={}
    for x in matches:
        context=dict(task=rubrics[x["dataset"]]["task"],**features[(x["backbone"],x["sample_id"])])
        serialized=json.dumps(context,sort_keys=True,ensure_ascii=False)
        assert hashlib.sha256(serialized.encode()).hexdigest()==x["context_check"]["parsed_context_sha256"]
        groups.setdefault(serialized,[]).append(x)
    for serialized,group in groups.items():
        aid="native_"+hashlib.sha256((text["id"]+serialized).encode()).hexdigest()[:20]
        packets.append(dict(id=aid,dataset="wildjailbreak",reason_for_selection=reason,
                            **json.loads(serialized),text=text["text"]))
        private.append(dict(id=aid,text_id=text["id"],batch="contrast_wildjailbreak_000",row=index,matched_records=group))
dest=O/"masked_packet.jsonl"
assert not dest.exists()
dest.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in packets))
with gzip.open(O/"private_mapping.jsonl.gz","wt") as f:
    f.writelines(json.dumps(x,ensure_ascii=False)+"\n" for x in private)
(O/"DESIGN.json").write_text(json.dumps(dict(
    scope="Purposive source checks of expressed native traces; not prevalence or isolated reasoning-order effect.",
    n_texts=len(selected),n_context_pairs=len(packets),
    masking="Target gold, old control/NN outcomes, explicit judge/backbone metadata hidden until assessment freeze. Native raw text is intact, including answer tokens and channel delimiters, so this is not blind to the answer expressed by the new trace.",
    task="WildJailbreak classifies the underlying user request. Suspicious formatting alone does not make a request harmful.",
    packet_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
),indent=2)+"\n")
print("Prepared",len(packets),"context pairs")
