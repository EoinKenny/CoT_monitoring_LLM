"""Prepare four purposive source checks without displaying outcome fields."""
from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json

R = Path(__file__).resolve().parents[1]
D = R/"context_convention_followup"
D.mkdir(exist_ok=True)
selection = {"contrast_wildjailbreak_007": [57,60], "contrast_wildjailbreak_008": [5,14]}
texts = []
for batch, indices in selection.items():
    rows = [json.loads(s) for s in (R/"packets"/(batch+".jsonl")).read_text().splitlines()]
    for i in indices:
        texts.append((batch,i,rows[i]))
ids = {x[2]["id"] for x in texts}
with gzip.open(R/"corpus/rows.jsonl.gz", "rt") as f:
    matches = [json.loads(s) for s in f if s.strip()]
matches = [r for r in matches if r["text_id"] in ids]
context_ids = {r["context_id"] for r in matches}
with gzip.open(R/"corpus/contexts.jsonl.gz", "rt") as f:
    contexts = {r["context_id"]:r for r in map(json.loads,f) if r["context_id"] in context_ids}
packets, private = [], []
fields = ["trace_id","judge","backbone","true_label","control_probability","nn_probability",
          "flip_type","reason_probability","score_valid","retains_original_nn",
          "rationale_correct","finish_reason","generated_tokens"]
for batch,i,text in texts:
    related = [r for r in matches if r["text_id"] == text["id"]]
    for context_id in sorted({r["context_id"] for r in related}):
        audit_id = "convention_"+hashlib.sha256((text["id"]+context_id).encode()).hexdigest()[:20]
        packets.append(dict(audit_id=audit_id,read_text_id=text["id"],batch=batch,row=i,
                            context=contexts[context_id],rationale=text["text"]))
        for r in related:
            if r["context_id"] == context_id:
                private.append(dict(audit_id=audit_id,**{k:r[k] for k in fields}))
assert len(packets) == 4
assert not (D/"packets.jsonl").exists()
(D/"packets.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in packets))
with gzip.open(D/"private_mapping.jsonl.gz","wt") as f:
    for r in private:f.write(json.dumps(r)+"\n")
(D/"DESIGN.json").write_text(json.dumps(dict(
    created_utc=datetime.now(timezone.utc).isoformat(), selection=selection,
    n_unique_texts=len(texts),n_context_packets=len(packets),n_trace_context_pairs=len(private),
    purpose="Purposive source check of explicit label precedent versus task-relevant differences. Selected from already-read prose, without selecting on gold, original outcomes or model identity.",
    limitations="Single-reader illustrations, not prevalence estimates or causal evidence. New numeric answer fragments can remain in rationale. No new inference.",
    reader="/root", source_corpus_sha256=hashlib.sha256((R/"corpus/rows.jsonl.gz").read_bytes()).hexdigest()
),indent=2)+"\n")
print("Prepared",len(packets),"masked context packets and",len(private),"private mappings.")
