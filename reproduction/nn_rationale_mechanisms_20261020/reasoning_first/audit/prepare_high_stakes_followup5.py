"""Check rejection of irrelevant reference and competing interpretations of a shared hygiene analogy, selected before outcome lookup."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json

R = Path(__file__).resolve().parents[1]
D = R / "context_high_stakes_followup5"
D.mkdir(exist_ok=True)
selection = {"contrast_high_stakes_023": [70], "contrast_high_stakes_032": [45, 49]}
texts = []
for batch, indices in selection.items():
    rows = list(map(json.loads, (R/"packets"/(batch+".jsonl")).read_text().splitlines()))
    texts.extend((batch, i, rows[i]) for i in indices)
ids = {x[2]["id"] for x in texts}
with gzip.open(R/"corpus/rows.jsonl.gz", "rt") as f:
    matches = [r for r in map(json.loads, f) if r["text_id"] in ids]
context_ids = {r["context_id"] for r in matches}
with gzip.open(R/"corpus/contexts.jsonl.gz", "rt") as f:
    contexts = {r["context_id"]: r for r in map(json.loads, f) if r["context_id"] in context_ids}
fields = ["trace_id", "judge", "backbone", "true_label", "control_probability",
          "nn_probability", "flip_type", "reason_probability", "score_valid",
          "retains_original_nn", "rationale_correct", "finish_reason", "generated_tokens"]
packets, private = [], []
for batch, i, text in texts:
    related = [r for r in matches if r["text_id"] == text["id"]]
    for context_id in sorted({r["context_id"] for r in related}):
        audit_id = "high5_" + hashlib.sha256((text["id"]+context_id).encode()).hexdigest()[:20]
        packets.append(dict(audit_id=audit_id, read_text_id=text["id"], batch=batch,
                            row=i, context=contexts[context_id], rationale=text["text"]))
        private.extend(dict(audit_id=audit_id, **{k: r[k] for k in fields})
                       for r in related if r["context_id"] == context_id)
assert not (D/"packets.jsonl").exists()
(D/"packets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in packets))
with gzip.open(D/"private_mapping.jsonl.gz", "wt") as f:
    f.write("".join(json.dumps(r)+"\n" for r in private))
(D/"DESIGN.json").write_text(json.dumps(dict(
    created_utc=datetime.now(timezone.utc).isoformat(), selection=selection,
    n_unique_texts=len(texts), n_context_packets=len(packets), n_trace_context_pairs=len(private),
    purpose="Check a potentially useful rejection of an irrelevant reference, plus two infant-care rationales invoking a pet-care neighbour with apparently different consequences. Selected from fully read prose before original outcomes or model identity lookup.",
    limitations="Three purposive single-reader checks, not prevalence or causal estimates. Raw new answer text visible, original outcome and identity withheld until coding freeze. These exact context identities and outcomes have not yet been inspected; neither classification is assumed correct based on the rationale alone.",
    source_corpus_sha256=hashlib.sha256((R/"corpus/rows.jsonl.gz").read_bytes()).hexdigest()
), indent=2)+"\n")
print("Prepared", len(packets), "masked packets and", len(private), "private mappings.")
