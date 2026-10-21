"""Verify frozen manual source checks against saved inputs and original raw records."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
D=R/"context_high_stakes_followup4"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))
freeze=json.loads((D/"CODING_FREEZE.json").read_text())
assert sha(D/"codes.jsonl")==freeze["code_sha256"]
assert sha(D/"packets.jsonl")==freeze["packet_sha256"]
examples=json.loads((D/"unmasked_examples.json").read_text())
ids={r["outcome"]["trace_id"] for r in examples}
with gzip.open(R/"corpus/rows.jsonl.gz","rt") as f:
    allrows=list(map(json.loads,f))
selected={r["trace_id"]:r for r in allrows if r["trace_id"] in ids}
with gzip.open(R/"corpus/contexts.jsonl.gz","rt") as f:
    contexts={r["context_id"]:r for r in map(json.loads,f)}
manifest=json.loads((R/"CORPUS_MANIFEST.json").read_text())
assert sha(R/"corpus/rows.jsonl.gz")==manifest["rows_sha256"]
assert sha(R/"corpus/contexts.jsonl.gz")==manifest["contexts_sha256"]
byfile=defaultdict(set)
for e in examples:
    row=selected[e["outcome"]["trace_id"]]
    assert e["rationale"]==row["reasoning"]
    assert e["read_text_id"]==row["text_id"]
    assert e["context"]==contexts[row["context_id"]]
    for k,v in e["outcome"].items():
        if k!="audit_id":
            assert row[k]==v,(k,row[k],v)
    byfile[row["source_judgment_file"]].add(canonical(row["source_record"]))
for path,wanted in byfile.items():
    actual={canonical(json.loads(line)) for line in Path(path).read_text().splitlines()}
    assert wanted<=actual,path
receipt=dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),n_traces=len(examples),
    n_contexts=len({e["context"]["context_id"] for e in examples}),
    saved_raw_record_membership_verified=len(examples),
    exact_contexts_and_outcomes_verified=len(examples),
    coding_freeze_verified=True,source_corpus_hashes_verified=True,
    source_files={p:sha(Path(p)) for p in sorted(byfile)},
)
(D/"VERIFIED.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_files"},indent=2))
