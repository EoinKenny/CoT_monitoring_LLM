"""Verify frozen manual source checks against saved inputs and original raw records."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
D=R/"context_high_stakes_followup"
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
mixed_neighbor=examples[2]["context"]["neighbor_messages"]
assert mixed_neighbor==examples[5]["context"]["neighbor_messages"]
matching_context_ids={k for k,v in contexts.items() if v["neighbor_messages"]==mixed_neighbor}
exposed=[r for r in allrows if r["context_id"] in matching_context_ids]
exposure=dict(
    scope="Descriptive reuse within the original fixed50 test-flip selection only; not prevalence among all test cases or all training examples.",
    selected_trace_rows=len(exposed),
    contrast_trace_rows=sum(bool(r["contrast"]) for r in exposed),
    distinct_contexts=len(matching_context_ids),
    distinct_dataset_targets=len({(r["dataset"],r["index"]) for r in exposed}),
    neighbor_sha256=hashlib.sha256(canonical(mixed_neighbor).encode()).hexdigest(),
    observation="Exact frozen neighbour contains dinner-choice prose, repeated data-breach drafts and editing chatter. No cleaning or removal performed.",
)
(D/"REFERENCE_REUSE.json").write_text(json.dumps(exposure,indent=2)+"\n")
receipt=dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),n_traces=len(examples),
    n_contexts=len({e["context"]["context_id"] for e in examples}),
    saved_raw_record_membership_verified=len(examples),
    exact_contexts_and_outcomes_verified=len(examples),
    coding_freeze_verified=True,source_corpus_hashes_verified=True,
    source_files={p:sha(Path(p)) for p in sorted(byfile)},
    reference_reuse=exposure,
)
(D/"VERIFIED.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_files"},indent=2))
