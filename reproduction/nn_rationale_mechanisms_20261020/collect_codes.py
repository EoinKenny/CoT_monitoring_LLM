"""Validate semantic read receipts, expand exact-text codes, and track coverage."""
from pathlib import Path
from datetime import datetime, timezone
import json, gzip, hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
ALLOWED=dict(relation=set("NSCBU"),bridge=set("TLDRQU"),conclusion=set("PNMU"),quality=set("AIU"))
AXES={"LABEL","INTENT","BEHAVIOUR","SEVERITY","CONTEXT","FACT","TOPIC","OTHER","NONE"}

def main():
    batches=pd.read_csv(ROOT/"BATCHES.csv")
    codes,progress=[],[]
    for b in batches.to_dict("records"):
        p=ROOT/"codes"/(b["batch_id"]+".jsonl")
        receipt=p.with_name(b["batch_id"]+".receipt.json")
        status="pending"
        if p.exists() and receipt.exists():
            packet=[json.loads(l) for l in Path(b["path"]).read_text().splitlines()]
            data=[json.loads(l) for l in p.read_text().splitlines()]
            rec=json.load(open(receipt))
            assert rec["read_complete"] is True, b["batch_id"]
            assert rec["n"]==b["n"]==len(data),b["batch_id"]
            assert rec["input_sha256"]==b["sha256"]==hashlib.sha256(Path(b["path"]).read_bytes()).hexdigest()
            assert len({x["id"] for x in data})==len(data), b["batch_id"]
            assert {x["id"] for x in data}=={x["id"] for x in packet},b["batch_id"]
            text={x["id"]:x["text"] for x in packet}
            for c in data:
                for field,allowed in ALLOWED.items():
                    assert c[field] in allowed,(b["batch_id"],c["id"],field)
                assert c["axes"] and set(c["axes"])<=AXES,(b["batch_id"],c["id"],c["axes"])
                assert c["evidence"] and all(isinstance(q,str) and q and q in text[c["id"]]
                                             for q in c["evidence"]),(b["batch_id"],c["id"],"quotes")
                codes.append(dict(c,batch_id=b["batch_id"],reader=rec["reader"],
                                  code_version=rec["code_version"],dataset=b["dataset"]))
            status="read_and_validated"
        elif p.exists() or receipt.exists():
            status="in_progress"
        progress.append(dict(**b,status=status))
    p=pd.DataFrame(progress)
    p.to_csv(ROOT/"results/reading_progress.csv",index=False)
    c=pd.DataFrame(codes)
    if len(c):
        assert not c.id.duplicated().any()
        c.to_json(ROOT/"results/validated_codes.jsonl",orient="records",lines=True,force_ascii=False)
    with gzip.open(ROOT/"corpus/rows.jsonl.gz","rt") as f:
        r=pd.DataFrame([json.loads(l) for l in f])
    r["available"]=r.reasoning.fillna("").str.strip().ne("")
    r["read_complete"]=r.text_id.isin(c.id if len(c) else [])
    r["coverage_state"]=np.select([~r.available,r.read_complete],["unavailable","read_and_validated"],default="pending")
    r[["trace_id","text_id","context_id","judge","backbone","dataset","contrast","similarity",
       "requested_census","similarity_harm","available","read_complete","coverage_state"]].to_csv(
        ROOT/"results/coverage_ledger.csv.gz",index=False)
    context_status=ROOT/"context_audit/STATUS.json"
    context_complete=context_status.exists() and json.load(open(context_status)).get("all_double_read",False)
    corpus_complete=bool((p.status=="read_and_validated").all())
    summary={"checked_utc":datetime.now(timezone.utc).isoformat(),
             "all_semantic_work_complete":corpus_complete and context_complete,
             "batches_read":int((p.status=="read_and_validated").sum()),
             "batches_total":len(p),"unique_texts_read":len(c),
             "unique_texts_total":int(batches.n.sum()),
             "contrast_records":int(r.requested_census.sum()),
             "contrast_available":int((r.requested_census&r.available).sum()),
             "contrast_records_read":int((r.requested_census&r.read_complete).sum()),
             "similarity_harm_records":int(r.similarity_harm.sum()),
             "similarity_harm_records_read":int((r.similarity_harm&r.read_complete).sum()),
             "remaining_contrast":int((r.requested_census&r.available&~r.read_complete).sum()),
             "remaining_similarity_harm":int((r.similarity_harm&r.available&~r.read_complete).sum()),
             "context_audit_complete":context_complete,
             "all_available_local_census_read":corpus_complete,
             "scope_limit":"Local saved probability-first flip collection. Remote reasoning-first and broader paused-run traces remain outside this receipt.",
             "semantic_scope":"Every read receipt covers full expressed rationale; contextual validity is a separate audit.",
             "verifier_scope":"Checks receipt integrity, IDs, schema and evidence substrings. Does not independently prove semantic correctness of codes."}
    (ROOT/"results/READING_STATUS.json").write_text(json.dumps(summary,indent=2)+"\n")
    if len(c):
        c=c.rename(columns={"id":"text_id","dataset":"code_dataset"})
        joined=r.merge(c,on="text_id",how="inner",validate="many_to_one")
        assert (joined.dataset==joined.code_dataset).all()
        joined["prose_assessable"]=joined.conclusion.isin(["P","N"])
        joined["score_prose_conflict"]=joined.prose_assessable & joined.score_valid & (
            (joined.conclusion=="P") != (joined.reason_probability>=50))
        joined["mixed_prose"]=joined.conclusion=="M"
        joined.drop(columns=["reasoning","generated_text"]).to_json(
            ROOT/"results/coded_trace_metadata.jsonl.gz",orient="records",lines=True,compression="gzip")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
