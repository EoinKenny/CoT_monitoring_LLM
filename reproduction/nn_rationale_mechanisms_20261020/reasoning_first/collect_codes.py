"""Validate separate reasoning-first census; never infer unread semantic labels."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import pandas as pd

R = Path(__file__).resolve().parent
ALLOWED = dict(relation=set("NSCBU"), bridge=set("TLDRQU"),
               conclusion=set("PNMU"), quality=set("AIU"))
AXES = {"LABEL", "INTENT", "BEHAVIOUR", "SEVERITY", "CONTEXT", "FACT", "TOPIC", "OTHER", "NONE"}


def check(code, text):
    for key, allowed in ALLOWED.items():
        assert code[key] in allowed, (code["id"], key)
    assert code["axes"] and set(code["axes"]) <= AXES
    assert code["evidence"] and all(isinstance(q, str) and q and q in text for q in code["evidence"])


def main():
    out = R / "results"
    out.mkdir(exist_ok=True)
    batches = pd.read_csv(R / "BATCHES.csv")
    rows = pd.DataFrame([json.loads(x) for x in gzip.open(R / "corpus/rows.jsonl.gz", "rt")])
    text_lookup = rows.drop_duplicates("text_id").set_index("text_id").reasoning.to_dict()
    codes, progress = [], []
    for b in batches.to_dict("records"):
        p = R / "codes" / (b["batch_id"] + ".jsonl")
        receipt = p.with_suffix(".receipt.json")
        status = "pending"
        if p.exists() and receipt.exists():
            packet_path = Path(b["path"])
            packet = [json.loads(x) for x in packet_path.read_text().splitlines()]
            data = [json.loads(x) for x in p.read_text().splitlines()]
            rec = json.loads(receipt.read_text())
            assert rec["read_complete"] and rec["n"] == b["n"] == len(data)
            assert rec["input_sha256"] == b["sha256"] == hashlib.sha256(packet_path.read_bytes()).hexdigest()
            assert len({x["id"] for x in data}) == len(data)
            assert {x["id"] for x in data} == {x["id"] for x in packet}
            for c in data:
                check(c, text_lookup[c["id"]])
                codes.append(dict(c, batch_id=b["batch_id"], reader=rec["reader"],
                                  dataset=b["dataset"], reused=False))
            status = "read_and_validated"
        elif p.exists() or receipt.exists():
            status = "in_progress"
        progress.append(dict(**b, status=status))
    for r in json.loads((R / "REUSED_EXPRESSED_CODES.json").read_text()):
        c = r["code"]
        check(c, text_lookup[c["id"]])
        assert hashlib.sha256((r["dataset"] + "\n" + text_lookup[c["id"]]).encode()).hexdigest() == c["id"]
        codes.append(dict(c, batch_id="exact_prior_text", reader="prior_full_read",
                          dataset=r["dataset"], reused=True))
    codes = pd.DataFrame(codes)
    assert not codes.id.duplicated().any()
    codes.to_json(out / "validated_codes.jsonl", orient="records", lines=True, force_ascii=False)
    p = pd.DataFrame(progress)
    p.to_csv(out / "reading_progress.csv", index=False)
    rows["available"] = rows.reasoning.fillna("").str.strip().ne("")
    rows["read_complete"] = rows.text_id.isin(codes.id)
    cols = ["trace_id", "text_id", "context_id", "judge", "backbone", "dataset",
            "contrast", "similarity", "requested_census", "similarity_harm",
            "available", "read_complete", "score_valid", "reasoning_complete"]
    rows[cols].to_csv(out / "coverage_ledger.csv.gz", index=False)
    required = rows.requested_census & rows.available
    cp = p[p.role == "contrast"]
    summary = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        required_scope="Every available contrast-labelled reasoning-first response, including capped/unscored prose.",
        contrast_records=int(rows.requested_census.sum()),
        contrast_available=int(required.sum()),
        contrast_records_read=int((required & rows.read_complete).sum()),
        remaining_contrast=int((required & ~rows.read_complete).sum()),
        contrast_batches_read=int(cp.status.eq("read_and_validated").sum()),
        contrast_batches_total=len(cp),
        new_unique_texts_read=int((~codes.reused).sum()),
        exact_prior_texts_reused=int(codes.reused.sum()),
        contrast_census_complete=bool(rows.loc[required, "read_complete"].all()),
        similarity_harm_records_read=int((rows.similarity_harm & rows.read_complete).sum()),
        similarity_harm_records=int(rows.similarity_harm.sum()),
        additional_similarity_scope="Prepared for possible later use; not part of required second-round census. Primary probability-first census covers every available similarity-harm response.",
        context_scope="Expressed prose only; primary 200-case full-context audit does not transfer automatically.",
    )
    (out / "READING_STATUS.json").write_text(json.dumps(summary, indent=2) + "\n")
    c = codes.rename(columns={"id": "text_id", "dataset": "code_dataset"})
    joined = rows.merge(c, on="text_id", how="inner", validate="many_to_one")
    assert joined.dataset.eq(joined.code_dataset).all()
    joined["prose_assessable"] = joined.conclusion.isin(["P", "N"])
    joined["score_prose_conflict"] = joined.prose_assessable & joined.score_valid & (
        joined.conclusion.eq("P") != joined.reason_probability.ge(50))
    joined["mixed_prose"] = joined.conclusion.eq("M")
    big = ["reasoning", "generated_text", "raw_generated_text", "saved_reasoning", "source_record"]
    joined.drop(columns=big, errors="ignore").to_json(
        out / "coded_trace_metadata.jsonl.gz", orient="records", lines=True, compression="gzip")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
