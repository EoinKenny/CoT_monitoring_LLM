"""Freeze an outcome-masked, full-rationale reading census from saved outputs."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "reproduction/nn_reason_flips_20261017/FINAL_ANALYSIS_SNAPSHOT.json.gz"
ANN = PROJECT / "reproduction/nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz"
FEATURES = PROJECT / "reproduction/nn_mechanisms_20261015/features.jsonl.gz"
TOX = {"aegis", "beavertails", "toxicchat", "toxigen"}
HAL = {"fava", "halueval", "ragtruth"}

def sha(v):
    return hashlib.sha256(v.encode()).hexdigest()

def writej(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def main():
    for d in ["corpus", "packets", "codes", "results", "audit"]:
        (ROOT / d).mkdir(exist_ok=True)
    rows = json.load(gzip.open(SOURCE, "rt"))["rows"]
    anns = pd.read_csv(ANN).set_index(["backbone", "sample_id"]).to_dict("index")
    contexts = {}
    with gzip.open(FEATURES, "rt") as f:
        for line in f:
            a = json.loads(line)
            if a["split"] == "test":
                contexts[(a["backbone"], a["sample_id"])] = {
                    k: a[k] for k in ["dataset", "target_messages", "neighbor_messages",
                                      "nearest_label", "nearest_prediction"]
                }
    corpus, texts = [], {}
    for r in rows:
        a = anns[(r["backbone"], r["sample_id"])]
        context = contexts[(r["backbone"], r["sample_id"])]
        reasoning = r.get("reasoning") or ""
        trace_id = sha("|".join([r["judge"], r["backbone"], r["sample_id"], "after256"]))
        # Dedup is ONLY for coding the words expressed, never contextual validity.
        text_id = sha(r["dataset"] + "\n" + reasoning)
        context_id = sha(json.dumps(context, ensure_ascii=False, sort_keys=True))
        contrast = a["majority_important_contrast"] == "yes"
        similarity = a["majority_labelled_similarity"] == "yes"
        p = r.get("reason_probability")
        valid = isinstance(p, (int, float)) and 0 <= p <= 100
        row = dict(r, trace_id=trace_id, text_id=text_id, context_id=context_id,
                   family="toxicity4" if r["dataset"] in TOX else
                          "hallucination3" if r["dataset"] in HAL else "other_safety3",
                   contrast=a["majority_important_contrast"],
                   similarity=a["majority_labelled_similarity"],
                   cluster=a["cluster"], source_index=int(a["source_index"]),
                   annotation_weight=float(a["weight"]),
                   paired_common_valid=bool(a["common_valid"]),
                   nearest_label=int(a["nearest_label"]),
                   nearest_prediction=int(a["nearest_prediction"]),
                   nearest_cosine=float(a["nearest_cosine"]),
                   score_valid=valid,
                   retains_original_nn=(bool((p >= 50) == (r["nn_probability"] >= 50))
                                        if valid else None),
                   rationale_correct=(bool((p >= 50) == r["true_label"]) if valid else None),
                   requested_census=bool(contrast),
                   similarity_harm=bool(similarity and r["flip_type"] == "harmed"))
        corpus.append(row)
        if contrast or row["similarity_harm"]:
            text = texts.setdefault(text_id, {
                "text_id": text_id, "dataset": r["dataset"], "rationale": reasoning,
                "contrast_requested": False, "similarity_harm_requested": False,
                "trace_ids": [], "context_ids": []
            })
            text["contrast_requested"] |= contrast
            text["similarity_harm_requested"] |= row["similarity_harm"]
            text["trace_ids"].append(trace_id)
            text["context_ids"].append(context_id)
    with gzip.open(ROOT/"corpus/rows.jsonl.gz", "wt") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gzip.open(ROOT/"corpus/contexts.jsonl.gz", "wt") as f:
        for context in contexts.values():
            cid = sha(json.dumps(context, ensure_ascii=False, sort_keys=True))
            f.write(json.dumps(dict(context, context_id=cid), ensure_ascii=False) + "\n")
    with gzip.open(ROOT/"corpus/text_map.jsonl.gz", "wt") as f:
        for text in texts.values():
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
    batches = []
    for role in ["contrast", "similarity_harm_only"]:
        eligible = [t for t in texts.values() if t["rationale"].strip() and
                    (t["contrast_requested"] if role == "contrast" else not t["contrast_requested"])]
        for dataset in sorted({t["dataset"] for t in eligible}):
            selected = sorted([t for t in eligible if t["dataset"] == dataset],
                              key=lambda t: t["text_id"])
            for i in range(0, len(selected), 100):
                bid = f"{role}_{dataset}_{i//100:03d}"
                packet = [{"id": t["text_id"], "text": t["rationale"]}
                          for t in selected[i:i+100]]
                path = ROOT / "packets" / (bid + ".jsonl")
                path.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in packet))
                batches.append(dict(batch_id=bid, dataset=dataset, role=role, n=len(packet),
                                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(batches).to_csv(ROOT/"BATCHES.csv", index=False)
    writej(ROOT/"CORPUS_MANIFEST.json", {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(ANN.read_bytes()).hexdigest(),
        "rows": len(corpus), "contrast_rows": sum(r["requested_census"] for r in corpus),
        "similarity_harm_rows": sum(r["similarity_harm"] for r in corpus),
        "unique_full_texts_to_read": sum(b["n"] for b in batches),
        "contrast_unique_full_texts_to_read": sum(b["n"] for b in batches if b["role"]=="contrast"),
        "batches": len(batches),
        "dedup_scope": "Only expressed-word codes: exact same dataset and full rationale. Context-specific correctness/failure codes cannot be inherited across different contexts.",
        "blinding": "Packets hide judge, backbone, original/new numeric scores, target gold and original flip outcome.",
        "source_corpus_selection": "Original fixed50 test flips; no original unchanged cases.",
        "context_audit": "Separate full-context semantic validation required for claims of imported facts, unsupported intent, erroneous rules or factual contradictions.",
    })
    print(json.dumps(json.load(open(ROOT/"CORPUS_MANIFEST.json")), indent=2))

if __name__ == "__main__":
    main()
