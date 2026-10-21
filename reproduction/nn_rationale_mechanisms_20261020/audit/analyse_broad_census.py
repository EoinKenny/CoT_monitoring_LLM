"""Describe complete auxiliary census, without pooling with selected flips."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math

R=Path(__file__).resolve().parents[1]
O=R/"broad_contrast/results"
status=json.loads((O/"READING_STATUS.json").read_text())
assert status["all_available_read"]
rows=list(map(json.loads,gzip.open(R/"broad_contrast/corpus/rows.jsonl.gz","rt")))
codes={x["id"]:x for x in map(json.loads,(O/"validated_codes.jsonl").read_text().splitlines())}
keys={(x["judge"],x["backbone"],x["sample_id"]) for x in rows}
assert len(keys)==len(rows)
source=R.parents[1]/"reproduction/nn_contrast_investigation_20261020/results/cases.csv.gz"
original={}
for x in csv.DictReader(gzip.open(source,"rt")):
    key=(x["model"],x["backbone"],x["sample_id"])
    if key in keys:
        item={k:x[k] for k in ["control_score","nearest_score","true_label"]}
        assert key not in original or original[key]==item
        original[key]=item
assert set(original)==keys
toc={"aegis","beavertails","toxicchat","toxigen"}
hal={"fava","halueval","ragtruth"}
data=[]
for row in rows:
    code=codes[row["text_id"]]
    old=original[(row["judge"],row["backbone"],row["sample_id"])]
    raw=row["original_source"]["source_record"]
    prob=raw["positive_probability"]
    valid=prob is not None and math.isfinite(float(prob)) and 0<=float(prob)<=100
    assert valid==row["original_source"]["score_valid"]
    gold=row["true_label"]
    assert int(old["true_label"])==gold
    cp,np=float(old["control_score"]),float(old["nearest_score"])
    oldflip=(cp>=50)!=(np>=50)
    assert oldflip==row["original_source"]["original_fixed50_flip_selected"]
    oldcorrect=(np>=50)==gold
    correct=(float(prob)>=50)==gold if valid else None
    newchange=(float(prob)>=50)!=(np>=50) if valid else None
    if not valid: change="unscored"
    elif not newchange: change="unchanged"
    elif correct: change="corrected_original_nn"
    else: change="undid_correct_original_nn"
    proseclass={"P":1,"N":0}.get(code["conclusion"])
    conflict=valid and proseclass is not None and (float(prob)>=50)!=proseclass
    data.append(dict(
        dataset=row["dataset"],family="toxicity4" if row["dataset"] in toc else "hallucination3" if row["dataset"] in hal else "other_safety3",
        judge=row["judge"],backbone=row["backbone"],sample_id=row["sample_id"],text_id=row["text_id"],
        gold=gold,control_probability=cp,nn_probability=np,new_probability=prob,valid=valid,
        original_control_nn_flip=oldflip,original_nn_correct=oldcorrect,new_correct=correct,
        new_answer_outcome=change,prose_relation=code["relation"],prose_bridge=code["bridge"],
        prose_conclusion=code["conclusion"],prose_quality=code["quality"],
        prose_class_available=proseclass is not None,prose_score_conflict=conflict,
        explanation_complete=raw["reasoning_complete"],finish_reason=raw["finish_reason"],
        smoke=row["original_source"]["smoke"]))
with (O/"answer_prose_records.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summaries=[]
for dimension in ["all","family","dataset","judge"]:
    groups=defaultdict(list)
    for row in data:groups[row[dimension] if dimension!="all" else "all"].append(row)
    for name,group in sorted(groups.items()):
        changes=Counter(x["new_answer_outcome"] for x in group)
        summary=dict(dimension=dimension,group=name,n=len(group),
                     unique_texts=len({x["text_id"] for x in group}),
                     original_flips=sum(x["original_control_nn_flip"] for x in group),
                     valid=sum(x["valid"] for x in group),
                     original_nn_correct=sum(x["original_nn_correct"] for x in group),
                     new_correct=sum(x["new_correct"] is True for x in group),
                     corrected=changes["corrected_original_nn"],
                     undone=changes["undid_correct_original_nn"],
                     unchanged=changes["unchanged"],
                     unscored=changes["unscored"],
                     capped=sum(not x["explanation_complete"] for x in group),
                     prose_class_available=sum(x["prose_class_available"] for x in group),
                     prose_score_conflict=sum(x["prose_score_conflict"] for x in group),
                     explicit_comparison=sum(x["prose_relation"] in ["S","C","B"] for x in group))
        summaries.append(summary)
with (O/"descriptive_summary.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
method=dict(
    generated_utc=datetime.now(timezone.utc).isoformat(),
    scope="All 1192 additional available test contrast explanations from paused broader collection, separate from primary flip-selected study.",
    selection="Opportunistic saved full/smoke outputs, heavily imbalanced across judges/backbones; not a random unchanged sample.",
    aggregation="Raw counts, no design-weighted or equal-cell performance claim. Repeated target contexts are not independent.",
    reading=status,original_scores_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256((O/"validated_codes.jsonl").read_bytes()).hexdigest(),
    records_sha256=hashlib.sha256((O/"answer_prose_records.csv").read_bytes()).hexdigest(),
    threshold="Every answer comparison uses probability >=50. First probability remains scored answer.",
    semantic_limits="Codes describe expressed prose only. Explicit comparison and score/prose agreement do not establish contextual validity or internal reasoning.",
    no_inference="No new model calls, retries or controller changes.")
(O/"DESCRIPTIVE_METHOD.json").write_text(json.dumps(method,indent=2)+"\n")
print(json.dumps(summaries,indent=2))
