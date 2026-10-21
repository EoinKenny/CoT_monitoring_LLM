"""Answer-level comparisons on frozen paired cases; no semantic inference."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
P = R.parent
PROJECT = P.parents[1]
KEY = ["judge", "backbone", "sample_id"]
BIG = {"reasoning", "generated_text", "raw_generated_text", "saved_reasoning", "source_record"}


def read_metadata(path):
    with gzip.open(path, "rt") as f:
        return pd.DataFrame([{k:v for k,v in json.loads(l).items() if k not in BIG} for l in f])


def subsets(d):
    yield "all_selected_flips", d
    yield "contrast_yes", d[d.contrast.eq("yes")]
    yield "similarity_yes", d[d.similarity.eq("yes")]
    yield "similarity_only", d[d.similarity.eq("yes") & d.contrast.eq("no")]


def groups(d):
    for family in ["all10"] + sorted(d.family.unique().tolist()):
        f = d if family == "all10" else d[d.family.eq(family)]
        for judge in ["all"] + sorted(f.judge.unique().tolist()):
            j = f if judge == "all" else f[f.judge.eq(judge)]
            for outcome in ["all", "helped", "harmed"]:
                z = j if outcome == "all" else j[j.flip_type.eq(outcome)]
                if len(z):
                    yield dict(family=family, judge=judge, original_outcome=outcome), z


def rate(w, v):
    return float(np.dot(w, np.asarray(v, float))/w.sum()) if w.sum() else None


def main():
    first = read_metadata(R/"corpus/rows.jsonl.gz")
    after = read_metadata(P/"corpus/rows.jsonl.gz")
    assert len(first) == len(after) == 35919
    assert not first.duplicated(KEY).any() and not after.duplicated(KEY).any()
    meta = ["dataset", "true_label", "control_probability", "nn_probability", "flip_type",
            "contrast", "similarity", "cluster"]
    paired = first.merge(after[KEY+meta+["score_valid", "reason_probability"]],
                         on=KEY, suffixes=("", "_after"), validate="one_to_one")
    for c in meta:
        assert paired[c].fillna("<NA>").equals(paired[c+"_after"].fillna("<NA>")), c
    w = pd.read_csv(PROJECT/"reproduction/nn_contrast_investigation_20261020/results/cases.csv.gz",
                    usecols=["policy","model","backbone","sample_id","w","cell"])
    w = w[w.policy.eq("judge50_direct")].rename(columns={"model":"judge"})
    assert not w.duplicated(KEY).any()
    paired = paired.merge(w[KEY+["w","cell"]], on=KEY, how="left", validate="one_to_one")
    paired["weight_available"] = paired.w.notna()
    paired["both_valid"] = paired.score_valid & paired.score_valid_after
    paired["first_correct"] = paired.score_valid & paired.reason_probability.ge(50).eq(paired.true_label.eq(1))
    paired["after_correct"] = paired.score_valid_after & paired.reason_probability_after.ge(50).eq(paired.true_label.eq(1))
    paired["first_retains"] = paired.score_valid & paired.reason_probability.ge(50).eq(paired.nn_probability.ge(50))
    paired["after_retains"] = paired.score_valid_after & paired.reason_probability_after.ge(50).eq(paired.nn_probability.ge(50))
    records, paired_records = [], []
    for subset, s in subsets(paired):
        for keys, q in groups(s):
            z = q[q.weight_available]
            for order, valid_col, correct_col, retain_col in [
                ("probability_first","score_valid_after","after_correct","after_retains"),
                ("reasoning_first","score_valid","first_correct","first_retains")]:
                valid = z[z[valid_col]]
                records.append(dict(subset=subset, **keys, order=order,
                    selected_rows=len(q), weighted_rows=len(z), valid=int(q[valid_col].sum()),
                    unscored=int((~q[valid_col]).sum()),
                    correct_valid=int(q[correct_col].sum()),
                    retains_valid=int(q[retain_col].sum()),
                    weighted_valid_rate=rate(z.w,z[valid_col]),
                    weighted_accuracy_valid=rate(valid.w,valid[correct_col]),
                    weighted_retention_valid=rate(valid.w,valid[retain_col]),
                    accuracy_lower_if_unscored_wrong=rate(z.w,z[correct_col]),
                    accuracy_upper_if_unscored_correct=rate(z.w,z[correct_col] | ~z[valid_col])))
            b = z[z.both_valid].copy()
            delta = b.first_correct.astype(int)-b.after_correct.astype(int)
            mean = rate(b.w,delta)
            if len(b) and mean is not None:
                influence = pd.DataFrame({"cluster":b.cluster,"v":b.w*(delta-mean)}).groupby("cluster").v.sum()
                n = len(influence)
                se = np.sqrt(np.square(influence).sum()*n/max(n-1,1))/b.w.sum()
            else:
                se = np.nan
            paired_records.append(dict(subset=subset, **keys, selected_rows=len(q),
                both_valid_rows=len(b), both_valid_targets=b.cluster.nunique(),
                first_correct_after_wrong=int((b.first_correct & ~b.after_correct).sum()),
                first_wrong_after_correct=int((~b.first_correct & b.after_correct).sum()),
                both_correct=int((b.first_correct & b.after_correct).sum()),
                both_wrong=int((~b.first_correct & ~b.after_correct).sum()),
                weighted_after_accuracy=rate(b.w,b.after_correct),
                weighted_first_accuracy=rate(b.w,b.first_correct),
                first_minus_after=mean,
                lower=mean-1.96*se if mean is not None else None,
                upper=mean+1.96*se if mean is not None else None))
    out=R/"results";out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/"answer_outcomes_by_order.csv",index=False)
    pd.DataFrame(paired_records).to_csv(out/"paired_answer_order_comparison.csv",index=False)
    paired.to_csv(out/"paired_metadata.csv.gz",index=False)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(), paired_rows=len(paired),
        valid_reasoning_first=int(paired.score_valid.sum()), valid_probability_first=int(paired.score_valid_after.sum()),
        both_valid=int(paired.both_valid.sum()), with_weight=int(paired.weight_available.sum()),
        frozen_metadata_equal=meta, no_new_calls=True,
        population="Originally observed fixed50 test flips only, not all test cases or prospective deployment.",
        weighting="Frozen original inverse-inclusion weights with equal cell mass; conditional weighted rates within displayed subset.",
        missing="Unscored first-order outputs retained in denominator and worst/best correctness bounds. Paired order comparison conditions on both valid; not assumed missing at random.",
        intervention="Reasoning-first preserves 640 prior 256-token responses; other calls had 512 cap. Probability-first used 256 cap. Order and output budget are not separately randomized.",
        uncertainty="Approximate target-cluster delta-method intervals conditional on frozen selected cases and both-valid outputs, not correction for intervention selection or missingness.",
        semantics="Answer-level analysis independent of still-incomplete second-round expressed-rationale census.")
    (out/"ANSWER_COMPARISON_METHOD.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
