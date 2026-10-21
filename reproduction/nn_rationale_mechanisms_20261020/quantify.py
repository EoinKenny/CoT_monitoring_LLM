"""Selection denominators and answer-retention audit, without semantic inference."""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TOX = ["aegis", "beavertails", "toxicchat", "toxigen"]
HAL = ["fava", "halueval", "ragtruth"]
OTHER = ["high_stakes", "wildguard", "wildjailbreak"]

def cohorts(df):
    yield "all10", df
    for name, ds in [("toxicity4", TOX), ("hallucination3", HAL), ("other_safety3", OTHER)]:
        yield name, df[df.dataset.isin(ds)]

def stats(df, weight="w"):
    w = df[weight].to_numpy(float)
    flip = df.flip.to_numpy(bool)
    h = df.helped.to_numpy(bool)
    a = np.sum(w*h)
    b = np.sum(w*flip)
    ratio = a/b if b else np.nan
    # Cluster-robust ratio delta-method interval conditional on this saved test set.
    z = pd.DataFrame({"cluster":df.cluster, "z": w*(h-ratio*flip)})
    cl = z.groupby("cluster").z.sum().to_numpy()
    se = np.sqrt(np.sum(cl*cl)*len(cl)/max(1,len(cl)-1))/b if b else np.nan
    return dict(rows=len(df), targets=df.cluster.nunique(),
                helped=int(h.sum()), harmed=int((flip & ~h).sum()),
                unchanged=int((~flip).sum()), weighted_helped_fraction=a/w.sum(),
                weighted_harmed_fraction=np.sum(w*(flip & ~h))/w.sum(),
                weighted_unchanged_fraction=np.sum(w*~flip)/w.sum(),
                raw_helped_among_flips=float(h.sum()/flip.sum()) if flip.sum() else None,
                weighted_helped_among_flips=ratio,
                lower=max(0,ratio-1.96*se), upper=min(1,ratio+1.96*se))

def main():
    cols = ["backbone","sample_id","dataset","model","w","cluster","similarity",
            "contrast","helped","harmed","unchanged","flip","policy","cell"]
    d = pd.read_csv(PROJECT/"reproduction/nn_contrast_investigation_20261020/results/cases.csv.gz",
                    usecols=cols)
    results = []
    for policy, p in d.groupby("policy"):
        for scope, s in cohorts(p):
            for subset, x in [("all",s), ("similarity_yes",s[s.similarity]),
                              ("contrast_yes",s[s.contrast]),
                              ("similarity_only",s[s.similarity & ~s.contrast]),
                              ("contrast_only",s[s.contrast & ~s.similarity]),
                              ("both",s[s.similarity & s.contrast])]:
                if len(x):
                    results.append(dict(policy=policy, scope=scope, subset=subset,
                                        judge="all", **stats(x)))
                    for judge, q in x.groupby("model"):
                        results.append(dict(policy=policy,scope=scope,subset=subset,
                                            judge=judge,**stats(q)))
    pd.DataFrame(results).to_csv(ROOT/"results/original_outcome_denominators.csv",index=False)
    rows = [json.loads(l) for l in gzip.open(ROOT/"corpus/rows.jsonl.gz","rt")]
    r = pd.DataFrame(rows)
    join = d[d.policy=="judge50_direct"][["backbone","sample_id","model","w","cell"]]
    assert not join.duplicated(["backbone","sample_id","model"]).any()
    r = r.merge(join.rename(columns={"model":"judge"}), on=["backbone","sample_id","judge"],
                how="left",validate="one_to_one")
    r["weight_available"] = r.w.notna()
    summaries=[]
    for scope, s in cohorts(r):
        for subset, x in [("all",s),("contrast_yes",s[s.contrast=="yes"]),
                          ("similarity_yes",s[s.similarity=="yes"]),
                          ("similarity_only",s[(s.similarity=="yes")&(s.contrast=="no")]),
                          ("both",s[(s.similarity=="yes")&(s.contrast=="yes")])]:
            for judge in ["all"] + sorted(x.judge.unique().tolist()):
                q=x if judge=="all" else x[x.judge==judge]
                for outcome in ["helped","harmed"]:
                    z=q[q.flip_type==outcome]
                    if not len(z): continue
                    valid=z[z.score_valid]
                    weighted=valid[valid.weight_available]
                    def wr(col):
                        return float(np.average(weighted[col].astype(float),weights=weighted.w)) if len(weighted) else None
                    summaries.append(dict(scope=scope,subset=subset,judge=judge,
                        original_outcome=outcome,rows=len(z),valid=len(valid),
                        missing=int((~z.score_valid).sum()),nonempty=int(z.reasoning.fillna("").str.strip().ne("").sum()),
                        capped=int((z.finish_reason=="length").sum()),
                        retained=int(valid.retains_original_nn.sum()),
                        reversed=int((~valid.retains_original_nn.astype(bool)).sum()),
                        retained_fraction=float(valid.retains_original_nn.mean()),
                        weighted_retained_fraction=wr("retains_original_nn"),
                        weighted_correct_fraction=wr("rationale_correct"),
                        weighted_rows=len(weighted), targets=z.cluster.nunique()))
    pd.DataFrame(summaries).to_csv(ROOT/"results/rationale_answer_retention.csv",index=False)
    r.drop(columns=["reasoning","generated_text"]).to_csv(ROOT/"corpus/metadata.csv.gz",index=False)
    receipt = dict(original_case_rows=len(d), rationale_rows=len(r),
                   rationale_with_paired_population_weight=int(r.weight_available.sum()),
                   rationale_without_paired_population_weight=int((~r.weight_available).sum()),
                   outcomes="Original probability-only control to NN flips at50; new probability only changes retention/correction columns.",
                   weight="Original inverse-inclusion weights normalized to equal mass per evaluation cell; subset percentages conditional on subset, not each-cell percentages averaged.",
                   uncertainty="Approximate target-text-cluster delta-method95% intervals for helped fraction among original flips. Conditional on frozen outcomes and labels.")
    (ROOT/"results/QUANTIFICATION.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    out=pd.DataFrame(results)
    print(out[(out.policy=="judge50_direct")&(out.judge=="all")&
              out.subset.isin(["similarity_yes","contrast_yes"])][
        ["scope","subset","rows","helped","harmed","unchanged","weighted_helped_among_flips",
         "weighted_unchanged_fraction"]].to_string(index=False))

if __name__=="__main__":
    main()
