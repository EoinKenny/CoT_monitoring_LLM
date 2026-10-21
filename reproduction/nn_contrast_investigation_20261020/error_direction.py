"""Separate false positives from false negatives in observed contrast outcomes."""
from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parent;O=R/'results';T=['aegis','beavertails','toxicchat','toxigen'];cols=['policy','model','dataset','cluster','w','true_label','nearest_label','helped','harmed','toward','away','majority_important_contrast']
d=pd.read_csv(O/'cases.csv.gz',usecols=cols)
rows=[]
for policy,z in d.groupby('policy'):
 for cohort,h in [('all10',z),('toxicity4',z[z.dataset.isin(T)])]:
  for c,v in h.groupby('majority_important_contrast'):
   for outcome in ['helped','harmed']:
    g=v[v[outcome]];den=g.w.sum()
    for gold in [0,1]:
     for direction in ['toward','away']:
      indicator=g.true_label.eq(gold)&g[direction];mu=np.average(indicator,weights=g.w) if len(g) else np.nan
      u=pd.Series(g.w.to_numpy()*(indicator.to_numpy()-mu)/den).groupby(g.cluster.to_numpy()).sum().to_numpy() if den else np.array([]);se=np.sqrt(u@u*len(u)/max(1,len(u)-1)) if len(u) else np.nan
      rows.append(dict(policy=policy,cohort=cohort,contrast=c,outcome=outcome,true_label=gold,error_type='false_positive' if gold==0 else 'false_negative',direction=direction,n=len(g),event_n=int(indicator.sum()),fraction=mu,lo=mu-1.96*se,hi=mu+1.96*se))
pd.DataFrame(rows).to_csv(O/'error_direction.csv',index=False)
print(pd.DataFrame(rows).query("cohort=='toxicity4' and contrast=='yes'").round(4).to_string(index=False))
