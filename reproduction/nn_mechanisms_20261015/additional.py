from pathlib import Path
import json,numpy as np,pandas as pd
from statsmodels.regression.linear_model import WLS
R=Path(__file__).resolve().parent;O=R/'results';f=pd.read_csv(O/'joined_features.csv.gz');d=pd.read_csv(O/'case_mechanisms.csv.gz');d=d[d.scope=='common'].copy()
s=[]
for (b,ds,split),g in f[f.common_valid].groupby(['backbone','dataset','split']):
 w=g.weight.to_numpy();s.append(dict(backbone=b,dataset=ds,split=split,n=len(g),positive_n=int(g.true_label.sum()),negative_n=int((1-g.true_label).sum()),ess=w.sum()**2/(w*w).sum(),max_normalized_weight=w.max()/w.sum()))
pd.DataFrame(s).to_csv(O/'sample_support.csv',index=False)
rows=[]
for (m,ds),g in d.groupby(['model','dataset']):
 row=dict(model=m,dataset=ds)
 for y,z in g.groupby('true_label'):
  for col in ['base_score','nn_score']:
   row[f'{col}_gold{y}']=np.mean([np.average(t[col],weights=t.weight) for _,t in z.groupby('backbone')])
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'score_direction_by_class.csv',index=False)
# Within-target, within-judge comparison over overlapping backbone samples. No population claim.
d['unit']=d.model+':'+d.cluster;counts=d.groupby('unit').backbone.transform('nunique');v=d[counts>=2].copy();v['n_unit']=v.groupby('unit').model.transform('size')
cols=['brier_gain','label_aligned','base_error','nearest_cosine','label_prediction_agree']
x=v[cols]-v.groupby('unit')[cols].transform('mean');model=WLS(x.brier_gain,x.drop(columns='brier_gain'),weights=1/v.n_unit).fit(cov_type='cluster',cov_kwds={'groups':v.cluster});ci=model.conf_int()
out=[dict(term=k,coef=model.params[k],lo=ci.loc[k,0],hi=ci.loc[k,1],p=model.pvalues[k],n_observations=len(v),unique_targets=v.cluster.nunique(),judge_target_groups=v.unit.nunique()) for k in model.params.index]
pd.DataFrame(out).to_csv(O/'within_target_associations.csv',index=False)
# Deterministic audit checks for key mechanism summaries and class-specific effects.
cells=pd.read_csv(O/'cell_mechanisms.csv');common=cells[cells.scope=='common']
assert len(common)==500 and len(cells[cells.scope=='local_complete'])==300
assert np.allclose(common.f1_gain,common.f1_score_effect+common.f1_policy_effect,atol=1e-12)
assert (f.exact_text_match==False).all()
print('within-target',pd.DataFrame(out).round(4).to_string(index=False))
print('support',pd.DataFrame(s).groupby(['dataset','split'])[['n','ess']].mean().round(1).to_string())
