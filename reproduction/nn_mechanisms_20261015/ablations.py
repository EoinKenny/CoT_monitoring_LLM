from pathlib import Path
import json,gzip,hashlib
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;O=R/'results'
assert hashlib.sha256((R/'ablation_cases.jsonl.gz').read_bytes()).hexdigest()==json.load(open(R/'ABLATION_EXPORT.json'))['sha256']
f=pd.read_csv(O/'joined_features.csv.gz').set_index(['backbone','sample_id']);out=[];case=[]
raw=[json.loads(x) for x in gzip.open(R/'ablation_cases.jsonl.gz','rt')]
for r in raw:
 if r['split']!='test':continue
 a=f.loc[r['backbone'],r['sample_id']];assert int(a.true_label)==r['true_label'];y=r['true_label'];w=1/r['inclusion_probability'];b=r['scores']['baseline']/100
 for c,p in r['scores'].items():
  p/=100;label=r['random_label'] if c=='random_full' else int(a.nearest_label)
  case.append(dict(model=r['model'],backbone=r['backbone'],dataset=r['dataset'],sample_id=r['sample_id'],cluster=a.cluster,condition=c,true_label=y,weight=w,score=p,base_score=b,correctness_gain=int((p>=.5)==y)-int((b>=.5)==y),brier_gain=(b-y)**2-(p-y)**2,toward_neighbor=(p-b)*(2*label-1),label_aligned=int(label==y),label=label,nearest_label=int(a.nearest_label),nearest_prediction=int(a.nearest_prediction),nearest_cosine=float(a.nearest_cosine)))
d=pd.DataFrame(case);d.to_csv(O/'ablation_case_effects.csv.gz',index=False)
for (m,bb,ds,c),g in d.groupby(['model','backbone','dataset','condition']):
 y=g.true_label.to_numpy();w=g.weight.to_numpy();s=g.score.to_numpy();b=g.base_score.to_numpy()
 row=dict(model=m,backbone=bb,dataset=ds,condition=c,n=len(g),auc=roc_auc_score(y,s,sample_weight=w),base_auc=roc_auc_score(y,b,sample_weight=w))
 for z in ['correctness_gain','brier_gain','toward_neighbor','label_aligned']:row[z]=np.average(g[z],weights=w)
 out.append(row)
pd.DataFrame(out).to_csv(O/'ablation_cells.csv',index=False)
pd.DataFrame(out).groupby('condition').mean(numeric_only=True).to_csv(O/'ablation_summary.csv')
rows=[]
for (m,bb,ds,c,align),g in d.groupby(['model','backbone','dataset','condition','label_aligned']):
 row=dict(model=m,backbone=bb,dataset=ds,condition=c,alignment=align,n=len(g))
 for z in ['correctness_gain','brier_gain','toward_neighbor']:row[z]=np.average(g[z],weights=g.weight)
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'ablation_alignment_cells.csv',index=False)
pd.DataFrame(rows).groupby(['condition','alignment']).mean(numeric_only=True).to_csv(O/'ablation_alignment_summary.csv')
print(pd.DataFrame(out).groupby('condition')[['correctness_gain','brier_gain','toward_neighbor']].mean().round(4).to_string())
