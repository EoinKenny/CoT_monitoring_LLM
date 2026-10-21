from pathlib import Path
import pandas as pd,numpy as np,json,gzip
R=Path(__file__).resolve().parent;O=R/'results';f=pd.read_csv(O/'joined_features.csv.gz');d=pd.read_csv(O/'case_mechanisms.csv.gz');d=d[d.scope=='common'].copy();d['case_id']=d.dataset+':'+d.cluster
# Match all five backbones, equal weight per shared underlying target; this is not a population estimator.
counts=d.groupby(['model','case_id']).backbone.nunique();keys=counts[counts==5].index
z=d.set_index(['model','case_id']).loc[keys].reset_index();assert (z.groupby(['model','case_id']).true_label.nunique()==1).all()
rows=[]
for (m,bb,ds),g in z.groupby(['model','backbone','dataset']):
 rows.append(dict(model=m,backbone=bb,dataset=ds,n=len(g),label_alignment=g.label_aligned.mean(),base_accuracy=np.mean((g.base_score>=.5)==g.true_label),nn_accuracy=np.mean((g.nn_score>=.5)==g.true_label),brier_gain=g.brier_gain.mean(),toward_neighbor=g.toward_neighbor.mean(),cosine=g.nearest_cosine.mean()))
pd.DataFrame(rows).to_csv(O/'matched_backbone_cells.csv',index=False)
pd.DataFrame(rows).groupby('backbone').mean(numeric_only=True).to_csv(O/'matched_backbone_summary.csv')
var=z.groupby(['model','case_id']).agg(base_range=('base_score',lambda x:x.max()-x.min()),nn_range=('nn_score',lambda x:x.max()-x.min()),neighbor_ids=('nearest_id','nunique'),neighbor_labels=('nearest_label','nunique'))
var.groupby(level=0).mean().to_csv(O/'matched_backbone_variation.csv');var.to_csv(O/'matched_target_variation.csv.gz')
# Per-class retrieval quality, to avoid treating majority-class matching as an informative neighbour.
rr=[]
for (bb,ds),g in f[(f.split=='test')&f.common_valid].groupby(['backbone','dataset']):
 rates={}
 for y,sg in g.groupby('true_label'):rates[y]=np.average(sg.label_aligned,weights=sg.weight)
 rr.append(dict(backbone=bb,dataset=ds,negative_alignment=rates.get(0,np.nan),positive_alignment=rates.get(1,np.nan),balanced_alignment=np.mean(list(rates.values()))))
pd.DataFrame(rr).to_csv(O/'retrieval_class_balance.csv',index=False)
print('all-five-matched targets',z[['dataset','cluster']].drop_duplicates().shape[0]);print(pd.DataFrame(rows).groupby('backbone')[['n','label_alignment','base_accuracy','nn_accuracy','brier_gain']].mean().round(4))
