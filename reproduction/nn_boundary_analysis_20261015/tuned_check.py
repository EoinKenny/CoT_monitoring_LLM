from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;A=R.parent/'api_results_20261015';f=pd.read_csv(R.parent/'nn_mechanisms_20261015/results/joined_features.csv.gz');s=np.load(A/'results/scores.npz');valid=np.logical_and.reduce([(s[m]>=0).all(1) for m in s.files]);pol=pd.read_csv(A/'results/detailed_metrics.csv');pol=pol[(pol.scope=='common_valid_all_judges')&(pol.policy=='joint')&(pol.condition=='baseline')];out=[]
def rank(x,w):
 z=pd.DataFrame({'x':np.asarray(x),'w':np.asarray(w)});g=z.groupby('x').w.sum();return z.x.map((g.cumsum()-g/2)/g.sum()).to_numpy()
for r in pol.itertuples():
 if r.probability_threshold==0:continue
 z=f[valid&f.split.eq('test')&f.backbone.eq(r.backbone)&f.dataset.eq(r.dataset)&f.probe_probability.ge(r.probe_threshold)].copy()
 if len(z)==0:continue
 ix=z.index.to_numpy();z['weight']=1/z.inclusion_probability;b=s[r.model][ix,0];n=s[r.model][ix,2];z['flip']=(b>=r.probability_threshold)!=(n>=r.probability_threshold);z['base_correct']=(b>=r.probability_threshold)==z.true_label;z['probe_rank']=rank(z.probe_probability,z.weight);z['cosine_rank']=rank(z.nearest_cosine,z.weight);z['judge_margin']=abs(b-r.probability_threshold)
 for (y,bc),g in z.groupby(['true_label','base_correct']):
  if g.flip.nunique()<2:continue
  row=dict(model=r.model,backbone=r.backbone,dataset=r.dataset,gold=y,base_correct=bc,n=len(g),cutoff=r.probability_threshold)
  for k in ['nearest_cosine','cosine_rank','probe_rank','judge_margin']:
   row[k+'_difference']=np.average(g.loc[g.flip,k],weights=g.loc[g.flip,'weight'])-np.average(g.loc[~g.flip,k],weights=g.loc[~g.flip,'weight'])
  out.append(row)
d=pd.DataFrame(out);d.to_csv(R/'results/tuned_matched_contrasts.csv',index=False);print(d.groupby('model')[[k+'_difference' for k in ['nearest_cosine','probe_rank','judge_margin']]].mean().to_string());print('Overall',len(d),d.filter(like='_difference').mean().to_dict())
