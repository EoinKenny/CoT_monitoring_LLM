from pathlib import Path
import json,time
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
R=Path(__file__).resolve().parent;O=R/'results';d=pd.read_csv(O/'case_mechanisms.csv.gz');f=pd.read_csv(O/'joined_features.csv.gz')
# Stronger baseline discrimination subset is a sensitivity, never a replacement for the primary grid.
strong=['qwen','gemma','qwen122','claude4.8-opus','claude4.5-sonnet','gpt-astra','gpt-4o']
rows=[]
for cohort,v in [('all10_common',d[d.scope=='common']),('seven_common',d[(d.scope=='common')&d.model.isin(strong)]),('six_local_complete',d[d.scope=='local_complete'])]:
 v=v.copy();v['cell']=v.model+':'+v.backbone+':'+v.dataset
 for axis in ['overall','label_aligned','label_prediction_agree','cosine_bin','target_length_bin','neighbor_length_bin','true_label']:
  groups=[('all',v)] if axis=='overall' else list(v.groupby(axis))
  for level,g in groups:
   g=g.copy();nc=g.cell.nunique();g['a']=g.weight/g.groupby('cell').weight.transform('sum')/nc
   for outcome in ['toward_neighbor','toward_gold','brier_gain','correctness_gain']:
    mu=float((g.a*g[outcome]).sum());means=g.groupby('cell').apply(lambda x:np.average(x[outcome],weights=x.weight));res=g[outcome]-g.cell.map(means)
    influences=(g.a*res).groupby(g.cluster).sum();G=len(influences);se=float(np.sqrt((influences**2).sum()*G/(G-1)))
    rows.append(dict(cohort=cohort,axis=axis,level=level,outcome=outcome,mean=mu,lo=mu-1.96*se,hi=mu+1.96*se,n=len(g),clusters=G,cells=nc))
 # Within-label-alignment cosine patterns avoid explaining label composition by similarity.
 for (aligned,bin_),g in v.groupby(['label_aligned','cosine_bin']):
  for outcome in ['toward_neighbor','brier_gain','correctness_gain']:
   means=g.groupby('cell').apply(lambda x:np.average(x[outcome],weights=x.weight))
   rows.append(dict(cohort=cohort,axis='alignment_cosine',level=f'{aligned}:{bin_}',outcome=outcome,mean=float(means.mean()),n=len(g),clusters=g.cluster.nunique(),cells=len(means)))
pd.DataFrame(rows).to_csv(O/'factor_cluster_intervals.csv',index=False)
print('factor intervals complete',flush=True)
reg=[]
for cohort,v in [('all10_common',d[d.scope=='common']),('seven_common',d[(d.scope=='common')&d.model.isin(strong)])]:
 v=v.copy();v['log_target']=np.log1p(v.target_chars);v['log_neighbor']=np.log1p(v.neighbor_chars)
 for y in ['brier_gain','toward_neighbor']:
  formula=f'{y} ~ label_aligned * cosine_rank + log_target + log_neighbor + lexical_jaccard + label_prediction_agree + true_label + base_error + probe_probability + C(model) + C(backbone) + C(dataset)'
  fit=smf.wls(formula,v,weights=v.macro_weight).fit(cov_type='cluster',cov_kwds={'groups':v.cluster})
  ci=fit.conf_int()
  for name in fit.params.index:
   reg.append(dict(cohort=cohort,outcome=y,term=name,coef=fit.params[name],se=fit.bse[name],lo=ci.loc[name,0],hi=ci.loc[name,1],p=fit.pvalues[name],n=len(v),clusters=v.cluster.nunique(),r_squared=fit.rsquared))
  print('regression',cohort,y,fit.rsquared,flush=True)
pd.DataFrame(reg).to_csv(O/'conditional_associations.csv',index=False)
# Paired Poisson cluster bootstrap, 2000 replicates. One weight per target text across judges/backbones.
# Recompute each cell's precision/recall/F1 from weighted confusion counts, then average cells.
v=d[d.scope=='common'].copy();v['cell']=v.model+':'+v.backbone+':'+v.dataset
clusters,uid=np.unique(v.cluster,return_inverse=True);cells,cid=np.unique(v.cell,return_inverse=True);ncell=len(cells);N=len(clusters)
y=v.true_label.to_numpy();w=v.weight.to_numpy();preds={k:v['decision_'+k].to_numpy() for k in ['BB','NB','BN','NN']}
base={k:np.stack([w*y*p,w*(1-y)*p,w*y],axis=1) for k,p in preds.items()}
model=np.array([x.split(':')[0] for x in cells]);dataset=np.array([x.split(':')[2] for x in cells]);bb=np.array([x.split(':')[1] for x in cells]);groups={'overall':np.ones(ncell,bool)}
for key,arr in [('judge',model),('dataset',dataset),('backbone',bb)]:
 for x in np.unique(arr):groups[key+':'+x]=arr==x
B=2000;rng=np.random.default_rng(20261015);out={k:[] for k in groups}
def calc(mult):
 result={}
 for name,mat in base.items():
  z=[np.bincount(cid,weights=mat[:,j]*mult,minlength=ncell) for j in range(3)];tp,fp,pos=z
  result[name]=np.stack([np.divide(2*tp,pos+tp+fp,out=np.zeros(ncell),where=pos+tp+fp>0),np.divide(tp,tp+fp,out=np.zeros(ncell),where=tp+fp>0),np.divide(tp,pos,out=np.zeros(ncell),where=pos>0)],axis=1)
 return result
point=calc(np.ones(len(v)))
for b in range(B):
 z=calc(rng.poisson(1,N)[uid])
 for group,mask in groups.items():
  a={k:t[mask].mean(axis=0) for k,t in z.items()}
  out[group].append(np.stack([a['NN']-a['BB'],a['NB']-a['BB'],.5*(a['NB']-a['BB']+a['NN']-a['BN']),.5*(a['BN']-a['BB']+a['NN']-a['NB'])]))
 if b%500==0:print('bootstrap',b,flush=True)
rows=[]
for group,mask in groups.items():
 a={k:t[mask].mean(axis=0) for k,t in point.items()};p=np.stack([a['NN']-a['BB'],a['NB']-a['BB'],.5*(a['NB']-a['BB']+a['NN']-a['BN']),.5*(a['BN']-a['BB']+a['NN']-a['NB'])]);arr=np.array(out[group])
 for i,effect in enumerate(['own_policy_gain','same_baseline_policy_gain','shapley_score','shapley_policy']):
  for j,metric in enumerate(['f1','precision','recall']):
   lo,hi=np.quantile(arr[:,i,j],[.025,.975]);rows.append(dict(group=group,effect=effect,metric=metric,mean=p[i,j],lo=lo,hi=hi,replicates=B,clusters=N,cells=int(mask.sum())))
pd.DataFrame(rows).to_csv(O/'paired_cluster_bootstrap.csv',index=False)
(O/'CONDITIONAL_COMPLETED.json').write_text(json.dumps(dict(time=time.time(),bootstrap_replicates=B,seed=20261015,n_clusters=N,n_observations=len(v),note='Intervals conditional on frozen validation-selected thresholds; no retraining or threshold reselection. Dataset-level correlations are exploratory.'),indent=2))
