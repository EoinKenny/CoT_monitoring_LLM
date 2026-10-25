"""Balanced descriptive variance decomposition of frozen main-results cells.
No significance tests or causal effects; no new inference or tuning.
"""
from pathlib import Path
import csv,json,hashlib,itertools,collections
import numpy as np
R=Path(__file__).resolve().parent;SRC=R.parent/'main_performance_1nn_2nn_all12_20261023/results/cells.csv'
rows=list(csv.DictReader(SRC.open()));axes=['model','backbone','dataset'];levels=[sorted({r[k] for r in rows}) for k in axes];lookup={}
for r in rows:
 key=(r['policy'],r['model'],r['backbone'],r['dataset'],r['condition']);lookup.setdefault(key,[]).append(float(r['f1'])*100)
lookup={k:float(np.mean(v)) for k,v in lookup.items()}
records=[];marginals=[];detail={}
metrics={'control_f1':('baseline',None),'one_nn_f1':('nearest',None),'two_nn_f1':('nearest2',None),'one_nn_gain':('nearest','baseline'),'two_nn_gain':('nearest2','baseline'),'second_neighbour_gain':('nearest2','nearest'),'saliency_gain':('saliency','baseline')}
for sensitivity in ['all12','without_glm']:
 ls=[([m for m in levels[0] if m!='glm'] if sensitivity=='without_glm' else levels[0]),*levels[1:]]
 for policy in ['fixed50','tuned_shared']:
  for metric,(a,b) in metrics.items():
   y=np.empty(tuple(map(len,ls)))
   for indices in itertools.product(*[range(len(x)) for x in ls]):
    vals=tuple(ls[i][k] for i,k in enumerate(indices));y[indices]=lookup[(policy,*vals,a)]-(lookup[(policy,*vals,b)] if b else 0)
   grand=float(y.mean());total=float(((y-grand)**2).sum());effects={():np.full((1,1,1),grand)};shares={}
   for size in range(1,4):
    for subset in itertools.combinations(range(3),size):
     collapse=tuple(i for i in range(3) if i not in subset)
     effect=y.mean(axis=collapse,keepdims=True) if collapse else y.copy()
     for previous,e in effects.items():
      if set(previous).issubset(subset):effect=effect-e
     effects[subset]=effect
     ss=float((np.broadcast_to(effect,y.shape)**2).sum());name=' x '.join(axes[i] for i in subset);share=100*ss/total
     shares[name]=share;records.append(dict(sensitivity=sensitivity,policy=policy,metric=metric,component=name,ss=ss,variance_share_percent=share,total_sd_pp=float(y.std()),grand_mean_percent_or_pp=grand,n_cells=int(y.size)))
   assert abs(sum(shares.values())-100)<1e-8
   recon=sum(effects.values());assert np.max(abs(recon-y))<1e-10
   for ax,factor in enumerate(axes):
    means=y.mean(axis=tuple(i for i in range(3) if i!=ax))
    for group,val in zip(ls[ax],means):marginals.append(dict(sensitivity=sensitivity,policy=policy,metric=metric,factor=factor,group=group,value=float(val),deviation=float(val-grand)))
   detail[sensitivity+'/'+policy+'/'+metric]=dict(mean=grand,sd=float(y.std()),shares=shares,cell_min=float(y.min()),cell_max=float(y.max()))
for name,rr in [('variance_components.csv',records),('marginal_means.csv',marginals)]:
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(R/'results.json').write_text(json.dumps(detail,indent=2))
(R/'METHOD.json').write_text(json.dumps(dict(source=str(SRC),sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),method='Exact balanced three-way ANOVA sum-of-squares decomposition of 600 equal-weight cell estimates; tuned metrics averaged across three shared anchors within each cell before decomposition. Factors model (judge), backbone and dataset. Two- and three-way interactions are included. No replication-based error term, p-value or causal identification.',weights='Upstream within-cell design weights retained in F1; each judge/backbone/dataset cell receives equal weight.',limitations='Descriptive variation of this fixed set, not population effect sizes; interactions include estimation noise. API masks are model-specific. Unequal numbers of factor levels and bounded F1 affect interpretation. Test data previously inspected.',new_model_calls=0),indent=2))
for key,d in detail.items():
 if key.split('/')[-1] in ['two_nn_f1','two_nn_gain','one_nn_gain']:
  print(key,'mean',round(d['mean'],3),'sd',round(d['sd'],3),{k:round(v,2) for k,v in d['shares'].items()})
for policy in ['fixed50','tuned_shared']:
 for factor in axes:
  rr=sorted([r for r in marginals if r['sensitivity']=='all12' and r['policy']==policy and r['metric']=='two_nn_gain' and r['factor']==factor],key=lambda x:x['value'],reverse=True)
  print(policy,factor,[(r['group'],round(r['value'],3)) for r in rr])
