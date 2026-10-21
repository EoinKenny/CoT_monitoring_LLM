from pathlib import Path
import json,gzip,collections,ast,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; SRC=ROOT.parent/'api_results_20261015'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PROTOCOL=json.loads((ROOT/'PROTOCOL.json').read_text())
source=ROOT.parent/'two_nn_20261014/code/numerics_source.py'
tree=ast.parse(source.read_text());ns={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['curve','grid','metrics']],type_ignores=[]),str(source),'exec'),ns)
with gzip.open(SRC/'frozen/design.json.gz','rt') as f:design=json.load(f)
cases=[dict(backbone=b,**r) for b in design['samples'] for r in design['samples'][b]]
scores=np.load(SRC/'results/scores.npz');CS=['baseline','saliency','nearest']
common=np.logical_and.reduce([(scores[m]>=0).all(axis=1) for m in scores.files])
groups=collections.defaultdict(list)
for i,r in enumerate(cases):
 if common[i]:groups[r['backbone'],r['dataset'],r['split']].append(i)
def choose(g,ts,target,support,min_support):
 tp,fp=g['tp'],g['fp'];rec=tp/g['positive'];prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
 objective=prec if target=='recall95' else rec
 valid=(rec>=.95-1e-12) if target=='recall95' else ((prec>=.95-1e-12)&(support>=min_support))
 valid &= tp+fp>1e-10
 if not valid.any():return None
 best=objective[valid].max();ix=np.argwhere(valid&np.isclose(objective,best,rtol=0,atol=1e-12))
 cut,k=max(ix,key=lambda a:(ts[a[1]],a[0]));return int(cut),int(k)
results=[]
for b in design['samples']:
 for ds,spec in sorted(design['thresholds'][b].items()):
  ts=np.array(spec['thresholds']);vi=np.array(groups[b,ds,'val']);ti=np.array(groups[b,ds,'test'])
  vr=[cases[i] for i in vi];tr=[cases[i] for i in ti]
  wv=np.array([1/r['inclusion_probability'] for r in vr]);yv=np.array([r['true_label'] for r in vr]);pv=np.array([r['probe_probability'] for r in vr])
  wt=np.array([1/r['inclusion_probability'] for r in tr]);yt=np.array([r['true_label'] for r in tr]);pt=np.array([r['probe_probability'] for r in tr])
  pop={'n':wv.sum(),'positive':(wv*yv).sum()}
  for m in scores.files:
   for ci,c in enumerate(CS):
    val=scores[m][vi,ci];test=scores[m][ti,ci]
    g=ns['grid'](vr,val,ts,pop);g={k:v[:101] for k,v in g.items()};g['positive']=pop['positive']
    # Raw validation-positive counts for the declared minimum-support rule.
    ix=np.argsort(pv,kind='stable');at=np.searchsorted(pv[ix],ts,side='left')
    support=[]
    for cut in range(101):
     sums=np.r_[0,np.cumsum(val[ix]>=cut)];support.append(sums[-1]-sums[at])
    support=np.array(support)
    for target,minimum in [('recall95',0),('precision95',20),('precision95',1)]:
     choice=choose(g,ts,target,support,minimum)
     base=dict(backbone=b,model=m,dataset=ds,condition=c,target=target,min_support=minimum,validation_n=len(vi),test_n=len(ti),feasible=choice is not None)
     if choice is None:results.append(base);continue
     cut,k=choice;vy=(pv>=ts[k])&(val>=cut);yes=(pt>=ts[k])&(test>=cut)
     vm=ns['metrics'](wv[vy&(yv==1)].sum(),wv[vy&(yv==0)].sum(),wv.sum(),(wv*yv).sum())
     tm=ns['metrics'](wt[yes&(yt==1)].sum(),wt[yes&(yt==0)].sum(),wt.sum(),(wt*yt).sum())
     assert abs(g['tp'][cut,k]-vm['tp'])<1e-7
     assert vm['recall' if target=='recall95' else 'precision']>=.95-1e-12
     results.append(base|dict(probe_threshold=float(ts[k]),judge_threshold=cut,validation_predicted_positive_n=int(vy.sum()),test_predicted_positive_n=int(yes.sum()),test_positive_n=int(yt.sum()),validation_positive_n=int(yv.sum()),**{'validation_'+z:vm[z] for z in ['precision','recall','f1']},**{z:tm[z] for z in ['precision','recall','f1','accuracy']},test_target_met=bool(tm['recall' if target=='recall95' else 'precision']>=.95-1e-12),referral_rate=float(wt[pt>=ts[k]].sum()/wt.sum())))
  print('done',b,ds,flush=True)
df=pd.DataFrame(results);df.to_csv(OUT/'all_cells.csv',index=False)
agg=[];coverage=[]
for (target,minimum,b,m),gg in df.groupby(['target','min_support','backbone','model']):
 eligible=gg.groupby('dataset').feasible.all();datasets=list(eligible[eligible].index)
 coverage.append(dict(target=target,min_support=int(minimum),backbone=b,model=m,matched_datasets=len(datasets),excluded_datasets=';'.join(eligible[~eligible].index)))
 for c in CS:
  rr=gg[(gg.condition==c)&gg.dataset.isin(datasets)]
  if not len(rr):continue
  out=dict(target=target,min_support=int(minimum),backbone=b,model=m,condition=c,datasets=len(rr))
  out.update({k:float(rr[k].mean()) for k in ['precision','recall','f1','accuracy','referral_rate']})
  out['test_target_met_cells']=int(rr.test_target_met.sum());agg.append(out)
pd.DataFrame(agg).to_csv(OUT/'combinations.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
summary=[]
for (target,minimum,c),rr in pd.DataFrame(agg).groupby(['target','min_support','condition']):
 cells=df[(df.target==target)&(df.min_support==minimum)&(df.condition==c)];ok=cells[cells.feasible]
 summary.append(dict(target=target,min_support=int(minimum),condition=c,combinations=len(rr),matched_dataset_cells=int(rr.datasets.sum()),validation_feasible=int(cells.feasible.sum()),validation_total=len(cells),test_target_met=int(ok.test_target_met.sum()),test_target_met_denominator=len(ok),**{k:float(rr[k].mean()) for k in ['precision','recall','f1','accuracy','referral_rate']}))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
manifest={'protocol':PROTOCOL,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'frozen/design.json.gz',SRC/'results/scores.npz',ROOT/'PROTOCOL.json']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'validation_cases':int(sum(common[i] and r['split']=='val' for i,r in enumerate(cases))),'test_cases':int(sum(common[i] and r['split']=='test' for i,r in enumerate(cases))),'rows':len(df),'inference_calls':0}
(OUT/'COMPLETED.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(summary,indent=2))
