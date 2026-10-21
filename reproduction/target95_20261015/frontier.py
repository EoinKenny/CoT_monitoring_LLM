from pathlib import Path
import json,gzip,ast,collections,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; SRC=ROOT.parent/'api_results_20261015'; OUT=ROOT/'results'
source=ROOT.parent/'two_nn_20261014/code/numerics_source.py'; tree=ast.parse(source.read_text());ns={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['curve','grid','metrics']],type_ignores=[]),str(source),'exec'),ns)
with gzip.open(SRC/'frozen/design.json.gz','rt') as f:design=json.load(f)
cases=[dict(backbone=b,**r) for b in design['samples'] for r in design['samples'][b]]
scores=np.load(SRC/'results/scores.npz');CS=['baseline','saliency','nearest']
common=np.logical_and.reduce([(scores[m]>=0).all(axis=1) for m in scores.files]);groups=collections.defaultdict(list)
for i,r in enumerate(cases):
 if common[i]:groups[r['backbone'],r['dataset'],r['split']].append(i)
betas=[0.0]+list(np.logspace(-1,1,61))+[float('inf')]
rows=[];violations=0
for b in design['samples']:
 for ds,spec in sorted(design['thresholds'][b].items()):
  ts=np.array(spec['thresholds']);vi=np.array(groups[b,ds,'val']);ti=np.array(groups[b,ds,'test'])
  vr=[cases[i] for i in vi];tr=[cases[i] for i in ti]
  wv=np.array([1/r['inclusion_probability'] for r in vr]);yv=np.array([r['true_label'] for r in vr]);pv=np.array([r['probe_probability'] for r in vr]);vp={'n':wv.sum(),'positive':(wv*yv).sum()}
  wt=np.array([1/r['inclusion_probability'] for r in tr]);yt=np.array([r['true_label'] for r in tr]);pt=np.array([r['probe_probability'] for r in tr])
  order=np.argsort(pv,kind='stable');at=np.searchsorted(pv[order],ts,side='left')
  # Identical validation decisions share a grid column. Retain the highest
  # original cutoff, exactly preserving the frozen tie-breaking rule.
  keep=np.r_[np.flatnonzero(np.diff(at)!=0),len(ts)-1];ts=ts[keep];at=at[keep]
  for m in scores.files:
   for ci,c in enumerate(CS):
    val=scores[m][vi,ci];test=scores[m][ti,ci];g=ns['grid'](vr,val,ts,vp)
    tp=g['tp'][:101];fp=g['fp'][:101];rec=tp/vp['positive'];prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
    support=[]
    for cut in range(101):
     sums=np.r_[0,np.cumsum(val[order]>=cut)];support.append(sums[-1]-sums[at])
    valid=(np.array(support)>=20)&(tp+fp>1e-10);assert valid.any()
    cache={}
    for bi,beta in enumerate(betas):
     if beta==0:obj=prec
     elif np.isinf(beta):obj=rec
     else:obj=(1+beta**2)*tp/(beta**2*vp['positive']+tp+fp)
     best=obj[valid].max();mask=valid&np.isclose(obj,best,rtol=0,atol=1e-12)
     if beta==0 or np.isinf(beta):
      secondary=rec if beta==0 else prec;mask &= np.isclose(secondary,secondary[mask].max(),rtol=0,atol=1e-12)
     choices=np.argwhere(mask);cut,k=max(choices,key=lambda a:(ts[a[1]],a[0]));cut=int(cut);k=int(k)
     # Exact held-out metric replay. The same decision may serve many betas.
     key=cut,k
     if key not in cache:
      yes=(pt>=ts[k])&(test>=cut);vyes=(pv>=ts[k])&(val>=cut)
      tm=ns['metrics'](wt[yes&(yt==1)].sum(),wt[yes&(yt==0)].sum(),wt.sum(),(wt*yt).sum())
      dominated=valid&(prec>=prec[cut,k]-1e-10)&(rec>=rec[cut,k]-1e-10)&((prec>prec[cut,k]+1e-10)|(rec>rec[cut,k]+1e-10))
      assert not dominated.any(),(b,m,ds,c,beta)
      cache[key]=dict(**{z:tm[z] for z in ['precision','recall','f1','accuracy']},referral_rate=float(wt[pt>=ts[k]].sum()/wt.sum()),validation_precision=float(prec[cut,k]),validation_recall=float(rec[cut,k]),validation_predicted_positive_n=int(vyes.sum()),test_predicted_positive_n=int(yes.sum()))
     rows.append(dict(backbone=b,model=m,dataset=ds,condition=c,beta_index=bi,beta='infinity' if np.isinf(beta) else float(beta),probe_threshold=float(ts[k]),judge_threshold=cut,**cache[key]))
  print('done',b,ds,flush=True)
df=pd.DataFrame(rows);assert len(df)==94500
df.to_csv(OUT/'frontier_cells.csv.gz',index=False)
keys=['condition','beta_index','beta'];metrics=['precision','recall','f1','accuracy','referral_rate','validation_precision','validation_recall']
macro=df.groupby(keys,as_index=False,sort=False)[metrics].mean();macro.to_csv(OUT/'frontier_macro.csv',index=False)
pair=df.groupby(['model','backbone']+keys,as_index=False,sort=False)[metrics].mean();pair.to_csv(OUT/'frontier_combinations.csv',index=False)
judge=df.groupby(['model']+keys,as_index=False,sort=False)[metrics].mean();judge.to_csv(OUT/'frontier_judges.csv',index=False)
f1old=pd.read_csv(SRC/'plots/combination_metrics.csv');new=pair[pair.beta_index==31]
assert len(new)==150
check=new.merge(f1old[f1old.policy=='joint'],on=['model','backbone','condition'],suffixes=('_supported','_original'))
summary={'rows':len(df),'beta_policies':63,'validation_pareto_checks_passed':True,'minimum_validation_predicted_positives':20,'beta1_vs_original_f1_mean_difference':float((check.f1_supported-check.f1_original).mean()),'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'frozen/design.json.gz',SRC/'results/scores.npz',ROOT/'FRONTIER_PROTOCOL.json',Path(__file__)]}}
(OUT/'FRONTIER_COMPLETED.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
