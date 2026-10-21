from pathlib import Path
import json,gzip,hashlib
import numpy as np,pandas as pd
from statsmodels.regression.linear_model import WLS
R=Path(__file__).resolve().parent;M=R.parent/'nn_mechanisms_20261015';A=R.parent/'api_results_20261015'
O=R/'results';O.mkdir(exist_ok=True)
f=pd.read_csv(M/'results/joined_features.csv.gz');scores=np.load(A/'results/scores.npz');models=scores.files
D=json.load(gzip.open(A/'frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b in D['samples'] for r in D['samples'][b]]
assert list(zip(f.backbone,f.sample_id))==[(r['backbone'],r['sample_id']) for r in cases]
assert np.array_equal(f.true_label,[r['true_label'] for r in cases])
manifest=json.load(open(M/'EXPORT_COMPLETED.json'));assert hashlib.sha256((M/'features.jsonl.gz').read_bytes()).hexdigest()==manifest['files']['features.jsonl.gz']
valid=np.logical_and.reduce([(scores[m]>=0).all(1) for m in models]);f['gate']=[D['thresholds'][r.backbone][r.dataset]['original_threshold'] for r in f.itertuples()]
f['flag']=f.probe_probability>=f.gate;f['weight']=1/f.inclusion_probability
assert f.gate.between(0,1,inclusive='neither').all()
logit=lambda a:np.log(np.clip(a,1e-12,1-1e-12)/(1-np.clip(a,1e-12,1-1e-12)))
f['probe_gap']=abs(logit(f.probe_probability.to_numpy())-logit(f.gate.to_numpy()))
f['probe_saturated']=f.probe_probability.le(1e-12)|f.probe_probability.ge(1-1e-12)
def wrank(x,w):
 z=pd.DataFrame({'x':np.asarray(x),'w':np.asarray(w)});g=z.groupby('x',sort=True).w.sum();v=(g.cumsum()-g/2)/g.sum();return z.x.map(v).to_numpy()
def wquant(x,w,q):
 ix=np.argsort(x);x=np.asarray(x)[ix];w=np.asarray(w)[ix];return np.interp(q,(w.cumsum()-w/2)/w.sum(),x)
base=f[valid & f.split.eq('test') & f.flag].copy()
for _,ix in base.groupby(['backbone','dataset']).groups.items():
 z=base.loc[ix];base.loc[ix,'probe_rank']=wrank(z.probe_gap,z.weight);base.loc[ix,'cosine_rank_local']=wrank(z.nearest_cosine,z.weight)
rows=[]
for m in models:
 z=base.copy();ix=z.index.to_numpy();z['model']=m;z['control_score']=scores[m][ix,0]/100;z['nn_score']=scores[m][ix,2]/100;rows.append(z)
d=pd.concat(rows,ignore_index=True);d['base_correct']=(d.control_score>=.5)==d.true_label;d['nn_correct']=(d.nn_score>=.5)==d.true_label
d['category']=np.select([d.base_correct&d.nn_correct,d.base_correct&~d.nn_correct,~d.base_correct&d.nn_correct],['correct_to_correct','correct_to_wrong','wrong_to_correct'],default='wrong_to_wrong')
d['flip']=d.base_correct!=d.nn_correct;d['flag_type']=np.where(d.true_label==1,'TP','FP');d['judge_margin']=abs(d.control_score-.5)*100;d['judge_near']=d.judge_margin<=10;d['judge_extreme']=d.judge_margin>=40;d['aligned']=(d.nearest_label==d.true_label).astype(int)
d['cell']=d.model+'|'+d.backbone+'|'+d.dataset;d['macro_weight']=d.weight/d.groupby('cell').weight.transform('sum');d['cluster']=d.dataset+':'+d.target_content_hash
d.to_csv(O/'cases.csv.gz',index=False)
metrics=['nearest_cosine','cosine_rank_local','probe_gap','probe_rank','judge_margin','judge_near','judge_extreme','aligned']
summary=[]
for keys,g in d.groupby(['flag_type','category']):
 row=dict(flag_type=keys[0],category=keys[1],n=len(g),unique_targets=g.cluster.nunique(),mass=float(g.macro_weight.sum()))
 for c in metrics:
  row[c+'_mean']=np.average(g[c],weights=g.macro_weight)
  for q,val in zip([10,25,50,75,90],wquant(g[c],g.macro_weight,np.array([.1,.25,.5,.75,.9]))):row[c+'_q'+str(q)]=val
 summary.append(row)
pd.DataFrame(summary).to_csv(O/'category_summary.csv',index=False)
# Flip-versus-stable contrasts within identical model, backbone, dataset, gold class, and prior correctness.
contrasts=[]
for keys,g in d.groupby(['model','backbone','dataset','flag_type','base_correct']):
 if g.flip.nunique()<2:continue
 row=dict(zip(['model','backbone','dataset','flag_type','base_correct'],keys));row['flip_n']=int(g.flip.sum());row['stable_n']=int((~g.flip).sum())
 for c in metrics:
  row[c+'_flip']=np.average(g.loc[g.flip,c],weights=g.loc[g.flip,'weight']);row[c+'_stable']=np.average(g.loc[~g.flip,c],weights=g.loc[~g.flip,'weight']);row[c+'_difference']=row[c+'_flip']-row[c+'_stable']
 contrasts.append(row)
cd=pd.DataFrame(contrasts);cd.to_csv(O/'matched_contrasts.csv',index=False)
for key in ['model','dataset','flag_type','base_correct']:
 cd.groupby(key).mean(numeric_only=True).to_csv(O/('contrasts_by_'+key+'.csv'))
# Cell / prior correctness fixed effects; covariance clustered by target text across backbones and judges.
d['cue_opposes_control']=(d.nearest_label!=(d.control_score>=.5).astype(int)).astype(int)
predictors=['cosine_rank_local','probe_rank','judge_margin','cue_opposes_control'];regs=[]
for scope,z in [('all10',d),('stronger7',d[~d.model.isin(['glm','mistral','olmo'])])]:
 for facet in ['all','TP','FP']:
  g=z.copy() if facet=='all' else z[z.flag_type.eq(facet)].copy()
  g['stratum']=g.cell+'|'+g.flag_type+'|'+g.base_correct.astype(str)
  cols=['flip']+predictors;X=g[cols].astype(float).copy();X['judge_margin']/=50
  w=g.macro_weight.to_numpy();weighted=X.mul(w,axis=0);means=weighted.groupby(g.stratum).transform('sum').div(pd.Series(w,index=g.index).groupby(g.stratum).transform('sum'),axis=0);v=X-means
  for spec,px in [('joint',predictors)]+[('single_'+c,[c]) for c in predictors]:
   fit=WLS(v.flip.to_numpy(),v[px].to_numpy(),weights=w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(g.cluster)[0]})
   ci=fit.conf_int()
   for j,c in enumerate(px):regs.append(dict(scope=scope,facet=facet,spec=spec,predictor=c,coefficient=fit.params[j],ci_low=ci[j,0],ci_high=ci[j,1],n=len(g),clusters=g.cluster.nunique()))
pd.DataFrame(regs).to_csv(O/'conditional_associations.csv',index=False)
# Per-judge flipped/stable summary, and full distributions available for plotting.
ss=[]
for (m,fl),g in d.groupby(['model','flip']):
 rr=dict(model=m,flip=bool(fl),n=len(g));rr.update({c:np.average(g[c],weights=g.macro_weight) for c in metrics});ss.append(rr)
pd.DataFrame(ss).to_csv(O/'by_judge.csv',index=False)
# Sensitivity: hold each control policy's validation-tuned gate and cutoff fixed for both prompt conditions.
pol=pd.read_csv(A/'results/detailed_metrics.csv');pol=pol[(pol.scope=='common_valid_all_judges')&(pol.policy=='joint')&(pol.condition=='baseline')]
tuned=[]
for r in pol.itertuples():
 z=f[valid&f.split.eq('test')&f.backbone.eq(r.backbone)&f.dataset.eq(r.dataset)&f.probe_probability.ge(r.probe_threshold)]
 if len(z)==0:continue
 ix=z.index.to_numpy();b=scores[r.model][ix,0];n=scores[r.model][ix,2];fl=(b>=r.probability_threshold)!=(n>=r.probability_threshold)
 tuned.append(dict(model=r.model,backbone=r.backbone,dataset=r.dataset,cutoff=r.probability_threshold,n=len(z),flip_rate=np.average(fl,weights=z.weight),bypass=r.probability_threshold==0))
pd.DataFrame(tuned).to_csv(O/'tuned_policy_sensitivity.csv',index=False)
meta=dict(judges=list(models),test_backbone_cases=int((valid&f.split.eq('test')).sum()),flagged_backbone_cases=len(base),unique_flagged_target_texts=base.cluster.nunique(),judge_case_observations=len(d),cell_count=d.cell.nunique(),probe_saturated_cases=int(base.probe_saturated.sum()),flip_observations=int(d.flip.sum()),weighted_flip_fraction=float(np.average(d.flip,weights=d.macro_weight)),weighted_accuracy_gain=float(np.average(d.nn_correct.astype(int)-d.base_correct.astype(int),weights=d.macro_weight)),matched_strata=len(cd),sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A/'results/scores.npz',A/'frozen/design.json.gz',M/'results/joined_features.csv.gz']},method='Common-valid test subset, frozen original validation F1 probe gate; common 50 judge cutoff; inverse-inclusion weights, equal mass per cell. Probe logit gaps clipped at 1e-12; within-probe rank handles differing coefficient norms; LLM output margin is not geometric distance. Cluster-robust WLS uses within-cell/gold/prior-correctness centered variables.')
(O/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k not in ['sources','method']},indent=2));print(pd.DataFrame(summary)[['flag_type','category','n','nearest_cosine_mean','probe_rank_mean','judge_margin_mean']].to_string(index=False));print(cd[[c+'_difference' for c in metrics]].mean().to_string());print(pd.DataFrame(regs).query("scope=='all10' and facet=='all' and spec=='joint'").to_string(index=False))
