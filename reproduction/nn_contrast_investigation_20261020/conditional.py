"""Opportunity-conditioned contrasts and descriptive boundary associations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,json
R=Path(__file__).resolve().parent;O=R/'results'
use=['policy','model','backbone','dataset','unit_id','cluster','cell','w','weight','true_label','nearest_label','nearest_prediction','probe_probability','probe_margin_rank','probe_saturated','cosine_rank','nearest_cosine','control_score','nearest_score','cue_agrees_control','aligned','label_prediction_agree','gain','helped','harmed','flip','away_flip','toward_flip','away','toward','attraction','similarity','contrast','judge_margin','probe_bin','cosine_bin5','judge_margin_bin']+[a+'_'+l for a in ['majority','qwen122','gemma','qwen'] for l in ['important_contrast','labelled_similarity']]
d=pd.concat([g[g.policy.eq('judge50_direct')] for g in pd.read_csv(O/'cases.csv.gz',usecols=use,chunksize=100000)],ignore_index=True)
T=['aegis','beavertails','toxicchat','toxigen']
def ci(g,x):
 w=g.w.to_numpy();x=np.asarray(x,float);mu=np.average(x,weights=w)
 v=pd.Series(w*(x-mu)/w.sum()).groupby(g.cluster.to_numpy()).sum().to_numpy()
 se=np.sqrt(v@v*len(v)/max(1,len(v)-1));return mu,mu-1.96*se,mu+1.96*se
def fit(g,ann,same_target=False):
 c=g[ann+'_important_contrast'].eq('yes').to_numpy(float);sim=g[ann+'_labelled_similarity'].eq('yes').to_numpy(float)
 r=g.cosine_rank.to_numpy();p=g.probe_margin_rank.to_numpy();m=g.judge_margin.to_numpy()/50
 names=['contrast','similarity','cosine_rank','cosine_rank2','probe_rank','probe_rank2','judge_margin','judge_margin2','judge_margin3','neighbor_label','label_prediction_agree']
 X=np.column_stack([c,sim,r,r*r,p,p*p,m,m*m,m**3,g.nearest_label,g.label_prediction_agree]);w=g.w.to_numpy()
 Z=np.column_stack([g.flip.to_numpy(float),X]);ids=[pd.factorize(g.cell)[0]]
 if same_target:ids.append(pd.factorize(g.model+'|'+g.cluster)[0])
 err=np.inf
 for it in range(600):
  before=Z.copy()
  for k in ids:
   den=np.bincount(k,weights=w)
   for j in range(Z.shape[1]):Z[:,j]-=np.bincount(k,weights=w*Z[:,j])[k]/den[k]
  err=np.max(abs(Z-before))
  if err<1e-8 or len(ids)==1:break
 y=Z[:,0];X=Z[:,1:];inv=np.linalg.pinv(X.T@(w[:,None]*X));coef=inv@(X.T@(w*y));res=y-X@coef
 k=pd.factorize(g.cluster)[0];G=k.max()+1
 u=np.column_stack([np.bincount(k,weights=w*res*X[:,j],minlength=G) for j in range(X.shape[1])])
 cov=inv@(u.T@u)@inv*G/max(1,G-1);se=np.sqrt(np.maximum(np.diag(cov),0))
 return [dict(term=n,estimate=coef[j],lo=coef[j]-1.96*se[j],hi=coef[j]+1.96*se[j],n=len(g),targets=G,iterations=it+1,converged=err<1e-8 or len(ids)==1) for j,n in enumerate(names)]
rows=[];rr=[];byjudge=[];decomp=[];boundary=[]
for cohort,z in [('all10',d),('toxicity4',d[d.dataset.isin(T)])]:
 for ann in ['majority','qwen122','gemma','qwen']:
  a=z[z[ann+'_important_contrast'].isin(['yes','no'])&z[ann+'_labelled_similarity'].isin(['yes','no'])]
  for (aligned,agrees),risk in a.groupby(['aligned','cue_agrees_control']):
   direction='away' if agrees else 'toward'
   outcome='helped' if aligned!=agrees else 'harmed'
   for c,g in risk.groupby(ann+'_important_contrast'):
    mean,lo,hi=ci(g,g.flip);rows.append(dict(cohort=cohort,annotator=ann,aligned=aligned,control_agrees_neighbor=agrees,possible_direction=direction,flip_outcome=outcome,contrast=c,n=len(g),mass=g.w.sum(),flip_rate=mean,lo=lo,hi=hi))
   for st in ([False,True] if ann=='majority' else [False]):
    for r in fit(risk,ann,st):rr.append(dict(cohort=cohort,annotator=ann,aligned=aligned,control_agrees_neighbor=agrees,possible_direction=direction,flip_outcome=outcome,spec='same_target' if st else 'cell_adjusted',**r))
  print('risk',cohort,ann,flush=True)
 for m,g in z[z.contrast].groupby('model'):
  den=g.w.sum();hf=g.w[g.harmed].sum();hh=g.w[g.helped].sum()
  byjudge.append(dict(cohort=cohort,model=m,n=len(g),contrast_gain=np.average(g.gain,weights=g.w),harm_mass=hf,help_mass=hh,harm_toward_fraction=g.w[g.harmed&g.toward].sum()/hf if hf else np.nan,help_away_fraction=g.w[g.helped&g.away].sum()/hh if hh else np.nan))
 for scope,h in [('all_contrast',z[z.contrast]),('cues_agree',z[z.contrast&z.label_prediction_agree])]:
  for out in ['helped','harmed']:
   g=h[h[out]];den=g.w.sum()
   for mv in ['away','toward']:
    mean,lo,hi=ci(g,g[mv]);decomp.append(dict(cohort=cohort,scope=scope,outcome=out,direction=mv,n=len(g),fraction=mean,lo=lo,hi=hi))
 for feature in ['probe_bin','judge_margin_bin','cosine_bin5']:
  for (bn,al,agree),g in z[z.contrast].groupby([feature,'aligned','cue_agrees_control']):
   mean,lo,hi=ci(g,g.flip);boundary.append(dict(cohort=cohort,axis=feature,bin=str(bn),aligned=al,control_agrees_neighbor=agree,n=len(g),flip_rate=mean,lo=lo,hi=hi))
for name,rs in [('opportunities',rows),('risk_regressions',rr),('judge_contrast_decomposition',byjudge),('contrast_flip_decomposition',decomp),('boundary_opportunity_rates',boundary)]:pd.DataFrame(rs).to_csv(O/(name+'.csv'),index=False)
print(pd.DataFrame(decomp).round(4).to_string(index=False))
print(pd.DataFrame(rows).query("annotator=='majority'").round(4).to_string(index=False))
