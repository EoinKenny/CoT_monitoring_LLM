import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,json,gzip,hashlib
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
protocol=dict(primary='Per original judge/backbone/dataset, keep probe and judge at50; choose NN cosine cutoff by maximum weighted validation F1. NN scores above cutoff, control below. Compare never NN and always NN on identical test cases.',candidates='Every unique validation cosine and never-NN (threshold1.0000001), plus always-NN (threshold-1); use >=. Max F1 tie within1e-12: fewer validation NN shown, then higher threshold. Include endpoints explicitly.',sensitivity='Direct judge50 without probe gating; existing control,NN,saliency-anchor common probe/judge pairs, each fixed before cosine search. Average three anchors as sensitivity, not independent repetitions. Reuse prior selection, do not retune probe/judge.',missingness='Original 3-condition matched-valid cases per judge. Local full population denominators as prior core evaluation; API estimates conditional on matched sampled support. Direct judge comparison uses sampled support for all judges.',labels='Only test annotations exist, so no contrast-label gate is fitted. Test labels are descriptive subgroups after freezing cosine cutoff.',limits='Previously inspected test set; exploratory follow-up. No new model calls. Test F1 does not select thresholds or policies.')
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
S=P/'api_results_20261015';d=read(S/'frozen/design.json.gz');s=np.load(S/'results/scores.npz');f=pd.read_csv(P/'nn_mechanisms_20261015/results/joined_features.csv.gz');case=pd.DataFrame([dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr]);assert list(zip(f.backbone,f.sample_id))==list(zip(case.backbone,case.sample_id));assert np.array_equal(f.true_label,case.true_label)
old=read(P/'two_policy_evaluation_20261018/results/COMPLETED.json')
for path in [S/'frozen/design.json.gz',S/'results/scores.npz']:assert sha(path)==old['inputs'][str(path)]
th=pd.DataFrame(read(P/'anchor_transfer_20261018/results/thresholds.json'));th=th[th.scope!='common_valid'];thresholds={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in th.itertuples()}
ann=pd.read_csv(P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz',usecols=['backbone','sample_id','majority_labelled_similarity','majority_important_contrast']);f=f.merge(ann,on=['backbone','sample_id'],how='left',validate='one_to_one');f['ix']=np.arange(len(f));f['w']=1/f.inclusion_probability
local=['qwen','gemma','mistral','olmo','glm','qwen122'];rows=[];selections=[];subgroups=[];caseout=[]
def metrics(y,p,w,pop):
 tp=w@(p&(y==1));fp=w@(p&(y==0));n,pos=pop;den=pos+tp+fp
 return dict(f1=2*tp/den if den else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else 0.,accuracy=(tp+n-pos-fp)/n,tp=tp,fp=fp,n=n,positive=pos)
def population(g,m,b,ds,split,policy):
 if m in local and policy!='judge50_direct':q=d['thresholds'][b][ds]['populations'][split];return q['n'],q['positive']
 return g.w.sum(),g.w@g.true_label
for m in s.files:
 valid=(s[m]>=0).all(axis=1)
 for (b,ds),gg in f[valid].groupby(['backbone','dataset'],sort=True):
  gv=gg[gg.split=='val'];gt=gg[gg.split=='test'];assert len(gv) and len(gt)
  for policy in ['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor']:
   pt,jt=(.5,50) if policy=='fixed50' else (0.,50) if policy=='judge50_direct' else thresholds[m,b,ds,policy.replace('_anchor','')]
   def inputs(g):
    y=g.true_label.to_numpy();w=g.w.to_numpy();ref=g.probe_probability.to_numpy()>=pt;ss=s[m][g.ix];return y,w,ref,(ss[:,0]>=jt)&ref,(ss[:,2]>=jt)&ref,g.nearest_cosine.to_numpy()
   yv,wv,rv,bv,nv,cv=inputs(gv);yt,wt,rt,bt,nt,ct=inputs(gt);pv=population(gv,m,b,ds,'val',policy);pop=population(gt,m,b,ds,'test',policy)
   # Sorted cumulative swap from control to NN; evaluate tied-cosine blocks together.
   order=np.argsort(-cv,kind='stable');cos=cv[order];diff=nv.astype(float)-bv.astype(float);base=metrics(yv,bv,wv,pv);tp=base['tp']+np.cumsum((wv*diff*(yv==1))[order]);fp=base['fp']+np.cumsum((wv*diff*(yv==0))[order]);ends=np.r_[np.where(cos[:-1]!=cos[1:])[0],len(cos)-1]
   cuts=np.r_[1.0000001,cos[ends],-1.];tps=np.r_[base['tp'],tp[ends],tp[-1]];fps=np.r_[base['fp'],fp[ends],fp[-1]];den=pv[1]+tps+fps;f1=np.divide(2*tps,den,out=np.zeros(len(cuts)),where=den>0)
   counts=np.r_[0,np.cumsum(rv[order])[ends],rv.sum()];best=np.flatnonzero(f1>=f1.max()-1e-12);best=sorted(best,key=lambda i:(counts[i],-cuts[i]))[0];cut=float(cuts[best]);gvpred=np.where(cv>=cut,nv,bv);assert abs(metrics(yv,gvpred,wv,pv)['f1']-f1[best])<1e-10
   shown=rt&(ct>=cut);gated=np.where(ct>=cut,nt,bt);key=dict(model=m,backbone=b,dataset=ds,policy=policy)
   selections.append(dict(**key,probe_threshold=pt,judge_threshold=jt,cosine_threshold=cut,validation_n=len(gv),test_n=len(gt),validation_f1=f1[best],validation_control_f1=base['f1'],validation_nn_f1=metrics(yv,nv,wv,pv)['f1'],validation_nn_shown=int(counts[best]),test_nn_shown=int(shown.sum()),test_referred=int(rt.sum()),test_nn_rate=wt@shown/pop[0],nn_fraction_of_referrals=wt@shown/(wt@rt) if wt@rt else 0.))
   for method,pred,show in [('control',bt,np.zeros(len(gt),bool)),('always_nn',nt,rt),('similarity_gate',gated,shown)]:
    rows.append(dict(**key,method=method,**metrics(yt,pred,wt,pop),nn_rate=wt@show/pop[0],referral_rate=wt@rt/pop[0]))
   for lab in ['labelled_similarity','important_contrast']:
    for lev in ['yes','no']:
     mask=gt['majority_'+lab].eq(lev).to_numpy();ww=wt[mask];n=ww.sum()
     for method,pred in [('always_nn',nt),('similarity_gate',gated)]:
      bc=bt[mask]==yt[mask];nc=pred[mask]==yt[mask];subgroups.append(dict(**key,label=lab,level=lev,method=method,mass=n/popup if False else n/wt.sum(),help_mass=ww@(~bc&nc)/wt.sum(),harm_mass=ww@(bc&~nc)/wt.sum(),show_mass=ww@shown[mask]/wt.sum(),count=int(mask.sum())))
   if policy in ['fixed50','judge50_direct']:
    for i,ix in enumerate(gt.ix):caseout.append(dict(**key,source_index=int(ix),cosine_threshold=cut,show_nn=bool(shown[i]),control=int(bt[i]),always_nn=int(nt[i]),gated=int(gated[i])))
 print('completed',m,flush=True)
res=pd.DataFrame(rows);sel=pd.DataFrame(selections);sub=pd.DataFrame(subgroups)
# Equal-cell macro means, also split into judge/backbone/dataset and toxicity subset.
summary=[];deltas=[]
for cohort in ['all','toxicity4']:
 rr=res if cohort=='all' else res[res.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])]
 for dim in ['overall','model','backbone','dataset']:
  groups=[('ALL',rr)] if dim=='overall' else rr.groupby(dim)
  for name,g in groups:
   for pol in ['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor','anchor_average']:
    q=g[g.policy.isin(['baseline_anchor','nearest_anchor','saliency_anchor'])] if pol=='anchor_average' else g[g.policy==pol]
    for method,h in q.groupby('method'):
     summary.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,method=method,cells=len(h)//(3 if pol=='anchor_average' else 1),**{k:h[k].mean() for k in ['f1','precision','recall','accuracy','nn_rate','referral_rate']}))
    for ref in ['control','always_nn']:
     ga=q[q.method=='similarity_gate'].set_index(['model','backbone','dataset','policy']);ba=q[q.method==ref].set_index(['model','backbone','dataset','policy']);dd=ga[['f1','precision','recall','accuracy']]-ba[['f1','precision','recall','accuracy']]
     if pol=='anchor_average':dd=dd.groupby(level=[0,1,2]).mean()
     deltas.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,reference=ref,cells=len(dd),f1_wins=int((dd.f1>1e-10).sum()),f1_losses=int((dd.f1< -1e-10).sum()),f1_ties=int((abs(dd.f1)<=1e-10).sum()),**{k:dd[k].mean() for k in dd}))
for name,df in [('cell_metrics',res),('selected_thresholds',sel),('summary',pd.DataFrame(summary)),('paired_deltas',pd.DataFrame(deltas)),('test_label_subgroups',sub)]:df.to_csv(O/(name+'.csv'),index=False)
pd.DataFrame(caseout).to_csv(O/'test_decisions.csv.gz',index=False)
# Replay same fixed and anchored conditions against original results.
oldanchor=pd.DataFrame(read(P/'anchor_transfer_20261018/results/cells.json'));oldanchor=oldanchor[oldanchor.scope!='common_valid'];checks=0;maxerr=0.
for r in res[res.policy.str.endswith('_anchor')&res.method.isin(['control','always_nn'])].itertuples():
 condition='baseline' if r.method=='control' else 'nearest';a=oldanchor[(oldanchor.model==r.model)&(oldanchor.backbone==r.backbone)&(oldanchor.dataset==r.dataset)&(oldanchor.anchor==r.policy.replace('_anchor',''))&(oldanchor.condition==condition)].iloc[0]
 for k in ['f1','precision','recall','accuracy']:maxerr=max(maxerr,abs(getattr(r,k)-a[k]));assert abs(getattr(r,k)-a[k])<1e-10
 checks+=1
receipt=dict(new_model_calls=0,cell_policies=len(sel),source_cases=len(f),validation_and_test_separate=True,anchor_replay_rows=checks,max_anchor_replay_error=maxerr,inputs={str(p):sha(p) for p in [S/'frozen/design.json.gz',S/'results/scores.npz',P/'nn_mechanisms_20261015/results/joined_features.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json',R/'PROTOCOL.json',Path(__file__)]})
(O/'COMPLETED.json').write_text(json.dumps(receipt,indent=2));print(pd.DataFrame(summary).query("cohort=='all' and dimension=='overall'").to_string(index=False));print(pd.DataFrame(deltas).query("cohort=='all' and dimension=='overall'").to_string(index=False))
