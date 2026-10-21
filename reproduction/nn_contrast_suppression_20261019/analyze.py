import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import pandas as pd,numpy as np,json,gzip,hashlib
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';S=P/'api_results_20261015'
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[S/'results/scores.npz',S/'frozen/design.json.gz',P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json']
old=read(P/'two_policy_evaluation_20261018/results/COMPLETED.json')
for p in inputs[:2]:assert sha(p)==old['inputs'][str(p)]
d=read(inputs[1]);s=np.load(inputs[0]);f=pd.DataFrame([dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr]);f['ix']=np.arange(len(f));f=f[f.split=='test'].copy();ANN=['majority','qwen122','gemma','qwen'];cols=[a+'_important_contrast' for a in ANN]
a=pd.read_csv(inputs[2],usecols=['backbone','sample_id']+cols);f=f.merge(a,on=['backbone','sample_id'],validate='one_to_one');assert len(f)==25000;f['w']=1/f.inclusion_probability
th=pd.DataFrame(read(inputs[3]));th=th[th.scope!='common_valid'];th={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in th.itertuples()};local=['qwen','gemma','mistral','olmo','glm','qwen122']
protocol=dict(rule='If contrast YES, use original control score; otherwise use original NN score. Includes both-labels-YES. Uncertain/unresolved/missing retain original NN. No examples excluded.',selection='Fixed user-proposed rule; no cutoff fitting or new model calls. Majority annotation primary, each annotator separately sensitivity.',evaluation='Same original paired valid support and same probe/judge thresholds across methods. Fixed50 primary; existing three shared-anchor pairs plus direct judge50 sensitivities.',limits='Post-hoc test-set policy evaluation, proposed after inspecting test associations. Not fresh confirmatory evidence. Annotation labels do not contain target gold or observed outcome, but were generated for test only.')
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
rows=[];effects=[]
for m in s.files:
 valid=(s[m]>=0).all(axis=1)
 for (b,ds),g in f[valid[f.ix]].groupby(['backbone','dataset']):
  y=g.true_label.to_numpy();w=g.w.to_numpy();sc=s[m][g.ix]
  for pol in ['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor']:
   pt,jt=(.5,50) if pol=='fixed50' else (0.,50) if pol=='judge50_direct' else th[m,b,ds,pol.replace('_anchor','')]
   ref=g.probe_probability.to_numpy()>=pt;bp=(sc[:,0]>=jt)&ref;npred=(sc[:,2]>=jt)&ref
   if m in local and pol!='judge50_direct':pop=d['thresholds'][b][ds]['populations']['test'];n,pos=pop['n'],pop['positive']
   else:n,pos=w.sum(),w@y
   for ann in ANN:
    hide=g[ann+'_important_contrast'].eq('yes').to_numpy();gp=np.where(hide,bp,npred);key=dict(model=m,backbone=b,dataset=ds,policy=pol,annotator=ann)
    for method,pred,show in [('control',bp,np.zeros(len(g),bool)),('always_nn',npred,ref),('suppress_contrast',gp,ref&~hide)]:
     tp=w@(pred&(y==1));fp=w@(pred&(y==0));rows.append(dict(**key,method=method,f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/pos if pos else 0,accuracy=(tp+n-pos-fp)/n,nn_rate=w@show/n,test_n=len(g)))
    nc=npred==y;gc=gp==y
    effects.append(dict(**key,corrected=int((~nc&gc).sum()),lost_correct=int((nc&~gc).sum()),changed=int((gp!=npred).sum()),corrected_rate=w@(~nc&gc)/n,lost_correct_rate=w@(nc&~gc)/n,hidden=int((hide&ref).sum()),unknown=int((~g[ann+'_important_contrast'].isin(['yes','no'])).sum())))
 print(m,flush=True)
r=pd.DataFrame(rows);e=pd.DataFrame(effects);out=[];deltas=[]
for cohort in ['all','toxicity4']:
 rr=r if cohort=='all' else r[r.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])]
 for dim in ['overall','model','dataset','backbone']:
  groups=[('ALL',rr)] if dim=='overall' else rr.groupby(dim)
  for name,g in groups:
   for pol in ['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor','anchor_average']:
    h=g[g.policy.isin(['baseline_anchor','nearest_anchor','saliency_anchor'])] if pol=='anchor_average' else g[g.policy==pol]
    for (ann,method),z in h.groupby(['annotator','method']):out.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,annotator=ann,method=method,**{k:z[k].mean() for k in ['f1','precision','recall','accuracy','nn_rate']}))
    for ann,z in h.groupby('annotator'):
     idx=['model','backbone','dataset','policy'];ga=z[z.method=='suppress_contrast'].set_index(idx);ba=z[z.method=='always_nn'].set_index(idx);dd=ga[['f1','precision','recall','accuracy']]-ba[['f1','precision','recall','accuracy']]
     if pol=='anchor_average':dd=dd.groupby(level=[0,1,2]).mean()
     deltas.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,annotator=ann,cells=len(dd),wins=int((dd.f1>1e-10).sum()),losses=int((dd.f1< -1e-10).sum()),ties=int((abs(dd.f1)<=1e-10).sum()),**{k:dd[k].mean() for k in dd}))
summary=pd.DataFrame(out);delta=pd.DataFrame(deltas)
for name,z in [('cell_metrics',r),('changed_outcomes',e),('summary',summary),('deltas',delta)]:z.to_csv(O/(name+'.csv'),index=False)
# Verify original control and always NN metrics exactly match last checked analysis.
prior=pd.read_csv(P/'nn_similarity_gate_20261019/results/cell_metrics.csv');base=r[(r.annotator=='majority')&r.method.isin(['control','always_nn'])];check=base.merge(prior,on=['model','backbone','dataset','policy','method'],validate='one_to_one',suffixes=('_new','_old'));assert len(check)==5000
err=max(abs(check[k+'_new']-check[k+'_old']).max() for k in ['f1','precision','recall','accuracy']);assert err<1e-10
(O/'COMPLETED.json').write_text(json.dumps(dict(new_model_calls=0,contexts=25000,original_metric_replays=len(check),max_error=err,inputs={str(p):sha(p) for p in inputs},script_sha256=sha(Path(__file__))),indent=2))
print(summary.query("cohort=='all' and dimension=='overall' and annotator=='majority'").to_string(index=False));print(delta.query("cohort=='all' and dimension=='overall'").to_string(index=False))
