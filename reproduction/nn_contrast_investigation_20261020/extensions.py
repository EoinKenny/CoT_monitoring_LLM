"""No new calls: add frozen shared-anchor and existing 2-NN sensitivities."""
from pathlib import Path
import pandas as pd,numpy as np,json,gzip,hashlib
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';T=['aegis','beavertails','toxicchat','toxigen']
use=['policy','model','backbone','dataset','sample_id','unit_id','cluster','cell','w','weight','true_label','nearest_label','probe_probability','control_score','nearest_score','aligned','toward','away','majority_important_contrast','majority_labelled_similarity']
d=pd.concat([g[g.policy.eq('judge50_direct')] for g in pd.read_csv(O/'cases.csv.gz',usecols=use,chunksize=100000)],ignore_index=True)
def mean_ci(g,x):
 w=g.w.to_numpy();x=np.asarray(x,float);mu=np.average(x,weights=w);u=pd.Series(w*(x-mu)/w.sum()).groupby(g.cluster.to_numpy()).sum().to_numpy();se=np.sqrt(u@u*len(u)/max(len(u)-1,1));return mu,mu-1.96*se,mu+1.96*se
th=P/'anchor_transfer_20261018/results/thresholds.json';t=pd.DataFrame(json.load(open(th)));t=t[t.scope.ne('common_valid')]
k=['model','backbone','dataset'];assert not t.duplicated(k+['anchor']).any()
a=d.merge(t[k+['anchor','probe_threshold','judge_threshold']],on=k,validate='many_to_many');assert len(a)==3*len(d)
ref=a.probe_probability.ge(a.probe_threshold);bc=(a.control_score.ge(a.judge_threshold)&ref).eq(a.true_label);nc=(a.nearest_score.ge(a.judge_threshold)&ref).eq(a.true_label)
a['helped']=~bc&nc;a['harmed']=bc&~nc;a['gain']=nc.astype(int)-bc.astype(int);a['zero_judge']=a.judge_threshold.eq(0);assert a.loc[a.zero_judge,'gain'].eq(0).all()
ar=[]
for cohort,z in [('all10',a),('toxicity4',a[a.dataset.isin(T)])]:
 for anchor,h in [('three_anchor_average',z)]+list(z.groupby('anchor')):
  for c,g in h.groupby('majority_important_contrast'):
   mu,lo,hi=mean_ci(g,g.gain);har=g[g.harmed];v,l,u=mean_ci(har,har.toward) if len(har) else (np.nan,)*3
   ar.append(dict(cohort=cohort,anchor=anchor,contrast=c,n=len(g),gain=mu,lo=lo,hi=hi,helped=np.average(g.helped,weights=g.w),harmed=np.average(g.harmed,weights=g.w),harm_toward=v,harm_toward_lo=l,harm_toward_hi=u,zero_judge_mass=np.average(g.zero_judge,weights=g.w)))
pd.DataFrame(ar).to_csv(O/'shared_anchor_sensitivity.csv',index=False)
p=P/'two_nn_20261014/export_verified_20261016/nearest2_scores.jsonl.gz';sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==json.load(open(p.parent/'VERIFIED.json'))['score_export_sha256']
rr=[]
with gzip.open(p,'rt') as f:
 for line in f:
  r=json.loads(line);assert r['condition']=='nearest2' and r['mode']=='score';rr.append({k:r[k] for k in ['model','backbone','dataset','sample_id','positive_probability']})
s=pd.DataFrame(rr);keys=['model','backbone','dataset','sample_id'];assert len(s)==300000 and not s.duplicated(keys).any();assert s.positive_probability.between(0,100).all()
g=d[d.model.isin(s.model.unique())].merge(s,on=keys,validate='one_to_one');assert len(g)==150000
rows=[];cells=[];heterogeneity=[];interactions=[];balanced=[]
for policy in ['judge50_direct','pipeline50']:
 ref=np.ones(len(g),bool) if policy=='judge50_direct' else g.probe_probability.ge(.5).to_numpy();y=g.true_label.to_numpy();b=(g.control_score.ge(50).to_numpy()&ref);n1=g.nearest_score.ge(50).to_numpy()&ref;n2=g.positive_probability.ge(50).to_numpy()&ref
 z=g.copy();z['corrected']=(n1!=y)&(n2==y);z['undid']=(n1==y)&(n2!=y);z['delta']=z.corrected.astype(int)-z.undid.astype(int);z['gain1']=(n1==y).astype(int)-(b==y).astype(int);z['gain2']=(n2==y).astype(int)-(b==y).astype(int)
 for cohort,h in [('all10',z),('toxicity4',z[z.dataset.isin(T)])]:
  for c,h0 in h.groupby('majority_important_contrast'):
   for alignment,v in [('all',h0)]+list(h0.groupby('aligned')):
    mu,lo,hi=mean_ci(v,v.delta);rows.append(dict(policy=policy,cohort=cohort,contrast=c,alignment=alignment,n=len(v),delta=mu,lo=lo,hi=hi,gain1=np.average(v.gain1,weights=v.w),gain2=np.average(v.gain2,weights=v.w),corrected=np.average(v.corrected,weights=v.w),undid=np.average(v.undid,weights=v.w)))
 for cohort,h in [('all10',z),('toxicity4',z[z.dataset.isin(T)])]:
  hh=h[h.majority_important_contrast.isin(['yes','no'])].copy()
  group=hh.groupby(['cell','majority_important_contrast']);cellmass=group.w.transform('sum');hh['wc']=hh.w/cellmass
  hh['weighted_delta']=hh.wc*hh.delta;cm=hh.groupby(['cell','majority_important_contrast']).weighted_delta.sum().unstack();cm=cm.dropna();hh=hh[hh.cell.isin(cm.index)].copy();ncells=len(cm)
  influence=[]
  for cc,v in hh.groupby('majority_important_contrast'):
   residual=v.delta.to_numpy()-v.cell.map(cm[cc]).to_numpy();sgn=1 if cc=='yes' else -1;influence.append(pd.Series(sgn*v.wc.to_numpy()*residual/ncells,index=v.cluster.to_numpy()))
  u=pd.concat(influence).groupby(level=0).sum().to_numpy();se=np.sqrt(u@u*len(u)/max(len(u)-1,1));diff=(cm.yes-cm.no).mean();balanced.append(dict(policy=policy,cohort=cohort,cells=ncells,contrast_delta=cm.yes.mean(),no_contrast_delta=cm.no.mean(),difference=diff,lo=diff-1.96*se,hi=diff+1.96*se))
  hh=h[h.majority_important_contrast.isin(['yes','no'])].copy();contributions=[];means={}
  for c,v in hh.groupby('majority_important_contrast'):
   w=v.w.to_numpy();mu=np.average(v.delta,weights=w);means[c]=mu
   contributions.append(pd.Series((1 if c=='yes' else -1)*w*(v.delta.to_numpy()-mu)/w.sum(),index=v.cluster.to_numpy()))
  assert all(c.notna().all() for c in contributions)
  u=pd.concat(contributions).groupby(level=0).sum().to_numpy();assert np.any(u!=0);se=np.sqrt(u@u*len(u)/max(len(u)-1,1));diff=means['yes']-means['no'];interactions.append(dict(policy=policy,cohort=cohort,difference=diff,lo=diff-1.96*se,hi=diff+1.96*se))
  for dim in ['model','dataset','backbone']:
   for (value,c),v in h.groupby([dim,'majority_important_contrast']):
    mu,lo,hi=mean_ci(v,v.delta);heterogeneity.append(dict(policy=policy,cohort=cohort,dimension=dim,value=value,contrast=c,n=len(v),delta=mu,lo=lo,hi=hi,gain1=np.average(v.gain1,weights=v.w),gain2=np.average(v.gain2,weights=v.w)))
 # F1 is calculated per original cell and averaged, not over pooled decisions.
 z['pred_control']=b;z['pred_nn1']=n1;z['pred_nn2']=n2
 for (model,backbone,ds),h in z.groupby(k):
  for c,v in [('all',h)]+list(h.groupby('majority_important_contrast')):
   for method in ['control','nn1','nn2']:
    pred=v['pred_'+method].to_numpy();yy=v.true_label.to_numpy();w=v.weight.to_numpy();tp=w@(pred&(yy==1));fp=w@(pred&(yy==0));fn=w@(~pred&(yy==1));f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    cells.append(dict(policy=policy,model=model,backbone=backbone,dataset=ds,contrast=c,method=method,n=len(v),f1=f1))
pd.DataFrame(balanced).to_csv(O/'second_nn_balanced_cells.csv',index=False);pd.DataFrame(heterogeneity).to_csv(O/'second_nn_heterogeneity.csv',index=False);pd.DataFrame(interactions).to_csv(O/'second_nn_interaction.csv',index=False);pd.DataFrame(rows).to_csv(O/'second_nn_sensitivity.csv',index=False);pd.DataFrame(cells).to_csv(O/'second_nn_cell_f1.csv',index=False)
(O/'EXTENSIONS_VERIFIED.json').write_text(json.dumps(dict(second_nn_score_hash=sha,second_nn_total=300000,second_nn_paired_test=150000,models=sorted(s.model.unique()),anchor_paired_rows=len(a),anchor_zero_threshold_no_effect=True,second_neighbor_label_metadata_available=False,limits='Original 1-NN mechanism annotations, not reannotations of 2-NN prompts. Second-neighbour labels unavailable locally; cannot test whether examples bracket a boundary. Six local judges only for 2-NN.',thresholds_sha256=hashlib.sha256(th.read_bytes()).hexdigest()),indent=2))
print(pd.DataFrame(rows).query("policy=='judge50_direct' and alignment=='all'").round(5).to_string(index=False));print(pd.DataFrame(ar).query("anchor=='three_anchor_average'").round(5).to_string(index=False))
