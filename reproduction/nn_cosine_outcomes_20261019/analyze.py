import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
R=Path(__file__).resolve().parent;O=R/'results';S=R.parent/'nn_mechanism_agreement_20261019/results/case_outcomes.csv.gz'
LAB=['labelled_similarity','important_contrast'];ANN=['majority','qwen122','gemma','qwen'];EV=['helped','harmed','unchanged']
cols=['policy','model','backbone','dataset','unit_id','cluster','w','cell','true_label','control_score','nearest_score','nearest_cosine','cosine_rank','label_aligned','helped','harmed','stable_correct','stable_wrong']+[a+'_'+l for a in ANN for l in LAB]
d=pd.concat([c[c.policy.isin(['fixed50','anchor_average'])] for c in pd.read_csv(S,usecols=cols,chunksize=100000)],ignore_index=True)
z=d[d.policy=='fixed50'].copy();bc=(z.control_score>=50)==z.true_label;nc=(z.nearest_score>=50)==z.true_label
z['policy']='judge50_direct';z['helped']=(~bc&nc).astype(float);z['harmed']=(bc&~nc).astype(float);z['stable_correct']=(bc&nc).astype(float);z['stable_wrong']=(~bc&~nc).astype(float)
d=pd.concat([z,d],ignore_index=True);d['unchanged']=d.stable_correct+d.stable_wrong
d['distance_bin']=np.minimum(np.ceil(d.cosine_rank*5).astype(int),5)
assert np.isfinite(d.nearest_cosine).all() and np.allclose(d[EV].sum(axis=1),1)
assert len(z)==240524

def ci(g,w,x):
 den=w.sum();mu=w@x/den;v=pd.Series(w*(x-mu)/den).groupby(g.cluster.to_numpy()).sum().to_numpy();se=np.sqrt(v@v*len(v)/max(len(v)-1,1));return mu,mu-1.96*se,mu+1.96*se

def quant(x,w):
 i=np.argsort(x);return np.interp([.25,.5,.75],np.cumsum(w[i])/w.sum(),x[i])
rows=[];hist=[];bins=[]
for policy,p in d.groupby('policy'):
 for ann in ANN:
  for lab in LAB:
   for lev in ['yes','no']:
    g=p[p[ann+'_'+lab]==lev]
    for ev in EV:
     w=(g.w*g[ev]).to_numpy();m=w>0;h=g[m];ww=w[m]
     for metric in ['nearest_cosine','cosine_rank']:
      x=h[metric].to_numpy();mu,lo,hi=ci(h,ww,x);q=quant(x,ww)
      rows.append(dict(policy=policy,annotator=ann,label=lab,level=lev,outcome=ev,metric=metric,n=float(g[ev].sum()),mean=mu,lo=lo,hi=hi,q25=q[0],median=q[1],q75=q[2]))
      if ann=='majority':
       edges=np.linspace(0,1,41);hh,_=np.histogram(x,bins=edges,weights=ww);hh=hh/hh.sum()
       for i,v in enumerate(hh):hist.append(dict(policy=policy,label=lab,level=lev,outcome=ev,metric=metric,left=edges[i],right=edges[i+1],mass=v))
    for bn,h in g.groupby('distance_bin'):
     w=h.w.to_numpy();flip=(h.helped+h.harmed).to_numpy();nflip=w@flip
     for metric,x,wt in [('helped',h.helped.to_numpy(),w),('harmed',h.harmed.to_numpy(),w),('gain',(h.helped-h.harmed).to_numpy(),w),('harm_among_flips',np.divide(h.harmed.to_numpy(),flip,out=np.zeros(len(h)),where=flip>0),w*flip)]:
      mu,lo,hi=ci(h,wt,x);bins.append(dict(policy=policy,annotator=ann,label=lab,level=lev,bin=bn,metric=metric,mean=mu,lo=lo,hi=hi,n=len(h),flip_count=float(flip.sum())))
 print('done distributions',policy,flush=True)
pd.DataFrame(rows).to_csv(O/'distribution_summary.csv',index=False);pd.DataFrame(hist).to_csv(O/'histograms.csv',index=False);pd.DataFrame(bins).to_csv(O/'distance_bins.csv',index=False)
# Same-cell weighted linear comparisons; cluster exact target text across repeats.
reg=[]
for policy,p in d.groupby('policy'):
 for ann in ANN:
  for outcome in ['gain','harm_among_flips']:
   g=p[p[ann+'_important_contrast'].isin(['yes','no']) & p[ann+'_labelled_similarity'].isin(['yes','no'])].copy()
   if outcome=='harm_among_flips':g=g[g.helped+g.harmed>0].copy();g['rw']=g.w*(g.helped+g.harmed);y=(g.harmed/(g.helped+g.harmed)).to_numpy()
   else:g['rw']=g.w;y=(g.helped-g.harmed).to_numpy()
   w=g.rw.to_numpy();c=g[ann+'_important_contrast'].eq('yes').astype(float).to_numpy();rank=g.cosine_rank.to_numpy();raw=g.nearest_cosine.to_numpy()
   for spec in ['cell_only','cell_distance','cell_distance_similarity','cell_distance_alignment','cell_distance_similarity_alignment']:
    xs=[c];names=['contrast_yes']
    if spec!='cell_only':
     for k in [1,2,3]:xs.extend([rank**k,raw**k]);names.extend(['rank'+str(k),'cosine'+str(k)])
    if 'similarity' in spec:xs.append(g[ann+'_labelled_similarity'].eq('yes').astype(float).to_numpy());names.append('similarity')
    if 'alignment' in spec:xs.extend([g.label_aligned.to_numpy(),g.true_label.to_numpy()]);names.extend(['alignment','true_label'])
    X=np.column_stack(xs);codes=pd.factorize(g.cell)[0];den=np.bincount(codes,weights=w);yc=y-np.bincount(codes,weights=w*y)[codes]/den[codes];Xc=X.copy()
    for j in range(X.shape[1]):Xc[:,j]-=np.bincount(codes,weights=w*X[:,j])[codes]/den[codes]
    inv=np.linalg.pinv(Xc.T@(w[:,None]*Xc));coef=inv@(Xc.T@(w*yc));res=yc-Xc@coef
    cl=pd.factorize(g.cluster)[0];G=cl.max()+1;scores=np.column_stack([np.bincount(cl,weights=w*res*Xc[:,j],minlength=G) for j in range(X.shape[1])]);cov=inv@(scores.T@scores)*G/(G-1)@inv;se=np.sqrt(max(cov[0,0],0))
    reg.append(dict(policy=policy,annotator=ann,outcome=outcome,spec=spec,n=len(g),clusters=G,contrast_yes_coefficient=coef[0],lo=coef[0]-1.96*se,hi=coef[0]+1.96*se))
pd.DataFrame(reg).to_csv(O/'distance_adjusted.csv',index=False)
receipt=dict(source=str(S),source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),judge_context_pairs=len(z),contexts=z.unit_id.nunique(),clusters=z.cluster.nunique(),new_model_calls=0,primary_policy='Direct original judge decisions at 50, no probe gate',weights='Inverse inclusion weights normalized within original judge/backbone/dataset cells; same weights as prior analysis',distance_rank='Rank within backbone,dataset,split; quintiles on test contexts. Raw cosine is not directly comparable across embedding spaces.',intervals='Cluster sandwich intervals across exact target text, repeated judges and backbones kept together',caveat='Observational distance adjustment cannot establish mediation or causality. Conditional harm among flips excludes unchanged decisions; net gain includes all events.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'COMPLETED.json').write_text(json.dumps(receipt,indent=2))
print(pd.DataFrame(reg).query("annotator=='majority' and policy=='judge50_direct'").to_string(index=False))
