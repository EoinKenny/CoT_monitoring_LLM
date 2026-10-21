"""Mechanism annotation agreement and frozen-score outcome associations. No inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,gzip,hashlib,itertools,time
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;P=R.parent;I=R/'inputs';O=R/'results';O.mkdir(exist_ok=True)
ANNOTATORS=['qwen122','gemma','qwen'];LAB=['labelled_similarity','important_contrast'];TOX=['aegis','beavertails','toxicchat','toxigen']
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,rows):
 x=pd.DataFrame(rows);x.to_csv(O/(name+'.csv'),index=False);return x
def dump(name,x): (O/(name+'.json')).write_text(json.dumps(x,indent=2,allow_nan=False))
source=P/'api_results_20261015';design=read(source/'frozen/design.json.gz');case=pd.DataFrame([dict(backbone=b,**r) for b,rr in design['samples'].items() for r in rr]);scores=np.load(source/'results/scores.npz');MODELS=list(scores.files)
receipt=read(P/'two_policy_evaluation_20261018/results/COMPLETED.json')
for p in [source/'frozen/design.json.gz',source/'results/scores.npz']:assert sha(p)==receipt['inputs'][str(p)]
features=pd.read_csv(P/'nn_mechanisms_20261015/results/joined_features.csv.gz')
assert list(zip(features.backbone,features.sample_id))==list(zip(case.backbone,case.sample_id))
assert np.array_equal(features.true_label,case.true_label)
for k in ['probe_probability','inclusion_probability']:
 assert np.allclose(features[k],case[k],rtol=1e-14,atol=1e-14),k
export=read(P/'nn_mechanisms_20261015/EXPORT_COMPLETED.json');rawpath=P/'nn_mechanisms_20261015/features.jsonl.gz';assert sha(rawpath)==export['files']['features.jsonl.gz']
with gzip.open(rawpath,'rt') as stream:
 rawfeatures=pd.DataFrame([{k:r[k] for k in ['backbone','sample_id','nearest_label','nearest_cosine','target_content_hash']} for r in map(json.loads,stream)])
check=features.merge(rawfeatures,on=['backbone','sample_id'],suffixes=('_derived','_raw'),validate='one_to_one')
assert np.array_equal(check.nearest_label_derived,check.nearest_label_raw) and np.allclose(check.nearest_cosine_derived,check.nearest_cosine_raw) and np.array_equal(check.target_content_hash_derived,check.target_content_hash_raw)
f=features.loc[case.split.eq('test')].copy();f['source_index']=f.index;f=f.reset_index(drop=True)
map_=pd.read_json(I/'mapping_PRIVATE.jsonl',lines=True)
assert len(map_)==25000 and not map_[['backbone','sample_id']].duplicated().any()
f=f.merge(map_[['backbone','sample_id','unit_id']],on=['backbone','sample_id'],validate='one_to_one')
quality=[];agreements=[];frequency=[]
for m in ANNOTATORS:
 comp=read(I/m/'COMPLETED.json');assert sha(I/m/'annotations.jsonl')==comp['annotations_sha256'];assert sha(I/m/'labels_by_case.csv')==comp['case_labels_sha256']
 a=pd.read_json(I/m/'annotations.jsonl',lines=True);assert len(a)==25000 and a.unit_id.is_unique and set(a.unit_id)==set(f.unit_id)
 f=f.merge(a[['unit_id']+LAB+['status']].rename(columns={k:m+'_'+k for k in LAB+['status']}),on='unit_id',validate='one_to_one')
 quality.append(dict(annotator=m,**comp['status_counts']))
 for lab in LAB:
  col=m+'_'+lab;f[col]=f[col].fillna('missing');assert f[col].isin(['yes','no','uncertain','missing']).all()
  for level,n in f[col].value_counts().items():frequency.append(dict(annotator=m,label=lab,level=level,n=int(n),fraction=n/len(f)))
for lab in LAB:
 cols=[m+'_'+lab for m in ANNOTATORS];v=f[cols].to_numpy();maj=[]
 for row in v:
  votes={k:sum(row==k) for k in ['yes','no','uncertain']};w=max(votes,key=votes.get);maj.append(w if votes[w]>=2 else 'unresolved')
 f['majority_'+lab]=maj;f['unanimous_'+lab]=np.where((v[:,0]==v[:,1])&(v[:,0]==v[:,2])&(v[:,0]!='missing'),v[:,0],'unresolved')
 for level,n in pd.Series(maj).value_counts().items():frequency.append(dict(annotator='majority',label=lab,level=level,n=int(n),fraction=n/len(f)))
 for a,b in itertools.combinations(ANNOTATORS,2):
  good=f[a+'_'+lab].ne('missing')&f[b+'_'+lab].ne('missing');x=f.loc[good,a+'_'+lab];y=f.loc[good,b+'_'+lab];obs=float((x==y).mean());expected=sum(float(x.eq(k).mean()*y.eq(k).mean()) for k in ['yes','no','uncertain']);kap=(obs-expected)/(1-expected) if expected<1 else np.nan
  agreements.append(dict(label=lab,annotator_a=a,annotator_b=b,n=len(x),agreement=obs,kappa=kap))
  save('confusion_'+lab+'_'+a+'_'+b,[dict(a=aa,b=bb,n=int(((x==aa)&(y==bb)).sum())) for aa in ['yes','no','uncertain'] for bb in ['yes','no','uncertain']])
 agreements.append(dict(label=lab,annotator_a='all_three',annotator_b='exact',n=len(f),agreement=float(f['unanimous_'+lab].ne('unresolved').mean()),kappa=np.nan))
save('quality',quality);save('agreement',agreements);save('label_frequencies',frequency)
f.to_csv(O/'joined_annotations.csv.gz',index=False)
print('Agreement complete',flush=True)
# Frozen threshold pairs from existing three-anchor experiment; each pair shared by baseline and NN.
th=pd.DataFrame(read(P/'anchor_transfer_20261018/results/thresholds.json'));th=th[th.scope.ne('common_valid')]
thresholds={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in th.itertuples()};assert len(thresholds)==1500
common=np.logical_and.reduce([(scores[m]>=0).all(axis=1) for m in MODELS]);long=[]
keep=['source_index','unit_id','backbone','dataset','sample_id','true_label','probe_probability','weight','cluster','nearest_label','nearest_cosine','cosine_rank','label_aligned','label_prediction_agree','target_chars','neighbor_chars']+[c for c in f if any(c.startswith(m+'_') for m in ANNOTATORS+['majority','unanimous']) and any(c.endswith(l) for l in LAB)]
for model in MODELS:
 ix=f.source_index.to_numpy();valid=(scores[model][ix]>=0).all(axis=1);z=f.loc[valid,keep].copy();ix=z.source_index.to_numpy();s=scores[model][ix];z['model']=model;z['common_valid']=common[ix];z['cell']=model+'|'+z.backbone+'|'+z.dataset;z['w']=z.weight/z.groupby('cell').weight.transform('sum');z['control_score']=s[:,0];z['nearest_score']=s[:,2];z['brier_gain']=((s[:,0]/100-z.true_label)**2-(s[:,2]/100-z.true_label)**2)
 vals={}
 for policy in ['fixed50','baseline_anchor','nearest_anchor','saliency_anchor']:
  if policy=='fixed50':p=np.full(len(z),.5);cut=np.full(len(z),50)
  else:
   pairs=np.array([thresholds[model,b,ds,policy.replace('_anchor','')] for b,ds in zip(z.backbone,z.dataset)]);p=pairs[:,0];cut=pairs[:,1]
  referred=z.probe_probability.to_numpy()>=p;y=z.true_label.to_numpy();bc=((s[:,0]>=cut)&referred)==y;nc=((s[:,2]>=cut)&referred)==y
  vals[policy]=dict(helped=(~bc&nc).astype(float),harmed=(bc&~nc).astype(float),stable_correct=(bc&nc).astype(float),stable_wrong=(~bc&~nc).astype(float),base_wrong=(~bc).astype(float),base_correct=bc.astype(float),referred=referred.astype(float))
 for field in vals['fixed50']:vals.setdefault('anchor_average',{})[field]=np.mean([vals[a+'_anchor'][field] for a in ['baseline','nearest','saliency']],axis=0)
 for policy,metrics in vals.items():
  v=z.copy();v['policy']=policy
  for k,a in metrics.items():v[k]=a
  v['gain']=v.helped-v.harmed;assert np.allclose(v.helped+v.harmed+v.stable_correct+v.stable_wrong,1)
  long.append(v)
d=pd.concat(long,ignore_index=True);d.to_csv(O/'case_outcomes.csv.gz',index=False)
print('Outcomes constructed',len(d),flush=True)
# Outcome weights: equal cell mass prior to label conditioning. No different implicit
# cell weighting for yes/no. Cluster influence intervals use repeated target identities.
def stats(g):
 w=g.w.to_numpy();den=w.sum();r=dict(n=len(g),contexts=g.unit_id.nunique(),targets=g.cluster.nunique(),cells=g.cell.nunique(),mass=float(den),effective_n=float(den**2/(w@w)))
 for k in ['helped','harmed','gain','stable_correct','stable_wrong','brier_gain','referred']:
  x=g[k].to_numpy();mu=float(w@x/den);r[k]=mu
  if k=='gain':
   infl=pd.Series(w*(x-mu)/den).groupby(g.cluster.to_numpy()).sum().to_numpy();n=len(infl);se=np.sqrt((infl@infl)*n/max(n-1,1));r.update(gain_lo=mu-1.96*se,gain_hi=mu+1.96*se)
 r['correction_rate']=float(w@g.helped.to_numpy()/(w@g.base_wrong.to_numpy())) if w@g.base_wrong.to_numpy()>0 else np.nan
 r['damage_rate']=float(w@g.harmed.to_numpy()/(w@g.base_correct.to_numpy())) if w@g.base_correct.to_numpy()>0 else np.nan
 r['help_share_of_flips']=r['helped']/(r['helped']+r['harmed']) if r['helped']+r['harmed']>0 else np.nan
 return r
summary=[];breakdowns=[];joint=[]
for policy in d.policy.unique():
 v=d[d.policy.eq(policy)]
 for cohort,g0 in [('all',v),('toxicity4',v[v.dataset.isin(TOX)])]:
  summary.append(dict(policy=policy,cohort=cohort,annotator='none',label='overall',level='all',**stats(g0)))
  for ann in ANNOTATORS+['majority','unanimous']:
   for lab in LAB:
    for level,g in g0.groupby(ann+'_'+lab):summary.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,level=level,**stats(g)))
  if policy in ['fixed50','anchor_average']:
   for (a,b),g in g0.groupby(['majority_'+LAB[0],'majority_'+LAB[1]]):joint.append(dict(policy=policy,cohort=cohort,similarity=a,contrast=b,**stats(g)))
   for dimension in ['model','dataset','backbone','label_aligned']:
    for value,g1 in g0.groupby(dimension):
     for lab in LAB:
      for level,g in g1.groupby('majority_'+lab):breakdowns.append(dict(policy=policy,cohort=cohort,dimension=dimension,group=value,label=lab,level=level,**stats(g)))
 print('summarized',policy,flush=True)
save('summary',summary);save('breakdowns',breakdowns);save('joint_labels',joint)
# Target-cluster paired Poisson bootstrap for headline yes-minus-no net effects.
# Hold original macro weights fixed, resample entire targets across all repeats.
contrasts=[];rng=np.random.default_rng(20261019);B=2000
for policy in ['fixed50','anchor_average']:
 for cohort in ['all','toxicity4']:
  v=d[d.policy.eq(policy)];v=v if cohort=='all' else v[v.dataset.isin(TOX)]
  keys,cid=np.unique(v.cluster,return_inverse=True);G=len(keys);mats=[];meta=[]
  for ann in ANNOTATORS+['majority']:
   for lab in LAB:
    col=v[ann+'_'+lab].to_numpy();w=v.w.to_numpy();gain=v.gain.to_numpy();parts=[]
    for level in ['yes','no']:
     ww=w*(col==level);parts.extend([np.bincount(cid,weights=ww*gain,minlength=G),np.bincount(cid,weights=ww,minlength=G)])
    mats.extend(parts);meta.append((ann,lab))
  mat=np.stack(mats,axis=1);sums=mat.sum(0).reshape(-1,4);point=sums[:,0]/sums[:,1]-sums[:,2]/sums[:,3];boot=[]
  for lo in range(0,B,100):
   a=rng.poisson(1,(min(100,B-lo),G))@mat;a=a.reshape(-1,len(meta),4);boot.append(a[:,:,0]/a[:,:,1]-a[:,:,2]/a[:,:,3])
  lo,hi=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
  for j,(ann,lab) in enumerate(meta):contrasts.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,yes_minus_no=point[j],lo=lo[j],hi=hi[j],clusters=G,replicates=B))
  print('bootstrap',policy,cohort,flush=True)
save('bootstrap_contrasts',contrasts)
# Weighted within-cell regression, sandwich covariance clustered on target text.
def regress(g,ann,alignment_interactions=False):
 a=g[ann+'_'+LAB[0]].eq('yes').to_numpy().astype(float);b=g[ann+'_'+LAB[1]].eq('yes').to_numpy().astype(float);al=g.label_aligned.to_numpy().astype(float)
 xs=[a,b,a*b,g.true_label.to_numpy().astype(float),al,g.cosine_rank.to_numpy(),g.probe_probability.to_numpy(),np.log1p(g.target_chars.to_numpy()),np.log1p(g.neighbor_chars.to_numpy())]
 names=['similarity_yes','contrast_yes','similarity_x_contrast','true_label','neighbor_label_aligned','cosine_rank','probe_probability','log_target_chars','log_neighbor_chars']
 if alignment_interactions:xs.extend([a*al,b*al]);names.extend(['similarity_x_alignment','contrast_x_alignment'])
 X=np.stack(xs,axis=1);y=g.gain.to_numpy();w=g.w.to_numpy();_,cell=np.unique(g.cell,return_inverse=True);C=cell.max()+1;den=np.bincount(cell,weights=w,minlength=C)
 xm=np.stack([np.bincount(cell,weights=w*X[:,i],minlength=C)/den for i in range(X.shape[1])],axis=1);ym=np.bincount(cell,weights=w*y,minlength=C)/den
 X=X-xm[cell];y=y-ym[cell];bread=np.linalg.pinv(X.T@(w[:,None]*X));coef=bread@(X.T@(w*y));res=y-X@coef
 _,cl=np.unique(g.cluster,return_inverse=True);G=cl.max()+1;u=np.stack([np.bincount(cl,weights=w*X[:,j]*res,minlength=G) for j in range(X.shape[1])],axis=1);cov=bread@(u.T@u)@bread*G/(G-1);se=np.sqrt(np.maximum(np.diag(cov),0))
 return [dict(term=k,coef=coef[i],lo=coef[i]-1.96*se[i],hi=coef[i]+1.96*se[i],n=len(g),clusters=G,cells=C,rank=int(np.linalg.matrix_rank(X))) for i,k in enumerate(names)]
regs=[];sensitivity=[]
for policy in ['fixed50','anchor_average']:
 for cohort in ['all','toxicity4']:
  v=d[d.policy.eq(policy)];v=v if cohort=='all' else v[v.dataset.isin(TOX)]
  for ann in ANNOTATORS+['majority']:
   g=v[v[ann+'_'+LAB[0]].isin(['yes','no'])&v[ann+'_'+LAB[1]].isin(['yes','no'])]
   for interactions in [False,True]:
    for row in regress(g,ann,interactions):regs.append(dict(policy=policy,cohort=cohort,annotator=ann,spec='alignment_interactions' if interactions else 'joint',**row))
  for scope,sub in [('common_all10',v[v.common_valid]),('other7_judges',v[~v.model.isin(ANNOTATORS)]),('six_local',v[v.model.isin(MODELS[:6])]),('four_api',v[v.model.isin(MODELS[6:])])]:
   sub=sub.copy();sub['w']=sub.weight/sub.groupby('cell').weight.transform('sum')
   for lab in LAB:
    for level,g in sub.groupby('majority_'+lab):sensitivity.append(dict(policy=policy,cohort=cohort,scope=scope,label=lab,level=level,**stats(g)))
  print('regression',policy,cohort,flush=True)
save('adjusted_associations',regs);save('sensitivities',sensitivity)
# Descriptive conditional correction/harm regressions under the actual binary fixed policy.
v=d[d.policy.eq('fixed50')];cond=[]
for ann in ANNOTATORS+['majority']:
 for originally_correct in [False,True]:
  g=v[v.base_correct.eq(int(originally_correct))&v[ann+'_'+LAB[0]].isin(['yes','no'])&v[ann+'_'+LAB[1]].isin(['yes','no'])].copy()
  g['gain']=g.harmed if originally_correct else g.helped
  for row in regress(g,ann):cond.append(dict(annotator=ann,outcome='damage_rate' if originally_correct else 'correction_rate',**row))
save('conditional_rates_adjusted',cond)
inputs=[R/'ANALYSIS_PLAN.json',Path(__file__),source/'results/scores.npz',source/'frozen/design.json.gz',P/'anchor_transfer_20261018/results/thresholds.json',P/'nn_mechanisms_20261015/results/joined_features.csv.gz']+[I/m/'annotations.jsonl' for m in ANNOTATORS]
dump('COMPLETED',dict(time=time.time(),test_contexts=len(f),distinct_targets=f.cluster.nunique(),outcome_judges=MODELS,rows_per_policy=d.groupby('policy').size().to_dict(),quality=quality,new_model_calls=0,inputs={str(p):sha(p) for p in inputs},outputs={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='COMPLETED.json'}))
print('COMPLETE',flush=True)
