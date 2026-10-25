import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MPLCONFIGDIR='/private/tmp/nn_paper_mpl')
from pathlib import Path
import json,hashlib,collections
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';O.mkdir(exist_ok=True)
S=P/'nn_trace_labels_qwen122_revision_20261024'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
comp=json.loads((S/'COMPLETED.json').read_text());assert sha(S/'annotations.jsonl')==comp['annotations_sha256']
a=pd.read_json(S/'annotations.jsonl',lines=True);assert a.unit_id.is_unique and len(a)==44089
m=pd.read_json(S/'mapping_PRIVATE.jsonl',lines=True);assert not m.duplicated(['judge','backbone','sample_id']).any()
d=m.merge(a,on='unit_id',how='left',validate='many_to_one');assert len(d)==45638;assert d.loc[d.eligibility.eq('eligible'),'status'].notna().all()
for key in ['similarity','contrast','assessable']:d['trace_'+key]=d.labels.map(lambda x:x.get(key) if isinstance(x,dict) else None)
d['outcome']=np.where(d.original_helped,'Corrects judge','Misleads judge');assert (d.original_helped^d.original_harmed).all()
d['score']=d.answer_lines_removed.map(lambda x:int(x[0]['value']) if x else np.nan)
d['probability_first']=d.answer_lines_removed.map(lambda x:bool(x and x[0]['edge']=='first'))
d['retained']=d.score.notna()&((d.score>=50)==(d.original_nn>=50))
d['complete_source']=~d.finish_reason.isin(['length','max_tokens','MAX_TOKENS'])
genres={'aegis':'Toxicity','beavertails':'Toxicity','toxicchat':'Toxicity','toxigen':'Toxicity','wildguard':'Safety','wildjailbreak':'Safety','fava':'Hallucination','halueval':'Hallucination','ragtruth':'Hallucination','high_stakes':'High stakes'};d['genre']=d.dataset.map(genres);assert d.genre.notna().all()
# Verify frozen pair labels and outcomes against original probability-only records.
f=pd.read_pickle(P/'nn_contrast_extremity_20261024/results/joined.pkl').set_index(['judge','backbone','sample_id'])
ix=pd.MultiIndex.from_frame(d[['judge','backbone','sample_id']]);v=f.loc[ix]
for col,orig in [('similarity','original_similarity'),('contrast','original_contrast'),('control','original_control'),('nn','original_nn'),('helped','original_helped')]:assert np.array_equal(v[col].to_numpy(),d[orig].to_numpy())
# Sensitivity: exclude both the source judge and Qwen122 from the original label consensus.
A=np.load(P/'mechanisms_all12_20261023/annotations.npz');mp=[json.loads(x) for x in (P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl').open()];lookup={(x['backbone'],x['sample_id']):i for i,x in enumerate(mp)};ai=np.array([lookup[b,s] for b,s in zip(d.backbone,d.sample_id)])
for name in ['similarity','contrast']:d['independent_'+name]=3
for judge,g in d.groupby('judge'):
 keep=~np.isin(A['models'],[judge,'qwen122']);arr=A['annotations'][keep];votes=np.stack([(arr==k).sum(axis=0) for k in range(3)]);cons=votes.argmax(0);cons[votes.max(0)<(int(keep.sum())//2+1)]=3
 for j,name in enumerate(['similarity','contrast']):d.loc[g.index,'independent_'+name]=cons[ai[g.index],j]
coverage=d.groupby('judge').agg(records=('sample_id','size'),eligible=('eligibility',lambda x:x.eq('eligible').sum()),valid=('status',lambda x:x.eq('valid').sum()),invalid=('status',lambda x:x.eq('invalid').sum()),assessable=('trace_assessable',lambda x:x.eq('yes').sum()),unassessable=('trace_assessable',lambda x:x.eq('no').sum()),uncertain_assessability=('trace_assessable',lambda x:x.eq('uncertain').sum()),source_capped=('complete_source',lambda x:(~x).sum()),retained=('retained','sum')).reset_index();coverage.to_json(O/'coverage.json',orient='records',indent=2)
# One target-level bootstrap weight shared across all source judges and backbones.
targets=d[['dataset','sample_id']].drop_duplicates().sort_values(['dataset','sample_id']).reset_index(drop=True);ti={(x.dataset,x.sample_id):i for i,x in targets.iterrows()};d['target_i']=[ti[x] for x in zip(d.dataset,d.sample_id)]
B=1000;rng=np.random.default_rng(20261025);mult=np.zeros((len(targets),B),np.float64)
for ds,g in targets.groupby('dataset'):
 n=len(g);mult[g.index]=rng.multinomial(n,np.repeat(1/n,n),size=B).T

def metrics(j):
 # j: 2 x 2 x replicates, original label row, expressed label column
 den=j.sum(axis=1);p=np.divide(j[:,1],den,out=np.full_like(den,np.nan),where=den>0)
 phiden=np.sqrt(np.prod(j.sum(axis=0),axis=0)*np.prod(den,axis=0));phi=np.divide(j[0,0]*j[1,1]-j[0,1]*j[1,0],phiden,out=np.full_like(phiden,np.nan),where=phiden>0)
 return np.stack([100*p[0],100*p[1],100*(p[1]-p[0]),phi])
rows=[];joints={};boots={};within=[]
def compute(mask,name,feature,ci=True,origprefix='original_'):
 valid=mask & d.status.eq('valid') & d.trace_assessable.eq('yes') & d['trace_'+feature].isin(['yes','no']) & d[origprefix+feature].isin([0,1]);g=d.loc[valid].copy();assert len(g)>0
 codes,cells=pd.factorize(pd.MultiIndex.from_frame(g[['judge','dataset','backbone']]),sort=True);nc=len(cells);cats=2*g[origprefix+feature].to_numpy(dtype=int)+g['trace_'+feature].eq('yes').to_numpy(dtype=int)
 mat=csr_matrix((g.weight.to_numpy(),(codes*4+cats,g.target_i.to_numpy())),shape=(nc*4,len(targets)))
 vals=np.asarray(mat.sum(axis=1)).reshape(nc,4,1)
 if ci:vals=np.concatenate([vals,(mat@mult).reshape(nc,4,B)],axis=2)
 den=vals.sum(1,keepdims=True);rates=np.divide(vals,den,out=np.full_like(vals,np.nan),where=den>0)
 # Equal backbones, then datasets within source judge, then source judges.
 grouped=collections.defaultdict(list)
 for i,(judge,ds,bb) in enumerate(cells):grouped[judge,ds].append(i)
 judgevals=collections.defaultdict(list)
 for (judge,ds),inds in grouped.items():judgevals[judge].append(np.nanmean(rates[inds],axis=0))
 joint=np.nanmean([np.nanmean(v,axis=0) for v in judgevals.values()],axis=0).reshape(2,2,-1);metric=metrics(joint)
 row=dict(scope=name,feature=feature,n=len(g),targets=g.sample_id.nunique(),cells=nc,original_no=int(g[origprefix+feature].eq(0).sum()),original_yes=int(g[origprefix+feature].eq(1).sum()))
 for k,key in enumerate(['stated_if_no','stated_if_yes','difference_pp','phi']):
  row[key]=float(metric[k,0])
  if ci:row[key+'_lo'],row[key+'_hi']=np.nanpercentile(metric[k,1:],[2.5,97.5]).tolist()
 if ci:boots[name,feature]=metric[2,1:]
 # Matched within-cell conditional differences, to separate within-cell association from composition.
 cond=rates.reshape(nc,2,2,-1);dd=cond.sum(axis=2);pp=np.divide(cond[:,:,1],dd,out=np.full_like(dd,np.nan),where=dd>0);diff=100*(pp[:,1]-pp[:,0]);support=np.isfinite(diff[:,0]);jj=collections.defaultdict(list)
 for (judge,ds),inds in grouped.items():
  inds=[i for i in inds if support[i]]
  if inds:jj[judge].append(np.nanmean(diff[inds],axis=0))
 if jj:
  bdiff=np.nanmean([np.nanmean(v,axis=0) for v in jj.values()],axis=0);wr=dict(scope=name,feature=feature,matched_cells=int(support.sum()),difference_pp=float(bdiff[0]))
  if ci:wr['lo'],wr['hi']=np.nanpercentile(bdiff[1:],[2.5,97.5]).tolist()
  within.append(wr)
 row['raw_joint']=pd.crosstab(g[origprefix+feature],g['trace_'+feature].eq('yes')).reindex(index=[0,1],columns=[False,True],fill_value=0).to_numpy().tolist()
 rows.append(row);joints[name,feature]=joint[:,:,0].tolist();return row
allmask=pd.Series(True,index=d.index)
for scope,mask in [('All original flips',allmask),('Corrects judge',d.original_helped),('Misleads judge',d.original_harmed),('Original NN answer retained',d.retained),('Source answer probability first',d.probability_first),('Complete source explanation',d.complete_source),('Exclude Qwen122 source judge',d.judge.ne('qwen122'))]:
 for feature in ['similarity','contrast']:compute(mask,scope,feature,ci=scope in ['All original flips','Corrects judge','Misleads judge','Original NN answer retained'])
for feature in ['similarity','contrast']:compute(allmask,'Exclude Qwen122 from original consensus',feature,ci=False,origprefix='independent_')
for dim in ['judge','genre','dataset']:
 for group in sorted(d[dim].unique()):
  for feature in ['similarity','contrast']:compute(d[dim].eq(group),dim+': '+group,feature,ci=dim in ['judge','genre'])
# Explicit retention-sensitive outcome breakdown (group defined by original flip).
for outcome,mask in [('Corrects judge',d.original_helped),('Misleads judge',d.original_harmed)]:
 for feature in ['similarity','contrast']:compute(mask&d.retained,outcome+'; answer retained',feature,ci=True)
interactions=[]
for feature in ['similarity','contrast']:
 for suffix in ['', '; answer retained']:
  x=boots['Corrects judge'+suffix,feature]-boots['Misleads judge'+suffix,feature];lo,hi=np.nanpercentile(x,[2.5,97.5]);rr={r['scope']:r for r in rows if r['feature']==feature};interactions.append(dict(feature=feature,subset='All flips' if not suffix else 'Answer retained',difference_of_differences_pp=rr['Corrects judge'+suffix]['difference_pp']-rr['Misleads judge'+suffix]['difference_pp'],lo=float(lo),hi=float(hi)))
(O/'outcome_interactions.json').write_text(json.dumps(interactions,indent=2));pd.DataFrame(within).to_json(O/'within_cell_sensitivity.json',orient='records',indent=2)
res=pd.DataFrame(rows);res.to_json(O/'associations.json',orient='records',indent=2)
(O/'joint_distributions.json').write_text(json.dumps({s+'|'+f:v for (s,f),v in joints.items()},indent=2))
cols=['judge','backbone','dataset','sample_id','genre','unit_id','eligibility','status','finish_reason','trace_assessable','trace_similarity','trace_contrast','original_similarity','original_contrast','original_control','original_nn','original_helped','original_harmed','weight','score','retained','probability_first','complete_source','independent_similarity','independent_contrast'];d[cols].to_pickle(O/'matched_annotations.pkl')
receipt=dict(records=len(d),unique_annotation_calls=len(a),annotation_call_status=a.status.value_counts().to_dict(),record_annotation_status=d.status.fillna('not_sent').value_counts().to_dict(),assessability=d.trace_assessable.fillna('unavailable').value_counts().to_dict(),distinct_targets=d.sample_id.nunique(),backbone_targets=len(d[['backbone','sample_id']].drop_duplicates()),source_judges=sorted(d.judge.unique()),original_flip_counts=d.outcome.value_counts().to_dict(),unknown_trace_labels={f:d['trace_'+f].fillna('unavailable').value_counts().to_dict() for f in ['similarity','contrast']},bootstrap_replicates=B,bootstrap_seed=20261025,source_sha256={str(p):sha(p) for p in [S/'annotations.jsonl',S/'mapping_PRIVATE.jsonl',P/'mechanisms_all12_20261023/annotations.npz',P/'nn_contrast_extremity_20261024/results/joined.pkl']},method='Design-weighted joint proportions within source judge/dataset/backbone; average backbones, datasets, source judges, then derive conditional rates and phi. 1000 target-ID resamples within dataset shared across judges/backbones. Labels fixed; pointwise intervals. Fixed-50 original direct-judge flips, no probe referral filter. Unknown excluded separately by feature. Source judge excluded from original consensus. One Qwen122 trace labeler; not three.',new_inference_calls=0)
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(res[res.scope.isin(['All original flips','Corrects judge','Misleads judge','Original NN answer retained','Exclude Qwen122 from original consensus'])].drop(columns='raw_joint').to_string(index=False));print(json.dumps(receipt,indent=2))
