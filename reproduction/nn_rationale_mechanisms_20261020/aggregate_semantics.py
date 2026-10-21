"""Aggregate full-rationale codes; partial outputs explicitly remain interim."""
from pathlib import Path
import json,gzip,argparse
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent

def weighted_rate(x,col):
 w=x.w.to_numpy(float);a=x[col].to_numpy(float)
 return float(np.sum(w*a)/w.sum()) if len(x) and w.sum()>0 else np.nan

def compare(x,col):
 h=x[x.flip_type=='helped'];b=x[x.flip_type=='harmed']
 rh,rb=weighted_rate(h,col),weighted_rate(b,col)
 if len(h) and len(b):
  influence=pd.concat([pd.DataFrame({'cluster':h.cluster,'z':h.w*(h[col].astype(float)-rh)/h.w.sum()}),
                       pd.DataFrame({'cluster':b.cluster,'z':-b.w*(b[col].astype(float)-rb)/b.w.sum()})])
  z=influence.groupby('cluster').z.sum().to_numpy()
  se=np.sqrt(np.sum(z*z)*len(z)/max(1,len(z)-1))
 else:se=np.nan
 return dict(helped_n=len(h),harmed_n=len(b),helped_positive=int(h[col].sum()),harmed_positive=int(b[col].sum()),
  helped_rate=rh,harmed_rate=rb,delta_pp=100*(rh-rb),lower_pp=100*(rh-rb-1.96*se),upper_pp=100*(rh-rb+1.96*se))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 status=json.load(open(R/'results/READING_STATUS.json'))
 if not args.partial:
  assert status['remaining_contrast']==0,'Contrast census not finished'
  assert status['remaining_similarity_harm']==0,'Similarity failure census not finished'
 out=R/'results'/('interim' if args.partial else 'semantic_census');out.mkdir(exist_ok=True)
 c=pd.read_json(R/'results/coded_trace_metadata.jsonl.gz',lines=True)
 m=pd.read_csv(R/'corpus/metadata.csv.gz',usecols=['trace_id','w','cell'])
 c=c.merge(m,on='trace_id',validate='one_to_one');assert c.w.notna().all()
 c['uses_reference']=c.relation.isin(['S','C','B'])
 c['explicit_contrast']=c.relation.isin(['C','B'])
 c['explicit_similarity']=c.relation.isin(['S','B'])
 features=['uses_reference','explicit_contrast','explicit_similarity','score_prose_conflict','mixed_prose']
 for val in ['N','S','C','B','U']:
  col='relation_'+val;c[col]=c.relation==val;features.append(col)
 for val in ['T','L','D','R','Q','U']:
  col='bridge_'+val;c[col]=c.bridge==val;features.append(col)
 for val in ['LABEL','INTENT','BEHAVIOUR','SEVERITY','CONTEXT','FACT','TOPIC','OTHER']:
        col='axis_'+val;c[col]=c['axes'].map(lambda z:val in z);features.append(col)
 def scopes(x):
  yield 'all10',x
  for f,y in x.groupby('family'):yield f,y
  for ds,y in x.groupby('dataset'):yield ds,y
 full=pd.read_csv(R/'results/coverage_ledger.csv.gz')
 ctr=full[full.requested_census & full.available]
 coverage=ctr.groupby('dataset').agg(n=('trace_id','size'),read=('read_complete','sum'))
 coverage['complete']=coverage.n==coverage.read;coverage.to_csv(out/'coverage.csv')
 comparisons=[];dist=[]
 for subset,base in [('contrast',c[c.requested_census]),('similarity_harm',c[c.similarity_harm])]:
  for scope,x in scopes(base):
   complete=bool(coverage.loc[x.dataset.unique(),'complete'].all()) if subset=='contrast' else bool(full.loc[full.similarity_harm & full.available,'read_complete'].all())
   for judge in ['all']+sorted(x.judge.unique()):
    q=x if judge=='all' else x[x.judge==judge]
    for retained in ['all','retained','changed']:
     z=q if retained=='all' else q[q.retains_original_nn==(retained=='retained')]
     if not len(z):continue
     for f in features:
      dist.append(dict(subset=subset,scope=scope,judge=judge,retention=retained,feature=f,n=len(z),positive=int(z[f].sum()),weighted_rate=weighted_rate(z,f),scope_complete=complete))
      if subset=='contrast':
       comparisons.append(dict(scope=scope,judge=judge,retention=retained,feature=f,scope_complete=complete,**compare(z,f)))
 pd.DataFrame(comparisons).to_csv(out/'helpful_harmful_codes.csv',index=False)
 pd.DataFrame(dist).to_csv(out/'code_distributions.csv',index=False)
 # Conflicts only among prose with an assessable explicit class; do not equate U with agreement.
 conf=[]
 for subset,base in [('contrast',c[c.requested_census]),('similarity_harm',c[c.similarity_harm])]:
  for scope,x in scopes(base):
   for judge in ['all']+sorted(x.judge.unique()):
    zz=x if judge=='all' else x[x.judge==judge]
    for retained in ['all','retained','changed']:
     z=zz if retained=='all' else zz[zz.retains_original_nn==(retained=='retained')]
     for exclude50 in [False,True]:
      q=z[z.prose_assessable & z.score_valid]
      if exclude50:q=q[q.reason_probability!=50]
      for outcome in ['all','helped','harmed']:
       q2=q if outcome=='all' else q[q.flip_type==outcome]
       if not len(q2):continue
       conf.append(dict(subset=subset,scope=scope,judge=judge,retention=retained,
         original_outcome=outcome,exclude50=exclude50,all_n=len(z),assessable_n=len(q2),
         conflict_n=int(q2.score_prose_conflict.sum()),conflict_rate=weighted_rate(q2,'score_prose_conflict'),
         mixed_n=int(z.mixed_prose.sum()),unclear_n=int((z.conclusion=='U').sum())))
 pd.DataFrame(conf).to_csv(out/'score_prose_conflicts.csv',index=False)
 (out/'METHOD.json').write_text(json.dumps(dict(interim=args.partial,read_unique=status['unique_texts_read'],remaining_contrast=status['remaining_contrast'],
  comparisons='Expressed codes among original helpful versus original harmful fixed50 flips. New answer retention shown separately. No private reasoning or causal inference.',
  weights='Original population inverse-inclusion/equal evaluation-cell weights, conditioned on specified subgroup.',
  uncertainty='Target-cluster delta-method conditional on frozen data/codes, excludes coding and model-selection uncertainty.',
  similarity='Census covers similarity harmful outcomes. Do not compare this failure-enriched corpus with all helpful similarity cases; use separately balanced context audit for that contrast.',
  conflict='Main conflict table conditions on clear positive/negative prose and valid scored answer; M/U retained separately.'),indent=2)+'\n')
 print('Wrote',out,'with',len(c),'coded trace rows; partial=',args.partial)
if __name__=='__main__':main()
