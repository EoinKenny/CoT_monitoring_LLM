"""Average adjusted contrasts and label-alignment strata; no new inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import ast
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';LAB=['labelled_similarity','important_contrast'];TOX=['aegis','beavertails','toxicchat','toxigen'];ANNOTATORS=['qwen122','gemma','qwen','majority']
s=(R/'analyze.py').read_text();tree=ast.parse(s);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='regress');code=ast.get_source_segment(s,fn)
code=code.replace(' return [dict(term=k,',' rows=[dict(term=k,')
code+='''
 for label,idx,other in [('similarity',0,b),('contrast',1,a)]:
  for subgroup in ['all','aligned','opposite']:
   mask=np.ones(len(g),bool) if subgroup=='all' else al==(1 if subgroup=='aligned' else 0)
   if not mask.any():continue
   v=np.zeros(len(names));v[idx]=1;v[2]=np.average(other[mask],weights=w[mask])
   if alignment_interactions:v[9+idx]=np.average(al[mask],weights=w[mask])
   effect=float(v@coef);err=float(np.sqrt(max(v@cov@v,0)))
   rows.append(dict(term=label+'_average_marginal_'+subgroup,coef=effect,lo=effect-1.96*err,hi=effect+1.96*err,n=int(mask.sum()),clusters=G,cells=C,rank=int(np.linalg.matrix_rank(X))))
 return rows
'''
exec(code)
cols=['policy','model','backbone','dataset','unit_id','cell','cluster','w','weight','gain','helped','harmed','base_correct','base_wrong','label_aligned','true_label','cosine_rank','probe_probability','target_chars','neighbor_chars']+[a+'_'+l for a in ANNOTATORS for l in LAB]
parts=[]
for chunk in pd.read_csv(O/'case_outcomes.csv.gz',usecols=cols,chunksize=100000):parts.append(chunk[chunk.policy.isin(['fixed50','anchor_average'])])
d=pd.concat(parts,ignore_index=True);out=[]
for policy in ['fixed50','anchor_average']:
 for cohort in ['all','toxicity4']:
  v=d[d.policy.eq(policy)];v=v if cohort=='all' else v[v.dataset.isin(TOX)]
  for ann in ANNOTATORS:
   g=v[v[ann+'_'+LAB[0]].isin(['yes','no'])&v[ann+'_'+LAB[1]].isin(['yes','no'])]
   for inter in [False,True]:
    for row in regress(g,ann,inter):
     if 'average_marginal' in row['term']:out.append(dict(policy=policy,cohort=cohort,annotator=ann,spec='alignment_interactions' if inter else 'joint',**row))
  print(policy,cohort,flush=True)
pd.DataFrame(out).to_csv(O/'adjusted_marginal_associations.csv',index=False)
# Conditional rates and majority label prevalence within alignment strata.
rows=[]
for keys,g in d.groupby(['policy','label_aligned','majority_labelled_similarity','majority_important_contrast']):
 w=g.w;rows.append(dict(zip(['policy','aligned','similarity','contrast'],keys),n=len(g),contexts=g.unit_id.nunique(),helped=np.average(g.helped,weights=w),harmed=np.average(g.harmed,weights=w),gain=np.average(g.gain,weights=w),correction_rate=(w*g.helped).sum()/(w*g.base_wrong).sum() if (w*g.base_wrong).sum()>0 else np.nan,damage_rate=(w*g.harmed).sum()/(w*g.base_correct).sum() if (w*g.base_correct).sum()>0 else np.nan))
pd.DataFrame(rows).to_csv(O/'alignment_joint.csv',index=False)
print(pd.DataFrame(out).query("annotator=='majority' and cohort=='all' and spec=='alignment_interactions'").to_string(index=False))
