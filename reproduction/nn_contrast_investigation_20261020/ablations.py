"""Stratify the existing five-pair component ablations by original NN annotations."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
A=P/'nn_mechanisms_20261015/results/ablation_case_effects.csv.gz'
F=P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz'
a=pd.read_csv(A);f=pd.read_csv(F);k=['model','backbone','dataset','sample_id']
assert not a.duplicated(k+['condition']).any()
wide=a.pivot(index=k,columns='condition',values='score').reset_index()
meta=a[a.condition.eq('nearest')].drop(columns=['condition','score','base_score','correctness_gain','brier_gain','toward_neighbor'])
g=wide.merge(meta,on=k,validate='one_to_one').merge(f[['backbone','sample_id','unit_id','majority_important_contrast','majority_labelled_similarity','unanimous_important_contrast']],on=['backbone','sample_id'],validate='many_to_one')
assert len(g)==25000 and g.majority_important_contrast.notna().all()
# Confirm original baseline and NN remain the same frozen measurements.
s=np.load(P/'api_results_20261015/results/scores.npz')
lookup={(b,si):i for b,si,i in zip(f.backbone,f.sample_id,f.source_index)}
for m,z in g.groupby('model'):
 ix=np.array([lookup[b,si] for b,si in zip(z.backbone,z.sample_id)])
 assert np.array_equal(np.rint(z.baseline*100).astype(int),s[m][ix,0]) and np.array_equal(np.rint(z.nearest*100).astype(int),s[m][ix,2])
g['cell']=g.model+'|'+g.backbone+'|'+g.dataset;g['w']=g.weight/g.groupby('cell').weight.transform('sum')
rows=[]
def summarize(h,cond):
 w=h.w.to_numpy();y=h.true_label.to_numpy();bc=(h.baseline.to_numpy()>=.5)==y;nc=(h.nearest.to_numpy()>=.5)==y
 cc=(h[cond].to_numpy()>=.5)==y;diff=cc.astype(int)-nc.astype(int)
 mu=np.average(diff,weights=w);u=pd.Series(w*(diff-mu)/w.sum()).groupby(h.cluster.to_numpy()).sum().to_numpy();se=np.sqrt(u@u*len(u)/max(len(u)-1,1))
 return dict(n=len(h),contexts=h.unit_id.nunique(),targets=h.cluster.nunique(),accuracy=np.average(cc,weights=w),gain_vs_control=np.average(cc.astype(int)-bc.astype(int),weights=w),delta_vs_full=mu,lo=mu-1.96*se,hi=mu+1.96*se,corrected_full_errors=np.average(~nc&cc,weights=w),undid_full_correct=np.average(nc&~cc,weights=w))
for cohort,z in [('all10',g),('toxicity4',g[g.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])])]:
 for c,h0 in z.groupby('majority_important_contrast'):
  if c not in ['yes','no']:continue
  for al,h in [('all',h0)]+[(str(v),h) for v,h in h0.groupby('label_aligned')]:
   for cond in a.condition.unique():rows.append(dict(cohort=cohort,contrast=c,alignment=al,condition=cond,**summarize(h,cond)))
pd.DataFrame(rows).to_csv(O/'contrast_component_ablations.csv',index=False)
(O/'ABLATIONS_VERIFIED.json').write_text(json.dumps(dict(cases=len(g),pairs=g[['model','backbone']].drop_duplicates().to_dict('records'),baseline_and_full_nn_exactly_match_original=True,annotation_context='Annotation applies to original full neighbour, including when evaluating text/cue removal or random replacement. Not a reannotation of each ablation.',scope='Five previously selected high-NN-benefit backbone/judge pairs. Not all ten judges.',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A,F]}),indent=2))
print(pd.DataFrame(rows).query("contrast=='yes' and alignment=='all'")[['cohort','condition','accuracy','gain_vs_control','delta_vs_full','lo','hi']].round(5).to_string(index=False))
