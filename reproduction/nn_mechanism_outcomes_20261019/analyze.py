"""Direct outcome-first label prevalence, preserving paired scores and design weights."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';S=R.parent/'nn_mechanism_agreement_20261019/results'
ANN=['qwen122','gemma','qwen','majority'];LAB=['labelled_similarity','important_contrast'];TOX=['aegis','beavertails','toxicchat','toxigen']
cols=['policy','model','backbone','dataset','unit_id','cluster','w','true_label','control_score','nearest_score','helped','harmed','stable_correct','stable_wrong','referred','label_aligned']+[a+'_'+l for a in ANN for l in LAB]
parts=[]
for c in pd.read_csv(S/'case_outcomes.csv.gz',usecols=cols,chunksize=100000):parts.append(c[c.policy.isin(['fixed50','anchor_average'])])
d=pd.concat(parts,ignore_index=True)
# Scores are original frozen probability-only scores, never the annotation output.
z=d[d.policy=='fixed50'].copy();y=z.true_label.to_numpy();bc=(z.control_score.to_numpy()>=50)==y;nc=(z.nearest_score.to_numpy()>=50)==y
z['policy']='judge50_direct';z['helped']=(~bc&nc).astype(float);z['harmed']=(bc&~nc).astype(float);z['stable_correct']=(bc&nc).astype(float);z['stable_wrong']=(~bc&~nc).astype(float)
d=pd.concat([z,d],ignore_index=True);d['unchanged']=d.stable_correct+d.stable_wrong;d['cell']=d.model+'|'+d.backbone+'|'+d.dataset
assert np.allclose(d.helped+d.harmed+d.unchanged,1)
# Cluster influence intervals: all repeats of an exact target text stay together.
def interval(g,influence):
 v=pd.Series(influence).groupby(g.cluster.to_numpy()).sum().to_numpy();n=len(v);se=np.sqrt(v@v*n/max(n-1,1));return 1.96*se
rows=[];contrast=[];conditional=[];eventcounts=[];balanced=[];breakdowns=[]
for policy,v in d.groupby('policy'):
 for cohort,g in [('all',v),('toxicity4',v[v.dataset.isin(TOX)])]:
  for ev in ['helped','harmed','unchanged','stable_correct','stable_wrong']:
   e=g[ev].to_numpy();w=g.w.to_numpy();eventcounts.append(dict(policy=policy,cohort=cohort,outcome=ev,judgment_count=float(e.sum()),contexts=int(g.loc[e>0,'unit_id'].nunique()),weighted_share=float(w@e/w.sum())))
  for ann in ANN:
   for lab in LAB:
    a=g[ann+'_'+lab];valid=a.isin(['yes','no']);h=g[valid];x=h[ann+'_'+lab].eq('yes').to_numpy().astype(float);w=h.w.to_numpy()
    rate={};infl={}
    for ev in ['helped','harmed','unchanged','stable_correct','stable_wrong']:
     e=h[ev].to_numpy();we=w*e;den=we.sum();mu=we@x/den if den else np.nan;err=interval(h,we*(x-mu)/den) if den else np.nan
     px=w@x/w.sum();pe=w@e/w.sum();pxe=w@(x*e)/w.sum();phi=(pxe-px*pe)/np.sqrt(px*(1-px)*pe*(1-pe)) if 0<px<1 and 0<pe<1 else np.nan
     rows.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,outcome=ev,valid_judgments=float(e.sum()),excluded_judgments=float(g.loc[~valid,ev].sum()),label_yes_fraction=mu,lo=mu-err,hi=mu+err,weighted_phi=phi,yes_overall=px))
     rate[ev]=mu;infl[ev]=we*(x-mu)/den
     for level in ['yes','no']:
      m=x==(1 if level=='yes' else 0);conditional.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,level=level,outcome=ev,outcome_fraction=float(w[m]@e[m]/w[m].sum())))
    delta=rate['helped']-rate['harmed'];err=interval(h,infl['helped']-infl['harmed']);contrast.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,helped_minus_harmed=delta,lo=delta-err,hi=delta+err,clusters=h.cluster.nunique()))
    # Equal-cell comparison restricted to cells containing both helpful and harmful events.
    t=pd.DataFrame(dict(cell=h.cell,w=w,x=x,helped=h.helped,harmed=h.harmed))
    pairs=[]
    for cell,c in t.groupby('cell'):
     dh=(c.w*c.helped).sum();dr=(c.w*c.harmed).sum()
     if dh>0 and dr>0:pairs.append(((c.w*c.helped*c.x).sum()/dh,(c.w*c.harmed*c.x).sum()/dr))
    b=np.array(pairs);balanced.append(dict(policy=policy,cohort=cohort,annotator=ann,label=lab,cells=len(b),helped_yes=b[:,0].mean(),harmed_yes=b[:,1].mean(),difference=(b[:,0]-b[:,1]).mean()))
  for dim in ['model','dataset','backbone','label_aligned']:
   for value,h0 in g.groupby(dim):
    for lab in LAB:
     h=h0[h0['majority_'+lab].isin(['yes','no'])];x=h['majority_'+lab].eq('yes').to_numpy();w=h.w.to_numpy()
     for ev in ['helped','harmed','unchanged']:
      e=h[ev].to_numpy();den=w@e
      breakdowns.append(dict(policy=policy,cohort=cohort,dimension=dim,group=value,label=lab,outcome=ev,n=float(e.sum()),label_yes_fraction=float((w*e)@x/den) if den else np.nan))
 print('done',policy,flush=True)
for name,data in [('prevalence_by_outcome',rows),('helpful_vs_harmful',contrast),('outcome_given_label',conditional),('outcome_counts',eventcounts),('equal_cell_sensitivity',balanced),('breakdowns',breakdowns)]:pd.DataFrame(data).to_csv(O/(name+'.csv'),index=False)
# Distribution counts use unique annotation contexts, not repeated outcome judges.
a=pd.read_csv(S/'joined_annotations.csv.gz');dist=[]
for ann in ANN:
 for lab in LAB:
  for value,n in a[ann+'_'+lab].value_counts(dropna=False).items():dist.append(dict(annotator=ann,label=lab,value=value,n=int(n),fraction=n/len(a)))
pd.DataFrame(dist).to_csv(O/'label_distribution.csv',index=False)
j=a.groupby(['majority_labelled_similarity','majority_important_contrast'],dropna=False).size().reset_index(name='n');j['fraction']=j.n/len(a);j.to_csv(O/'joint_distribution.csv',index=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'COMPLETED.json').write_text(json.dumps(dict(time=time.time(),test_contexts=len(a),judge_context_pairs=len(z),source_sha256=sha(S/'case_outcomes.csv.gz'),annotation_sha256=sha(S/'joined_annotations.csv.gz'),script_sha256=sha(Path(__file__)),interpretation='Main: direct original LLM decisions at common judge50, ignoring probe gating, on all annotated cases with original valid paired scores. Pipeline fixed50 and same-threshold three-anchor average are separate sensitivity views. Matched-tuned event fractions represent averaging actual binary events over three anchors, never classifying an averaged decision. Label prevalence conditions on each outcome; inverse-inclusion weights and equal original cell mass retained. Intervals cluster target texts across repeated cases, thresholds held fixed. Uncertain/missing votes excluded explicitly from yes/no association.',new_model_calls=0),indent=2))
print(pd.DataFrame(rows).query("policy=='judge50_direct' and cohort=='all' and annotator=='majority' and outcome in ['helped','harmed','unchanged']").to_string(index=False))
print(pd.DataFrame(contrast).query("policy=='judge50_direct' and cohort=='all'").to_string(index=False))
