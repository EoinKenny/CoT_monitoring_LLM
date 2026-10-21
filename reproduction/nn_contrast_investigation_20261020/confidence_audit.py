"""Check whether contrast effects are predominantly marginal changes around judge50."""
from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parent;O=R/'results';T=['aegis','beavertails','toxicchat','toxigen']
cols=['policy','model','backbone','dataset','cell','unit_id','cluster','w','weight','control_score','nearest_score','majority_important_contrast','aligned','helped','harmed','gain','outcome','away','toward','probe_margin_rank']
d=pd.concat([x[x.policy.eq('judge50_direct')] for x in pd.read_csv(O/'cases.csv.gz',usecols=cols,chunksize=100000)],ignore_index=True)
def stats(g):
 w=g.w.to_numpy();a=g.control_score.to_numpy();b=g.nearest_score.to_numpy();return dict(n=len(g),contexts=g.unit_id.nunique(),judge_high_confidence=np.average((a<=10)|(a>=90),weights=w),judge_within10=np.average((a>=40)&(a<=60),weights=w),probe_nearest_quintile=np.average(g.probe_margin_rank<=.2,weights=w),mean_absolute_shift=np.average(abs(b-a),weights=w),fifty_tie=np.average((a==50)|(b==50),weights=w),endpoint_to_opposite_endpoint=np.average(((a==0)&(b==100))|((a==100)&(b==0)),weights=w))
rows=[];sens=[]
for co,z in [('all10',d),('toxicity4',d[d.dataset.isin(T)])]:
 for c,h in z.groupby('majority_important_contrast'):
  for out,g in [('all',h)]+list(h.groupby('outcome')):rows.append(dict(cohort=co,contrast=c,outcome=out,**stats(g)))
 for cut in [25,40,50,60,75]:
  # Fixed descriptive cutoffs only. No test-based cutoff selection.
  y=z.copy();origbase=y.control_score>=cut;pred=y.nearest_score>=cut
  # Original helped/harmed identify original truth: gain and outcome are retained only at50, so recover gold from source instead below.
  # Recover actual target labels from verified case source via a keyed merge, not scores.
  truth=pd.read_csv(R.parent/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz',usecols=['unit_id','true_label'])
  y=y.merge(truth,on='unit_id',validate='many_to_one');a=y.control_score.to_numpy()>=cut;b=y.nearest_score.to_numpy()>=cut;gold=y.true_label.to_numpy();y['harm']=(a==gold)&(b!=gold);y['help']=(a!=gold)&(b==gold)
  for scope,h in [('all',y),('exclude_exact50',y[y.control_score.ne(50)&y.nearest_score.ne(50)])]:
   for c,g in h.groupby('majority_important_contrast'):
    gh=g[g.harm];den=gh.w.sum();sens.append(dict(cohort=co,cutoff=cut,scope=scope,contrast=c,n=len(g),net_accuracy=np.average(g.help.astype(int)-g.harm.astype(int),weights=g.w),harm_toward=np.average(gh.toward,weights=gh.w) if den else np.nan,harm_n=len(gh)))
pd.DataFrame(rows).to_csv(O/'confidence_summary.csv',index=False);pd.DataFrame(sens).to_csv(O/'fixed_cutoff_sensitivity.csv',index=False)
print(pd.DataFrame(rows).query("contrast=='yes'").round(4).to_string(index=False));print(pd.DataFrame(sens).query("contrast=='yes' and scope=='all'").round(4).to_string(index=False))
