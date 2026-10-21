"""Paired group bootstrap for the exploratory FAVA training-method contrast."""
from pathlib import Path
import json,numpy as np
R=Path('/home/ubuntu/experiments/training-balance-comparison-20261004')
out=json.loads((R/'RESULTS.json').read_text());rows=json.loads((R/'fava_selected.json').read_text());ix=[i for i,r in enumerate(rows) if r['split']=='test'];groups=sorted({rows[i]['group'] for i in ix});gid={g:i for i,g in enumerate(groups)}
counts=np.zeros((len(groups),2,3),dtype=np.int64)
for m,name in enumerate(['undersampled_weighted','full_weighted']):
 p=np.load(R/(name+'_probabilities.npy'));t=out['results'][name]['threshold']
 for i in ix:
  y=rows[i]['label'];pred=int(p[i]>=t);g=gid[rows[i]['group']]
  if y and pred:counts[g,m,0]+=1
  elif pred:counts[g,m,1]+=1
  elif y:counts[g,m,2]+=1
rng=np.random.default_rng(20261004);delta=[]
for _ in range(2000):
 c=counts[rng.integers(0,len(groups),len(groups))].sum(axis=0);f=2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1);delta.append(float(f[1]-f[0]))
report=dict(full_minus_undersampled_f1=out['results']['full_weighted']['test']['f1']-out['results']['undersampled_weighted']['test']['f1'],paired_group_bootstrap_ci95=np.quantile(delta,[.025,.975]).tolist(),bootstrap_seed=20261004,replicates=2000,test_groups=len(groups),scope='Conditional on fixed fitted probes and this test pool. Groups follow source/reference grouping. Does not correct for choosing the training method using the test result.')
(R/'COMPARISON_UNCERTAINTY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
(R/'SELECTED_TRAINING_POLICY.json').write_text(json.dumps(dict(selected=out['winner_by_test_f1'],class_weight='balanced',undersample_training=out['winner_by_test_f1']=='undersampled_weighted',basis='User-requested exploratory FAVA/Llama-1B test-F1 comparison; individual thresholds tuned on validation only. Generalisation across datasets/backbones not established.',results=out['results'],uncertainty=report),indent=2))
