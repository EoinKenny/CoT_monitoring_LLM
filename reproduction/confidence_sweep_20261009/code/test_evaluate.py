import numpy as np
from evaluate import grid,choose,bootstrap,boot_metrics
rows=[dict(probe_probability=s,true_label=y,inclusion_probability=p,stratum=f'{y}:{p}') for s,y,p in [(.1,1,.5),(.3,0,.5),(.6,1,1),(.9,0,1)]]
ts=np.array([0,.1,.3,.6,.9,1.01]);prob=np.array([10,50,80,100]);pop={'n':6,'positive':3};g=grid(rows,prob,ts,pop)
for cut in range(102):
 for k,t in enumerate(ts):
  tp=sum(1/r['inclusion_probability'] for r,p in zip(rows,prob) if r['true_label']==1 and r['probe_probability']>=t and p>=cut)
  fp=sum(1/r['inclusion_probability'] for r,p in zip(rows,prob) if r['true_label']==0 and r['probe_probability']>=t and p>=cut)
  assert g['tp'][cut,k]==tp and g['fp'][cut,k]==fp
  assert abs(g['f1'][cut,k]-2*tp/(3+tp+fp))<1e-12
assert choose(np.zeros((102,6)),ts)==(101,5)
assert choose(np.zeros((102,6)),ts,50)==(50,5)
mult=bootstrap(rows,np.random.default_rng(1));assert np.all(mult[:,[2,3]]==1)
a=boot_metrics(mult,rows,np.array([1,0,1,0]),pop);b=boot_metrics(mult,rows,np.array([1,0,1,0]),pop)
assert np.all(a['f1']-b['f1']==0)
assert np.all(grid(rows,np.zeros(4),ts,pop)['tp'][101]==0)
print('Passed: exhaustive weighted confusion counts, >= boundary semantics, highest-threshold ties, reject-all, census-fixed bootstrap, matched identical-condition differences.')
