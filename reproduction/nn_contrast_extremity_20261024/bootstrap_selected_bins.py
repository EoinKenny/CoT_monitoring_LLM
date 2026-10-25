import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';B=1000;SEED=20261024;RADIUS=15.
full=pd.read_pickle(O/'joined.pkl');full=full[(full.contrast==1)&full.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])].copy().reset_index(drop=True)
assert full.control.between(0,100).all() and full.nn.between(0,100).all()
assert (full.helped.astype(int)+full.harmed.astype(int)+full.unchanged.astype(int)).eq(1).all()
cluster,cids=pd.factorize(full.sample_id,sort=True);full['cluster']=cluster
rng=np.random.default_rng(SEED);mult=np.zeros((B,len(cids)))
for ds,v in full.groupby('dataset'):
 ids=np.unique(cluster[v.index]);mult[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
mult=np.vstack([np.ones((1,len(cids))),mult])
sd=float(full.control.std(ddof=1));low=max(0,50-sd);high=min(100,50+sd)
scopes=[('selected','Judge probability 50–70',full.control.between(50,70))]
old=pd.read_csv(O/'summary.csv');results={};audit={};allbins=[];allcoarse=[];boot={}
for key,title,mask in scopes:
 g=full.loc[mask].copy().reset_index(drop=True);x=100*g.neighbor_probe_probability.to_numpy();w=g.w.to_numpy();h=g.helped.to_numpy(float);a=g.harmed.to_numpy(float);cl=g.cluster.to_numpy()
 ci,cells=pd.factorize(pd.MultiIndex.from_frame(g[['judge','backbone','dataset']]),sort=True);C=len(cells);cf=cells.to_frame(index=False);cf.columns=['judge','backbone','dataset'];jd,jds=pd.factorize(pd.MultiIndex.from_frame(cf[['judge','dataset']]),sort=True);jf=jds.to_frame(index=False);jf.columns=['judge','dataset'];ji,judges=pd.factorize(jf.judge,sort=True);M1=np.eye(len(jds))[jd];M2=np.eye(len(judges))[ji]
 def avg(v,M):
  valid=np.isfinite(v);num=np.nan_to_num(v)@M;den=valid.astype(float)@M
  return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 def macro(num,den):
  v=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
  return 100*np.nanmean(avg(avg(v,M1),M2),axis=1)
 def estimate(k):
  wk=w*k;mat=sparse.csr_matrix((np.r_[wk,wk*h,wk*a],(np.tile(cl,3),np.r_[ci,ci+C,ci+2*C])),shape=(len(cids),3*C));mat.eliminate_zeros()
  z=(mat.T@mult.T).T;den=z[:,:C]
  return macro(z[:,C:2*C],den),macro(z[:,2*C:],den)
 for left in range(0,100,10):
  k=(x>=left)&((x<left+10) if left<90 else x<=100);hh,aa=estimate(k)
  row=dict(scope=key,left=left,right=left+10,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(hh[0]-aa[0]),rows=int(k.sum()),targets=int(len(np.unique(cl[k]))))
  for name,val in [('helped',hh),('harmed',aa),('net',hh-aa)]:
   assert np.isfinite(val).all()
   row[name+'_lo'],row[name+'_hi']=map(float,np.percentile(val[1:],[2.5,97.5]))
  allbins.append(row)
oldbins=pd.read_json(O/'selected_spectrum_bins.json');newbins=pd.DataFrame(allbins)
for col in ['helped','harmed','net','rows']:assert np.allclose(oldbins[col],newbins[col])
(O/'selected_spectrum_bins_ci.json').write_text(json.dumps(allbins,indent=2))
(O/'BIN_CI_VERIFIED.json').write_text(json.dumps(dict(replicates=B,seed=SEED,method='Target-ID cluster bootstrap within dataset; repeats across backbones and judges kept together. Paired net change computed within each replicate. 95% pointwise percentile intervals conditional on selected band; no correction for test selection.',point_estimates_unchanged=True,source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest()),indent=2))
