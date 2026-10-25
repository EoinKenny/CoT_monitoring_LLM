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
scopes=[('opposite','Opposite side: neighbour <50',full.control.between(50,70)&full.neighbor_probe_probability.lt(.5)),('same','Same side: neighbour ≥50',full.control.between(50,70)&full.neighbor_probe_probability.ge(.5))]
assert sum(int(m.sum()) for _,_,m in scopes)==int(full.control.between(50,70).sum())
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
 checks=[]
 for band,k in [('All scores on this side',np.ones(len(g)))]:
  hh,aa=estimate(k);nn=hh-aa;z=dict(scope=key,band=band,rows=int((k>0).sum()),helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(nn[0]),net_lo=float(np.percentile(nn[1:],2.5)),net_hi=float(np.percentile(nn[1:],97.5)))
  allcoarse.append(z)
 curves=[];draws=[]
 for center in (range(50) if key=='opposite' else range(50,101)):
  u=abs(x-center)/RADIUS;k=np.where(u<1,(1-np.minimum(u,1)**3)**3,0);hh,aa=estimate(k);nn=hh-aa
  row=dict(score=center,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(nn[0]),rows=int((k>0).sum()),targets=int(len(np.unique(cl[k>0]))),cells=int(len(np.unique(ci[k>0]))),judges=int(g.loc[k>0,'judge'].nunique()))
  for name,val in [('helped',hh),('harmed',aa),('net',nn)]:row[name+'_lo'],row[name+'_hi']=map(float,np.nanpercentile(val[1:],[2.5,97.5]))
  curves.append(row);draws.append(np.stack([hh[1:],aa[1:],nn[1:]]))
 results[key]=curves;boot[key]=np.stack(draws)
 for left in (range(0,50,10) if key=='opposite' else range(50,100,10)):
  k=(x>=left)&((x<left+10) if left<90 else x<=100);hh,aa=estimate(k)
  allbins.append(dict(scope=key,left=left,right=left+10,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(hh[0]-aa[0]),rows=int(k.sum()),targets=int(len(np.unique(cl[k])))))
 audit[key]=dict(rows=len(g),targets=int(g.sample_id.nunique()),contexts=int(g[['backbone','sample_id']].drop_duplicates().shape[0]),judges=int(g.judge.nunique()),backbones=int(g.backbone.nunique()),checks=checks,minimum_window_targets=min(r['targets'] for r in curves))
 print(key,json.dumps(audit[key]),flush=True)
(O/'side_spectrum_curves.json').write_text(json.dumps(results,indent=2));(O/'side_spectrum_bins.json').write_text(json.dumps(allbins,indent=2));(O/'side_spectrum_coarse.json').write_text(json.dumps(allcoarse,indent=2));np.savez_compressed(O/'side_spectrum_bootstrap.npz',**boot)



plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(11,9),sharey='row')
for col,(key,title,_) in enumerate(scopes):
 c=pd.DataFrame(results[key]);x=c.score.to_numpy()
 for k,label,color in [('helped','Corrects judge','#258466'),('harmed','Misleads judge','#c66d66')]:
  ax[0,col].plot(x,c[k].to_numpy(),color=color,lw=2.3,label=label);ax[0,col].fill_between(x,c[k+'_lo'].to_numpy(),c[k+'_hi'].to_numpy(),color=color,alpha=.14)
 ax[0,col].set_title(title+f'\n{audit[key]["rows"]:,} judgments');ax[0,col].set_ylim(0,60)
 ax[1,col].plot(x,c.net.to_numpy(),color='#5a72a1',lw=2.3);ax[1,col].fill_between(x,c.net_lo.to_numpy(),c.net_hi.to_numpy(),color='#5a72a1',alpha=.17);ax[1,col].axhline(0,color='#777',ls='--',lw=1);ax[1,col].set_ylim(-25,40)
 for a in ax[:,col]:a.set_xlim(0,100);a.axvline(50,color='#aaa',ls=':',lw=1);a.set_box_aspect(.8)
 ax[1,col].set_xlabel('Neighbour probe probability (%)')
ax[0,0].set_ylabel('Percentage of cases (%)');ax[1,0].set_ylabel('Net accuracy change (percentage points)');ax[0,0].legend(frameon=False)
fig.suptitle('Exploratory · Original judge probability 50–70',fontsize=15);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'contrast_sides.{ext}',dpi=180,bbox_inches='tight')
summary=pd.DataFrame(allcoarse);print(summary.to_string(index=False))
receipt=dict(scopes=audit,split='Neighbour probe probability <0.5 versus >=0.5; original judge control 50–70 inclusive.',source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),partition_verified=True,bootstrap='1000 target-ID clusters within dataset; pointwise intervals conditional on test-selected band; no selection correction',new_calls=0)
(O/'SIDES_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
html=f"""<!doctype html><meta charset='utf-8'><title>Contrasting NN: which side?</title><style>body{{font:17px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style><h1>Which side does the contrasting neighbour come from?</h1><p>The original judge is positive (probability 50–70). We split neighbours at probe probability 50: <b>opposite-side neighbours &lt;50</b> versus <b>same-side neighbours ≥50</b>. This refers to probability sides, not the neighbour's ground-truth label or its frozen binary prediction.</p><img src='contrast_sides.png'><h2>Average outcomes within each side</h2>{summary.to_html(index=False,float_format=lambda x:f'{x:.2f}')}<p>All rates retain unchanged cases in their denominator. Net gain = corrections minus harmful flips. Each side is independently averaged: design-weighted within judge/backbone/dataset, then equal averages over available backbones, datasets and judges. Group composition differs, so this table need not recombine to the pooled macro estimate.</p><p>Curves use ±15-point tricube windows restricted to the indicated side; no smoothing crosses 50. Both columns have identical axes. Shading: 95% pointwise intervals, 1,000 target-ID bootstrap resamples within dataset, keeping repeated judge/backbone outcomes together.</p><p>Exploratory: the 50–70 judge band was selected using test outcomes. These intervals do not account for that selection. Same four datasets (Aegis, BeaverTails, ToxicChat, ToxiGen), independent Contrasting NN consensus, paired valid test outcomes, fixed-50 judge decisions and no probe-referral filter. Numerical neighbour probability was not shown to judges.</p><p><a href='contrast_sides.pdf'>PDF</a> · <a href='exploratory_selected.html'>Pooled band</a></p>"""
(O/'contrast_sides.html').write_text(html)
