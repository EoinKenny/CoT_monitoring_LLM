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
scopes=[('one_sd',f'50 ± 1 SD: {low:.1f}–{high:.1f}',full.control.between(low,high)),('middle','Judge probability 30–70',full.control.between(30,70)),('central','Judge probability 40–60',full.control.between(40,60))]
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
 for band,k in [('All neighbour scores',np.ones(len(g))),('Negative NN (0–20)',x<=20),('Positive NN (80–100)',x>=80)]:
  hh,aa=estimate(k);nn=hh-aa;z=dict(scope=key,band=band,rows=int((k>0).sum()),helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(nn[0]),net_lo=float(np.percentile(nn[1:],2.5)),net_hi=float(np.percentile(nn[1:],97.5)))
  allcoarse.append(z)
 curves=[];draws=[]
 for center in range(101):
  u=abs(x-center)/RADIUS;k=np.where(u<1,(1-np.minimum(u,1)**3)**3,0);hh,aa=estimate(k);nn=hh-aa
  row=dict(score=center,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(nn[0]),rows=int((k>0).sum()),targets=int(len(np.unique(cl[k>0]))),cells=int(len(np.unique(ci[k>0]))),judges=int(g.loc[k>0,'judge'].nunique()))
  for name,val in [('helped',hh),('harmed',aa),('net',nn)]:row[name+'_lo'],row[name+'_hi']=map(float,np.nanpercentile(val[1:],[2.5,97.5]))
  curves.append(row);draws.append(np.stack([hh[1:],aa[1:],nn[1:]]))
 results[key]=curves;boot[key]=np.stack(draws)
 for left in range(0,100,10):
  k=(x>=left)&((x<left+10) if left<90 else x<=100);hh,aa=estimate(k)
  allbins.append(dict(scope=key,left=left,right=left+10,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(hh[0]-aa[0]),rows=int(k.sum()),targets=int(len(np.unique(cl[k])))))
 audit[key]=dict(rows=len(g),targets=int(g.sample_id.nunique()),contexts=int(g[['backbone','sample_id']].drop_duplicates().shape[0]),judges=int(g.judge.nunique()),backbones=int(g.backbone.nunique()),checks=checks,minimum_window_targets=min(r['targets'] for r in curves))
 print(key,json.dumps(audit[key]),flush=True)
(O/'middle_spectrum_curves.json').write_text(json.dumps(results,indent=2));(O/'middle_spectrum_bins.json').write_text(json.dumps(allbins,indent=2));(O/'middle_spectrum_coarse.json').write_text(json.dumps(allcoarse,indent=2));np.savez_compressed(O/'middle_spectrum_bootstrap.npz',**boot)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'helped':'#258466','harmed':'#c66d66','net':'#5a72a1','unchanged':'#94a3b8'}
fig,ax=plt.subplots(3,3,figsize=(15,11),sharex=True,gridspec_kw={'height_ratios':[3,3,1]})
for col,(key,title,_) in enumerate(scopes):
 c=pd.DataFrame(results[key]);x=c.score.to_numpy()
 for metric,label in [('helped','Corrects judge'),('harmed','Misleads judge'),('unchanged','Unchanged')]:
  ax[0,col].plot(x,c[metric].to_numpy(),color=colors[metric],lw=2,label=label,ls='--' if metric=='unchanged' else '-')
  if metric!='unchanged':ax[0,col].fill_between(x,c[metric+'_lo'].to_numpy(),c[metric+'_hi'].to_numpy(),color=colors[metric],alpha=.13)
 ax[0,col].set_title(title+f'\n{audit[key]["rows"]:,} paired judgments');ax[0,col].set_ylim(0,100)
 ax[1,col].axhline(0,color='#666',ls='--',lw=1);ax[1,col].plot(x,c.net.to_numpy(),color=colors['net'],lw=2)
 ax[1,col].fill_between(x,c.net_lo.to_numpy(),c.net_hi.to_numpy(),color=colors['net'],alpha=.18);ax[1,col].set_ylim(-20,40)
 ax[2,col].fill_between(x,c.targets.to_numpy(),color='#94a3b8',alpha=.6);ax[2,col].set_ylim(0,max(z['targets'] for v in results.values() for z in v)*1.05)
 for rr in range(3):ax[rr,col].set_xlim(0,100)
 ax[2,col].set_xlabel('Neighbour probe probability (%)')
ax[0,0].set_ylabel('Percentage of cases (%)');ax[1,0].set_ylabel('Corrections − harmful flips (pp)');ax[2,0].set_ylabel('Distinct targets')
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False)
fig.tight_layout(rect=[0,0,1,.96])
for ext in ['png','pdf']:fig.savefig(O/f'middle_probability_spectrum.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
receipt=dict(sd=sd,band=[low,high],sd_definition='Unweighted sample standard deviation of original judge scores across all 39429 eligible paired judgments, before central filtering; descriptive band centered on 50, not mean or confidence interval.',scopes=audit,bootstrap_replicates=B,source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),new_model_calls=0)
(O/'MIDDLE_SPECTRUM_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
summary=pd.DataFrame(allcoarse)
html=f"""<!doctype html><meta charset='utf-8'><title>Central judge probability spectrum</title><style>body{{font:17px/1.5 system-ui;max-width:1450px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>Do clear neighbours help uncertain judges?</h1><p>Original judge probabilities are pooled within each central band. The horizontal axis shows the neighbour's probe probability. All panels retain the same four datasets, independently labelled Contrasting NNs, valid paired test outcomes, and fixed-50 judge decisions. There is no probe-referral filter.</p><p>The requested one-standard-deviation band is <b>{low:.2f}–{high:.2f}</b>: the empirical sample SD is {sd:.2f} percentage points across all 39,429 eligible judgments. This is centered on 50, not the empirical mean; it is not a confidence interval. The 30–70 and 40–60 panels are explicit sensitivity views, not optimized cutoffs.</p><img src='middle_probability_spectrum.png'><p>Top: percentages corrected, misled, or unchanged; these sum to 100 at each position. Middle: corrections minus harmful flips, so positive values favour showing the NN. Bottom: distinct target IDs contributing to each local window. Clear neighbours here means extreme probe scores, not an independently measured semantic property.</p><p>Tricube smoothing within ±15 probability points. Design weights within judge/backbone/dataset, then equal averaging over available backbones, datasets and judges. Shading: 95% pointwise intervals from 1,000 target-ID bootstrap resamples within dataset, preserving repeated outcomes across judges/backbones. Group composition varies along each curve. Labels and the observed SD band are held fixed. These are descriptive associations; the neighbour's numerical probe probability was not displayed to the judge.</p><h2>Overall and extreme-neighbour summaries</h2>{summary.to_html(index=False,float_format=lambda v:f'{v:.2f}')}<p><a href='middle_probability_spectrum.pdf'>PDF</a> · <a href='joint_spectrum.html'>Two-dimensional spectrum</a></p>"""
(O/'middle_spectrum.html').write_text(html)
print(summary.to_string(index=False),flush=True)
