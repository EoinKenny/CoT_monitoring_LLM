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
(O/'selected_spectrum_curves.json').write_text(json.dumps(results,indent=2));(O/'selected_spectrum_bins.json').write_text(json.dumps(allbins,indent=2));(O/'selected_spectrum_coarse.json').write_text(json.dumps(allcoarse,indent=2));np.savez_compressed(O/'selected_spectrum_bootstrap.npz',**boot)


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
c=pd.DataFrame(results['selected']);x=c.score.to_numpy()
fig,ax=plt.subplots(1,2,figsize=(11,5))
for k,label,color in [('helped','Corrects judge','#258466'),('harmed','Misleads judge','#c66d66')]:
 ax[0].plot(x,c[k].to_numpy(),color=color,lw=2.4,label=label);ax[0].fill_between(x,c[k+'_lo'].to_numpy(),c[k+'_hi'].to_numpy(),color=color,alpha=.14)
ax[0].set_ylim(0,60);ax[0].set_ylabel('Percentage of cases (%)');ax[0].legend(frameon=False)
ax[1].plot(x,c.net.to_numpy(),color='#5a72a1',lw=2.4);ax[1].fill_between(x,c.net_lo.to_numpy(),c.net_hi.to_numpy(),color='#5a72a1',alpha=.17);ax[1].axhline(0,color='#777',ls='--',lw=1);ax[1].set_ylabel('Net accuracy change (percentage points)')
for a in ax:a.set_xlim(0,100);a.set_xlabel('Neighbour probe probability (%)');a.set_box_aspect(1)
fig.suptitle('Exploratory: original judge probability 50–70',fontsize=15);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'exploratory_selected_spectrum.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
scan=pd.read_json(O/'exploratory_band_scan.json');top=scan[scan.width==15].sort_values('net',ascending=False)
receipt=dict(selected_band=[50,70],selection='Maximum observed macro net accuracy gain across widths 10,15,20,30 and lower bounds every 5 points; inclusive band endpoints. Selected using test outcomes, exploratory only.',scopes=audit,bootstrap='1000 target-ID clustered resamples within dataset, fixed selected band; pointwise intervals do not account for band selection.',source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),new_model_calls=0)
(O/'EXPLORATORY_SELECTED_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
html=f"""<!doctype html><meta charset='utf-8'><title>Exploratory contrast benefit: judge 50–70</title><style>body{{font:17px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>Exploratory: contrast when the judge starts at 50–70</h1><p><b>Selected for illustration using observed test outcomes.</b> We scanned judge-probability bands of width 10, 15, 20 and 30, starting every five points, and selected the largest mean corrections-minus-harm rate. Endpoints are inclusive. This is not a validation-selected policy or an unbiased estimate of a general benefit.</p><img src='exploratory_selected_spectrum.png'><p>3,143 paired judgments; 1,088 distinct target IDs; 11 judges and five backbones. Four toxicity/harmful-content datasets: Aegis, BeaverTails, ToxicChat and ToxiGen. Independently labelled Contrasting NN cases only; fixed-50 decisions; no probe-referral gate.</p><p>Overall: 33.86% corrected, 13.79% misled, 52.35% unchanged: net +20.07 percentage points. Percentages are design-weighted within judge/backbone/dataset, then equally averaged over available backbones, datasets and judges. They are not raw count percentages. The plot pools original judge probabilities 50–70 and varies the neighbour probe probability.</p><p>Curves use ±15-point tricube windows. Shading is a 95% pointwise target-cluster bootstrap interval, holding the selected band fixed; it does not correct for selecting the band on these outcomes. Groups vary along the curve. The numerical neighbour probability was not shown to the judge.</p><h2>Is 50–65 the strongest band?</h2><p>It is strongest among the 15-point bands checked (+18.49 pp). The broader 50–70 band is strongest across the complete scan (+20.07 pp). This describes the explored grid, not a uniquely optimal threshold.</p><h2>Support and estimates along the curve</h2>{c.iloc[::10][['score','rows','targets','helped','harmed','unchanged','net']].to_html(index=False,float_format=lambda v:f'{v:.2f}')}<h2>All scanned bands</h2>{scan.sort_values('net',ascending=False).to_html(index=False,float_format=lambda v:f'{v:.2f}')}<p><a href='exploratory_selected_spectrum.pdf'>Download PDF</a> · <a href='joint_spectrum.html'>Full two-dimensional view</a></p>"""
(O/'exploratory_selected.html').write_text(html)
print(pd.DataFrame(allcoarse).to_string(index=False))
