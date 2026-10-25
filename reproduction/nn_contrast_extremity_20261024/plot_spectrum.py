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
g=pd.read_pickle(O/'joined.pkl');g=g[(g.contrast==1)&g.control.between(50,65)&g.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])].copy().reset_index(drop=True)
assert (g.helped.astype(int)+g.harmed.astype(int)+g.unchanged.astype(int)).eq(1).all()
# Same hierarchical averaging as the existing coarse-band analysis.
ck=pd.MultiIndex.from_frame(g[['judge','backbone','dataset']]);ci,cells=pd.factorize(ck,sort=True);C=len(cells);cframe=cells.to_frame(index=False);cframe.columns=["judge","backbone","dataset"];jd,jds=pd.factorize(pd.MultiIndex.from_frame(cframe[['judge','dataset']]),sort=True);jf=jds.to_frame(index=False);jf.columns=["judge","dataset"];ji,judges=pd.factorize(jf['judge'],sort=True)
cellmat=sparse.csr_matrix((np.ones(len(g)),(np.arange(len(g)),ci)),shape=(len(g),C))
M1=np.eye(len(jds))[jd];M2=np.eye(len(judges))[ji]
def avg(v,M):
 valid=np.isfinite(v);num=np.nan_to_num(v)@M;den=valid.astype(float)@M
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
def macro(num,den):
 v=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);v=avg(avg(v,M1),M2)
 return np.nanmean(v,axis=1)*100
# Resample immutable target IDs within dataset; keep repeated backbones/judges together.
cluster,cids=pd.factorize(g.sample_id,sort=True);rng=np.random.default_rng(SEED);mult=np.zeros((B,len(cids)),dtype=np.int32)
for ds,v in g.groupby('dataset'):
 ids=np.unique(cluster[v.index]);mult[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
mult=np.vstack([np.ones((1,len(cids)),dtype=int),mult])[:,cluster]
w=g.w.to_numpy();x=100*g.neighbor_probe_probability.to_numpy();h=g.helped.to_numpy(float);a=g.harmed.to_numpy(float)
def estimate(kernel):
 ww=mult*(w*kernel)[None,:];den=ww@cellmat
 return macro((ww*h)@cellmat,den),macro((ww*a)@cellmat,den)
# Check that the plotting estimator reproduces the previously reported coarse-band numbers.
checks=[]
old=pd.read_csv(O/'summary.csv');old=old[(old.scope=='Four toxicity datasets')&(old.side=='Positive control 50–65')]
for name,k in [('Positive NN (80–100)',x>=80),('Negative NN (0–20)',x<=20)]:
 hh,aa=estimate(k);row=old[old.anchor==name].iloc[0]
 assert np.allclose([hh[0],aa[0]],[row.helped,row.harmed],rtol=0,atol=1e-10)
 checks.append(dict(band=name,helped=float(hh[0]),harmed=float(aa[0])))
curves=[];boot=[]
for center in np.arange(101):
 # Compact tricube kernel: smooth weighting inside a fixed +/-15-point window.
 u=abs(x-center)/RADIUS;k=np.where(u<1,(1-np.minimum(u,1)**3)**3,0)
 hh,aa=estimate(k);nn=hh-aa
 row=dict(score=int(center),helped=float(hh[0]),harmed=float(aa[0]),net=float(nn[0]),rows=int((k>0).sum()),targets=int(len(np.unique(cluster[k>0]))),cells=int(len(np.unique(ci[k>0]))),judges=int(g.loc[k>0,'judge'].nunique()))
 for name,vals in [('helped',hh),('harmed',aa),('net',nn)]:
  row[name+'_lo'],row[name+'_hi']=map(float,np.nanpercentile(vals[1:],[2.5,97.5]))
 curves.append(row);boot.append(np.stack([hh[1:],aa[1:],nn[1:]],axis=0))
c=pd.DataFrame(curves);(O/'spectrum_curve.json').write_text(json.dumps(curves,indent=2));np.savez_compressed(O/'spectrum_bootstrap.npz',draws=np.stack(boot),score=c.score.to_numpy())
# Empirical ten-point bins, for a less-smoothed sensitivity view.
bins=[]
for left in range(0,100,10):
 k=(x>=left)&((x<left+10) if left<90 else (x<=100));hh,aa=estimate(k)
 bins.append(dict(left=left,right=left+10,helped=float(hh[0]),harmed=float(aa[0]),net=float(hh[0]-aa[0]),rows=int(k.sum()),targets=int(len(np.unique(cluster[k])))))
(O/'spectrum_bins.json').write_text(json.dumps(bins,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig=plt.figure(figsize=(11.4,7.0));gs=fig.add_gridspec(2,2,height_ratios=[4,1.1],hspace=.48,wspace=.28)
ax=fig.add_subplot(gs[0,0]);an=fig.add_subplot(gs[0,1]);support=fig.add_subplot(gs[1,:])
colors={'helped':'#258466','harmed':'#c66d66','net':'#5a72a1'}
for key,label in [('helped','Corrects judge'),('harmed','Misleads judge')]:
 ax.plot(c.score.to_numpy(),c[key].to_numpy(),lw=2.3,color=colors[key],label=label);ax.fill_between(c.score.to_numpy(),c[key+'_lo'].to_numpy(),c[key+'_hi'].to_numpy(),color=colors[key],alpha=.14,lw=0)
ax.set(ylabel='Mean outcome rate (%)',ylim=(0,100),xlabel='Neighbour probe probability of positive class (%)')
an.axhline(0,color='#555',lw=1,ls='--');an.plot(c.score.to_numpy(),c.net.to_numpy(),lw=2.3,color=colors['net']);an.fill_between(c.score.to_numpy(),c.net_lo.to_numpy(),c.net_hi.to_numpy(),color=colors['net'],alpha=.18,lw=0)
an.set(ylabel='Net accuracy change (percentage points)',xlabel='Neighbour probe probability of positive class (%)')
limit=max(30,10*np.ceil(max(abs(c.net_lo).max(),abs(c.net_hi).max())/10));an.set_ylim(-limit,limit)
ax.legend(frameon=False,loc='upper right');ax.set_title('Corrections versus harmful flips',fontsize=12);an.set_title('Corrections minus harmful flips',fontsize=12)
for aa in [ax,an]:aa.set_xlim(0,100);aa.set_xticks(np.arange(0,101,20));aa.set_box_aspect(1)
support.fill_between(c.score.to_numpy(),0,c.targets.to_numpy(),color='#94a3b8',alpha=.55);support.set(xlim=(0,100),ylim=(0,None),ylabel='Distinct targets\nin local window',xlabel='Neighbour probe probability of positive class (%)');support.set_xticks(np.arange(0,101,10))
fig.suptitle('Contrasting NNs · Original judge probability 50–65',fontsize=14,y=.99)
fig.subplots_adjust(top=.90,bottom=.09,left=.075,right=.97)
for ext in ['png','pdf']:fig.savefig(O/f'contrast_probability_spectrum.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.4));b=pd.DataFrame(bins);xx=((b.left+b.right)/2).to_numpy()
for key in ['helped','harmed']:axes[0].plot(xx,b[key].to_numpy(),marker='o',color=colors[key],label='Corrects judge' if key=='helped' else 'Misleads judge')
axes[1].plot(xx,b.net.to_numpy(),marker='o',color=colors['net']);axes[1].axhline(0,color='#555',ls='--',lw=1)
for aa in axes:aa.set(xlim=(0,100),xlabel='Neighbour probe probability (%)');aa.set_box_aspect(1)
axes[0].set(ylabel='Mean outcome rate (%)',ylim=(0,100));axes[1].set(ylabel='Net accuracy change (pp)');axes[0].legend(frameon=False);fig.tight_layout()
fig.savefig(O/'contrast_probability_bins.png',dpi=180,bbox_inches='tight');plt.close(fig)
receipt=dict(rows=len(g),distinct_target_ids=int(g.sample_id.nunique()),backbone_target_contexts=int(g[['backbone','sample_id']].drop_duplicates().shape[0]),judges=int(g.judge.nunique()),backbones=int(g.backbone.nunique()),datasets=sorted(g.dataset.unique()),filter='Test only; independent contrast consensus yes; original control score 50 through 65 inclusive; valid paired control and 1-NN judgments; no probe-referral restriction.',kernel='Tricube, fixed radius 15 probability points, centres 0..100 inclusive; local weights multiply design weights. Each plotted position uses only observed data in its window; edge windows are one-sided.',aggregation='Design-weighted within judge/backbone/dataset, equal mean over available backbones, then datasets, then judges at each x. Composition can vary along x.',bootstrap=dict(replicates=B,seed=SEED,unit='Immutable target sample_id shared across backbones and judges',strata='dataset',bands='95% pointwise percentile, not simultaneous; does not include label uncertainty or every source of sampling-design uncertainty.'),checks=checks,source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),minimum_targets_per_window=int(c.targets.min()),minimum_judges_per_window=int(c.judges.min()),new_model_calls=0)
(O/'SPECTRUM_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
report='''<!doctype html><meta charset="utf-8"><title>Neighbour probability spectrum</title><style>body{max-width:1200px;margin:35px auto;padding:0 25px;font:17px/1.5 system-ui;color:#263547}img{width:100%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}</style><h1>When does a contrasting NN help or hurt?</h1><p>Same subset as the preceding comparison: Aegis, BeaverTails, ToxicChat and ToxiGen; independently labelled Contrasting NN; original judge probability 50–65 inclusive. These are direct judge outcomes, without restricting the probe referral gate.</p><img src="contrast_probability_spectrum.png"><p>Green: wrong → correct. Red: correct → wrong. Remaining cases are unchanged. The right plot subtracts harmful-flip rate from correction rate. The bottom plot counts distinct target IDs in each local window; these are not independent observations across neighbouring windows.</p><p>The x-axis is the original neighbour probe probability—not cosine similarity, the judge's score on the neighbour, or a measure of semantic severity. The judge saw the neighbour's text, label and binary prediction, not this probability.</p><p>Curves use a fixed ±15-point tricube smoothing window, with design weights and the same hierarchical averaging as the coarse-band analysis. Shading: 95% pointwise target-cluster bootstrap intervals (1,000 draws, stratified by dataset). Group and case composition vary along the curve; this is descriptive and does not isolate a causal contrast mechanism.</p>'''
report+=f'<p>{len(g):,} paired judge outcomes; {g.sample_id.nunique():,} distinct target IDs; {g.judge.nunique()} judges and {g.backbone.nunique()} backbones. Smallest local window: {c.targets.min()} distinct targets.</p>'
report+='<h2>Ten-point bins: sensitivity to smoothing</h2><img src="contrast_probability_bins.png">'+b.to_html(index=False,float_format=lambda x:f'{x:.2f}')+'<p><a href="contrast_probability_spectrum.pdf">Download PDF</a> · <a href="report.html">Earlier coarse-band results and examples</a></p>'
(O/'spectrum.html').write_text(report)
p=O/'report.html';s=p.read_text();link='<p><a href="spectrum.html"><strong>NEW: full 0–100 probability spectrum</strong></a></p>'
if link not in s:s=s.replace('<h1>Do strongly positive and negative neighbours differ?</h1>','<h1>Do strongly positive and negative neighbours differ?</h1>'+link);p.write_text(s)
print(json.dumps(receipt,indent=2));print(c.iloc[::10].to_string(index=False))
