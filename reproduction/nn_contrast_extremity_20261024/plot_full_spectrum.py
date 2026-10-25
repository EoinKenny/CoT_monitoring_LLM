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
scopes=[('all','All initial decisions (0–100)',np.ones(len(full),dtype=bool)),('negative','Initially negative (0–49)',full.control.lt(50)),('positive','Initially positive (50–100)',full.control.ge(50))]
assert int(scopes[1][2].sum()+scopes[2][2].sum())==len(full)
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
  if key!='all' and band!='All neighbour scores':
   rr=old[(old.scope=='Four toxicity datasets')&(old.side==('Positive control' if key=='positive' else 'Negative control'))&(old.anchor==band)].iloc[0]
   assert np.allclose([hh[0],aa[0]],[rr.helped,rr.harmed],atol=1e-10,rtol=0);checks.append(z)
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
(O/'full_spectrum_curves.json').write_text(json.dumps(results,indent=2));(O/'full_spectrum_bins.json').write_text(json.dumps(allbins,indent=2));(O/'full_spectrum_coarse.json').write_text(json.dumps(allcoarse,indent=2));np.savez_compressed(O/'full_spectrum_bootstrap.npz',**boot)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'helped':'#258466','harmed':'#c66d66','net':'#5a72a1'}
fig,ax=plt.subplots(3,3,figsize=(15.3,10),gridspec_kw={'height_ratios':[3,3,1]},sharex=True)
maxrate=max(max(r['helped_hi'],r['harmed_hi']) for rs in results.values() for r in rs);ratecap=min(100,max(40,10*np.ceil(maxrate/10)))
netlim=max(10,5*np.ceil(max(max(abs(r['net_lo']),abs(r['net_hi'])) for rs in results.values() for r in rs)/5))
for col,(key,title,_) in enumerate(scopes):
 c=pd.DataFrame(results[key]);xx=c.score.to_numpy()
 for metric,label in [('helped','Corrects judge'),('harmed','Misleads judge')]:
  ax[0,col].plot(xx,c[metric].to_numpy(),color=colors[metric],lw=2.1,label=label);ax[0,col].fill_between(xx,c[metric+'_lo'].to_numpy(),c[metric+'_hi'].to_numpy(),color=colors[metric],alpha=.16,lw=0)
 ax[0,col].set_title(title+f'\n{audit[key]["rows"]:,} paired judgments',fontsize=12);ax[0,col].set_ylim(0,ratecap)
 ax[1,col].axhline(0,color='#555',lw=1,ls='--');ax[1,col].plot(xx,c.net.to_numpy(),lw=2.1,color=colors['net']);ax[1,col].fill_between(xx,c.net_lo.to_numpy(),c.net_hi.to_numpy(),color=colors['net'],alpha=.18,lw=0);ax[1,col].set_ylim(-netlim,netlim)
 ax[2,col].fill_between(xx,0,c.targets.to_numpy(),color='#94a3b8',alpha=.6);ax[2,col].set_ylim(0,max(r['targets'] for rs in results.values() for r in rs)*1.05)
 for rr in range(3):ax[rr,col].set_xlim(0,100);ax[rr,col].set_xticks(np.arange(0,101,20))
 for rr in range(2):ax[rr,col].set_box_aspect(.85)
 ax[2,col].set_xlabel('Neighbour probe probability (%)')
 if col:
  for rr in range(3):ax[rr,col].tick_params(labelleft=False)
ax[0,0].set_ylabel('Percentage of cases (%)');ax[1,0].set_ylabel('Net accuracy change (percentage points)');ax[2,0].set_ylabel('Distinct targets\nin local window')
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.975),ncol=2,frameon=False)
fig.suptitle('Contrasting NNs · Full range of original judge scores',y=1.01,fontsize=16)
fig.subplots_adjust(left=.07,right=.98,bottom=.08,top=.90,hspace=.28,wspace=.16)
for ext in ['png','pdf']:fig.savefig(O/f'contrast_full_spectrum.{ext}',dpi=185,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(2,3,figsize=(14,7.7),sharex=True);bb=pd.DataFrame(allbins)
for col,(key,title,_) in enumerate(scopes):
 b=bb[bb.scope==key];x=((b.left+b.right)/2).to_numpy()
 for k in ['helped','harmed']:ax[0,col].plot(x,b[k].to_numpy(),marker='o',color=colors[k],label='Corrects judge' if k=='helped' else 'Misleads judge')
 ax[0,col].set_title(title);ax[0,col].set_ylim(0,ratecap);ax[1,col].plot(x,b.net.to_numpy(),marker='o',color=colors['net']);ax[1,col].axhline(0,color='#555',ls='--',lw=1);ax[1,col].set_ylim(-netlim,netlim)
 for aa in ax[:,col]:aa.set_xlim(0,100)
 ax[1,col].set_xlabel('Neighbour probe probability (%)')
ax[0,0].set_ylabel('Percentage of cases (%)');ax[1,0].set_ylabel('Net accuracy change (pp)');ax[0,0].legend(frameon=False);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'contrast_full_bins.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
receipt=dict(scopes=audit,filter='Test; independent Contrasting NN consensus yes; Aegis, BeaverTails, ToxicChat, ToxiGen; all original judge scores 0–100; paired valid control/1-NN; no probe-referral filter.',partition_verified=True,kernel_radius=RADIUS,bootstrap_replicates=B,seed=SEED,bootstrap='Resample immutable target IDs within dataset, shared across all backbones and judges and scopes; 95% pointwise percentile bands. Not simultaneous bands; labels treated as fixed.',aggregation='Design weights within judge/backbone/dataset, equally average available backbones, datasets, judges at every x. Local group composition varies.',source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),new_model_calls=0)
(O/'FULL_SPECTRUM_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
oldpage=O/'spectrum.html';archive=O/'spectrum_borderline.html'
if not archive.exists():archive.write_bytes(oldpage.read_bytes())
summary=pd.DataFrame(allcoarse)
body='''<!doctype html><meta charset="utf-8"><title>Full-range contrast spectrum</title><style>body{max-width:1450px;margin:35px auto;padding:0 25px;font:17px/1.5 system-ui;color:#263547}img{width:100%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}</style><h1>Contrasting NNs across all original judge probabilities</h1><p><b>The 50–65 restriction has been removed.</b> All original judge scores 0–100 are included. We retain the same four datasets (Aegis, BeaverTails, ToxicChat, ToxiGen), independent Contrasting NN labels, test split and valid paired control/1-NN judgments. This is the full score range within that subset, not all datasets or all NN types.</p><img src="contrast_full_spectrum.png"><p>Columns: all decisions; initially negative (&lt;50); initially positive (≥50). Top: percentage corrected or misled; the remainder is unchanged. Middle: corrections minus harmful flips. Bottom: distinct target IDs within the local window. Identical axes support comparisons.</p><p>The x-axis is the neighbour's probe probability of the positive class. It is not the judge's confidence on the target, cosine similarity, or semantic severity. Judges saw the neighbour's text, class label and binary prediction, not its numerical probability. Initially negative cases can flip upward; initially positive cases can flip downward.</p><p>Tricube smoothing within ±15 probability points; design-weighted within judge/backbone/dataset, then equally averaged over available backbones, datasets and judges. Bootstrap shading is a 95% pointwise interval from 1,000 resamples of target IDs within dataset, keeping repeated target outcomes across backbones and judges together. Cases and available groups change along the curve. This is descriptive, not a causal contrast-mechanism estimate.</p>'''
body+=f'<p>{len(full):,} paired judgments; {full.sample_id.nunique():,} distinct target IDs; {full.judge.nunique()} judges; {full.backbone.nunique()} backbones. Target counts in the two decision subsets overlap because judges can disagree on the same target.</p>'
body+='<h2>Overall and coarse-band summaries</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.2f}')+'<h2>Ten-point bins</h2><img src="contrast_full_bins.png"><p><a href="contrast_full_spectrum.pdf">Full spectrum PDF</a> · <a href="contrast_full_bins.pdf">Binned PDF</a> · <a href="spectrum_borderline.html">Earlier 50–65 subset</a> · <a href="report.html">Original examples</a></p>'
(O/'spectrum.html').write_text(body);(O/'full_spectrum.html').write_text(body)
print(summary.to_string(index=False));print('DONE',len(full),flush=True)
