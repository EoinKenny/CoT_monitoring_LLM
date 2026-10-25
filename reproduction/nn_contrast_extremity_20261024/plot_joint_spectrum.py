import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent;O=R/'results';SEED=20261024
G=pd.read_pickle(O/'joined.pkl');g=G[(G.contrast==1)&G.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])].copy().reset_index(drop=True)
assert g.control.between(0,100).all() and g.nn.between(0,100).all()
assert (g.helped.astype(int)+g.harmed.astype(int)+g.unchanged.astype(int)).eq(1).all()
g['x']=g.control.astype(float);g['y']=100*g.neighbor_probe_probability;g['xb']=np.minimum((g.x/10).astype(int),9);g['yb']=np.minimum((g.y/10).astype(int),9)
def macro(v):
 rows=[]
 for (j,b,d),z in v.groupby(['judge','backbone','dataset']):
  w=z.w.to_numpy();rows.append(dict(judge=j,backbone=b,dataset=d,helped=100*np.average(z.helped,weights=w),harmed=100*np.average(z.harmed,weights=w)))
 if not rows:return dict(helped=np.nan,harmed=np.nan,net=np.nan,unchanged=np.nan,cells=0)
 z=pd.DataFrame(rows);m=z.groupby(['judge','dataset'])[['helped','harmed']].mean().groupby('judge').mean().mean()
 return dict(helped=float(m.helped),harmed=float(m.harmed),net=float(m.helped-m.harmed),unchanged=float(100-m.helped-m.harmed),cells=len(rows))
rows=[];matrix=np.full((10,10),np.nan);nmat=np.zeros((10,10),int);tmat=nmat.copy()
for (xb,yb),v in g.groupby(['xb','yb']):
 m=macro(v);n=len(v);nt=v.sample_id.nunique();row=dict(judge_bin_left=int(xb*10),neighbor_bin_left=int(yb*10),judgments=n,distinct_targets=int(nt),raw_helped=int(v.helped.sum()),raw_harmed=int(v.harmed.sum()),raw_unchanged=int(v.unchanged.sum()),**m);rows.append(row);nmat[yb,xb]=n;tmat[yb,xb]=nt
 if nt>=20:matrix[yb,xb]=m['net']
assert nmat.sum()==len(g)
coarse=[]
for name,v in [('All',g),('Judge 40–60',g[g.control.between(40,60)]),('Judge outside 40–60',g[~g.control.between(40,60)])]:coarse.append(dict(group=name,judgments=len(v),targets=int(v.sample_id.nunique()),**macro(v)))
# Reproduce prior full-range aggregate before plotting.
prior=json.load(open(O/'full_spectrum_coarse.json'));prior=next(r for r in prior if r['scope']=='all' and r['band']=='All neighbour scores')
assert np.allclose([coarse[0][k] for k in ['helped','harmed','net']],[prior[k] for k in ['helped','harmed','net']],atol=1e-10,rtol=0)
(O/'joint_spectrum_bins.json').write_text(json.dumps(rows,indent=2));(O/'joint_spectrum_summary.json').write_text(json.dumps(coarse,indent=2))
rng=np.random.default_rng(SEED)
x=g.x.to_numpy();y=g.y.to_numpy();jx=rng.uniform(np.maximum(0,x-.45),np.minimum(100,x+.45));jy=rng.uniform(np.maximum(0,y-.45),np.minimum(100,y+.45));h=g.helped.to_numpy(bool);a=g.harmed.to_numpy(bool);u=g.unchanged.to_numpy(bool)
colors={'helped':'#238b67','harmed':'#d26661','unchanged':'#9ca6af'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def decorate(ax):
 ax.set(xlim=(0,100),ylim=(0,100),xlabel='Original judge probability of positive class (%)');ax.set_xticks(np.arange(0,101,20));ax.set_yticks(np.arange(0,101,20));ax.set_box_aspect(1);ax.axvline(50,lw=.8,ls='--',color='#53616c',alpha=.65);ax.axhline(50,lw=.8,ls=':',color='#53616c',alpha=.5)
fig,ax=plt.subplots(1,2,figsize=(13.2,6.4),gridspec_kw={'width_ratios':[1,1.12]})
ax[0].scatter(jx[u],jy[u],s=4,color=colors['unchanged'],alpha=.10,lw=0,rasterized=True)
# Interleave helpful and harmful points randomly to avoid painting one category on top.
idx=rng.permutation(np.flatnonzero(h|a));cc=np.where(h[idx],colors['helped'],colors['harmed']);ax[0].scatter(jx[idx],jy[idx],s=10,c=cc,alpha=.36,lw=0,rasterized=True)
ax[0].set_title('Individual paired judgments',fontsize=13);ax[0].set_ylabel('Neighbour probe probability of positive class (%)');decorate(ax[0])
cmap=LinearSegmentedColormap.from_list('harm_help',['#bd4e4b','#fbfbf8','#218262']);cmap.set_bad('#e5e7eb');limit=max(10,5*np.ceil(np.nanmax(abs(matrix))/5));im=ax[1].pcolormesh(np.arange(0,101,10),np.arange(0,101,10),np.ma.masked_invalid(matrix),cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='flat',edgecolors='white',linewidth=.4)
for iy in range(10):
 for ix in range(10):
  if np.isfinite(matrix[iy,ix]):
   col='white' if abs(matrix[iy,ix])>.6*limit else '#25333e';ax[1].text(ix*10+5,iy*10+5,f'{matrix[iy,ix]:+.1f}',ha='center',va='center',fontsize=7.2,color=col)
  else:ax[1].text(ix*10+5,iy*10+5,'—',ha='center',va='center',fontsize=8,color='#8a929a')
ax[1].set_title('Net benefit within 10 × 10 probability bins',fontsize=13);decorate(ax[1]);ax[1].tick_params(labelleft=False)
cb=fig.colorbar(im,ax=ax[1],fraction=.045,pad=.04);cb.set_label('Corrections minus harmful flips (percentage points)')
handles=[Line2D([0],[0],marker='o',ls='',color=colors[k],label=f'{name} ({int(mask.sum()):,})',markersize=6) for k,name,mask in [('helped','Corrects judge',h),('harmed','Misleads judge',a),('unchanged','Unchanged',u)]]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.49,.015));fig.suptitle('Contrasting NNs · All original judge probabilities',y=.98,fontsize=15);fig.subplots_adjust(left=.075,right=.95,bottom=.16,top=.88,wspace=.13)
for ext in ['png','pdf']:fig.savefig(O/f'joint_probability_spectrum.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(14.1,5.0),sharex=True,sharey=True)
for aa,(k,name,mask) in zip(ax,[('helped','Corrects judge',h),('harmed','Misleads judge',a),('unchanged','Unchanged',u)]):
 aa.scatter(jx[mask],jy[mask],s=7 if k!='unchanged' else 4,color=colors[k],alpha=.24 if k!='unchanged' else .10,lw=0,rasterized=True);decorate(aa);aa.set_title(f'{name}\n{mask.sum():,} paired judgments')
ax[0].set_ylabel('Neighbour probe probability (%)');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'joint_probability_outcomes.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
# Support counts for every displayed or suppressed tile.
fig,ax=plt.subplots(figsize=(6,5.5));im=ax.imshow(tmat,origin='lower',extent=(0,100,0,100),aspect='equal',cmap='Blues')
for iy in range(10):
 for ix in range(10):ax.text(ix*10+5,iy*10+5,str(tmat[iy,ix]),ha='center',va='center',fontsize=7,color='white' if tmat[iy,ix]>.65*tmat.max() else '#233')
ax.set(xlabel='Original judge probability (%)',ylabel='Neighbour probe probability (%)',title='Distinct target IDs per probability bin');fig.colorbar(im,ax=ax,label='Distinct target IDs');fig.tight_layout();fig.savefig(O/'joint_probability_support.png',dpi=170);plt.close(fig)
receipt=dict(rows=len(g),targets=int(g.sample_id.nunique()),judges=int(g.judge.nunique()),backbones=int(g.backbone.nunique()),datasets=sorted(g.dataset.unique()),filter='Same four-dataset full-spectrum contrast-labelled test subset, all original judge probabilities, no probe referral restriction.',raw_outcomes=dict(helped=int(h.sum()),harmed=int(a.sum()),unchanged=int(u.sum())),scatter='All paired judgments plotted; tiny uniform display-only jitter within +/-0.45 probability points, clipped bounds 0..100; fixed seed; unchanged behind randomly interleaved helpful/harmful dots. Dots are raw observations, not weighted population estimates.',heatmap='10-point bins: [0,10),...,[90,100]. Weighted outcome rates within judge/backbone/dataset, then equal means across available backbones, datasets, judges. Grey if fewer than 20 distinct target IDs; sparse tiles retained in machine-readable output. Point estimates, no significance claim; cell composition varies.',heatmap_scale_pp=float(limit),shown_bins=int(np.isfinite(matrix).sum()),all_bins=int(len(rows)),seed=SEED,coarse_summary=coarse,source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),new_model_calls=0)
(O/'JOINT_SPECTRUM_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
body='''<!doctype html><meta charset="utf-8"><title>Judge versus neighbour probability</title><style>body{max-width:1400px;margin:32px auto;padding:0 25px;font:17px/1.5 system-ui;color:#263547}img{width:100%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}</style><h1>Judge probability versus neighbour probability</h1><p><b>X:</b> the judge's original control probability on the target. <b>Y:</b> the original probe's probability on the training neighbour. Both axes cover 0–100. Green points are wrong → correct; red points correct → wrong; grey points unchanged. Flips use cutoff 50 for the judge before and after adding 1-NN.</p><p>Same scope as the full-spectrum plots: Aegis, BeaverTails, ToxicChat, ToxiGen; independently labelled Contrasting NN; test cases; valid paired judgments; all original judge probabilities, no probe-referral filter. The neighbour axis is probe confidence, not semantic severity or a judge score on the neighbour.</p><img src="joint_probability_spectrum.png"><p>Left: every paired judgment is plotted, with tiny display-only jitter (at most 0.45 probability points) to reveal overlaps. Raw point density is not sampling-weight adjusted. Right: design-weighted net accuracy gain, equally averaging available backbones, datasets and judges within each tile. Green tiles indicate more corrections; red tiles more harm. Grey tiles have fewer than 20 distinct targets. Tile colours are descriptive point estimates, not significance tests.</p><p>The dashed vertical line marks the judge's classification cutoff. The dotted horizontal line marks the neighbour probe's 50% score, which need not equal its historical frozen binary-prediction threshold. The judge saw neighbour text, label and binary prediction, not this numeric probability.</p><h2>Separate views of the three outcomes</h2><img src="joint_probability_outcomes.png"><p>Separating the colours makes overlap easier to inspect. These plots use exactly the same observations and coordinates.</p>'''
body+=f'<p>{len(g):,} paired judgments, {g.sample_id.nunique():,} distinct target IDs, {g.judge.nunique()} judges and {g.backbone.nunique()} backbones. Repeated targets across judges/backbones are not independent observations.</p>'
body+='<h2>Descriptive comparison near the judge cutoff</h2>'+pd.DataFrame(coarse).to_html(index=False,float_format=lambda z:f'{z:.2f}')+'<p>The 40–60 band is a descriptive illustration of proximity to the classification threshold. It was not tuned. Different regions contain different targets and model/dataset compositions.</p><h2>Sample support</h2><img style="max-width:700px" src="joint_probability_support.png"><p><a href="joint_probability_spectrum.pdf">Download main figure PDF</a> · <a href="joint_probability_outcomes.pdf">Download separate-outcome PDF</a> · <a href="full_spectrum.html">Earlier one-dimensional curves</a></p>'
(O/'joint_spectrum.html').write_text(body)
for name in ['full_spectrum.html','spectrum.html']:
 p=O/name;s=p.read_text();link='<p><a href="joint_spectrum.html"><strong>NEW: judge × neighbour probability plots</strong></a></p>'
 if link not in s:s=s.replace('<h1>Contrasting NNs across all original judge probabilities</h1>','<h1>Contrasting NNs across all original judge probabilities</h1>'+link);p.write_text(s)
print(json.dumps(receipt,indent=2))
