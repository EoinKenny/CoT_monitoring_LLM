import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
O=Path(__file__).resolve().parent/'results'
r=json.loads((O/'joint_spectrum_bins.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
cmap=LinearSegmentedColormap.from_list('net',['#bd4e4b','#fbfbf8','#218262']);cmap.set_bad('#e5e7eb')
lim=json.loads((O/'JOINT_SPECTRUM_VERIFIED.json').read_text())['heatmap_scale_pp']
def matrix(key,n):
 m=np.full((10,10),np.nan)
 for v in r:
  if v['distinct_targets']>=n:m[v['neighbor_bin_left']//10,v['judge_bin_left']//10]=v[key]
 return m
def draw(ax,m,title,cm,norm):
 im=ax.pcolormesh(np.arange(0,101,10),np.arange(0,101,10),np.ma.masked_invalid(m),cmap=cm,norm=norm,edgecolors='white',linewidth=.4)
 for y in range(10):
  for x in range(10):
   val=m[y,x];ax.text(x*10+5,y*10+5,(f'{val:+.1f}' if norm.vmin<0 else f'{val:.1f}') if np.isfinite(val) else '—',ha='center',va='center',fontsize=7.7,color='white' if np.isfinite(val) and (abs(val)>.65*norm.vmax) else '#25333e')
 ax.set(xlim=(0,100),ylim=(0,100),xlabel='Original judge probability (%)',title=title);ax.set_box_aspect(1);ax.set_xticks(range(0,101,20));ax.set_yticks(range(0,101,20));ax.axvline(50,color='#53616c',ls='--',lw=.8);ax.axhline(50,color='#53616c',ls=':',lw=.8)
 return im
norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
fig,axs=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
for ax,n in zip(axs,[20,30]):im=draw(ax,matrix('net',n),f'At least {n} distinct instances per bucket',cmap,norm)
axs[0].set_ylabel('Neighbour probe probability (%)');fig.colorbar(im,ax=axs,label='Corrections minus harmful flips (percentage points)',shrink=.84)
for ext in ['png','pdf']:fig.savefig(O/f'joint_min30_comparison.{ext}',dpi=190)
plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
from matplotlib.colors import Normalize
for ax,key,title,cm in zip(axs,['helped','harmed','net'],['Corrects judge (%)','Misleads judge (%)','Net accuracy gain (pp)'],['Greens','Reds',cmap]):
 if isinstance(cm,str):cm=plt.get_cmap(cm).copy();cm.set_bad('#e5e7eb')
 nn=norm if key=='net' else Normalize(0,60)
 im=draw(ax,matrix(key,30),title,cm,nn);fig.colorbar(im,ax=ax,shrink=.76)
axs[0].set_ylabel('Neighbour probe probability (%)')
for ext in ['png','pdf']:fig.savefig(O/f'joint_min30_outcomes.{ext}',dpi=190)
plt.close(fig)
keep=[v for v in r if v['distinct_targets']>=30];old=[v for v in r if v['distinct_targets']>=20]
stats={'retained_buckets':len(keep),'previous_buckets':len(old),'total_buckets':len(r),'retained_judgments':sum(v['judgments'] for v in keep),'total_judgments':sum(v['judgments'] for v in r),'positive_buckets':sum(v['net']>0 for v in keep),'negative_buckets':sum(v['net']<0 for v in keep),'minimum_distinct_targets':30}
(O/'joint_min30_summary.json').write_text(json.dumps(stats,indent=2))
(O/'joint_min30_bins.json').write_text(json.dumps(keep,indent=2))
body=f'''<!doctype html><meta charset="utf-8"><title>Probability buckets with at least 30 instances</title><style>body{{max-width:1500px;margin:32px auto;padding:0 25px;font:17px/1.5 system-ui;color:#263547}}img{{width:100%}}</style><h1>At least 30 distinct instances per bucket</h1><p>Same four toxicity/harmful-content datasets and independently labelled Contrasting NN examples as before. Both probability axes span 0–100; binary judge decisions use cutoff 50. A bucket counts distinct target IDs, not repeated judgments across models/backbones.</p><p><b>{len(keep)} of {len(r)} buckets retained</b> (previously {len(old)}). These contain {stats['retained_judgments']:,} of {stats['total_judgments']:,} paired judgments. Grey buckets have fewer than 30 distinct targets in the new plots.</p><img src="joint_min30_comparison.png"><p>Identical colour scales and unchanged estimates for retained buckets. Removing sparse buckets does not change the numbers in surviving buckets.</p><img src="joint_min30_outcomes.png"><p>Rates retain the original design weights and equal averaging over available backbones, datasets and judges within each bucket. Corrects and misleads use all paired judgments within each group as denominator, including unchanged decisions. Net gain is corrects minus misleads. These are descriptive estimates; 30 distinct instances does not guarantee precision or equal representation of all models/datasets.</p><p><a href="joint_min30_comparison.pdf">Comparison PDF</a> · <a href="joint_min30_outcomes.pdf">Outcome panels PDF</a> · <a href="joint_spectrum.html">Original report</a></p>'''
(O/'joint_min30.html').write_text(body)
print(json.dumps(stats,indent=2))
print('Judge 50-70 retained bins:')
for v in keep:
 if v['judge_bin_left'] in [50,60]:print(v['judge_bin_left'],v['neighbor_bin_left'],v['distinct_targets'],round(v['net'],2))
