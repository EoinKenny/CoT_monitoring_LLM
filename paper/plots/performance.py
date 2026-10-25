# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
import os
os.environ['MPLBACKEND']='Agg'
import csv,collections,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=OUTPUT;O=OUTPUT
conds=['baseline','saliency','nearest','nearest2'];labels=['No explanation','Saliency','1-NN','2-NN'];colors=['#63758b','#cc9630','#8562aa','#247a59']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
coords=list(csv.DictReader(gzip.open(DATA/'main_curves.csv.gz','rt')));cells=list(csv.DictReader(gzip.open(DATA/'main_cells.csv.gz','rt')))
fig,axes=plt.subplots(1,4,figsize=(23,6.5))
for ax,policy,title in zip(axes[:2],['fixed','tuned'],['(a) Probe fixed at 50; judge swept','(b) Validation-tuned thresholds']):
 for c,label,color in zip(conds,labels,colors):
  rs=sorted([r for r in coords if r['dimension']=='overall' and r['policy']==policy and r['condition']==c],key=lambda r:int(r['operating_index']))
  xy=np.array([[float(r['mean_test_recall']),float(r['mean_test_precision'])] for r in rs])*100
  ax.plot(xy[:,0],xy[:,1],color=color,lw=2.2,label=label)
  i=50 if policy=='fixed' else 20
  ax.scatter(*xy[i],s=48,color=color,edgecolors='white',linewidth=.7,zorder=5)
 ax.set_xlim(55,83 if policy=='fixed' else 90);ax.set_ylim(80,100)
 ax.set_xticks(range(55,81 if policy=='fixed' else 91,5));ax.set_yticks(range(80,101,5))
 ax.set_xlabel('Mean test recall (%)');ax.set_box_aspect(1);ax.grid(False)
axes[0].set_ylabel('Mean test precision (%)');axes[1].tick_params(axis='y',labelleft=False)
for ax,policy,title in zip(axes[2:],['fixed50','tuned_shared'],['(c) Fixed probe 50 / judge 50','(d) Validation F1-optimal thresholds']):
 groups=collections.defaultdict(list)
 for r in cells:
  if r['policy']==policy:groups[r['model'],r['dataset'],r['condition']].append(float(r['f1'])*100)
 for i,(c,color) in enumerate(zip(conds,colors)):
  keys=sorted(k for k in groups if k[2]==c);assert len(keys)==120
  assert all(len(groups[k])==(5 if policy=='fixed50' else 20) for k in keys)
  v=np.array([np.mean(groups[k]) for k in keys]);j=np.random.default_rng(20261024).uniform(-.25,.25,120)
  ax.scatter(i+j,v,s=20,alpha=.45,color=color,edgecolors='none')
  ax.plot([i-.24,i+.24],[v.mean()]*2,color=color,lw=4)
  ax.text(i,v.mean()+.8,f'{v.mean():.2f}',color=color,ha='center',weight='bold')
 ax.set_xticks(range(4));ax.set_xticklabels(['No\nexplanation','Saliency','1-NN','2-NN'])
 ax.set_ylim(65,100);ax.set_yticks(range(65,101,5));ax.set_box_aspect(1);ax.grid(False)
axes[2].set_ylabel('Mean test F1 (%)');axes[3].tick_params(axis='y',labelleft=False)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,frameon=False,fontsize=12)
for ax,letter in zip(axes,'ABCD'):
 ax.text(-.18 if letter in 'AC' else -.06,1.0,'('+letter+')',transform=ax.transAxes,ha='right',va='top',fontsize=18,clip_on=False)
fig.tight_layout(rect=[0,.21,1,.99],w_pad=3)
fig.canvas.draw()
for ax in axes:
 b=ax.get_window_extent();assert abs(b.width/b.height-1)<1e-8
 assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
for ext in ['png','pdf']:fig.savefig(O/('overall_four_panel.'+ext),dpi=200,bbox_inches='tight')
(O/'overall_four_panel.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Experiment 1: four panels</title><style>body{font:16px system-ui;margin:24px;color:#182738}img{width:100%}</style><img src="overall_four_panel.png" alt="Precision recall and F1: fixed and validation-tuned thresholds"><p><a href="overall_four_panel.pdf">Download vector PDF</a> · <a href="overall_four_panel.png">Full-size PNG</a></p></html>''')
print('Four square panels verified; no grid lines; original scores and all-point means retained.')
