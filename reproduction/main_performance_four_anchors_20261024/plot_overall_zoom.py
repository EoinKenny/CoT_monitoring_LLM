import os
os.environ['MPLBACKEND']='Agg'
import csv,collections,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; O=R/'breakdowns'
conds=['baseline','saliency','nearest','nearest2']
labels=['No explanation','Saliency','1-NN','2-NN']
colors=['#63758b','#cc9630','#8562aa','#247a59']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
coords=list(csv.DictReader((O/'curve_coordinates.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(13,5.8),sharex=True,sharey=True)
for ax,policy,title in zip(axs,['fixed','tuned'],['Probe fixed at 50; judge cutoff swept','Validation-selected probe + judge thresholds']):
 for c,label,color in zip(conds,labels,colors):
  rows=sorted([r for r in coords if r['dimension']=='overall' and r['policy']==policy and r['condition']==c],key=lambda r:int(r['operating_index']))
  xy=np.array([[float(r['mean_test_recall']),float(r['mean_test_precision'])] for r in rows])*100
  ax.plot(xy[:,0],xy[:,1],color=color,lw=2.3,label=label)
  marker=50 if policy=='fixed' else 20
  ax.scatter(*xy[marker],s=62,color=color,edgecolors='white',linewidth=.8,zorder=5)
 ax.set_xlim(55,90);ax.set_ylim(80,96);ax.set_xticks(range(55,91,5));ax.set_yticks(range(80,97,2))
 ax.set_title(title,fontsize=12);ax.set_xlabel('Mean test recall (%)');ax.grid(alpha=.15)
axs[0].set_ylabel('Mean test precision (%)')
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.04),ncol=4,frameon=False)
fig.suptitle('Overall precision–recall trade-offs · zoomed view',fontsize=16)
fig.text(.5,.01,'Dots: 50/50 (left); validation F1 optimum, averaged over four shared anchors (right).',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.12,1,.93])
for ext in ['png','pdf']:fig.savefig(O/('overall_zoom.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
rows=list(csv.DictReader((R/'results/cells.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(13,5.8),sharey=True);clipped={}
for ax,policy,title in zip(axs,['fixed50','tuned_shared'],['Fixed probe 50 / judge 50','Validation F1 tuning: four shared anchors']):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['policy']==policy:groups[r['model'],r['dataset'],r['condition']].append(float(r['f1'])*100)
 clipped[policy]={}
 for i,(c,label,color) in enumerate(zip(conds,labels,colors)):
  keys=sorted(k for k in groups if k[2]==c);assert len(keys)==120
  assert all(len(groups[k])==(5 if policy=='fixed50' else 20) for k in keys)
  v=np.array([np.mean(groups[k]) for k in keys])
  jitter=np.random.default_rng(20261024).uniform(-.25,.25,len(v))
  ax.scatter(i+jitter,v,s=24,alpha=.45,color=color,edgecolors='none')
  ax.plot([i-.24,i+.24],[v.mean()]*2,color=color,lw=4)
  ax.text(i,v.mean()+.8,f'{v.mean():.2f}',ha='center',color=color,weight='bold')
  clipped[policy][c]={'below_65':int((v<65).sum()),'above_95':int((v>95).sum()),'total':120,'mean_all_points':float(v.mean())}
 ax.set_title(title,fontsize=12);ax.set_xticks(range(4));ax.set_xticklabels(labels)
 ax.set_ylim(65,95);ax.set_yticks(range(65,96,5));ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Mean test F1 (%)')
fig.suptitle('F1 distributions · averaged over five backbones',fontsize=16)
fig.text(.5,.025,'Display restricted to 65–95%. Mean bars include all 120 judge–dataset pairs, including points outside the view.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.08,1,.93])
for ext in ['png','pdf']:fig.savefig(O/('f1_swarm_backbone_averaged_zoom.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
receipt={'display_only':True,'pr_axes':{'recall':[55,90],'precision':[80,96]},'swarm_y':[65,95],'clipped_points':clipped,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'curve_coordinates.csv',R/'results/cells.csv']}}
(O/'ZOOM_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
(O/'overall_zoom.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Experiment 1: zoomed overall plots</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#182738}img{width:100%}a{color:#265e9d}</style><h1>Experiment 1: zoomed overall results</h1><p>All 12 judges and 10 datasets, averaged over five backbones. Validation tuning averages four shared anchors: control, saliency, 1NN and 2NN. Only display limits have changed; all data contribute to the averages.</p><h2>Precision–recall curves</h2><img src="overall_zoom.png"><p><a href="overall_zoom.pdf">PDF</a> · <a href="overall.png">Full range</a></p><h2>F1 distributions</h2><img src="f1_swarm_backbone_averaged_zoom.png"><p>Each dot is a judge–dataset pair. Points outside 65–95% are hidden by the axis limits, not excluded from the calculations. High ceiling scores are also outside this view; they are not necessarily statistical outliers.</p><p><a href="f1_swarm_backbone_averaged_zoom.pdf">PDF</a> · <a href="f1_swarm_backbone_averaged.png">Full range</a> · <a href="ZOOM_VERIFIED.json">Display ranges and clipped-point counts</a></p></html>''')
print(json.dumps(receipt['clipped_points']))
