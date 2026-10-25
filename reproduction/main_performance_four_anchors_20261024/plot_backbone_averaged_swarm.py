import os
os.environ['MPLBACKEND']='Agg'
import csv,collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'results/cells.csv').open()))
conditions=['baseline','saliency','nearest','nearest2']
labels=['No explanation','Saliency','1-NN','2-NN']
colors=['#63758b','#cc9630','#8562aa','#247a59']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
for ax,policy,title in zip(axs,['fixed50','tuned_shared'],['Fixed probe 50 / judge 50','Validation F1 tuning: four shared anchors']):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['policy']==policy:groups[r['model'],r['dataset'],r['condition']].append(float(r['f1'])*100)
 assert len(groups)==480
 for i,(c,label,color) in enumerate(zip(conditions,labels,colors)):
  keys=sorted(k for k in groups if k[2]==c)
  assert len(keys)==120
  assert all(len(groups[k])==(5 if policy=='fixed50' else 20) for k in keys)
  v=np.array([np.mean(groups[k]) for k in keys])
  jitter=np.random.default_rng(20261024).uniform(-.25,.25,120)
  ax.scatter(i+jitter,v,s=19,alpha=.38,color=color,edgecolors='none')
  ax.plot([i-.24,i+.24],[v.mean()]*2,color=color,lw=4)
  ax.text(i,v.mean()+2,f'{v.mean():.2f}',ha='center',color=color,weight='bold')
 ax.set_title(title,fontsize=12);ax.set_xticks(range(4));ax.set_xticklabels(labels)
 ax.set_ylim(0,103);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Mean test F1 (%)')
fig.suptitle('12 judges × 10 datasets · averaged over five backbones',fontsize=15)
fig.text(.5,.015,'Each dot: one judge–dataset pair. Tuned dots also average all four shared-threshold anchors.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.94])
for ext in ['png','pdf']:fig.savefig(R/'breakdowns'/('f1_swarm_backbone_averaged.'+ext),dpi=180,bbox_inches='tight')
print('Verified 120 dots per condition per policy; five-backbone means.')
