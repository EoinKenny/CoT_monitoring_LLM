"""Dataset-level boxplots of missing-output bounds, without midpoint imputation."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = Path(__file__).resolve().parent
with (root/'api_metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
judges = [('claude4.6-sonnet','Claude Sonnet'),('claude4.8-opus','Claude Opus')]
backbones = [('llama','Llama 1B'),('qwen','Qwen 4B'),('gemma','Gemma 31B')]
conditions = ['baseline','saliency','nearest']
colors = ['#3476A8','#DB9147']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,3,figsize=(14,8),sharey=True)
for ri,(judge,judge_name) in enumerate(judges):
    for ci,(backbone,backbone_name) in enumerate(backbones):
        ax = axes[ri,ci]
        for j,condition in enumerate(conditions):
            group = sorted([r for r in rows if (r['judge'],r['backbone'],r['condition'])==(judge,backbone,condition)],key=lambda r:r['dataset'])
            assert len(group)==10 and len({r['dataset'] for r in group})==10
            lower = np.array([float(r['pipeline_f1_lower'])*100 for r in group])
            upper = np.array([float(r['pipeline_f1_upper'])*100 for r in group])
            assert np.all(lower<=upper+1e-9)
            pos = [j-.18,j+.18]
            bp=ax.boxplot([lower,upper],positions=pos,widths=.27,patch_artist=True,showfliers=False,manage_ticks=False,medianprops={'color':'#111111','linewidth':1.7},whiskerprops={'linewidth':1},capprops={'linewidth':1})
            for box,color in zip(bp['boxes'],colors):box.set(facecolor=color,alpha=.28,edgecolor=color,linewidth=1.4)
            jitter=np.linspace(-.055,.055,10)
            for k in range(10):
                ax.plot([pos[0]+jitter[k],pos[1]+jitter[k]],[lower[k],upper[k]],color='#8A9299',alpha=.42,lw=.65,zorder=2)
            for p,values,color in zip(pos,[lower,upper],colors):ax.scatter(p+jitter,values,s=15,color=color,alpha=.85,zorder=3)
        ax.set_title(judge_name+' · '+backbone_name,pad=12,fontsize=12)
        ax.set_xticks(range(3),['No explanation','Saliency','Nearest neighbour'],fontsize=10)
        ax.set_xlim(-.6,2.6);ax.set_ylim(0,102);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.16)
        if ci==0:ax.set_ylabel('Full-test pipeline F1 (×100)')
fig.suptitle('Claude judges: performance across ten datasets',fontsize=18,y=.98)
fig.legend(handles=[Patch(facecolor=colors[0],alpha=.45,label='Lower bound'),Patch(facecolor=colors[1],alpha=.45,label='Upper bound')],loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False)
fig.text(.5,.045,'Each box summarizes 10 dataset scores; dots joined by a line are the same dataset.\nBounds reflect unresolved API responses, not confidence intervals. Boxes show quartiles and median; whiskers use 1.5×IQR.',ha='center',va='center',fontsize=10,color='#444444')
fig.subplots_adjust(top=.85,bottom=.14,wspace=.12,hspace=.30)
fig.savefig(root/'claude_explanation_boxplots.png',dpi=180,facecolor='white')
fig.savefig(root/'claude_explanation_boxplots.pdf',facecolor='white')
print(root/'claude_explanation_boxplots.png')
