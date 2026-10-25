import os
os.environ['MPLCONFIGDIR']='/private/tmp/nn_paper_mpl'
from pathlib import Path
import json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parent
ex=json.loads((R.parents[1]/'mechanism_paper_figures_20261024/panel_examples.json').read_text())[0]
assert ex['choice_id']=='S7' and ex['control']==15 and ex['nn']==75
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(15,5.9));ax=fig.add_axes([.025,.05,.95,.91]);ax.set_xlim(0,100);ax.set_ylim(0,40);ax.axis('off')
ink='#243747';green='#257d61';muted='#647485';red='#b8635b'
def txt(x,y,s,size=12,color=ink,weight='normal',ha='left',va='center'):ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.35)
def box(x,y,w,h,color,edge='#dce3e7'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.4,rounding_size=1',facecolor=color,edgecolor=edge,lw=1))
def arrow(x,y,X,Y):ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=18,lw=1.7,color='#8b9aa6'))
txt(0,38,'A similar labelled example can correct the judge',20,weight='bold')
txt(0,34.8,'One real example · ToxiGen · Claude Sonnet 4.5',11,color=muted)
# Target and probe
box(.5,7,29,24,'#f6f8fa');txt(2,28,'1  Probe flags the target',14,weight='bold')
txt(2,24.8,'TARGET',9.5,color=muted,weight='bold')
txt(2,22.8,textwrap.fill(ex['context']['target_messages'][0]['content'],width=40),12,va='top')
txt(2,11.5,'Probe: 74% toxic → send to judge',11.5,weight='bold');txt(2,8.7,'Dataset label: toxic',10,color=muted)
arrow(30.2,19,34,19)
# NN with small schematic manifold
box(35,7,30,24,'#edf6f1');txt(36.5,28,'2  Add a nearby training example',13.5,weight='bold')
# small manifold along upper band
rng=np.random.default_rng(123);xx=rng.uniform(37,53,24);yy=23.2+.8*np.sin(xx)+rng.normal(0,.55,24)
ax.scatter(xx,yy,s=18,c='#9dbbac',alpha=.65)
ax.scatter([48],[23.3],s=125,marker='*',c=ink,zorder=5);ax.scatter([51],[22.6],s=65,facecolors='none',edgecolors=green,lw=2,zorder=5)
ax.plot([48,51],[23.3,22.6],ls=':',color=ink,lw=1)
txt(54,24,'★ target',9.5);txt(54,21.9,'○ neighbour',9.5,color=green)
txt(36.5,19.2,'NEIGHBOUR',9.5,color=green,weight='bold')
txt(36.5,17.5,textwrap.fill(ex['context']['neighbor_messages'][0]['content'],width=42),11.7,va='top')
txt(36.5,8.7,'Training label: toxic  ·  Probe: toxic',10,color=green,weight='bold')
arrow(65.8,19,69.5,19)
# pairwise outcome
box(70.5,7,29,24,'#fafbfc');txt(72,28,'3  Judge changes its answer',14,weight='bold')
txt(72,24,'Without the neighbour',11,color=muted)
txt(72,20.6,'15% → non-toxic',17,color=red,weight='bold')
arrow(83.9,18.5,83.9,16)
txt(72,13.7,'With the neighbour',11,color=muted)
txt(72,10.4,'75% → toxic ✓',17,color=green,weight='bold')
txt(50,3.1,'The labelled neighbour offers a possible precedent for how this dataset treats related statements.',11.5,ha='center')
txt(50,.6,'Scores are P(toxic); decisions use 50%. Paired prompts, not a continuing conversation. Activation-space sketch is schematic.',9,color=muted,ha='center')
for ext in ['png','pdf','svg']:fig.savefig(R/f'similarity_pipeline.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Similarity pipeline</title><style>body{font:17px system-ui;padding:22px;color:#243747}img{width:100%}</style><img src="similarity_pipeline.png"><p><a href="similarity_pipeline.pdf">PDF</a> · <a href="similarity_pipeline.svg">Editable SVG</a></p>')
