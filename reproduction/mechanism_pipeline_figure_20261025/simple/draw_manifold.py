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
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,7.3));ax=fig.add_axes([.025,.04,.95,.92]);ax.set_xlim(0,100);ax.set_ylim(0,60);ax.axis('off')
ink='#213849';green='#257d61';red='#c7736b';blue='#7e9fbb';muted='#677887'
def txt(x,y,s,size=12,color=ink,weight='normal',ha='left',va='center'):ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.32,zorder=8)
def box(x,y,w,h,fc,ec='#dbe4e9',lw=1):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.4,rounding_size=1',facecolor=fc,edgecolor=ec,lw=lw,zorder=2))
def arrow(x,y,X,Y,c=muted,rad=0,ls='-'):
 ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=16,color=c,lw=1.6,connectionstyle=f'arc3,rad={rad}',linestyle=ls,zorder=7))
txt(0,58,'A nearby labelled example can change the judge’s decision',20,weight='bold')
txt(0,54.8,'ToxiGen · Llama 1B representations · Claude Sonnet 4.5 judge',11,color=muted)
# The representation manifold is the main visual, not a small pipeline node.
box(.5,10,64,41,'#f6f8fa')
txt(2.3,48.5,'Training examples in activation space',13,weight='bold')
rng=np.random.default_rng(42)
t=np.linspace(0,1,240);xx=5+53*t;yy=29+8*np.sin(1.45*np.pi*t-.7)
ax.plot(xx,yy,color='#dbe3e8',lw=48,alpha=.28,zorder=3,solid_capstyle='round')
xxp=5+53*rng.uniform(0,1,140);yyp=29+8*np.sin(1.45*np.pi*(xxp-5)/53-.7)+rng.normal(0,2.3,len(xxp))
# Keep the highlighted neighbour visibly nearest in the schematic.
keep=(xxp-37.8)**2 + (.68*(yyp-32.4))**2 > 5.0**2
xxp,yyp=xxp[keep],yyp[keep]
col=np.where(xxp<31,blue,red)
ax.scatter(xxp,yyp,s=30,c=col,alpha=.68,edgecolors='white',linewidths=.35,zorder=4)
# Illustrative linear probe boundary; not a fitted 2-D boundary.
ax.plot([30,30],[18,42],ls=(0,(4,4)),color='#98a6b1',lw=1.4,zorder=5)
txt(30,43.3,'Probe boundary (schematic)',9.5,color=muted,ha='center')
# Star and its nearest training point on the positive side of the probe boundary.
tx,ty=37.8,32.4;nx,ny=42.0,30.8
# Clear dots in a small local region for unambiguous highlighted pair.
ax.scatter([tx],[ty],s=390,marker='*',c=ink,edgecolors='white',linewidths=1,zorder=7)
ax.scatter([nx],[ny],s=230,c=red,edgecolors=green,linewidths=2.8,zorder=7)
ax.plot([tx,nx],[ty,ny],color=green,lw=2.4,zorder=6)
# Small labels, outside dense patch.
txt(35,39.4,'TEST INSTANCE',10,weight='bold');arrow(39,37.8,tx,ty+1,ink)
txt(46.5,25.5,'NEAREST\nNEIGHBOUR',10,color=green,weight='bold');arrow(46.8,27.6,nx+.5,ny-.5,green)
txt(36,19.4,'Probe: 74% toxic → flagged',11,weight='bold')
txt(36,16.7,'Retrieve the closest training example',9.8,color=green)
ax.scatter([3.5,18],[13.6,13.6],s=40,c=[blue,red],zorder=6)
txt(5,13.6,'Non-toxic label',9.5,color=muted);txt(19.5,13.6,'Toxic label',9.5,color=muted)
# Judge stays compact and downstream from retrieval.
arrow(65,32,70,32,green)
box(71,15,28,32,'#ffffff')
txt(73,44,'LLM judge',15,weight='bold')
txt(73,40,'Task + target',11,color=muted)
txt(73,36,'15% → non-toxic',17,color='#b85e55',weight='bold')
arrow(84.5,33.4,84.5,29.5,green)
txt(73,26.8,'Add the labelled neighbour',11,color=muted)
txt(73,22.8,'75% → toxic ✓',17,color=green,weight='bold')
txt(73,18.4,'Now matches the dataset label',9.5,color=green)
# Source examples below the manifold, compact and linked by colour.
box(.5,-9.7,48,17.0,'#f3f6f8');box(51,-9.7,48,17.0,'#edf6f1')
txt(2.4,5.2,'★ TARGET · dataset label: toxic',10.5,weight='bold')
txt(2.4,2.6,textwrap.fill(ex['context']['target_messages'][0]['content'],width=65),11.5,va='top')
txt(53,5.2,'○ NEIGHBOUR · training label: toxic · probe: toxic',10.5,color=green,weight='bold')
txt(53,2.6,textwrap.fill(ex['context']['neighbor_messages'][0]['content'],width=65),11.5,va='top')
# Expand canvas to accommodate exact source text and concise technical note.
ax.set_ylim(-15,60)
txt(50,-12.4,'Schematic projection, not measured geometry. Scores are P(toxic); cutoff 50. Before/after are separate prompts.',9,color=muted,ha='center')
for ext in ['png','pdf','svg']:fig.savefig(R/f'similarity_manifold.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
plt.close(fig)
(R/'manifold.html').write_text('<!doctype html><meta charset="utf-8"><title>Similarity: manifold view</title><style>body{font:17px system-ui;padding:20px}img{width:100%}</style><img src="similarity_manifold.png"><p><a href="similarity_manifold.pdf">PDF</a> · <a href="similarity_manifold.svg">Editable SVG</a></p>')
