# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
import os

from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=OUTPUT;c=pd.read_csv(DATA/'label_cells.csv.gz');p=pd.read_json(DATA/'label_prevalence.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
rng=np.random.default_rng(20261024)
for mode in ['all_cells','judge_dataset']:
 data=c if mode=='all_cells' else c.groupby(['judge','dataset','label','outcome'],as_index=False).rate.mean()
 fig,ax=plt.subplots(1,2,figsize=(11,5.5));counts=[]
 for i,(label,short,title) in enumerate([('labelled_similarity','similarity','Similar NN'),('important_contrast','contrast','Contrasting NN')]):
  for x,(out,code,color) in enumerate([('unchanged','unchanged','#84909c'),('helpful','helped','#258466'),('harmful','harmed','#c66d66')]):
   vals=data[(data.label==label)&(data.outcome==out)].rate.dropna().to_numpy();row=p[(p.label==short)&(p.outcome==code)].iloc[0]
   ax[i].scatter(x+rng.uniform(-.27,.27,len(vals)),vals,s=11 if mode=='all_cells' else 20,color=color,alpha=.22,edgecolors='none',rasterized=True)
   ax[i].vlines(x,row.lo,row.hi,color='black',lw=1.5,zorder=5);ax[i].plot([x-.29,x+.29],[row['mean']]*2,color='black',lw=3,zorder=6)
   ax[i].text(x,107,f'{row["mean"]:.1f}%',ha='center',fontsize=11);ax[i].text(x,-10,f'n={len(vals)}',ha='center',fontsize=9,color='#555')
   counts.append(dict(label=label,outcome=out,points=len(vals)))
  ax[i].set_title(title,pad=15);ax[i].set_xticks(range(3));ax[i].set_xticklabels(['Unchanged','Corrects\njudge','Misleads\njudge']);ax[i].set_ylim(-13,113);ax[i].set_yticks(range(0,101,20));ax[i].set_box_aspect(1)
 ax[0].set_ylabel('Label prevalence within outcome (%)');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(R/f'label_prevalence_{mode}.{ext}',dpi=190,bbox_inches='tight')
 plt.close(fig);pd.DataFrame(counts).to_json(R/f'label_prevalence_{mode}_counts.json',orient='records',indent=2)
html='''<!doctype html><meta charset="utf-8"><title>All label-prevalence points</title><style>body{font:17px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 25px;color:#263547}img{width:100%}</style><h1>Label associations: show every cell</h1><h2>Every backbone × judge × dataset</h2><p>Each dot is one weighted label-prevalence estimate within an outcome group: up to 600 dots per group. Undefined groups with no eligible cases are omitted, not plotted as zero. The displayed n counts valid cells, not independent examples.</p><img src="label_prevalence_all_cells.png"><p><a href="label_prevalence_all_cells.pdf">PDF</a></p><h2>Average backbones only: every judge × dataset</h2><p>Each dot now averages backbones within one judge and dataset, leaving up to 120 dots per group.</p><img src="label_prevalence_judge_dataset.png"><p><a href="label_prevalence_judge_dataset.pdf">PDF</a></p><p>Black bars retain the original hierarchical means: equal averages over available backbones within judge/dataset, then datasets within judge, then judges. They are not a raw average of all visible dots when availability differs. Black vertical intervals retain the 95% target-cluster bootstrap intervals. Small and large outcome cells are both shown, so extreme dots can reflect sparse groups. Repeated targets across cells mean dots are not independent replicates.</p><p>All ten datasets, twelve judges, full judge-probability range; paired valid control/1-NN outcomes at cutoff 50, no probe-referral gate. Independently annotated yes/no labels, uncertain/unresolved excluded separately for each label. <a href="label_associations.html">Original summary plots</a></p>'''
(R/'label_all_points.html').write_text(html)
