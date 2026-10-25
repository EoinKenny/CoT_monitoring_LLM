# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
import os
os.environ['MPLBACKEND']='Agg'
import csv,json,hashlib,collections,html
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

R=OUTPUT
S=DATA
rows=list(csv.DictReader(gzip.open(DATA/'ablation_cells.csv.gz','rt')))
C=['baseline','nearest','text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full']
L=['No explanation','Full 1NN: text + label + prediction','Text + label','Text + prediction','Text only','Label only','Prediction only','Label + prediction (no text)','Random neighbour: full information']
colors=['#63758b','#247a59','#508c76','#6d9a87','#8eaaa0','#508c76','#6d9a87','#8eaaa0','#cc9630']
genres={'Overall':None,'Toxicity / harmful content':['aegis','beavertails','toxicchat','toxigen'],'Safety / jailbreaks':['wildguard','wildjailbreak'],'Hallucination':['fava','halueval','ragtruth'],'High-stakes':['high_stakes']}
metrics=['precision','recall','f1'];summary={};out=[]
assert len(rows)==5400
for genre,datasets in genres.items():
 sub=[r for r in rows if datasets is None or r['dataset'] in datasets]
 paired=collections.defaultdict(dict)
 for r in sub:paired[r['model'],r['backbone'],r['dataset']][r['condition']]=r
 assert all(set(v)==set(C) for v in paired.values())
 assert len(paired)==60*(10 if datasets is None else len(datasets))
 summary[genre]={}
 for c in C:
  r={'genre':genre,'datasets':';'.join(datasets or sorted({v['dataset'] for v in sub})),'condition':c,'cells':len(paired)}
  for metric in metrics:
   vals=np.array([float(v[c][metric]) for v in paired.values()])*100
   assert np.isfinite(vals).all()
   r[metric]=float(vals.mean())
   r['delta_'+metric+'_vs_full']=float(np.mean([(float(v[c][metric])-float(v['nearest'][metric]))*100 for v in paired.values()]))
   r['delta_'+metric+'_vs_control']=float(np.mean([(float(v[c][metric])-float(v['baseline'][metric]))*100 for v in paired.values()]))
  summary[genre][c]=r;out.append(r)
with (R/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
def save(fig,slug):
 for ext in ['png','pdf']:fig.savefig(R/(slug+'.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)
def metric_plot(genre,slug):
 fig,axs=plt.subplots(1,3,figsize=(14,6.2),sharey=True)
 for ax,m,letter in zip(axs,metrics,'ABC'):
  values=[summary[genre][c][m] for c in C]
  low=np.floor(min(values)-.4);high=np.ceil(max(values)+.5)
  width=max(high-low,2);high=low+width
  ax.axvline(summary[genre]['nearest'][m],color='#247a59',ls='--',lw=1,alpha=.65,zorder=1)
  for i,(v,col) in enumerate(zip(values,colors)):
   ax.scatter(v,i,s=58,color=col,zorder=3)
   ax.text(v+.035*width,i,f'{v:.2f}',va='center',fontsize=10,color=col)
  ax.set_xlim(low,high+.12*width);ax.set_ylim(8.7,-.7)
  ax.set_yticks(range(9));ax.set_yticklabels(L)
  ax.set_xlabel(('F1' if m=='f1' else m.title())+' (%)')
  ax.text(-.08,1.02,'('+letter+')',transform=ax.transAxes,fontsize=13,ha='right')
  ax.tick_params(axis='y',length=0);ax.grid(False)
 fig.tight_layout(w_pad=2)
 save(fig,slug)
metric_plot('Overall','overall_metrics')

# Paired differences show what each component removal loses or gains.
for metric in metrics:
 fig,axs=plt.subplots(1,4,figsize=(17,6.2),sharey=True,sharex=True)
 gs=list(genres)[1:]
 values=[summary[g][c]['delta_'+metric+'_vs_full'] for g in gs for c in C]
 lo=np.floor(min(values))-1;hi=np.ceil(max(values))+1
 for ax,g,letter in zip(axs,gs,'ABCD'):
  ax.axvline(0,color='#247a59',ls='--',lw=1,zorder=1)
  for i,(c,col) in enumerate(zip(C,colors)):
   v=summary[g][c]['delta_'+metric+'_vs_full']
   ax.plot([0,v],[i,i],color=col,alpha=.4,lw=2,zorder=1)
   ax.scatter(v,i,s=43,color=col,zorder=3)
   ax.annotate(f'{v:+.2f}',(v,i),xytext=(-6 if v<0 else 6,0),textcoords='offset points',ha='right' if v<0 else 'left',va='center',fontsize=9,color=col)
  ax.set_xlim(lo,hi);ax.set_ylim(8.7,-.7);ax.set_yticks(range(9));ax.set_yticklabels(L)
  ax.set_title(g,fontsize=11,pad=16)
  ax.set_xlabel(('F1' if metric=='f1' else metric.title())+' change vs full 1NN (pp)')
  ax.tick_params(axis='y',length=0);ax.grid(False)
 fig.tight_layout(w_pad=2)
 save(fig,'genre_'+metric+'_differences')

def table(genre):
 best=max(summary[genre][c]['f1'] for c in C)
 result='<table><tr><th>Condition</th><th>Precision</th><th>Recall</th><th>F1</th><th>ΔF1 vs control</th><th>ΔF1 vs full 1NN</th></tr>'
 for c,label in zip(C,L):
  r=summary[genre][c];f=f"{r['f1']:.2f}"
  if abs(r['f1']-best)<1e-12:f='<u>'+f+'</u>'
  result+=f"<tr><td>{html.escape(label)}</td><td>{r['precision']:.2f}</td><td>{r['recall']:.2f}</td><td>{f}</td><td>{r['delta_f1_vs_control']:+.2f}</td><td>{r['delta_f1_vs_full']:+.2f}</td></tr>"
 return result+'</table>'
body='<h1>Experiment 2: what information in a neighbour helps?</h1><p>Fixed probe 50 / judge 50 · 12 judges · five backbones · ten datasets. No threshold tuning and no new model calls.</p><h2>Overall precision, recall and F1</h2><img src="overall_metrics.png"><p>The dashed green line marks the full 1NN explanation in each panel. All values are equal-cell macro means, preserving design weights inside each cell.</p>'+table('Overall')
body+='<h2>Component effects by dataset genre</h2><p>Negative values mean worse performance than the full 1NN explanation. Zero is the full-1NN reference. Dots show descriptive paired mean differences, not confidence intervals.</p>'
for m in metrics:
 body+=f'<h3>{m.upper() if m=="f1" else m.title()}</h3><img src="genre_{m}_differences.png"><p><a href="genre_{m}_differences.pdf">Vector PDF</a></p>'
for g,ds in list(genres.items())[1:]:body+='<h2>'+html.escape(g)+'</h2><p>Datasets: '+', '.join(ds)+'.</p>'+table(g)
body+='<h2>Coverage and interpretation</h2><p>Each judge uses test cases matched across all nine ablation conditions. A failed API response excludes a case only from that judge’s comparison. This differs from Experiment 1’s four-condition mask, so the baseline and full-NN means differ slightly. Local population denominators and API matched-cohort weights follow the verified source analysis. Results are descriptive on a repeatedly inspected test set. The four genre groups do not overlap; “toxicity / harmful content” includes broader safety content in Aegis and BeaverTails.</p><p><a href="summary.csv">All plotted values</a> · <a href="VERIFIED.json">Verification</a> · <a href="overall_metrics.pdf">Overall PDF</a></p>'
(R/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Experiment 2: NN ablations</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:35px auto;padding:0 22px;color:#182738}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}u{text-underline-offset:3px;text-decoration-thickness:2px}a{color:#265e9d}</style>'+body+'</html>')
print(json.dumps({g:{c:round(summary[g][c]['f1'],2) for c in C} for g in genres},indent=2))
