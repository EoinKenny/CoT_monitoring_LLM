import os
os.environ['MPLBACKEND']='Agg'
import csv,collections,json,hashlib,html,math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parent;O=R/'group_figures';O.mkdir(exist_ok=True)
C=['baseline','saliency','nearest','nearest2'];L=['No explanation','Saliency','1-NN','2-NN'];COL=['#63758b','#cc9630','#8562aa','#247a59']
GENRES={'toxicity':('Toxicity / harmful content',['aegis','beavertails','toxicchat','toxigen']),'safety':('Safety / jailbreaks',['wildguard','wildjailbreak']),'hallucination':('Hallucination',['fava','halueval','ragtruth']),'high_stakes':('High-stakes',['high_stakes'])}
NAMES={'qwen':'Qwen 27B','qwen122':'Qwen 122B','gemma':'Gemma','mistral':'Mistral','olmo':'OLMo','glm':'GLM','claude4.5-sonnet':'Claude Sonnet 4.5','claude4.8-opus':'Claude Opus 4.8','claude-opus-5.5':'Claude Opus 5.5','gpt-4o':'GPT-4o','gpt-astra':'GPT Astra','gpt-5.5':'GPT-5.5'}
cells=list(csv.DictReader((R/'results/cells.csv').open()))
coords=list(csv.DictReader((R/'breakdowns/curve_coordinates.csv').open()))
curve_index=collections.defaultdict(list)
for r in coords:
 if r['dimension'] in ['dataset','model']:curve_index[r['dimension'],r['group'],r['policy'],r['condition']].append(r)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
records=[];plot_values=[]
def make(kind,key,title,datasets,pdf):
 selected=[r for r in cells if (r['dataset'] in datasets if kind=='genre' else r['model']==key)]
 expected_dots=12*len(datasets) if kind=='genre' else 10
 dim='dataset' if kind=='genre' else 'model';groups=datasets if kind=='genre' else [key]
 curves={}
 for pol in ['fixed','tuned']:
  for c in C:
   raw=collections.defaultdict(list)
   for group in groups:
    for r in curve_index[dim,group,pol,c]:raw[int(r['operating_index'])].append([float(r['mean_test_recall'])*100,float(r['mean_test_precision'])*100])
   assert all(len(a)==len(groups) for a in raw.values())
   assert len(raw)==(102 if pol=='fixed' else 41)
   curves[pol,c]=np.array([np.mean(raw[i],axis=0) for i in sorted(raw)])
 fig,axes=plt.subplots(1,4,figsize=(23,6.5))
 # Widen the original zoom only where needed to keep group operating points visible.
 marked=[curves[p,c][50 if p=='fixed' else 20] for p in ['fixed','tuned'] for c in C]
 ymin=min(80,5*math.floor(min(x[1] for x in marked)/5))
 axis_ranges={}
 for ax,pol in zip(axes[:2],['fixed','tuned']):
  marker=50 if pol=='fixed' else 20
  xmin=max(0,min(55,5*math.floor((min(curves[pol,c][marker,0] for c in C)-10)/5)))
  xmax=min(100,math.ceil(max(curves[pol,c][:,0].max() for c in C)+1)) if pol=='fixed' else min(100,max(90,5*math.ceil((max(curves[pol,c][marker,0] for c in C)+3)/5)))
  axis_ranges[pol]={'recall':[xmin,xmax],'precision':[ymin,100]}
  for c,label,color in zip(C,L,COL):
   xy=curves[pol,c];ax.plot(xy[:,0],xy[:,1],color=color,lw=2.2,label=label)
   assert xmin<=xy[marker,0]<=xmax and ymin<=xy[marker,1]<=100
   ax.scatter(*xy[marker],s=48,color=color,edgecolors='white',linewidth=.7,zorder=5)
   for i,(x,y) in enumerate(xy):plot_values.append({'kind':kind,'group':key,'policy':pol,'condition':c,'operating_index':i,'recall':x,'precision':y})
  ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,100);ax.set_yticks(range(ymin,101,5))
  ax.set_xlabel('Mean test recall (%)');ax.set_box_aspect(1);ax.grid(False)
 axes[0].set_ylabel('Mean test precision (%)');axes[1].tick_params(axis='y',labelleft=False)
 vals={}
 for pol in ['fixed50','tuned_shared']:
  grouped=collections.defaultdict(list)
  for r in selected:
   if r['policy']==pol:grouped[r['model'],r['dataset'],r['condition']].append(float(r['f1'])*100)
  for c in C:
   keys=sorted(k for k in grouped if k[2]==c)
   assert len(keys)==expected_dots
   assert all(len(grouped[k])==(5 if pol=='fixed50' else 20) for k in keys)
   vals[pol,c]=np.array([np.mean(grouped[k]) for k in keys])
 lower=max(0,min(65,5*math.floor((min(v.mean() for v in vals.values())-3)/5)))
 stats=[]
 for ax,pol in zip(axes[2:],['fixed50','tuned_shared']):
  for i,(c,col) in enumerate(zip(C,COL)):
   v=vals[pol,c];j=np.random.default_rng(20261024).uniform(-.25,.25,len(v))
   ax.scatter(i+j,v,s=20 if expected_dots>20 else 34,alpha=.48,color=col,edgecolors='none')
   ax.plot([i-.24,i+.24],[v.mean()]*2,color=col,lw=4)
   # Keep the label inside the frame for near-ceiling means.
   ypos=v.mean()+.8 if v.mean()<98.7 else v.mean()-1.5
   ax.text(i,ypos,f'{v.mean():.2f}',color=col,ha='center',weight='bold')
   stats.append({'policy':pol,'condition':c,'dots':len(v),'mean_f1':float(v.mean()),'below_display':int((v<lower).sum())})
  ax.set_xticks(range(4));ax.set_xticklabels(['No\nexplanation','Saliency','1-NN','2-NN'])
  ax.set_ylim(lower,100);ax.set_yticks(range(lower,101,5));ax.set_box_aspect(1);ax.grid(False)
 axes[2].set_ylabel('Mean test F1 (%)');axes[3].tick_params(axis='y',labelleft=False)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,frameon=False,fontsize=12)
 for ax,letter in zip(axes,'ABCD'):ax.text(-.18 if letter in 'AC' else -.06,.5,'('+letter+')',transform=ax.transAxes,ha='right',va='center',fontsize=24,clip_on=False)
 fig.tight_layout(rect=[0,.21,1,.99],w_pad=3);fig.canvas.draw()
 for ax in axes:
  b=ax.get_window_extent();assert abs(b.width/b.height-1)<1e-8
  assert not any(t.get_visible() for t in ax.get_xgridlines()+ax.get_ygridlines())
 slug=kind+'_'+key
 for ext in ['png','pdf']:fig.savefig(O/(slug+'.'+ext),dpi=180,bbox_inches='tight')
 pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
 records.append({'kind':kind,'group':key,'title':title,'datasets':datasets,'slug':slug,'axes':axis_ranges,'swarm_y':[lower,100],'stats':stats})
 print(title,flush=True)

with PdfPages(O/'genres.pdf') as pdf:
 for key,(title,ds) in GENRES.items():make('genre',key,title,ds,pdf)
with PdfPages(O/'judges.pdf') as pdf:
 for key in sorted({r['model'] for r in cells}):make('judge',key,NAMES.get(key,key),sorted({r['dataset'] for r in cells}),pdf)

with (O/'curve_coordinates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(plot_values[0]));w.writeheader();w.writerows(plot_values)
(O/'VERIFIED.json').write_text(json.dumps({'inputs':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'results/cells.csv',R/'breakdowns/curve_coordinates.csv']},'anchors':C,'new_model_calls':0,'group_figures':records,'interpretation':'Equal-weight macro means over backbone/judge/dataset cells. Every swarm point averages five backbones; tuned points also average four shared anchors. All observations contribute to means even if outside display. PR zooms adapt only to retain operating points and reduce empty fixed-probe space; y scale shared within each pair. API coverage is judge-specific.'},indent=2))
body='<h1>Experiment 1: dataset genres and judges</h1><p>All figures use the same four panels: <b>(A)</b> probe fixed at 50, judge cutoff swept; <b>(B)</b> validation-selected probe/judge cutoffs across F-beta objectives; <b>(C)</b> F1 at 50/50; <b>(D)</b> F1 at validation-selected F1 optima.</p><p>Tuning averages all four shared anchors: control, saliency, 1NN and 2NN. Backbones are averaged throughout. No titles, footnotes or grid lines inside figures. Swarm means include all points, including those outside the zoom. Plot limits widen where needed to retain group means and operating points; read the axes when comparing figures.</p><p><a href="genres.pdf">All four genres: PDF</a> · <a href="judges.pdf">All 12 judges: PDF</a></p><nav><a href="#genres">Dataset genres</a> · <a href="#judges">Judge models</a></nav>'
for kind,heading in [('genre','Dataset genres'),('judge','Judge models')]:
 body+='<h2 id="'+('genres' if kind=='genre' else 'judges')+'">'+heading+'</h2>'
 for r in [r for r in records if r['kind']==kind]:
  body+='<section id="'+r['slug']+'"><h3>'+html.escape(r['title'])+'</h3>'
  if kind=='genre':body+='<p>Datasets: '+', '.join(r['datasets'])+'. Each swarm has '+str(r['stats'][0]['dots'])+' judge–dataset points.</p>'
  else:body+='<p>Each swarm has ten points: one per dataset, averaged over five backbones.</p>'
  body+='<a href="'+r['slug']+'.png"><img loading="lazy" src="'+r['slug']+'.png"></a><p><a href="'+r['slug']+'.pdf">PDF</a> · <a href="'+r['slug']+'.png">Full-size PNG</a></p></section>'
body+='<p>Genre groups are non-overlapping. Aegis and BeaverTails cover broader harmful content than toxicity alone. Judge comparisons retain each judge’s own matched cases. These are descriptive results; the dots do not show confidence intervals.</p><p><a href="VERIFIED.json">Verification, plot limits and means</a> · <a href="curve_coordinates.csv">Curve coordinates</a></p>'
(O/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Experiment 1 by genre and judge</title><style>body{font:16px/1.6 system-ui;max-width:1800px;margin:35px auto;padding:0 22px;color:#182738}img{width:100%}h2{margin-top:50px}h3{margin-top:35px}a{color:#265e9d}section{border-bottom:1px solid #ddd}</style>'+body+'</html>')
