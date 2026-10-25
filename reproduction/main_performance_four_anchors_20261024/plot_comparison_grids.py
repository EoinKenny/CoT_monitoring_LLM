import os
os.environ['MPLBACKEND']='Agg'
import csv,json,collections,hashlib,math,html
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

R=Path(__file__).resolve().parent;O=R/'comparison_grids';O.mkdir(exist_ok=True)
META=json.loads((R/'group_figures/VERIFIED.json').read_text())
records=META['group_figures']
coords=list(csv.DictReader((R/'group_figures/curve_coordinates.csv').open()))
cells=list(csv.DictReader((R/'results/cells.csv').open()))
C=['baseline','saliency','nearest','nearest2'];L=['No explanation','Saliency','1-NN','2-NN'];COL=['#63758b','#cc9630','#8562aa','#247a59']
curve=collections.defaultdict(list)
for r in coords:curve[r['kind'],r['group'],r['policy'],r['condition']].append(r)
for key,rs in list(curve.items()):curve[key]=np.array([[float(r['recall']),float(r['precision'])] for r in sorted(rs,key=lambda r:int(r['operating_index']))])
swarm={};checked=0
for rec in records:
 grouped=collections.defaultdict(list)
 for r in cells:
  if (r['model']==rec['group'] if rec['kind']=='judge' else r['dataset'] in rec['datasets']):
   grouped[r['policy'],r['condition'],r['model'],r['dataset']].append(float(r['f1'])*100)
 for stat in rec['stats']:
  pol,c=stat['policy'],stat['condition']
  vals=[v for k,v in sorted(grouped.items()) if k[:2]==(pol,c)]
  assert all(len(v)==(5 if pol=='fixed50' else 20) for v in vals)
  v=np.array([np.mean(x) for x in vals]);assert len(v)==stat['dots']
  assert abs(v.mean()-stat['mean_f1'])<1e-10
  swarm[rec['kind'],rec['group'],pol,c]=v;checked+=1
assert checked==128

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
handles=[Line2D([0],[0],color=col,lw=2.5,label=lab) for col,lab in zip(COL,L)]
definitions=[('pr_fixed','Precision–recall · probe fixed at 50','fixed'),('pr_tuned','Precision–recall · validation-selected thresholds','tuned'),('f1_fixed','F1 · fixed probe 50 / judge 50','fixed50'),('f1_tuned','F1 · validation-optimal thresholds','tuned_shared')]
outputs=[]
with PdfPages(O/'all_eight_comparisons.pdf') as allpdf:
 for kind in ['judge','genre']:
  rs=[r for r in records if r['kind']==kind]
  ncol,nrow=(4,3) if kind=='judge' else (2,2)
  with PdfPages(O/(kind+'_comparisons.pdf')) as kindpdf:
   for typ,heading,pol in definitions:
    is_pr=typ.startswith('pr')
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*4.65,nrow*5.2+1.15),sharex=True,sharey=True)
    if is_pr:
     xmin=min(r['axes'][pol]['recall'][0] for r in rs)
     xmax=min(100,5*math.ceil(max(r['axes'][pol]['recall'][1] for r in rs)/5))
     ymin=min(r['axes'][pol]['precision'][0] for r in rs)
     limits={'x':[xmin,xmax],'y':[ymin,100]}
    else:
     lowest=min(swarm[kind,r['group'],pol,c].mean() for r in rs for c in C)
     ymin=max(0,min(65,5*math.floor((lowest-3)/5)))
     limits={'y':[ymin,100]}
    for i,(ax,rec) in enumerate(zip(axes.flat,rs)):
     if is_pr:
      mark=50 if pol=='fixed' else 20
      for c,col in zip(C,COL):
       xy=curve[kind,rec['group'],pol,c]
       ax.plot(xy[:,0],xy[:,1],color=col,lw=2.1)
       ax.scatter(*xy[mark],s=45,color=col,edgecolors='white',linewidth=.7,zorder=5)
       assert xmin<=xy[mark,0]<=xmax and ymin<=xy[mark,1]<=100
      ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,100)
      ax.set_xticks(np.arange(math.ceil(xmin/10)*10,xmax+1,10))
      ax.set_yticks(range(ymin,101,5))
     else:
      for j,(c,col) in enumerate(zip(C,COL)):
       v=swarm[kind,rec['group'],pol,c];jitter=np.random.default_rng(20261024).uniform(-.25,.25,len(v))
       ax.scatter(j+jitter,v,s=32 if kind=='judge' else 23,color=col,alpha=.5,edgecolors='none')
       ax.plot([j-.24,j+.24],[v.mean()]*2,color=col,lw=3.5)
       ax.text(j,v.mean()+.9 if v.mean()<98 else v.mean()-1.6,f'{v.mean():.2f}',ha='center',color=col,weight='bold',fontsize=10)
      ax.set_ylim(ymin,100);ax.set_yticks(range(ymin,101,5))
      ax.set_xticks(range(4));ax.set_xticklabels(['No\nexplanation','Saliency','1-NN','2-NN'],fontsize=10)
      ax.tick_params(axis='x',labelbottom=True)
     ax.set_title(rec['title'],fontsize=13,pad=12)
     ax.set_box_aspect(1);ax.grid(False)
     if i%ncol==0:ax.set_ylabel('Mean test precision (%)' if is_pr else 'Mean test F1 (%)')
     if is_pr and i//ncol==nrow-1:ax.set_xlabel('Mean test recall (%)')
    title=('All 12 judges' if kind=='judge' else 'All four dataset genres')+' · '+heading
    fig.suptitle(title,fontsize=17,y=.985)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=12)
    fig.tight_layout(rect=[0,.07,1,.96],w_pad=1.6,h_pad=3.8)
    fig.canvas.draw()
    for ax in axes.flat:
     box=ax.get_window_extent();assert abs(box.width/box.height-1)<1e-8
     assert not any(l.get_visible() for l in ax.get_xgridlines()+ax.get_ygridlines())
    slug=kind+'_'+typ
    for ext in ['png','pdf']:fig.savefig(O/(slug+'.'+ext),dpi=180,bbox_inches='tight')
    kindpdf.savefig(fig,bbox_inches='tight');allpdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    outputs.append({'kind':kind,'plot_type':typ,'title':title,'slug':slug,'axes':limits,'panels':len(rs)})
    print(slug,flush=True)

body='<h1>Experiment 1: eight large comparison grids</h1><p>Each grid compares every judge or every dataset genre for one plot type. All panels within a grid share axes, use the same colours, and average over five backbones. Tuning uses four shared anchors: control, saliency, 1NN and 2NN.</p><p>For precision–recall curves, dots mark 50/50 on the fixed-probe curve, or the validation F1 optimum on the tuned curve. The fixed curve sweeps judge cutoffs; the tuned curve selects both cutoffs on validation across F-beta objectives. Each swarm point is one judge–dataset pair. All observations contribute to mean bars, including those below the displayed range.</p><p><a href="all_eight_comparisons.pdf">All eight grids: PDF</a> · <a href="judge_comparisons.pdf">Judge grids: PDF</a> · <a href="genre_comparisons.pdf">Genre grids: PDF</a></p><nav><a href="#judges">Judges</a> · <a href="#genres">Dataset genres</a></nav>'
for kind,heading,anchor in [('judge','Judge models','judges'),('genre','Dataset genres','genres')]:
 body+='<h2 id="'+anchor+'">'+heading+'</h2>'
 if kind=='judge':body+='<p>Each swarm panel contains ten dataset points, each averaged over five backbones.</p>'
 else:
  body+='<ul>'+''.join('<li><b>'+html.escape(r['title'])+'</b>: '+', '.join(r['datasets'])+'</li>' for r in records if r['kind']=='genre')+'</ul>'
 for o in [o for o in outputs if o['kind']==kind]:
  body+='<section id="'+o['slug']+'"><h3>'+html.escape(o['title'])+'</h3><a href="'+o['slug']+'.png"><img loading="lazy" src="'+o['slug']+'.png"></a><p><a href="'+o['slug']+'.pdf">Vector PDF</a> · <a href="'+o['slug']+'.png">Full-size PNG</a></p></section>'
body+='<p>All results reproduce the preceding four-panel group figures. Display scales are common within each grid; fixed and tuned grids can have different ranges. API judges retain their own matched usable cases. The reject-all precision convention can produce rising line segments on fixed-probe curves; these are not an optimised test frontier.</p>'
(O/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Experiment 1: comparison grids</title><style>body{font:16px/1.6 system-ui;max-width:1800px;margin:35px auto;padding:0 24px;color:#182738}img{width:100%}h2{margin-top:50px}section{margin-top:35px;border-bottom:1px solid #ddd}a{color:#265e9d}</style>'+body+'</html>')
(O/'VERIFIED.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'group_figures/curve_coordinates.csv',R/'results/cells.csv',R/'group_figures/VERIFIED.json']},'verified_mean_comparisons':checked,'figures':outputs,'anchors':C,'new_model_calls':0,'display_only_regrouping':True},indent=2))
