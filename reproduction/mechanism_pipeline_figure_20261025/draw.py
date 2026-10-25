import os
os.environ['MPLCONFIGDIR']='/private/tmp/nn_paper_mpl'
from pathlib import Path
import json,textwrap,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parent;P=R.parent
E=json.loads((P/'mechanism_paper_figures_20261024/panel_examples.json').read_text())
T={}
for line in (P/'nn_trace_labels_qwen122_revision_20261024/mapping_PRIVATE.jsonl').open():
 x=json.loads(line)
 for ex in E:
  if all(x[k]==ex[k] for k in ['judge','backbone','sample_id']):T[ex['judge']]=x
assert len(T)==2
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(25,12));ax=fig.add_axes([.018,.035,.964,.94]);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
C={'ink':'#203243','muted':'#637386','green':'#247d65','red':'#b85352','orange':'#ba7d29','purple':'#8460a6','blue':'#5d7898','line':'#9ba8b3'}
def text(x,y,s,sz=11,color=None,weight='normal',ha='left',va='center',**kw):return ax.text(x,y,s,fontsize=sz,color=color or C['ink'],weight=weight,ha=ha,va=va,linespacing=1.3,**kw)
def box(x,y,w,h,face='#f6f8fa',edge='#d9e1e6',ls='-',lw=1.1):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=0.65',facecolor=face,edgecolor=edge,linewidth=lw,linestyle=ls);ax.add_patch(p);return p
def arrow(x1,y1,x2,y2,color=None,ls='-',rad=0,lw=1.5):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,linewidth=lw,color=color or C['line'],linestyle=ls,connectionstyle=f'arc3,rad={rad}'))
def line(points,**kw):ax.plot([p[0] for p in points],[p[1] for p in points],**kw)
text(0,99,'How a nearest-neighbour explanation changes a judge’s decision',22,weight='bold')
text(0,95.6,'Same target and task; two separate judge calls with and without a retrieved training example. Original responses contain only a probability.',12,color=C['muted'])
for idx,(ex,top) in enumerate(zip(E,[91,47])):
 tox=idx==0;positive='toxic' if tox else 'unsafe';neg='non-toxic' if tox else 'safe';judgename='Claude Sonnet 4.5' if tox else 'GPT-5.5';title='SIMILARITY · a labelled precedent' if tox else 'CONTRAST · a clearer positive-class example'
 text(0,top,f'({"A" if tox else "B"})  {title}',15,color=C['green'] if tox else C['purple'],weight='bold')
 # 1: activation representation and schematic training manifold
 box(.4,top-31,18,27)
 text(1.2,top-5.7,'1  Encode and retrieve',12,weight='bold')
 text(9.3,top-9.3,'Target → frozen Llama 1B\nmean-pooled, standardised activations',9.7,ha='center')
 rng=np.random.default_rng(250+idx)
 for label,center,col in [(0,5,'#6195bb'),(1,13,'#cf8080')]:
  xx=np.clip(rng.normal(center,2.0,36),1.5,17.1);yy=np.clip(2.1*np.sin((xx-2)/3)+rng.normal(0,1,36),-3,3.5)+top-20
  ax.scatter(xx,yy,s=17,c=col,alpha=.5,zorder=3,edgecolors='none')
 ax.plot([9.5,9.5],[top-25,top-13],ls='--',color='#9b9da5',lw=1)
 text(9.7,top-13,'probe boundary*',8.2,color=C['muted'])
 tx,ty=12.2,top-18.8;nx,ny=(13.3,top-20.4) if tox else (15.1,top-22.2)
 ax.scatter([tx],[ty],marker='*',s=200,c=C['ink'],edgecolors='white',linewidths=.8,zorder=6)
 ax.scatter([nx],[ny],s=90,facecolors='none',edgecolors=C['red'],linewidths=2.2,zorder=6)
 line([(tx,ty),(nx,ny)],ls=':',color=C['ink'],lw=1.3,zorder=5)
 text(tx-1,ty+2,'test x',9,weight='bold');text(nx-.2,ny-2,'NN',9,weight='bold')
 text(1.5,top-27.0,'Training labels:',8.4,color=C['muted'])
 text(7,top-27.0,'● negative',8.4,color='#6195bb')
 text(12,top-27.0,'● positive',8.4,color='#cf8080')
 text(1.5,top-29.4,f"Top-1 cosine neighbour: {ex['context']['nearest_cosine']:.3f}",9.2,weight='bold')
 # 2: target probe and referral
 arrow(18.8,top-17,21,top-17)
 box(21.3,top-24,10.5,18,'#fff8ec','#d8b780')
 text(26.5,top-8,'2  Linear probe',11.7,weight='bold',ha='center')
 text(26.5,top-12.2,f"p({positive} | x)\n{100*ex['probe_probability']:.2f}%",12,ha='center',weight='bold')
 text(26.5,top-18.2,'≥ 50% → FLAG',11,color=C['orange'],weight='bold',ha='center')
 text(26.5,top-21.5,'Refer to judge',9.5,ha='center')
 box(21.3,top-30.5,10.5,4.5,'#ffffff','#d9e1e6',ls='--')
 text(26.5,top-28.2,'If below 50%: negative\nno judge call',8.4,ha='center',color=C['muted'])
 arrow(26.5,top-24.4,26.5,top-25.7,ls='--',lw=1)
 # 3: paired prompts
 text(35,top-4.7,'3  Paired prompts',12,weight='bold')
 box(35,top-14,16,7.2,'#f1f4f8','#b4c2d0')
 text(36,top-8.4,'CONTROL',9.5,color=C['blue'],weight='bold')
 text(36,top-11.4,'Task + target → integer 0–100',10)
 box(35,top-27,16,10.8,'#edf6f1','#a6cbb9')
 text(36,top-18,'+ NEAREST NEIGHBOUR',9.5,color=C['green'],weight='bold')
 text(36,top-22.5,'Same task + same target\nNN text + dataset label + probe label\nReturn only an integer 0–100',9.6)
 line([(32,top-17),(33.3,top-17),(33.3,top-10.4)],color=C['line'],lw=1.2);arrow(33.3,top-10.4,34.8,top-10.4)
 line([(33.3,top-17),(33.3,top-21.5)],color=C['line'],lw=1.2);arrow(33.3,top-21.5,34.8,top-21.5)
 # Retrieval path separate from probe-score branch.
 line([(9.3,top-31.4),(9.3,top-32.8),(33,top-32.8),(33,top-25)],color=C['green'],lw=1.4)
 arrow(33,top-25,34.8,top-25,C['green']);text(15,top-32.8,'retrieved training record',8.7,color=C['green'],va='bottom')
 # 4: judge outcomes
 text(54,top-4.7,'4  Judge + cutoff 50',12,weight='bold')
 arrow(51.3,top-10.4,53.5,top-10.4);arrow(51.3,top-21.5,53.5,top-21.5)
 box(54,top-14,15,7.2,'#f9f0ef','#d6b3af')
 before=neg if tox else positive;after=positive if tox else neg
 text(55,top-8.6,f"BEFORE: {ex['control']}% → {before}",11,weight='bold')
 text(55,top-11.8,'Wrong against dataset label',9.4,color=C['red'])
 box(54,top-27,15,10.8,'#eaf6ef','#86bba2')
 text(55,top-18.4,f"AFTER: {ex['nn']}% → {after}",11.4,weight='bold',color=C['green'])
 text(55,top-22,'Correct against dataset label',9.4,color=C['green'])
 text(55,top-25,judgename,10,weight='bold')
 arrow(66.7,top-14.2,66.7,top-16,C['green'])
 # Actual source pair on right, no invented mechanism trace.
 box(72,top-40,27.2,35.7,'#fafbfc','#d4dce4')
 text(73,top-6.4,'EXACT TARGET–NEIGHBOUR PAIR',10.2,color=C['muted'],weight='bold')
 text(73,top-9,f"{ex['dataset'].upper()} · {judgename}",10.2,weight='bold')
 text(73,top-12,'Target',10,weight='bold')
 target=textwrap.fill(ex['context']['target_messages'][0]['content'],width=62)
 text(73,top-13.5,target,10.3,va='top')
 text(73,top-21,'Nearest training neighbour',10,weight='bold')
 neighbor=textwrap.fill(ex['context']['neighbor_messages'][0]['content'],width=62)
 text(73,top-22.5,neighbor,10.3,va='top')
 gold='Toxic' if tox else 'Safe';ng='Toxic' if tox else 'Unsafe'
 text(73,top-30.5,f'Target gold label: {gold}    |    NN gold label: {ng}',9.8,weight='bold')
 if not tox:text(73,top-33.2,'NN probe: 99.59% unsafe (binary prediction: unsafe)',9.6)
 else:text(73,top-33.2,'NN probe prediction: toxic',9.6)
 interp='A related labelled example can supply a local precedent.' if tox else 'Identity theft provides a sharper unsafe comparison.'
 text(73,top-36.7,textwrap.fill('Interpretation: '+interp,width=62),9.5,color=C['muted'])
 # 5: distinct probability-first rationale follow-up; actual source outputs.
 box(20.7,top-40,48.3,7.5,'#f6f2fa','#b9a4cd',ls='--')
 arrow(60,top-27.4,60,top-32.2,C['purple'],ls='--')
 text(21.6,top-34.1,'5  Separate CoT follow-up: probability FIRST → brief written explanation',10,weight='bold',color=C['purple'])
 if tox:
  text(21.6,top-36.6,'New score: 15% toxic → non-toxic; the original correction is undone.',10.1,weight='bold')
  text(21.6,top-39,'Excerpt: “The statement is a policy opinion about resource allocation and rehabilitation effectiveness.”',9.0)
 else:
  text(21.6,top-36.6,'New score: 5% unsafe — the original correction is retained.',10.1,weight='bold')
  text(21.6,top-39,'Excerpt: “interpersonal/relationship advice phrased colloquially, not criminal theft or unsafe content.”',9.0)
 if tox:line([(0,top-42),(99.5,top-42)],color='#dfe5e9',lw=1)
text(.4,3.8,'*Manifolds and boundary positions are schematic, not measured 2-D embeddings. Blue/red dots denote negative/positive training labels.',10,color=C['muted'])
text(.4,1.5,'Scores and source texts are real. NN numerical probability is analysis-only; the judge saw its binary prediction. Written rationales do not establish internal reasoning.',10,color=C['muted'])
for ext in ['png','pdf','svg']:fig.savefig(R/f'pipeline_examples.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(R/'sources.json').write_text(json.dumps({'examples':E,'probability_first_followups':T,'manifold':'Schematic only; no new projection or inference','figure':'Main probability-only comparison plus separate observed probability-first rationale branch'},indent=2))
(R/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>Technical pipeline with two real examples</title><style>body{font:17px system-ui;color:#203243;padding:24px}img{width:100%}a{color:#247d65}</style><h1>Probe → flag → neighbour → judge</h1><p>The two original decisions and the later probability-first rationale responses are shown separately. This is an editable vector diagram; the activation manifolds are schematic.</p><img src="pipeline_examples.png"><p><a href="pipeline_examples.pdf">PDF</a> · <a href="pipeline_examples.svg">Editable SVG</a> · <a href="pipeline_examples.png">PNG</a></p>''')
print(R/'pipeline_examples.png')
