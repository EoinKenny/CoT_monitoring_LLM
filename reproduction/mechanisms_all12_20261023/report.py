import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import csv,json,collections
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'results'
def read(n):return list(csv.DictReader(open(R/(n+'.csv'))))
s=read('summary');p=read('prevalence');freq=read('label_distribution');agr=read('agreement');ev=read('outcome_counts');bal=read('equal_cell_sensitivity');bd=read('breakdowns')
LAB=['labelled_similarity','important_contrast'];NAMES=['Labelled similarity','Important contrast'];EVENTS=['helpful','harmful','unchanged'];COL=['#247a59','#bb7168','#8795a8']
CO=[('all10','All datasets'),('safety6','Safety / toxicity (6)'),('hallucination3','Hallucination (3)'),('high_stakes','High-stakes (1)'),('toxicity4','Toxicity subset (4)')]
def select(co='all10',policy='judge50_direct',ann='independent_majority'):return [r for r in s if r['cohort']==co and r['policy']==policy and r['annotator']==ann]
def table(h,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def pct(x):return f'{100*float(x):.1f}%'
def rates(rr,lab):return {r['outcome']:float(r['yes_fraction']) for r in rr if r['label']==lab}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,lab,title in zip(axs,LAB,NAMES):
 g=rates(select(),lab);v=[100*g[e] for e in EVENTS];ax.bar(range(3),v,color=COL);ax.set_xticks(range(3));ax.set_xticklabels(['Helpful flip','Harmful flip','Unchanged']);ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.15)
 for i,x in enumerate(v):ax.text(i,x+1,f'{x:.1f}%',ha='center')
axs[0].set_ylabel('Feature labelled yes (%)');fig.suptitle('Mechanism prevalence by NN outcome · independent annotator majority',fontweight='bold');fig.tight_layout();fig.savefig(R/'outcomes.png');fig.savefig(R/'outcomes.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,(co,title) in zip(axs,CO[1:4]):
 for ei,e in enumerate(EVENTS):
  v=[rates(select(co),lab)[e]*100 for lab in LAB];ax.bar(np.arange(2)+(ei-1)*.24,v,width=.24,label=e.capitalize(),color=COL[ei])
 ax.set_xticks(range(2));ax.set_xticklabels(['Labelled\nsimilarity','Important\ncontrast']);ax.set_title(title);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Feature labelled yes (%)');axs[-1].legend(frameon=False);fig.tight_layout();fig.savefig(R/'genres.png');fig.savefig(R/'genres.pdf');plt.close(fig)
models=sorted({r['annotator_a'] for r in agr}|{r['annotator_b'] for r in agr})
fig,axs=plt.subplots(1,2,figsize=(14,7))
for ax,lab,title in zip(axs,LAB,NAMES):
 mat=np.eye(12)
 for r in agr:
  if r['label']==lab:i=models.index(r['annotator_a']);j=models.index(r['annotator_b']);mat[i,j]=mat[j,i]=float(r['kappa'])
 im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(12));ax.set_xticklabels(models,rotation=90,fontsize=8);ax.set_yticks(range(12));ax.set_yticklabels(models,fontsize=8);ax.set_title(title+' · Cohen κ')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65);fig.subplots_adjust(bottom=.28,wspace=.55,right=.85);fig.savefig(R/'agreement.png');fig.savefig(R/'agreement.pdf');plt.close(fig)
b='<h1>NN mechanism labels and helpful versus harmful changes</h1><p>23 October 2026 · 25,000 test contexts · 12 annotators · 12 original decision judges · no new model calls.</p><p><b>Labelled similarity</b>: the neighbour illustrates how similar content is classified. <b>Important contrast</b>: the neighbour highlights a decision-relevant difference. These labels can co-occur; neither is a ground-truth explanation of internal reasoning.</p><p>Main analysis compares the original control and one-NN probability-only decisions at threshold 50. Helpful means wrong→correct; harmful means correct→wrong; unchanged includes correct→correct and wrong→wrong. The probe-gated pipeline is reported separately below.</p><p>For each decision judge, its own mechanism annotation is excluded. A strict majority of at least 6 of the other 11 annotators assigns yes/no/uncertain; unresolved and uncertain cases are explicitly excluded from binary associations. This is a stronger separation than asking a judge to validate its own decision, though models can share biases.</p><img src="outcomes.png">'
b+='<p><b>Main finding:</b> labelled similarity is more common in helpful than harmful flips; important contrast shows the reverse association. Both directions hold for all 12 individual annotators when self-comparisons are excluded, and persist within judge/backbone/dataset cells. Contrast-labelled cases still have positive net NN accuracy gain overall (+2.14 points), but less than cases without contrast (+9.06 points). This does not establish that contrast itself causes errors.</p>'
for co,title in CO:
 rr=select(co);tr=[]
 for lab,name in zip(LAB,NAMES):
  g=rates(rr,lab);tr.append([name,*[pct(g[e]) for e in EVENTS],f"{100*(g['helpful']-g['harmful']):+.1f} pp"])
 b+='<h2>'+title+'</h2>'+table(['Feature','Helpful: yes','Harmful: yes','Unchanged: yes','Helpful − harmful'],tr)
b+='<img src="genres.png"><p>These percentages answer P(label=yes | outcome), not P(help | label). The next table gives that reverse view. The toxicity subset overlaps safety6.</p><h2>What happens when each label is present?</h2>'
rr=select();tr=[]
for lab,name in zip(LAB,NAMES):
 ll=[r for r in rr if r['label']==lab]
 for level in ['yes','no']:
  mass={r['outcome']:float(r['yes_mass']) if level=='yes' else float(r['mass'])-float(r['yes_mass']) for r in ll};den=sum(mass.values());tr.append([name,level,*[pct(mass[e]/den) for e in EVENTS],f"{100*(mass['helpful']-mass['harmful'])/den:+.2f} pp"])
b+=table(['Feature','Label','Helpful','Harmful','Unchanged','Net accuracy change'],tr)
b+='<h2>Label distribution</h2><p>Counts below use 25,000 distinct target/backbone contexts, not repeated outcomes across judges. The all-12 consensus requires at least seven matching votes. These are unweighted corpus frequencies; outcome association plots above use design weights with equal cell mass.</p>'
tr=[]
for ann in ['majority12']+models:
 for lab,name in zip(LAB,NAMES):
  g={r['level']:r for r in freq if r['annotator']==ann and r['label']==lab and r['cohort']=='all10'};tr.append([ann,name]+[f"{int(g[k]['n']):,} ({pct(g[k]['fraction'])})" for k in ['yes','no','uncertain',('unresolved' if ann=='majority12' else 'missing')]])
b+=table(['Annotator','Feature','Yes','No','Uncertain','Missing / unresolved'],tr)
b+='<h2>Agreement between annotators</h2><img src="agreement.png"><p>Pairwise agreement includes yes/no/uncertain and excludes missing responses. Cohen κ corrects for agreement expected from each model’s label frequencies. Similarity and contrast are analysed separately.</p>'
b+=table(['Feature','Mean raw agreement','Mean κ','Min κ','Max κ'],[[name,pct(np.mean([float(r['agreement']) for r in agr if r['label']==lab])),f"{np.mean([float(r['kappa']) for r in agr if r['label']==lab]):.3f}",f"{min(float(r['kappa']) for r in agr if r['label']==lab):.3f}",f"{max(float(r['kappa']) for r in agr if r['label']==lab):.3f}"] for lab,name in zip(LAB,NAMES)])
b+='<h2>Does each annotator give the same conclusion?</h2><p>Each annotator’s labels are evaluated against the other eleven decision judges, excluding self-comparisons. Values are percentage-point differences in feature prevalence: helpful minus harmful.</p>';tr=[]
for ann in models:
 row=[ann]
 for lab in LAB:
  ss=[r for r in p if r['annotator']==ann and r['judge']!=ann and r['policy']=='judge50_direct' and r['cohort']=='all10' and r['label']==lab];g={e:sum(float(r['yes_mass']) for r in ss if r['outcome']==e)/sum(float(r['mass']) for r in ss if r['outcome']==e) for e in ['helpful','harmful']};row.append(f"{100*(g['helpful']-g['harmful']):+.1f}")
 tr.append(row)
b+=table(['Annotator','Labelled similarity','Important contrast'],tr)
b+='<h2>By decision judge</h2>';tr=[]
for m in models:
 row=[m]
 for lab in LAB:
  ss=[r for r in p if r['annotator']=='independent_majority' and r['judge']==m and r['policy']=='judge50_direct' and r['cohort']=='all10' and r['label']==lab];g={r['outcome']:float(r['yes_mass'])/float(r['mass']) for r in ss};row.extend([pct(g['helpful']),pct(g['harmful']),f"{100*(g['helpful']-g['harmful']):+.1f}"])
 tr.append(row)
b+=table(['Judge','Similarity: helpful','Similarity: harmful','Δ pp','Contrast: helpful','Contrast: harmful','Δ pp'],tr)
b+='<h2>Within-cell sensitivity</h2><p>To check whether pooled differences reflect different dataset/judge mixes, this comparison averages helpful-minus-harmful label prevalence within each judge/backbone/dataset cell containing both outcomes.</p>';tr=[]
for co,title in CO:
 sets={'all10':None,'safety6':{'aegis','beavertails','toxicchat','toxigen','wildguard','wildjailbreak'},'hallucination3':{'fava','halueval','ragtruth'},'high_stakes':{'high_stakes'},'toxicity4':{'aegis','beavertails','toxicchat','toxigen'}}
 for lab,name in zip(LAB,NAMES):
  ss=[r for r in bal if r['policy']=='judge50_direct' and r['label']==lab and (sets[co] is None or r['dataset'] in sets[co])];tr.append([title,name,len(ss),f"{100*np.mean([float(r['delta']) for r in ss]):+.1f} pp"])
b+=table(['Group','Feature','Eligible cells','Within-cell helpful − harmful'],tr)
b+='<h2>Pipeline sensitivity: probe 50 / judge 50</h2><p>Here cases below the probe referral threshold remain negative under both prompts and therefore unchanged. This separates pipeline behaviour from the direct judge comparison.</p>';tr=[]
for lab,name in zip(LAB,NAMES):
 g=rates(select(policy='pipeline50'),lab);tr.append([name,*[pct(g[e]) for e in EVENTS],f"{100*(g['helpful']-g['harmful']):+.1f} pp"])
b+=table(['Feature','Helpful','Harmful','Unchanged','Helpful − harmful'],tr)
b+='<h2>Sample sizes and limits</h2>';tr=[]
for m in models:
 ss=[r for r in ev if r['judge']==m and r['policy']=='judge50_direct' and r['cohort']=='all10'];tr.append([m]+[int(next(r['n'] for r in ss if r['outcome']==e)) for e in EVENTS])
b+=table(['Judge','Helpful','Harmful','Unchanged'],tr)
b+='<p>Each judge retains its own valid control/NN pairs; saliency failures do not remove those pairs. Annotation failures do not remove another annotator’s labels. Weights are inverse inclusion probabilities normalized within each judge/backbone/dataset cell before outcome conditioning. These are descriptive, exploratory associations on previously examined test data, not independent evidence of internal reasoning or causal mediation. Repeated contexts across decision judges are not independent observations; no significance claims are made from the repeated row counts. Missing/uncertain denominators are retained in the downloadable tables. Reasoning traces are separate corroborating evidence and are not used to assign these outcomes or labels.</p><p>'+ ' · '.join('<a href="'+n+'.csv">'+n.replace('_',' ')+'</a>' for n in ['summary','prevalence','breakdowns','label_distribution','agreement','outcome_counts','equal_cell_sensitivity'])+' · <a href="VERIFIED.json">Verification</a></p>'
(R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN mechanisms</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#182738;background:#fafbfd}img{width:100%}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#edf2f5}.scroll{overflow:auto}</style>'+b)
for co,title in CO:
 for lab in LAB:print(co,lab,rates(select(co),lab))
print('REPORT READY')
