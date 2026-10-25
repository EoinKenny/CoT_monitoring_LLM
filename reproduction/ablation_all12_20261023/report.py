import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import csv,json,html
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'results'
def read(n):return list(csv.DictReader(open(R/(n+'.csv'))))
s=read('summary');cov=read('coverage');cells=read('cells')
C=['baseline','nearest','text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full']
L=['No explanation','Full NN: text + label + prediction','Text + label','Text + prediction','Text only','Label only','Prediction only','Label + prediction (no text)','Random neighbour: all components']
cohorts=[('all10','All ten datasets'),('safety6','Safety / toxicity (6)'),('hallucination3','Hallucination (3)'),('high_stakes','High-stakes (1)'),('toxicity4','Toxicity subset (4)')]
def get(co='all10',dim='overall',group='ALL'):
 return {r['condition']:r for r in s if r['cohort']==co and r['dimension']==dim and r['group']==group}
def table(h,rr):return '<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rr)+'</table></div>'
def fmt(v):return f'{float(v)*100:.2f}'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,3,figsize=(14,5),sharey=True)
for ax,metric in zip(axs,['precision','recall','f1']):
 g=get();v=[100*float(g[c][metric]) for c in C];ax.barh(range(9),v,color=['#63758b','#247a59']+['#81a99a']*6+['#cc9630']);ax.invert_yaxis();ax.set_yticks(range(9));ax.set_yticklabels(L);ax.set_xlim(0,100);ax.set_xlabel(metric.title()+' (%)');ax.grid(axis='x',alpha=.15)
 for i,x in enumerate(v):ax.text(x+.4,i,f'{x:.2f}',va='center',fontsize=9)
fig.suptitle('NN component ablations · fixed probe 50 / judge 50 · all 12 judges',fontweight='bold');fig.tight_layout();fig.savefig(R/'overall.png');fig.savefig(R/'overall.pdf');plt.close(fig)
# Contrast each condition with the full NN, paired within every cell.
fig,axs=plt.subplots(1,3,figsize=(14,5),sharey=True)
for ax,(co,title) in zip(axs,cohorts[1:4]):
 g=get(co);v=[100*(float(g[c]['f1'])-float(g['nearest']['f1'])) for c in C];ax.barh(range(9),v,color=['#63758b','#247a59']+['#81a99a']*6+['#cc9630']);ax.invert_yaxis();ax.set_yticks(range(9));ax.set_yticklabels(L);ax.axvline(0,color='black',lw=.8);ax.set_title(title);ax.set_xlabel('F1 difference from full NN (pp)');ax.grid(axis='x',alpha=.15)
lo=min(ax.get_xlim()[0] for ax in axs);hi=max(ax.get_xlim()[1] for ax in axs)
for ax in axs:ax.set_xlim(lo,hi)
fig.tight_layout();fig.savefig(R/'genres.png');fig.savefig(R/'genres.pdf');plt.close(fig)
b='<h1>NN component ablations</h1><p>23 October 2026 · 12 judges × 5 backbones × 10 datasets · existing responses only.</p><p>Fixed probe threshold 50% and judge threshold 50 for every condition. All nine conditions share the same test cases within a judge; failures never filter other judges. Scores are weighted, then averaged equally over the 600 judge/backbone/dataset cells. Best F1 is underlined.</p><img src="overall.png">'
for co,title in cohorts:
 g=get(co);best=max(float(r['f1']) for r in g.values());rr=[]
 for c,label in zip(C,L):
  r=g[c];f=fmt(r['f1']);rr.append([label,fmt(r['precision']),fmt(r['recall']),'<u>'+f+'</u>' if float(r['f1'])==best else f,f"{100*(float(r['f1'])-float(g['baseline']['f1'])):+.2f}",f"{100*(float(r['f1'])-float(g['nearest']['f1'])):+.2f}"])
 b+='<h2>'+title+'</h2>'+table(['Condition','Precision','Recall','F1','ΔF1 vs control','ΔF1 vs full NN'],rr)
b+='<h2>Component effects by genre</h2><img src="genres.png"><p>Negative values mean performance below full NN. The toxicity subset overlaps the six-dataset safety group.</p>'
for dim,title in [('model','By judge'),('dataset','By dataset'),('backbone','By backbone')]:
 b+='<h2>'+title+'</h2><p>F1 (%), using matched cases across all conditions within each judge.</p>';rr=[]
 for group in sorted({r['group'] for r in s if r['cohort']=='all10' and r['dimension']==dim}):
  g=get(dim=dim,group=group);best=max(float(r['f1']) for r in g.values());rr.append([group]+[('<u>'+fmt(g[c]['f1'])+'</u>') if float(g[c]['f1'])==best else fmt(g[c]['f1']) for c in C])
 b+=table(['Group']+L,rr)
b+='<h2>Coverage</h2><p>Each judge had 25,000 selected test cases across five backbones. API cases are retained only when all nine conditions are scorable. This conservative mask provides one consistent comparison; it differs from the three-condition main experiment, so its control/full-NN averages can differ. Local estimators retain original full-population denominators; API estimators describe the matched weighted cohort. Cross-judge ranks also reflect differing response coverage.</p>'
b+=table(['Judge','Retained test cases','Available','Retention'],[[m,sum(int(r['retained']) for r in cov if r['model']==m),25000,f"{sum(int(r['retained']) for r in cov if r['model']==m)/250:.1f}%"] for m in sorted({r['model'] for r in cov})])
b+='<h2>Interpretation</h2><p>These are descriptive ablations on a previously inspected test set, with no threshold retuning or new calls. Text+label isolates removing prediction; text+prediction isolates removing label. Random-full tests the choice of neighbour while retaining all components. Differences describe prompt interventions, not proof of private internal reasoning. The approved final standalone integer rule is applied to normally completed API responses, with raw responses preserved; failures, blocks and truncated responses remain unavailable.</p><p><a href="summary.csv">Full precision/recall/F1 breakdowns</a> · <a href="cells.csv">Per-cell results</a> · <a href="coverage.csv">Coverage</a> · <a href="VERIFIED.json">Verification receipt</a> · <a href="overall.pdf">Overall PDF</a> · <a href="genres.pdf">Genre PDF</a></p>'
(R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN ablations</title><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#182738;background:#fafbfd}img{width:100%}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#edf2f5}.scroll{overflow-x:auto}u{text-underline-offset:3px;text-decoration-thickness:2px}</style>'+b)
assert all(np.isfinite(float(r[k])) and 0<=float(r[k])<=1 for r in cells for k in ['precision','recall','f1','accuracy'])
assert len(cells)==5400
print('Report verified and saved')
