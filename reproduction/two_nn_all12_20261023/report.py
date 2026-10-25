import os
os.environ['MPLBACKEND']='Agg'
import csv,json,collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'results'
s=list(csv.DictReader(open(R/'summary.csv')));rows=list(csv.DictReader(open(R/'cells.csv')));cov=list(csv.DictReader(open(R/'coverage.csv')))
C=['baseline','nearest','nearest2'];L=['No explanation','One NN','Two NNs'];colors=['#63758b','#247a59','#8f59a0']
def get(co='all10',dim='overall',group='ALL'):
 return {r['condition']:r for r in s if r['cohort']==co and r['dimension']==dim and r['group']==group}
def f(x):return f'{100*float(x):.2f}'
def table(h,rr):return '<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rr)+'</table></div>'
def compare(g):
 best=max(float(r['f1']) for r in g.values())
 return [[L[i],f(g[c]['precision']),f(g[c]['recall']),'<u>'+f(g[c]['f1'])+'</u>' if float(g[c]['f1'])==best else f(g[c]['f1']),f"{100*(float(g[c]['f1'])-float(g['nearest']['f1'])):+.2f}"] for i,c in enumerate(C)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,k in zip(axs,['precision','recall','f1']):
 g=get();v=[float(g[c][k])*100 for c in C];ax.bar(range(3),v,color=colors);ax.set_xticks(range(3));ax.set_xticklabels(L,rotation=12);ax.set_ylim(0,100);ax.set_title(k.title()+' (%)');ax.grid(axis='y',alpha=.15)
 for i,x in enumerate(v):ax.text(i,x+1,f'{x:.2f}',ha='center')
fig.suptitle('One versus two neighbours · all 12 judges · fixed 50/50',fontweight='bold');fig.tight_layout();fig.savefig(R/'overall.png');fig.savefig(R/'overall.pdf');plt.close(fig)
models=sorted({r['model'] for r in rows},key=lambda m:float(get(dim='model',group=m)['nearest2']['f1'])-float(get(dim='model',group=m)['nearest']['f1']))
fig,ax=plt.subplots(figsize=(9,6));v=[100*(float(get(dim='model',group=m)['nearest2']['f1'])-float(get(dim='model',group=m)['nearest']['f1'])) for m in models];ax.barh(models,v,color=['#247a59' if x>=0 else '#b46d66' for x in v]);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Two NNs minus one NN: F1 percentage points');ax.set_title('Does a second neighbour help?');ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(R/'judges.png');fig.savefig(R/'judges.pdf');plt.close(fig)
b='<h1>One versus two nearest neighbours</h1><p>23 October 2026 · 12 judges × 5 backbones × 10 datasets · Fixed probe 50%, judge 50.</p><p>Existing responses only. Each judge uses identical test cases across control, one NN and two NNs; failures do not exclude cases from other judges. Full neighbour information is retained. Original sampling weights and equal-cell macro averaging are preserved.</p><img src="overall.png">'
for co,title in [('all10','Overall'),('safety6','Safety / toxicity (6 datasets)'),('hallucination3','Hallucination (3 datasets)'),('high_stakes','High-stakes (1 dataset)'),('toxicity4','Toxicity subset (4 datasets)')]:
 b+='<h2>'+title+'</h2>'+table(['Condition','Precision','Recall','F1','ΔF1 vs one NN'],compare(get(co)))
for title,local in [('Local judges',True),('API judges',False)]:
 gg={}
 for c in C:
  rr=[r for r in rows if r['condition']==c and ((r['model'] in ['qwen','gemma','mistral','olmo','glm','qwen122'])==local)]
  gg[c]={k:np.mean([float(r[k]) for r in rr]) for k in ['precision','recall','f1']}
 b+='<h2>'+title+'</h2>'+table(['Condition','Precision','Recall','F1','ΔF1 vs one NN'],compare(gg))
b+='<h2>By judge</h2><img src="judges.png">'
for dim,title in [('model','Judge'),('dataset','Dataset'),('backbone','Backbone')]:
 rr=[]
 for group in sorted({r['group'] for r in s if r['cohort']=='all10' and r['dimension']==dim}):
  g=get(dim=dim,group=group);best=max(float(r['f1']) for r in g.values());rr.append([group]+[('<u>'+f(g[c]['f1'])+'</u>') if float(g[c]['f1'])==best else f(g[c]['f1']) for c in C]+[f"{100*(float(g['nearest2'][k])-float(g['nearest'][k])):+.2f}" for k in ['f1','precision','recall']])
 b+='<h2>By '+title.lower()+'</h2>'+table([title,'Control F1','One NN F1','Two NNs F1','ΔF1','Δprecision','Δrecall'],rr)
b+='<h2>Coverage and interpretation</h2>'+table(['Judge','Matched test cases','Available'],[[m,sum(int(r['retained']) for r in cov if r['model']==m),25000] for m in sorted({r['model'] for r in cov})])
b+='<p>API scores accept the final standalone integer 0–100 only for normally completed ok/invalid responses; raw text is preserved. Blocks, errors, ambiguous and truncated responses are unavailable. Local results use original full-population denominators; API results use matched weighted cohorts. Cross-judge comparisons can reflect differing coverage. These are descriptive results from a previously inspected test set, not a fresh significance test. No thresholds were retuned. The four-dataset toxicity subset overlaps the six-dataset safety group.</p><p><a href="cells.csv">Per-cell metrics</a> · <a href="summary.csv">All precision/recall/F1 breakdowns</a> · <a href="coverage.csv">Coverage</a> · <a href="VERIFIED.json">Verification</a> · <a href="overall.pdf">Overall PDF</a> · <a href="judges.pdf">Judge PDF</a></p>'
(R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>One versus two NNs</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#182738;background:#fafbfd}img{width:100%}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#edf2f5}.scroll{overflow-x:auto}u{text-underline-offset:3px;text-decoration-thickness:2px}</style>'+b)
assert len(rows)==1800 and all(np.isfinite(float(r[k])) and -1e-12<=float(r[k])<=1+1e-12 for r in rows for k in ['precision','recall','f1','accuracy'])
old=list(csv.DictReader(open(R.parents[1]/'main_performance_all12_20261023/results/cells.csv')));idx={(r['model'],r['backbone'],r['dataset'],r['condition']):r for r in old if r['policy']=='fixed50'};n=0
for r in rows:
 if r['model'] in ['qwen','gemma','mistral','olmo','glm','qwen122'] and r['condition'] in ['baseline','nearest']:
  prev=idx[tuple(r[k] for k in ['model','backbone','dataset','condition'])]
  for k in ['precision','recall','f1','accuracy']:assert abs(float(prev[k])-float(r[k]))<1e-12
  n+=1
receipt=json.load(open(R/'VERIFIED.json'));receipt['historical_main_rows_replayed']=n;(R/'VERIFIED.json').write_text(json.dumps(receipt,indent=2))
for co in ['all10','safety6','hallucination3','high_stakes','toxicity4']:print(co,compare(get(co)))
print('Verified',n,'historical rows')
