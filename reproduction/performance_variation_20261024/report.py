from pathlib import Path
import json,csv,html,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;d=json.loads((R/'results.json').read_text())
rows=list(csv.DictReader((R.parent/'main_performance_1nn_2nn_all12_20261023/results/cells.csv').open()))
models=sorted({r['model'] for r in rows});datasets=sorted({r['dataset'] for r in rows})
names={'qwen':'Qwen27','gemma':'Gemma31','qwen122':'Qwen122','mistral':'Mistral','olmo':'OLMo','glm':'GLM','claude4.5-sonnet':'Sonnet 4.5','claude4.8-opus':'Opus 4.8','claude-opus-5.5':'Opus 5.5','gpt-4o':'GPT-4o','gpt-astra':'GPT Astra','gpt-5.5':'GPT-5.5'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,2,figsize=(15,6.5));out=[]
for ax,policy,title in zip(axs,['fixed50','tuned_shared'],['Fixed probe 50 / judge 50','Validation tuning: three shared anchors']):
 groups=collections.defaultdict(dict)
 for r in rows:
  if r['policy']==policy and r['condition'] in ['baseline','nearest2']:groups[r['model'],r['dataset'],r['backbone'],r['anchor']][r['condition']]=float(r['f1'])*100
 matrix=np.zeros((len(models),len(datasets)))
 for i,m in enumerate(models):
  for j,ds in enumerate(datasets):
   v=[x['nearest2']-x['baseline'] for k,x in groups.items() if k[0]==m and k[1]==ds];matrix[i,j]=np.mean(v);out.append(dict(policy=policy,judge=m,dataset=ds,delta_f1_pp=float(matrix[i,j])))
 bound=np.ceil(abs(matrix).max());im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_title(title)
 ax.set_xticks(range(len(datasets)));ax.set_xticklabels([x.replace('_',' ') for x in datasets],rotation=55,ha='right');ax.set_yticks(range(len(models)));ax.set_yticklabels([names[x] for x in models]);ax.set_ylabel('Judge')
 for i in range(len(models)):
  for j in range(len(datasets)):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.55*bound else '#14202b',fontsize=8)
 fig.colorbar(im,ax=ax,shrink=.8,label='2NN minus control F1 (percentage points)')
fig.suptitle('Neighbour benefit depends on the judge–dataset pairing',fontsize=15)
fig.text(.5,.01,'Each entry averages five backbones. Colour scales differ between panels; numbers are directly comparable.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(R/'judge_dataset.png',dpi=160);fig.savefig(R/'judge_dataset.pdf');plt.close(fig)
with (R/'judge_dataset.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
labels=['Fixed: 2NN F1','Tuned: 2NN F1','Fixed: 2NN gain','Tuned: 2NN gain'];keys=['all12/fixed50/two_nn_f1','all12/tuned_shared/two_nn_f1','all12/fixed50/two_nn_gain','all12/tuned_shared/two_nn_gain'];factors=['dataset','model','backbone','model x dataset','other interactions'];colors=['#4873a1','#d39b39','#377e67','#9a6aa8','#bec7d0'];fig,ax=plt.subplots(figsize=(10,4));left=np.zeros(4)
for factor,color in zip(factors,colors):
 vals=[]
 for key in keys:
  sh=d[key]['shares'];vals.append(sh[factor] if factor!='other interactions' else sum(v for k,v in sh.items() if k not in factors))
 ax.barh(range(4),vals,left=left,color=color,label={'model':'Judge','dataset':'Dataset','backbone':'Backbone','model x dataset':'Judge × dataset','other interactions':'Other interactions'}[factor])
 for i,v in enumerate(vals):
  if v>=5:ax.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',color='white' if factor!='other interactions' else '#223044',fontsize=9)
 left+=vals
ax.set_yticks(range(4));ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Share of observed between-cell variance (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False);ax.set_title('Dataset dominates F1; judge–dataset interactions dominate tuned NN gains');fig.tight_layout();fig.savefig(R/'variance_partition.png',dpi=160);fig.savefig(R/'variance_partition.pdf');plt.close(fig)
def tab(headers,rr):return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rr)+'</table>'
body='<h1>Which factors vary most?</h1><p>Exact descriptive decomposition across 600 judge × backbone × dataset cells. Fixed and tuned scores come from the four-condition matched analysis; tuned cell scores average the three shared-threshold anchors before decomposition. No model calls, threshold changes or significance tests.</p><img src="variance_partition.png"><h2>Judge–dataset divergence</h2><img src="judge_dataset.png"><p>The charts partition observed variation, not causal effects. Dataset prevalence and difficulty affect F1; model-specific API availability also affects estimates. With one aggregated value per cell, interaction components contain genuine heterogeneity and estimation noise. A small backbone main effect does not imply identical performance for every backbone.</p><h2>GLM sensitivity</h2>'
rr=[]
for policy in ['fixed50','tuned_shared']:
 for sens in ['all12','without_glm']:
  a=d[sens+'/'+policy+'/two_nn_gain'];rr.append([policy,sens,f"{a['mean']:+.3f}",*[f"{a['shares'][k]:.2f}" for k in ['model','backbone','dataset','model x dataset']]])
body+=tab(['Policy','Judges','Mean ΔF1 (pp)','Judge share (%)','Backbone share (%)','Dataset share (%)','Judge × dataset share (%)'],rr)
body+='<p><a href="variance_components.csv">All variance components</a> · <a href="marginal_means.csv">All marginal means</a> · <a href="judge_dataset.csv">Judge/dataset deltas</a> · <a href="METHOD.json">Method and source hash</a></p>'
(R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Performance variation</title><style>body{font:16px system-ui;color:#1b2b3a;max-width:1250px;margin:35px auto;padding:0 20px;line-height:1.6}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}</style>'+body)
for policy in ['fixed50','tuned_shared']:
 print(policy)
 for ds in ['toxigen','halueval','toxicchat','fava']:
  a=sorted([r for r in out if r['policy']==policy and r['dataset']==ds],key=lambda x:x['delta_f1_pp']);print(ds,'min',a[0]['judge'],round(a[0]['delta_f1_pp'],2),'max',a[-1]['judge'],round(a[-1]['delta_f1_pp'],2))
