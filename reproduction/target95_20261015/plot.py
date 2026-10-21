from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
R=Path(__file__).resolve().parent;O=R/'plots';O.mkdir(exist_ok=True)
P={'baseline':'#64748b','saliency':'#d88b25','nearest':'#2989ac'}
L={'baseline':'No explanation','saliency':'Saliency','nearest':'1NN'};CS=list(P)
sns.set_theme(style='whitegrid',font_scale=1.08)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=180,facecolor='white')
 plt.close(fig)
pair=pd.read_csv(R/'results/combinations.csv');pair=pair[(pair.target=='recall95')|(pair.min_support==20)]
fig,axs=plt.subplots(2,2,figsize=(13,10));fig.subplots_adjust(left=.09,right=.97,bottom=.17,top=.86,hspace=.4,wspace=.19)
summary=[]
for ri,(target,tlabel) in enumerate([('recall95','Validation recall ≥95%'),('precision95','Validation precision ≥95%')]):
 for co,metric in enumerate(['precision','recall']):
  ax=axs[ri,co];sub=pair[pair.target==target].copy();sub[metric]*=100
  sns.swarmplot(data=sub,x='condition',y=metric,hue='condition',order=CS,hue_order=CS,palette=P,legend=False,size=4,ax=ax)
  constrained=(target=='recall95' and metric=='recall')or(target=='precision95' and metric=='precision')
  if constrained:ax.axhline(95,ls='--',lw=1.2,color='#263244')
  lo=max(0,np.floor(sub[metric].min()/5)*5-3);ax.set_ylim(lo,103)
  for i,c in enumerate(CS):
   mean=sub[sub.condition==c][metric].mean();ax.plot([i-.22,i+.22],[mean]*2,color='#172132',lw=2.5)
   ax.text(i,101.5,f'{mean:.2f}%',ha='center',va='top',weight='bold',color=P[c])
  ax.set_title(tlabel+'\nTest '+metric+(' (target)' if constrained else ' (maximised on validation)'),fontsize=12)
  ax.set_ylabel('Test '+metric+' (%)');ax.set_xlabel('');ax.set_xticks(range(3));ax.set_xticklabels([L[c] for c in CS]);ax.grid(False,axis='x')
fig.suptitle('What happens when we target 95%?',fontsize=20,weight='bold',x=.06,ha='left',y=.97)
fig.text(.06,.915,'Each dot = one backbone–judge average. Dashed line = the validation target, evaluated on unseen test cases.',fontsize=11)
fig.text(.06,.055,'50 points per condition. Recall target: all 10 datasets per point. Precision target: 6–10 matched feasible datasets per point\n(397 of 500 cells feasible for all conditions). Same datasets within each paired comparison; no test-based filtering.\nAt least 20 validation predicted positives for precision targeting. Weighted common valid-case cohort across all ten judges.',fontsize=10,color='#536174')
save(fig,'target95_swarm')

macro=pd.read_csv(R/'results/frontier_macro.csv');jp=pd.read_csv(R/'results/frontier_judges.csv')
def contours(ax,xlim,ylim):
 x=np.linspace(max(.01,xlim[0]),xlim[1],200);y=np.linspace(max(.01,ylim[0]),ylim[1],200);xx,yy=np.meshgrid(x,y)
 z=2*xx*yy/(xx+yy)
 c=ax.contour(xx,yy,z,levels=[50,60,70,75,80,82,84,86,88,90,92,95],colors='#c5cbd3',linewidths=.7,alpha=.8)
 ax.clabel(c,inline=True,fontsize=8,fmt=lambda v:f'F1 {v:.0f}')
def curves(ax,frame,legend=False):
 for c in CS:
  sub=frame[frame.condition==c].sort_values('beta_index')
  ax.plot(sub.recall.to_numpy()*100,sub.precision.to_numpy()*100,color=P[c],label=L[c],lw=2.2,marker='.',ms=3,alpha=.95)
  point=sub[sub.beta_index==31];ax.scatter(point.recall.to_numpy()*100,point.precision.to_numpy()*100,marker='*',s=150,color=P[c],edgecolors='white',linewidths=.6,zorder=5)
 if legend:ax.legend(frameon=False,loc='lower left')
 ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)')
fig,axs=plt.subplots(1,2,figsize=(14,7));fig.subplots_adjust(left=.075,right=.97,bottom=.22,top=.80,wspace=.20)
xlim=(max(0,np.floor(macro.recall.min()*100/5)*5-2),100);ylim=(max(0,np.floor(macro.precision.min()*100/5)*5-2),100)
central=macro[macro.beta_index.between(20,42)];zx=(central.recall.min()*100-1,central.recall.max()*100+1);zy=(central.precision.min()*100-1,central.precision.max()*100+1)
for ax,x,y,title in zip(axs,[xlim,zx],[ylim,zy],['Full range of operating preferences','Closer view around balanced preferences']):
 contours(ax,x,y);curves(ax,macro,ax is axs[0]);ax.set_xlim(x);ax.set_ylim(y);ax.set_title(title,fontsize=12)
fig.suptitle('Precision–recall trade-offs across explanation types',fontsize=20,weight='bold',x=.06,ha='left',y=.97)
fig.text(.06,.90,'63 validation-selected preferences, from precision-first to recall-first. Stars mark F1 optimisation (β = 1).',fontsize=11)
fig.text(.06,.055,'Each point averages the same 500 backbone–judge–dataset cells. Candidate policies require ≥20 validation predicted positives.\nCutoffs are selected on validation Pareto fronts; curves show test performance and can be non-monotonic. No test-selected envelope.\nGrey contours show F1 of the plotted mean precision and recall, not the mean per-cell F1. Common valid-case cohort.',fontsize=10,color='#536174')
save(fig,'precision_recall_tradeoffs')

fig,axs=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.23,wspace=.2)
for c in CS:
 sub=macro[(macro.condition==c)&macro.beta_index.between(1,61)].sort_values('beta_index');beta=pd.to_numeric(sub.beta).to_numpy()
 axs[0].plot(beta,sub.f1.to_numpy()*100,label=L[c],color=P[c],lw=2)
 base=macro[(macro.condition=='baseline')&macro.beta_index.between(1,61)].sort_values('beta_index')
 axs[1].plot(beta,(sub.f1.values-base.f1.values)*100,label=L[c],color=P[c],lw=2)
for ax in axs:
 ax.set_xscale('log');ax.set_xticks([.1,.25,.5,1,2,4,10]);ax.set_xticklabels(['0.1','0.25','0.5','1','2','4','10']);ax.set_xlabel('β preference: precision-first ← 1 → recall-first');ax.axvline(1,color='#b6bfcb',lw=1,ls='--')
axs[0].set_ylabel('Mean test F1 (%)');axs[0].legend(frameon=False)
axs[1].set_ylabel('Change in mean test F1 (pp)\nversus no explanation');axs[1].axhline(0,color='#9199a4',ls='--',lw=1)
fig.suptitle('How the explanation benefit changes with the objective',x=.06,ha='left',y=.96,weight='bold',fontsize=18)
fig.text(.06,.87,'Same validation preference for every explanation; all 500 cells retained at every point.',fontsize=11)
fig.text(.06,.055,'Fβ weights recall relative to precision: β = 1 is F1; larger β favours recall. Plotted outcomes are mean per-cell test F1.\nThese are descriptive paired comparisons, not independent replications or confidence intervals.',fontsize=10,color='#536174')
save(fig,'f1_by_preference')

names={'qwen':'Qwen27B','gemma':'Gemma31B','qwen122':'Qwen122B','glm':'GLM','olmo':'OLMo','mistral':'Mistral','claude4.5-sonnet':'Claude Sonnet','claude4.8-opus':'Claude Opus','gpt-astra':'GPT-Astra','gpt-4o':'GPT-4o'}
fig,axs=plt.subplots(2,5,figsize=(17,8),sharex=True,sharey=True);fig.subplots_adjust(left=.06,right=.98,bottom=.16,top=.84,wspace=.17,hspace=.28)
for ax,m in zip(axs.ravel(),names):
 curves(ax,jp[jp.model==m]);ax.set_title(names[m],fontsize=12);ax.set_xlim(35,100);ax.set_ylim(35,100);ax.set_xlabel('Recall (%)');ax.set_ylabel('Precision (%)')
axs[0,0].legend(frameon=False,fontsize=9,loc='lower left')
fig.suptitle('Precision–recall trade-offs by judge',x=.06,ha='left',y=.97,fontsize=20,weight='bold')
fig.text(.06,.90,'Each point averages five backbones × ten datasets. Stars mark validation F1 optimisation.',fontsize=11)
fig.text(.06,.045,'Same matched valid-case cohort across all ten judges. Validation-selected cutoffs; held-out test metrics.\nLines connect operating preferences, not a test-selected Pareto envelope. Axes are shared across panels.',fontsize=10,color='#536174')
save(fig,'tradeoffs_by_judge')

dataset=pd.read_csv(R/'results/dataset_nn_gains.csv')
frontds=pd.read_csv(R/'results/frontier_datasets.csv')
fig,axs=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1,2]});fig.subplots_adjust(left=.12,right=.97,bottom=.21,top=.80,wspace=.32)
order=list(dataset.dataset)
axs[0].barh(range(10),dataset.f1_nn_gain_pp,color=[P['nearest'] if x>=0 else '#c67845' for x in dataset.f1_nn_gain_pp])
axs[0].set_yticks(range(10));axs[0].set_yticklabels(order);axs[0].invert_yaxis();axs[0].axvline(0,color='#596373',lw=1);axs[0].set_xlabel('NN − no explanation (F1 points)');axs[0].set_title('Existing jointly F1-tuned policy',fontsize=12)
for i,v in enumerate(dataset.f1_nn_gain_pp):axs[0].text(v+(.025 if v>=0 else -.025),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=10)
axs[0].set_xlim(-.5,1.12)
z=frontds.pivot(index=['dataset','beta_index'],columns='condition',values='f1');z=((z.nearest-z.baseline)*100).unstack('beta_index').reindex(order)[list(range(1,62))]
sns.heatmap(z,cmap='RdBu',center=0,vmin=-3,vmax=3,ax=axs[1],yticklabels=True,xticklabels=False,cbar_kws={'label':'NN − no explanation (mean test F1 points)','shrink':.8})
axs[1].set_xticks([.5,12.5,21.5,30.5,39.5,48.5,60.5]);axs[1].set_xticklabels(['0.1','0.25','0.5','1','2','4','10']);axs[1].set_xlabel('β preference: precision-first ← 1 → recall-first');axs[1].set_ylabel('');axs[1].set_title('How the NN benefit changes across objectives',fontsize=12)
fig.suptitle('Where do nearest neighbours help?',fontsize=20,weight='bold',x=.06,ha='left',y=.97)
fig.text(.06,.90,'Each dataset averages the same five backbones and ten judges. Positive / blue = NN improves test F1.',fontsize=11)
fig.text(.06,.055,'Left: original validation-F1 policies. Right: the new Pareto sweep requires ≥20 validation predicted positives.\nAll results use the common valid-case cohort; dataset effects are descriptive averages. Heatmap colours saturate at ±3 pp.',fontsize=10,color='#536174')
save(fig,'dataset_nn_advantage')

html='<!doctype html><meta charset="utf-8"><title>Precision–recall trade-off study</title><style>body{font:17px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;color:#243044}img{width:100%}p{line-height:1.5}section{margin:40px 0}a{color:#216d8d}</style><h1>Explanations under different precision–recall objectives</h1><p>Ten judges, five backbones, ten datasets. Thresholds are chosen on validation only; every frontier curve retains the same test cohort. No new inference. F1 is one operating preference, not a guarantee of high precision or recall.</p>'
for name in ['precision_recall_tradeoffs','f1_by_preference','tradeoffs_by_judge','dataset_nn_advantage','target95_swarm']:
 html+=f'<section><img src="{name}.png" alt="{name}"><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></section>'
html+='<p>95% policies: infeasible cells remain recorded as unavailable. Their swarm compares identical feasible datasets across explanation conditions within each backbone–judge pair. This differs from the fixed-cohort trade-off curves. A validation point-estimate target is not a test guarantee.</p><p><a href="../results/summary.json">95% summaries and feasibility</a> · <a href="../results/frontier_macro.csv">Trade-off data</a> · <a href="../results/coverage.csv">Dataset coverage</a></p>'
(O/'report.html').write_text(html)
print('Plots complete')
