from pathlib import Path
import os,json,html
os.environ['MPLCONFIGDIR']='/tmp/nn-selector-granularity-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results'
s=pd.read_csv(O/'summary.csv');c=pd.read_csv(O/'training_counts.csv');d=pd.read_csv(O/'deltas.csv');choices=pd.read_csv(O/'model_choices.csv')
scope_names={'pool_all':'Pooled per judge (all 10 datasets)','pool_toxicity':'Pooled per judge (toxicity only)','dataset':'Separate judge × dataset','cell':'Separate judge × dataset × backbone'}
names={'always_nn':'Always NN','cosine_gate':'Cosine gate','control':'No explanation'}
for scope,description in scope_names.items():
 for variant,label in [('logistic','Logistic'),('knn','k-NN (validation selected)'),('knn_euclidean','k-NN Euclidean'),('knn_cosine','k-NN cosine')]:names[scope+'__'+variant]=description+' — '+label
main=['always_nn','cosine_gate']+[a+'__'+b for a in scope_names for b in ['logistic','knn']]
def table(df):
 return df.to_html(index=False,escape=False,border=0,classes='results')
def formatted(df,numeric):
 df=df.copy()
 for col in numeric:
  values=df[col].astype(float);best=values.max();df[col]=[('<u>'+f'{x:.3f}'+'</u>') if abs(x-best)<1e-9 else f'{x:.3f}' for x in values]
 return table(df)
parts=['''<!doctype html><html><head><meta charset="utf-8"><title>Separate versus pooled NN selectors</title><style>body{font-family:system-ui,sans-serif;margin:32px auto;max-width:1280px;color:#182234;padding:0 24px;line-height:1.5}h1{font-size:32px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{padding:9px 12px;border-bottom:1px solid #dfe4eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef2f7}u{text-decoration-thickness:2px;text-underline-offset:4px}img{width:100%}.note{padding:16px;background:#f1f5fb;border-left:4px solid #4775aa}.scroll{overflow:auto}a{color:#1b5c9e}</style></head><body><h1>How local should the NN selector be?</h1><p>Logistic regression and k-NN, with the same validation folds, features, original judge/probe thresholds, test cases and weights. Toxicity means Aegis, BeaverTails, ToxicChat and ToxiGen. Underlining marks the largest F1 in each comparison column.</p>''']
for cohort,label in [('toxicity4','Four toxicity datasets'),('all','All ten datasets')]:
 h=s[(s.cohort==cohort)&(s.dimension=='overall')&s.policy.isin(['fixed50','anchor_average'])&s.method.isin(main)]
 p=h.pivot(index='method',columns='policy',values='f1').reindex([m for m in main if m in h.method.unique()])*100
 p=p.rename(columns={'fixed50':'Fixed 50/50 F1 (%)','anchor_average':'Shared-anchor average F1 (%)'}).reset_index();p['method']=p.method.map(names);p=p.rename(columns={'method':'Selector'})
 parts += ['<h2>'+label+'</h2>',formatted(p,list(p.columns[1:]))]
parts+=['<div class="note">Fixed 50/50 keeps both the probe and judge at 50. The shared-anchor analysis reuses the existing control-, NN- and saliency-selected probe/judge thresholds, holding each pair fixed across all selector methods, then averages the three evaluations. Only the inclusion selector and its cutoff are newly fitted here. F1 is macro-averaged across backbone–judge–dataset cells.</div>','<img src="granularity.png" alt="F1 improvement over always showing NN by training granularity">']
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=False)
for row,cohort in enumerate(['toxicity4','all']):
 for col,pol in enumerate(['fixed50','anchor_average']):
  ax=axes[row,col];hh=s[(s.cohort==cohort)&(s.dimension=='overall')&(s.policy==pol)].set_index('method');ref=hh.loc['always_nn','f1'];scopes=list(scope_names) if cohort=='toxicity4' else ['pool_all','dataset','cell'];ys=np.arange(len(scopes))
  for off,var,color,label in [(-.13,'logistic','#2864a1','Logistic'),(.13,'knn','#db8538','k-NN')]:
   vals=np.array([(hh.loc[sc+'__'+var,'f1']-ref)*100 for sc in scopes]);ax.scatter(vals,ys+off,color=color,label=label,s=52)
   for x,y in zip(vals,ys+off):ax.annotate(f'{x:+.2f}',(x,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
  ax.axvline(0,color='#999999',lw=1);ax.set_yticks(ys);ax.set_yticklabels([{'pool_all':'All-data pool','pool_toxicity':'Toxicity pool','dataset':'Judge × dataset','cell':'Judge × dataset × backbone'}[sc] for sc in scopes]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(('Toxicity 4' if row==0 else 'All 10 datasets')+' · '+('Fixed 50/50' if col==0 else 'Shared-anchor average'));ax.set_xlabel('F1 change versus always NN (percentage points)');ax.margins(x=.25)
axes[0,0].legend(loc='best',fontsize=9);fig.tight_layout();fig.savefig(O/'granularity.png',dpi=180);fig.savefig(O/'granularity.pdf');plt.close(fig)
counts=[]
for cohort in ['all','toxicity4']:
 for scope in scope_names:
  h=c[(c.policy=='fixed50')&(c.scope==scope)]
  if cohort=='all' and scope=='pool_toxicity':continue
  if cohort=='toxicity4' and scope in ['dataset','cell']:h=h[h.group.str.split('__').str[-1].isin(['aegis','beavertails','toxicchat','toxigen'])]
  counts.append(dict(cohort=cohort,scope=scope,models=len(h),min=int(h.training_flips.min()),median=float(h.training_flips.median()),max=int(h.training_flips.max()),constant=int(h.constant_final.sum()),empty=int(h.empty_final.sum()),below20=int((h.training_flips<20).sum()),cv_constant=int((h.empty_cv_fits+h.oneclass_cv_fits).sum())))
pd.DataFrame(counts).to_csv(O/'training_size_summary.csv',index=False)
parts+=['<h2>How much usable training data does each selector get?</h2><p>Fixed 50/50 counts below are validation disagreements: cases where control and NN make different binary decisions. The target is whether NN is correct. The remaining validation cases still contribute to selection of the inclusion cutoff. No new labels were collected.</p>']
for cohort,label in [('toxicity4','Toxicity evaluation'),('all','All datasets')]:
 h=pd.DataFrame(counts);h=h[h.cohort==cohort].drop(columns='cohort');h['scope']=h.scope.map(scope_names);parts+=['<h3>'+label+'</h3>',table(h)]
parts+=['<p>Empty or one-class training subsets use the original constant-score fallback; no cells are discarded. “CV constant” counts the three fold fits, whereas “constant” counts final refits. Pooled all-data counts remain all-data counts even when evaluation is restricted to toxicity.</p>']
for cohort,title in [('toxicity4','Toxicity by judge'),('all','All datasets by judge')]:
 parts+=['<h2>'+title+'</h2>']
 for pol,label in [('fixed50','Fixed 50/50'),('anchor_average','Shared-anchor average')]:
  h=s[(s.cohort==cohort)&(s.dimension=='model')&(s.policy==pol)&s.method.isin(main)];p=h.pivot(index='group',columns='method',values='f1')*100;p=p.reindex(columns=[m for m in main if m in p.columns]);p.columns=[names[x] for x in p.columns];p=p.reset_index().rename(columns={'group':'Judge'})
  # Underline the best method within each judge row, not across unrelated judges.
  for i in p.index:
   best=max(float(p.loc[i,x]) for x in p.columns[1:])
   for col in p.columns[1:]:
    x=float(p.loc[i,col]);p.loc[i,col]=('<u>'+f'{x:.2f}'+'</u>') if abs(x-best)<1e-9 else f'{x:.2f}'
  parts+=['<h3>'+label+' F1 (%)</h3><div class="scroll">',table(p),'</div>']
parts+=['<h2>Dataset breakdown</h2>']
for pol,label in [('fixed50','Fixed 50/50'),('anchor_average','Shared-anchor average')]:
 h=s[(s.cohort=='all')&(s.dimension=='dataset')&(s.policy==pol)&s.method.isin(main)];p=h.pivot(index='group',columns='method',values='f1')*100;p=p.reindex(columns=[m for m in main if m in p.columns]);p.columns=[names[x] for x in p.columns];p=p.reset_index().rename(columns={'group':'Dataset'});parts+=['<h3>'+label+' F1 (%)</h3><div class="scroll">',table(p.round(3)),'</div>']
parts+=['<h2>Precision, recall, accuracy, and both k-NN metrics</h2><details><summary>Full overall tables</summary>']
for cohort in ['toxicity4','all']:
 for pol in ['fixed50','anchor_average']:
  h=s[(s.cohort==cohort)&(s.dimension=='overall')&(s.policy==pol)].copy();h['method']=h.method.map(names);h=h[['method','f1','precision','recall','accuracy','nn_rate']];h.iloc[:,1:]=h.iloc[:,1:]*100;parts+=['<h3>'+cohort+' · '+pol+'</h3>',table(h.round(3))]
parts+=['</details><h2>Validation selection and reproducibility</h2><p>Logistic C ∈ {0.1, 1}. k-NN k ∈ {1, 3, 5, 15, 51, 151}, capped to available training rows, with Euclidean or cosine distances on the same standardized 31 features. The two small k values were added to every training scope, including the pooled comparison. These distances are between selector feature vectors, not original language-model embeddings. The original cosine gate instead uses target–training-neighbour embedding similarity.</p><p>For every scope, use three identical target-grouped validation folds; fit only on control/NN disagreements; choose the gate cutoff separately per backbone–judge–dataset cell. Select hyperparameters by mean cell validation F1. All choices and models were frozen before evaluating test. The original pooled logistic scores reproduce exactly. Group size varies substantially; sparse final fits and fold fits are explicitly retained.</p><p>This is an exploratory follow-up on an already examined test set. The reported best test mean is descriptive; no claim that the largest test mean was selected prospectively. All methods keep the same cases, splits, probability scores and weights. No LLM inference or probe retraining.</p><p><a href="summary.csv">All aggregate results</a> · <a href="cell_metrics.csv">Every cell</a> · <a href="training_counts.csv">Training counts</a> · <a href="model_choices.csv">Validation-selected settings</a> · <a href="deltas.csv">Paired deltas and wins/losses</a> · <a href="COMPLETED.json">Integrity receipt</a> · <a href="granularity.pdf">Plot PDF</a></p></body></html>']
(O/'report.html').write_text('\n'.join(parts))
print(pd.DataFrame(counts).to_string(index=False))
