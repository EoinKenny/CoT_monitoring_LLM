import os
os.environ['MPLCONFIGDIR']='/tmp/nn-selector-truth-mpl'
from pathlib import Path
import json,hashlib,gzip
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';SRC=P/'nn_selector_labels_20261019/results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cols=['source_index','model','scope','policy','backbone','dataset','true_label','weight','referred','show_nn','control','nearest','gate_prediction','outcome','majority_labelled_similarity','majority_important_contrast']
z=pd.read_csv(SRC/'joined_gate_annotations.csv.gz',usecols=cols)
with gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt') as handle:design=json.load(handle)
local=['qwen','gemma','mistral','olmo','glm','qwen122']
assert (z.gate_prediction==np.where(z.show_nn,z.nearest,z.control)).all()
z['control_correct']=z.control==z.true_label;z['nn_correct']=z.nearest==z.true_label;z['gate_correct']=z.gate_prediction==z.true_label
z['actual_outcome']=np.where(z.control!=z.nearest,np.where(z.nn_correct,'helpful','harmful'),np.where(z.nn_correct,'both_correct','both_wrong'))
assert (z.loc[z.control!=z.nearest,'actual_outcome']==z.loc[z.control!=z.nearest,'outcome']).all()
matrices=[];summaries=[];outcomes=[];basepol=['fixed50','baseline_anchor','nearest_anchor','saliency_anchor']
for scope in ['pool_all','pool_toxicity']:
 for pol in basepol+['anchor_average']:
  h=z[(z.scope==scope)&(z.policy.isin(basepol[1:]) if pol=='anchor_average' else z.policy.eq(pol))].copy()
  denominators=h.groupby(['model','backbone','dataset','policy']).weight.transform('sum').to_numpy()
  for i,row in enumerate(h[['model','backbone','dataset']].itertuples(index=False)):
   if row.model in local:denominators[i]=design['thresholds'][row.backbone][row.dataset]['populations']['test']['n']
  h['analysis_weight']=h.weight/denominators
  h=h[h.referred].copy()
  for dim in ['overall','model','dataset']:
   for group,g in ([('ALL',h)] if dim=='overall' else h.groupby(dim)):
    meta=dict(scope=scope,policy=pol,dimension=dim,group=group)
    for outcome,v in g.groupby('actual_outcome'):
     outcomes.append(dict(**meta,outcome=outcome,n=len(v),include=int(v.show_nn.sum()),suppress=int((~v.show_nn).sum()),include_rate=v.show_nn.mean(),unique_contexts=v.source_index.nunique()))
    flip=g[g.control!=g.nearest].copy()
    for rule in ['logistic','similarity_yes','contrast_no']:
     v=flip.copy()
     if rule=='logistic':decision=v.show_nn.to_numpy()
     else:
      col='majority_labelled_similarity' if rule=='similarity_yes' else 'majority_important_contrast';v=v[v[col].isin(['yes','no'])];decision=v[col].eq('yes' if rule=='similarity_yes' else 'no').to_numpy()
     actual=v.nn_correct.to_numpy()
     for weighting,w in [('raw',np.ones(len(v))),('weighted',v.analysis_weight.to_numpy())]:
      tab=np.zeros((2,2));ns=np.zeros((2,2),int)
      for i,truth in enumerate([True,False]):
       for j,include in enumerate([True,False]):
        mask=(actual==truth)&(decision==include);ns[i,j]=mask.sum();tab[i,j]=w[mask].sum()
      tp,fn=tab[0];fp,tn=tab[1];total=tab.sum();denom=np.sqrt(tab.sum(axis=1).prod()*tab.sum(axis=0).prod());phi=(tp*tn-fp*fn)/denom if denom else np.nan
      summ=dict(**meta,rule=rule,weighting=weighting,n=len(v),excluded_unknown=len(flip)-len(v),unique_contexts=v.source_index.nunique(),selection_accuracy=(tp+tn)/total if total else np.nan,keep_helpful=tp/(tp+fn) if tp+fn else np.nan,suppress_harmful=tn/(tn+fp) if tn+fp else np.nan,helpful_given_include=tp/(tp+fp) if tp+fp else np.nan,harmful_given_suppress=tn/(tn+fn) if tn+fn else np.nan,balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if (tp+fn)*(tn+fp) else np.nan,phi=phi,net_correct_vs_always_nn=tn-fn,net_correct_vs_control=tp-fp,always_nn_accuracy=(tp+fn)/total if total else np.nan,control_accuracy=(tn+fp)/total if total else np.nan)
      summaries.append(summ)
      for i,truth in enumerate(['helpful','harmful']):
       for j,decisionname in enumerate(['include','suppress']):matrices.append(dict(**meta,rule=rule,weighting=weighting,truth=truth,decision=decisionname,n=int(ns[i,j]),mass=tab[i,j],row_fraction=tab[i,j]/tab[i].sum() if tab[i].sum() else np.nan))
pd.DataFrame(matrices).to_csv(O/'matrices.csv',index=False);pd.DataFrame(summaries).to_csv(O/'summary.csv',index=False);pd.DataFrame(outcomes).to_csv(O/'all_outcomes.csv',index=False)
m=pd.DataFrame(matrices);s=pd.DataFrame(summaries);out=pd.DataFrame(outcomes)
fig,axes=plt.subplots(1,3,figsize=(14,4.9))
for ax,rule,title in zip(axes,['logistic','similarity_yes','contrast_no'],['Logistic selector','Include if similarity = yes','Include if contrast = no']):
 h=m[(m.scope=='pool_all')&(m.policy=='fixed50')&(m.dimension=='overall')&(m.rule==rule)&(m.weighting=='raw')];mat=h.pivot(index='truth',columns='decision',values='row_fraction').reindex(index=['helpful','harmful'],columns=['include','suppress']);ns=h.pivot(index='truth',columns='decision',values='n').reindex(index=['helpful','harmful'],columns=['include','suppress']);ax.imshow(mat.to_numpy(),vmin=0,vmax=1,cmap='Blues')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{mat.iloc[i,j]*100:.1f}%\n(n={int(ns.iloc[i,j]):,})',ha='center',va='center',fontsize=12,color='white' if mat.iloc[i,j]>.55 else '#172332')
 ax.set_xticks([0,1]);ax.set_xticklabels(['Include NN','Suppress NN']);ax.set_yticks([0,1]);ax.set_yticklabels(['NN helps\nControl wrong → NN right','NN harms\nControl right → NN wrong'] if rule=='logistic' else ['NN helps','NN harms']);ax.set_title(title,fontsize=12)
fig.suptitle('Against actual test labels · toxicity · fixed 50/50\nOnly cases where control and NN disagree; raw percentages within each row',fontsize=14);fig.tight_layout(rect=[0,0,1,.89]);fig.savefig(O/'ground_truth_matrices.png',dpi=180,bbox_inches='tight');fig.savefig(O/'ground_truth_matrices.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for ax,rule,title in zip(axes,['logistic','similarity_yes','contrast_no'],['Logistic selector','Include if similarity = yes','Include if contrast = no']):
 h=m[(m.scope=='pool_all')&(m.policy=='fixed50')&(m.dimension=='overall')&(m.rule==rule)&(m.weighting=='weighted')];mat=h.pivot(index='truth',columns='decision',values='row_fraction').reindex(index=['helpful','harmful'],columns=['include','suppress']);ax.imshow(mat.to_numpy(),vmin=0,vmax=1,cmap='Blues')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{mat.iloc[i,j]*100:.1f}%',ha='center',va='center',fontsize=15,color='white' if mat.iloc[i,j]>.55 else '#172332')
 ax.set_xticks([0,1]);ax.set_xticklabels(['Include NN','Suppress NN']);ax.set_yticks([0,1]);ax.set_yticklabels(['NN helps','NN harms']);ax.set_title(title,fontsize=12)
fig.suptitle('Same test decisions, using the main evaluation weights\nDesign weights and equal cell population mass; row percentages',fontsize=14);fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(O/'weighted_matrices.png',dpi=180,bbox_inches='tight');fig.savefig(O/'weighted_matrices.pdf',bbox_inches='tight');plt.close(fig)
parts=['<!doctype html><html><head><meta charset="utf-8"><title>Selector decisions against test ground truth</title><style>body{font-family:system-ui;max-width:1280px;margin:32px auto;padding:0 24px;color:#172332;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%}h2{margin-top:36px}.scroll{overflow:auto}</style></head><body><h1>Does the selector make the right inclusion decision?</h1><p>Ground truth is whether the original NN decision is correct according to the test dataset label. If control and NN disagree, exactly one is correct: a helpful NN should be included; a harmful NN should be suppressed. Unchanged predictions are evaluated separately because neither inclusion choice changes correctness.</p><p>Primary analysis: frozen all-dataset pooled logistic selector, four toxicity datasets, fixed probe/judge 50. The annotation rules are descriptive comparisons using earlier majority labels, not newly trained policies: include when labelled similarity=yes, or important contrast=no. Unknown annotations are excluded from the respective comparison. No new inference or fitting.</p><img src="ground_truth_matrices.png">']
parts+=['<h2>The weighting distinction matters</h2><p>In raw counts, logistic suppresses 1,016 harmful flips but also suppresses 1,129 helpful flips: 113 fewer correct answers than always NN. The evaluation weights can change this comparison because sampled cases represent different population mass and cells are averaged equally. The weighted matrices below use the exact original population denominator for each cell, so their net correctness differences reconcile with the main macro accuracy comparison.</p><img src="weighted_matrices.png">']
for scope,title in [('pool_all','Original pooled logistic'),('pool_toxicity','Toxicity-only pooled logistic')]:
 parts+=['<h2>'+title+'</h2>']
 for pol in ['fixed50','anchor_average']:
  h=s[(s.scope==scope)&(s.policy==pol)&(s.dimension=='overall')].copy()
  num=['selection_accuracy','always_nn_accuracy','control_accuracy','keep_helpful','suppress_harmful','helpful_given_include','harmful_given_suppress','balanced_accuracy']
  for col in num:h[col]*=100
  parts+=['<h3>'+pol+'</h3><div class="scroll">',h[['rule','weighting','n','excluded_unknown']+num+['phi','net_correct_vs_always_nn']].round(3).to_html(index=False,border=0),'</div>']
parts+=['<h2>All referrals, including unchanged predictions</h2>',out.query("scope=='pool_all' and policy=='fixed50' and dimension=='overall'").to_html(index=False,border=0),'<p>Both-correct and both-wrong cases are not false positives or true negatives for an inclusion selector: showing or suppressing the explanation leaves the class prediction unchanged.</p>','<h2>By outcome judge: fixed 50/50 logistic selector</h2>']
h=s.query("scope=='pool_all' and policy=='fixed50' and dimension=='model' and rule=='logistic' and weighting=='raw'").copy()
for col in ['selection_accuracy','keep_helpful','suppress_harmful','always_nn_accuracy']:h[col]*=100
parts+=[h[['group','n','selection_accuracy','always_nn_accuracy','keep_helpful','suppress_harmful','net_correct_vs_always_nn']].round(3).to_html(index=False,border=0),'<p>Numbers count outcome-judge × backbone × target decisions and can repeat an annotated context. Weighted summaries divide original design weights by the original evaluation population denominator in each backbone–judge–dataset–policy cell. Local judges use full population sizes; API judges use original weighted valid support, matching the main evaluation. Weighted net-correct entries are population-normalized mass, not numbers of examples. Selection accuracy here is measured only on control/NN disagreements; it is not the full pipeline F1 or overall accuracy. Anchor averages combine the three frozen shared-threshold evaluations. Correlations and annotation comparisons are descriptive, not causal evidence of a mechanism.</p><p><a href="matrices.csv">All confusion matrices</a> · <a href="summary.csv">Judge and dataset summaries</a> · <a href="all_outcomes.csv">Unchanged cases</a> · <a href="ground_truth_matrices.pdf">PDF figure</a></p></body></html>']
(O/'report.html').write_text('\n'.join(parts))
# Explicit arithmetic and partition checks.
main=m.query("scope=='pool_all' and policy=='fixed50' and dimension=='overall' and weighting=='raw'")
assert (main.groupby('rule').n.sum()==7047).all();assert (main.groupby(['rule','truth']).row_fraction.sum()-1).abs().max()<1e-10
counts=out.query("scope=='pool_all' and policy=='fixed50' and dimension=='overall'");assert counts.n.sum()==36743
truthcounts=main[main.rule=='logistic'].set_index(['truth','decision']).n
q=z[(z.scope=='pool_all')&(z.policy=='fixed50')&z.referred];net=int(q.gate_correct.astype(int).sum()-q.nn_correct.astype(int).sum());assert net==int(truthcounts['harmful','suppress']-truthcounts['helpful','suppress'])
prior=pd.read_csv(P/'nn_selector_granularity_20261019/results/summary.csv');prior=prior.query("cohort=='toxicity4' and dimension=='overall' and policy=='fixed50'").set_index('method');weightednet=float(s.query("scope=='pool_all' and policy=='fixed50' and dimension=='overall' and rule=='logistic' and weighting=='weighted'").net_correct_vs_always_nn.iloc[0]);assert abs(weightednet/200-(prior.loc['pool_all__logistic','accuracy']-prior.loc['always_nn','accuracy']))<1e-10
(O/'COMPLETED.json').write_text(json.dumps(dict(source_receipt_sha256=sha(SRC/'COMPLETED.json'),joined_source_sha256=sha(SRC/'joined_gate_annotations.csv.gz'),script_sha256=sha(Path(__file__)),referral_decisions=len(q),unique_contexts=q.source_index.nunique(),disagreement_decisions=7047,net_correct_vs_always_nn=net,validated=True,new_fits=0,new_llm_calls=0),indent=2))
print(main.to_string(index=False));print(s.query("scope=='pool_all' and policy in ['fixed50','anchor_average'] and dimension=='overall' and weighting=='raw'").to_string(index=False));print(counts.to_string(index=False))
