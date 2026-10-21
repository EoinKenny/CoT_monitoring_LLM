from pathlib import Path
import pandas as pd,numpy as np,json,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results'
s=pd.read_csv(O/'summary.csv');a=pd.read_csv(O/'agreement.csv');freq=pd.read_csv(O/'label_frequencies.csv');bc=pd.read_csv(O/'bootstrap_contrasts.csv');br=pd.read_csv(O/'breakdowns.csv');joint=pd.read_csv(O/'joint_labels.csv');reg=pd.read_csv(O/'adjusted_associations.csv');sens=pd.read_csv(O/'sensitivities.csv');meta=json.loads((O/'COMPLETED.json').read_text())
names={'qwen122':'Qwen122','gemma':'Gemma31B','qwen':'Qwen27B','majority':'Majority vote','labelled_similarity':'Labelled similarity','important_contrast':'Important contrast','fixed50':'Fixed probe50 / judge50','anchor_average':'Matched thresholds: average of 3 anchors','all':'All ten datasets','toxicity4':'Four toxicity / content-safety datasets'}
colors={'labelled_similarity':'#2a699b','important_contrast':'#b6662a'}
fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
for ri,policy in enumerate(['fixed50','anchor_average']):
 for ci,cohort in enumerate(['all','toxicity4']):
  ax=axs[ri,ci]
  for j,ann in enumerate(['qwen122','gemma','qwen','majority']):
   for lab,off in [('labelled_similarity',-.13),('important_contrast',.13)]:
    r=bc[(bc.policy==policy)&(bc.cohort==cohort)&(bc.annotator==ann)&(bc.label==lab)].iloc[0]
    ax.errorbar(r.yes_minus_no*100,j+off,xerr=[[max(0,(r.yes_minus_no-r.lo)*100)],[max(0,(r.hi-r.yes_minus_no)*100)]],fmt='o',capsize=3,color=colors[lab],label=names[lab] if j==0 else None)
  ax.axvline(0,color='#8b949e',lw=1);ax.set_yticks(range(4));ax.set_yticklabels([names[k] for k in ['qwen122','gemma','qwen','majority']]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_title(names[cohort]+'\n'+names[policy],fontsize=11);ax.set_xlabel('NN net accuracy gain: feature YES minus NO (percentage points)')
axs[0,0].legend(loc='best',fontsize=9)
fig.suptitle('Do the annotated features predict a larger benefit from nearest neighbours?',fontsize=15)
fig.text(.5,.008,'2,000 target-cluster bootstrap replicates; 95% intervals. Positive = more benefit when the feature is present. Observational, not causal.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.025,1,.95]);fig.savefig(O/'feature_associations.png',dpi=180);fig.savefig(O/'feature_associations.pdf');plt.close(fig)

def table(frame,perc=(),digits=2):
 z=frame.copy()
 for c in perc:
  if c in z:z[c]=z[c].map(lambda x:'—' if pd.isna(x) else f'{x*100:.{digits}f}')
 return z.to_html(index=False,escape=True,border=0,na_rep='—')
parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Nearest-neighbour mechanism labels: three-judge comparison</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#182b3b}p,li{line-height:1.55}table{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0 30px}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f6}img{width:100%;height:auto}.note{background:#f1f5f8;padding:16px;border-radius:6px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}h2{margin-top:40px}</style><h1>Do similarity and contrast explain when NNs help?</h1>''']
parts.append('<p>Three annotation models independently labelled the same 25,000 test contexts: five backbones × ten datasets × 500 cases. Outcomes come from the original ten probability-only judges. The annotations were made without target gold labels, judge outcomes or flip categories.</p>')
parts.append('<div class="note"><b>What this analysis measures.</b> A helpful flip is wrong → correct; a harmful flip is correct → wrong. Net accuracy gain = helpful rate − harmful rate, including cases whose decisions stay unchanged. Percentages below are design-weighted over the annotated sample support, with equal original judge/backbone/dataset cell mass. They are not the unconditional full-population accuracy figures from the main evaluation.</div>')
parts.append('<h2>Main findings</h2><p><b>Partial, conditional support.</b> All three annotators identify larger pooled NN gains for labelled-similarity cases. However, this is not an independent general benefit: the association changes substantially after accounting for target class, neighbour-label alignment, probe score and dataset/model/backbone differences.</p><p>At fixed50/50, similarity with an appropriate neighbour label is associated with greater benefit, while similarity to an opposite-labelled neighbour can amplify harm. Contrast is not an unconditional benefit in pooled comparisons. Under matched tuned thresholds, its adjusted association is positive across all three annotators. These patterns support a conditional labelled-reference account, rather than a claim that either feature always helps.</p>')
parts.append('<h2>Collection and agreement</h2>'+table(pd.DataFrame(meta['quality']).fillna(0)))
parts.append('<p>All raw-output checksums and final files were verified; case membership and order match across annotators. Gemma’s Markdown JSON wrappers were normalized, preserving raw text and previous invalid records. The four invalid/truncated responses were not retried. Majority vote requires two matching votes and is a summary, not ground truth.</p>')
z=a.copy();z['annotator_a']=z.annotator_a.map(lambda x:names.get(x,x));z['annotator_b']=z.annotator_b.map(lambda x:names.get(x,x));z['label']=z.label.map(names);z.kappa=z.kappa.round(3);parts.append(table(z,perc=['agreement']))
parts.append('<p>Agreement is over yes/no/uncertain on common valid pairs. Three-way agreement uses all 25,000 contexts. Kappa adjusts for agreement expected from marginal label frequencies; similar annotation models may still share biases.</p>')
parts.append('<details><summary>Label prevalence by annotator</summary>'+table(freq,perc=['fraction'])+'</details>')
parts.append('<h2>Does feature presence predict greater benefit?</h2><img src="feature_associations.png" alt="Forest plots of feature presence versus absence associations"><p>Each point compares net NN accuracy gain when that feature is labelled yes versus no. The intervals resample whole target texts together across backbones and outcome judges, so repeated judgments are not counted as independent examples. These are exploratory, pointwise intervals conditional on the existing thresholds.</p>')
for cohort in ['all','toxicity4']:
 parts.append('<h2>'+names[cohort]+'</h2>')
 for policy in ['fixed50','anchor_average']:
  parts.append('<h3>'+names[policy]+'</h3>')
  z=s[(s.cohort==cohort)&(s.policy==policy)&(s.annotator=='majority')&s.level.isin(['yes','no'])][['label','level','contexts','n','helped','harmed','gain','correction_rate','damage_rate']].copy();z.label=z.label.map(names)
  z=z.rename(columns={'contexts':'NN contexts','n':'judge–context pairs','helped':'Helped %','harmed':'Harmed %','gain':'Net gain pp','correction_rate':'Correction % of originally wrong','damage_rate':'Damage % of originally correct'})
  parts.append(table(z,perc=list(z.columns[4:])))
  j=joint[(joint.cohort==cohort)&(joint.policy==policy)][['similarity','contrast','contexts','helped','harmed','gain']]
  parts.append('<details><summary>Both-label combinations, including unresolved cases</summary>'+table(j,perc=['helped','harmed','gain'])+'</details>')
 parts.append('<p>Correction and damage rates use different opportunity denominators: originally wrong and originally correct decisions, respectively. Raw helpful/harmful rates also reflect how many errors were available to correct.</p>')
for dim in ['dataset','model','backbone','label_aligned']:
 z=br[(br.cohort=='all')&(br.dimension==dim)&br.level.isin(['yes','no'])].pivot_table(index=['group','label'],columns=['policy','level'],values='gain')
 z.columns=[p+'_'+l for p,l in z.columns]
 for pol in ['fixed50','anchor_average']:z[pol+'_yes_minus_no']=z[pol+'_yes']-z[pol+'_no']
 z=z.reset_index();parts.append('<h2>Breakdown by '+dim.replace('_',' ')+'</h2><p>Net accuracy gains and yes-minus-no differences, in percentage points. Here “model” means the original outcome judge; labels use the annotation majority.</p><div class="scroll">'+table(z,perc=list(z.columns[2:]))+'</div>')
ame=pd.read_csv(O/'adjusted_marginal_associations.csv')
parts.append('<h2>What changes after adjustment?</h2><p>The pooled YES-versus-NO contrast mixes different target classes, neighbour-label alignment and case difficulty. The following average adjusted associations compare the features within the same outcome judge/backbone/dataset cell, controlling for both labels and their interactions, target class, neighbour-label alignment, cosine rank, probe probability and text lengths. They are observational regression summaries, not causal effects.</p>')
x=ame[(ame.cohort=='all')&(ame.spec=='alignment_interactions')&ame.term.isin(['similarity_average_marginal_all','contrast_average_marginal_all'])][['policy','annotator','term','coef','lo','hi']]
parts.append(table(x,perc=['coef','lo','hi']))
x=ame[(ame.cohort=='all')&(ame.spec=='alignment_interactions')&(ame.annotator=='majority')&~ame.term.str.endswith('_all')][['policy','term','coef','lo','hi','n']]
parts.append('<details open><summary>Adjusted associations conditional on neighbour-label alignment (majority labels)</summary>'+table(x,perc=['coef','lo','hi'])+'</details>')
parts.append('<p>At fixed50/50 the majority-labelled similarity association is +0.81 pp with aligned neighbour labels (95% interval +0.37 to +1.26), and −1.72 pp with opposite neighbour labels (−3.12 to −0.32). Under matched tuned thresholds, the pooled similarity advantage does not persist after adjustment. Contrast instead has an average adjusted association of +0.93 pp (+0.66 to +1.19). These estimates depend on the regression specification and the available observed controls.</p>')
parts.append('<p>The fixed50 pooled similarity association is positive for all ten outcome judges, but its size is particularly large for GLM, OLMo and Mistral. With matched tuning, the unadjusted similarity association is positive for five of ten outcome judges. This is heterogeneity, not a universal model effect.</p>')
parts.append('<h2>Adjusted associations and sensitivity checks</h2><p>Regressions compare cases within the same judge/backbone/dataset cell, controlling for target class, neighbour-label alignment, cosine rank, probe probability and text lengths. Both feature indicators and their interaction are included. The displayed main coefficients are conditional on the other feature being absent; the interaction must also be considered. Cluster intervals use target text as the unit.</p>')
z=reg[(reg.spec=='joint')&reg.term.isin(['similarity_yes','contrast_yes','similarity_x_contrast'])]
parts.append('<details><summary>Adjusted coefficients by annotation model</summary>'+table(z[['policy','cohort','annotator','term','coef','lo','hi','n','clusters']],perc=['coef','lo','hi'])+'</details>')
parts.append('<details><summary>Common-valid, non-annotator judges, local-only and API-only sensitivity results</summary>'+table(sens[sens.level.isin(['yes','no'])][['policy','cohort','scope','label','level','n','gain','gain_lo','gain_hi']],perc=['gain','gain_lo','gain_hi'])+'</details>')
parts.append('<h2>Interpretation limits</h2><ul><li>Only about one third to one half of some label-defined groups are referred at fixed probe50; non-referred cases are retained with zero NN-versus-control change. Tuned anchors also retain non-referred cases. The displayed net gains describe the pipeline on annotated support.</li><li>The annotation prompt included the target probe probability as approved context, although the original nearest-only judge user message did not display it. Regression controls for that probability, but annotation and outcome prompts are not identical tasks.</li><li>“Labelled similarity” records similar meaning or behaviour. It does not specifically establish that the target was ambiguous or that the judge learned a dataset convention.</li><li>“Important contrast” records a relevant difference, not whether a judge noticed it or used it correctly. These labels are not reasoning traces.</li><li>Label presence is associated with target content, dataset, difficulty and neighbour-label alignment. Adjustment reduces measured confounding but cannot establish a causal mechanism.</li><li>Two annotators are from the Qwen family. Results for each annotator and for other outcome judges are provided; agreement is not independent validation.</li><li>The three tuned comparisons use one shared pair of probe/judge cutoffs within each anchor, then average the three paired outcomes. They are repeated uses of the same cases, not three replications. No test tuning or cutoff exclusions were introduced.</li><li>API missing responses remain excluded using their original three-condition matched-valid sets. All-ten-common-valid sensitivity is shown. Four content-safety datasets are an exploratory subset chosen earlier.</li><li>Uncertainty is conditional on the existing samples, thresholds and annotation labels; it does not include threshold-selection or annotation error. Subgroup comparisons are exploratory and not multiplicity-adjusted.</li></ul>')
parts.append('<h2>Files</h2><p><a href="summary.csv">All annotation-conditioned outcomes</a> · <a href="agreement.csv">Agreement</a> · <a href="bootstrap_contrasts.csv">Paired bootstrap comparisons</a> · <a href="breakdowns.csv">Dataset/judge/backbone breakdowns</a> · <a href="adjusted_associations.csv">Adjusted associations</a> · <a href="feature_associations.pdf">Figure PDF</a></p></html>')
(O/'report.html').write_text(''.join(parts))
print(O/'report.html')
