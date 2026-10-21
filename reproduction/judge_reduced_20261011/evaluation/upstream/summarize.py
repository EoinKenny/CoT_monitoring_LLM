try:
 from analyze_probability_20261011 import *
except ModuleNotFoundError:
 from analyze import *
from collections import Counter
pairs=read(OUT/'pairs.json'); effects=read(OUT/'effects.json'); budgets=read(OUT/'budgets.json'); calibration=read(OUT/'calibration.json'); stability=read(OUT/'stability.json')
variants=list(pairs[0]['variants']);datasets=sorted({p['dataset'] for p in pairs})
def mean(rows,k):return float(np.mean([r[k] for r in rows]))
summary={v:{k:mean([p['variants'][v] for p in pairs],k) for k in ['f1','precision','recall','routing_rate']} for v in variants}
summary['validation_selected']={k:mean([p['selected_test'] for p in pairs],k) for k in ['f1','precision','recall','routing_rate']}
rng=np.random.default_rng(20261011);ix=rng.integers(0,len(datasets),(10000,len(datasets)));comparisons={}
for c in sorted({e['comparison'] for e in effects}):
 rows=[e for e in effects if e['comparison']==c]; d=np.array([mean([e for e in rows if e['dataset']==ds],'delta_f1') for ds in datasets]);comparisons[c]=dict(mean_delta=float(d.mean()),dataset_block_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),positive_pairs=sum(e['delta_f1']>1e-12 for e in rows),negative_pairs=sum(e['delta_f1']< -1e-12 for e in rows),positive_datasets=int(sum(d>0)),per_dataset=dict(zip(datasets,d.tolist())))
heterogeneity={}
for factor in ['dataset','backbone','judge']:
 heterogeneity[factor]={value:{v:{k:mean([p['variants'][v] for p in pairs if p[factor]==value],k) for k in ['f1','precision','recall','routing_rate']} for v in variants} for value in sorted({p[factor] for p in pairs})}
write('summary.json',dict(macro=summary,comparisons=comparisons,heterogeneity=heterogeneity,selected_counts=dict(Counter(p['selected'] for p in pairs)),mean_selection_stability=mean(stability,'original_choice_frequency')))
# Precision-recall plot: all points are validation-selected configurations; no test optimization.
colors={'baseline':'#4c78a8','saliency':'#f58518','nearest':'#54a24b'}
fig,axs=plt.subplots(1,3,figsize=(15,4.6))
for ax,j in zip(axs[:2],JS):
 for c in CS:
  for suffix,mark in [('parent_hard','o'),('joint','s'),('fixed50','^')]:
   v=c+'_'+suffix;h=heterogeneity['judge'][j][v];ax.scatter(h['recall']*100,h['precision']*100,c=colors[c],marker=mark,s=75,label=f'{c}: {suffix.replace("parent_","")}')
 ax.set(title=j+' judge',xlabel='Recall (%)',ylabel='Precision (%)');ax.grid(alpha=.2)
for c in CS:
 for mode,ls in [('hard','--'),('joint','-')]:
  rows=[[r for r in budgets if r['condition']==c and r['mode']==mode and r['budget']==b] for b in BUDGETS];axs[2].plot([mean(r,'test_routing')*100 for r in rows],[mean(r,'f1')*100 for r in rows],ls,marker='o',color=colors[c],label=c+' '+mode)
axs[2].axhline(summary['probe']['f1']*100,color='gray',ls=':',label='probe only');axs[2].set(title='Validation-budget selected policies',xlabel='Actual test referral rate (%)',ylabel='Test F1 (%)');axs[2].grid(alpha=.2);axs[0].legend(fontsize=7);axs[2].legend(fontsize=7);fig.suptitle('Population estimates; macro averages across datasets and backbones');fig.tight_layout();fig.savefig(OUT/'performance_and_cost.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5));cmp=['baseline_joint_minus_nearest_parent_hard','nearest_joint_minus_baseline_joint']
for ax,c,title in zip(axs,cmp,['Tuned no-explanation probability − hard-label NN','Tuned NN probability − tuned no-explanation probability']):
 vals=comparisons[c]['per_dataset'];ax.barh(datasets,[100*vals[d] for d in datasets],color=['#54a24b' if vals[d]>0 else '#e45756' for d in datasets]);ax.axvline(0,color='black',lw=.8);ax.set(title=title,xlabel='F1 difference (percentage points)')
fig.tight_layout();fig.savefig(OUT/'dataset_effects.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4.8));cal=[]
for ax,j in zip(axs,JS):
 for c in CS:
  rr=[r for r in calibration if r['condition']==c and r['judge']==j];bins={}
  for r in rr:
   for x in r['bins']:
    if not x['weighted_n']:continue
    b=x['low'];bins.setdefault(b,[]).append(x)
  xx=[];yy=[]
  for b,rs in sorted(bins.items()):
   w=np.array([r['weighted_n'] for r in rs]);xx.append(np.average([r['mean_probability'] for r in rs],weights=w));yy.append(np.average([r['observed_positive_rate'] for r in rs],weights=w))
  ax.plot(xx,yy,'o-',color=colors[c],label=c);cal.append(dict(judge=j,condition=c,brier=mean(rr,'weighted_brier'),zero_mass=mean(rr,'zero_mass'),hundred_mass=mean(rr,'hundred_mass'),multiple10=mean(rr,'multiple_of_10_mass')))
 ax.plot([0,1],[0,1],'k:',alpha=.5);ax.set(title=j+' reliability (population pooled)',xlabel='Elicited positive-class probability',ylabel='Observed positive fraction');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'calibration.png',dpi=170);plt.close(fig)
write('calibration_summary.json',cal)
# Representative post-hoc examples, chosen by immutable sample ID, one help/hurt per dataset.
cases=read(OUT/'cases.json'); examples=[]
for ds in datasets:
 for kind in ['nn_only_correct','nn_hurt']:
  rr=sorted([r for r in cases if r['dataset']==ds and r['kind']==kind and r['mode']=='joint'],key=lambda r:(r['sample_id'],r['backbone'],r['judge']))
  if not rr:continue
  r=rr[0];es={e['sample_id']:e for e in jl(P/r['backbone']/'judge_inputs/explanations.jsonl')};e=es[r['sample_id']];examples.append(dict(**r,current_messages=e.get('messages'),neighbour=e['nearest'],saliency=e['saliency']))
write('matched_examples.json',examples)
# NN stratifications are observational and include post-hoc true-label relationships.
ng=read(OUT/'neighbour_groups.json');grouped=[]
for mode in ['joint','parent_hard']:
 for field in ['agreement','label_matches_target','score_band']:
  for value in sorted({x[field] for x in ng}):
   rr=[x for x in ng if x['mode']==mode and x[field]==value];w=np.array([x['weight'] for x in rr]);grouped.append(dict(mode=mode,field=field,value=value,n=len(rr),improved_rate=float(np.average([x['improved'] for x in rr],weights=w)),worse_rate=float(np.average([x['worse'] for x in rr],weights=w))))
write('neighbour_diagnostics.json',grouped)
print(json.dumps(dict(summary=summary,comparisons=comparisons,calibration=cal,selection=Counter(p['selected'] for p in pairs),stability=mean(stability,'original_choice_frequency')),indent=2))
import html
parts=['<!doctype html><meta charset="utf-8"><title>Probe → LLM analysis</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#19212c}h1,h2{color:#183e60}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px;font-variant-numeric:tabular-nums}th{background:#edf3f8}img{width:100%}p{line-height:1.6}.note{background:#f4f6f8;padding:15px}</style><h1>Probe → LLM: explanations and probability tuning</h1><p>11 October 2026. Five backbones, ten datasets, two judges. All 600,000 outputs verified; no additional model calls.</p>']
def para(s):parts.append('<p>'+s+'</p>')
def header(s):parts.append('<h2>'+s+'</h2>')
def table(head,rows):parts.append('<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>')
def pct(x):return f'{100*x:.2f}'
header('Main finding and recommendation')
para('Probability threshold tuning is an important baseline. Tuned no-explanation probabilities slightly exceed hard-label NN on mean test F1. NN retains a small average advantage when it receives the same joint tuning, but this is heterogeneous. Choose the explanation and output mode on validation for each task; do not claim universal NN superiority or Pareto dominance. No explanation plus tuned probability is a competitive simpler default, with NN a validation-tested option.')
header('Estimated full-test performance')
para('Equal-weight macro means over 100 backbone–dataset–judge combinations. Probe-only is exact full-test performance, repeated across judges only for equal-weight comparison. LLM pipelines are population estimates from stratified samples; thresholds are selected on validation. Fixed-50 also tunes the probe gate. Hard-label rows also tune the probe gate.')
table(['Policy','F1 (%)','Precision (%)','Recall (%)','Referral (%)'],[[v]+[pct(summary[v][k]) for k in ['f1','precision','recall','routing_rate']] for v in variants+['validation_selected']])
para('“validation_selected” chooses among seven core policies separately within each backbone–dataset–judge pair, using validation F1, then lower referral cost on ties. It does not select a backbone or judge on test performance. The mean of cell F1 values is not F1 of a pooled confusion matrix.')
header('Paired comparisons and uncertainty')
table(['Comparison','Mean ΔF1 (pp)','Dataset-block 95% interval (pp)','Positive / negative cells','Positive datasets'],[[k,pct(v['mean_delta']),' to '.join(pct(x) for x in v['dataset_block_ci95']),f"{v['positive_pairs']} / {v['negative_pairs']}",f"{v['positive_datasets']} / 10"] for k,v in comparisons.items()])
para('Per-cell intervals use 2,000 paired test resamples within original score-band × gold-label strata, keeping census strata fixed. Macro intervals use 10,000 paired dataset-block resamples retaining all backbones, judges and methods together. There are only ten dataset blocks and shared benchmark origins; these intervals are exploratory summaries, not evidence of independence or a confirmatory significance test. Thresholds are fixed in these intervals. Separate selection sensitivity resamples validation and test and retunes the policies.')
parts.append('<img src="dataset_effects.png" alt="Dataset-specific primary effects">')
header('Heterogeneity')
for factor,groups in heterogeneity.items():
 parts.append('<h3>'+factor+'</h3>');table([factor,'Probe F1','Hard baseline','Hard saliency','Hard NN','Tuned baseline','Tuned saliency','Tuned NN'],[[key]+[pct(vals[v]['f1']) for v in ['probe','baseline_parent_hard','saliency_parent_hard','nearest_parent_hard','baseline_joint','saliency_joint','nearest_joint']] for key,vals in groups.items()])
header('Referral cost and trade-offs')
parts.append('<img src="performance_and_cost.png" alt="Precision recall and cost comparisons">')
para('Each cost policy maximizes validation F1 under a validation referral budget. The plotted x coordinate is actual full-test referral rate, which can differ from the validation budget. These are validation-selected operating points, not test-optimized frontiers. The pipeline accepts unflagged examples as negative; every positive output requires passing the probe gate and judge decision. At a zero gate the pipeline becomes LLM-only. Different explanations change input length as well as accuracy.')
table(['Condition','Mode','Validation budget','Actual test referral','Test F1'],[[c,m,pct(b),pct(mean([r for r in budgets if r['condition']==c and r['mode']==m and r['budget']==b],'test_routing')),pct(mean([r for r in budgets if r['condition']==c and r['mode']==m and r['budget']==b],'f1'))] for c in CS for m in ['hard','joint'] for b in BUDGETS])
rt=read(OUT/'runtime.json');table(['Condition','Output','Mean input tokens/call','Observed batch inference hours'],[[c,m,f"{sum(r['input_tokens'] for r in rt if r['condition']==c and r['mode']==m)/sum(r['n'] for r in rt if r['condition']==c and r['mode']==m):.1f}",f"{sum(r['measured_inference_seconds'] for r in rt if r['condition']==c and r['mode']==m)/3600:.2f}"] for c in CS for m in ['hard','probability']]);para('Measured batch inference time excludes model load, preparation and analysis, and is not a causal latency benchmark. It covers the actual sample sweep, not a predicted deployment workload.')
eps=read(OUT/'endpoints.json');table(['LLM-only explanation','Output','F1 (%)'],[[c,m,pct(mean([r for r in eps if r['condition']==c and r['mode']==m],'f1'))] for c in CS for m in ['hard','fixed50','joint']])
header('Calibration and threshold sensitivity')
parts.append('<img src="calibration.png" alt="Reliability diagrams">');table(['Judge','Explanation','Brier','0 mass (%)','100 mass (%)','Multiple-of-10 mass (%)'],[[r['judge'],r['condition'],f"{r['brier']:.4f}",pct(r['zero_mass']),pct(r['hundred_mass']),pct(r['multiple10'])] for r in cal]);para('Reliability curves pool design-weighted populations and therefore give larger datasets more weight, unlike the macro performance tables. Brier scores and score masses are cell-macro means. Elicited probabilities are not assumed calibrated. Threshold tuning can improve decisions even when the numerical probabilities are poorly calibrated.')
table(['Validation-chosen policy','Cells'],sorted(Counter(p['selected'] for p in pairs).items()));para('Across 200 paired validation/test resamples per cell, the original selected configuration is reselected on average '+pct(mean(stability,'original_choice_frequency'))+'% of the time. Per-cell cutoff quantiles and selection frequencies are in stability.json. This sensitivity holds the exact complete-validation probe-only baseline fixed and resamples the LLM selection samples. It does not repeat model training or inference. Configurations selected among many close validation scores can be unstable.')
header('Matched errors and neighbour diagnostics')
for field in ['agreement','label_matches_target']:
 table(['Output','Group field','Value','Sample records','NN helps baseline (%)','NN hurts baseline (%)'],[[r['mode'],r['field'],r['value'],r['n'],pct(r['improved_rate']),pct(r['worse_rate'])] for r in grouped if r['field']==field])
para('These diagnostics compare final pipeline decisions at their independently validation-selected thresholds. Thus an improvement may reflect gate/cutoff changes as well as a changed judge response. Neighbour label matching the target is a post-hoc diagnostic using test gold, not a deployable selection rule. Label/probe agreement and probe-score band summaries are observational; they do not establish why NN works. effects.json contains recovered/lost positives and added/removed false positives. matched_examples.json contains deterministic illustrative cases where NN alone succeeds and where NN hurts, including exact text and the retrieved example. Cases are examples, not prevalence estimates.')
header('Design, coverage and scientific limits')
para('There are 50,000 backbone–dataset cases: 25,000 validation and 25,000 test records, 500 per split for each backbone–dataset pair. Cases are shared across judges, explanations and output formats within a backbone; backbone case sets may differ. The 600,000 judgments are not 600,000 independent examples. Inverse inclusion probabilities recover population confusion counts; exact population positives determine false negatives. F1 is computed from weighted counts for each cell. Routed sample counts and effective sizes are in support.json.')
para('All committed raw batch checksums, configuration/prompt hashes, record order and completed judgment hashes were verified. Parent and probability case metadata and threshold manifests match exactly. The independent weighted-grid implementation reproduced frozen validation choices for all conditions; synthetic grids were checked against brute force. The new analysis initially tested unnamespaced group IDs across datasets; this was corrected to dataset-scoped groups, and all within-dataset sampled validation/test group checks passed. Aegis and BeaverTails share some source-group hashes across datasets, so benchmark independence should not be assumed. Frozen splits were not altered.')
para('The study has already involved test inspection, so these are exploratory held-out evaluations of validation-selected policies, not a pristine preregistered confirmation. Sample-based nonlinear metrics and broad threshold selection have uncertainty. No random-neighbour or label-only control was run, so the NN effect cannot isolate explanation fidelity from labelled-example information. Only two local judges were used in this sweep. No guarantee of universal safety gains follows from benchmark F1. Prior removal of numerical replay agreement gates remains documented; finite-value and artifact-integrity checks remain. vLLM generations are frozen for replay, but fresh inference can differ.')
header('Reproducibility and files')
para('The machine-readable outputs accompany this report: summary.json, pairs.json, effects.json, budgets.json, endpoints.json, calibration.json, stability.json, support.json, runtime.json, cases.json, matched_examples.json and neighbour_diagnostics.json. VERIFIED.json and ADDITIONAL_CHECKS.json record integrity checks. analyze.py and summarize.py reproduce the analysis against the authoritative server artifacts. Random analysis seed is 20261011; per-cell bootstrap seeds derive from SHA-256 of that seed and cell identity. No model call or probe retraining was performed.')
(OUT/'report.html').write_text('\n'.join(parts))
write('REPORT_COMPLETED.json',dict(time=time.time(),files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='REPORT_COMPLETED.json'}))
