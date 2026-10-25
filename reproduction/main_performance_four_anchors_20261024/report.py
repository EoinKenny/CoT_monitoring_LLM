import os
os.environ['MPLBACKEND']='Agg'
import pathlib,csv,json,gzip,collections,html,hashlib
import numpy as np
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parent;O=R/'results';C=['baseline','saliency','nearest2','nearest'];labels=['No explanation','Saliency','2-NN','1-NN'];colors=['#63758b','#cc9630','#247a59','#8562aa']
def readcsv(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))
s=readcsv(O/'summary.csv');cells=readcsv(O/'cells.csv');cover=readcsv(O/'coverage.csv');th=readcsv(O/'thresholds.csv');parse={}
def subset(cohort='all10',scope='combined',policy='fixed50',dim='overall'):return [r for r in s if r['cohort']==cohort and r['scope']==scope and r['policy']==policy and r['dimension']==dim]
def f(x):return f'{100*float(x):.2f}'
def delta(x):return f'{100*float(x):+.2f}'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def comp(rows):
 out=[]
 for r in rows:
  vals=[float(r[c+'_f1']) for c in C];best=max(vals);v=[('<u>'+f(x)+'</u>') if np.isclose(x,best,rtol=0,atol=1e-12) else f(x) for x in vals]
  out.append([html.escape(r['group']),*v,delta(r['nn_minus_control_f1']),delta(r['nn_minus_control_precision']),delta(r['nn_minus_control_recall'])])
 return table(['Group','Control F1','Saliency F1','2NN F1','1NN F1','2NN ΔF1','2NN Δprecision','2NN Δrecall'],out)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True);rng=np.random.default_rng(20261023)
for ax,pol,title in zip(axs,['fixed50','tuned_shared'],['Fixed probe 50 / judge 50','Validation tuning: mean of four shared-threshold anchors']):
 vals=collections.defaultdict(list)
 grouped=collections.defaultdict(list)
 for r in cells:
  if r['policy']==pol:grouped[r['model'],r['backbone'],r['dataset'],r['condition']].append(float(r['f1']))
 for k,v in grouped.items():vals[k[3]].append(np.mean(v))
 for i,c in enumerate(C):
  v=np.array(vals[c])*100;assert len(v)==600;ax.scatter(i+rng.uniform(-.25,.25,len(v)),v,s=8,alpha=.18,color=colors[i],edgecolors='none');ax.plot([i-.22,i+.22],[v.mean()]*2,color=colors[i],linewidth=4);ax.text(i,v.mean()+2.2,f'{v.mean():.2f}',ha='center',color=colors[i],fontweight='bold')
 ax.set_xticks(range(4));ax.set_xticklabels(labels);ax.set_title(title,fontsize=11);ax.set_ylim(0,103);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Test F1 (%)');fig.suptitle('All 12 judges · 5 backbones · 10 datasets',fontweight='bold');fig.tight_layout();fig.savefig(O/'f1_swarm.png');fig.savefig(O/'f1_swarm.pdf');plt.close(fig)
# Average PR operating curves: no test-selected winners.
curves=readcsv(O/'tradeoff_points.csv.gz');fixed=readcsv(O/'fixed_probe_curves.csv.gz');fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,rr,title,key in [(axs[0],curves,'Validation-selected threshold pairs','beta_index'),(axs[1],fixed,'Probe fixed at 50; judge cutoff swept','judge_threshold')]:
 for ci in [0,1,3,2]:
  c=C[ci]
  g=collections.defaultdict(list)
  for r in rr:
   if r['condition']==c:g[int(r[key])].append((float(r['recall']),float(r['precision'])))
  a=np.array([np.mean(g[k],axis=0) for k in sorted(g)]);ax.plot(a[:,0]*100,a[:,1]*100,color=colors[ci],label=labels[ci],linewidth=2)
 ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)');ax.set_title(title);ax.grid(alpha=.15);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(O/'precision_recall_full.png');fig.savefig(O/'precision_recall_full.pdf')
# Display-only zoom: retain all curve coordinates and the full-range figure.
for ax in axs:
 ax.set_xlim(55,90);ax.set_ylim(80,96)
for ax in axs:ax.legend(frameon=False,loc='lower left')
fig.tight_layout();fig.savefig(O/'precision_recall.png');fig.savefig(O/'precision_recall.pdf');plt.close(fig)
# Genre-specific curves use the same saved operating points and macro averaging.
genres=[('safety6','Safety / toxicity',['aegis','beavertails','toxicchat','toxigen','wildguard','wildjailbreak']),('hallucination3','Hallucination',['fava','halueval','ragtruth']),('high_stakes','High-stakes',['high_stakes']),('safety2','Safety / jailbreaks',['wildguard','wildjailbreak']),('toxicity4','Toxicity subset',['aegis','beavertails','toxicchat','toxigen'])]
for slug,title,datasets in genres:
 fig,axs=plt.subplots(1,2,figsize=(12,5))
 for ax,rr,heading,key in [(axs[0],curves,'Validation-selected threshold pairs','beta_index'),(axs[1],fixed,'Probe fixed at 50; judge cutoff swept','judge_threshold')]:
  for ci in [0,1,3,2]:
   c=C[ci]
   g=collections.defaultdict(list)
   for r in rr:
    if r['condition']==c and r['dataset'] in datasets:g[int(r[key])].append((float(r['recall']),float(r['precision'])))
   assert g,(slug,c)
   a=np.array([np.mean(g[k],axis=0) for k in sorted(g)])
   ax.plot(a[:,0]*100,a[:,1]*100,color=colors[ci],label=labels[ci],linewidth=2)
  ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)');ax.set_title(heading,fontsize=11);ax.grid(alpha=.15);ax.legend(frameon=False,loc='lower left')
 fig.suptitle(title+' · '+str(len(datasets))+' dataset'+('s' if len(datasets)>1 else ''),fontweight='bold')
 fig.tight_layout();fig.savefig(O/('precision_recall_'+slug+'_full.png'));fig.savefig(O/('precision_recall_'+slug+'_full.pdf'))
 for ax in axs:
  ax.set_xlim(20 if slug=='high_stakes' else 55,95 if slug=='hallucination3' else 90)
  ax.set_ylim(80,100 if slug in ['high_stakes','hallucination3'] else 96)
 fig.tight_layout();fig.savefig(O/('precision_recall_'+slug+'.png'));fig.savefig(O/('precision_recall_'+slug+'.pdf'));plt.close(fig)
body='<h1>Main performance experiment: one and two NNs, all 12 judges</h1><p>24 October 2026 · Six local judges and six API aliases · Reanalysis only; no new model calls.</p>'
body+='<div class="lead">Every model keeps its own usable cases. A failed response from one judge never removes the case from another judge. Within a judge, all four explanation conditions use identical validation/test membership and identical thresholds for each comparison.</div>'
body+='<h2>Overall performance</h2><p>Equal-weight macro means across 600 judge × backbone × dataset cells. Values are percentages; differences are percentage points. Best F1 in each comparison is underlined.</p>'
for pol,title in [('fixed50','Fixed 50/50'),('average_of_four','Validation-tuned, average of four anchors')]:
 r=subset(policy=pol)[0];best=max(float(r[c+'_f1']) for c in C);body+='<h3>'+title+'</h3>'+table(['Explanation','Precision','Recall','F1','ΔF1 versus control'],[[labels[i],f(r[c+'_precision']),f(r[c+'_recall']),('<u>'+f(r[c+'_f1'])+'</u>') if float(r[c+'_f1'])==best else f(r[c+'_f1']),delta(float(r[c+'_f1'])-float(r['baseline_f1']))] for i,c in enumerate(C)])
body+='<img src="f1_swarm.png" alt="F1 of every judge backbone dataset cell, by explanation condition"><p>Each point is one cell. In the tuned panel, each point averages the four anchor-specific F1 scores; it is not the F1 of pooled predictions.</p>'
body+='<h2>How tuning is controlled</h2><p>For each judge/backbone/dataset, jointly select the probe and judge thresholds using validation F1 under control; apply that pair unchanged to control, saliency, one NN and two NNs on test. Repeat using saliency, one NN and two NNs as anchors. Average the four paired deltas. Thresholds are never selected on test performance. The frozen grid, including cutoff 0 and reject-all endpoints, and tie-breaking are unchanged.</p>'
body+=table(['Tuning anchor','Control F1','Saliency F1','2NN F1','1NN F1','2NN ΔF1'],[[pol,*[f(subset(policy=pol)[0][c+'_f1']) for c in C],delta(subset(policy=pol)[0]['nn_minus_control_f1'])] for pol in C])
body+='<h2>Dataset families</h2><p>Safety6 comprises Aegis, BeaverTails, ToxicChat, ToxiGen, WildGuard and WildJailbreak. Hallucination3 comprises FAVA, HaluEval and RAGTruth. High-stakes is separate. The previously examined toxicity4 subset excludes WildGuard/WildJailbreak and is also retained below.</p>'
for pol in ['fixed50','average_of_four']:
 body+='<h3>'+pol+'</h3>'+comp([dict(subset(cohort=co,policy=pol)[0],group=co) for co in ['toxicity4','safety2','hallucination3','high_stakes']])
body+='<h2>By judge</h2><p>These are within-judge comparisons. API judges can have different retained cases from each other, so differences between judges also reflect coverage. Local results retain the original full-population weighted estimator; API results describe each judge’s matched weighted cohort.</p>'
for pol in ['fixed50','average_of_four']:body+='<h3>'+pol+'</h3>'+comp(sorted(subset(policy=pol,dim='model'),key=lambda r:-float(r['nn_minus_control_f1'])))
body+='<h2>By dataset</h2>'
for pol in ['fixed50','average_of_four']:body+='<h3>'+pol+'</h3>'+comp(sorted(subset(policy=pol,dim='dataset'),key=lambda r:-float(r['nn_minus_control_f1'])))
body+='<h2>By backbone</h2>'
for pol in ['fixed50','average_of_four']:body+='<h3>'+pol+'</h3>'+comp(subset(policy=pol,dim='backbone'))
body+='<h2>Precision–recall trade-offs</h2><img src="precision_recall.png" alt="Precision recall operating curves"><p>Identical zoomed axes on both panels: recall 55–90% and precision 80–96%. Only the display ranges change; all scores and averaging remain unchanged. <a href="precision_recall.pdf">Download zoomed PDF</a>.</p><details><summary>Show full-range curves</summary><img src="precision_recall_full.png" alt="Full-range precision recall curves"><a href="precision_recall_full.pdf">Download full-range PDF</a></details><p>Left: threshold pairs selected on validation across 41 predeclared F-beta objectives, averaged over all four anchors and all cells, then evaluated on test. Right: a descriptive test curve with probe threshold fixed at 50 and judge cutoffs 0–101; no best test cutoff is selected. These average curves are not an optimized test Pareto envelope. Plot coordinates at a common beta/cutoff need not have identical recall.</p>'
body+='<h2>Precision–recall by dataset genre</h2><p>Both panels within each genre have identical axes. Safety and toxicity use recall 55–90%, precision 80–96%; hallucination uses recall 55–95%, precision 80–100%; high-stakes uses recall 20–90%, precision 80–100% to include its lower fixed-probe recall ceiling. Portions outside this window are visible in the full-range versions. The four tuning anchors remain control, saliency, 1NN and 2NN, including the newly added 1NN anchor. Every condition is evaluated at those same thresholds. Each genre averages its judge × backbone × dataset cells equally; the tuned curves also average the four shared-threshold anchors. The toxicity subset overlaps the safety group and is not an additional disjoint genre.</p>'
for slug,title,datasets in genres:
 body+='<h3>'+title+'</h3><p>Datasets: '+', '.join(datasets)+'.</p><img src="precision_recall_'+slug+'.png" alt="'+title+' precision recall curves"><p><a href="precision_recall_'+slug+'.pdf">Download zoomed PDF</a></p><details><summary>Show full-range curves</summary><img src="precision_recall_'+slug+'_full.png" alt="Full-range '+title+' curves"><a href="precision_recall_'+slug+'_full.pdf">Download full-range PDF</a></details>'
body+='<h2>Coverage and parsing</h2>'
cov=collections.defaultdict(collections.Counter)
for r in cover:
 for k in ['validation_n','test_n','validation_available','test_available']:cov[r['model']][k]+=int(r[k])
body+=table(['Judge','Validation retained / 25,000','Test retained / 25,000','Retained case %'],[[m,f"{c['validation_n']:,}",f"{c['test_n']:,}",f"{100*(c['validation_n']+c['test_n'])/50000:.2f}"] for m,c in cov.items()])
body+='<p>The relaxed rule takes only a standalone integer 0–100 on the final nonempty line, provided the request ended normally and is not an error, ambiguous or blocked. Final-integer recovery follows the approved rule for both original main responses and two-neighbour responses; score provenance is linked below. Raw explanations are retained but not used as features or scored answers. Eight older Sonnet invalid responses have an integer tail but no finish metadata; they remain excluded because truncation cannot be ruled out.</p><p>For each judge only, a case with any remaining unusable control/saliency/one-NN/two-NN response is omitted from that judge’s four-condition comparison. Other judges retain the case. Original inverse inclusion weights and frozen splits are preserved. No failure is imputed as a negative prediction.</p>'
body+='<h2>Judge cutoff zero</h2>'+table(['Anchor','Selected cells','Judge cutoff 0'],[[a,sum(r['anchor']==a for r in th),sum(r['anchor']==a and int(r['judge_threshold'])==0 for r in th)] for a in C])
body+='<p>When the selected judge cutoff is zero, all referred cases are accepted under every explanation. Those comparisons therefore contribute zero explanation effect. They remain explicit rather than being silently discarded.</p>'
body+='<h2>Interpretation and reproducibility</h2><p>This is an updated exploratory analysis of a repeatedly inspected test set. The displayed macro differences are descriptive, not a new confirmatory significance test. API aliases and configuration provenance remain as supplied; accepting unsolicited text for score extraction does not erase its generation. Earlier results remain preserved.</p><p>The previously verified evaluator now includes both one-NN and two-NN scores on four-way matched cases. The four tuning anchors remain control, saliency, 1NN and 2NN; all four conditions use each selected pair. This analysis contains 12,000 metric rows and 2,400 validation-selected threshold pairs. The original one-NN report remains preserved. There is no common-valid-across-models analysis in this rerun.</p>'
body+='<ul>'+''.join(f'<li><a href="{x}">{x}</a></li>' for x in ['cells.csv','summary.csv','thresholds.csv','coverage.csv','probe_only.csv','tradeoff_points.csv.gz','fixed_probe_curves.csv.gz','VERIFIED.json','SCORE_INPUTS.json'])+'</ul>'
css='body{font:16px/1.6 system-ui;max-width:1160px;margin:40px auto;padding:0 22px;color:#182738;background:#fafbfd}h1{font-size:32px}h2{margin-top:38px}h3{font-size:18px}.lead{background:#e8f1ed;border-left:5px solid #278158;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:8px 10px;border-bottom:1px solid #dce2e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef1f5}u{text-decoration-thickness:2px;text-underline-offset:3px}img{max-width:100%;height:auto}a{color:#265e9d}'
(O/'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>All-judge main performance results</title><style>'+css+'</style></head><body>'+body+'</body></html>')
print('Overall',[(pol,{k:v for k,v in subset(policy=pol)[0].items() if k.endswith(('_f1','_precision','_recall'))}) for pol in ['fixed50','average_of_four']]);print('Report ready')
