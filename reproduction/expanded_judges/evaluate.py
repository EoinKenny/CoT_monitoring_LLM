from pathlib import Path
import json,sys,numpy as np
R=Path(sys.argv[1])
rows=json.loads((R/'private.json').read_text());base=json.loads((R/'weighted_results.json').read_text())['results']
conditions=[m+'_'+c for m in ['glm','olmo'] for c in ['baseline','saliency','nearest']]
if len(sys.argv)>2:conditions=[sys.argv[2]]
allpred={};reuse={}
for c in conditions:
 p=R/'judges'/c/'run';assert (p/'COMPLETED').exists()
 rr=[json.loads(l) for l in (p/'judgments.jsonl').read_text().splitlines()];by={r['sample_id']:r for r in rr}
 assert len(rr)==len(by)==len(rows) and set(by)=={r['sample_id'] for r in rows}
 assert all(by[r['sample_id']]['dataset']==r['dataset'] and by[r['sample_id']]['prediction'] in (0,1) for r in rows)
 allpred[c]=by;reuse[c]=sum('reused_from' in r for r in rr)
def metrics(tp,fp,fn,tn):
 def div(a,b):return a/b if b else 0.
 return dict(tp=float(tp),fp=float(fp),fn=float(fn),tn=float(tn),accuracy=div(tp+tn,tp+fp+fn+tn),precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),balanced_accuracy=.5*(div(tp,tp+fn)+div(tn,tn+fp)))
results={};comparisons={};rng=np.random.default_rng(20261002)
for ds,b0 in base.items():
 b=b0['test'];ss=[r for r in rows if r['dataset']==ds];y=np.array([r['true_label'] for r in ss]);strata=[np.flatnonzero(y==v) for v in [0,1]]
 assert all(len(s)>0 or b['tp' if v else 'fp']==0 for v,s in enumerate(strata)), 'Missing class in sample; cannot post-stratify'
 # Identical bootstrap draws for all models/conditions, conditional on observed class counts.
 draws=[rng.integers(0,len(s),(2000,len(s))) if len(s) else None for s in strata]
 boot={};results[ds]={}
 for c in conditions:
  pred=np.array([allpred[c][r['sample_id']]['prediction'] for r in ss]);rates=[float(pred[s].mean()) if len(s) else 0. for s in strata]
  fp=b['fp']*rates[0];tp=b['tp']*rates[1];cm=metrics(tp,fp,b['fn']+b['tp']-tp,b['tn']+b['fp']-fp)
  bt=[]
  for v,s in enumerate(strata):
   pop=b['tp' if v else 'fp']
   bt.append(np.full(2000,pop*rates[v]) if len(s)==pop or not len(s) else pop*pred[s][draws[v]].mean(axis=1))
  bf,btp=bt;bf1=2*btp/(btp+bf+b['fn']+b['tp']);boot[c]=bf1
  observed=metrics(np.sum((y==1)&(pred==1)),np.sum((y==0)&(pred==1)),np.sum((y==1)&(pred==0)),np.sum((y==0)&(pred==0)))
  results[ds][c]=dict(probe=b,cascade_estimate=cm,cascade_f1_ci95=np.quantile(bf1,[.025,.975]).tolist(),f1_change=cm['f1']-b['f1'],flagged_sample_metrics=observed,sampled_flags=len(ss),population_flags=b['tp']+b['fp'],sample_positives=int(y.sum()),sample_negatives=int(len(y)-y.sum()))
 comparisons[ds]={}
 for m in ['glm','olmo']:
  for a,z in [('saliency','baseline'),('nearest','baseline'),('nearest','saliency')]:
   ca=m+'_'+a;cz=m+'_'+z
   if ca in boot and cz in boot:comparisons[ds][ca+'_minus_'+cz]=dict(f1_difference=results[ds][ca]['cascade_estimate']['f1']-results[ds][cz]['cascade_estimate']['f1'],ci95=np.quantile(boot[ca]-boot[cz],[.025,.975]).tolist())
output=dict(scope='Exact full-test probe, sampled flagged-only LLM metrics, estimated full-test cascades. Post-stratify sampled judge acceptance rates using known full flag gold-class counts; unflagged TN/FN fixed. 2000 paired row-bootstrap draws within gold-class strata; census strata fixed. Intervals conditional on fixed test pool/probes, omit finite-population correction for sampled strata, and do not measure training or backend variability.',results=results,paired_comparisons=comparisons,reused=reuse)
p=R/('results.json' if len(conditions)>1 else conditions[0]+'_results.json');p.write_text(json.dumps(output,indent=2))
if len(conditions)>1:
 lines=['# Ten-dataset sampled evaluation','','Exact probe results; estimated full-test cascade F1. Flags sampled independently of labels; 500 per dataset maximum. See results.json for intervals, class counts, flagged-only metrics and paired comparisons.','','| Dataset | Probe F1 | GLM baseline | GLM saliency | GLM NN | OLMo baseline | OLMo saliency | OLMo NN |','|---|'+'---:|'*7]
 for ds,d in results.items():lines.append('| '+ds+' | '+ ' | '.join(f'{x:.4f}' for x in [next(iter(d.values()))['probe']['f1']]+[d[c]['cascade_estimate']['f1'] for c in conditions])+' |')
 lines+=['','FAVA has very few negative test examples; inspect its negative sample count and intervals. HaluEval has a strong length shortcut. Expanded evaluation reuses previously inspected pilot cases and is exploratory. Token-identical vLLM regeneration is not guaranteed; archived outputs are authoritative.']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(evaluated=conditions,datasets=len(results))))
