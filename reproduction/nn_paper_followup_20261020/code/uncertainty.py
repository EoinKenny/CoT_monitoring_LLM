exec(open(__file__.replace('uncertainty.py','analyse_results.py')).read().split('results=[];')[0])
# Paired Poisson group bootstrap, dataset stratified by independently resampling clusters;
# same multiplicity reused across backbones, judges and retrieval conditions.
ts=[t for t in targets if t['arm']=='retrieval' and t['split']=='test'];keys=sorted({(t['dataset'],t['group']) for t in ts});idx={k:i for i,k in enumerate(keys)};rng=np.random.default_rng(20261021);mult=rng.poisson(1,(2000,len(keys)))
th=json.loads((O/'thresholds.json').read_text());ans={p:np.zeros(2000) for p in ['fixed50','shared_tuned']};counts=collections.Counter()
for j in ['qwen122','gemma','glm']:
 rows=jl(I/'runs/natural'/j/'responses.jsonl');lookup={(r['backbone'],r['sample_id'],r['condition']):r for r in rows if r['arm']=='retrieval'}
 for b,d in sorted({(t['backbone'],t['dataset']) for t in ts}):
  a=[t for t in ts if t['backbone']==b and t['dataset']==d];y=np.array([t['true_label'] for t in a],bool);w=mult[:,[idx[t['dataset'],t['group']] for t in a]]*np.array([t['weight'] for t in a]);probe=np.array([t['probe_probability']*100 for t in a]);scores={c:np.array([lookup[b,t['sample_id'],c]['positive_probability'] for t in a]) for c in ['activation_reference','semantic_reference']}
  pairs=[('fixed50',50,50)]+[('shared_tuned',r['probe'],r['judge_threshold']) for r in th if r['arm']=='retrieval' and r['judge']==j and r['backbone']==b and r['dataset']==d]
  for policy,pt,qt in pairs:
   fs=[]
   for c in scores:
    pred=(probe>=pt)&(scores[c]>=qt);tp=w@(pred&y);fp=w@(pred&~y);fn=w@(~pred&y);den=2*tp+fp+fn;fs.append(np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0))
   ans[policy]+=fs[0]-fs[1];counts[policy]+=1
out={p:dict(delta_percentile95_pp=(np.quantile(v/counts[p]*100,[.025,.975])).tolist(),replicates=2000,method='Paired Poisson group bootstrap shared across judges/backbones; thresholds held fixed; descriptive sampling uncertainty, not threshold-selection uncertainty') for p,v in ans.items()};(O/'uncertainty.json').write_text(json.dumps(out,indent=2));print(out)
