from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/ubuntu/experiments/reasoning-benchmark-quick-20261011')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prepared=json.loads((R/'PREPARED.json').read_text());assert sha(R/'cases.jsonl')==prepared['cases_sha256']
source=[json.loads(l) for l in (R/'cases.jsonl').open()]
summary={'denominator':'300000 judgments/model: 5 backbones x 10 datasets x 1000 frozen validation/test cases x 3 explanations x 2 output formats. Each output format alone is 150000.','sample':prepared,'models':{}}
for model in ['qwen','gemma','glm','gptoss']:
 d=R/model
 if not (d/'COMPLETED.json').exists():continue
 done=json.loads((d/'COMPLETED.json').read_text());assert sha(d/'RESULTS.json')==done['results_sha256'];results=json.loads((d/'RESULTS.json').read_text());batches=results['batches'];records=[]
 for b in batches:
  assert sha(d/(b['tag']+'.raw.json'))==b['raw_sha256'];records.extend(b['records'])
 assert len(records)==300 and {r['id'] for r in records}=={r['id'] for r in source}
 rng=np.random.default_rng(20261011);boot=np.zeros(5000);groups=[]
 for condition in ['baseline','saliency','nearest']:
  for mode in ['hard','probability']:
   bs=[b for b in batches if b['records'][0]['condition']==condition and b['records'][0]['mode']==mode and b['n']==16]
   sec=np.array([b['seconds'] for b in bs]);n=np.array([b['n'] for b in bs]);idx=rng.integers(0,len(bs),(5000,len(bs)))
   boot+=50000*sec[idx].sum(axis=1)/n[idx].sum(axis=1)
   rr=[r for b in bs for r in b['records']]
   groups.append(dict(condition=condition,mode=mode,n=len(rr),valid=sum(r['valid'] for r in rr),truncated=sum(r['finish_reason']=='length' for r in rr),seconds=float(sec.sum()),full_hours=float(sec.sum()/n.sum()*50000/3600),mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in rr])),max_generated_tokens=max(r['generated_tokens'] for r in rr)))
 hours=sum(g['full_hours'] for g in groups)
 summary['models'][model]=dict(n=len(records),valid=sum(r['valid'] for r in records),truncated=sum(r['finish_reason']=='length' for r in records),reasoning_delimiter=sum(r['reasoning_delimiter'] for r in records),groups=groups,full_hours=hours,batch_bootstrap_95_hours=(np.quantile(boot,[.025,.975])/3600).tolist(),hard_hours=sum(g['full_hours'] for g in groups if g['mode']=='hard'),probability_hours=sum(g['full_hours'] for g in groups if g['mode']=='probability'),load_seconds=results['load_seconds'],input_mean=float(np.mean([r['input_tokens'] for r in records])),generated_mean=float(np.mean([r['generated_tokens'] for r in records])),generated_p95=float(np.quantile([r['generated_tokens'] for r in records],.95)),batch_median_seconds=float(np.median([b['seconds'] for b in batches])),batch_p95_seconds=float(np.quantile([b['seconds'] for b in batches],.95)))
summary['limits']=['Conditional estimate for batch16, eager/no-prefix-cache, native reasoning, greedy seed20261001, 2048 total generated-token ceiling.','Intervals resample batches within six condition/output strata; these quantify pilot timing variability, not all system or rare-case uncertainty.','Unconstrained final answers were validated after native reasoning. Invalid/truncated outputs are never silently retried; a production budget/decoding change requires a new timing estimate.','Full-300000 timing uses full batch16 timings only (288 of 300 calls); 12 tail calls are retained for validity but excluded from throughput extrapolation. Initialization/checksum/write overhead excluded. Native final-output constrained decoding is not included.']
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['models'],indent=2))
