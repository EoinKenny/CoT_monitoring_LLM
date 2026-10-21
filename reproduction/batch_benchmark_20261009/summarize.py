import json,hashlib,statistics
from pathlib import Path
R=Path('/home/ubuntu/experiments/batch-benchmark-20261009')
out=[]
for model in ['qwen','gemma']:
 baseline=json.loads((R/model/'b16_t4096/COMPLETED.json').read_text())
 for p in sorted((R/model).glob('b*/COMPLETED.json')):
  z=json.loads(p.read_text());labels=z.pop('labels');z['speedup_over_original']=z['cases_per_second']/baseline['cases_per_second'];z['label_changes_vs_original']=sum(a!=b for a,b in zip(labels,baseline['labels']));z['completed_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();out.append(z)
summary={'runs':out,'judgments_in_timed_passes':sum(2*z['n'] for z in out),'note':'8 deterministic examples plus longest per dataset and condition; all settings use same prompts within model. Two timed passes per configuration; loading and warmup excluded, durable output writes included. Benchmark outputs never merged into canonical results. Only batching/scheduler token budget changed. Fastest tested configuration is not a global optimum.'}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
