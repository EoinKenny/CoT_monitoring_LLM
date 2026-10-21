from pathlib import Path
import json,gzip,hashlib,time
P=Path('/home/ubuntu/experiments/probability-500-20261012');R=Path('/home/ubuntu/experiments/api-results-20261015');R.mkdir(exist_ok=True)
BBS=['llama','qwen','gemma','llama70','qwen122'];JS=['qwen','gemma','mistral','olmo','glm','qwen122'];CS=['baseline','saliency','nearest']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*2**20),b''):h.update(x)
 return h.hexdigest()
def jl(p):
 with p.open() as f:
  for line in f:yield json.loads(line)
assert read(P/'COMPLETED.json')['n']==900000
sources={};samples={};thresholds={}
for b in BBS:
 samples[b]=read(P/'score'/b/'selected_flags.json');thresholds[b]=read(P/'score'/b/'THRESHOLDS.json')
 for name in ['selected_flags.json','THRESHOLDS.json']:sources[str(P/'score'/b/name)]=sha(P/'score'/b/name)
with gzip.open(R/'design.json.gz','wt') as f:json.dump(dict(samples=samples,thresholds=thresholds),f)
canonical=P/'canonical/cases.jsonl';assert sha(canonical)==read(P/'canonical/PREPARED.json')['cases_sha256'];sources[str(canonical)]=sha(canonical)
with gzip.open(R/'prompt_index.jsonl.gz','wt') as f:
 for r in jl(canonical):f.write(json.dumps({k:r[k] for k in ['prompt_id','backbone','dataset','sample_id','split','condition','messages_sha256']})+'\n')
with gzip.open(R/'local_scores.jsonl.gz','wt') as f:
 for b in BBS:
  for j in JS:
   for c in CS:
    run=P/'score'/b/'judges'/f'{j}_{c}'/'run';done=read(run/'COMPLETED');p=run/'judgments.jsonl';assert done['n']==10000 and sha(p)==done['judgments_sha256'];assert sha(run/'config.json')==done['config_sha256'];sources[str(p)]=done['judgments_sha256'];sources[str(run/'config.json')]=done['config_sha256']
    rows=list(jl(p));assert [x['sample_id'] for x in rows]==[x['sample_id'] for x in samples[b]]
    for x in rows:f.write(json.dumps(dict(model=j,**{k:x[k] for k in ['prompt_id','backbone','dataset','sample_id','condition','positive_probability','input_tokens','generated_tokens','batch_number','batch_seconds']}))+'\n')
T=P.parent/'two-nn-20261014';partial=[];effects=[]
for b in BBS:
 for j in JS:
  d=T/'analysis'/f'{b}_{j}'
  if not (d/'COMPLETED.json').exists():continue
  done=read(d/'COMPLETED.json')
  for name,h in done['files'].items():assert sha(d/name)==h
  for x in read(d/'summary.json'):partial.append(dict(backbone=b,judge=j,**x))
  for x in read(d/'comparisons.json'):effects.append(dict(backbone=b,judge=j,**x))
(R/'two_nn_partial_summary.json').write_text(json.dumps(partial));(R/'two_nn_partial_comparisons.json').write_text(json.dumps(effects))
manifest=dict(time=time.time(),sources=sources,files={p.name:sha(p) for p in R.iterdir() if p.name.endswith(('.gz','.json')) and p.name!='EXPORT_MANIFEST.json'})
(R/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('exported',len(samples)*10000,'cases',len(partial),'twoNN summary rows')
