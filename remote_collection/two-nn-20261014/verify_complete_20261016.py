import json,hashlib,gzip,time
from pathlib import Path
R=Path('/home/ubuntu/experiments/two-nn-20261014')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
done=read(R/'COMPLETED.json')
assert done['new_judgments']==300000
for name,h in done['files'].items():assert sha(R/'results'/name)==h
for name,h in done['sources'].items():assert sha(R/'analysis'/name/'COMPLETED.json')==h
n=0;batches=0
out=R/'export_verified_20261016';out.mkdir(exist_ok=True)
with gzip.open(out/'nearest2_scores.jsonl.gz','wt') as f:
 for b in ['llama','qwen','gemma','llama70','qwen122']:
  selected=read(R/'score'/b/'selected_flags.json')
  for j in ['qwen','gemma','mistral','olmo','glm','qwen122']:
   run=R/'score'/b/'judges'/f'{j}_nearest2'/'run'
   d=read(run/'COMPLETED');assert d['n']==10000
   assert sha(run/'judgments.jsonl')==d['judgments_sha256']
   assert sha(run/'config.json')==d['config_sha256']
   rows=[json.loads(l) for l in (run/'judgments.jsonl').open()]
   assert [x['sample_id'] for x in rows]==[x['sample_id'] for x in selected]
   committed=[]
   for p in sorted((run/'batches').glob('*.json')):
    if p.name.endswith(('.raw.json','.attempt.json')):continue
    z=read(p);assert sha(p.with_suffix('.raw.json'))==z['raw_sha256']
    assert hashlib.sha256(json.dumps(z['records'],sort_keys=True).encode()).hexdigest()==z['records_sha256']
    committed.extend(z['records']);batches+=1
   assert committed==rows
   for x in rows:
    assert x['condition']=='nearest2' and 0<=x['positive_probability']<=100
    f.write(json.dumps(dict(model=j,**x))+'\n');n+=1
assert n==300000
receipt=dict(time=time.time(),judgments=n,raw_batches_verified=batches,paired_membership=True,score_export_sha256=sha(out/'nearest2_scores.jsonl.gz'),completion_sha256=sha(R/'COMPLETED.json'))
(out/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
