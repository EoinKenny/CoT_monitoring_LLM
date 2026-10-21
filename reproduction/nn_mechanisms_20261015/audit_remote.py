from pathlib import Path
import json,hashlib
R=Path('/home/ubuntu/experiments/nn-mechanisms-20261015');P=R.parent/'probability-500-20261012';req=json.load(open(R/'raw_audit_request.json'));wanted={(r['backbone'],r['sample_id']) for r in req};out=[]
canonical={}
for line in open(P/'canonical/cases.jsonl'):
 x=json.loads(line)
 if (x['backbone'],x['sample_id']) in wanted and x['condition'] in ['baseline','nearest']:canonical[x['backbone'],x['sample_id'],x['condition']]=x
for r in req:
 b=r['backbone'];sid=r['sample_id'];row=dict(request=r,prompts={c:canonical[b,sid,c]['messages'] for c in ['baseline','nearest']},outputs={})
 for m in ['glm','mistral','olmo','gemma']:
  row['outputs'][m]={}
  for c,ci in [('baseline',0),('nearest',2)]:
   run=P/'score'/b/'judges'/f'{m}_{c}'/'run';p=run/'judgments.jsonl';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==json.load(open(run/'COMPLETED'))['judgments_sha256']
   for line in open(p):
    x=json.loads(line)
    if x['sample_id']==sid:
     assert x['positive_probability']==r['expected_scores'][m][ci];assert x['prompt_id']==canonical[b,sid,c]['prompt_id'];row['outputs'][m][c]=x;break
 (R/'RAW_AUDIT.json').touch(exist_ok=True);out.append(row)
(R/'RAW_AUDIT.json').write_text(json.dumps(out,indent=2));print('verified',len(out),'cases and',len(out)*8,'raw outputs')
