import json,hashlib
from pathlib import Path
ROOT=Path('reproduction/nn_rationale_mechanisms_20261020')
def write(batch,rows):
 p=ROOT/'packets'/f'{batch}.jsonl';src=[json.loads(l) for l in p.read_text().splitlines()]
 assert len(rows)==len(src),(len(rows),len(src))
 out=[]
 for i,(r,a,b,c,q) in enumerate(rows):
  evidence=q if isinstance(q,list) else [q]
  assert all(e and e in src[i]['text'] for e in evidence),(i,evidence)
  out.append(dict(id=src[i]['id'],relation=r,axes=a.split(','),bridge=b,conclusion=c,quality='A',evidence=evidence))
 assert len({x['id'] for x in out})==len(src)
 (ROOT/'codes'/f'{batch}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 rec=dict(batch_id=batch,reader='/root/mechanism_design',read_complete=True,n=len(src),input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),notes='Every complete expressed rationale read semantically; task/context factual validity deferred to separate contextual audit. Exact IDs and evidence substrings validated. No keyword coding.',code_version='expressed-v1')
 (ROOT/'codes'/f'{batch}.receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(batch,len(out),'validated')
