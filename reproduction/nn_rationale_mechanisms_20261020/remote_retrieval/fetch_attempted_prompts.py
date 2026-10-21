from pathlib import Path
import subprocess,gzip,json,hashlib,datetime
R=Path(__file__).resolve().parent
name='nn-reason-after-20261016';dest=R/(name+'.attempted_prompts.jsonl.gz');assert not dest.exists()
remote=r'''
from pathlib import Path
import json,hashlib,sys
root=Path('/home/ubuntu/experiments/nn-reason-after-20261016');man=[]
for d in sorted(root.glob('score/*/judges/*')):
 wanted=set()
 for p in (d/'run/batches').glob('*.attempt.json'):wanted.update(json.loads(p.read_text())['ids'])
 for p in (d/'run/batches').glob('*.json'):
  if p.stem.isdigit():wanted.update(r['prompt_id'] for r in json.loads(p.read_text())['records'])
 ph=hashlib.sha256();n=0;matched=[]
 with (d/'prompts.jsonl').open('rb') as f:
  for line in f:
   ph.update(line);r=json.loads(line);n+=1
   if r['prompt_id'] in wanted:
    matched.append(r['prompt_id']);print(json.dumps({'source_file':str(d/'prompts.jsonl'),'source_line':n,'line_sha256':hashlib.sha256(line).hexdigest(),'row':r}))
 assert set(matched)==wanted,(d,len(matched),len(wanted))
 approved=json.loads((d/'approved_config.json').read_text());assert approved['prompt_sha256']==ph.hexdigest()
 man.append(dict(source_file=str(d/'prompts.jsonl'),source_sha256=ph.hexdigest(),all_source_rows=n,attempted_or_committed_rows=len(matched)))
print(json.dumps({'manifest':man}))
'''
cmd=['ssh','-i','/Users/eoinkenny/.ssh/llm_interpretable.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@ec2-52-14-51-0.us-east-2.compute.amazonaws.com','python3 -']
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);p.stdin.write(remote.encode());p.stdin.close()
with gzip.open(dest,'wb') as f:
 while b:=p.stdout.read(1024*1024):f.write(b)
err=p.stderr.read();code=p.wait();assert code==0,err.decode()
receipt=dict(file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),read_only=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/(name+'.attempted_prompts.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
