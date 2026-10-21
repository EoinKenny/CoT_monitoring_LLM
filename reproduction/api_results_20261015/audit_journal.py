from pathlib import Path
import zipfile,json,collections,hashlib,csv
R=Path(__file__).resolve().parent;p=Path('/Users/eoinkenny/Downloads/request_journal.zip')
counts=collections.Counter();fields=collections.Counter();keys=collections.Counter();status=collections.Counter();bounds={};examples={};other=[]
with zipfile.ZipFile(p) as z:
 with z.open('request_journal.txt') as f:
  for line in f:
   r=json.loads(line);event=r.get('event');model=r.get('model');counts[model,event]+=1;fields.update(r.keys());keys[model,r.get('prompt_id'),event]+=1
   if event not in examples:examples[event]={k:v for k,v in r.items() if k not in ['response','error']}
   if 'status' in r:status[model,r['status']]+=1
   if r.get('ts'):
    prior=bounds.setdefault(model,[r['ts'],r['ts']]);prior[0]=min(prior[0],r['ts']);prior[1]=max(prior[1],r['ts'])
   if event not in ['dispatch','complete','result'] and len(other)<3:other.append(r)
response_keys={(r['model'],r['prompt_id']) for r in csv.DictReader((R/'responses.csv').open())}
seen={(m,p) for m,p,e in keys if p is not None}
result=dict(zip_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fields=dict(fields),events=[dict(model=m,event=e,n=n) for (m,e),n in counts.items()],duplicate_event_keys=sum(n-1 for n in keys.values() if n>1),duplicate_examples=[dict(model=m,prompt_id=p,event=e,n=n) for (m,p,e),n in keys.items() if n>1][:10],response_keys_not_in_journal=len(response_keys-seen),journal_keys_not_in_responses=len(seen-response_keys),time_ranges=bounds,event_examples=examples)
(R/'JOURNAL_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2)[:7000])
