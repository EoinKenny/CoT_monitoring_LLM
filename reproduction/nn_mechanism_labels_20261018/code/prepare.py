from common import *
import collections

def main():
 reserve();assert not (ROOT/'PREPARED.json').exists()
 canonical=PARENT/'canonical/cases.jsonl';assert sha(canonical)==read(PARENT/'canonical/PREPARED.json')['cases_sha256']
 rubric=(ROOT/'RUBRIC.txt').read_text();cases={};units={};mapping=[];sourcehash={str(canonical):sha(canonical)}
 def add(kind,payload,meta):
  text=json.dumps(payload,ensure_ascii=False,separators=(',',':'));uid=digest(kind+':'+text)
  if uid not in units:units[uid]=dict(unit_id=uid,kind=kind,payload=payload,messages=[{'role':'system','content':rubric},{'role':'user','content':text}],dataset=meta['dataset'])
  mapping.append(dict(unit_id=uid,**meta))
 for r in jl(canonical):
  key=r['backbone'],r['sample_id'];assert key not in cases
  task=r['messages'][0]['content'].split('Task:',1)[1].split('\n\n',1)[0].strip();body=r['messages'][1]['content'];cue='Positive-class probability (0-100):';assert body.endswith(cue)
  cases[key]=dict(task_definition=task,target_and_reference=body[:-len(cue)].strip())
  add('case',dict(annotation_type='CASE',**cases[key]),dict(dataset=r['dataset'],backbone=r['backbone'],sample_id=r['sample_id'],split=r['split'],kind='case'))
 assert len(cases)==50000
 coverage=collections.Counter()
 for order,src,cond in [('after',AFTER,'nearest_reason_after'),('first',FIRST,'nearest_reason_first')]:
  for j in ['qwen','gemma','mistral','olmo','glm','qwen122']:
   for b in BBS:
    d=src/f'score/{b}/judges/{j}_{cond}';sel=read(d/'selection.json');by={r['sample_id']:r for r in sel};p=d/'run/judgments.jsonl';done=read(d/'run/COMPLETED');assert sha(p)==done['judgments_sha256'];sourcehash[str(p)]=sha(p)
    for r in jl(p):
     coverage[order+'_all']+=1;n=r.get('positive_probability');prior=by[r['sample_id']]
     if n is None:coverage[order+'_invalid']+=1;continue
     if (n>=50)!=(prior['nn_probability']>=50):coverage[order+'_changed']+=1;continue
     reason=r.get('reasoning','').strip()
     if not reason:coverage[order+'_empty_rationale']+=1;continue
     coverage[order+'_unchanged']+=1
     add('rationale',dict(annotation_type='RATIONALE',**cases[b,r['sample_id']],written_explanation=reason),dict(dataset=r['dataset'],backbone=b,sample_id=r['sample_id'],split='test',kind='rationale',source_judge=j,source_order=order,source_probability=n,original_nn_probability=prior['nn_probability'],reasoning_complete=r.get('reasoning_complete'),finish_reason=r.get('finish_reason')))
 # Deduplicate only byte-identical evidence; retain every original mapping privately.
 ids=sorted(units,key=lambda u:digest('20261018:'+u));smoke=[]
 for kind in ['case','rationale']:
  queues={ds:[u for u in ids if units[u]['kind']==kind and units[u]['dataset']==ds] for ds in sorted({r['dataset'] for r in units.values()})}
  while len([u for u in smoke if units[u]['kind']==kind])<16:
   for ds,q in queues.items():
    if q and len([u for u in smoke if units[u]['kind']==kind])<16:smoke.append(q.pop(0))
 used=set(smoke);ordered=smoke+[u for u in ids if u not in used]
 with open(ROOT/'units.jsonl','w') as f:
  for u in ordered:f.write(json.dumps(units[u],ensure_ascii=False)+'\n')
 with gzip.open(ROOT/'mapping_PRIVATE.jsonl.gz','wt') as f:
  for r in mapping:f.write(json.dumps(r)+'\n')
 write(ROOT/'INPUT_SHA256.json',sourcehash);write(ROOT/'SCHEMA.json',SCHEMA)
 write(ROOT/'PREPARED.json',dict(time=time.time(),original_cases=50000,unique_units=len(units),kind_counts=dict(collections.Counter(u['kind'] for u in units.values())),source_rationale_coverage=dict(coverage),original_mappings=len(mapping),smoke_units=32,units_sha256=sha(ROOT/'units.jsonl'),mapping_sha256=sha(ROOT/'mapping_PRIVATE.jsonl.gz'),rubric_sha256=sha(ROOT/'RUBRIC.txt'),schema_sha256=sha(ROOT/'SCHEMA.json'),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
 print(json.dumps(read(ROOT/'PREPARED.json')),flush=True)
if __name__=='__main__':main()
