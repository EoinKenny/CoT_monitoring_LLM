"""Validate independent context coding and report coverage/agreement, never infer codes."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,itertools
A=Path(__file__).resolve().parent
SUPPORT={'supported','unsupported_inference','misreads_text','insufficient_context','no_checkable_claim'}
TASK={'consistent','criterion_shift','wrong_object','unclear'}
FLAGS={'imports_neighbour_facts_or_intent','topic_equated_to_class','relative_severity_crossing_without_rule','misses_stance_negation_or_role','neighbour_substitutes_for_source_check','changes_task_criterion','reference_label_prediction_conflict','unsupported_target_inference','contradictory_prose','none_identified','uncertain'}
AXES={'LABEL','INTENT','BEHAVIOUR','SEVERITY','CONTEXT','FACT','TOPIC','OTHER','NONE'}

records=[];coverage=[]
for p in sorted((A/'packets').glob('context_*.jsonl')):
 ds=p.stem.removeprefix('context_');packet=[json.loads(l) for l in p.read_text().splitlines()];src={x['audit_id']:x for x in packet}
 readers=[]
 for d in sorted((A/'codes').iterdir()):
  if not d.is_dir():continue
  q=d/(ds+'.jsonl');r=d/(ds+'.receipt.json')
  if not q.exists() or not r.exists():continue
  rows=[json.loads(l) for l in q.read_text().splitlines()];receipt=json.load(open(r))
  assert receipt['all_contexts_and_rationales_read'] is True
  assert receipt['n']==len(rows)==len(packet)
  assert receipt['input_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
  assert len({x['audit_id'] for x in rows})==len(rows) and {x['audit_id'] for x in rows}==set(src)
  for x in rows:
   assert x['relation'] in 'NSCBU' and x['bridge'] in 'TLDRQU' and x['conclusion'] in 'PNMU' and x['quality'] in 'AIU'
   assert set(x['axes'])<=AXES and x['axes']
   assert x['context_support'] in SUPPORT and x['task_rule'] in TASK
   assert x['flags'] and set(x['flags'])<=FLAGS
   assert 'none_identified' not in x['flags'] or len(x['flags'])==1
   assert x['evidence']
   s=src[x['audit_id']]
   for e in x['evidence']:
    assert e['rationale_quote'] and e['rationale_quote'] in s['rationale'],(d.name,ds,x['audit_id'],'rationale')
    side=e['context_side'];assert side in ['target','neighbor','task']
    txt=s['task'] if side=='task' else '\n'.join(m['content'] for m in s['context'][side+'_messages'])
    assert e['context_quote'] and e['context_quote'] in txt,(d.name,ds,x['audit_id'],'context')
   records.append(dict(x,reader=d.name,dataset=ds))
  readers.append(d.name)
 coverage.append(dict(dataset=ds,n=len(packet),readers=readers,at_least_one=bool(readers),independent_double_read=len(readers)>=2))
byid=defaultdict(list)
for x in records:byid[x['audit_id']].append(x)
agreement=[];disagreements=[]
fields=['relation','bridge','conclusion','quality','context_support','task_rule']
for pair in itertools.combinations(sorted({x['reader'] for x in records}),2):
 matched=[]
 for aid,rr in byid.items():
  a={x['reader']:x for x in rr}
  if all(k in a for k in pair):matched.append((aid,a[pair[0]],a[pair[1]]))
 if not matched:continue
 for field in fields:
  a=Counter(x[field] for _,x,y in matched);b=Counter(y[field] for _,x,y in matched);n=len(matched)
  observed=sum(x[field]==y[field] for _,x,y in matched)/n
  expected=sum(a[k]*b[k] for k in set(a)|set(b))/n**2
  agreement.append(dict(readers=list(pair),field=field,n=n,agreement=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None))
 for aid,x,y in matched:
  diff=[f for f in fields if x[f]!=y[f]]
  if set(x['flags'])!=set(y['flags']):diff.append('flags')
  if diff:disagreements.append(dict(audit_id=aid,readers=list(pair),fields=diff,codes={pair[0]:x,pair[1]:y}))
(A/'validated_records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
(A/'disagreements.json').write_text(json.dumps(disagreements,indent=2)+'\n')
(A/'agreement.json').write_text(json.dumps(agreement,indent=2)+'\n')
status=dict(total_cases=200,read_cases=len(byid),code_records=len(records),double_read_cases=sum(len(v)>=2 for v in byid.values()),all_primary_read=len(byid)==200,all_double_read=len(byid)==200 and all(len(v)>=2 for v in byid.values()),adjudicated=False,coverage=coverage,scope='Verifies IDs and exact quotes; semantic correctness requires reading and independent adjudication. Independent assistant readers are not human validation.')
(A/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({k:v for k,v in status.items() if k!='coverage'},indent=2))
