from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2]/'context_audit'
def save(ds,spec):
 p=R/'packets'/f'context_{ds}.jsonl';rs=[json.loads(x) for x in p.read_text().splitlines()]
 out=[]
 for i,s in enumerate(spec):
  relation,axes,bridge,conclusion,quality,support,rule,flags,ev,note=s
  r=rs[i];o=dict(audit_id=r['audit_id'],relation=relation,axes=axes.split(','),bridge=bridge,conclusion=conclusion,quality=quality,context_support=support,task_rule=rule,flags=flags.split(','),evidence=[],note=note)
  for rq,side,cq,desc in ev:
   assert rq in r['rationale'],(ds,i,'rationale',rq)
   texts=[r['task']] if side=='task' else [x['content'] for x in r['context'][side+'_messages']]
   assert any(cq in x for x in texts),(ds,i,side,cq)
   o['evidence'].append(dict(rationale_quote=rq,context_side=side,context_quote=cq,supports=desc))
  out.append(o)
 assert len(out)==len(rs)
 dest=R/'codes/trace_inventory'/f'{ds}.jsonl';dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 dest.with_suffix('.receipt.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=len(out),all_contexts_and_rationales_read=True,reader='trace_inventory',coding_version='context-v1'),indent=2)+'\n')
 print(ds,len(out),'validated')
