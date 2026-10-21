from pathlib import Path
import json,hashlib
A=Path(__file__).resolve().parent
def write_audit(dataset,data,reader='root'):
 p=A/'packets'/('context_'+dataset+'.jsonl');rows=[json.loads(l) for l in p.read_text().splitlines()]
 assert len(data)==len(rows)
 out=[]
 for row,d in zip(rows,data):
  rel,axes,bridge,conclusion,support,taskrule,flags,rq,side,cq,note=d
  assert rq in row['rationale'],(row['audit_id'],rq)
  source=row['task'] if side=='task' else '\n'.join(m['content'] for m in row['context'][side+'_messages'])
  assert cq in source,(row['audit_id'],cq)
  out.append(dict(audit_id=row['audit_id'],relation=rel,axes=axes.split(','),bridge=bridge,conclusion=conclusion,quality='A',context_support=support,task_rule=taskrule,flags=flags,evidence=[dict(rationale_quote=rq,context_side=side,context_quote=cq,supports=note)],note=note))
 dest=A/'codes'/reader;dest.mkdir(exist_ok=True)
 (dest/(dataset+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 (dest/(dataset+'.receipt.json')).write_text(json.dumps(dict(reader='/'+reader,n=len(rows),all_contexts_and_rationales_read=True,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),coding_version='context-v1',outcomes_masked=True,notes='All complete task, target and neighbour texts and full rationales read. Original outcomes and first probability masked. Flags are reviewer assessments against supplied text, not inferences from gold or claims of private reasoning.'),indent=2)+'\n')
 print('Validated',len(rows),dataset,'masked full-context assessments')
