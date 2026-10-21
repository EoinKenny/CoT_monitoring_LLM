import json,pathlib,hashlib
ROOT=pathlib.Path('reproduction/nn_rationale_mechanisms_20261020/context_audit')
OUT=ROOT/'codes/mechanism_design';OUT.mkdir(parents=True,exist_ok=True)
def add(dataset,i,relation,axes,bridge,conclusion,support,rule,flags,evidence,note,quality='A'):
 p=ROOT/'packets'/f'context_{dataset}.jsonl';src=[json.loads(x) for x in p.read_text().splitlines()];s=src[i]
 e=[]
 for rq,side,cq,why in evidence:
  assert rq in s['rationale'],(dataset,i,'rationale',rq)
  contexts=[s['task']] if side=='task' else [m['content'] for m in s['context'][f'{"neighbor" if side=="neighbor" else "target"}_messages']]
  assert any(cq in t for t in contexts),(dataset,i,side,cq)
  e.append(dict(rationale_quote=rq,context_side=side,context_quote=cq,supports=why))
 out=dict(audit_id=s['audit_id'],relation=relation,axes=axes.split(','),bridge=bridge,conclusion=conclusion,quality=quality,context_support=support,task_rule=rule,flags=flags.split(','),evidence=e,note=note)
 path=OUT/f'{dataset}.jsonl';a=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [];assert out['audit_id'] not in {x['audit_id'] for x in a},('duplicate',dataset,i);a.append(out);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in a));print(dataset,i,'validated')
 if len(a)==len(src):
  assert {x['audit_id'] for x in a}=={x['audit_id'] for x in src}
  (OUT/f'{dataset}.receipt.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=len(a),all_contexts_and_rationales_read=True,reader='/root/mechanism_design',coding_version='context-v1'),indent=2))
