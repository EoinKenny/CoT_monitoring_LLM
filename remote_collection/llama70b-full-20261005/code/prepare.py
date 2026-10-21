"""Retokenize every frozen eligible example; preserve memberships and content rules."""
from common import *
import ast,numpy as np
from transformers import AutoTokenizer
from runtime import batches
B.mkdir(exist_ok=True);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
mask_source=E/'logistic-probes-20261001/train.py';tree=ast.parse(mask_source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='content_mask')
ns=dict(np=np,json=json,tokenizer=tok,special_ids=set(tok.all_special_ids),config={'template_date_string':'01 Oct 2026'})
exec(compile(ast.Module(body=[fn],type_ignores=[]),'audited_content_mask','exec'),ns)
policy=read(ROOT/'BATCH_POLICY.json');summary={}
for ds in DATASETS:
 reserve();done=B/f'{ds}_PREPARED.json'
 if done.exists():
  mark=read(done)
  for name,h in mark['sha256'].items():assert sha(B/name)==h
  assert mark['source_sha256']==sha(PRIOR/'llama'/f'{ds}_metadata.json');summary[ds]=mark;continue
 old=read(PRIOR/'llama'/f'{ds}_metadata.json');rows=[]
 for i,r in enumerate(old):
  text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026');ids=tok(text,add_special_tokens=False)['input_ids'];proxy={'assistant':'wildguard','user':'toxicchat','all':'high_stakes'}[r['role']];mask=ns['content_mask'](r['messages'],np.array(ids),proxy);indices=np.flatnonzero(mask).tolist();assert indices and not any(ids[j] in ns['special_ids'] for j in indices)
  record={k:r[k] for k in ['id','messages','label','split','group','input_hash','role','immutable_id']};assert record['immutable_id']==identity(ds,record)
  record.update(token_ids=ids,mask=indices,n_tokens=len(ids));rows.append(record)
  if (i+1)%10000==0:print(json.dumps(dict(event='tokenizing',dataset=ds,completed=i+1,total=len(old))),flush=True)
 assert len({r['immutable_id'] for r in rows})==len(rows)
 for a in ['train','val','test']:
  for c in ['train','val','test']:
   if a!=c:
    for key in ['group','input_hash']:assert not ({r[key] for r in rows if r['split']==a}&{r[key] for r in rows if r['split']==c})
 schedule=batches(rows,policy['max_batch'],policy['max_tokens']);assert sorted(i for ix in schedule for i in ix)==list(range(len(rows)))
 write(B/f'{ds}_metadata.json',rows);write(B/f'{ds}_BATCHES.json',schedule)
 write(B/f'{ds}_SPLITS.json',dict(source_sha256=sha(PRIOR/'llama'/f'{ds}_metadata.json'),mask_source_sha256=sha(mask_source),split_records=[{k:r[k] for k in ['immutable_id','split','label','group','input_hash']} for r in rows]))
 mark=dict(rows=len(rows),batches=len(schedule),padded_tokens=sum(len(ix)*max(rows[i]['n_tokens'] for i in ix) for ix in schedule),source_sha256=sha(PRIOR/'llama'/f'{ds}_metadata.json'),sha256={f'{ds}_{s}':sha(B/f'{ds}_{s}') for s in ['metadata.json','BATCHES.json','SPLITS.json']});write(done,mark);summary[ds]=mark
 print(json.dumps(dict(event='prepared',dataset=ds,rows=len(rows),batches=len(schedule))),flush=True)
write(ROOT/'PREPARED.json',dict(datasets=summary,rows=sum(s['rows'] for s in summary.values()),batches=sum(s['batches'] for s in summary.values()),extraction_hours=sum(s['batches']*policy['seconds_per_batch']+s['padded_tokens']*policy['seconds_per_padded_token'] for s in summary.values())/3600,policy_sha256=sha(ROOT/'BATCH_POLICY.json')))
