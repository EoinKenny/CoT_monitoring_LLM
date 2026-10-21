"""Native Qwen template, locate rendered content by sentinel substitution with exact reconstruction."""
import copy,hashlib
from render_content import render_messages

def tokenize_record(tokenizer,record):
 messages=record['messages']
 def render(mm):return render_messages(tokenizer,mm)
 text=render(messages);spans=[]
 for mi,m in enumerate(messages):
  if record['role']!='all' and m['role']!=record['role']:continue
  content=m['content'];assert isinstance(content,str)
  if not content.strip():continue
  marker='PROBE_CONTENT_SENTINEL_'+hashlib.sha256((str(mi)+content).encode()).hexdigest()
  assert marker not in text
  changed=copy.deepcopy(messages);changed[mi]['content']=marker
  marked=render(changed);assert marked.count(marker)==1,(record['id'],mi,'sentinel missing')
  before,after=marked.split(marker)
  # These equality checks prevent template markers or another message from entering the pool.
  if text==before+content+after:body=content
  elif text==before+content.strip()+after:body=content.strip()
  else:raise ValueError(f'Nonliteral native template content at record {record["id"]}, message {mi}; needs explicit handling, no silent truncation')
  spans.append(dict(start=len(before),end=len(before)+len(body),message_index=mi,role=m['role'],text=body))
 enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 special=set(tokenizer.all_special_ids)
 mask=[i for i,(tid,(a,b)) in enumerate(zip(enc['input_ids'],enc['offset_mapping'])) if tid not in special and b>a and any(a<s['end'] and b>s['start'] for s in spans)]
 assert mask,(record['id'],'empty content mask')
 return dict(token_ids=enc['input_ids'],mask=mask,n_tokens=len(enc['input_ids']),content_spans=spans,rendered_sha256=hashlib.sha256(text.encode()).hexdigest())
