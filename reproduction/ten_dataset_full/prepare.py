import os,json,gzip,re,hashlib,random,time,traceback
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd
from transformers import AutoTokenizer
R=Path('/home/ubuntu/experiments/ten-dataset-full-20261002');R.mkdir(exist_ok=True);P=Path('/home/ubuntu/experiments/seven-dataset-pilot-20261002');M=json.loads((P/'downloads.json').read_text());(R/'downloads.json').write_text(json.dumps(M,indent=2));SEED=20261002
MODEL='/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);special=set(tok.all_special_ids)
def norm(s):return re.sub(r'\s+',' ',s).strip().casefold()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def read(p):
 if p.endswith('.parquet'):return pd.read_parquet(p).to_dict('records')
 if p.endswith('.tsv'):return pd.read_csv(p,sep='\t',keep_default_na=False).to_dict('records')
 if p.endswith('.gz'):
  with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
 s=Path(p).read_text()
 try:return json.loads(s)
 except json.JSONDecodeError:return [json.loads(l) for l in s.splitlines()]
D=defaultdict(list);stats=defaultdict(Counter)
def add(ds,ident,split,group,messages,label,role,links=None):
 assert label in [0,1]
 if any(not isinstance(m['content'],str) or not m['content'].strip() for m in messages):stats[ds]['empty']+=1;return
 D[ds].append(dict(id=str(ident),official_split=split,group=digest(norm(group)),messages=messages,label=int(label),role=role,links=links or []))
def msg(role,text):return dict(role=role,content=text.strip())
for ds in ['beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']:
 if ds not in M or not M[ds]['files']:continue
 files=M[ds]['files']
 if ds=='ragtruth':sources={r['source_id']:r for r in read(files['dataset/source_info.jsonl'])}
 for filename,path in files.items():
  if 'annotations.json' in filename or 'source_info' in filename:continue
  data=read(path);stats[ds]['raw_rows']+=len(data)
  split='test' if ('test' in filename or filename.startswith('eval/')) else ('val' if 'validation' in filename else 'train')
  for j,r in enumerate(data):
   ident=filename+':'+str(j)
   if ds=='beavertails':add(ds,ident,split,r['prompt'],[msg('user',r['prompt']),msg('assistant',r['response'])],int(not r['is_safe']),'assistant')
   elif ds=='aegis':
    if r['prompt']=='REDACTED' or r['prompt_label'] not in ['safe','unsafe']:stats[ds]['redacted_or_invalid']+=1;continue
    add(ds,ident,split,r['prompt'],[msg('user',r['prompt'])],int(r['prompt_label']=='unsafe'),'user')
   elif ds=='toxigen':
    y=float(r['toxicity_human'])
    if not 1<=y<=5:stats[ds]['invalid_rating']+=1;continue
    add(ds,ident,split,r['text'],[msg('user',r['text'])],int(y>3),'user')
   elif ds=='halueval':
    ctx='Evidence:\n'+r['knowledge']+'\n\nQuestion:\n'+r['question']
    for y,k in [(0,'right_answer'),(1,'hallucinated_answer')]:add(ds,ident+':'+str(y),'unsplit',r['knowledge'],[msg('user',ctx),msg('assistant',r[k])],y,'assistant')
   elif ds=='ragtruth':
    if r['quality']!='good':stats[ds]['quality_excluded']+=1;continue
    y=int(any(not lab.get('implicit_true',False) for lab in r['labels']))
    add(ds,r['id'],r['split'],r['source_id'],[msg('user',sources[r['source_id']]['prompt']),msg('assistant',r['response'])],y,'assistant')
   elif ds=='fava':
    c=r['completion'];tags=set(re.findall(r'</?(\w+)>',c));allowed={'entity','relation','contradictory','invented','subjective','unverifiable','mark','delete'}
    if tags-allowed:stats[ds]['unrecognized_tags']+=1;continue
    # Recover the ORIGINAL passage: mark=proposed correction, delete=original error.
    text=re.sub(r'<mark>.*?</mark>','',c,flags=re.S);text=re.sub(r'</?\w+>','',text).strip()
    if text not in r['prompt']:stats[ds]['reconstruction_mismatch']+=1;continue
    ctx=r['prompt'].split('Please identify')[0].strip()
    refs=re.split(r'Reference \[\d+\]:',ctx)[1:]
    if not refs:stats[ds]['missing_references']+=1;continue
    links=[digest(norm(z)) for z in refs]
    add(ds,ident,'unsplit',ctx,[msg('user',ctx),msg('assistant',text)],int(bool(tags & (allowed-{'mark','delete'}))),'assistant',links)
   elif ds=='wildjailbreak':
    typ=r['data_type']
    if typ not in ['vanilla_harmful','vanilla_benign','adversarial_harmful','adversarial_benign']:stats[ds]['invalid_type']+=1;continue
    text=r['adversarial'] if typ.startswith('adversarial') else r['vanilla']
    group=r.get('vanilla') or text
    add(ds,ident,split,group,[msg('user',text)],int(typ.endswith('_harmful')),'user')
 print('adapted',ds,len(D[ds]),flush=True)
 del data
inventory={}
for ds,rows in D.items():
 # Union groups sharing an identical input or FAVA reference passage.
 parent={r['group']:r['group'] for r in rows}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 seen={};byinput=defaultdict(list)
 for i,r in enumerate(rows):
  h=digest(json.dumps(r['messages'],sort_keys=True));r['input_hash']=h;byinput[h].append(i)
  for key in ['input:'+h]+['ref:'+z for z in r['links']]:
   if key in seen:union(r['group'],seen[key])
   else:seen[key]=r['group']
 conflicts={h for h,ix in byinput.items() if len({rows[i]['label'] for i in ix})>1}
 stats[ds]['conflicting_inputs']=len(conflicts)
 rows=[r for r in rows if r['input_hash'] not in conflicts]
 priority={'unsplit':0,'train':1,'val':2,'test':3};assignment={}
 for r in rows:
  r['group']=find(r['group']);g=r['group'];s=r['official_split']
  if g not in assignment or priority[s]>priority[assignment[g]]:assignment[g]=s
 pools=defaultdict(list);seen=set()
 for r in sorted(rows,key=lambda r:-priority[r['official_split']]):
  if r['input_hash'] in seen:stats[ds]['duplicates_removed']+=1;continue
  seen.add(r['input_hash']);g=r['group'];s=assignment[g]
  if r['official_split']!=s:stats[ds]['cross_split_group_excluded']+=1;continue
  u=int(digest(str(SEED)+g)[:12],16)/16**12
  if s=='unsplit':s='train' if u<.6 else ('val' if u<.8 else 'test')
  elif s=='train':s='val' if u<.2 else 'train'
  r['split']=s;pools[s,r['label']].append(r)
 selected=[];available={};rng=random.Random(SEED)
 for s in ['train','val','test']:
  quota=min(len(pools[s,0]),len(pools[s,1])) if s=='train' else None
  bins=[]
  for y in [0,1]:
   pool=pools[s,y];available[s+':'+str(y)]=len(pool);rng.shuffle(pool);chosen=[]
   for r in pool:
    text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026');enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    if len(enc['input_ids'])>4096:stats[ds]['overlength_candidates_skipped']+=1;continue
    spans=[]
    for k,m in enumerate(r['messages']):
     if m['role']!=r['role']:continue
     prefix=tok.apply_chat_template(r['messages'][:k+1],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026');assert text.startswith(prefix)
     a=prefix.rfind(m['content']);assert a>=0;spans.append((a,a+len(m['content'])))
    mask=[i for i,(t,(a,b)) in enumerate(zip(enc['input_ids'],enc['offset_mapping'])) if t not in special and b>a and any(b>x and a<z for x,z in spans)]
    if not mask:stats[ds]['empty_mask']+=1;continue
    r.update(token_ids=enc['input_ids'],mask=mask,n_tokens=len(enc['input_ids']));r.pop('links',None);chosen.append(r)
    if quota is not None and len(chosen)==quota:break
   bins.append(chosen)
  n=min(len(bins[0]),len(bins[1]));assert n>=20,(ds,s,n)
  selected+=(bins[0][:n]+bins[1][:n]) if s=='train' else bins[0]+bins[1]
 for a in ['train','val','test']:
  for b in ['train','val','test']:
   if a!=b:assert not ({r['group'] for r in selected if r['split']==a}&{r['group'] for r in selected if r['split']==b})
 p=R/(ds+'_selected.json');assert not p.exists();p.write_text(json.dumps(selected))
 inventory[ds]=dict(counts=dict(Counter(r['split']+':'+str(r['label']) for r in selected)),available=available,exclusions=dict(stats[ds]),examples=len(selected),tokens=sum(r['n_tokens'] for r in selected),max_tokens=max(r['n_tokens'] for r in selected),mean_tokens=sum(r['n_tokens'] for r in selected)/len(selected))
 print('selected',ds,json.dumps(inventory[ds]),flush=True)
(R/'new_inventory.json').write_text(json.dumps(inventory,indent=2));(R/'NEW_PREPARED').write_text('ok')
