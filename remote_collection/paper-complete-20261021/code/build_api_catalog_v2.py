from common import *
import sqlite3,collections,csv,sys
OUT=ROOT/'api_export';OUT.mkdir(exist_ok=True)
OLD=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra'];NEW=['claude-opus-5.5','gpt-5.5'];ALL=OLD+NEW
old=(ROOT/'OLD_OUTPUT.txt').read_text();after=(ROOT/'AFTER_OUTPUT.txt').read_text();before=(ROOT/'BEFORE_OUTPUT.txt').read_text().rstrip('\n')
assert (ROOT/'PREPARED.json').exists();path=OUT/'prompts.sqlite';assert not path.exists();db=sqlite3.connect(path);db.executescript('CREATE TABLE strings(id INTEGER PRIMARY KEY,text TEXT NOT NULL);CREATE TABLE prompts(prompt_id TEXT PRIMARY KEY,study TEXT,recipe TEXT,messages_sha256 TEXT,case_key TEXT,split TEXT,condition TEXT);CREATE TABLE tasks(model TEXT,prompt_id TEXT,PRIMARY KEY(model,prompt_id));CREATE TABLE pairs(case_key TEXT PRIMARY KEY,baseline TEXT,nearest TEXT,reason_after TEXT,reason_first TEXT);CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);')
strings={};counts=collections.Counter();pairs={};proof={}
def intern(s):
 if s not in strings:
  i=len(strings)+1;strings[s]=i;db.execute('INSERT INTO strings VALUES (?,?)',(i,s))
 return strings[s]
def add(r,study,models,pid=None):
 msgs=r['messages'];pid=pid or r['prompt_id'];recipe=[[m['role'],[intern(v) for v in m['content'].splitlines(keepends=True)]] for m in msgs];mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')));case=digest(r.get('backbone','')+':'+r.get('sample_id',pid))
 db.execute('INSERT INTO prompts VALUES (?,?,?,?,?,?,?)',(pid,study,json.dumps(recipe,separators=(',',':')),mh,case,r.get('split','test'),r.get('condition',study)))
 for m in models:db.execute('INSERT INTO tasks VALUES (?,?)',(m,pid));counts[m,study]+=1
 return pid
parent=PARENT/'canonical/cases.jsonl';proof[str(parent)]=sha(parent)
for i,r in enumerate(jl(parent)):
 add(r,'main',NEW)
 if r['split']=='test' and r['condition'] in ['baseline','nearest']:
  key=digest(r['backbone']+':'+r['sample_id']);pairs.setdefault(key,{})[r['condition']]=r['prompt_id']
  if r['condition']=='nearest':
   for study,new in [('reason_after',after),('reason_first',before)]:
    msgs=[dict(m) for m in r['messages']];assert msgs[0]['content'].count(old)==1;msgs[0]['content']=msgs[0]['content'].replace(old,new,1)
    if study=='reason_first':
     cue='Positive-class probability (0-100):';assert msgs[1]['content'].endswith(cue);msgs[1]['content']=msgs[1]['content'][:-len(cue)]+'Assessment (VERY brief reasoning, then positive-class probability):'
    mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')));pid=('apir_' if study=='reason_after' else 'apif_')+digest(r['prompt_id']+':'+mh);add(dict(r,messages=msgs,condition=study),study,[],pid);pairs[key][study]=pid
 if i%10000==0:db.commit();print('main',i,flush=True)
assert len(pairs)==25000 and all(len(v)==4 for v in pairs.values())
for key,v in pairs.items():db.execute('INSERT INTO pairs VALUES (?,?,?,?,?)',(key,v['baseline'],v['nearest'],v['reason_after'],v['reason_first']))
# Old API selections exactly reproduce the previously exported 7122 after prompts.
oldreason=ROOT/'old_api_reason_prompts.csv';proof[str(oldreason)]=sha(oldreason);n=0
for r in csv.DictReader(oldreason.open()):
 pid=r['prompt_id'];stored=db.execute('SELECT recipe FROM prompts WHERE prompt_id=?',(pid,)).fetchone();assert stored is not None,pid
 reconstructed=[dict(role=role,content=''.join(db.execute('SELECT text FROM strings WHERE id=?',(i,)).fetchone()[0] for i in ids)) for role,ids in json.loads(stored[0])];assert reconstructed==[dict(role='system',content=r['system_prompt']),dict(role='user',content=r['user_prompt'])]
 first=db.execute('SELECT reason_first FROM pairs WHERE reason_after=?',(pid,)).fetchone()[0]
 for m in json.loads(r['models']):
  assert m in OLD
  for study,p in [('reason_after',pid),('reason_first',first)]:db.execute('INSERT INTO tasks VALUES (?,?)',(m,p));counts[m,study]+=1
  n+=1
assert n==7122
for b in BBS:
 f=ROOT/'canonical'/f'{b}.jsonl';proof[str(f)]=sha(f)
 for r in jl(f):add(r,'ablation',ALL,'apiabl_'+digest(r['backbone']+':'+r['prompt_id']))
 db.commit();print('ablation',b,flush=True)
f=BASE/'two-nn-20261014/canonical/cases.jsonl';proof[str(f)]=sha(f)
for r in jl(f):add(r,'two_nn',ALL)
f=ROOT/'canonical/mechanism.jsonl';proof[str(f)]=sha(f)
for r in jl(f):add(r,'mechanism',ALL)
assert sum(counts.values())==2864244
assert all(counts[m,'ablation']==350000 and counts[m,'two_nn']==50000 and counts[m,'mechanism']==25000 for m in ALL)
db.commit();db.execute('CREATE INDEX tasks_prompt ON tasks(prompt_id)');db.execute('CREATE INDEX prompts_study ON prompts(study)');db.execute('CREATE INDEX pairs_after ON pairs(reason_after)');db.commit()
manifest=dict(models=ALL,static_calls=sum(counts.values()),calls_per_model={m:{s:n for (mm,s),n in counts.items() if mm==m} for m in ALL},unique_prompts=db.execute('SELECT count(*) FROM prompts').fetchone()[0],new_judge_trace_calls='Dynamic: twice the number of own valid unambiguous control-versus-NN test flips at50; max50000 per new model',native_thinking=False,no_target_gold_labels=True,retained_reference_labels=True,old_trace_selections_reused=True,format='Lossless string-interned SQLite catalog; materialize CSV or stream exact messages with kit.py. Metadata never sent to judge.')
db.execute('INSERT INTO metadata VALUES (?,?)',('manifest',json.dumps(manifest)));db.commit();db.execute('VACUUM');db.close();write(OUT/'manifest.json',manifest);write(ROOT/'API_SOURCE_PROOF_PRIVATE.json',proof);print(json.dumps(manifest),flush=True)
