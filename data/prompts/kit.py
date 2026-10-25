"""Offline, provider-independent task/validation helper. Makes NO API calls."""
from pathlib import Path
import argparse,collections,csv,functools,hashlib,json,re,sqlite3,sys
ROOT=Path(__file__).resolve().parent
MODELS=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra','claude-opus-5.5','gpt-5.5'];NEW=MODELS[-2:]
CAPS={'main':8,'ablation':8,'two_nn':8,'mechanism':128,'reason_after':256,'reason_first':512}
NORMAL={'stop','end_turn','completed','eos','stop_sequence'}
FIELDS=['prompt_id','model','response','status','finish_reason','input_tokens','output_tokens','latency_ms']
INT=r'(?:0|[1-9][0-9]?|100)'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def connect():return sqlite3.connect('file:'+str(ROOT/'prompts.sqlite')+'?mode=ro',uri=True)
def strict_object(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate JSON key')
  d[k]=v
 return d
def parse(study,text,finish):
 """Return parsed answer and status; caller retains untouched raw response."""
 text=text.strip();complete=finish in NORMAL
 if study in ['main','ablation','two_nn']:
  return (int(text),'ok') if complete and re.fullmatch(INT,text) else (None,'invalid')
 if study=='mechanism':
  try:
   x=json.loads(text,object_pairs_hook=strict_object);assert complete and isinstance(x,dict) and set(x)=={'labelled_similarity','important_contrast'} and all(v in ['yes','no','uncertain'] for v in x.values());return x,'ok'
  except Exception:return None,'invalid'
 if study=='reason_after':
  first=text.split('\n',1)[0].strip()
  # First probability remains scored if a later explanation is capped or contradictory.
  return (int(first),'ok') if re.fullmatch(INT,first) and finish in NORMAL|{'length','max_tokens','max_output_tokens'} else (None,'invalid')
 if study=='reason_first':
  parts=text.rsplit('\n',1)
  return (int(parts[1].strip()),'ok') if complete and len(parts)==2 and parts[0].strip() and re.fullmatch(INT,parts[1].strip()) else (None,'invalid')
 raise ValueError(study)
def verify():
 checks=json.loads((ROOT/'CHECKSUMS.json').read_text())
 for name,h in checks.items():assert sha(ROOT/name)==h,name
 from restore import verify_catalog
 db=connect();verify_catalog(db);assert db.execute('SELECT count(*) FROM tasks LEFT JOIN prompts USING(prompt_id) WHERE prompts.prompt_id IS NULL').fetchone()[0]==0
 print(json.dumps(json.loads((ROOT/'manifest.json').read_text()),indent=2))
def decode(db,row):
 pid,study,recipe,mh=row;messages=[dict(role=role,content=''.join(string(db,i) for i in ids)) for role,ids in json.loads(recipe)];assert digest(json.dumps(messages,ensure_ascii=False,separators=(',',':')))==mh
 return dict(prompt_id=pid,study=study,messages=messages,messages_sha256=mh,max_output_tokens=CAPS[study],temperature=0,seed=20261001,native_reasoning=False)
@functools.lru_cache(maxsize=150000)
def string(db,i):return db.execute('SELECT text FROM strings WHERE id=?',(i,)).fetchone()[0]
def iter_jobs(model,study,dynamic=None):
 assert model in MODELS and study in CAPS;db=connect()
 if dynamic:
  selection=json.loads(Path(dynamic).read_text());assert selection['model']==model and selection['catalog_sha256']==sha(ROOT/'prompts.sqlite')
  if study not in ['reason_after','reason_first']:raise ValueError('Dynamic selection only for reasoning')
  for pid in selection['prompt_ids'][study]:
   row=db.execute('SELECT prompt_id,study,recipe,messages_sha256 FROM prompts WHERE prompt_id=?',(pid,)).fetchone();assert row and row[1]==study;yield dict(model=model,**decode(db,row))
 else:
  if model in NEW and study in ['reason_after','reason_first']:raise ValueError('First run main, then select-traces; supply --selection')
  for row in db.execute('SELECT p.prompt_id,p.study,p.recipe,p.messages_sha256 FROM tasks t JOIN prompts p USING(prompt_id) WHERE t.model=? AND p.study=? ORDER BY p.rowid',(model,study)):yield dict(model=model,**decode(db,row))
 db.close()
def journal_state(path,model):
 dispatch=collections.Counter();results=collections.Counter();result_status={}
 with Path(path).open() as f:
  for line in f:
   r=json.loads(line)
   if r['model']!=model:continue
   pid=r['prompt_id']
   if r['event']=='dispatch':dispatch[pid]+=1
   elif r['event']=='result':results[pid]+=1;result_status[pid]=r['status']
 return dispatch,results,result_status
def select_traces(model,response_path,journal_path,out):
 assert model in NEW;db=connect();need={x[0] for x in db.execute('SELECT prompt_id FROM prompts WHERE study="main"')};responses={}
 with Path(response_path).open(newline='') as f:
  for r in csv.DictReader(f):
   if r['model']==model and r['prompt_id'] in need:
    assert r['prompt_id'] not in responses,'Duplicate response key';responses[r['prompt_id']]=r
 assert set(responses)==need,'Main stage must account for every one of the 150000 prompts before trace selection'
 dispatched,returned,status=journal_state(journal_path,model);selected={'reason_after':[],'reason_first':[]};counts=collections.Counter()
 for case,a,b,after,first in db.execute('SELECT * FROM pairs ORDER BY case_key'):
  vals=[]
  for pid in [a,b]:
   r=responses[pid];v,s=parse('main',r['response'],r['finish_reason']);valid=r['status']=='ok' and s=='ok' and dispatched[pid]==1 and returned[pid]==1 and status.get(pid)=='ok';vals.append(v if valid else None)
  if None in vals:counts['excluded_invalid_missing_or_ambiguous_pair']+=1
  elif (vals[0]>=50)==(vals[1]>=50):counts['unchanged']+=1
  else:counts['selected_flips']+=1;selected['reason_after'].append(after);selected['reason_first'].append(first)
 dest=Path(out);assert not dest.exists();dest.write_text(json.dumps(dict(model=model,counts=dict(counts),prompt_ids=selected,catalog_sha256=sha(ROOT/'prompts.sqlite'),response_sha256=sha(response_path),journal_sha256=sha(journal_path),rule='Own test-only valid unambiguous control-versus-NN decisions differ at50; no gold labels used'),indent=2));print(dict(counts))
def validate(response_path,journal_path):
 db=connect();seen=set();counts=collections.Counter();issues=[]
 with Path(response_path).open(newline='') as f:
  rd=csv.DictReader(f);assert rd.fieldnames==FIELDS
  for r in rd:
   key=r['model'],r['prompt_id'];assert key not in seen,'Duplicate response key';seen.add(key);assert r['model'] in MODELS
   meta=db.execute('SELECT study FROM prompts WHERE prompt_id=?',(r['prompt_id'],)).fetchone();assert meta,'Unknown prompt ID'
   value,parsed=parse(meta[0],r['response'],r['finish_reason'])
   assert r['status'] in ['ok','invalid','error','ambiguous','skipped']
   if r['status']=='ok':assert parsed=='ok',key
   counts[r['model']+'/'+meta[0]+'/'+r['status']]+=1
 for m in MODELS:
  d,ret,st=journal_state(journal_path,m)
  for (model,pid) in seen:
   if model==m and (d[pid]>1 or ret[pid]>1):issues.append([m,pid,'multiple attempts/results; retain and report, do not silently select one'])
 print(json.dumps(dict(counts=dict(counts),attempt_issues=issues),indent=2))
def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('verify')
 e=sub.add_parser('export-csv');e.add_argument('--model',required=True,choices=MODELS);e.add_argument('--study',required=True,choices=CAPS);e.add_argument('--out',required=True);e.add_argument('--selection')
 s=sub.add_parser('select-traces');s.add_argument('--model',required=True,choices=NEW);s.add_argument('--responses',required=True);s.add_argument('--journal',required=True);s.add_argument('--out',required=True)
 v=sub.add_parser('validate');v.add_argument('--responses',required=True);v.add_argument('--journal',required=True)
 a=ap.parse_args()
 if a.cmd=='verify':verify()
 elif a.cmd=='select-traces':select_traces(a.model,a.responses,a.journal,a.out)
 elif a.cmd=='validate':validate(a.responses,a.journal)
 else:
  with open(a.out,'x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['prompt_id','model','study','messages_json','max_output_tokens']);w.writeheader()
   for r in iter_jobs(a.model,a.study,a.selection):w.writerow(dict(prompt_id=r['prompt_id'],model=r['model'],study=r['study'],messages_json=json.dumps(r['messages'],ensure_ascii=False),max_output_tokens=r['max_output_tokens']))
if __name__=='__main__':main()
