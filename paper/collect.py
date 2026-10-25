"""Fresh collection on frozen prompts, using local or hosted HTTP endpoints.

No implicit retries. A durable dispatch record precedes each request. A dispatched
request without a result stays ambiguous on resume and is never resubmitted.
"""
import concurrent.futures,contextlib,functools,gzip,hashlib,importlib.util,json,os,re,shutil,sqlite3,time,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CAPS={'main':8,'two_nn':8,'ablation':8,'mechanism':128,'reason_after':256,'reason_first':512,'trace_labels':128}
NORMAL={'stop','end_turn','completed','eos','stop_sequence'}
INT=re.compile(r'(?:0|[1-9][0-9]?|100)')

def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()

def parse(study,text,finish):
 lines=text.strip().splitlines();tail=lines[-1].strip() if lines else '';head=lines[0].strip() if lines else ''
 if study in ['main','two_nn','ablation','reason_first']:
  if finish in NORMAL and INT.fullmatch(tail):return int(tail),'valid'
 elif study=='reason_after':
  if finish in NORMAL|{'length','max_tokens','max_output_tokens'} and INT.fullmatch(head):return int(head),'valid'
 else:
  def strict(pairs):
   d={}
   for k,v in pairs:
    if k in d:raise ValueError('duplicate key')
    d[k]=v
   return d
  t=text.strip()
  if t.startswith('```') and t.endswith('```'):t='\n'.join(t.splitlines()[1:-1])
  try:
   value=json.loads(t,object_pairs_hook=strict)
   expected={'labelled_similarity','important_contrast'} if study=='mechanism' else {'similarity','contrast','assessable'}
   assert finish in NORMAL and set(value)==expected and all(v in ['yes','no','uncertain'] for v in value.values())
   if study=='trace_labels' and value['assessable']=='no':assert value['similarity']==value['contrast']=='uncertain'
   return value,'valid'
  except (ValueError,TypeError,AssertionError):pass
 return None,'invalid'

def load_catalog(out):
 directory=out/'catalog';directory.mkdir(exist_ok=True)
 for name in ['catalog.jsonl.xz','CATALOG_LOGICAL.json','CHECKSUMS.json','restore.py','kit.py','manifest.json','model_manifest.template.json','README.txt','responses_template.csv','task_counts.csv','prompt_examples.csv','QA.json','test_kit.py']:
  target=directory/name
  if not target.exists():shutil.copy2(ROOT/'data/prompts'/name,target)
 spec=importlib.util.spec_from_file_location('nn_catalog_restore',directory/'restore.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.main()
 return sqlite3.connect('file:'+str(directory/'prompts.sqlite')+'?mode=ro',uri=True)

def initialise(path,config):
 con=sqlite3.connect(path);con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
 con.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
 con.execute('CREATE TABLE IF NOT EXISTS requests(model TEXT,prompt_id TEXT,study TEXT,case_key TEXT,condition TEXT,split TEXT,messages_hash TEXT,status TEXT,dispatch_time REAL,result_time REAL,response TEXT,parsed TEXT,finish_reason TEXT,returned_model TEXT,raw_json TEXT,error TEXT,latency_ms REAL,input_tokens INTEGER,output_tokens INTEGER,PRIMARY KEY(model,prompt_id))')
 con.execute('CREATE TABLE IF NOT EXISTS trace_sources(source_model TEXT,source_prompt TEXT,label_model TEXT,label_prompt TEXT,PRIMARY KEY(source_model,source_prompt,label_model))')
 cfg=json.dumps(config,sort_keys=True);prior=con.execute('SELECT value FROM metadata WHERE key="config"').fetchone()
 if prior and prior[0]!=cfg:raise ValueError('Configuration differs from this collection. Use a new output directory.')
 con.execute('INSERT OR IGNORE INTO metadata VALUES ("config",?)',(cfg,));con.commit();return con

def payload(model,job):
 provider=model['provider'];cap=CAPS[job['study']]
 body={'model':model['model'],'temperature':0}
 messages=job['messages']
 if provider=='anthropic':
  system=[x['content'] for x in messages if x['role']=='system']
  body.update(messages=[x for x in messages if x['role']!='system'],max_tokens=cap)
  if system:body['system']='\n\n'.join(system)
 else:
  body.update(messages=messages)
  body[model.get('token_parameter','max_completion_tokens')]=cap
 extra=model.get('extra_body',{})
 forbidden={'model','messages','system','temperature','max_tokens','max_completion_tokens','stream'}&extra.keys()
 if forbidden:raise ValueError(f'Protocol fields cannot be overridden: {forbidden}')
 body.update(extra);return body

def request(model,job,transport=None):
 body=payload(model,job);start=time.monotonic()
 headers={'Content-Type':'application/json'}
 env=model.get('api_key_env')
 if env:
  key=os.environ.get(env)
  if not key:raise ValueError(f'Missing environment variable {env}')
  headers['x-api-key' if model['provider']=='anthropic' else 'Authorization']=key if model['provider']=='anthropic' else 'Bearer '+key
 if model['provider']=='anthropic':headers['anthropic-version']='2023-06-01'
 r=None;raw_body=None
 try:
  if transport:r=transport(body)
  else:
   req=urllib.request.Request(model['url'],data=json.dumps(body).encode(),headers=headers,method='POST')
   with urllib.request.urlopen(req,timeout=model.get('timeout_seconds',300)) as response:
    raw_body=response.read().decode('utf-8');r=json.loads(raw_body)
  if model['provider']=='anthropic':
   text=''.join(x['text'] for x in r.get('content',[]) if x.get('type')=='text');finish=r.get('stop_reason','');use=r.get('usage',{});it=use.get('input_tokens');ot=use.get('output_tokens')
  else:
   choice=r['choices'][0];text=choice['message'].get('content') or '';finish=choice.get('finish_reason','');use=r.get('usage',{});it=use.get('prompt_tokens');ot=use.get('completion_tokens')
  parsed,status=parse(job['study'],text,finish)
  return dict(response=text,parsed=parsed,status=status,finish_reason=finish,returned_model=r.get('model'),raw_json=r,error=None,latency_ms=(time.monotonic()-start)*1000,input_tokens=it,output_tokens=ot)
 except urllib.error.HTTPError as e:
  raw=e.read().decode(errors='replace');return dict(response='',parsed=None,status='error',finish_reason='http_error',returned_model=None,raw_json={'http_status':e.code,'body':raw},error=f'HTTP {e.code}',latency_ms=(time.monotonic()-start)*1000,input_tokens=None,output_tokens=None)
 except Exception as e:
  # Timeout/network or malformed provider envelopes may have consumed a request.
  return dict(response='',parsed=None,status='ambiguous',finish_reason='exception',returned_model=None,raw_json=r if r is not None else {'raw_body':raw_body},error=type(e).__name__+': '+str(e),latency_ms=(time.monotonic()-start)*1000,input_tokens=None,output_tokens=None)

def run_jobs(con,model,jobs,workers,limit=None,transport=None):
 alias=model['alias'];count=0;done=0
 attempted={x[0] for x in con.execute('SELECT prompt_id FROM requests WHERE model=?',(alias,))}
 def next_job(iterator):
  nonlocal count
  for job in iterator:
   if job['prompt_id'] in attempted:continue
   if limit is not None and count>=limit:return None
   if shutil.disk_usage(Path(con.execute('PRAGMA database_list').fetchone()[2]).parent).free < 20*2**30:raise RuntimeError('Stopped: less than 20 GiB free disk remains.')
   payload(model,job) # Validate before recording a dispatch.
   con.execute('INSERT INTO requests(model,prompt_id,study,case_key,condition,split,messages_hash,status,dispatch_time) VALUES (?,?,?,?,?,?,?,?,?)',(alias,job['prompt_id'],job['study'],job.get('case_key'),job.get('condition'),job.get('split'),digest(job['messages']),'dispatched',time.time()));con.commit()
   attempted.add(job['prompt_id']);count+=1;return job
  return None
 iterator=iter(jobs)
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  pending={}
  while True:
   while len(pending)<workers:
    job=next_job(iterator)
    if job is None:break
    pending[pool.submit(request,model,job,transport)]=job
   if not pending:break
   completed,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for fut in completed:
    job=pending.pop(fut);r=fut.result()
    con.execute('UPDATE requests SET status=?,result_time=?,response=?,parsed=?,finish_reason=?,returned_model=?,raw_json=?,error=?,latency_ms=?,input_tokens=?,output_tokens=? WHERE model=? AND prompt_id=?',(r['status'],time.time(),r['response'],json.dumps(r['parsed']),r['finish_reason'],r['returned_model'],json.dumps(r['raw_json']),r['error'],r['latency_ms'],r['input_tokens'],r['output_tokens'],alias,job['prompt_id']));con.commit();done+=1
    if done%100==0:print(alias,job['study'],done,'new results',flush=True)
 return count

def static_jobs(catalog,study):
 @functools.lru_cache(maxsize=150000)
 def string(i):return catalog.execute('SELECT text FROM strings WHERE id=?',(i,)).fetchone()[0]
 for pid,recipe,mh,case,condition,split in catalog.execute('SELECT prompt_id,recipe,messages_sha256,case_key,condition,split FROM prompts WHERE study=? ORDER BY rowid',(study,)):
  messages=[dict(role=role,content=''.join(string(i) for i in ids)) for role,ids in json.loads(recipe)]
  assert hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==mh
  yield dict(prompt_id=pid,study=study,messages=messages,case_key=case,condition=condition,split=split)

def flip_jobs(con,catalog,alias,study):
 valid={pid:json.loads(p) for pid,p in con.execute('SELECT prompt_id,parsed FROM requests WHERE model=? AND study="main" AND status="valid"',(alias,))}
 selected=set()
 for case,a,b,after,first in catalog.execute('SELECT * FROM pairs ORDER BY case_key'):
  if a in valid and b in valid and (valid[a]>=50)!=(valid[b]>=50):selected.add(after if study=='reason_after' else first)
 for job in static_jobs(catalog,study):
  if job['prompt_id'] in selected:yield job

def trace_jobs(con,labeler):
 template=(ROOT/'data/prompts/trace_label_prompt.txt').read_text();seen=set()
 # Only the written explanation is sent; no target label, earlier scores, or outcome.
 for source,pid,text in con.execute('SELECT model,prompt_id,response FROM requests WHERE study="reason_after" AND status="valid" ORDER BY model,prompt_id').fetchall():
  lines=text.strip().splitlines()
  if lines and INT.fullmatch(lines[0].strip()):lines=lines[1:]
  if lines and INT.fullmatch(lines[-1].strip()):lines=lines[:-1]
  trace='\n'.join(lines).strip()
  if not trace:continue
  messages=[dict(role='user',content=template+json.dumps(trace,ensure_ascii=False))];key='trace_'+digest(messages)
  con.execute('INSERT OR IGNORE INTO trace_sources VALUES (?,?,?,?)',(source,pid,labeler,key));con.commit()
  if key not in seen:
   seen.add(key);yield dict(prompt_id=key,study='trace_labels',messages=messages)

def export_scores(con,out):
 import numpy as np
 design=json.load(gzip.open(ROOT/'data/paper/analysis/design.json.gz','rt'))
 cases=[dict(backbone=b,**r) for b,rr in design['samples'].items() for r in rr]
 index={hashlib.sha256((r['backbone']+':'+r['sample_id']).encode()).hexdigest():i for i,r in enumerate(cases)}
 main=['baseline','saliency','nearest2','nearest'];abl=['baseline','nearest','text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full']
 models=[x[0] for x in con.execute('SELECT DISTINCT model FROM requests WHERE study="main" ORDER BY model')]
 for filename,conditions,studies in [('main_scores.npz',main,{'main','two_nn'}),('ablation_scores.npz',abl,{'main','ablation'})]:
  arrays={m:np.full((len(cases),len(conditions)),-1,np.int16) for m in models}
  for m,case,c,study,value in con.execute('SELECT model,case_key,condition,study,parsed FROM requests WHERE status="valid"'):
   if m in arrays and study in studies and c in conditions:arrays[m][index[case],conditions.index(c)]=json.loads(value)
  np.savez_compressed(out/filename,**arrays)
 summary=[dict(model=m,study=s,status=t,n=n) for m,s,t,n in con.execute('SELECT model,study,status,count(*) FROM requests GROUP BY model,study,status')]
 (out/'COVERAGE.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary

def collect(config_path,out,studies,limit,dry):
 config=json.loads(config_path.read_text());models=config['models']
 if not models or len({m['alias'] for m in models})!=len(models):raise ValueError('Supply unique model aliases.')
 if set(studies)-CAPS.keys():raise ValueError('Unknown study')
 if limit is not None and limit<1:raise ValueError('Limit must be positive')
 for m in models:
  if m['provider'] not in ['openai-compatible','anthropic']:raise ValueError('Unsupported provider')
  if m.get('native_thinking') is not False:raise ValueError('Record native_thinking:false and configure its model-specific disabling option.')
  if not m.get('model') or not m.get('url'):raise ValueError('Exact backend model ID and endpoint are required.')
 counts={'main':150000,'two_nn':50000,'ablation':350000,'mechanism':25000,'reason_after':'own-model valid test flips at 50','reason_first':'own-model valid test flips at 50','trace_labels':'deduplicated available probability-first explanations'}
 if dry:
  print(json.dumps({'models':[m['alias'] for m in models],'calls_per_model':{s:counts[s] for s in studies},'limit_per_model_per_stage':limit,'network_requests':0,'inputs':'frozen probes, splits, references and exact prompts'},indent=2));return
 for m in models:
  if 'YOUR_' in m['model'] or 'YOUR_' in m['url']:raise ValueError('Replace example model ID/endpoint before collection.')
  if m.get('api_key_env') and not os.environ.get(m['api_key_env']):raise ValueError('Missing '+m['api_key_env'])
 if 'trace_labels' in studies and config.get('trace_labeler') not in {m['alias'] for m in models}:raise ValueError('trace_labeler must name a configured model alias.')
 out.mkdir(parents=True,exist_ok=True)
 # OS lock prevents concurrent controllers; a stale lock file itself is harmless.
 import fcntl
 with (out/'collection.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  con=initialise(out/'responses.sqlite',config);catalog=load_catalog(out)
  for m in models:
   for study in studies:
    if study=='trace_labels':continue
    if study.startswith('reason_'):
     attempted=con.execute('SELECT count(*) FROM requests WHERE model=? AND study="main"',(m['alias'],)).fetchone()[0]
     if attempted!=150000 and limit is None:raise ValueError('Account for every main request before selecting reasoning cases; run main first.')
     jobs=flip_jobs(con,catalog,m['alias'],study)
    else:jobs=static_jobs(catalog,study)
    n=run_jobs(con,m,jobs,config.get('concurrency',4),limit);print(m['alias'],study,n,'new dispatches',flush=True)
  if 'trace_labels' in studies:
   m=next(x for x in models if x['alias']==config['trace_labeler']);run_jobs(con,m,trace_jobs(con,m['alias']),config.get('concurrency',4),limit)
  summary=export_scores(con,out);catalog.close();con.close()
  print(json.dumps(summary,indent=2));print('Raw responses, attempts, configuration and scores saved in',out)
