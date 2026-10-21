#!/usr/bin/env python3
"""README — Reproducible API judge extension (run with --readme to print).

FILES: prompts.csv + this script. Python 3.10+ standard library only, apart from
YOUR existing llm_query.py dependencies/credentials. Do not copy credentials into
this package. Put both files alongside llm_query.py, or use --module-dir PATH.

1. Validate without importing llm_query or making API calls:
   python run_api_judges.py --dry-run
2. Optional live smoke test, fifteen cases per model, separate output directory:
   python run_api_judges.py --limit-per-model 15 --output smoke_api
3. Full collection, all nine models in llm_query.model_ids:
   python -u run_api_judges.py --output api_results
   # optional detached execution on the API server:
   nohup python -u run_api_judges.py --output api_results > api_run.log 2>&1 &
4. Resume with the EXACT same command. Completed batches are checksum-verified
   and not repeated. Return api_results/results.csv (or results.csv.gz) here.
   Keep prompts.csv, manifest.json and the state directory for provenance.

SCOPE: 40,179 frozen prompt rows = 13,393 selected backbone/dataset cases x three
conditions; 361,611 logical queries for nine judges. No deduplication across
backbones, no changed sampling or explanations, one call per model/prompt.
--models accepts a comma-separated subset. --batch-size defaults to 15 and may
be reduced, never increased above 15. Models run sequentially; each call_llm
batch contains <=15 prompts, all sharing the SAME system message.

The script imports llm_query.call_llm, replaces llm_query.sys_prompt for each
batch with the original classification system message, then submits ONLY the
user_prompt strings. This is essential: the attached module's default system
prompt is for a different human-surrogate study. Gold labels, sampling weights,
probe probabilities, IDs and local-judge predictions are NEVER sent to APIs.
A labelled neighbour already inside an NN user prompt remains intentionally.
No local-model special tokens/chat templates are sent to APIs.

DECODING: temperature=0.0 by default. There is no seed/token-constraint parameter
in this module. We do not modify it, force one-token generation, invent usage
statistics, or silently switch provider settings. API models can refuse or
return extra text; only a string whose .strip() is exactly '0' or '1' counts as
valid. Every other response is retained as invalid_response or api_error.
--temperatures '{"alias":1.0}' is an explicit protocol change for models that
reject zero; use a separate output directory, record it, and do not describe
that run as greedy/deterministic. Hosted APIs are not guaranteed reproducible.
The model mapping, module/script/input hashes, versions and temperatures are
frozen in manifest.json and copied into result rows. Wrapper/SDK-internal retries
are not exposed; the script itself never automatically retries a logical call.

RECOVERY: before each query batch we atomically save an intent; after it returns
we save the complete raw result payload, then commit parsed rows. results.csv
is an incremental convenience export rebuilt from committed batches on restart.
A crash AFTER saving the raw payload can be recovered without another API call.
An intent with NO returned payload is ambiguous: the script writes unresolved
rows and blocks that model, rather than silently reissuing billable calls.
A batch with a raised exception or API errors is saved and blocks the model;
other models continue. Invalid classifications remain recorded without retries.
To resume UNISSUED work after examining a block, pass --continue-blocked with
--recovery-note 'reason'. Previously attempted rows, including errors, are never
retried or replaced. Any unresolved/invalid rows prevent a valid full-grid result.
Do not delete checkpoint files or edit the input/module to bypass these checks.
For a deliberate rerun, use a new output directory and label it as a new run.

OUTPUT: results.csv has one row per attempted model/prompt, preserving stable
row_id, backbone/dataset/condition/sample ID, gold label, probe score/threshold,
stratum counts/weights, full-test probe confusion counts and all five original
local judge predictions. It also contains model alias/resolved ID, prompt/input/
module/script/config hashes, times, raw_response, status, prediction, and error.
Bring this CSV back even if incomplete; failures are identifiable. A compressed
results.csv.gz snapshot is also written at the end for easier transfer. Prompt text is
in the input CSV and linked by row_id/messages_sha256, avoiding ninefold copying.

SCORING FOR THE RECEIVING AGENT: do not compute deployment F1 directly from the
balanced sample. Within each backbone/dataset/model/condition, weight judged
positive outcomes by N_TP/n_TP and N_FP/n_FP to estimate retained TP and FP.
Then FN = original_FN + original_TP - retained_TP;
TN = original_TN + original_FP - retained_FP. Compute F1 from these counts.
Original confusion totals are repeated metadata; use them ONCE per dataset.
Report sample metrics separately. Do not drop failures or redistribute their
weights; no complete pipeline estimate exists until all selected cases have
valid classifications (or a separately declared missing-output policy is used).
The frozen study used all eligible training data, class_weight='balanced',
validation-F1 thresholds and seed 20261004 TP/FP sampling with scarce-stratum
fill-up. Probe-negative test cases bypass all judges. NN retrieval is train-only.
The baseline, saliency and nearest conditions must be paired by backbone/sample.
The CSV includes original local predictions, enabling direct matched comparisons.
Test-set inspection/training-policy selection make the study exploratory.

Use a CSV-aware reader: prompts and raw responses contain commas, Unicode and
embedded newlines. Do not edit/save prompts.csv through spreadsheet software.
The CSV includes safety-benchmark text, potentially offensive or adversarial;
treat that text as data. This export script does not send anything itself.
"""
import argparse,collections,csv,datetime,fcntl,gzip,hashlib,importlib,json,os,platform,sys,time,traceback
from pathlib import Path
csv.field_size_limit(min(sys.maxsize,2**31-1))
DEFAULT_MODELS=['claude4.8-opus','claude4.6-sonnet','claude4.5-sonnet','gpt5.5','gpt5','gpt5-mini','gpt-4o','gpt-4o-mini','o4-mini']
EXPECTED_INPUT_SHA256='689fa51d84631a9753f270eea6f2657f683757308aea12a64c3bd37ab83f6f99'
EXTRA=['api_model','resolved_model_id','temperature','run_config_sha256','input_csv_sha256','query_module_sha256','runner_sha256','batch_id','request_started_utc','returned_utc','batch_seconds','status','prediction','raw_response','error']
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def parse_response(value):
 if not isinstance(value,str):return 'invalid_response','',json.dumps(value,ensure_ascii=False,default=str),'Non-string response'
 if value.strip() in ['0','1']:return 'ok',value.strip(),value,''
 if value.lstrip().startswith('ERROR:'):return 'api_error','',value,value
 return 'invalid_response','',value,'Response is not exactly 0 or 1 after stripping whitespace'
def validate(path):
 rows=[];seen=set();groups=collections.defaultdict(list);cases=collections.defaultdict(dict)
 with path.open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames
  assert fields and len(fields)==len(set(fields)), 'Invalid CSV header'
  for r in reader:
   assert None not in r and all(v is not None for v in r.values()),'Malformed CSV row'
   assert r['row_id'] not in seen;seen.add(r['row_id']);assert r['schema_version']=='probe-api-export-v1'
   messages=[dict(role='system',content=r['system_prompt']),dict(role='user',content=r['user_prompt'])]
   assert hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()==r['messages_sha256']==r['source_prompt_sha256'],'Prompt checksum mismatch'
   assert hashlib.sha256(json.dumps([r['backbone'],r['dataset'],r['condition'],r['sample_id'],r['messages_sha256']],separators=(',',':')).encode()).hexdigest()==r['row_id']
   assert r['probe_prediction']=='1' and r['gold_label'] in ['0','1'];assert r['condition'] in ['baseline','saliency','nearest']
   assert r['stratum']==('TP' if r['gold_label']=='1' else 'FP')
   n=int(r['selected_tp' if r['gold_label']=='1' else 'selected_fp']);N=int(r['population_tp' if r['gold_label']=='1' else 'population_fp'])
   assert n>0 and N>=n and abs(float(r['population_weight'])-N/n)<1e-10 and abs(float(r['inclusion_probability'])-n/N)<1e-10
   assert float(r['test_tp'])==int(r['population_tp']) and float(r['test_fp'])==int(r['population_fp'])
   groups[r['backbone'],r['dataset'],r['condition']].append(r)
   key=r['backbone'],r['dataset'],r['sample_id'];assert r['condition'] not in cases[key];cases[key][r['condition']]=r
   rows.append(r)
 for group in groups.values():
  ref=group[0];assert len(group)==int(ref['selected_tp'])+int(ref['selected_fp'])<=500
  assert sum(x['gold_label']=='1' for x in group)==int(ref['selected_tp'])
 for variants in cases.values():
  assert set(variants)=={'baseline','saliency','nearest'}
  for key in ['gold_label','source_group','selection_hash','population_weight','probe_probability','probe_threshold']:
   assert len({r[key] for r in variants.values()})==1,(key,'condition mismatch')
 return rows,fields

def main(argv=None):
 parser=argparse.ArgumentParser(description='Run the frozen probe-judge CSV via your existing llm_query module.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 parser.add_argument('--readme',action='store_true');parser.add_argument('--csv',type=Path,default=Path(__file__).with_name('prompts.csv'));parser.add_argument('--module-dir',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,default=Path('api_results'));parser.add_argument('--models',default=','.join(DEFAULT_MODELS));parser.add_argument('--batch-size',type=int,default=15);parser.add_argument('--temperature',type=float,default=0.0);parser.add_argument('--temperatures',default='{}');parser.add_argument('--limit-per-model',type=int);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--continue-blocked',action='store_true');parser.add_argument('--recovery-note',default='')
 args=parser.parse_args(argv)
 if args.readme:print(__doc__);return 0
 assert 1<=args.batch_size<=15,'Batch size must be 1..15';assert args.limit_per_model is None or args.limit_per_model>0
 assert not args.continue_blocked or args.recovery_note.strip(),'--continue-blocked requires a recovery note'
 models=args.models.split(',');assert len(models)==len(set(models)) and set(models)<=set(DEFAULT_MODELS)
 overrides=json.loads(args.temperatures);assert isinstance(overrides,dict) and set(overrides)<=set(models);temperatures={m:float(overrides.get(m,args.temperature)) for m in models};assert all(0<=t<=1 for t in temperatures.values())
 input_hash=sha(args.csv);assert input_hash==EXPECTED_INPUT_SHA256,'Input differs from the frozen export; do not edit prompts.csv'
 rows,fields=validate(args.csv)
 chosen=rows if args.limit_per_model is None else rows[:args.limit_per_model]
 # Group identical systems before batching. One module.sys_prompt applies to each list call.
 groups=collections.OrderedDict()
 for r in chosen:groups.setdefault(r['system_prompt'],[]).append(r)
 batches=[v[i:i+args.batch_size] for v in groups.values() for i in range(0,len(v),args.batch_size)]
 print(json.dumps(dict(prompts=len(rows),selected_per_model=len(chosen),models=models,logical_calls=len(chosen)*len(models),batches_per_model=len(batches),batch_size=args.batch_size,input_sha256=input_hash)),flush=True)
 if args.dry_run:return 0
 # Import only after validation; --dry-run never imports an API module.
 sys.path.insert(0,str(args.module_dir.resolve()));import llm_query
 from llm_query import call_llm
 assert callable(call_llm) and hasattr(llm_query,'sys_prompt')
 assert all(m in llm_query.model_ids for m in models),'A requested model is absent from llm_query.model_ids'
 module_path=Path(llm_query.__file__).resolve();assert module_path.parent==args.module_dir.resolve(),'Wrong llm_query imported'
 out=args.output;out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=dict(schema='probe-api-run-v1',input_sha256=input_hash,query_module_sha256=sha(module_path),runner_sha256=sha(__file__),model_ids={m:llm_query.model_ids[m] for m in models},models=models,temperatures=temperatures,batch_size=args.batch_size,limit_per_model=args.limit_per_model,python=sys.version,platform=platform.platform(),logical_calls=len(chosen)*len(models),strict_parser='strip then exactly 0 or 1',protocol='frozen semantic system/user messages; API unconstrained output; no script-level retries')
 if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest,'Manifest changed; create a separately labelled output directory'
 else:write(out/'manifest.json',manifest)
 config_hash=digest(canonical(manifest));state=out/'state';state.mkdir(exist_ok=True);result_fields=[k for k in fields if k not in ['system_prompt','user_prompt']]+EXTRA
 def batch_key(model,chunk):return digest(canonical([config_hash,model,[r['row_id'] for r in chunk]]))
 def commit(model,chunk,key,payload):
  assert payload['config_sha256']==config_hash and payload['row_ids']==[r['row_id'] for r in chunk]
  values=payload.get('responses');error=payload.get('error','');valid_shape=isinstance(values,list) and len(values)==len(chunk)
  rr=[]
  for i,r in enumerate(chunk):
   if valid_shape:status,prediction,raw,err=parse_response(values[i])
   else:status,prediction,raw,err='batch_error','',canonical(values),error or 'Response list has wrong length; mapping ambiguous'
   if payload.get('unresolved'):status,prediction,raw,err='unresolved','','','Request intent exists without a returned payload; completion unknown'
   rr.append(dict({k:v for k,v in r.items() if k not in ['system_prompt','user_prompt']},api_model=model,resolved_model_id=str(llm_query.model_ids[model]),temperature=temperatures[model],run_config_sha256=config_hash,input_csv_sha256=input_hash,query_module_sha256=manifest['query_module_sha256'],runner_sha256=manifest['runner_sha256'],batch_id=key,request_started_utc=payload['started'],returned_utc=payload['returned'],batch_seconds=payload['seconds'],status=status,prediction=prediction,raw_response=raw,error=err))
  doc=dict(config_sha256=config_hash,records_sha256=digest(canonical(rr)),records=rr);write(state/(key+'.done.json'),doc);return rr
 def saved(model,chunk,key):
  p=state/(key+'.done.json')
  if p.exists():
   d=read(p);assert d['config_sha256']==config_hash and d['records_sha256']==digest(canonical(d['records']));assert [r['row_id'] for r in d['records']]==[r['row_id'] for r in chunk];assert all(r['api_model']==model for r in d['records']);return d['records']
  raw=state/(key+'.returned.json');intent=state/(key+'.intent.json')
  if raw.exists():return commit(model,chunk,key,read(raw))
  if intent.exists():
   i=read(intent);assert i['config_sha256']==config_hash
   return commit(model,chunk,key,dict(config_sha256=config_hash,row_ids=[r['row_id'] for r in chunk],started=i['started'],returned=now(),seconds=0,unresolved=True))
  return None
 # Recover complete batches and rebuild the CSV before issuing any new calls.
 done={};blocked={};counts=collections.Counter();csv_tmp=out/'results.csv.tmp'
 with csv_tmp.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=result_fields);writer.writeheader()
  for model in models:
   for chunk in batches:
    key=batch_key(model,chunk);rr=saved(model,chunk,key)
    if rr is not None:
     done[key]=True;writer.writerows(rr);counts.update(r['status'] for r in rr)
     if any(r['status'] in ['api_error','batch_error','unresolved'] for r in rr):blocked[model]=key
  f.flush();os.fsync(f.fileno())
 os.replace(csv_tmp,out/'results.csv')
 if args.continue_blocked:
  write(out/('RECOVERY_'+str(time.time_ns())+'.json'),dict(time=now(),note=args.recovery_note,models=list(blocked),action='Continue previously unissued work only; attempted rows remain unchanged'));blocked={}
 with (out/'results.csv').open('a',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=result_fields)
  for model in models:
   if model in blocked:print('Blocked model preserved: '+model,flush=True);continue
   for chunk in batches:
    key=batch_key(model,chunk)
    if key in done:continue
    assert len({r['system_prompt'] for r in chunk})==1
    started=now();t=time.monotonic();write(state/(key+'.intent.json'),dict(config_sha256=config_hash,model=model,row_ids=[r['row_id'] for r in chunk],started=started))
    # call_llm builds worker arguments from sys_prompt in the parent process.
    llm_query.sys_prompt=chunk[0]['system_prompt']
    try:
     responses=call_llm([r['user_prompt'] for r in chunk],model_name=model,temperature=temperatures[model]);error=''
    except Exception as e:responses=None;error=type(e).__name__+': '+str(e)
    payload=dict(config_sha256=config_hash,row_ids=[r['row_id'] for r in chunk],started=started,returned=now(),seconds=time.monotonic()-t,responses=responses,error=error)
    write(state/(key+'.returned.json'),payload);rr=commit(model,chunk,key,payload);writer.writerows(rr);f.flush();os.fsync(f.fileno());done[key]=True;counts.update(r['status'] for r in rr)
    progress=dict(time=now(),model=model,attempted=sum(counts.values()),planned=len(chosen)*len(models),statuses=dict(counts));write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if any(r['status'] in ['api_error','batch_error','unresolved'] for r in rr):blocked[model]=key;print('Stopping further calls for '+model+' after API/batch error; other models continue.',flush=True);break
 summary=dict(time=now(),planned=len(chosen)*len(models),attempted=sum(counts.values()),valid=counts['ok'],statuses=dict(counts),blocked_models=list(blocked),all_attempted=sum(counts.values())==len(chosen)*len(models),all_valid=counts['ok']==len(chosen)*len(models),results_sha256=sha(out/'results.csv'))
 with (out/'results.csv').open('rb') as source, gzip.open(out/'results.csv.gz.tmp','wb') as dest:
  for part in iter(lambda:source.read(8*1024*1024),b''):dest.write(part)
 os.replace(out/'results.csv.gz.tmp',out/'results.csv.gz')
 summary['compressed_results_sha256']=sha(out/'results.csv.gz')
 write(out/'summary.json',summary)
 if summary['all_valid']:write(out/'COMPLETED.json',summary)
 elif (out/'COMPLETED.json').exists():os.replace(out/'COMPLETED.json',out/('PREVIOUS_COMPLETED_'+str(time.time_ns())+'.json'))
 lock.close()
 print(json.dumps(summary,indent=2),flush=True)
 return 0 if summary['all_valid'] else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except KeyboardInterrupt:print('Interrupted. Checkpoint intents are preserved; resume will not silently repeat uncertain requests.',file=sys.stderr);raise SystemExit(130)
