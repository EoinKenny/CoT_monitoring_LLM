"""Idempotent judging stage; skip only verified completed runs, never retry partial runs."""
from pathlib import Path
import os,json,subprocess,sys,time,hashlib,fcntl,traceback
R=Path(os.environ.get('EXPERIMENT_ROOT','/home/ubuntu/experiments/gemma31b-backbone-20261003'))
lock=(R/'judge_stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());logs=R/'resume_logs'/stamp;logs.mkdir(parents=True)
completed=[];os.environ['PYTHONPATH']=str(R);os.environ['EXPERIMENT_ROOT']=str(R)
def status(state,**kw):
 p=R/'JUDGE_STAGE_STATUS.json';t=p.with_suffix('.tmp');d=dict(state=state,seconds=time.time()-start,completed=completed,**kw);t.write_text(json.dumps(d,indent=2));os.replace(t,p);q=R/'PIPELINE_STATUS.json';qt=q.with_suffix('.tmp');qt.write_text(json.dumps(dict(d,stage_driver='judge_stage.py'),indent=2));os.replace(qt,q);print(json.dumps(d),flush=True)
def call(script,args=[],name=None):
 with (logs/((name or script.removesuffix('.py'))+'.log')).open('x') as f:subprocess.run([sys.executable,str(R/script)]+args,stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ)
def ensure_run(root,name):
 p=root/'run';expected=[json.loads(l) for l in (root/'prompts.jsonl').open()];config=json.loads((root/'approved_config.json').read_text())
 assert hashlib.sha256((root/'prompts.jsonl').read_bytes()).hexdigest()==config['prompt_sha256']
 if not (p/'COMPLETED').exists():
  assert not p.exists(),'Unresolved partial run: preserve and inspect before resuming'
  call('judge_run.py',[str(root)],name)
 rr=[json.loads(l) for l in (p/'judgments.jsonl').open()];by={x['sample_id']:x for x in rr}
 assert len(rr)==len(by)==len(expected) and set(by)=={x['sample_id'] for x in expected}
 assert all(by[x['sample_id']]['dataset']==x['dataset'] and by[x['sample_id']]['prediction'] in [0,1] for x in expected)
 assert (p/'prompts.jsonl').read_bytes()==(root/'prompts.jsonl').read_bytes()
 assert json.loads((p/'config.json').read_text())['prompt_sha256']==config['prompt_sha256']
try:
 (logs/'SOURCE_CHECKSUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},indent=2))
 status('checking_missing_strata');call('prepare_supplement.py')
 supplemental=bool(json.loads((R/'SUPPLEMENT_MANIFEST.json').read_text())['frozen']['rows'])
 for m in ['qwen','gemma','llama','glm','olmo']:
  for c in ['baseline','saliency','nearest']:
   name=m+'_'+c;status('judging_original',condition=name);ensure_run(R/'judges'/name,name)
   if supplemental:status('judging_supplement',condition=name);ensure_run(R/'supplement/judges'/name,name+'_supplement')
   status('evaluating',condition=name);call('evaluate.py',[name],'evaluate_'+name);completed.append(name)
 status('aggregating');call('evaluate.py');call('compare.py')
 (R/'PIPELINE_COMPLETED').write_text(json.dumps(dict(conditions=15,resumed_seconds=time.time()-start,logs=str(logs),supplement_manifest='SUPPLEMENT_MANIFEST.json')))
 status('completed')
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise
