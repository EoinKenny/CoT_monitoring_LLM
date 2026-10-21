from pathlib import Path
import subprocess,time,json,os,traceback,fcntl
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python';JUDGE='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';start=time.time()
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'PIPELINE_STATUS.json').exists()
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,**kw);p=R/'PIPELINE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
def call(script,py=CPU,args=[]):
 with (R/(script.replace('.py','')+('_'+'_'.join(args) if args else '')+'.log')).open('x') as f:subprocess.run([py,str(R/script)]+args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('downloading_backbone')
 while not (R/'backbone.json').exists():
  p=Path('/proc/363728/cmdline')
  if not p.exists() or b'download.py' not in p.read_bytes():raise RuntimeError('Download stopped without pinned backbone manifest; inspect download.log')
  time.sleep(15)
 status('checking_extraction');call('smoke.py')
 status('preparing_frozen_splits');call('prepare.py')
 status('extracting_and_training_probes');call('run.py')
 status('auditing_probes');call('audit.py')
 status('sampling_and_explaining');call('explanations.py')
 for m in ['qwen','gemma','llama']:
  status('preparing_judge_prompts',model=m);call('prepare_judges.py',CPU if m=='qwen' else JUDGE,[m])
 for c in ['baseline','saliency','nearest']:
  cfg=[json.loads((R/'judges'/f'{m}_{c}'/'approved_config.json').read_text()) for m in ['qwen','gemma','llama']]
  assert len({x['semantic_sha256'] for x in cfg})==1 and len({x['n'] for x in cfg})==1
 call('record_environment.py',JUDGE)
 for m in ['qwen','gemma','llama']:
  for c in ['baseline','saliency','nearest']:
   condition=m+'_'+c;status('judging',condition=condition)
   with (R/(condition+'.log')).open('x') as f:subprocess.run([JUDGE,str(R/'judge_run.py'),str(R/'judges'/condition)],stdout=f,stderr=subprocess.STDOUT,check=True)
   call('evaluate.py',args=[condition])
 status('aggregating');call('evaluate.py')
 status('completed',conditions=9);(R/'PIPELINE_COMPLETED').write_text(json.dumps(dict(conditions=9,seconds=time.time()-start)))
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise
