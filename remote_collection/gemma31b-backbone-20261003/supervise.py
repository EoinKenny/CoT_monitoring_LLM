from pathlib import Path
import subprocess,time,json,os,traceback,fcntl,hashlib
R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003');PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';start=time.time()
os.environ['PYTHONPATH']=str(R)
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'PIPELINE_STATUS.json').exists()
completed=[]
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,completed=completed,**kw);p=R/'PIPELINE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
def call(script,args=[]):
 with (R/(script.replace('.py','')+('_'+'_'.join(args) if args else '')+'.log')).open('x') as f:subprocess.run([PY,str(R/script)]+args,stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ)
try:
 (R/'SOURCE_CHECKSUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},indent=2))
 status('waiting_for_pinned_backbone_download')
 pid=int((R/'download.pid').read_text())
 while not (R/'backbone.json').exists():
  p=Path(f'/proc/{pid}/cmdline')
  if not p.exists() or str(R/'download.py').encode() not in p.read_bytes():raise RuntimeError('Download stopped; inspect download.log. No automatic restart.')
  time.sleep(15)
 status('preparing_exact_frozen_splits');call('prepare.py')
 status('checking_prefix_equivalence_memory_and_throughput');call('smoke.py')
 status('extracting_and_training_probes');call('run.py')
 status('auditing_probes');call('audit.py')
 # Preserve probe results before the judge evaluator writes the aggregate results file.
 (R/'probe_results.json').write_bytes((R/'results.json').read_bytes())
 status('sampling_and_explaining');call('explanations.py')
 for m in ['qwen','gemma','llama','glm','olmo']:
  status('preparing_judge_prompts',model=m);call('prepare_judges.py',[m])
 for c in ['baseline','saliency','nearest']:
  cfg=[json.loads((R/'judges'/f'{m}_{c}'/'approved_config.json').read_text()) for m in ['qwen','gemma','llama','glm','olmo']]
  assert len({x['semantic_sha256'] for x in cfg})==1 and len({x['n'] for x in cfg})==1
 call('record_environment.py')
 status('judging_with_automatic_missing_stratum_supplements');call('judge_stage.py')
 status('completed',conditions=15)
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise
