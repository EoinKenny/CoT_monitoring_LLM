from pathlib import Path
import subprocess,time,json,os,traceback,fcntl,hashlib
R=Path('/home/ubuntu/experiments/ten-dataset-sampled-20261002');CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python';JUDGE='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';start=time.time()
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'PIPELINE_STATUS.json').exists()
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,**kw);p=R/'PIPELINE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
def call(args,log):
 with (R/log).open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 assert (R/'PREPARED.json').exists()
 for m in ['qwen','gemma','llama']:
  for c in ['baseline','saliency','nearest']:
   condition=m+'_'+c;status('judging',condition=condition)
   call([JUDGE,str(R/'judge_run.py'),str(R/'judges'/condition)],condition+'.log')
   call([CPU,str(R/'evaluate.py'),condition],condition+'_evaluation.log')
 status('aggregating');call([CPU,str(R/'evaluate.py')],'evaluation.log')
 status('completed',conditions=9);(R/'PIPELINE_COMPLETED').write_text(json.dumps(dict(conditions=9,seconds=time.time()-start)))
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise
