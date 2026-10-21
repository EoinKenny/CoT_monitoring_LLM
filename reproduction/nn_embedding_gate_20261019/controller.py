from pathlib import Path
import os,sys,json,time,subprocess,concurrent.futures,fcntl,hashlib
R=Path(__file__).resolve().parent;R.joinpath('logs').mkdir(exist_ok=True);R.joinpath('results').mkdir(exist_ok=True)
lock=open(R/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
for name,digest in json.loads((R/'INPUTS.json').read_text()).items():assert sha(R/'inputs'/name)==digest
write('LAUNCH.json',dict(pid=os.getpid(),python=sys.executable,time=time.time(),code={p.name:sha(p) for p in [R/'common.py',R/'prepare.py',R/'train.py',R/'PROTOCOL.json']}))
def worker(stage,b):
 with open(R/'logs'/(stage+'_'+b+'.log'),'a') as out:
  p=subprocess.Popen([sys.executable,'-u',str(R/(stage+'.py')),b],stdout=out,stderr=subprocess.STDOUT,cwd=R)
  write('CHILD_'+stage+'_'+b+'.json',dict(pid=p.pid,stage=stage,backbone=b,time=time.time()));rc=p.wait()
  if rc:raise RuntimeError(f'{stage} {b} exited {rc}; see logs')
 return b
try:
 for stage in ['prepare','train']:
  write('STATUS.json',dict(stage=stage,time=time.time()))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(worker,stage,b) for b in ['llama','qwen','gemma','llama70','qwen122']]
   for future in concurrent.futures.as_completed(jobs):print('FINISHED',stage,future.result(),flush=True)
 write('TRAINING_COMPLETED.json',dict(time=time.time(),test_evaluated=False));write('STATUS.json',dict(stage='training_complete_awaiting_frozen_test_evaluation',time=time.time()))
except BaseException as e:
 write('FAILED.json',dict(time=time.time(),error=repr(e)));raise
