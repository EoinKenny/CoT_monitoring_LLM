from pathlib import Path
import os,sys,json,time,subprocess,concurrent.futures,fcntl,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent/'nn-embedding-gate-20261019';(R/'logs').mkdir(exist_ok=True);(R/'results').mkdir(exist_ok=True)
lock=open(R/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
assert (R/'IMPLEMENTATION_AMENDMENT.json').exists();assert not (R/'RECOVERY_LAUNCH.json').exists(),'Existing recovery launch'
for name,digest in json.loads((P/'INPUTS.json').read_text()).items():assert sha(P/'inputs'/name)==digest
assert shutil.disk_usage(R).free>20*1024**3
write('RECOVERY_LAUNCH.json',dict(pid=os.getpid(),python=sys.executable,time=time.time(),code={p.name:sha(p) for p in R.glob('*.py')},protocol_sha256=sha(R/'PROTOCOL.json')))
def worker(b):
 with open(R/'logs'/('train_'+b+'.log'),'a') as out:
  p=subprocess.Popen([sys.executable,'-u',str(R/'train.py'),b],stdout=out,stderr=subprocess.STDOUT,cwd=R)
  write('CHILD_'+b+'.json',dict(pid=p.pid,backbone=b,time=time.time()));rc=p.wait()
  if rc:raise RuntimeError(f'Train {b} exited {rc}; see logs')
 return b
try:
 write('STATUS.json',dict(stage='training',time=time.time()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  jobs=[pool.submit(worker,b) for b in ['llama','qwen','gemma','llama70','qwen122']]
  for future in concurrent.futures.as_completed(jobs):print('FINISHED',future.result(),flush=True)
 write('TRAINING_COMPLETED.json',dict(time=time.time(),test_evaluated=False));write('STATUS.json',dict(stage='evaluating_frozen_models',time=time.time()))
 with open(R/'logs/evaluate.log','a') as out:subprocess.run([sys.executable,'-u',str(R/'evaluate.py')],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=R)
 write('STATUS.json',dict(stage='evaluation_complete',time=time.time()))
except BaseException as e:
 write('FAILED.json',dict(time=time.time(),error=repr(e)));raise
