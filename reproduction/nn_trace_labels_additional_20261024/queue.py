from pathlib import Path
import json,time,hashlib,subprocess,fcntl,os,signal,traceback
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def verify(r):
 c=read(r/'COMPLETED.json');assert sha(r/'annotations.jsonl')==c['annotations_sha256'];assert sha(r/'OUTPUT_SHA256.json')==c['output_manifest_sha256']
 for n,h in read(r/'OUTPUT_SHA256.json').items():assert sha(r/n)==h,n
 assert c['total']==44089
 return c
def main():
 lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(R/'QUEUE_PLAN.json');parent=Path(plan['parent'])
 while not (parent/'COMPLETED.json').exists():
  if list(parent.glob('FAILED_*.json')) or not alive(plan['parent_pid']):raise RuntimeError('Parent stopped or failed; inspect before proceeding')
  write(R/'STATUS.json',{'state':'waiting_for_qwen122','time':time.time()});time.sleep(30)
 verify(parent)
 while alive(plan['parent_pid']):
  write(R/'STATUS.json',{'state':'waiting_for_parent_process_exit','time':time.time()});time.sleep(30)
 for child in plan['children']:
  r=Path(child['root'])
  if (r/'COMPLETED.json').exists():verify(r);continue
  assert not (r/'LAUNCH_ATTEMPT.json').exists(),'Already attempted; manual review required'
  write(r/'LAUNCH_ATTEMPT.json',{'time':time.time(),'queue_pid':os.getpid(),'run_sha256':sha(r/'run.py'),'model_config_sha256':sha(r/'MODEL_CONFIG.json')})
  with (r/'run.log').open('ab') as log:
   p=subprocess.Popen([plan['python'],'-u',str(r/'run.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(r/'LAUNCH.json',{'pid':p.pid,'time':time.time()})
  while p.poll() is None:
   write(R/'STATUS.json',{'state':'running_child','judge':child['judge'],'pid':p.pid,'time':time.time()})
   if list(r.glob('FAILED_*.json')):
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise RuntimeError('Child failed; outputs preserved; no retry: '+child['judge'])
   time.sleep(20)
  assert p.returncode==0,'Child exited unsuccessfully'
  verify(r)
 write(R/'COMPLETED.json',{'time':time.time(),'children':plan['children']});write(R/'STATUS.json',{'state':'completed','time':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  write(R/'FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
