from common import *
import subprocess,fcntl,traceback,collections
D=ROOT/'recovery_20261023'
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard=D/'STARTED.json';assert not guard.exists(),'Recovery is launch-once; inspect attempts before any restart'
 tasks=read(ROOT/'PREPARED.json')['tasks'];indices=[i for i,t in enumerate(tasks) if t['judge']=='mistral'];assert indices==list(range(6,12))
 proof={}
 for i in indices:
  t=tasks[i];old=ROOT/'runs'/t['study']/f"{t['backbone']}_mistral"
  assert (old/'FAILED.json').exists() and not (old/'responses.jsonl').exists() and not (old/'COMPLETED.json').exists()
  assert not list(old.rglob('*.attempt.json')) and not list(old.rglob('*.raw.json'))
  assert sha(t['path'])==t['sha256']
  proof[str(i)]={str(f.relative_to(ROOT)):sha(f) for f in old.rglob('*') if f.is_file()}
 write(guard,dict(pid=os.getpid(),indices=indices,no_generation_attempts_verified=True,preserved_failures=proof,environment=read(ROOT/'judge_specs.json')['models']['mistral']['environment'],time=time.time()))
 done=[]
 for i in indices:
  reserve();t=tasks[i];env=dict(os.environ);env.update(read(ROOT/'judge_specs.json')['models']['mistral']['environment'])
  write(D/'STATUS.json',dict(state='running',index=i,task=t,completed_tasks=done,pid=os.getpid(),time=time.time()))
  with (D/f'task_{i:02d}.log').open('x') as f:
   r=subprocess.run([GPU,str(D/'run_task_recovery.py'),str(i)],env=env,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:
   write(D/'STATUS.json',dict(state='failed',index=i,completed_tasks=done,returncode=r.returncode,time=time.time()));return
  done.append(i)
 resolved=[]
 for i,t in enumerate(tasks):
  rd=(D if t['judge']=='mistral' else ROOT)/'runs'/t['study']/f"{t['backbone']}_{t['judge']}"
  c=read(rd/'COMPLETED.json');assert c['n']==t['n'] and sha(rd/'responses.jsonl')==c['responses_sha256']
  path=rd/'responses.jsonl';statuses=c['statuses']
  if t['judge']=='glm' and t['study']=='mechanism':
   g=read(D/'GLM_RECOVERED.json');path=Path(g['output']);assert sha(path)==g['output_sha256'];statuses=g['statuses']
  resolved.append(dict(index=i,task=t,path=str(path),sha256=sha(path),n=c['n'],statuses=statuses,original_completion=str(rd/'COMPLETED.json')))
 assert sum(x['n'] for x in resolved)==1825000
 write(D/'COLLECTION_RESOLVED.json',dict(tasks=resolved,n=1825000,scope='Final additional collection only; prior reused studies separate.',time=time.time()))
 write(D/'STATUS.json',dict(state='completed',completed_tasks=done,resolved_total=1825000,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(D/'CONTROLLER_FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
